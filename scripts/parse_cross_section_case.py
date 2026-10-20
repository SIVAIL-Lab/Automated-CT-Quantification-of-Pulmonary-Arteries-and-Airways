#!/usr/bin/env python3
"""Measure one PARSE prediction's sections and mean areas from its saved EDT tree."""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
from pathlib import Path
import resource
import sys
import time

import numpy as np

from measurement_io import verify_hashes as _verify
from measurement_io import write_csv, write_json
from centerline_cross_sections import Config
from section_measurement import _measure_source
from parse_radius_case import computation_identity as radius_computation_identity
from run_edt_experiment import sha256
from segment_cross_sections import METHOD, summarize_measurement


def _read(path: Path) -> dict:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected an object: {path}")
    return value


def computation_identity(config: Config = Config()) -> dict:
    config.validate()
    names = ("parse_cross_section_case.py", "centerline_cross_sections.py",
             "section_measurement.py", "segment_cross_sections.py", "edt_tree_radius.py",
             "measurement_io.py", "run_edt_experiment.py", "parse_radius_case.py",
             "edt_centerline_tree.py")
    return {
        "parameters": {**asdict(config), "segment_area_method": METHOD,
                       "source": "prediction", "measurement_scope": "largest_26",
                       "gap_bridging": False, "included_statuses": ["ok", "needs_review"]},
        "code_sha256": {str(Path(__file__).with_name(name).resolve()):
                        sha256(Path(__file__).with_name(name)) for name in names},
        "versions": {name: importlib.metadata.version(name)
                     for name in ("numpy", "scipy", "nibabel", "scikit-image")},
    }


def _add_hashes(target: dict, incoming: dict) -> None:
    if not isinstance(incoming, dict):
        raise ValueError("A source hash mapping is required")
    for name, digest in incoming.items():
        if not Path(name).is_absolute() or not isinstance(digest, str) or len(digest) != 64:
            raise ValueError("Input hashes require absolute paths and SHA-256 digests")
        if name in target and target[name] != digest:
            raise ValueError(f"Conflicting source hash: {name}")
        target[name] = digest


def load_source_record(checkpoint: Path, case: str) -> tuple[dict, dict]:
    """Verify the calibrated prediction artifacts without changing the EDT tree."""
    import re

    if not re.fullmatch(r"PA[0-9]{6}", case):
        raise ValueError("Expected a PARSE case ID")
    checkpoint = Path(checkpoint).resolve()
    raw = checkpoint.read_bytes()
    record = json.loads(raw)
    if (record.get("schema_version") != 1 or record.get("case") != case
            or record.get("source") != "prediction" or record.get("status") != "ok"
            or record.get("technical_validation") != "passed"
            or record.get("input_integrity") != "verified_after_computation"):
        raise ValueError("A successful verified prediction radius checkpoint is required")
    expected_identity = radius_computation_identity()
    if (record.get("computation_identity") != expected_identity
            or any(record.get(key) != expected_identity[key]
                   for key in ("parameters", "code_sha256", "versions"))):
        raise ValueError("Saved radius computation identity differs from the current scientific implementation")
    hashes = {str(checkpoint): hashlib.sha256(raw).hexdigest()}
    for key in ("source_sha256", "artifact_sha256", "code_sha256"):
        if not record.get(key):
            raise ValueError(f"Missing saved {key}")
        _add_hashes(hashes, record[key])
    report_path = Path(record["worker_report_path"]).resolve()
    _add_hashes(hashes, {str(report_path): record["worker_report_sha256"]})
    _verify(hashes)
    worker = _read(report_path)
    for key, value in worker.items():
        if key in ("visual_review", "review_observations"):
            continue
        if key == "artifact_sha256":
            if any(record[key].get(path) != digest for path, digest in value.items()):
                raise ValueError("Worker artifact identities differ from checkpoint")
        elif record.get(key) != value:
            raise ValueError(f"Worker report {key} differs from checkpoint")
    for key in ("tree_path", "tree_mask_path", "mask_path", "edt_path", "points_path", "segments_path"):
        if str(Path(record[key]).resolve()) not in hashes:
            raise ValueError(f"Unhashed required radius input: {key}")
    provenance = record.get("segmentation_provenance_status")
    if (provenance not in {"inference_verified", "legacy_adopted"}
            or record.get("model_input_verified") is not (provenance == "inference_verified")):
        raise ValueError("Invalid inherited segmentation provenance")
    record = deepcopy(record)
    record["source_checkpoint_path"] = str(checkpoint)
    return record, hashes


def _protect_output(output: Path, checkpoint: Path, hashes: dict) -> None:
    if output.exists():
        raise ValueError("Each case attempt requires a new output directory")
    # A checkpoint may be an individual file beside the requested new output;
    # its parent alone does not turn every sibling directory into an input.
    protected = {Path(name).parent for name in hashes if Path(name) != checkpoint}
    if any(output == folder or folder in output.parents or output in folder.parents for folder in protected):
        raise ValueError("Case output must be separate from all source directories")


def _check_radius_segments(measured: dict, record: dict) -> None:
    import csv

    with Path(record["segments_path"]).open(newline="", encoding="utf-8") as stream:
        original = list(csv.DictReader(stream))
    radii = {int(row["branch_id"]): row for row in original}
    if (len(radii) != len(original) or len(measured["segments"]) != len(radii)
            or set(radii) != {row["branch_id"] for row in measured["segments"]}):
        raise ValueError("Cross-section segment identities differ from the saved radius table")
    for row in measured["segments"]:
        previous = radii[row["branch_id"]]
        if (not np.isclose(row["length_mm"], float(previous["length_mm"]), rtol=1e-12, atol=1e-8)
                or not np.isclose(row["edt_segment_radius_mm"], float(previous["segment_radius_mm"]),
                                  rtol=1e-12, atol=1e-8)):
            raise ValueError("Cross-section export changes existing segment length or EDT radius")
    if (measured["summary"]["segment_count"] != record["summary"]["segment_count"]
            or not np.isclose(measured["summary"]["total_length_mm"], record["summary"]["unique_length_mm"],
                              rtol=1e-12, atol=1e-8)):
        raise ValueError("Cross-section export changes the original tree summary")


def compute_case(case: str, source_checkpoint: Path, output_dir: Path,
                 config: Config = Config()) -> dict:
    """Measure one saved prediction with the final study parameters."""
    started = time.monotonic()
    config.validate()
    source_checkpoint, output = Path(source_checkpoint).resolve(), Path(output_dir).resolve()
    record, hashes = load_source_record(source_checkpoint, case)
    identity = computation_identity(config)
    _protect_output(output, source_checkpoint, hashes)
    result = {
        "schema_version": 1, "dataset": "PARSE", "case": case, "source": "prediction",
        "status": "running", "technical_validation": "not_completed", "source_sha256": hashes,
        "computation_identity": identity, "code_sha256": identity["code_sha256"],
        "parameters": identity["parameters"], "versions": identity["versions"],
        "source_checkpoint_path": str(source_checkpoint), "source_report_path": record["worker_report_path"],
        "radius_segments_path": record["segments_path"], "original_tree_path": record["tree_path"],
        **{key: record[key] for key in ("mask_path", "edt_path", "tree_path", "tree_mask_path", "shape",
                                       "spacing_mm", "affine_mm", "segmentation_provenance_status",
                                       "model_input_verified", "anatomical_validation", "visual_review")},
        "review_observations": record.get("review_observations", []),
        "provenance": {"radius_computation_identity": record["computation_identity"],
                       "selected_component": record["selected_component"],
                       "original_radius_summary": record["summary"]},
        "geometry_changed": False, "edt_recomputed": False,
        "cross_sections_recomputed": True,
    }

    def progress(update):
        result["progress"] = update
        result["elapsed_seconds"] = time.monotonic() - started
        write_json(output / "progress.json", result)
        print(json.dumps({"case": case, **update}), flush=True)

    output.mkdir(parents=True)
    try:
        progress({"stage": "verify_tree"})
        tree = _read(Path(record["tree_path"]))
        if tree.get("case") != case or tree.get("source") != "prediction":
            raise ValueError("Saved tree case or source differs from its checkpoint")
        review = {"status": record["visual_review"], "observations": record.get("review_observations", [])}
        section = _measure_source("prediction", record, tree, review, output / "sections", config, progress)
        measured = _read(Path(section["measurement_path"]))
        _check_radius_segments(measured, record)
        progress({"stage": "summarize_areas"})
        areas = summarize_measurement(measured, "prediction")
        areas_path, areas_json_path = output / "segment_areas.csv", output / "segment_areas.json"
        write_csv(areas_path, areas["segments"])
        write_json(areas_json_path, areas)
        result.update(summary=areas["summary"], sample_summary=measured["summary"],
                      samples_path=section["samples_path"], measurement_path=section["measurement_path"],
                      segments_path=section["segments_path"], segment_areas_path=str(areas_path),
                      segment_areas_json_path=str(areas_json_path))
        if computation_identity(config) != identity:
            raise ValueError("Cross-section computation code, parameters or dependencies changed")
        _verify({**hashes, **identity["code_sha256"]})
        result["artifact_sha256"] = {str(path): sha256(path) for path in sorted(output.rglob("*"))
                                     if path.is_file() and path.name not in {"progress.json", "report.json", "failure.json"}}
        _verify(result["artifact_sha256"])
        result.update(status="ok", technical_validation="passed", input_integrity="verified_after_computation",
                      elapsed_seconds=time.monotonic() - started,
                      peak_memory_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.)
        result.pop("progress", None)
        write_json(output / "report.json", result)
        write_json(output / "progress.json", result)
        return result
    except Exception as error:
        result.update(status="error", technical_validation="failed", error=f"{type(error).__name__}: {error}",
                      elapsed_seconds=time.monotonic() - started)
        write_json(output / "failure.json", result)
        write_json(output / "progress.json", result)
        raise


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--parameters-json", default="{}", help="JSON object of cross-section Config fields")
    parser.add_argument("--memory-limit-gb", type=float, default=6.)
    args = parser.parse_args(argv)
    try:
        parameters = json.loads(args.parameters_json)
        if not isinstance(parameters, dict):
            raise ValueError("--parameters-json must encode an object")
        config = Config(**parameters)
        config.validate()
        if not np.isfinite(args.memory_limit_gb) or args.memory_limit_gb <= 0:
            raise ValueError("--memory-limit-gb must be finite and positive")
    except (TypeError, ValueError) as error:
        parser.error(str(error))
    limit = int(args.memory_limit_gb * 1024 ** 3)
    resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
    try:
        result = compute_case(args.case, args.source_checkpoint, args.output_dir, config)
    except Exception as error:
        print(f"PARSE cross-section case failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    print(json.dumps({"case": args.case, "status": result["status"], "summary": result["summary"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
