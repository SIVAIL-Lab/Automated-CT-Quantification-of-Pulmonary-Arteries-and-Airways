#!/usr/bin/env python3
"""Compute final PARSE bronchial parameters with coupled local surfaces.

Inputs are completed airway measurements from reproduce_parse.py. Trachea is
included. Generations are topology-derived. Optional external engineering
adjustments do not establish anatomical or wall-boundary accuracy.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import time
import uuid

from measurement_scope import load_profile, source_results_root
from run_airway_parse import _measure_input, _check_source_hashes
from run_lung_volume_parse import _sha256
from airway_geometry import inspect_inputs
from airway_wall_surfaces import SurfaceConfig, METHOD
from airway_bronchial_parameters import summarize_cohort

CONFIG_ROOT = Path(__file__).resolve().parents[1] / "configs"

def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)

def _read(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value

def _verify(hashes: dict) -> None:
    for name, expected in hashes.items():
        if _sha256(Path(name)) != expected:
            raise ValueError(f"Source changed: {name}")

def resolve_input(root: Path, case: str) -> dict:
    """Read actual paths from the saved airway record and recheck its association."""
    path = root / "airway" / case / "airway_report.json"
    saved = _read(path)
    image = Path(saved["image"]).resolve()
    raw_mask = Path(saved.get("original_mask", saved["mask"])).resolve()
    _, current = _measure_input(image, raw_mask.parent, case, measurement_root=root)
    for key in ("image", "mask", "input_sha256", "mask_sha256", "segmentation_provenance_status", "model_input_verified"):
        if current[key] != saved[key]:
            raise ValueError(f"Saved airway {key} differs from the current source")
    geometry = inspect_inputs(image, Path(current["mask"]))
    if not geometry["geometry_usable"]:
        raise ValueError("Input geometry/values are unusable: " + "; ".join(geometry["reasons"]))
    hashes = {str(path.resolve()): _sha256(path), str(image): current["input_sha256"],
              current["mask"]: current["mask_sha256"], current["segmentation_provenance"]: current["segmentation_provenance_sha256"]}
    hashes.update(current.get("measurement_selection", {}).get("source_sha256", {}))
    skeleton = root / "airway" / case / "lung_airways_skeleton.nii.gz"
    artery = source_results_root(root) / "totalsegmentator" / case / "lung_arteries.nii.gz"
    for optional in (skeleton, artery):
        if optional.is_file():
            hashes[str(optional.resolve())] = _sha256(optional)
    _check_source_hashes(current)
    return {**current, "geometry": geometry, "source_report": str(path.resolve()), "source_sha256": hashes,
            "skeleton_path": str(skeleton.resolve()) if skeleton.is_file() else None,
            "artery_mask": str(artery.resolve()) if artery.is_file() else None,
            "trachea_policy": "include_as_segmented", "input_visual_review": "not_reviewed"}

def _surface_csv(path: Path, rows: list[dict], fields=None) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or sorted({key for row in rows for key in row})
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow({key: json.dumps(value, allow_nan=False) if isinstance(value, (list, dict)) else value
                                 for key, value in row.items() if key in fields})
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)

def _init_surface_worker() -> None:
    """Bound native thread pools inside each spawned, case-level worker."""
    global _SURFACE_WORKER_THREAD_LIMIT
    from threadpoolctl import threadpool_limits

    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    _SURFACE_WORKER_THREAD_LIMIT = threadpool_limits(limits=1)

def _run_surface_case(case: str, root: Path, run_dir: Path, profile: dict,
                      timeout_seconds: float, method: str, generation_review=None,
                      *, input_record: dict | None = None) -> tuple:
    """Compute and publish one independent case; keep failures in its record."""
    import gzip
    import nibabel as nib
    import numpy as np
    from airway_wall_surfaces import SurfaceConfig, estimate_branch, build_estimated_surfaces
    from airway_generations import build_generations, sample_branch
    from airway_bronchial_parameters import summarize_branch, summarize_case

    config = SurfaceConfig(**profile)
    case_dir = run_dir / case
    case_dir.mkdir()
    case_started = time.monotonic()
    branch_rows, sample_rows, artifacts = [], [], []
    inputs, generations = {}, {}
    try:
        # Dataset adapters may supply verified reference-label inputs. They still
        # pass the shared graph/skeleton/mask hash checks in build_generations;
        # reference masks must never be represented as model predictions.
        inputs = resolve_input(root, case) if input_record is None else {
            **input_record, "source_sha256": dict(input_record.get("source_sha256", {}))}
        generation_options = {"generation_review": generation_review} if generation_review is not None else {}
        generations = build_generations(root, case, inputs, max_generation=6, **generation_options)
        inputs["source_sha256"].update(generations["source_sha256"])
        _write(case_dir / "generations.json", generations)
        image = nib.load(inputs["image"])
        ct = image.get_fdata(dtype=np.float32)
        mask = np.asarray(nib.load(inputs["mask"]).dataobj) != 0
        artery = None
        if inputs.get("artery_mask"):
            vessel = nib.load(inputs["artery_mask"])
            if vessel.shape != image.shape or not np.allclose(vessel.affine, image.affine):
                raise ValueError("Artery mask must align with the CT")
            artery = np.asarray(vessel.dataobj) != 0
        selected = [branch for branch in generations["branches"] if isinstance(branch.get("generation"), int)
                    and 0 <= branch["generation"] <= 6]
        for index, branch in enumerate(selected):
            branch = {**branch, "case": case}
            samples = sample_branch(branch, step_mm=config.step_mm, tangent_window_mm=config.tangent_window_mm)
            if time.monotonic() - case_started > timeout_seconds:
                result = {"sections": [{**s, "status": "unavailable", "flags": ["case_runtime_limit"]} for s in samples],
                          "diagnostics": {"reason": "case_runtime_limit"}}
            else:
                try:
                    result = estimate_branch(ct, mask, image.affine, samples, config, artery_mask=artery)
                except (ValueError, RuntimeError, MemoryError) as error:
                    result = {"sections": [{**s, "status": "unavailable", "flags": [f"{type(error).__name__}: {error}"]} for s in samples],
                              "diagnostics": {"reason": str(error)}}
            for section in result["sections"]:
                section.update(case=case, branch_id=branch["branch_id"], generation=branch["generation"],
                               component_id=branch["component_id"], site_id=branch["branch_id"])
            result["local_surfaces"] = build_estimated_surfaces(result["sections"])
            path = case_dir / "branches" / f"{branch['branch_id']}.json.gz"
            path.parent.mkdir(exist_ok=True)
            temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
            try:
                with gzip.open(temporary, "wt", encoding="utf-8") as stream:
                    json.dump(result, stream, allow_nan=False, separators=(",", ":"))
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)
            measured = summarize_branch(branch, result["sections"])
            sample_rows.extend(measured.pop("radius_samples", []))
            measured["contours_path"] = str(path)
            measured["source_branches"] = branch.get("source_branches", [])
            branch_rows.append(measured)
            artifacts.append({"branch_id": branch["branch_id"], "path": str(path), "sha256": _sha256(path)})
            print(f"{case}: branch {index + 1}/{len(selected)}, generation {branch['generation']}, {len(result['sections'])} sections", flush=True)
        _verify(inputs["source_sha256"])
        del ct, mask, artery
        features = summarize_case(case, branch_rows, method=method, configuration=profile)
        features["source_graph_flags"] = generations.get("flags", [])
        features["unassigned_graph_branch_count"] = sum(branch.get("generation") is None for branch in generations["branches"])
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, MemoryError) as error:
        partial = summarize_case(case, branch_rows, method=method, configuration=profile)
        features = {**partial, **dict.fromkeys(("LA_g3_5_mm2", "WT_g3_5_mm", "WAP_g3_5_pct", "Pi10_g0_6_mm")),
                    "status": "error", "execution_error": f"{type(error).__name__}: {error}",
                    "partial_features": partial, "reviewed_only": None,
                    "unavailable_reasons": {key: "case_technical_error" for key in
                        ("LA_g3_5_mm2", "WT_g3_5_mm", "WAP_g3_5_pct", "Pi10_g0_6_mm")},
                    "pi10": {**partial["pi10"], "Pi10_mm": None, "status": "unavailable",
                             "unavailable_reason": "case_technical_error"}}
    features["segmentation_provenance_status"] = inputs.get("segmentation_provenance_status")
    features["model_input_verified"] = inputs.get("model_input_verified")
    _surface_csv(case_dir / "airway_radius_samples.csv", sample_rows)
    _surface_csv(case_dir / "airway_branch_parameters.csv", branch_rows)
    _write(case_dir / "branch_parameters.json", {"branches": branch_rows})
    _write(case_dir / "case_features.json", features)
    row = {"case": case, "backend": "surfaces", "input": inputs, "features": features,
           "branch_artifacts": artifacts, "elapsed_seconds": time.monotonic() - case_started,
           "artifact_sha256": {str(path): _sha256(path) for path in case_dir.rglob("*") if path.is_file()}}
    _write(case_dir / "report.json", row)
    return row, branch_rows, sample_rows

def run(args) -> dict:
    """Measure every requested case, with optional external topology adjustments."""
    measurement_root = args.measurement_root.expanduser().resolve()
    output = args.output_root.expanduser().resolve()
    if output == measurement_root or output.is_relative_to(measurement_root) or measurement_root.is_relative_to(output):
        raise ValueError("Airway output must be separate from the measurement root")
    if output.exists():
        raise ValueError("Use a new airway output directory")
    if isinstance(args.workers, bool) or args.workers < 1:
        raise ValueError("Workers must be a positive integer")
    if not math.isfinite(args.timeout_seconds) or args.timeout_seconds <= 0:
        raise ValueError("Case timeout must be finite and positive")
    profile = asdict(SurfaceConfig(**_read(args.surface_profile)))
    if profile["step_mm"] != .5:
        raise ValueError("The final bronchial protocol uses 0.5-mm physical sampling")
    adjustment_path = getattr(args, "generation_adjustments", None)
    adjustments = _read(adjustment_path) if adjustment_path is not None else {"schema_version": 2, "cases": {}}
    if adjustments.get("schema_version") != 2 or not isinstance(adjustments.get("cases"), dict):
        raise ValueError("Expected final generation-adjustment configuration schema 2")
    eligible = {p.parent.name for p in (measurement_root / "airway").glob("PA*/airway_report.json")
                if _read(p).get("status") in ("ok", "skipped")}
    cases = sorted(eligible) if args.all else list(args.case or [])
    if not cases or len(cases) != len(set(cases)) or any(not re.fullmatch(r"PA\d{6}", c) or c not in eligible for c in cases):
        raise ValueError("Select distinct completed PARSE cases using --case or --all")
    workers = min(args.workers, len(cases))
    selected_adjustments = {c: adjustments["cases"][c] for c in cases if c in adjustments["cases"]}
    source_names = ("run_parse_airway.py", "airway_geometry.py", "airway_wall_surfaces.py",
                    "airway_generations.py", "airway_bronchial_parameters.py", "airway_surface_geometry.py",
                    "airway_graph_inputs.py", "centerline_cross_sections.py")
    manifest = {"schema_version": 1, "dataset": "PARSE", "backend": "surfaces", "method": METHOD,
                "created_utc": datetime.now(timezone.utc).isoformat(), "cases": cases,
                "measurement_root": str(measurement_root),
                "measurement_scope": "largest_26" if load_profile(measurement_root) else "all",
                "profile": profile, "profile_sha256": _sha256(args.surface_profile),
                "adjustment_config_sha256": _sha256(adjustment_path) if adjustment_path is not None else None,
                "generation_adjustments": selected_adjustments,
                "engineering_adjustments": "external_configuration" if selected_adjustments else "none_applied",
                "evidence_rechecked": False if selected_adjustments else None,
                "anatomical_validation": "not_performed", "includes_trachea": True,
                "workers": workers, "code_sha256": {n: _sha256(Path(__file__).with_name(n)) for n in source_names}}
    output.mkdir(parents=True)
    _write(output / "run_manifest.json", manifest)
    _write(output / "profile.json", profile)
    if adjustment_path is not None:
        _write(output / "generation_adjustments.json", adjustments)
    records, branches, samples = [], [], []
    if workers == 1:
        outcomes = (_run_surface_case(c, measurement_root, output, profile, args.timeout_seconds,
                                      METHOD, adjustments["cases"].get(c)) for c in cases)
        for record, branch_rows, sample_rows in outcomes:
            records.append(record)
            branches.extend(branch_rows)
            samples.extend(sample_rows)
    else:
        from concurrent.futures import ProcessPoolExecutor
        import multiprocessing
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
                                 initializer=_init_surface_worker) as executor:
            futures = [executor.submit(_run_surface_case, c, measurement_root, output, profile,
                                       args.timeout_seconds, METHOD, adjustments["cases"].get(c)) for c in cases]
            for future in futures:
                record, branch_rows, sample_rows = future.result()
                records.append(record)
                branches.extend(branch_rows)
                samples.extend(sample_rows)
    features = [record["features"] for record in records]
    summary = summarize_cohort(features)
    _surface_csv(output / "airway_case_features.csv", features)
    _surface_csv(output / "airway_branch_parameters.csv", branches)
    _surface_csv(output / "airway_radius_samples.csv", samples)
    _surface_csv(output / "airway_parameter_summary.csv", summary)
    _write(output / "cohort.json", {"cases": features, "summary": summary})
    status = "complete" if all(row.get("status") != "error" for row in features) else "complete_with_errors"
    _write(output / "execution_status.json", {"status": status, "cases": len(cases),
           "errors": sum(row.get("status") == "error" for row in features)})
    return {"output_root": str(output), "status": status, "records": records, "plots": None}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measurement-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--case", action="append")
    selection.add_argument("--all", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--timeout-seconds", type=float, default=3600.)
    parser.add_argument("--surface-profile", type=Path, default=CONFIG_ROOT / "airway_surface.json")
    parser.add_argument("--generation-adjustments", type=Path, default=None,
                        help="Optional external schema-2 topology adjustments; default: automatic generations only.")
    args = parser.parse_args(argv)
    try:
        result = run(args)
    except (ValueError, OSError) as error:
        parser.exit(1, f"Airway quantification failed: {error}\n")
    print(f"{result['status']}: {result['output_root']}")
    return 0 if result["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
