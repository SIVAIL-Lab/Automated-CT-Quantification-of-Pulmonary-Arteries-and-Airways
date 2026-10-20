#!/usr/bin/env python3
"""Quantify complete saved airway lumen masks, including trachea as segmented."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import nibabel as nib

from dataset_conversion import verify_case_conversion

import run_analyzeskeleton_parse as analyze
import run_skeletonize3d_parse as skeletonize
from measure_arterial_volume_parse import _atomic_write, _csv_bytes, _json_bytes
from run_lung_volume_parse import (
    _case_file, _check_geometry, _discover_cases, _sha256, _voxel_volume_mm3,
    calculate_mask_volume,
)


MASK_NAME = "lung_airways.nii.gz"
SKELETON_NAME = "lung_airways_skeleton.nii.gz"
SCHEMA_VERSION = 1
TRACHEA_POLICY = "include_as_segmented"
MEASUREMENT = (
    "Complete saved airway lumen, including tracheal voxels present in the mask; "
    "tracheal completeness is not independently verified. No airway wall, lung clipping, "
    "HU filtering, morphology or pruning. Volume is foreground voxel count times the "
    "absolute affine determinant in mL. Branch count is all-component AnalyzeSkeleton "
    "graph edge count, not anatomical generations; total length sums plugin lengths in mm."
)
METRIC_FIELDS = (
    "airway_volume_ml", "airway_skeleton_voxels", "airway_total_branch_length_mm",
    "airway_branch_count",
)
STAGES = ("volume", "skeletonize3d", "analyzeskeleton")
CSV_FIELDS = [
    "case", "status", "failed_stage", *METRIC_FIELDS, "airway_voxel_count",
    "voxel_volume_mm3", *(f"airway_{stage}_status" for stage in STAGES),
    "trachea_policy", "image", "input_sha256", "mask", "mask_sha256",
    "segmentation_provenance_status", "model_input_verified", "report", "runtime_sec", "error",
]
DERIVED_ARTIFACTS = (
    SKELETON_NAME, "skeleton_report.json", "analysis_report.json", *analyze.ARTIFACTS,
)


def _measure_input(
    image_path: Path, prediction_dir: Path, case: str,
    *, measurement_root: Path | None = None,
) -> tuple[nib.Nifti1Image, dict[str, Any]]:
    """Validate the existing CT association and measure only the lumen mask."""
    conversion = verify_case_conversion(image_path.parent.parent, groups=("ct",))
    image_path = image_path.resolve()
    mask_path = (prediction_dir / MASK_NAME).resolve()
    provenance_path = (prediction_dir / "segmentation_provenance.json").resolve()
    details = {
        "image": str(image_path), "input_sha256": _sha256(image_path),
        "mask": str(mask_path), "mask_sha256": _sha256(mask_path),
        "segmentation_provenance": str(provenance_path),
        "segmentation_provenance_sha256": _sha256(provenance_path),
    }
    if conversion["sources"]:
        details["dataset_conversion"] = conversion
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if not isinstance(provenance, dict):
        raise ValueError("Segmentation provenance must be a JSON object")
    state = provenance.get("provenance_status")
    artifacts = provenance.get("artifact_sha256")
    if (
        provenance.get("schema_version") != 1 or provenance.get("case") != case
        or provenance.get("input_sha256") != details["input_sha256"]
        or state not in {"inference_verified", "legacy_adopted"}
        or provenance.get("model_input_verified") is not (state == "inference_verified")
        or not isinstance(artifacts, dict)
        or artifacts.get(MASK_NAME) != details["mask_sha256"]
    ):
        raise ValueError("Missing, invalid, or stale airway segmentation provenance")
    if measurement_root is not None:
        from measurement_scope import get_mask, scope_for_root
        if scope_for_root(measurement_root) != "all":
            selected, selection = get_mask(image_path.parent.parent, measurement_root, "airway", "prediction")
            if selection["original_mask_path"] != str(mask_path):
                raise ValueError("Selected airway belongs to a different raw prediction")
            details.update(original_mask=str(mask_path), original_mask_sha256=details["mask_sha256"],
                           mask=str(selected), mask_sha256=selection["selected_mask_sha256"],
                           measurement_scope=selection["measurement_scope"], measurement_selection=selection)
            mask_path = selected
    reference, mask = nib.load(image_path), nib.load(mask_path)
    _voxel_volume_mm3(reference)
    _check_geometry(reference, mask)
    measured = calculate_mask_volume(mask)
    details.update(
        airway_volume_ml=measured["volume_ml"], airway_voxel_count=measured["voxel_count"],
        voxel_volume_mm3=measured["voxel_volume_mm3"],
        segmentation_provenance_status=state,
        model_input_verified=provenance["model_input_verified"],
        geometry={
            "shape": list(mask.shape), "affine": mask.affine.tolist(),
            "voxel_spacing": [float(value) for value in mask.header.get_zooms()[:3]],
            "spatial_unit": mask.header.get_xyzt_units()[0],
        },
    )
    for form in ("qform", "sform"):
        matrix, code = getattr(mask, f"get_{form}")(coded=True)
        details["geometry"][form] = matrix.tolist() if matrix is not None else None
        details["geometry"][f"{form}_code"] = int(code)
    _check_source_hashes(details)
    if verify_case_conversion(image_path.parent.parent, groups=("ct",)) != conversion:
        raise ValueError("CT conversion sources changed during airway measurement")
    return mask, details


def _check_source_hashes(row: dict[str, Any]) -> None:
    for path, digest in row.get("measurement_selection", {}).get("source_sha256", {}).items():
        if _sha256(Path(path)) != digest:
            raise ValueError(f"Airway measurement selection changed: {path}")
    for path, digest in row.get("dataset_conversion", {}).get("source_sha256", {}).items():
        if _sha256(Path(path)) != digest:
            raise ValueError(f"Airway conversion source changed during measurement: {path}")
    for path_field, hash_field in (
        ("image", "input_sha256"), ("mask", "mask_sha256"),
        ("segmentation_provenance", "segmentation_provenance_sha256"),
    ):
        if _sha256(Path(row[path_field])) != row[hash_field]:
            raise ValueError(f"Airway input changed during measurement: {row[path_field]}")


def _validate_output_root(args: argparse.Namespace, cases: list[Path]) -> None:
    output = args.output_root.expanduser().resolve()
    prediction = args.prediction_root.expanduser().resolve()
    protected = [prediction, *(case.resolve() for case in cases)]
    if getattr(args, "input_root", None) is not None:
        protected.append(args.input_root.expanduser().resolve())
    protected.extend((prediction.parent / name).resolve() for name in (
        "skeletonize3d", "analyzeskeleton", "lung_volume", "arterial_volume",
        "branch_count_validation",
    ))
    if any(output.is_relative_to(path) or path.is_relative_to(output) for path in protected):
        raise ValueError("Airway output must not overlap input or existing computation directories")
    for case in cases:
        if not (output / case.name).resolve().is_relative_to(output):
            raise ValueError("Airway case path escapes its output directory")
    if output.exists():
        for path in output.rglob("*"):
            if path.is_symlink() and not path.resolve().is_relative_to(output):
                raise ValueError(f"Airway output contains an external symlink: {path}")


def _run_case(
    case_dir: Path, args: argparse.Namespace, fiji: Path | None, fiji_error: str | None = None,
) -> dict[str, Any]:
    _validate_output_root(args, [case_dir])
    started = time.perf_counter()
    output_dir = (args.output_root / case_dir.name).resolve()
    row = dict.fromkeys(CSV_FIELDS)
    row.update(
        case=case_dir.name, status="error", stages={}, schema_version=SCHEMA_VERSION,
        measurement=MEASUREMENT, trachea_policy=TRACHEA_POLICY,
        report=str(output_dir / "airway_report.json"),
    )
    for stage in STAGES:
        row[f"airway_{stage}_status"] = "not_run"
    stage = "volume"
    try:
        measurement_root = getattr(args, "measurement_root", None) or args.output_root.parent
        from measurement_scope import scope_for_root
        scope_options = ({"measurement_root": measurement_root}
                         if scope_for_root(measurement_root) != "all" else {})
        _, measured = _measure_input(_case_file(case_dir), args.prediction_root / case_dir.name,
                                     case_dir.name, **scope_options)
        row.update(measured, airway_volume_status="ok")
        if row.get("measurement_scope") == "largest_26":
            row["measurement"] = (
                "Largest 26-connected component of the saved airway lumen, including retained tracheal voxels; "
                "no airway wall, clipping, or pruning. Excluded components are audited. "
                "Volume counts selected foreground voxels; length and branch count use its unpruned Fiji graph.")
        row["stages"]["volume"] = {"status": "ok", **measured}
        stage = "skeletonize3d"
        # Retain the measured volume even if a long-running Fiji process is interrupted.
        row.update(status="in_progress", runtime_sec=round(time.perf_counter() - started, 3))
        _atomic_write(Path(row["report"]), _json_bytes(row))
        if fiji is None:
            raise RuntimeError(fiji_error or "Fiji is unavailable")
        skeleton = skeletonize._run_case(
            args.prediction_root / case_dir.name, args.output_root, fiji, args.overwrite,
            input_path=Path(row["mask"]), output_name=SKELETON_NAME,
        )
        row["stages"][stage] = skeleton
        row[f"airway_{stage}_status"] = skeleton["status"]
        if skeleton["status"] not in {"ok", "skipped"}:
            raise RuntimeError(skeleton.get("error") or "Airway skeletonization failed")
        if skeleton["input_sha256"] != row["mask_sha256"]:
            raise ValueError("Airway skeleton does not match measured lumen")
        row["airway_skeleton_voxels"] = skeleton["skeleton_voxels"]
        stage = "analyzeskeleton"
        analysis = analyze._run_case(
            output_dir, args.output_root, fiji,
            args.overwrite or skeleton["status"] == "ok",
            input_path=output_dir / SKELETON_NAME,
        )
        row["stages"][stage] = analysis
        row[f"airway_{stage}_status"] = analysis["status"]
        if analysis["status"] not in {"ok", "skipped"}:
            raise RuntimeError(analysis.get("error") or "Airway skeleton analysis failed")
        if analysis["provenance"]["input_sha256"] != skeleton["output_sha256"]:
            raise ValueError("Airway analysis does not match the computed skeleton")
        row.update(
            airway_total_branch_length_mm=analysis["total_branch_length_mm"],
            airway_branch_count=analysis["branch_count"],
            artifact_sha256={name: _sha256(output_dir / name) for name in DERIVED_ARTIFACTS},
            status="ok",
        )
    except Exception as error:
        row.update(status="error", failed_stage=stage, error=f"{type(error).__name__}: {error}")
        row[f"airway_{stage}_status"] = "error"
        row["stages"].setdefault(stage, {})
        row["stages"][stage].update(status="error", error=row["error"])
    if row["airway_volume_status"] == "ok":
        try:
            _check_source_hashes(row)
        except Exception as error:
            row.update(dict.fromkeys((*METRIC_FIELDS, "airway_voxel_count", "voxel_volume_mm3")))
            row.update(status="error", failed_stage="source_validation", artifact_sha256=None,
                       error=f"{row['error'] + '; ' if row['error'] else ''}{type(error).__name__}: {error}")
            for name, result in row["stages"].items():
                result.update(status="error", error="Input changed during airway computation")
                row[f"airway_{name}_status"] = "error"
            row["stages"]["source_validation"] = {"status": "error", "error": str(error)}
    row["runtime_sec"] = round(time.perf_counter() - started, 3)
    try:
        _atomic_write(Path(row["report"]), _json_bytes(row))
    except OSError as error:
        row.update(status="error", failed_stage=row["failed_stage"] or "report",
                   error=f"{row['error'] + '; ' if row['error'] else ''}Cannot save airway report: {error}")
    return row


def _load_validated_result(
    image_path: Path, prediction_dir: Path, directory: Path,
) -> tuple[nib.Nifti1Image, dict[str, Any]]:
    """Read a complete result for visualization without invoking Fiji or inference."""
    report = json.loads((directory / "airway_report.json").read_text(encoding="utf-8"))
    if (report.get("schema_version") != SCHEMA_VERSION or report.get("case") != prediction_dir.name
            or directory.name != prediction_dir.name or report.get("status") != "ok"
            or report.get("trachea_policy") != TRACHEA_POLICY):
        raise ValueError("Missing, invalid, or incomplete airway report; rerun run_airway_parse.py")
    from measurement_scope import scope_for_root
    measurement_root = directory.resolve().parent.parent
    scope_options = ({"measurement_root": measurement_root}
                     if scope_for_root(measurement_root) != "all" else {})
    mask, measured = _measure_input(image_path, prediction_dir, report["case"], **scope_options)
    if report.get("measurement_scope", "all") != scope_for_root(measurement_root):
        raise ValueError("Airway result measurement scope differs from its results root")
    # A refreshed reference label may update the shared conversion manifest while
    # leaving the CT, predicted lumen, skeleton, and analysis unchanged.
    if any(report.get(key) != value for key, value in measured.items() if key != "dataset_conversion"):
        raise ValueError("Airway measurement or source provenance is stale")
    hashes = {name: _sha256(directory / name) for name in DERIVED_ARTIFACTS}
    if report.get("artifact_sha256") != hashes:
        raise ValueError("Airway artifact hashes are stale")
    skeleton = skeletonize._load_binary_image(directory / SKELETON_NAME)
    voxels = skeletonize._validate_skeleton(mask, skeleton)
    skeleton_report = json.loads((directory / "skeleton_report.json").read_text(encoding="utf-8"))
    analysis_report = json.loads((directory / "analysis_report.json").read_text(encoding="utf-8"))
    if (skeleton_report.get("status") not in {"ok", "skipped"}
            or skeleton_report.get("input_sha256") != measured["mask_sha256"]
            or skeleton_report.get("output_sha256") != hashes[SKELETON_NAME]
            or skeleton_report.get("skeleton_voxels") != voxels
            or report.get("airway_skeleton_voxels") != voxels
            or analysis_report.get("provenance", {}).get("input_sha256") != hashes[SKELETON_NAME]):
        raise ValueError("Airway skeleton provenance is stale")
    metrics = analyze._cached_metrics(
        directory / "analysis_report.json", analysis_report["provenance"], skeleton, voxels,
        reference_image=SKELETON_NAME,
    )
    if metrics is None or any(report.get(key) != metrics[field] for key, field in (
        ("airway_total_branch_length_mm", "total_branch_length_mm"),
        ("airway_branch_count", "branch_count"),
    )):
        raise ValueError("Airway analysis tables, graph or measurements are stale")
    if "dataset_conversion" in measured:
        report = dict(report, dataset_conversion=measured["dataset_conversion"])
    return mask, report


def _write_summary(output_root: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    scopes = {row.get("measurement_scope", "all") for row in rows
              if row.get("airway_volume_status") in {"ok", "skipped"}}
    if len(scopes) > 1:
        raise ValueError("Airway summary cannot mix measurement scopes")
    measurement = next((row["measurement"] for row in rows
                        if row.get("measurement_scope") == "largest_26" and "measurement" in row), MEASUREMENT)
    volumes = [row for row in rows if row.get("airway_volume_status") in {"ok", "skipped"}]
    analyses = [row for row in rows if row.get("airway_analyzeskeleton_status") in {"ok", "skipped"}]
    report = {
        "schema_version": SCHEMA_VERSION, "measurement": measurement, "trachea_policy": TRACHEA_POLICY,
        "total_cases": len(rows), "successful_cases": sum(row["status"] == "ok" for row in rows),
        "failed_cases": sum(row["status"] != "ok" for row in rows),
        "volume_successful_cases": len(volumes), "analysis_successful_cases": len(analyses),
        "airway_volume_ml": math.fsum(row["airway_volume_ml"] for row in volumes),
        "airway_branch_count": sum(row["airway_branch_count"] for row in analyses),
        "airway_total_branch_length_mm": math.fsum(row["airway_total_branch_length_mm"] for row in analyses),
        "cases": rows,
    }
    if scopes == {"largest_26"}:
        report["measurement_scope"] = "largest_26"
    _atomic_write(output_root / "airway_metrics.csv", _csv_bytes(rows, CSV_FIELDS))
    _atomic_write(output_root / "summary.json", _json_bytes(report))
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=Path("data/parse/train"))
    parser.add_argument("--prediction-root", type=Path, default=Path("data/parse/results/totalsegmentator"))
    parser.add_argument("--output-root", type=Path, default=Path("data/parse/results/airway"))
    parser.add_argument("--case", dest="cases", action="append")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--fiji", type=Path)
    parser.add_argument("--overwrite", action="store_true", help="Recompute airway derivatives; never run inference.")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be a positive integer")
    args.input_root = args.input_root.expanduser()
    args.prediction_root = args.prediction_root.expanduser()
    args.output_root = args.output_root.expanduser()
    return args


def main() -> int:
    args = _parse_args()
    try:
        cases = _discover_cases(args.input_root, args.cases, args.limit)
        _validate_output_root(args, cases)
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    fiji_error = None
    try:
        fiji = skeletonize._find_fiji(args.fiji)
    except Exception as error:
        fiji, fiji_error = None, f"{type(error).__name__}: {error}"
    rows = []
    for index, case in enumerate(cases, 1):
        if not args.quiet:
            print(f"[{index}/{len(cases)}] {case.name}: airway lumen including trachea as segmented", flush=True)
        row = _run_case(case, args, fiji, fiji_error)
        rows.append(row)
        if row["status"] != "ok":
            print(f"{case.name}: {row['failed_stage']} failed: {row['error']}", file=sys.stderr)
        elif not args.quiet:
            print(f"  {row['airway_volume_ml']:.3f} mL, {row['airway_total_branch_length_mm']:.3f} mm, "
                  f"{row['airway_branch_count']} branches; "
                  f"skeleton={row['airway_skeletonize3d_status']}, analysis={row['airway_analyzeskeleton_status']}",
                  flush=True)
    _write_summary(args.output_root, rows)
    return 1 if any(row["status"] != "ok" for row in rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())
