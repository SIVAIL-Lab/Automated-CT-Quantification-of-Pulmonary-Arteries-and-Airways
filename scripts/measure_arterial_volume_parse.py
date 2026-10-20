#!/usr/bin/env python3
"""Measure complete saved pulmonary artery masks without inference or skeletonization."""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import statistics
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import nibabel as nib

from dataset_conversion import verify_case_conversion

from run_lung_volume_parse import (
    _case_file, _check_geometry, _discover_cases, _sha256, _voxel_volume_mm3,
    calculate_mask_volume,
)


MASK_NAME = "lung_arteries.nii.gz"
SCHEMA_VERSION = 1
MEASUREMENT = (
    "CT-derived pulmonary arterial volume of the complete saved binary mask; "
    "foreground voxel count times absolute affine determinant, converted to mL. "
    "No lung clipping, HU filtering, morphology, or skeleton measurement."
)
CSV_FIELDS = [
    "case", "status", "arterial_volume_ml", "arterial_voxel_count", "voxel_volume_mm3",
    "image", "input_sha256", "mask", "mask_sha256", "segmentation_provenance_status",
    "model_input_verified", "report", "runtime_sec", "error",
]


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, allow_nan=False) + "\n").encode("utf-8")


def _stage_bytes(path: Path, content: bytes) -> Path:
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        temporary = Path(handle.name)
        try:
            handle.write(content)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    return temporary


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _stage_bytes(path, content)
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _csv_bytes(rows: list[dict[str, Any]], fields: list[str]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def _measure_saved_mask(case_dir: Path, prediction_root: Path, row: dict[str, Any],
                        *, measurement_root: Path | None = None) -> None:
    conversion = verify_case_conversion(case_dir, groups=("ct",))
    if conversion["sources"]:
        row["dataset_conversion"] = conversion
    image_path = _case_file(case_dir).resolve()
    mask_path = (prediction_root / case_dir.name / MASK_NAME).resolve()
    provenance_path = mask_path.parent / "segmentation_provenance.json"
    row.update(image=str(image_path), mask=str(mask_path))
    row["input_sha256"] = _sha256(image_path)
    row["mask_sha256"] = _sha256(mask_path)
    row["segmentation_provenance_sha256"] = _sha256(provenance_path)
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if not isinstance(provenance, dict):
        raise ValueError("Segmentation provenance must be a JSON object")
    state = provenance.get("provenance_status")
    artifacts = provenance.get("artifact_sha256")
    if (
        provenance.get("schema_version") != 1
        or provenance.get("case") != case_dir.name
        or provenance.get("input_sha256") != row["input_sha256"]
        or state not in {"inference_verified", "legacy_adopted"}
        or provenance.get("model_input_verified") is not (state == "inference_verified")
        or not isinstance(artifacts, dict)
        or artifacts.get(MASK_NAME) != row["mask_sha256"]
    ):
        raise ValueError("Missing, invalid, or stale artery segmentation provenance")
    row.update(segmentation_provenance_status=state,
               model_input_verified=provenance["model_input_verified"])
    original_mask_path, original_mask_hash = mask_path, row["mask_sha256"]
    if measurement_root is not None:
        from measurement_scope import get_mask, scope_for_root
        if scope_for_root(measurement_root) != "all":
            mask_path, selection = get_mask(case_dir, measurement_root, "artery", "prediction")
            if selection["original_mask_path"] != str(original_mask_path):
                raise ValueError("Selected artery mask belongs to a different raw prediction")
            row.update(mask=str(mask_path), mask_sha256=selection["selected_mask_sha256"],
                       original_mask=str(original_mask_path), original_mask_sha256=original_mask_hash,
                       measurement_scope=selection["measurement_scope"], measurement_selection=selection)
    reference, mask = nib.load(image_path), nib.load(mask_path)
    _voxel_volume_mm3(reference)
    _check_geometry(reference, mask)
    measured = calculate_mask_volume(mask)
    qform, qcode = mask.get_qform(coded=True)
    sform, scode = mask.get_sform(coded=True)
    row["geometry"] = {
        "shape": list(mask.shape), "affine": mask.affine.tolist(),
        "voxel_spacing": [float(value) for value in mask.header.get_zooms()[:3]],
        "spatial_unit": mask.header.get_xyzt_units()[0],
        "qform": qform.tolist() if qform is not None else None, "qform_code": int(qcode),
        "sform": sform.tolist() if sform is not None else None, "sform_code": int(scode),
    }
    for path, digest in (
        (image_path, row["input_sha256"]), (mask_path, row["mask_sha256"]),
        (original_mask_path, original_mask_hash),
        (provenance_path, row["segmentation_provenance_sha256"]),
    ):
        if _sha256(path) != digest:
            raise ValueError(f"Input changed during volume measurement: {path}")
    for name, digest in row.get("measurement_selection", {}).get("source_sha256", {}).items():
        if _sha256(Path(name)) != digest:
            raise ValueError(f"Measurement selection changed during volume measurement: {name}")
    if verify_case_conversion(case_dir, groups=("ct",)) != conversion:
        raise ValueError("CT conversion sources changed during artery volume measurement")
    row.update(status="ok", arterial_volume_ml=measured["volume_ml"],
               arterial_voxel_count=measured["voxel_count"],
               voxel_volume_mm3=measured["voxel_volume_mm3"])


def measure_saved_mask(case_dir: Path, prediction_root: Path,
                       *, measurement_root: Path | None = None) -> dict[str, Any]:
    """Validate scan association and measure one saved mask, without any writes."""
    row = {field: None for field in CSV_FIELDS}
    row.update(case=case_dir.name, status="error", geometry=None)
    _measure_saved_mask(case_dir, prediction_root, row, measurement_root=measurement_root)
    return row


def _validate_output_root(args: argparse.Namespace, cases: list[Path]) -> None:
    output = args.output_root.expanduser().resolve()
    prediction = args.prediction_root.expanduser().resolve()
    protected = [prediction, *(case.resolve() for case in cases)]
    if getattr(args, "input_root", None) is not None:
        protected.append(args.input_root.expanduser().resolve())
    protected.extend((prediction.parent / name).resolve() for name in (
        "skeletonize3d", "analyzeskeleton", "lung_volume", "branch_count_validation", "airway",
    ))
    if any(output.is_relative_to(path) or path.is_relative_to(output) for path in protected):
        raise ValueError("Arterial volume output must not overlap input or existing computation directories")
    for case in cases:
        target = (output / case.name / "arterial_volume_report.json").resolve()
        if not target.is_relative_to(output):
            raise ValueError("Arterial report path escapes its output directory")
    if output.exists():
        for path in output.rglob("*"):
            if path.is_symlink() and not path.resolve().is_relative_to(output):
                raise ValueError(f"Arterial volume output contains an external symlink: {path}")


def _run_case(case_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    _validate_output_root(args, [case_dir])
    started = time.perf_counter()
    report_path = (args.output_root / case_dir.name / "arterial_volume_report.json").resolve()
    row = {field: None for field in CSV_FIELDS}
    row.update(case=case_dir.name, status="error", geometry=None, report=str(report_path),
               mask=str((args.prediction_root / case_dir.name / MASK_NAME).resolve()))
    try:
        _measure_saved_mask(case_dir, args.prediction_root, row,
                            measurement_root=getattr(args, "measurement_root", None))
    except Exception as error:
        row.update(status="error", arterial_volume_ml=None, arterial_voxel_count=None,
                   voxel_volume_mm3=None, error=f"{type(error).__name__}: {error}")
    row["runtime_sec"] = round(time.perf_counter() - started, 3)
    measurement = ("CT-derived pulmonary arterial volume of the largest 26-connected component; "
                   "original foreground voxel values and physical grid retained; excluded components are audited."
                   if row.get("measurement_scope") == "largest_26" else MEASUREMENT)
    if row.get("measurement_scope") == "largest_26":
        row["measurement"] = measurement
    _atomic_write(report_path, _json_bytes({
        "schema_version": SCHEMA_VERSION, "measurement": measurement, **row,
    }))
    return row


def _write_summary(output_root: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    scopes = {row.get("measurement_scope", "all") for row in rows if row["status"] in {"ok", "skipped"}}
    if len(scopes) > 1:
        raise ValueError("Arterial summary cannot mix measurement scopes")
    measurement = next((row["measurement"] for row in rows
                        if row.get("measurement_scope") == "largest_26" and "measurement" in row), MEASUREMENT)
    values = [row["arterial_volume_ml"] for row in rows if row["status"] in {"ok", "skipped"}]
    report = {
        "schema_version": SCHEMA_VERSION, "measurement": measurement,
        "total_cases": len(rows), "successful_cases": len(values),
        "failed_cases": len(rows) - len(values),
        "arterial_volume_ml": {
            "mean": statistics.fmean(values) if values else None,
            "median": statistics.median(values) if values else None,
            "min": min(values) if values else None, "max": max(values) if values else None,
        },
        "cases": rows,
    }
    if scopes == {"largest_26"}:
        report["measurement_scope"] = "largest_26"
    _atomic_write(output_root / "arterial_volume_metrics.csv", _csv_bytes(rows, CSV_FIELDS))
    _atomic_write(output_root / "summary.json", _json_bytes(report))
    return report


def _indexed_cases(rows: list[dict], source: str) -> dict[str, dict]:
    result = {}
    for row in rows:
        case = row.get("case") if isinstance(row, dict) else None
        if not isinstance(case, str) or not case or case in result:
            raise ValueError(f"Missing or duplicate case in {source}: {case!r}")
        result[case] = row
    return result


def _check_summary_association(measured: dict, existing: dict) -> None:
    stage = existing.get("stages", {}).get("totalsegmentator", {})
    if not isinstance(stage, dict) or not stage.get("input_sha256"):
        raise ValueError(f"Missing pipeline CT association for {measured['case']}")
    if measured.get("input_sha256") and measured["input_sha256"] != stage["input_sha256"]:
        raise ValueError(f"Pipeline CT hash differs for {measured['case']}")
    for source_key, stage_key in (("image", "image"), ("mask", "prediction")):
        if not stage.get(stage_key) or (measured.get(source_key) and
                Path(measured[source_key]).resolve() != Path(stage[stage_key]).resolve()):
            raise ValueError(f"Pipeline {source_key} association differs for {measured['case']}")
    skeleton = existing.get("stages", {}).get("skeletonize3d", {})
    prior_mask_hash = skeleton.get("input_sha256") if isinstance(skeleton, dict) else None
    if measured.get("mask_sha256") and prior_mask_hash and measured["mask_sha256"] != prior_mask_hash:
        raise ValueError(f"Pipeline artery mask hash differs for {measured['case']}")


def _update_pipeline_summary(prediction_root: Path, rows: list[dict[str, Any]]) -> None:
    """Append two case fields, preserving existing order, stages and cohort totals."""
    directory = prediction_root.parent
    csv_path, json_path = directory / "pipeline_metrics.csv", directory / "pipeline_report.json"
    csv_original, json_original = csv_path.read_bytes(), json_path.read_bytes()
    reader = csv.DictReader(io.StringIO(csv_original.decode("utf-8"), newline=""))
    fields = reader.fieldnames
    if not fields or len(set(fields)) != len(fields) or "case" not in fields:
        raise ValueError("Pipeline CSV has missing or duplicate column names")
    csv_rows = list(reader)
    if any(None in row for row in csv_rows):
        raise ValueError("Pipeline CSV contains extra values outside its header")
    report = json.loads(json_original)
    if not isinstance(report, dict) or not isinstance(report.get("cases"), list):
        raise ValueError("Pipeline JSON must contain a cases list")
    by_csv = _indexed_cases(csv_rows, "pipeline CSV")
    by_json = _indexed_cases(report["cases"], "pipeline JSON")
    measured = _indexed_cases(rows, "measurements")
    if by_csv.keys() != by_json.keys():
        raise ValueError("Pipeline CSV and JSON contain different case IDs")
    if not measured.keys() <= by_csv.keys():
        raise ValueError("Measured cases are absent from existing pipeline summaries")
    updated_fields = ("arterial_volume_ml", "arterial_volume_status")
    for case, record in by_csv.items():
        for field in fields:
            if field in updated_fields or field not in by_json[case]:
                continue
            value = by_json[case][field]
            expected = "" if value is None else str(value)
            if record[field] != expected:
                raise ValueError(f"Pipeline CSV and JSON disagree for {case}, field {field}")
    for case, row in measured.items():
        _check_summary_association(row, by_json[case])
        if row.get("status") not in {"ok", "skipped", "error"}:
            raise ValueError(f"Invalid arterial measurement status for {case}")
        success = row["status"] in {"ok", "skipped"}
        volume = row.get("arterial_volume_ml") if success else None
        if success and (isinstance(volume, bool) or not isinstance(volume, (int, float))
                        or not math.isfinite(volume) or volume < 0):
            raise ValueError(f"Invalid arterial volume for {case}")
        changes = {"arterial_volume_ml": volume, "arterial_volume_status": row["status"]}
        by_csv[case].update(changes)
        by_json[case].update(changes)
    fields.extend(field for field in updated_fields if field not in fields)
    changed_csv, changed_json = _csv_bytes(csv_rows, fields), _json_bytes(report)
    staged_csv = _stage_bytes(csv_path, changed_csv)
    staged_json = None
    replaced_csv = False
    try:
        staged_json = _stage_bytes(json_path, changed_json)
        if csv_path.read_bytes() != csv_original or json_path.read_bytes() != json_original:
            raise RuntimeError("Pipeline summaries changed concurrently; no update was published")
        staged_csv.replace(csv_path)
        replaced_csv = True
        staged_json.replace(json_path)
    except BaseException:
        if replaced_csv and csv_path.read_bytes() == changed_csv:
            _atomic_write(csv_path, csv_original)
        raise
    finally:
        staged_csv.unlink(missing_ok=True)
        if staged_json is not None:
            staged_json.unlink(missing_ok=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=Path("data/parse/train"))
    parser.add_argument("--prediction-root", type=Path, default=Path("data/parse/results/totalsegmentator"))
    parser.add_argument("--output-root", type=Path, default=Path("data/parse/results/arterial_volume"))
    parser.add_argument("--case", dest="cases", action="append", help="Case ID; repeat to select several.")
    parser.add_argument("--limit", type=int, help="Maximum number of selected cases.")
    parser.add_argument("--update-pipeline-summary", action="store_true",
                        help="Append arterial volume/status to existing pipeline CSV and JSON by case ID.")
    parser.add_argument("--quiet", action="store_true", help="Suppress successful progress messages.")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be a positive integer")
    for key in ("input_root", "prediction_root", "output_root"):
        setattr(args, key, getattr(args, key).expanduser().resolve())
    return args


def main() -> int:
    args = _parse_args()
    try:
        cases = _discover_cases(args.input_root, args.cases, args.limit)
        _validate_output_root(args, cases)
        rows = []
        for index, case_dir in enumerate(cases, 1):
            row = _run_case(case_dir, args)
            rows.append(row)
            _write_summary(args.output_root, rows)
            if row["status"] == "error":
                print(f"{case_dir.name}: {row['error']}", file=sys.stderr, flush=True)
            elif not args.quiet:
                print(f"[{index}/{len(cases)}] {case_dir.name}: {row['arterial_volume_ml']:.4f} mL", flush=True)
        if args.update_pipeline_summary:
            _update_pipeline_summary(args.prediction_root, rows)
    except Exception as error:
        print(f"error: {type(error).__name__}: {error}", file=sys.stderr)
        return 2
    return 1 if any(row["status"] == "error" for row in rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())
