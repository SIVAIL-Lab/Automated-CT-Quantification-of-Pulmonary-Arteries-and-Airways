#!/usr/bin/env python3
"""Run artery analysis and volumes, optionally adding lung volumes and airway analysis."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
import time
from pathlib import Path
from typing import Any

import measure_arterial_volume_parse as arterial_volume
import run_airway_parse as airway
import run_analyzeskeleton_parse as analyze
import run_lung_volume_parse as lung_volume
import run_skeletonize3d_parse as skeletonize
from dataset_cases import discover_ct_cases


STAGES = ("totalsegmentator", "skeletonize3d", "analyzeskeleton")
ANALYSIS_METRICS = (
    "component_count", "branch_count", "endpoint_count", "junction_count",
    "junction_voxel_count", "total_branch_length_mm", "mean_branch_length_mm",
    "max_branch_length_mm",
)
AIRWAY_METRICS = (
    "airway_volume_ml", "airway_skeleton_voxels", "airway_total_branch_length_mm",
    "airway_branch_count",
)
AIRWAY_STAGE_FIELDS = (
    "airway_volume_status", "airway_skeletonize3d_status", "airway_analyzeskeleton_status",
)
NORMALIZED_FIELDS = (
    "branches_per_lung_l", "arterial_length_m_per_lung_l", "arterial_volume_fraction",
    "airway_length_m_per_lung_l", "airway_volume_fraction", "airway_branches_per_lung_l",
)
CSV_FIELDS = [
    "case", "status", "failed_stage", "dice", "iou", "skeleton_voxels",
    *ANALYSIS_METRICS, *(f"{stage}_status" for stage in STAGES), "runtime_sec", "error",
    "lung_volume_status", *lung_volume.VOLUME_FIELDS, "branches_per_lung_l",
    "segmentation_provenance_status",
    "arterial_volume_ml", "arterial_volume_status",
    "airway_status", *AIRWAY_METRICS, *AIRWAY_STAGE_FIELDS,
    *NORMALIZED_FIELDS[1:], "normalization_status", "measurement_scope",
    "main_component_dice", "main_component_iou",
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=Path("data/parse/train"),
                        help="Cases containing image/ and label/ (default: data/parse/train).")
    parser.add_argument("--output-root", type=Path, default=Path("data/parse/results"),
                        help="Root for computation outputs (default: data/parse/results).")
    parser.add_argument("--case", dest="cases", action="append",
                        help="Case ID; repeat to select several (default: all CT cases).")
    parser.add_argument("--limit", type=int, help="Maximum number of selected cases.")
    parser.add_argument("--device", default="gpu", help="gpu, gpu:N, cpu or mps (default: gpu).")
    parser.add_argument("--fiji", type=Path, help="Fiji executable; otherwise use FIJI_BIN or discovery.")
    parser.add_argument("--with-lung-volume", action="store_true",
                        help="Also measure CT lung volumes and normalized branch, length and volume metrics.")
    parser.add_argument("--with-airway", action="store_true",
                        help="Also quantify the saved airway lumen, including its segmented trachea.")
    parser.add_argument("--overwrite", action="store_true", help="Rerun all requested stages for selected cases.")
    parser.add_argument("--quiet", action="store_true", help="Suppress successful case and inference progress.")
    parser.add_argument("--measurement-scope", choices=("all", "largest_26"), default=None,
                        help="Use all masks explicitly, or the published PARSE main-component analysis.")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be a positive integer")
    if args.device not in {"gpu", "cpu", "mps"} and not re.fullmatch(r"gpu:\d+", args.device):
        parser.error("--device must be gpu, gpu:N, cpu or mps")
    return args


def _segment_case(case_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    # Keep CLI help and downstream-only tooling independent of torch startup.
    import run_totalsegmentator_parse as segmentation

    from measurement_scope import load_profile, source_results_root
    if load_profile(args.output_root):
        import nibabel as nib
        import numpy as np
        raw = source_results_root(args.output_root)
        image_path = segmentation._case_file(case_dir, "image").resolve()
        label_path = segmentation._case_file(case_dir, "label").resolve()
        prediction = raw / "totalsegmentator" / case_dir.name
        identity = segmentation._cached_provenance(
            prediction / "segmentation_provenance.json", case_dir.name,
            segmentation._sha256(image_path),
        )
        if identity is None:
            raise ValueError("Main-component analysis requires verified saved segmentation provenance")
        image = nib.load(image_path)
        mask = skeletonize._load_binary_image(prediction / segmentation.TARGET_MASK)
        label = skeletonize._load_binary_image(label_path)
        lung_volume._check_geometry(image, mask)
        lung_volume._check_geometry(image, label)
        scores = segmentation._overlap_metrics(np.asarray(mask.dataobj), np.asarray(label.dataobj),
                                               lung_volume._voxel_volume_mm3(image))
        return {"case": case_dir.name, "status": "ok", "inference_ran": False,
                "image": str(image_path), "prediction": str(prediction / segmentation.TARGET_MASK),
                "input_sha256": segmentation._sha256(image_path), **scores,
                "provenance_status": identity["provenance_status"],
                "model_input_verified": identity["model_input_verified"],
                "measurement_scope": "all", "source_reuse": "validated_saved_segmentation"}

    options = argparse.Namespace(
        output_root=args.output_root / "totalsegmentator", device=args.device,
        overwrite=args.overwrite, quiet=args.quiet,
        inference_lock=getattr(args, "inference_lock", None),
        stage_callback=getattr(args, "stage_callback", None),
    )
    return segmentation._run_case(case_dir, options)


def _lung_volume_case(case_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    from measurement_scope import load_profile, source_results_root
    if load_profile(args.output_root):
        from main_component_lung_cache import load_lungs
        _, saved = load_lungs(case_dir, args.output_root)
        return saved
    options = argparse.Namespace(
        output_root=args.output_root / "lung_volume", device=args.device,
        overwrite=args.overwrite, quiet=args.quiet, measure_only=False,
        inference_lock=getattr(args, "inference_lock", None),
        stage_callback=getattr(args, "stage_callback", None),
    )
    return lung_volume._run_case(case_dir, options)


def _arterial_volume_case(case_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    from measurement_scope import source_results_root
    options = argparse.Namespace(
        input_root=case_dir.parent,
        prediction_root=source_results_root(args.output_root) / "totalsegmentator",
        measurement_root=args.output_root,
        output_root=args.output_root / "arterial_volume", quiet=args.quiet,
    )
    return arterial_volume._run_case(case_dir, options)


def _airway_case(
    case_dir: Path, args: argparse.Namespace, fiji: Path | None, fiji_error: str | None,
) -> dict[str, Any]:
    from measurement_scope import source_results_root
    options = argparse.Namespace(
        input_root=getattr(args, "input_root", case_dir.parent),
        prediction_root=source_results_root(args.output_root) / "totalsegmentator",
        measurement_root=args.output_root,
        output_root=args.output_root / "airway", overwrite=args.overwrite, quiet=args.quiet,
    )
    return airway._run_case(case_dir, options, fiji, fiji_error)


def _record_failure(row: dict[str, Any], stage: str, error: Exception) -> None:
    row["status"] = "error"
    row["failed_stage"] = row["failed_stage"] or stage
    row[f"{stage}_status"] = "error"
    message = f"{type(error).__name__}: {error}"
    row["error"] = f"{row['error']}; {stage}: {message}" if row["error"] else message
    row["stages"].setdefault(stage, {"case": row["case"], "status": "error", "error": message})
    if row["stages"][stage]["status"] in {"ok", "skipped"}:
        row["stages"][stage].update(status="error", error=message)


def _stage_progress(args: argparse.Namespace, stage: str, event: str, details=None) -> None:
    callback = getattr(args, "stage_callback", None)
    if callback is not None:
        callback(stage, event, details)


def _nonnegative_number(value: Any, name: str, *, positive: bool = False) -> float:
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value < 0 or (positive and value == 0)):
        raise ValueError(f"{name} must be finite and {'positive' if positive else 'nonnegative'}")
    return value


def normalize_case(row: dict[str, Any], *, with_lung_volume: bool, with_airway: bool) -> None:
    """Normalize validated measurements independently; failures never become zeros."""
    row.update(dict.fromkeys(NORMALIZED_FIELDS))
    row["normalization_status"] = "not_requested"
    if not with_lung_volume:
        row["stages"].pop("normalization", None)
        return
    stages = row["stages"]
    lung = stages.get("lung_volume", {})
    # (output field, numerator, successful prerequisite, CT association, unit scale)
    rules = [
        ("branches_per_lung_l", "branch_count", "analyzeskeleton", "arterial_analysis", 1.0),
        ("arterial_length_m_per_lung_l", "total_branch_length_mm", "analyzeskeleton", "arterial_analysis", .001),
        ("arterial_volume_fraction", "arterial_volume_ml", "arterial_volume", "arterial_volume", .001),
    ]
    if with_airway:
        rules.extend([
            ("airway_length_m_per_lung_l", "airway_total_branch_length_mm", "airway_analyzeskeleton", "airway", .001),
            ("airway_volume_fraction", "airway_volume_ml", "airway_volume", "airway", .001),
            ("airway_branches_per_lung_l", "airway_branch_count", "airway_analyzeskeleton", "airway", 1.0),
        ])
    errors = {}
    sources = {}
    for field, numerator, prerequisite, source, scale in rules:
        try:
            if row.get("lung_volume_status") not in {"ok", "skipped"}:
                raise ValueError("CT lung volume is unavailable")
            liters = _nonnegative_number(row.get("lung_total_volume_l"), "CT lung volume L", positive=True)
            ml = _nonnegative_number(row.get("lung_total_volume_ml"), "CT lung volume mL", positive=True)
            if not math.isclose(ml / 1000, liters, rel_tol=1e-9, abs_tol=1e-12):
                raise ValueError("CT lung volume mL and L disagree")
            if row.get(f"{prerequisite}_status") not in {"ok", "skipped"}:
                raise ValueError(f"{prerequisite} measurement is unavailable")
            if source == "arterial_analysis":
                segmented = stages.get("totalsegmentator", {})
                skeleton = stages.get("skeletonize3d", {})
                analysis = stages.get("analyzeskeleton", {})
                if (not skeleton.get("input_sha256") or not skeleton.get("output_sha256")
                        or analysis.get("provenance", {}).get("input_sha256") != skeleton["output_sha256"]):
                    raise ValueError("Arterial analysis does not match its source skeleton")
                measured_mask = stages.get("arterial_volume", {}).get("mask_sha256")
                if measured_mask and measured_mask != skeleton["input_sha256"]:
                    raise ValueError("Arterial length and volume use different masks")
                input_hash = segmented.get("input_sha256")
            else:
                input_hash = stages.get(source, {}).get("input_sha256")
            if not input_hash or input_hash != lung.get("input_sha256"):
                raise ValueError(f"{source} and lung results do not share the same CT hash")
            sources[source] = input_hash
            value = _nonnegative_number(row.get(numerator), numerator)
            normalized = value / ml if field.endswith("_fraction") else value * scale / liters
            _nonnegative_number(normalized, field)
            row[field] = normalized
        except (ValueError, KeyError, TypeError) as error:
            errors[field] = str(error)
    row["normalization_status"] = "error" if errors else "ok"
    stages["normalization"] = {
        "case": row["case"], "status": row["normalization_status"],
        "input_sha256": lung.get("input_sha256"), "source_ct_sha256": sources,
        "errors": errors,
        "measurement": ("Largest 26-connected artery and airway masks; original lung-lobe union; "
                        "length m/L, volume mL/mL, branches/L."
                        if row.get("measurement_scope") == "largest_26" else
                        "Complete masks and all-component graph lengths; length m/L, volume mL/mL, branches/L."),
    }
    if errors:
        _record_failure(row, "normalization", ValueError(
            "; ".join(f"{field}: {message}" for field, message in errors.items())
        ))


def _run_case(
    case_dir: Path, args: argparse.Namespace, fiji: Path | None, fiji_error: str | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    row = {field: None for field in CSV_FIELDS}
    row.update(case=case_dir.name, status="error", stages={})
    from measurement_scope import scope_for_root, source_results_root, prepare_case_masks, get_mask
    scope = scope_for_root(args.output_root)
    row["measurement_scope"] = scope
    if getattr(args, "measurement_scope", None) not in (None, scope):
        raise ValueError("Requested measurement scope disagrees with the results profile")
    for stage in STAGES:
        row[f"{stage}_status"] = "not_run"
    with_volume = getattr(args, "with_lung_volume", False)
    with_airway = getattr(args, "with_airway", False)
    row["lung_volume_status"] = "not_run" if with_volume else "not_requested"
    row["airway_status"] = "not_run" if with_airway else "not_requested"
    row.update(dict.fromkeys(AIRWAY_STAGE_FIELDS, "not_run"))
    row["arterial_volume_status"] = "not_run"
    stage = STAGES[0]
    try:
        if scope == "largest_26":
            _stage_progress(args, "measurement_masks", "start")
            prepare_case_masks(case_dir, args.output_root)
            _stage_progress(args, "measurement_masks", "complete")
        _stage_progress(args, stage, "start")
        segmented = _segment_case(case_dir, args)
        row["stages"][stage] = segmented
        row[f"{stage}_status"] = (
            "skipped" if segmented["status"] == "ok" and not segmented["inference_ran"]
            else segmented["status"]
        )
        if segmented["status"] != "ok":
            raise RuntimeError(segmented.get("error") or "Segmentation failed")
        row.update(dice=segmented["dice"], iou=segmented["iou"])
        row["segmentation_provenance_status"] = segmented.get("provenance_status")
        if scope == "largest_26":
            import nibabel as nib
            import numpy as np
            import run_totalsegmentator_parse as segmentation
            selected_path, selected_info = get_mask(case_dir, args.output_root)
            reference_path, reference_info = get_mask(case_dir, args.output_root, source="reference")
            selected, reference = nib.load(selected_path), nib.load(reference_path)
            scores = segmentation._overlap_metrics(np.asarray(selected.dataobj), np.asarray(reference.dataobj),
                                                   lung_volume._voxel_volume_mm3(selected))
            row.update(main_component_dice=scores["dice"], main_component_iou=scores["iou"])
            row["measurement_masks"] = {"prediction": selected_info, "reference": reference_info}
        _stage_progress(args, stage, "complete", {"status": row[f"{stage}_status"]})

        # Mask volume is independent of Fiji and later skeleton failures.
        try:
            _stage_progress(args, "arterial_volume", "start")
            arterial = _arterial_volume_case(case_dir, args)
            row["stages"]["arterial_volume"] = arterial
            row["arterial_volume_status"] = arterial["status"]
            if arterial["status"] not in {"ok", "skipped"}:
                raise RuntimeError(arterial.get("error") or "Arterial volume measurement failed")
            milliliters = arterial["arterial_volume_ml"]
            if (not isinstance(milliliters, (int, float))
                    or not math.isfinite(milliliters) or milliliters < 0):
                raise ValueError("CT arterial volume must be finite and nonnegative")
            row["arterial_volume_ml"] = milliliters
            _stage_progress(args, "arterial_volume", "complete", {"status": arterial["status"]})
        except Exception as error:
            _record_failure(row, "arterial_volume", error)
            _stage_progress(args, "arterial_volume", "error", {"error": str(error)})

        stage = STAGES[1]
        _stage_progress(args, stage, "start")
        if fiji is None:
            raise RuntimeError(fiji_error or "Fiji is unavailable")
        skeleton = skeletonize._run_case(
            source_results_root(args.output_root) / "totalsegmentator" / case_dir.name,
            args.output_root / stage, fiji,
            overwrite=args.overwrite or segmented["inference_ran"],
            **({"input_path": get_mask(case_dir, args.output_root)[0]} if scope == "largest_26" else {}),
        )
        row["stages"][stage] = skeleton
        row[f"{stage}_status"] = skeleton["status"]
        if skeleton["status"] not in {"ok", "skipped"}:
            raise RuntimeError(skeleton.get("error") or "Skeletonization failed")
        row["skeleton_voxels"] = skeleton["skeleton_voxels"]
        _stage_progress(args, stage, "complete", {"status": skeleton["status"]})

        stage = STAGES[2]
        _stage_progress(args, stage, "start")
        analysis = analyze._run_case(
            args.output_root / "skeletonize3d" / case_dir.name,
            args.output_root / stage, fiji,
            overwrite=args.overwrite or skeleton["status"] == "ok",
        )
        row["stages"][stage] = analysis
        row[f"{stage}_status"] = analysis["status"]
        if analysis["status"] not in {"ok", "skipped"}:
            raise RuntimeError(analysis.get("error") or "Skeleton analysis failed")
        row.update({field: analysis[field] for field in ANALYSIS_METRICS})
        row["status"] = "error" if row["failed_stage"] else "ok"
        _stage_progress(args, stage, "complete", {"status": analysis["status"]})
    except Exception as error:
        _record_failure(row, stage, error)
        _stage_progress(args, stage, "error", {"error": str(error)})

    # Saved airway masks remain usable independently of the vascular result.
    if with_airway:
        try:
            _stage_progress(args, "airway", "start")
            airway_result = _airway_case(case_dir, args, fiji, fiji_error)
            row["stages"]["airway"] = airway_result
            row["airway_status"] = airway_result["status"]
            # Keep successful volume/skeleton work even if a later airway stage failed.
            row.update({field: airway_result.get(field) for field in AIRWAY_METRICS})
            row.update({field: airway_result.get(field, "not_run") for field in AIRWAY_STAGE_FIELDS})
            if airway_result["status"] != "ok":
                raise RuntimeError(airway_result.get("error") or "Airway quantification failed")
            _stage_progress(args, "airway", "complete", {"status": airway_result["status"]})
        except Exception as error:
            _record_failure(row, "airway", error)
            _stage_progress(args, "airway", "error", {"error": str(error)})
            for field in (*AIRWAY_METRICS, *AIRWAY_STAGE_FIELDS):
                row["stages"]["airway"].setdefault(field, row[field])

    # The lung branch depends only on the CT, even if the vascular branch failed.
    if with_volume:
        try:
            _stage_progress(args, "lung_volume", "start")
            volume = _lung_volume_case(case_dir, args)
            row["stages"]["lung_volume"] = volume
            row["lung_volume_status"] = volume["status"]
            if volume["status"] not in {"ok", "skipped"}:
                raise RuntimeError(volume.get("error") or "Lung volume measurement failed")
            liters = volume["lung_total_volume_l"]
            if not isinstance(liters, (int, float)) or not math.isfinite(liters) or liters <= 0:
                raise ValueError("CT lung volume must be finite and positive")
            row.update({field: volume[field] for field in lung_volume.VOLUME_FIELDS})
            _stage_progress(args, "lung_volume", "complete", {"status": volume["status"]})
        except Exception as error:
            _record_failure(row, "lung_volume", error)
            _stage_progress(args, "lung_volume", "error", {"error": str(error)})
    _stage_progress(args, "normalization", "start")
    normalize_case(row, with_lung_volume=with_volume, with_airway=with_airway)
    normalization_status = row.get("normalization_status")
    _stage_progress(args, "normalization", "error" if normalization_status == "error" else "complete",
                    {"status": normalization_status})
    row["runtime_sec"] = round(time.perf_counter() - started, 3)
    return row


def _write_reports(output_root: Path, rows: list[dict[str, Any]]) -> None:
    scopes = {row.get("measurement_scope", "all") for row in rows}
    if len(scopes) > 1:
        raise ValueError("Cannot publish a cohort with mixed measurement scopes")
    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / "pipeline_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    valid = [row for row in rows if row["status"] == "ok"]
    vascular = [row for row in rows if row["analyzeskeleton_status"] in {"ok", "skipped"}]
    branches = sum(row["branch_count"] for row in vascular)
    total_length = sum(row["total_branch_length_mm"] for row in vascular)
    report = {
        "measurement_scope": next(iter(scopes), "all"),
        "total_cases": len(rows), "successful_cases": len(valid),
        "failed_cases": len(rows) - len(valid),
        "vascular_successful_cases": len(vascular),
        "branch_count": branches, "total_branch_length_mm": total_length,
        "mean_branch_length_mm": total_length / branches if branches else 0.0,
        "cases": rows,
    }
    (output_root / "pipeline_report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def main() -> int:
    args = _parse_args()
    with_volume = getattr(args, "with_lung_volume", False)
    with_airway = getattr(args, "with_airway", False)
    requested = (*STAGES, "arterial_volume")
    if with_volume:
        requested += ("lung_volume",)
    if with_airway:
        requested += ("airway",)
    try:
        cases = discover_ct_cases(args.input_root, args.cases, args.limit)
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    fiji_error = None
    try:
        fiji = skeletonize._find_fiji(args.fiji)
        for plugin in ("Skeletonize3D_", "AnalyzeSkeleton_"):
            if not list((fiji.parent / "plugins").glob(f"{plugin}*.jar")):
                raise FileNotFoundError(f"Missing {plugin} plugin in {fiji.parent / 'plugins'}")
    except Exception as error:
        fiji, fiji_error = None, f"{type(error).__name__}: {error}"
    rows = []
    for index, case_dir in enumerate(cases, 1):
        if not args.quiet:
            print(f"[{index}/{len(cases)}] {case_dir.name}", flush=True)
        row = _run_case(case_dir, args, fiji, fiji_error)
        rows.append(row)
        if row["status"] != "ok":
            print(f"{case_dir.name}: {row['failed_stage']} failed: {row['error']}", file=sys.stderr)
        elif not args.quiet:
            states = ", ".join(f"{stage}={row[f'{stage}_status']}" for stage in requested)
            print(f"  {states}; {row['branch_count']} branches, "
                  f"mean length {row['mean_branch_length_mm']:.3f} mm", flush=True)

    _write_reports(args.output_root, rows)
    # Match standalone summary conventions: each summary describes this invocation.
    for stage in requested:
        from measurement_scope import scope_for_root
        if scope_for_root(args.output_root) == "largest_26" and stage in {"totalsegmentator", "lung_volume"}:
            continue
        results = [row["stages"][stage] for row in rows if stage in row["stages"]]
        if not results:
            continue
        directory = args.output_root / stage
        directory.mkdir(parents=True, exist_ok=True)
        if stage == "totalsegmentator":
            import run_totalsegmentator_parse as segmentation
            segmentation._write_metrics(results, directory)
        elif stage == "skeletonize3d":
            skeletonize._write_summary(directory, results)
        elif stage == "analyzeskeleton":
            analyze._write_summary(directory, results)
        elif stage == "arterial_volume":
            arterial_volume._write_summary(directory, results)
        elif stage == "lung_volume":
            lung_volume._write_summary(directory, results)
        elif stage == "airway":
            airway._write_summary(directory, results)
    return 1 if any(row["status"] != "ok" for row in rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())
