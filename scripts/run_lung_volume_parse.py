#!/usr/bin/env python3
"""Segment five lung lobes and measure CT lung volumes on the saved NIfTI grid."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import re
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np

from dataset_cases import discover_ct_cases
from dataset_conversion import verify_case_conversion


LUNG_LOBES = (
    "lung_upper_lobe_left", "lung_lower_lobe_left", "lung_upper_lobe_right",
    "lung_middle_lobe_right", "lung_lower_lobe_right",
)
REGIONS = (*LUNG_LOBES, "lung_left", "lung_right", "lung_total")
VOLUME_FIELDS = [
    "lung_total_volume_ml", "lung_total_volume_l", "lung_left_volume_ml",
    "lung_right_volume_ml", *(f"{lobe}_volume_ml" for lobe in LUNG_LOBES),
]
CSV_FIELDS = [
    "case", "status", *VOLUME_FIELDS, "image", "input_sha256", "report",
    "inference_ran", "runtime_sec", "error",
]
SCHEMA_VERSION = 1
PARAMETERS = {
    "task": "total", "roi_subset": list(LUNG_LOBES), "fast": False,
    "fastest": False, "ml": False, "save_lowres": False, "statistics": False,
    "remove_small_blobs": False,
}


def _sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _case_file(case_dir: Path) -> Path:
    expected = case_dir / "image" / f"{case_dir.name}.nii.gz"
    if expected.is_file():
        return expected
    candidates = sorted((case_dir / "image").glob("*.nii*"))
    candidates = [path for path in candidates if path.name.endswith((".nii", ".nii.gz"))]
    if len(candidates) != 1:
        raise FileNotFoundError(f"Expected one CT NIfTI in {case_dir / 'image'}")
    return candidates[0]


def _discover_cases(root: Path, selected: list[str] | None, limit: int | None) -> list[Path]:
    return discover_ct_cases(root, selected, limit)


def _voxel_volume_mm3(image: nib.Nifti1Image) -> float:
    if len(image.shape) != 3:
        raise ValueError("Expected a 3-D image")
    unit = image.header.get_xyzt_units()[0]
    scale = {"mm": 1.0, "meter": 1000.0, "micron": 0.001}.get(unit)
    if scale is None:
        raise ValueError(f"Known NIfTI spatial units are required, got {unit!r}")
    if not np.all(np.isfinite(image.affine)):
        raise ValueError("Image affine must be finite")
    volume = float(abs(np.linalg.det(image.affine[:3, :3]))) * scale**3
    if not np.isfinite(volume) or volume <= 0:
        raise ValueError("Voxel volume must be finite and positive")
    return volume


def _binary_data(image: nib.Nifti1Image) -> np.ndarray:
    data = np.asarray(image.dataobj)
    values = np.unique(data)
    if not (np.all(np.isin(values, (0, 1))) or np.all(np.isin(values, (0, 255)))):
        raise ValueError("Expected a finite binary mask (0/1 or 0/255)")
    return data > 0


def calculate_mask_volume(image: nib.Nifti1Image) -> dict[str, Any]:
    """Count mask voxels; affine determinant and declared units define volume."""
    voxel_volume = _voxel_volume_mm3(image)
    count = int(np.count_nonzero(_binary_data(image)))
    return {"voxel_count": count, "voxel_volume_mm3": voxel_volume,
            "volume_ml": count * voxel_volume / 1000.0,
            "volume_l": count * voxel_volume / 1_000_000.0}


def _check_geometry(reference: nib.Nifti1Image, image: nib.Nifti1Image) -> None:
    _voxel_volume_mm3(image)
    if reference.shape != image.shape or not np.allclose(
        reference.affine, image.affine, rtol=1e-5, atol=1e-4
    ) or reference.header.get_xyzt_units()[0] != image.header.get_xyzt_units()[0]:
        raise ValueError("Mask shape, affine or spatial units differ from the CT")
    if not np.allclose(reference.header.get_zooms()[:3], image.header.get_zooms()[:3]):
        raise ValueError("Mask voxel spacing differs from the CT")


def _mask_image(reference: nib.Nifti1Image, data: np.ndarray) -> nib.Nifti1Image:
    header = reference.header.copy()
    header.set_data_dtype(np.uint8)
    result = nib.Nifti1Image(data.astype(np.uint8), reference.affine, header)
    result.set_qform(*reference.get_qform(coded=True))
    result.set_sform(*reference.get_sform(coded=True))
    return result


def merge_lung_masks(lobes: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    left = lobes[LUNG_LOBES[0]] | lobes[LUNG_LOBES[1]]
    right = lobes[LUNG_LOBES[2]] | lobes[LUNG_LOBES[3]] | lobes[LUNG_LOBES[4]]
    return {"lung_left": left, "lung_right": right, "lung_total": left | right}


def _measure_lobes(reference: nib.Nifti1Image, directory: Path) -> tuple[dict, dict, dict, list]:
    arrays = {}
    images = {}
    for name in LUNG_LOBES:
        image = nib.load(directory / f"{name}.nii.gz")
        _check_geometry(reference, image)
        arrays[name] = _binary_data(image)
        images[name] = image
    merged = merge_lung_masks(arrays)
    images.update({name: _mask_image(reference, array) for name, array in merged.items()})
    regions = {name: calculate_mask_volume(image) for name, image in images.items()}
    if regions["lung_total"]["voxel_count"] == 0:
        raise ValueError("Lung union is empty; CT lung volume cannot be zero")
    warnings = [f"Empty lobe: {name}" for name in LUNG_LOBES
                if regions[name]["voxel_count"] == 0]
    excess = sum(regions[name]["voxel_count"] for name in LUNG_LOBES) - regions["lung_total"]["voxel_count"]
    assignments = np.zeros(reference.shape, dtype=np.uint8)
    for array in arrays.values():
        assignments += array
    overlap_count = int(np.count_nonzero(assignments > 1))
    if excess:
        warnings.append(f"Lobe masks overlap: {excess} excess voxel assignments; union defines total volume")
    metrics = {f"{name}_volume_ml": region["volume_ml"] for name, region in regions.items()}
    metrics["lung_total_volume_l"] = regions["lung_total"]["volume_l"]
    metrics["overlap_voxel_count"] = overlap_count
    metrics["overlap_excess_voxel_assignments"] = excess
    return images, regions, metrics, warnings


def segment_lung_lobes(
    image_path: Path, output: Path, device: str, quiet: bool, *,
    inference_lock: Path | None = None, stage_callback=None,
) -> None:
    if inference_lock is not None:
        from inference_runtime import run_isolated_inference

        run_isolated_inference(
            image_path=image_path, output=output, report=output / "totalsegmentator_report.json",
            device=device, quiet=quiet, parameters=PARAMETERS, lock_path=inference_lock,
            stage="lung_volume", stage_callback=stage_callback,
        )
        return
    # Import only for inference: measurement, CLI help and tests do not need torch.
    from totalsegmentator.python_api import totalsegmentator

    totalsegmentator(input=image_path, output=output, device=device, quiet=quiet,
                    report=output / "totalsegmentator_report.json", **PARAMETERS)


def _read_report(path: Path) -> dict | None:
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
        return report if isinstance(report, dict) else None
    except (OSError, ValueError):
        return None


def _version() -> str | None:
    try:
        return importlib.metadata.version("TotalSegmentator")
    except importlib.metadata.PackageNotFoundError:
        return None


def _load_validated_result(image_path: Path, case_output: Path) -> tuple[dict, dict]:
    """Read masks/report only after checking scan identity and artifact integrity."""
    conversion = verify_case_conversion(image_path.parent.parent, groups=("ct",))
    report = _read_report(case_output / "lung_volume_report.json")
    if not report or report.get("status") not in {"ok", "skipped"}:
        raise ValueError("A successful lung volume report is required")
    if report.get("schema_version") != SCHEMA_VERSION or report.get("parameters") != PARAMETERS:
        raise ValueError("Lung volume report settings or schema are stale")
    inference = report.get("inference")
    if not isinstance(inference, dict) or inference.get("parameters") != PARAMETERS:
        raise ValueError("Lung segmentation provenance is missing or stale")
    input_hash = _sha256(image_path)
    if report.get("input_sha256") != input_hash or inference.get("input_sha256") != input_hash:
        raise ValueError("Lung volume results belong to a different CT; rerun lung volume analysis")
    artifacts = report.get("artifacts_sha256")
    if not isinstance(artifacts, dict):
        raise ValueError("Lung volume artifact hashes are missing")
    for filename in (*(f"{name}.nii.gz" for name in REGIONS), "totalsegmentator_report.json"):
        if artifacts.get(filename) != _sha256(case_output / filename):
            raise ValueError(f"Lung mask changed: {filename}; rerun with --measure-only")
    reference = nib.load(image_path)
    images, regions, metrics, _ = _measure_lobes(reference, case_output)
    if report.get("regions") != regions or any(report.get(key) != value for key, value in metrics.items()):
        raise ValueError("Lung volume report does not match its masks")
    for name in ("lung_left", "lung_right", "lung_total"):
        actual = nib.load(case_output / f"{name}.nii.gz")
        _check_geometry(reference, actual)
        if not np.array_equal(_binary_data(actual), np.asarray(images[name].dataobj) > 0):
            raise ValueError(f"{name} does not equal its lobe union")
    if verify_case_conversion(image_path.parent.parent, groups=("ct",)) != conversion:
        raise ValueError("CT conversion sources changed during lung volume validation")
    return images, report


def _run_case(case_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    output = args.output_root / case_dir.name
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "lung_volume_report.json"
    previous = _read_report(report_path)
    row = {field: None for field in CSV_FIELDS}
    row.update(case=case_dir.name, status="error", report=str(report_path), inference_ran=False)
    report: dict[str, Any] = {"schema_version": SCHEMA_VERSION, "parameters": PARAMETERS}
    try:
        if args.overwrite and getattr(args, "measure_only", False):
            raise ValueError("--measure-only and --overwrite cannot be combined")
        conversion = verify_case_conversion(case_dir, groups=("ct",))
        if conversion["sources"]:
            report["dataset_conversion"] = conversion
        image_path = _case_file(case_dir)
        row.update(image=str(image_path), input_sha256=_sha256(image_path))
        reference = nib.load(image_path)
        voxel_volume = _voxel_volume_mm3(reference)
        all_lobes = all((output / f"{name}.nii.gz").is_file() for name in LUNG_LOBES)
        inference = previous.get("inference", {}) if previous else {}
        if not isinstance(inference, dict):
            inference = {}
        same_input = bool(previous and previous.get("input_sha256") == row["input_sha256"]
                          and inference.get("input_sha256") == row["input_sha256"])
        same_settings = bool(previous and previous.get("parameters") == PARAMETERS
                             and inference.get("parameters") == PARAMETERS)
        native_report = _read_report(output / "totalsegmentator_report.json")
        native_valid = bool(native_report and native_report.get("task") == "total")
        measure_only = getattr(args, "measure_only", False)
        if measure_only and (not all_lobes or not same_input or not same_settings or not native_valid):
            raise ValueError("--measure-only requires five existing lobes and matching CT/settings and TotalSegmentator reports")
        version = _version()
        needs_inference = not measure_only and (
            args.overwrite or not all_lobes or not same_input or not same_settings or not native_valid
            or inference.get("totalsegmentator_version") != version
        )
        if not needs_inference:
            try:
                _, cached = _load_validated_result(image_path, output)
            except (OSError, ValueError, KeyError, TypeError):
                pass
            else:
                row.update({field: cached[field] for field in VOLUME_FIELDS})
                row["status"] = "skipped"
                row["runtime_sec"] = round(time.perf_counter() - started, 3)
                return row
        with tempfile.TemporaryDirectory(prefix=".lung-volume-", dir=output) as temporary:
            staging = Path(temporary)
            source = output
            if needs_inference:
                if reference.header.get_xyzt_units()[0] != "mm":
                    raise ValueError("TotalSegmentator inference requires a CT with spatial units in mm")
                row["inference_ran"] = True
                inference_options = {}
                if getattr(args, "inference_lock", None) is not None:
                    inference_options.update(inference_lock=args.inference_lock,
                                             stage_callback=getattr(args, "stage_callback", None))
                segment_lung_lobes(image_path, staging, args.device, args.quiet, **inference_options)
                native_report = _read_report(staging / "totalsegmentator_report.json")
                if not native_report or native_report.get("task") != "total":
                    raise ValueError("TotalSegmentator did not write its inference report")
                inference = {
                    "totalsegmentator_version": version, "device": args.device,
                    "parameters": PARAMETERS,
                    "model_task_ids": [291], "crop_model_task_id": 298,
                    "model_spacing_mm": 1.5, "crop_spacing_mm": 6.0,
                    "input_sha256": row["input_sha256"],
                    "lobe_sha256": {f"{name}.nii.gz": _sha256(staging / f"{name}.nii.gz") for name in LUNG_LOBES},
                }
                source = staging
            lobe_hashes = {f"{name}.nii.gz": _sha256(source / f"{name}.nii.gz") for name in LUNG_LOBES}
            images, regions, metrics, warnings = _measure_lobes(reference, source)
            for name in ("lung_left", "lung_right", "lung_total"):
                nib.save(images[name], staging / f"{name}.nii.gz")
            if _sha256(image_path) != row["input_sha256"]:
                raise ValueError("CT changed during lung volume processing")
            if verify_case_conversion(case_dir, groups=("ct",)) != conversion:
                raise ValueError("CT conversion sources changed during lung volume processing")
            if any(_sha256(source / name) != value for name, value in lobe_hashes.items()):
                raise ValueError("Lobe masks changed during lung volume processing")
            names = (*REGIONS, "totalsegmentator_report") if needs_inference else ("lung_left", "lung_right", "lung_total")
            for name in names:
                filename = f"{name}.json" if name == "totalsegmentator_report" else f"{name}.nii.gz"
                (staging / filename).replace(output / filename)
            hashes = {f"{name}.nii.gz": _sha256(output / f"{name}.nii.gz") for name in REGIONS}
            hashes["totalsegmentator_report.json"] = _sha256(output / "totalsegmentator_report.json")
            report.update(
                inference=inference, geometry={"shape": list(reference.shape), "affine": reference.affine.tolist(),
                    "spatial_unit": reference.header.get_xyzt_units()[0], "voxel_volume_mm3": voxel_volume},
                regions=regions, warnings=warnings, artifacts_sha256=hashes,
                masks_modified_since_inference=any(hashes.get(name) != value for name, value in inference.get("lobe_sha256", {}).items()),
                measurement="CT lung volume: union of five lung-lobe masks; no HU threshold or vascular subtraction",
            )
            row.update(metrics, status="ok")
    except Exception as error:
        row["error"] = f"{type(error).__name__}: {error}"
        # Keep successful provenance readable if a failed forced run left its masks intact.
        if previous and previous.get("input_sha256") == row.get("input_sha256"):
            report["inference"] = previous.get("inference", {})
    row["runtime_sec"] = round(time.perf_counter() - started, 3)
    report.update(row)
    with tempfile.NamedTemporaryFile("w", dir=output, suffix=".json", delete=False, encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")
        temporary_report = Path(handle.name)
    temporary_report.replace(report_path)
    return row


def _write_summary(output_root: Path, rows: list[dict[str, Any]]) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / "lung_volume_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    successful = sum(row["status"] in {"ok", "skipped"} for row in rows)
    report = {"total_cases": len(rows), "successful_cases": successful,
              "failed_cases": len(rows) - successful, "cases": rows}
    (output_root / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=Path("data/parse/train"), help="Cases containing image/<case>.nii.gz; labels are not required.")
    parser.add_argument("--output-root", type=Path, default=Path("data/parse/results/lung_volume"))
    parser.add_argument("--case", dest="cases", action="append")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--device", default="gpu")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--overwrite", action="store_true", help="Rerun lung-lobe inference only.")
    mode.add_argument("--measure-only", action="store_true", help="Measure existing lobes without loading or running the model.")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.device not in {"cpu", "gpu", "mps"} and not re.fullmatch(r"gpu:\d+", args.device):
        parser.error("--device must be cpu, gpu, gpu:N or mps")
    return args


def main() -> int:
    args = _parse_args()
    try:
        cases = _discover_cases(args.input_root, args.cases, args.limit)
    except (OSError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    rows = []
    for case in cases:
        row = _run_case(case, args)
        rows.append(row)
        if row["status"] == "error":
            print(f"{case.name}: {row['error']}", file=sys.stderr)
        elif not args.quiet:
            print(f"{case.name}: {row['status']} CT lung volume={row['lung_total_volume_l']:.4f} L", flush=True)
    _write_summary(args.output_root, rows)
    return int(any(row["status"] == "error" for row in rows))


if __name__ == "__main__":
    sys.exit(main())
