#!/usr/bin/env python3
"""Run TotalSegmentator on PARSE cases and compare lung-artery masks to labels.

The expected input layout is::

    data/parse/train/PA000005/image/PA000005.nii.gz
    data/parse/train/PA000005/label/PA000005.nii.gz

TotalSegmentator writes all masks from the ``lung_vessels`` task into the
case output directory. The ``lung_arteries.nii.gz`` mask is compared with the
binary PARSE label using Dice and IoU.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np

from dataset_cases import discover_ct_cases
from dataset_conversion import verify_case_conversion

TASK = "lung_vessels"
TARGET_MASK = "lung_arteries.nii.gz"
EXPECTED_MASKS = (
    "lung_airways.nii.gz",
    "lung_airways_wall.nii.gz",
    "lung_arteries.nii.gz",
    "lung_veins.nii.gz",
)
CSV_FIELDS = [
    "case",
    "status",
    "image",
    "label",
    "prediction",
    "report",
    "inference_ran",
    "input_sha256",
    "provenance_status",
    "dice",
    "iou",
    "pred_voxels",
    "label_voxels",
    "pred_volume_mm3",
    "label_volume_mm3",
    "volume_ratio",
    "shape",
    "voxel_spacing",
    "runtime_sec",
    "error",
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_hashes(directory: Path) -> dict[str, str]:
    return {name: _sha256(directory / name) for name in (*EXPECTED_MASKS, "run_report.json")}


def _write_provenance(path: Path, report: dict[str, Any]) -> None:
    temporary = path.with_suffix(".tmp")
    try:
        temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _cached_provenance(path: Path, case: str, input_hash: str) -> dict[str, Any] | None:
    try:
        cached = json.loads(path.read_text(encoding="utf-8"))
        if (
            cached["schema_version"] == 1
            and cached["case"] == case
            and cached["input_sha256"] == input_hash
            and cached["provenance_status"] in {"inference_verified", "legacy_adopted"}
            and cached["model_input_verified"] == (cached["provenance_status"] == "inference_verified")
            and cached["artifact_sha256"] == _artifact_hashes(path.parent)
        ):
            return cached
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return None


def _validate_masks(image: nib.Nifti1Image, directory: Path) -> None:
    for name in EXPECTED_MASKS:
        mask = nib.load(directory / name)
        errors = _geometry_errors(image, mask)
        if image.header.get_xyzt_units()[0] != mask.header.get_xyzt_units()[0]:
            errors.append("spatial units differ")
        if errors:
            raise ValueError(f"{name} geometry mismatch: {'; '.join(errors)}")
        data = np.asarray(mask.dataobj)
        if data.ndim != 3 or not np.all((data == 0) | (data == 1)):
            raise ValueError(f"{name} must be a finite binary 3-D mask")


def _infer(image_path: Path, case_output: Path, report_path: Path, args: argparse.Namespace) -> None:
    inference_limit = getattr(args, "max_concurrent_inference", 1)
    if type(inference_limit) is not int or inference_limit not in (1, 2, 4):
        raise ValueError("max_concurrent_inference must be 1, 2 or 4")
    if inference_limit > 1 and getattr(args, "inference_lock", None) is None:
        raise ValueError("Multiple inference slots require an explicit shared inference_lock")
    if getattr(args, "inference_lock", None) is not None:
        from inference_runtime import run_isolated_inference

        run_isolated_inference(
            image_path=image_path, output=case_output, report=report_path,
            device=args.device, quiet=args.quiet, parameters={"task": TASK},
            lock_path=args.inference_lock, stage="totalsegmentator",
            stage_callback=getattr(args, "stage_callback", None),
            max_concurrent_inference=inference_limit,
        )
        return
    from totalsegmentator.python_api import totalsegmentator

    totalsegmentator(
        input=image_path, output=case_output, task=TASK,
        device=args.device, quiet=args.quiet, report=report_path,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run TotalSegmentator lung-vessel segmentation on PARSE cases."
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("data/parse/train"),
        help="Directory containing case/image and case/label (default: data/parse/train).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/parse/results/totalsegmentator"),
        help="Directory for masks, reports, and metrics (default: data/parse/results/totalsegmentator).",
    )
    parser.add_argument(
        "--case",
        dest="cases",
        action="append",
        help="Case ID to run; repeat for multiple cases (default: all CT cases).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Process at most this many cases after case selection.",
    )
    parser.add_argument(
        "--device",
        default="gpu",
        help="TotalSegmentator device: gpu, gpu:N, cpu, or mps (default: gpu).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rerun inference even when lung_arteries.nii.gz already exists.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress TotalSegmentator progress output.",
    )
    parser.add_argument("--inference-workers", dest="max_concurrent_inference", type=int, choices=(1, 2, 4), default=1,
                        help="Isolated inference slots on the shared lock (default: 1).")
    parser.add_argument("--inference-lock", type=Path,
                        help="Shared dataset inference lock; required for multiple inference slots.")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be a positive integer")
    if args.device not in {"cpu", "gpu", "mps"} and not re.fullmatch(r"gpu:\d+", args.device):
        parser.error("--device must be cpu, gpu, gpu:N, or mps")
    if args.max_concurrent_inference > 1 and args.inference_lock is None:
        parser.error("Multiple --inference-workers require --inference-lock")
    return args


def _case_file(case_dir: Path, kind: str) -> Path:
    expected = case_dir / kind / f"{case_dir.name}.nii.gz"
    if expected.exists():
        return expected
    candidates = sorted((case_dir / kind).glob("*.nii")) + sorted(
        (case_dir / kind).glob("*.nii.gz")
    )
    if len(candidates) == 1:
        return candidates[0]
    raise FileNotFoundError(
        f"Expected one NIfTI file in {case_dir / kind}; found {len(candidates)}"
    )


def _discover_cases(input_root: Path, selected: list[str] | None, limit: int | None) -> list[Path]:
    return discover_ct_cases(input_root, selected, limit)


def _geometry_errors(reference: nib.Nifti1Image, other: nib.Nifti1Image) -> list[str]:
    errors: list[str] = []
    if reference.shape != other.shape:
        errors.append(f"shape {other.shape} != {reference.shape}")
    if not np.allclose(reference.affine, other.affine, rtol=1e-5, atol=1e-3):
        errors.append("affine differs")
    if not np.allclose(
        reference.header.get_zooms()[:3], other.header.get_zooms()[:3], rtol=1e-5, atol=1e-4
    ):
        errors.append("voxel spacing differs")
    return errors


def _overlap_metrics(prediction: np.ndarray, label: np.ndarray, voxel_volume: float) -> dict[str, Any]:
    prediction = prediction > 0
    label = label > 0
    pred_voxels = int(np.count_nonzero(prediction))
    label_voxels = int(np.count_nonzero(label))
    intersection = int(np.count_nonzero(prediction & label))
    union = pred_voxels + label_voxels - intersection

    if pred_voxels + label_voxels == 0:
        dice = 1.0
    else:
        dice = 2.0 * intersection / (pred_voxels + label_voxels)
    iou = 1.0 if union == 0 else intersection / union

    pred_volume = pred_voxels * voxel_volume
    label_volume = label_voxels * voxel_volume
    volume_ratio = pred_volume / label_volume if label_volume else None
    return {
        "dice": dice,
        "iou": iou,
        "pred_voxels": pred_voxels,
        "label_voxels": label_voxels,
        "pred_volume_mm3": pred_volume,
        "label_volume_mm3": label_volume,
        "volume_ratio": volume_ratio,
    }


def _run_case(case_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    case_output = args.output_root / case_dir.name
    case_output.mkdir(parents=True, exist_ok=True)
    image_path = case_dir / "image" / f"{case_dir.name}.nii.gz"
    label_path = case_dir / "label" / f"{case_dir.name}.nii.gz"
    prediction_path = case_output / TARGET_MASK
    report_path = case_output / "run_report.json"
    provenance_path = case_output / "segmentation_provenance.json"
    row: dict[str, Any] = {
        "case": case_dir.name,
        "status": "error",
        "image": str(image_path),
        "label": str(label_path),
        "prediction": str(prediction_path),
        "report": str(report_path),
        "inference_ran": False,
        "input_sha256": None,
        "provenance_status": None,
        "dice": None,
        "iou": None,
        "pred_voxels": None,
        "label_voxels": None,
        "pred_volume_mm3": None,
        "label_volume_mm3": None,
        "volume_ratio": None,
        "shape": None,
        "voxel_spacing": None,
        "runtime_sec": None,
        "error": None,
    }

    try:
        conversion = verify_case_conversion(case_dir, groups=("ct",))
        image_path = _case_file(case_dir, "image")
        row["image"] = str(image_path)
        image = nib.load(str(image_path))
        input_hash = _sha256(image_path)
        row["input_sha256"] = input_hash
        complete_output = all((case_output / mask).exists() for mask in EXPECTED_MASKS) and report_path.exists()
        cached = _cached_provenance(provenance_path, case_dir.name, input_hash) if complete_output else None
        stale = provenance_path.exists() and cached is None
        if cached is not None:
            row["provenance_status"] = cached["provenance_status"]
        # Check the required reference before starting expensive inference, while
        # keeping reference identities out of the CT prediction cache key.
        reference_conversion = verify_case_conversion(case_dir, groups=("artery",))
        label_path = _case_file(case_dir, "label")
        row["label"] = str(label_path)
        label_hash = _sha256(label_path)
        label_img = nib.load(str(label_path))
        geometry_errors = _geometry_errors(image, label_img)
        if geometry_errors:
            row["status"] = "geometry_mismatch"
            row["error"] = "; ".join(geometry_errors)
            return row
        label_data = np.asarray(label_img.dataobj)
        label_values = np.unique(label_data)
        if not np.all(np.isfinite(label_values)) or not np.all(np.isin(label_values, (0, 1))):
            raise ValueError(f"Artery label must contain only 0/1 values; found {label_values.tolist()}")
        if not complete_output or args.overwrite or stale:
            row["inference_ran"] = True
            # An interrupted/failed inference must never be adopted as a legacy cache.
            _write_provenance(provenance_path, {"schema_version": 1, "status": "incomplete"})
            _infer(image_path, case_output, report_path, args)
        if not prediction_path.exists():
            raise FileNotFoundError(f"TotalSegmentator did not create {prediction_path}")

        prediction_img = nib.load(str(prediction_path))
        geometry_errors = _geometry_errors(image, prediction_img)
        row["shape"] = list(image.shape)
        row["voxel_spacing"] = [float(value) for value in image.header.get_zooms()[:3]]
        if geometry_errors:
            row["status"] = "geometry_mismatch"
            row["error"] = "; ".join(geometry_errors)
            return row

        _validate_masks(image, case_output)

        if _sha256(image_path) != input_hash:
            raise RuntimeError("Input CT changed during segmentation or validation")
        if verify_case_conversion(case_dir, groups=("ct",)) != conversion:
            raise RuntimeError("CT conversion sources changed during segmentation")
        provenance_status = (
            "inference_verified" if row["inference_ran"]
            else cached["provenance_status"] if cached is not None else "legacy_adopted"
        )
        if provenance_status == "legacy_adopted" and cached is None:
            native_report = json.loads(report_path.read_text(encoding="utf-8"))
            original_input = native_report.get("input")
            if original_input and Path(original_input).name != image_path.name:
                raise ValueError("Legacy segmentation report names a different CT; use --overwrite")
        provenance = {
            "schema_version": 1, "case": case_dir.name, "input": str(image_path),
            "input_sha256": input_hash,
            "provenance_status": provenance_status,
            "model_input_verified": provenance_status == "inference_verified",
            "artifact_sha256": _artifact_hashes(case_output),
        }
        if conversion["sources"]:
            provenance["dataset_conversion"] = conversion
        if provenance_status == "legacy_adopted":
            provenance["note"] = (
                "Existing masks were adopted for this CT after geometry validation. "
                "The original model input was not hashed and cannot be verified retrospectively."
            )
        if cached is None or row["inference_ran"]:
            _write_provenance(provenance_path, provenance)
        row["provenance_status"] = provenance_status

        metrics = _overlap_metrics(
            np.asarray(prediction_img.dataobj), label_data,
            float(np.prod(image.header.get_zooms()[:3])),
        )
        if _sha256(label_path) != label_hash:
            raise RuntimeError("Reference artery label changed during validation")
        if verify_case_conversion(case_dir, groups=("artery",)) != reference_conversion:
            raise RuntimeError("Reference artery conversion sources changed during validation")
        row.update(metrics)
        row["status"] = "ok"
    except Exception as exc:  # keep the batch running and report the failed case
        row["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        row["runtime_sec"] = round(time.perf_counter() - started, 3)
    return row


def _write_metrics(rows: list[dict[str, Any]], output_root: Path) -> None:
    metrics_path = output_root / "metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    with (output_root / "failures.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(row for row in rows if row["status"] != "ok")

    valid = [row for row in rows if row["status"] == "ok"]
    summary: dict[str, Any] = {
        "task": TASK,
        "target_mask": TARGET_MASK,
        "total_cases": len(rows),
        "successful_cases": len(valid),
        "failed_cases": len(rows) - len(valid),
        "dice_mean": float(np.mean([row["dice"] for row in valid])) if valid else None,
        "dice_median": float(np.median([row["dice"] for row in valid])) if valid else None,
        "iou_mean": float(np.mean([row["iou"] for row in valid])) if valid else None,
        "iou_median": float(np.median([row["iou"] for row in valid])) if valid else None,
        "cases": rows,
    }
    with (output_root / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)


def main() -> int:
    args = _parse_args()
    cases = _discover_cases(args.input_root, args.cases, args.limit)
    args.output_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for index, case_dir in enumerate(cases, start=1):
        print(f"[{index}/{len(cases)}] {case_dir.name}", flush=True)
        row = _run_case(case_dir, args)
        rows.append(row)
        if row["status"] == "ok":
            print(f"  status=ok dice={row['dice']:.4f} iou={row['iou']:.4f}", flush=True)
        else:
            print(f"  status={row['status']} error={row['error']}", flush=True)

    _write_metrics(rows, args.output_root)
    failed = sum(row["status"] != "ok" for row in rows)
    print(f"Finished {len(rows)} case(s): {len(rows) - failed} ok, {failed} failed.")
    print(f"Metrics: {args.output_root / 'metrics.csv'}")
    print(f"Summary: {args.output_root / 'summary.json'}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
