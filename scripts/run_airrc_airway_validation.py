#!/usr/bin/env python3
"""Validate raw airway lumen/wall Dice and IoU against prepared AirRC labels.

Training-case overlap with the deployed model remains unverified.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import re
import sys
import uuid
from types import SimpleNamespace

import nibabel as nib
import numpy as np

import run_skeletonize3d_parse as skeletonize
import run_totalsegmentator_parse as segmentation
from dataset_cases import discover_ct_cases
from run_lung_volume_parse import _check_geometry, _sha256, _voxel_volume_mm3


REFERENCE_LIMITATIONS = [
    "One-mm reference grid limits wall-boundary resolution.",
    "Lumen/wall separation was initialized by morphology before expert review.",
    "Training-case overlap with the deployed lung_vessels model is unverified.",
]
SEGMENTATION_CODE_NAMES = (
    "run_airrc_airway_validation.py", "prepare_airrc.py", "run_totalsegmentator_parse.py",
    "dataset_conversion.py", "inference_runtime.py", "dataset_cases.py",
    "run_skeletonize3d_parse.py", "run_lung_volume_parse.py",
)
SEGMENTATION_COMPUTATION_FILES = SEGMENTATION_CODE_NAMES
OVERLAP_POLICY = {
    "definition": "raw_voxel_dice_iou_v1", "mask_selection": "none",
    "reference_labels": {"airways": 1, "airways_wall": 2},
    "dice": "2 * intersection_voxels / (prediction_voxels + reference_voxels)",
    "iou": "intersection_voxels / union_voxels", "both_empty": "unavailable",
}


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _verify(hashes: dict) -> None:
    for name, expected in hashes.items():
        if _sha256(Path(name)) != expected:
            raise ValueError(f"Source changed: {name}")


def _file(case_dir: Path, kind: str) -> Path:
    return case_dir / kind / f"{case_dir.name}.nii.gz"


def _binary_image(path: Path, reference=None):
    image = skeletonize._load_binary_image(path)
    _voxel_volume_mm3(image)
    if reference is not None:
        _check_geometry(reference, image)
    return image


def overlap_metrics(prediction, reference) -> dict:
    """Voxel Dice/IoU, with both-empty results unavailable rather than perfect."""
    pred, ref = np.asarray(prediction, bool), np.asarray(reference, bool)
    if pred.shape != ref.shape:
        raise ValueError("Overlap arrays must have the same shape")
    p, r = int(pred.sum()), int(ref.sum())
    intersection = int(np.count_nonzero(pred & ref))
    union = p + r - intersection
    return {"dice": 2 * intersection / (p + r) if p + r else None,
            "iou": intersection / union if union else None,
            "prediction_voxels": p, "reference_voxels": r,
            "intersection_voxels": intersection, "union_voxels": union,
            "status": "available" if union else "both_masks_empty"}


def _model_identity():
    model = Path.home() / ".totalsegmentator/nnunet/results/Dataset117_lung_airways_arteries_veins_282subj/nnUNetTrainerSkeletonRecall__nnUNetPlans__3d_fullres"
    paths = [model / "dataset.json", model / "plans.json", model / "fold_0/checkpoint_final.pth"]
    return {"task": "lung_vessels", "package_version": importlib.metadata.version("totalsegmentator"),
            "installation_files_sha256": {str(p): _sha256(p) for p in paths if p.is_file()},
            "cache_identity_policy": "Current installation is recorded; original cached inference weights require their own recorded identity."}


def _csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _segmentation_reports(report_dir, manifest, records):
    """Publish raw mask overlaps without placeholder airway measurement tables."""
    report_dir.mkdir(parents=True, exist_ok=True)
    rows = [{"case": record["case"], "mask": mask, **scores}
            for record in records for mask, scores in record.get("overlap", {}).items()]
    _csv(report_dir / "segmentation_overlap.csv", rows,
         ["case", "mask", "dice", "iou", "prediction_voxels", "reference_voxels",
          "intersection_voxels", "union_voxels", "status"])
    _write(report_dir / "validation_report.json", {"manifest": manifest, "cases": records})
    lines = ["# AirRC airway segmentation overlap", "",
             "Raw airway lumen and airway wall masks are compared on the verified annotation grid. "
             "Tracheal voxels are included as segmented. No component selection or derived airway measurements are applied.",
             "", "Both-empty masks have unavailable Dice and IoU rather than a perfect score.",
             "", "| Case | Overlap execution | Prediction |", "|---|---|---|"]
    lines.extend(f"| {r['case']} | {r['status']} | {r['prediction_status']} |" for r in records)
    lines += ["", "- [Raw lumen and wall Dice/IoU with voxel counts](segmentation_overlap.csv)",
              "- [Input geometry, inference identities and provenance](validation_report.json)",
              "", "## Interpretation limits", "",
              "This is label-comparison validation. Model training-case overlap is unverified; "
              "independent external generalization has not been established.", ""]
    lines.extend(f"- {item}" for item in manifest["reference_limitations"])
    (report_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _input_identity(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": _sha256(path)}


def _overlap_geometry(image):
    """Record the grid checked before any raw voxel overlap is computed."""
    forms = {}
    for name in ("qform", "sform"):
        matrix, code = getattr(image, f"get_{name}")(coded=True)
        forms[name] = {"code": int(code), "matrix": matrix.tolist() if matrix is not None else None}
    return {"shape": list(image.shape), "affine": image.affine.tolist(),
            "spatial_units": image.header.get_xyzt_units()[0],
            "voxel_spacing": [float(v) for v in image.header.get_zooms()[:3]], **forms}


def _run_segmentation_only(args, cases, dataset):
    """Reuse shared inference/cache validation, then measure two unselected masks."""
    from prepare_airrc import verify_airrc_conversion
    import shutil

    root = args.output_root.resolve() / args.run_id
    report_dir = args.report_root.resolve() / args.run_id
    if root.exists() or report_dir.exists():
        raise ValueError("Use a new run ID; validation publications are immutable")
    manifest = {
        "schema_version": 1, "dataset": "AirRC", "run_id": args.run_id,
        "created_utc": datetime.now(timezone.utc).isoformat(), "cases": [p.name for p in cases],
        "input_root": str(args.input_root.resolve()), "scope": "segmentation_overlap",
        "validation_scope": "segmentation_overlap", "measurement_scope": "raw_voxel_masks",
        "metrics": ["dice", "iou"], "profile": None, "profile_source": None, "profile_sha256": None,
        "overlap_policy": OVERLAP_POLICY, "trachea_policy": "include_as_segmented",
        "external_independence": "unverified", "potential_training_overlap": True,
        "reference_limitations": REFERENCE_LIMITATIONS,
        "device": args.device, "reference_only": False, "model_identity": _model_identity(),
        "inference_workers": getattr(args, "inference_workers", 1),
        "prediction_cases": [p.name for p in cases], "reference_initialized_wall_cases": [],
        "versions": {p: importlib.metadata.version(p) for p in ("numpy", "scipy", "nibabel", "totalsegmentator")},
        "code_sha256": {str(Path(__file__).with_name(name).resolve()): _sha256(Path(__file__).with_name(name))
                        for name in SEGMENTATION_CODE_NAMES},
    }
    root.mkdir(parents=True)
    snapshot = root / "source_code"
    snapshot.mkdir()
    for path in manifest["code_sha256"]:
        shutil.copyfile(path, snapshot / Path(path).name)
    manifest["code_snapshot_sha256"] = {str(snapshot / Path(path).name): digest
                                        for path, digest in manifest["code_sha256"].items()}
    _verify(manifest["code_sha256"])
    _verify(manifest["code_snapshot_sha256"])
    _write(root / "run_manifest.json", manifest)
    _write(root / "execution_status.json", {"status": "running", "scope": "segmentation_overlap",
           "validation_scope": "segmentation_overlap", "cases_requested": len(cases),
           "segmentation_cases_completed": 0, "prediction_cases_completed": 0,
           "accuracy_acceptance": "not_assessed"})
    records = []
    for case_dir in cases:
        case = case_dir.name
        print(f"{case}: raw lumen/wall segmentation overlap", flush=True)
        record = {"case": case, "validation_scope": "segmentation_overlap", "status": "error",
                  "prediction_status": "not_started"}
        try:
            # The shared segmentation driver also verifies its required artery
            # reference; no vascular measurements are included in this publication.
            conversion = verify_airrc_conversion(case_dir, groups=("ct", "airway", "airway_wall", "artery"))
            image_path = _file(case_dir, "image").resolve()
            image = nib.load(image_path)
            _voxel_volume_mm3(image)
            if image.header.get_xyzt_units()[0] != "mm":
                raise ValueError("AIRRC overlap requires the prepared millimeter annotation grid")
            references = {organ: _binary_image(_file(case_dir, kind), image)
                          for organ, kind in (("airways", "label_airway"), ("airways_wall", "label_airway_wall"))}
            prediction_dir = args.prediction_root.resolve() / case
            provenance = prediction_dir / "segmentation_provenance.json"
            cached = segmentation._cached_provenance(provenance, case, _sha256(image_path))
            if cached is not None and cached.get("provenance_status") != "inference_verified":
                raise ValueError("AIRRC validation requires verified model inference, not adopted masks")
            if cached is not None and cached.get("model_identity") is not None and (
                    cached["model_identity"] != manifest["model_identity"]):
                raise ValueError("Cached inference model identity differs from the frozen installed model identity")
            if args.device == "gpu" and cached is None:
                import torch
                if not torch.cuda.is_available():
                    record["prediction_status"] = "pending_gpu_unavailable"
                    raise RuntimeError("GPU unavailable and no verified inference cache is available")
            outcome = segmentation._run_case(case_dir, SimpleNamespace(
                output_root=args.prediction_root.resolve(), device=args.device, overwrite=False, quiet=False,
                inference_lock=dataset / "results" / ".inference.lock",
                max_concurrent_inference=getattr(args, "inference_workers", 1)))
            # Retain execution/cache evidence, excluding the shared driver's
            # compatibility artery measurements from the Dice/IoU-only report.
            record["segmentation"] = {key: outcome.get(key) for key in
                                      ("case", "status", "image", "report", "inference_ran",
                                       "input_sha256", "provenance_status", "runtime_sec", "error")}
            if outcome.get("status") != "ok":
                record["prediction_status"] = "inference_error"
                raise RuntimeError(outcome.get("error") or "Shared segmentation failed")
            state = segmentation._cached_provenance(provenance, case, _sha256(image_path))
            if state is None or state.get("provenance_status") != "inference_verified":
                raise ValueError("AIRRC validation requires a complete, hash-verified inference cache")
            if not outcome.get("inference_ran") and state.get("model_identity") is not None and (
                    state["model_identity"] != manifest["model_identity"]):
                raise ValueError("Cached inference model identity differs from the frozen installed model identity")
            if outcome.get("inference_ran"):
                state["model_identity"] = manifest["model_identity"]
                _write(provenance, state)
            record["prediction_model_identity"] = state.get("model_identity")
            native_report = json.loads((prediction_dir / "run_report.json").read_text())
            record["prediction_execution_policy"] = native_report.get("execution_policy")
            record["prediction_inference_concurrency_status"] = (
                "recorded" if isinstance(record["prediction_execution_policy"], dict)
                and record["prediction_execution_policy"].get("inference_concurrency") else "legacy_not_recorded")
            record["prediction_weights_identity_status"] = (
                "recorded" if state.get("model_identity") else "legacy_original_weights_unverified")
            inputs = {"image": _input_identity(image_path), "prediction": {}, "reference": {},
                      "segmentation_provenance": _input_identity(provenance)}
            hashes = dict(conversion["source_sha256"])
            hashes[inputs["segmentation_provenance"]["path"]] = inputs["segmentation_provenance"]["sha256"]
            hashes.update({str(prediction_dir / name): digest for name, digest in state["artifact_sha256"].items()})
            overlaps, geometries = {}, {"image": _overlap_geometry(image), "prediction": {}, "reference": {}}
            for organ, kind in (("airways", "label_airway"), ("airways_wall", "label_airway_wall")):
                prediction_path = prediction_dir / f"lung_{organ}.nii.gz"
                reference_path = _file(case_dir, kind)
                inputs["prediction"][organ] = _input_identity(prediction_path)
                inputs["reference"][organ] = _input_identity(reference_path)
                prediction = _binary_image(prediction_path, image)
                reference = references[organ]
                geometries["prediction"][organ] = _overlap_geometry(prediction)
                geometries["reference"][organ] = _overlap_geometry(reference)
                overlaps[organ] = overlap_metrics(np.asarray(prediction.dataobj), np.asarray(reference.dataobj))
                for identity in (inputs["prediction"][organ], inputs["reference"][organ]):
                    hashes[identity["path"]] = identity["sha256"]
            hashes[inputs["image"]["path"]] = inputs["image"]["sha256"]
            _verify(hashes)
            record.update(status="segmentation_ready", prediction_status="complete", overlap=overlaps,
                          overlap_inputs=inputs, overlap_geometry=geometries, source_sha256=hashes,
                          dataset_conversion=conversion, cache_verified=True)
        except (OSError, ValueError, KeyError, RuntimeError, TypeError, MemoryError) as error:
            record.update(status="error", error=f"{type(error).__name__}: {error}")
            if record["prediction_status"] == "not_started":
                record["prediction_status"] = "validation_error"
        records.append(record)
        _write(root / case / "validation.json", record)
        _segmentation_reports(report_dir, manifest, records)
        print(f"{case}: {record['status']}; prediction={record['prediction_status']}", flush=True)
    _verify(manifest["code_sha256"])
    _verify(manifest["code_snapshot_sha256"])
    for record in records:
        _verify(record.get("source_sha256", {}))
    complete = sum(r["status"] == "segmentation_ready" and r["prediction_status"] == "complete" for r in records)
    _write(root / "execution_status.json", {"status": "complete" if complete == len(cases) else "partial",
           "scope": "segmentation_overlap", "validation_scope": "segmentation_overlap",
           "segmentation_cases_completed": complete, "prediction_cases_completed": complete,
           "accuracy_acceptance": "not_assessed", "cases_requested": len(cases)})
    return records


def run(args):
    cases = discover_ct_cases(args.input_root, args.case, args.limit)
    if not cases or any(not re.fullmatch(r"AIRRC\d{6}", case.name) for case in cases):
        raise ValueError("Prepared AIRRC case directories are required")
    dataset = args.input_root.resolve().parent
    if dataset.name != "airrc":
        raise ValueError("Prepared inputs must be under the airrc dataset tree")
    for path in (args.output_root, args.report_root, args.prediction_root):
        if not path.resolve().is_relative_to(dataset) or path.resolve().is_relative_to(args.input_root.resolve()):
            raise ValueError("All generated outputs must stay in a separate part of data/airrc")
        if any(part.is_symlink() for part in (path, *path.parents)):
            raise ValueError("Validation output paths must not contain symlinks")
        if any(path.resolve().is_relative_to(dataset / name) or (dataset / name).is_relative_to(path.resolve())
               for name in ("source_ct", "source_annotations")):
            raise ValueError("Generated validation outputs must not overlap original source directories")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", args.run_id):
        raise ValueError("Invalid run ID")
    if getattr(args, "scope", "segmentation_overlap") != "segmentation_overlap":
        raise ValueError("Only segmentation_overlap validation is supported")
    if getattr(args, "inference_workers", 1) not in (1, 2, 4):
        raise ValueError("Inference workers must be 1, 2 or 4")
    return _run_segmentation_only(args, cases, dataset)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=Path("data/airrc/cases"))
    parser.add_argument("--output-root", type=Path, default=Path("data/airrc/results/airway_validation"))
    parser.add_argument("--report-root", type=Path, default=Path("data/airrc/reports/airway_validation"))
    parser.add_argument("--prediction-root", type=Path, default=Path("data/airrc/results/totalsegmentator"))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--case", action="append")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--scope", choices=("segmentation_overlap",), default="segmentation_overlap")
    parser.add_argument("--segmentation-only", action="store_true", default=True,
                        help="Compatibility alias for the default raw lumen/wall Dice and IoU validation.")
    parser.add_argument("--inference-workers", type=int, choices=(1, 2, 4), default=1,
                        help="Shared inference slots for segmentation-only validation (default: 1).")
    parser.add_argument("--device", choices=("gpu", "cpu"), default="gpu")
    args = parser.parse_args(argv)
    try:
        records = run(args)
    except (OSError, ValueError, RuntimeError) as error:
        print(f"Validation failed: {error}", file=sys.stderr)
        return 1
    # Pending inference is a partial validation, not a successful accuracy test.
    return 0 if all(r["status"] == "segmentation_ready" and r["prediction_status"] == "complete"
                    for r in records) else 2


if __name__ == "__main__":
    raise SystemExit(main())
