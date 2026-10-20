#!/usr/bin/env python3
"""Build case-weighted AirRC lumen/wall Dice and IoU tables from frozen runs.

Sources are read-only. Execution completion is not an accuracy acceptance.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile

import numpy as np


CASE_PATTERN = re.compile(r"AIRRC\d{6}")
DICE_IOU_METRICS = ("dice", "iou")
RECORDED_CHILD_VERSIONS = ("numpy", "scipy", "nibabel", "totalsegmentator")
OVERLAP_POLICY = {"measurement_scope": "raw_voxel_masks", "trachea_policy": "include_as_segmented"}


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_hash_manifest(mapping, *, required=True):
    """Check recorded digest identities without opening their source files."""
    if not isinstance(mapping, dict) or (required and not mapping):
        raise ValueError("A nonempty hash manifest is required")
    for filename, digest in mapping.items():
        if not isinstance(filename, str) or not filename:
            raise ValueError("Invalid source path in hash manifest")
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError(f"Invalid SHA256 for {filename}")


def _verify(mapping, *, required=True):
    _validate_hash_manifest(mapping, required=required)
    for filename, digest in mapping.items():
        if _sha256(filename) != digest:
            raise ValueError(f"Source checksum mismatch: {filename}")


def _atomic_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _json(path, value):
    _atomic_text(path, json.dumps(value, indent=2, allow_nan=False) + "\n")


def _csv(path, rows, fields):
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    _atomic_text(path, buffer.getvalue())


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _code_identity(manifest):
    hashes = manifest.get("code_sha256")
    if not isinstance(hashes, dict) or not hashes:
        raise ValueError("Missing frozen pipeline code identity")
    by_name = {Path(path).name: digest for path, digest in hashes.items()}
    if len(by_name) != len(hashes):
        raise ValueError("Ambiguous pipeline code identities")
    return by_name


def _algorithm_group(manifest):
    value = {"scope": _scope(manifest), "profile": manifest.get("profile"), "code": _computation_identity(manifest),
             "model_identity": manifest["model_identity"],
             "recorded_dependency_versions": _dependency_identity(manifest), **_policy(manifest)}
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()[:16]


def _computation_identity(manifest):
    code = _code_identity(manifest)
    names = manifest.get("computation_files", sorted(code))
    if not isinstance(names, list) or not names or len(names) != len(set(names)) or any(name not in code for name in names):
        raise ValueError("Invalid computation algorithm identity")
    return {name: code[name] for name in names}


def _verify_cohort_code(manifest):
    expected = _code_identity(manifest)
    snapshots = manifest.get("code_snapshot_sha256")
    if snapshots:
        _verify(snapshots)
        if {Path(path).name: digest for path, digest in snapshots.items()} != expected:
            raise ValueError("Cohort snapshots differ from the frozen source code")
    else:
        _verify(manifest["code_sha256"])


def _dependency_identity(manifest):
    versions = manifest.get("dependency_versions", {})
    if not isinstance(versions, dict) or any(not isinstance(versions.get(name), str) or not versions[name]
                                             for name in RECORDED_CHILD_VERSIONS):
        raise ValueError("Missing frozen cohort dependency versions")
    return {name: versions[name] for name in RECORDED_CHILD_VERSIONS}


def _scope(manifest):
    scope = manifest.get("scope")
    if scope != "segmentation_overlap":
        raise ValueError("Unsupported AirRC validation scope")
    return scope


def _policy(manifest):
    _scope(manifest)
    return OVERLAP_POLICY


def derive_overlap(values, metrics=DICE_IOU_METRICS):
    """Independently verify Dice/IoU from recorded raw voxel counts."""
    if tuple(metrics) != DICE_IOU_METRICS:
        raise ValueError("AirRC overlap supports only Dice and IoU")
    if not isinstance(values, dict):
        raise ValueError("Invalid overlap metric record")
    counts = {}
    for key in ("prediction_voxels", "reference_voxels", "intersection_voxels", "union_voxels"):
        value = values.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"Invalid overlap voxel count: {key}")
        counts[key] = value
    p, r, i, u = (counts[key] for key in counts)
    if i > min(p, r) or u != p + r - i:
        raise ValueError("Inconsistent overlap voxel counts")
    result = {"dice": 2 * i / (p + r) if p + r else None,
              "iou": i / u if u else None}
    for key in metrics:
        if key not in values:
            raise ValueError(f"Missing recorded overlap metric: {key}")
        given, actual = values[key], result[key]
        if (given is None) != (actual is None) or (actual is not None and (
                not _finite(given) or not math.isclose(given, actual, rel_tol=1e-10, abs_tol=1e-12))):
            raise ValueError(f"Recorded {key} disagrees with voxel counts")
    return {**counts, **result, "status": "available" if u else "both_masks_empty"}


def _validate_overlap_inputs(record, case, model_identity, *, audit_masks=False,
                             input_root=None, prediction_root=None):
    """Check saved raw-mask identities; read masks only for an explicit audit."""
    if record.get("validation_scope") != "segmentation_overlap":
        raise ValueError("Case does not declare segmentation-only validation")
    overlap = record.get("overlap", {})
    expected_masks = {"airways", "airways_wall"}
    if not isinstance(overlap, dict) or set(overlap) != expected_masks:
        raise ValueError("Segmentation-only results must contain raw lumen/wall overlap")
    inputs = record["overlap_inputs"]
    if not isinstance(inputs, dict) or any(not isinstance(inputs.get(kind), dict)
                                          or set(inputs[kind]) != expected_masks
                                          for kind in ("prediction", "reference")):
        raise ValueError("Segmentation-only inputs must identify exactly the raw lumen/wall masks")
    recorded_sources = record.get("source_sha256", {})
    _validate_hash_manifest(recorded_sources)
    hashes = {}

    def source(info):
        if not isinstance(info, dict) or not isinstance(info.get("path"), str) or not info["path"]:
            raise ValueError("Invalid overlap input identity")
        path = Path(info["path"]).resolve()
        _validate_hash_manifest({str(path): info["sha256"]})
        if recorded_sources.get(str(path)) != info["sha256"]:
            raise ValueError("Overlap input differs from its recorded source identity")
        if audit_masks:
            _verify({str(path): info["sha256"]})
        hashes[str(path)] = info["sha256"]
        return path

    image_path = source(inputs["image"])
    if image_path.name != f"{case}.nii.gz" or image_path.parent.name != "image" or image_path.parent.parent.name != case:
        raise ValueError("CT input does not belong to the prepared case")
    case_root = image_path.parent.parent
    if input_root is not None and case_root != Path(input_root).resolve() / case:
        raise ValueError("CT input differs from the frozen cohort input root")
    provenance_path = source(inputs["segmentation_provenance"])
    if provenance_path.name != "segmentation_provenance.json" or provenance_path.parent.name != case:
        raise ValueError("Prediction cache does not belong to the case")
    if prediction_root is not None and provenance_path.parent != Path(prediction_root).resolve() / case:
        raise ValueError("Prediction cache differs from the frozen cohort prediction root")
    if not audit_masks:
        _verify({str(provenance_path): inputs["segmentation_provenance"]["sha256"]})
    provenance = _read(provenance_path)
    if not isinstance(provenance, dict):
        raise ValueError("Invalid segmentation provenance JSON record")
    if provenance.get("case") != case or provenance.get("provenance_status") != "inference_verified" or provenance.get("model_input_verified") is not True:
        raise ValueError("Segmentation provenance is not verified inference for this case")
    if Path(provenance["input"]).resolve() != image_path or provenance.get("input_sha256") != inputs["image"]["sha256"]:
        raise ValueError("Model input and validation CT identities differ")
    if provenance.get("model_identity") is not None and provenance["model_identity"] != model_identity:
        raise ValueError("Prediction cache model identity differs from the cohort")
    prediction_artifacts = provenance.get("artifact_sha256")
    if not isinstance(prediction_artifacts, dict):
        raise ValueError("Invalid inference artifact hash manifest")
    _validate_hash_manifest(prediction_artifacts)
    if audit_masks:
        import nibabel as nib

        image = nib.load(image_path)
        if len(image.shape) != 3 or image.header.get_xyzt_units()[0] != "mm" or not np.isfinite(image.affine).all() or abs(np.linalg.det(image.affine[:3, :3])) <= 0:
            raise ValueError("CT geometry must be finite three-dimensional world-mm geometry")
    for mask in sorted(overlap):
        scores = derive_overlap(overlap[mask], DICE_IOU_METRICS)
        if overlap[mask].get("status") != scores["status"]:
            raise ValueError("Recorded overlap availability disagrees with voxel counts")
        arrays = []
        for kind in ("prediction", "reference"):
            info = inputs[kind][mask]
            path = source(info)
            if kind == "prediction":
                if path != provenance_path.parent / f"lung_{mask}.nii.gz" or prediction_artifacts.get(path.name) != info["sha256"]:
                    raise ValueError("Prediction mask differs from its inference provenance")
            else:
                label = "label_airway" if mask == "airways" else "label_airway_wall"
                if path != case_root / label / f"{case}.nii.gz":
                    raise ValueError("Reference mask does not match its declared label kind")
            if not audit_masks:
                continue
            mask_image = nib.load(path)
            if mask_image.shape != image.shape or mask_image.header.get_xyzt_units()[0] != "mm" or not np.allclose(mask_image.affine, image.affine, rtol=0, atol=1e-5):
                raise ValueError("Prediction/reference mask geometry differs from CT")
            array = np.asanyarray(mask_image.dataobj)
            if not np.isfinite(array).all() or not np.all((array == 0) | (array == 1)):
                raise ValueError("Overlap input is not a finite binary mask")
            arrays.append(array.astype(bool, copy=False))
        if not audit_masks:
            continue
        p, r = int(arrays[0].sum()), int(arrays[1].sum())
        intersection = int(np.count_nonzero(arrays[0] & arrays[1]))
        counts = {"prediction_voxels": p, "reference_voxels": r,
                  "intersection_voxels": intersection, "union_voxels": p + r - intersection}
        if any(overlap[mask].get(key) != value for key, value in counts.items()):
            raise ValueError("Recorded overlap voxel counts differ from the raw masks")
    return hashes


def _validate_publication(entry, cohort, *, audit_masks=False):
    """Accept saved metrics by default; explicitly audit source masks when requested.

    The return value is ``(record, manifest, recorded_source_hashes)``. For
    segmentation overlap, JSON checksums and saved identities/arithmetic are
    checked without reading CT or masks.
    """
    case = entry["case"]
    if not isinstance(case, str) or not CASE_PATTERN.fullmatch(case):
        raise ValueError("Invalid case publication identity")
    root = Path(entry["run_dir"]).resolve()
    path = Path(entry["validation_path"]).resolve()
    if path != root / case / "validation.json":
        raise ValueError("Validation path does not belong to the case publication")
    _verify({str(path): entry["validation_sha256"]})
    artifacts, sources = entry.get("artifact_sha256", {}), entry.get("source_sha256", {})
    _validate_hash_manifest(artifacts)
    _validate_hash_manifest(sources)
    strict = audit_masks
    if strict:
        _verify(artifacts)
        _verify(sources)
    for required in (root / "run_manifest.json", root / "execution_status.json"):
        if str(required) not in artifacts:
            raise ValueError(f"Missing immutable publication checksum: {required.name}")
        if not strict:
            _verify({str(required): artifacts[str(required)]})
    if str(path) in artifacts and artifacts[str(path)] != entry["validation_sha256"]:
        raise ValueError("Conflicting case validation checksums")
    manifest = _read(root / "run_manifest.json")
    execution = _read(root / "execution_status.json")
    if not isinstance(manifest, dict) or not isinstance(execution, dict):
        raise ValueError("Invalid publication manifest or execution JSON record")
    if str(manifest.get("dataset", "")).lower() != "airrc" or manifest.get("run_id") != entry.get("run_id"):
        raise ValueError("Run identity mismatch")
    if case not in manifest.get("cases", []) or len(set(manifest["cases"])) != len(manifest["cases"]):
        raise ValueError("Case is absent or duplicated in publication manifest")
    if execution.get("status") not in ("complete", "partial"):
        raise ValueError("Source execution has not finished")
    if _scope(manifest) != _scope(cohort):
        raise ValueError("Incompatible validation scopes")
    policy = _policy(cohort)
    if any(manifest.get(key) != value or cohort.get(key) != value for key, value in policy.items()):
        raise ValueError("Incompatible measurement scope or trachea policy")
    if _code_identity(manifest) != _computation_identity(cohort):
        raise ValueError("Incompatible frozen pipeline algorithms")
    if not isinstance(cohort.get("model_identity"), dict) or not cohort["model_identity"] or manifest.get("model_identity") != cohort["model_identity"]:
        raise ValueError("Incompatible model/weight installation identities")
    expected_versions = _dependency_identity(cohort)
    child_versions = manifest.get("versions", {})
    if not isinstance(child_versions, dict) or any(child_versions.get(name) != version
                                                  for name, version in expected_versions.items()):
        raise ValueError("Incompatible or unrecorded child dependency versions")
    _validate_hash_manifest(manifest.get("code_snapshot_sha256", {}))
    if any(not Path(path).resolve().is_relative_to(root / "source_code") for path in manifest["code_snapshot_sha256"]):
        raise ValueError("Code snapshot is outside its immutable publication")
    snapshot = {Path(p).name: digest for p, digest in manifest["code_snapshot_sha256"].items()}
    if snapshot != _code_identity(manifest):
        raise ValueError("Code snapshots differ from the frozen algorithm identity")
    _verify(manifest["code_snapshot_sha256"])
    record = _read(path)
    if not isinstance(record, dict):
        raise ValueError("Invalid case validation JSON record")
    if record.get("case") != case:
        raise ValueError("Case validation identity mismatch")
    if case not in manifest.get("prediction_cases", []):
        raise ValueError("Case is not a declared prediction experiment")
    if manifest.get("validation_scope") != "segmentation_overlap" or manifest.get("metrics") != list(DICE_IOU_METRICS):
        raise ValueError("Publication is not Dice/IoU-only segmentation validation")
    if manifest.get("input_root") != cohort.get("input_root") or any(
            manifest.get(key) != cohort.get(key) for key in ("overlap_policy", "inference_workers")):
        raise ValueError("Incompatible prepared inputs or segmentation policy")
    hashes = _validate_overlap_inputs(record, case, manifest["model_identity"],
                                      audit_masks=audit_masks, input_root=cohort.get("input_root"),
                                      prediction_root=cohort.get("prediction_root"))
    recorded_sources = record["source_sha256"]
    if any(sources.get(filename) != digest for filename, digest in recorded_sources.items()):
        raise ValueError("Publication source identities differ from the saved case record")
    return record, manifest, {**artifacts, **sources,
                              **manifest["code_snapshot_sha256"], **hashes,
                              str(path): entry["validation_sha256"]}


def _safe_output(output, state_path, cohort):
    output = Path(output)
    if any(path.is_symlink() for path in (output, *output.parents)):
        raise ValueError("Report output must not contain symbolic links")
    output = output.resolve()
    input_root = Path(cohort["input_root"]).resolve()
    dataset = input_root.parent
    if dataset.name != "airrc" or dataset.parent.name != "data":
        raise ValueError("Cohort inputs must remain under data/airrc")
    reports = dataset / "reports"
    if output == reports or not output.is_relative_to(reports):
        raise ValueError("Cohort report must be a separate directory under data/airrc/reports")
    if Path(state_path).resolve().is_relative_to(output):
        raise ValueError("Report output overlaps source state")
    return output


def summarize_values(values, expected):
    data = np.asarray([value for value in values if _finite(value)], dtype=float)
    n = len(data)
    if n > expected:
        raise ValueError("Summary has more values than cohort cases")
    result = {"n_expected": expected, "n_available": n, "n_missing": expected - n,
              "mean": None, "std": None, "median": None, "min": None, "max": None,
              "p10": None, "p25": None, "p50": None, "p75": None}
    if n:
        result.update(mean=float(data.mean()), std=float(data.std(ddof=1)) if n > 1 else None,
                      median=float(np.median(data)), min=float(data.min()), max=float(data.max()))
        result.update(zip(("p10", "p25", "p50", "p75"), map(float, np.percentile(data, (10, 25, 50, 75)))))
    return result


def _summary(overlaps, expected):
    groups = defaultdict(list)
    for row in overlaps:
        for metric in DICE_IOU_METRICS:
            key = (row["mask"], metric, row["algorithm_group"])
            groups[key].append(row.get(metric))
    return [{"domain": "segmentation", "experiment": "model_prediction", "mask": key[0],
             "metric": key[1], "value_kind": "value", "algorithm_group": key[2],
             "quality_status": "label_comparison", "aggregation": "equal_case_weight",
             **summarize_values(values, expected)}
            for key, values in sorted(groups.items())]


def build_report(state_json, output_root, *, expected_cases=254, audit_masks=False):
    """Build an incremental report; invalid completed sources are explicitly excluded."""
    state_json = Path(state_json).resolve()
    state_text = state_json.read_text(encoding="utf-8")
    state_hash = hashlib.sha256(state_text.encode("utf-8")).hexdigest()
    state = json.loads(state_text)
    if str(state.get("dataset", "")).lower() != "airrc":
        raise ValueError("State is not an AirRC cohort")
    manifest_path = Path(state["cohort_manifest"])
    _verify({str(manifest_path): state["cohort_manifest_sha256"]})
    cohort = _read(manifest_path)
    if str(cohort.get("dataset", "")).lower() != "airrc":
        raise ValueError("Cohort manifest is not AirRC")
    cases = cohort.get("cases", [])
    if len(cases) != expected_cases or len(set(cases)) != len(cases) or any(not CASE_PATTERN.fullmatch(case) for case in cases):
        raise ValueError("Cohort case roster is invalid or has the wrong denominator")
    entries = state.get("cases", [])
    if len(entries) != len(cases) or Counter(row.get("case") for row in entries) != Counter(cases):
        raise ValueError("Publication state does not match the cohort roster")
    output = _safe_output(output_root, state_json, cohort)
    _verify_cohort_code(cohort)
    scope = _scope(cohort)
    verification_mode = "mask_audit" if audit_masks else "saved_metrics"
    verification = {"mode": verification_mode, "publication_json_checksums_verified": True,
                    "metric_arithmetic_checked": True, "source_hashes_rechecked": audit_masks,
                    "raw_masks_remeasured": audit_masks,
                    "note": ("Source hashes, CT/mask geometry and raw voxel counts were independently audited."
                             if audit_masks else "Saved metric JSON, identities and count arithmetic were checked. "
                             "Clinical source hashes are retained as recorded; CT/masks were not read or remeasured.")}
    metrics = DICE_IOU_METRICS
    if cohort.get("metrics") != list(metrics):
        raise ValueError("Segmentation-only cohort metrics must be Dice and IoU")
    if any(cohort.get(key) != value for key, value in _policy(cohort).items()):
        raise ValueError("Unsupported cohort measurement policy")
    overlaps, statuses, sources = [], [], {str(manifest_path): state["cohort_manifest_sha256"]}
    group = _algorithm_group(cohort)
    for entry in sorted(entries, key=lambda row: row["case"]):
        case = entry["case"]
        status = {"case": case, "execution_status": entry.get("status", "pending"),
                  "report_status": entry.get("status", "pending"), "run_id": entry.get("run_id"),
                  "accuracy_acceptance": "not_assessed", "error": entry.get("error"),
                  "model_identity_status": entry.get("model_identity_status", "unverified")}
        if entry.get("status") in ("complete", "partial", "failed") and entry.get("validation_path"):
            try:
                record, manifest, hashes = _validate_publication(entry, cohort, audit_masks=audit_masks)
                rows = [{"case": case, "mask": mask, "run_id": manifest["run_id"],
                         "algorithm_group": group, "external_independence": "unverified",
                         "model_identity_status": status["model_identity_status"], **derive_overlap(value, metrics)}
                        for mask, value in sorted(record["overlap"].items())]
                overlaps.extend(rows)
                sources.update(hashes)
                fully_complete = record.get("status") == "segmentation_ready" and record.get("prediction_status") == "complete"
                metric_status = "included" if fully_complete else "partial_metrics"
                if not rows:
                    metric_status = "no_available_metrics"
                status.update(report_status=metric_status, error=record.get("error") or entry.get("error"),
                              verification_mode=verification_mode, validation_status=record["status"],
                              prediction_status=record["prediction_status"])
            except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
                status.update(report_status="invalid_provenance", error=f"{type(error).__name__}: {error}")
        statuses.append(status)
    summary = _summary(overlaps, expected_cases)
    completed = sum(row["report_status"] == "included" for row in statuses)
    available = sum(row["report_status"] in ("included", "partial_metrics") for row in statuses)
    counts = dict(Counter(row["report_status"] for row in statuses))
    result = {"schema_version": 1, "dataset": "AirRC", "scope": scope, "metrics": list(metrics), "cohort_run_id": state.get("cohort_run_id"),
              "created_utc": datetime.now(timezone.utc).isoformat(), "cases_expected": expected_cases,
              "cases_included": available, "cases_completed": completed, "cases_not_included": expected_cases - available,
              "case_status_counts": counts, "accuracy_acceptance": "not_assessed",
              "verification": verification,
              "external_independence": "unverified", "potential_training_overlap": True,
              "measurement_policy": {**_policy(cohort), "overlap_scope": "raw_lumen_and_wall_masks",
                                     "aggregation": "equal_case_weight", "std": "sample_ddof_1",
                                     "percentiles": "linear interpolation"},
              "dependency_version_policy": {
                  "compared_child_versions": list(RECORDED_CHILD_VERSIONS),
                  "cohort_dependency_versions": cohort["dependency_versions"],
                  "historical_unrecorded_versions_not_verified": sorted(set(cohort["dependency_versions"]) - set(RECORDED_CHILD_VERSIONS)),
                  "note": "Historical runs did not record all dependencies; missing historical version identities are not retrospectively verified."},
              "cases": statuses, "segmentation_overlap": overlaps,
              "cohort_summary": summary,
              "source_sha256": sources, "state_snapshot_sha256": state_hash,
              "reporter_code_sha256": _sha256(__file__)}
    overlap_fields = ["case", "mask", "run_id", "algorithm_group", *metrics,
                      "prediction_voxels", "reference_voxels", "intersection_voxels", "union_voxels",
                      "status", "external_independence", "model_identity_status"]
    summary_fields = ["domain", "experiment", "mask", "metric", "value_kind", "algorithm_group",
                      "quality_status", "aggregation", "n_expected", "n_available", "n_missing",
                      "mean", "std", "median", "min", "max", "p10", "p25", "p50", "p75"]
    _csv(output / "segmentation_overlap.csv", overlaps, overlap_fields)
    _csv(output / "cohort_summary.csv", summary, summary_fields)
    status_fields = ["case", "execution_status", "report_status", "run_id",
                     "accuracy_acceptance", "model_identity_status", "verification_mode",
                     "prediction_status", "validation_status", "error"]
    _csv(output / "case_status.csv", statuses, status_fields)
    _atomic_text(output / "state_snapshot.json", state_text)
    _json(output / "sources.json", {"source_sha256": sources, "state_snapshot_sha256": state_hash,
                                    "reporter_code_sha256": result["reporter_code_sha256"],
                                    "verification": verification})
    _atomic_text(output / "report.md", _overlap_markdown(result))
    # The complete JSON is the final report-generation marker.
    _json(output / "validation_report.json", result)
    return result


def _format(value):
    return f"{value:.4f}" if _finite(value) else "Unavailable"


def _overlap_markdown(report):
    audited = report["verification"]["mode"] == "mask_audit"
    verification_text = ("Source hashes, CT/mask geometry and raw voxel counts are independently audited. "
                         if audited else "Saved publication JSON checksums, case/model identities and count arithmetic are checked. "
                         "Source hashes are retained as recorded; CT and mask files are not read or independently remeasured. ")
    lines = ["# AirRC cohort segmentation validation", "",
             f"Available case results: **{report['cases_included']} / {report['cases_expected']} cases**. "
             f"Completed cases: {report['cases_completed']}. Every metric retains the full cohort denominator.",
             "", "This report evaluates raw airway lumen and wall segmentation using Dice and IoU. "
             "Execution completion does not establish accuracy acceptance. Model training-case overlap is unverified; "
             "independent external generalization has not been established.",
             "", "## Dice and IoU", "",
             verification_text + "Dice and IoU are recalculated from the recorded intersection and voxel counts. "
             "Empty mask pairs remain unavailable.",
             "", "| Mask | Metric | Available / Total | Mean | Sample SD | Median |",
             "|---|---|---:|---:|---:|---:|"]
    for row in report["cohort_summary"]:
        lines.append(f"| {row['mask']} | {row['metric']} | {row['n_available']} / {row['n_expected']} | "
                     f"{_format(row['mean'])} | {_format(row['std'])} | {_format(row['median'])} |")
    if not report["segmentation_overlap"]:
        lines.append(f"| No accepted saved case results yet | — | 0 / {report['cases_expected']} | Unavailable | Unavailable | Unavailable |")
    lines += ["", "## Percentile distribution", "",
              "Percentiles describe the distribution across available cases with equal case weight. "
              "Scores range from 0 to 1; P50 is the median. Linear interpolation is used.",
              "", "| Mask | Metric | N | P10 | P25 | P50 | P75 |",
              "|---|---|---:|---:|---:|---:|---:|"]
    for row in report["cohort_summary"]:
        lines.append(f"| {row['mask']} | {row['metric']} | {row['n_available']} | "
                     f"{_format(row['p10'])} | {_format(row['p25'])} | "
                     f"{_format(row['p50'])} | {_format(row['p75'])} |")
    if not report["cohort_summary"]:
        lines.append("| No accepted saved case results yet | — | 0 | Unavailable | Unavailable | Unavailable | Unavailable |")
    lines += ["", "## Measurement policy", "",
              "- Compare raw model lumen/wall masks with the corresponding AirRC lumen/wall labels on the same physical grid.",
              "- Tracheal voxels are included as segmented. All connected components contribute to overlap.",
              "- Statistics assign equal weight to each case with an available value. Sample SD uses one degree-of-freedom correction and is unavailable for one case. Percentiles use linear interpolation.",
              "- The 1-mm reference label grid limits boundary resolution. Segmentation overlap does not establish histological wall-thickness accuracy.",
              "- Historical versions missing from the original inference/validation records remain unverified.",
              "", "## Availability", "", "| Report status | Cases |", "|---|---:|"]
    lines.extend(f"| {status} | {n} |" for status, n in sorted(report["case_status_counts"].items()))
    lines += ["", "## Tables and provenance", "",
              "- [Per-case lumen/wall Dice and IoU](segmentation_overlap.csv)",
              "- [Case-weighted statistics and percentiles](cohort_summary.csv)",
              "- [Every requested case and its status](case_status.csv)",
              "- [Complete numeric report](validation_report.json)",
              f"- [{'Audited' if audited else 'Recorded'} source checksums and verification mode](sources.json)"]
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-json", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True, help="Final incremental report directory under data/airrc/reports.")
    parser.add_argument("--expected-cases", type=int, default=254)
    parser.add_argument("--audit-masks", action="store_true",
                        help="Rehash clinical sources and independently verify CT/mask geometry and raw overlap voxels; "
                             "segmentation reports otherwise use saved metric JSON.")
    args = parser.parse_args(argv)
    try:
        report = build_report(args.state_json, args.output_root, expected_cases=args.expected_cases,
                              audit_masks=args.audit_masks)
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"Cohort report failed: {error}", file=sys.stderr)
        return 1
    print(f"Report cases: {report['cases_included']}/{report['cases_expected']}; "
          f"verification mode: {report['verification']['mode']}; accuracy acceptance not assessed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
