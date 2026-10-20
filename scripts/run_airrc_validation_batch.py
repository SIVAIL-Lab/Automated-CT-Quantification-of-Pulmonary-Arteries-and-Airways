#!/usr/bin/env python3
"""Run resumable AirRC validation with lightweight saved-metric aggregation."""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import uuid
import zipfile

import prepare_airrc as preparation
import run_airrc_airway_validation as validation
from inference_runtime import inference_environment, EXECUTION_POLICY_NAME, EXECUTION_POLICY_VERSION
from batch_runtime import (_child_subreaper, _interrupt_handlers, _terminate_workers,
                           _descendant_pids, _terminate_serial_descendants, BatchInterrupted)

SCHEMA_VERSION = 1
CASE_RE = re.compile(r"AIRRC\d{6}")
RUN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*")
COMPUTATION_FILES = validation.SEGMENTATION_COMPUTATION_FILES
TRACKED_FILES = (*COMPUTATION_FILES, "run_airrc_validation_batch.py", "batch_runtime.py",
                 "build_airrc_cohort_report.py")


def _now():
    return datetime.now(timezone.utc).isoformat()


def _read(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object")
    return value


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False,
                                         encoding="utf-8") as handle:
            temporary = Path(handle.name)
            json.dump(value, handle, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
        descriptor = os.open(path.parent, os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


@contextmanager
def _lock(dataset):
    directory = dataset / "results"
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".airrc_validation_batch.lock").open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another AirRC validation batch owns the dataset") from error
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _safe_path(path, dataset, protected=()):
    path = Path(os.path.abspath(Path(path).expanduser()))
    if any(part.is_symlink() for part in (path, *path.parents)) or not path.is_relative_to(dataset):
        raise ValueError("All paths must stay within the real AirRC dataset tree")
    for original in (dataset / "source_ct", dataset / "source_annotations", *protected):
        if path.is_relative_to(original) or original.is_relative_to(path):
            raise ValueError("Generated outputs must not overlap inputs or originals")
    return path


def _verify(hashes):
    if not isinstance(hashes, dict) or not hashes:
        raise ValueError("A nonempty hash manifest is required")
    for name, expected in hashes.items():
        path = Path(name)
        if not path.is_absolute() or any(part.is_symlink() for part in (path, *path.parents)) or not path.is_file():
            raise ValueError("A recorded input or artifact is missing or redirected")
        if preparation.sha256(path) != expected:
            raise ValueError("A recorded input or artifact changed")


def _roster(inventory, expected_count=254):
    rows = inventory.get("records")
    if not isinstance(rows, list) or len(rows) != expected_count:
        raise ValueError("The formal AirRC inventory has an unexpected case count")
    names = [row.get("case") for row in rows]
    if any(not isinstance(name, str) or not CASE_RE.fullmatch(name) for name in names):
        raise ValueError("Invalid de-identified AirRC case ID")
    if len(set(names)) != len(names):
        raise ValueError("Duplicate AirRC case IDs")
    return sorted(names)


def _dependency_versions():
    values = {"python": platform.python_version(), "python_executable": sys.executable}
    for name in ("numpy", "scipy", "nibabel", "totalsegmentator", "statsmodels", "scikit-image",
                 "SimpleITK", "torch", "PyMaxflow", "networkx"):
        try:
            values[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            values[name] = None
    return values


def _lightweight(cohort):
    return cohort.get("scope") == "segmentation_overlap" and not cohort.get("audit_masks", False)


def _check_archive_members(inventory):
    """Compare ZIP directory metadata once; do not read compressed CT contents."""
    archives = {}
    for row in inventory["records"]:
        archives.setdefault(row["ct_archive"], []).append(row)
    for filename, rows in archives.items():
        with zipfile.ZipFile(filename) as archive:
            for row in rows:
                expected = row["ct_archive_entries"]
                for kind in ("header", "voxel"):
                    member = row["ct_header_member" if kind == "header" else "ct_voxel_member"]
                    info = archive.getinfo(member)
                    expected_size = expected.get(f"{kind}_size_bytes")
                    if (info.CRC != expected[f"{kind}_crc32"]
                            or expected_size is not None and info.file_size != expected_size):
                        raise ValueError("CT archive member metadata differs from the inventory")


def _identity(args, inventory):
    cases = _roster(inventory)
    selected = sorted(set(getattr(args, "case", None) or [])) or None
    if selected is not None and (any(not CASE_RE.fullmatch(case) for case in selected) or set(selected) - set(cases)):
        raise ValueError("Selected cases must be de-identified aliases in the formal AirRC inventory")
    source_paths = {args.dataset_root / "26878867.zip", args.inventory}
    source_paths.update(Path(row["ct_archive"]) for row in inventory["records"])
    source_paths.update(Path(row["label_path"]) for row in inventory["records"])
    for path in source_paths:
        if any(part.is_symlink() for part in (path, *path.parents)) or not path.resolve().is_relative_to(args.dataset_root):
            raise ValueError("Cohort source paths must remain within the AirRC dataset")
    scope = getattr(args, "scope", "segmentation_overlap")
    audit_masks = getattr(args, "audit_masks", False)
    package = str((args.dataset_root / "26878867.zip").resolve())
    if not audit_masks:
        _check_archive_members(inventory)
        package_hash = inventory.get("annotation_package_sha256")
        if not isinstance(package_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", package_hash):
            raise ValueError("Inventory annotation package identity is missing")
        source_hashes = {str(args.inventory): preparation.sha256(args.inventory), package: package_hash}
    else:
        source_hashes = {str(path.resolve()): preparation.sha256(path) for path in sorted(source_paths)}
        if source_hashes[package] != inventory.get("annotation_package_sha256"):
            raise ValueError("Inventory reference package identity changed")
    runtime = {"segmentation_task": validation.segmentation.TASK,
               "segmentation_masks": list(validation.segmentation.EXPECTED_MASKS),
               "inference_policy": EXECUTION_POLICY_NAME, "inference_policy_version": EXECUTION_POLICY_VERSION,
               "max_concurrent_inference": getattr(args, "inference_workers", 1)}
    code_hashes = {str(Path(__file__).with_name(name).resolve()): preparation.sha256(Path(__file__).with_name(name))
                   for name in TRACKED_FILES}
    return {
        "schema_version": SCHEMA_VERSION, "dataset": "AIRRC", "cohort_run_id": args.run_id,
        "cases": cases, "selected_cases": selected, "input_root": str(args.input_root), "output_root": str(args.output_root),
        "child_report_root": str(args.child_report_root), "prediction_root": str(args.prediction_root),
        "report_root": str(args.report_root), "inventory": str(args.inventory),
        "annotation_package_sha256": source_hashes[package], "source_sha256": source_hashes,
        "audit_masks": audit_masks,
        "verification_policy": {
            "mode": "saved_metrics" if not audit_masks else "mask_audit",
            "archive_checks": "ZIP member CRC and size metadata once; no archive rehash"
                if not audit_masks else "full source SHA-256",
            "annotation_package_hash": "recorded inventory identity; not rehashed"
                if not audit_masks else "verified SHA-256",
            "case_inputs": "Shared preparation and first metric calculation verify inputs and masks",
            "receipt": "Case, request, exit status, small result JSON and overlap arithmetic"
                if not audit_masks else "full publication verification",
            "reports": "Saved JSON metrics; no CT or mask reads"
                if not audit_masks else "full source and mask verification",
        },
        "scope": scope, "validation_scope": scope, "overlap_policy": validation.OVERLAP_POLICY,
        "metrics": ["dice", "iou"], "profile": None, "profile_source": None,
        "profile_sha256": None, "code_sha256": code_hashes,
        "model_identity": validation._model_identity(), "runtime_identity": runtime,
        "computation_files": list(COMPUTATION_FILES), "dependency_versions": _dependency_versions(),
        "device": args.device,
        "case_timeout_seconds": args.case_timeout_seconds,
        "min_free_gib": args.min_free_gib, "report_every": args.report_every, "workers": getattr(args, "workers", 1),
        "inference_workers": getattr(args, "inference_workers", 1),
        "parallel_resource_policy": {"max_workers": 8, "minimum_case_memory_gib": 4, "host_memory_reserve_gib": 2,
                                     "disk_reservation": "floor plus candidate and active case budgets",
                                     "memory_reservation": "candidate budget plus unallocated active budgets and host reserve",
                                     "active_memory_measurement": "process-group proportional anonymous and shared memory"},
        "measurement_scope": "raw_voxel_masks", "trachea_policy": "include_as_segmented",
        "prediction_policy": "all_cases", "reference_initialized_wall_policy": "not_requested",
        "external_independence": "unverified", "accuracy_acceptance": "not_assessed",
        "execution_policy": "Serial isolated cases; shared deterministic isolated inference; no parameter tuning.",
    }


def _publication_files(root, case):
    files = [root / "run_manifest.json", root / "execution_status.json", root / case / "validation.json"]
    directories = [root / "source_code", root / case]
    for directory in directories:
        files.extend(path for path in directory.rglob("*") if path.is_file())
    if any(path.is_symlink() or any(parent.is_symlink() for parent in path.parents) for path in files):
        raise ValueError("Publication artifacts must not contain symlinks")
    return sorted(set(files))


def _saved_publication(root, case, cohort, *, published_entry=None):
    """Read case results without rescanning clinical sources or remeasuring masks."""
    from build_airrc_cohort_report import _validate_publication as read_publication

    root = Path(root).resolve()
    record_path = root / case / "validation.json"
    record = _read(record_path)
    manifest = _read(root / "run_manifest.json")
    execution = _read(root / "execution_status.json")
    if manifest.get("scope") != "segmentation_overlap" or cohort.get("scope") != "segmentation_overlap":
        raise ValueError("Publication validation scope differs from the frozen cohort")
    if (execution.get("status") != "complete" or record.get("status") != "segmentation_ready"
            or record.get("prediction_status") != "complete" or manifest.get("reference_only")):
        raise ValueError("A completed prediction and reference publication is required")
    if published_entry is None:
        files = [root / "run_manifest.json", root / "execution_status.json", record_path]
        artifacts = {str(path): preparation.sha256(path) for path in files}
        entry = {"case": case, "run_id": manifest["run_id"], "run_dir": str(root),
                 "validation_path": str(record_path), "validation_sha256": artifacts[str(record_path)],
                 "artifact_sha256": artifacts, "source_sha256": record["source_sha256"]}
    else:
        entry = dict(published_entry)
        if entry.get("case") != case or Path(entry["run_dir"]).resolve() != root:
            raise ValueError("Saved publication case or directory differs")
    record, manifest, _ = read_publication(entry, cohort)
    versions = manifest.get("versions", {})
    return {**entry, "status": "complete", "receipt_verification": "saved_metrics",
            "model_identity_status": "recorded" if record.get("prediction_model_identity")
                else "legacy_original_weights_unverified",
            "dependency_versions_reported": versions,
            "dependency_versions_not_recorded": sorted(name for name, value in cohort.get("dependency_versions", {}).items()
                                                       if value is not None and name not in versions)}


def _validate_publication(root, case, cohort, runtime, *, published_entry=None):
    """Verify a complete case without changing any saved measurement or mask."""
    if _lightweight(cohort):
        return _saved_publication(root, case, cohort, published_entry=published_entry)
    root = Path(root).resolve()
    manifest = _read(root / "run_manifest.json")
    execution = _read(root / "execution_status.json")
    record_path = root / case / "validation.json"
    record = _read(record_path)
    if manifest.get("scope") != "segmentation_overlap" or cohort.get("scope") != "segmentation_overlap":
        raise ValueError("Publication validation scope differs from the frozen cohort")
    if (execution.get("status") != "complete" or case not in manifest.get("cases", [])
            or case not in manifest.get("prediction_cases", []) or manifest.get("reference_only")
            or record.get("case") != case or record.get("status") != "segmentation_ready"
            or record.get("prediction_status") != "complete"):
        raise ValueError("A completed prediction and reference publication is required")
    policies = ("measurement_scope", "trachea_policy", "input_root")
    if manifest.get("inference_workers", 1) != cohort.get("inference_workers", 1):
        raise ValueError("Publication inference worker policy differs from the frozen cohort")
    if manifest.get("metrics") != ["dice", "iou"] or manifest.get("overlap_policy") != cohort.get("overlap_policy"):
        raise ValueError("Segmentation publications must request only Dice and IoU")
    for key in policies:
        if manifest.get(key) != cohort[key]:
            raise ValueError("Publication profile, measurement policy or prepared inputs differ")
    versions = manifest.get("versions", {})
    recorded_packages = ("numpy", "scipy", "nibabel", "totalsegmentator")
    if cohort.get("dependency_versions") and any(versions.get(name) != cohort["dependency_versions"].get(name)
                                                for name in recorded_packages):
        raise ValueError("Publication recorded dependency versions differ from the frozen cohort")
    expected_codes = {Path(name).name: digest for name, digest in cohort["code_sha256"].items()}
    codes = {Path(name).name: digest for name, digest in manifest.get("code_sha256", {}).items()}
    critical = cohort.get("computation_files") or COMPUTATION_FILES
    if set(codes) != set(critical) or any(codes[name] != expected_codes.get(name) for name in critical):
        raise ValueError("Publication calculation code differs from the frozen cohort")
    snapshots = manifest.get("code_snapshot_sha256", {})
    if {Path(name).name: value for name, value in snapshots.items()} != codes:
        raise ValueError("Publication frozen code snapshots are incomplete")
    _verify(snapshots)
    if manifest.get("model_identity") != cohort["model_identity"]:
        raise ValueError("Publication installed model identity differs")
    case_dir = Path(cohort["input_root"]) / case
    converted = preparation.verify_airrc_conversion(case_dir, groups=tuple(preparation.GROUP_DIRECTORIES))
    source_hashes = dict(converted["source_sha256"])
    for name, digest in source_hashes.items():
        if name in cohort.get("source_sha256", {}) and cohort["source_sha256"][name] != digest:
            raise ValueError("Prepared case sources differ from the frozen cohort inputs")
    prediction = Path(cohort["prediction_root"]) / case
    cached = validation.segmentation._cached_provenance(prediction / "segmentation_provenance.json", case,
                                                       preparation.sha256(case_dir / "image" / f"{case}.nii.gz"))
    if cached is None or cached.get("provenance_status") != "inference_verified":
        raise ValueError("Prediction model input and mask cache are not verified")
    recorded_model = cached.get("model_identity")
    if recorded_model is not None and recorded_model != cohort["model_identity"]:
        raise ValueError("Prediction original weights identity differs")
    source_hashes.update({str(path.resolve()): preparation.sha256(path)
                          for path in (prediction / "segmentation_provenance.json", prediction / "run_report.json",
                                       *(prediction / name for name in validation.segmentation.EXPECTED_MASKS))})
    _verify(source_hashes)
    _verify_overlap_inputs(record, case_dir, prediction, source_hashes)
    _verify_actual_overlap(record, case_dir, prediction)
    _verify(source_hashes)
    masks = ("airways", "airways_wall")
    for mask in masks:
        scores = record.get("overlap", {}).get(mask, {})
        p, r, n, u = (scores.get(key) for key in ("prediction_voxels", "reference_voxels", "intersection_voxels", "union_voxels"))
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in (p, r, n, u)):
            raise ValueError("Invalid voxel overlap counts")
        if n > min(p, r) or u != p + r - n:
            raise ValueError("Invalid voxel overlap relationships")
        if not u:
            if scores.get("dice") is not None or scores.get("iou") is not None or scores.get("status") != "both_masks_empty":
                raise ValueError("Both-empty masks must remain unavailable")
            continue
        for key, target in (("dice", 2 * n / (p + r)), ("iou", n / u)):
            if not isinstance(scores.get(key), (float, int)) or not math.isclose(scores[key], target, abs_tol=1e-12, rel_tol=1e-12):
                raise ValueError("Saved Dice or IoU differs from its voxel counts")
    files = _publication_files(root, case)
    artifact_hashes = {str(path): preparation.sha256(path) for path in files}
    return {"case": case, "status": "complete", "run_id": manifest["run_id"], "run_dir": str(root),
            "validation_path": str(record_path), "validation_sha256": artifact_hashes[str(record_path)],
            "artifact_sha256": artifact_hashes, "source_sha256": source_hashes,
            "model_identity_status": "recorded" if recorded_model else "legacy_original_weights_unverified",
            "dependency_versions_reported": versions,
            "dependency_versions_not_recorded": sorted(name for name, value in cohort.get("dependency_versions", {}).items()
                                                       if value is not None and name not in versions)}


def _verify_overlap_inputs(record, case_dir, prediction, source_hashes):
    paths = {"image": case_dir / "image" / f"{case_dir.name}.nii.gz",
             "segmentation_provenance": prediction / "segmentation_provenance.json"}
    inputs = record.get("overlap_inputs", {})
    for key, path in paths.items():
        entry = inputs.get(key, {})
        if entry.get("path") != str(path.resolve()) or entry.get("sha256") != preparation.sha256(path):
            raise ValueError("Segmentation overlap input identity differs")
        source_hashes[str(path.resolve())] = entry["sha256"]
    for role, directory in (("prediction", prediction), ("reference", case_dir)):
        for mask, group in (("airways", "label_airway"), ("airways_wall", "label_airway_wall")):
            path = directory / f"lung_{mask}.nii.gz" if role == "prediction" else directory / group / f"{case_dir.name}.nii.gz"
            entry = inputs.get(role, {}).get(mask, {})
            if entry.get("path") != str(path.resolve()) or entry.get("sha256") != preparation.sha256(path):
                raise ValueError("Segmentation overlap mask identity differs")
            source_hashes[str(path.resolve())] = entry["sha256"]


def _verify_actual_overlap(record, case_dir, prediction):
    image = validation.nib.load(case_dir / "image" / f"{case_dir.name}.nii.gz")
    validation.segmentation._validate_masks(image, prediction)
    if set(record.get("overlap", {})) != {"airways", "airways_wall"}:
        raise ValueError("Segmentation overlap must contain only the two raw airway masks")
    for mask, group in (("airways", "label_airway"), ("airways_wall", "label_airway_wall")):
        left = validation._binary_image(prediction / f"lung_{mask}.nii.gz", image)
        right = validation._binary_image(case_dir / group / f"{case_dir.name}.nii.gz", image)
        expected = validation.overlap_metrics(validation.np.asarray(left.dataobj), validation.np.asarray(right.dataobj))
        actual = record["overlap"][mask]
        for key, value in expected.items():
            if actual.get(key) != value:
                raise ValueError("Saved overlap differs from independently measured masks")


def _preserve_incomplete_inputs(args, row):
    """Keep interrupted generated files in their original attempt before a retry."""
    if not row.get("attempts"):
        return
    previous = row["attempts"][-1]
    if previous.get("status") not in ("failed", "interrupted"):
        return
    previous_dir = Path(previous["log_dir"])
    allowed = args.state_root / args.run_id / "attempts" / row["case"]
    if not previous_dir.is_relative_to(allowed) or any(part.is_symlink() for part in (previous_dir, *previous_dir.parents)):
        raise ValueError("Interrupted attempt log location changed identity")
    candidates = []
    case_dir = args.input_root / row["case"]
    if (previous.get("stage") == "conversion" and case_dir.exists()
            and not (case_dir / preparation.REPORT_NAME).exists()):
        candidates.append((case_dir, previous_dir / "incomplete_conversion_inputs"))
    prediction = args.prediction_root / row["case"]
    marker = prediction / "segmentation_provenance.json"
    if previous.get("stage") == "validation" and marker.is_file() and _read(marker).get("status") == "incomplete":
        candidates.append((prediction, previous_dir / "incomplete_prediction_cache"))
    for source, destination in candidates:
        if destination.exists() or any(path.is_symlink() for path in (source, *source.rglob("*"))):
            raise ValueError("Interrupted generated inputs cannot be archived safely")
        previous_dir.mkdir(parents=True, exist_ok=True)
        source.rename(destination)
        previous.setdefault("preserved_incomplete_inputs", []).append(str(destination))


def _disk_budget(record):
    """Reserve raw source + native conversion, dense temporary grids and bounded graphs."""
    native_bytes = int(record["ct_archive_entries"]["voxel_size_bytes"])
    voxels = math.prod(record["reference_geometry"]["shape"])
    return 2 * native_bytes + 80 * voxels + 1024 ** 3


def _check_disk(path, record, floor_gib):
    available = shutil.disk_usage(path).free
    required = int(floor_gib * 1024 ** 3) + _disk_budget(record)
    if available < required:
        raise RuntimeError("disk_space_floor: Not enough free space for the next case and reserve")


def _subprocess(command, log, timeout):
    log.parent.mkdir(parents=True, exist_ok=True)
    handle = log.open("wb")
    try:
        process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=handle,
                                   stderr=subprocess.STDOUT, env=dict(inference_environment(), PYTHONUNBUFFERED="1"),
                                   start_new_session=True)
    except BaseException:
        handle.close()
        raise
    job = {"process": process, "log_handle": handle}
    try:
        result = process.wait(timeout=timeout)
        # Even a normally exiting worker must not leave inference/Fiji writers.
        _terminate_workers([job], grace=1.)
        return result
    except BaseException:
        _terminate_workers([job])
        raise
    finally:
        if not handle.closed:
            handle.close()


def _save_state(path, state):
    state["updated_utc"] = _now()
    state["counts"] = dict(Counter(row["status"] for row in state["cases"]))
    _write(path, state)


def _refresh_report(args, root, state_path, state):
    if not args.build_report:
        return
    reporter = Path(__file__).with_name("build_airrc_cohort_report.py")
    command = [sys.executable, str(reporter), "--state-json", str(state_path),
               "--output-root", str(args.report_root / args.run_id)]
    if getattr(args, "audit_masks", False):
        command.append("--audit-masks")
    try:
        result = _subprocess(command, root / "report.log", args.case_timeout_seconds)
        state["report_status"] = "complete" if result == 0 else "error"
        state["report_updated_utc"] = _now()
    except (OSError, RuntimeError, subprocess.TimeoutExpired) as error:
        state.update(report_status="error", report_error={"type": type(error).__name__, "message": str(error)[:1000]})
    _save_state(state_path, state)


def _normalize(args):
    args.dataset_root = args.dataset_root.expanduser().resolve()
    if args.dataset_root.name != "airrc":
        raise ValueError("The dataset root must be data/airrc")
    args.inventory = args.inventory or args.dataset_root / "audit" / "airrc_case_inventory.json"
    args.inventory = args.inventory.expanduser().resolve()
    args.input_root = _safe_path(args.input_root or args.dataset_root / "cases", args.dataset_root)
    for key, default in (("output_root", "results/airway_validation"), ("child_report_root", "reports/airway_validation"),
                         ("prediction_root", "results/totalsegmentator"), ("state_root", "results/airway_validation_batch"),
                         ("report_root", "reports/airway_validation_batch")):
        setattr(args, key, _safe_path(getattr(args, key) or args.dataset_root / default, args.dataset_root, (args.input_root,)))
    outputs = [getattr(args, key) for key in ("output_root", "child_report_root", "prediction_root", "state_root", "report_root")]
    if any(a.is_relative_to(b) or b.is_relative_to(a) for index, a in enumerate(outputs) for b in outputs[index + 1:]):
        raise ValueError("Computation, state and report output roots must remain separate")
    if not RUN_RE.fullmatch(args.run_id):
        raise ValueError("Invalid cohort run ID")
    for key in ("case_timeout_seconds", "min_free_gib"):
        if not math.isfinite(getattr(args, key)) or getattr(args, key) <= 0:
            raise ValueError("Timeouts and disk reserve must be finite and positive")
    if args.report_every < 0:
        raise ValueError("--report-every must be nonnegative")
    if args.max_cases is not None and args.max_cases < 1:
        raise ValueError("--max-cases must be positive")
    args.scope = getattr(args, "scope", "segmentation_overlap")
    args.workers = getattr(args, "workers", 1)
    args.inference_workers = getattr(args, "inference_workers", 1)
    if args.workers not in range(1, 9) or args.inference_workers not in (1, 2, 4):
        raise ValueError("Case workers must be 1–8 and inference workers must be 1, 2 or 4")
    if args.scope != "segmentation_overlap":
        raise ValueError("Only segmentation_overlap validation is supported")
    return args


def _child_arguments(args, case, child_id):
    command = ["--case", case, "--run-id", child_id, "--input-root", str(args.input_root),
               "--output-root", str(args.output_root), "--report-root", str(args.child_report_root),
               "--prediction-root", str(args.prediction_root), "--device", args.device]
    command.extend(["--segmentation-only", "--inference-workers", str(getattr(args, "inference_workers", 1))])
    return command


def _active_identity(args, identity, source):
    if _lightweight(identity):
        return
    _verify(identity["code_sha256"])
    specific = [source["ct_archive"], source["label_path"], str(args.dataset_root / "26878867.zip")]
    _verify({name: identity["source_sha256"][name] for name in specific})
    if _dependency_versions() != identity["dependency_versions"]:
        raise ValueError("Dependency versions changed during the cohort")
    if validation._model_identity() != identity["model_identity"]:
        raise ValueError("Installed model identity changed during the cohort")


class ResourcesUnavailable(RuntimeError):
    pass


def _memory_available():
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("Available host memory could not be checked")


def _case_memory_budget(record):
    return max(4 * 1024 ** 3, 64 * math.prod(record["reference_geometry"]["shape"])
               + 3 * int(record["ct_archive_entries"]["voxel_size_bytes"]))


def _process_group_memory(pgid):
    """Measure allocated non-reclaimable case memory without counting fork shares twice."""
    allocated = 0
    for status in Path("/proc").glob("[0-9]*/stat"):
        try:
            fields = status.read_text().rsplit(")", 1)[1].split()
            if int(fields[2]) != pgid:
                continue
            rows = (status.parent / "smaps_rollup").read_text().splitlines()
            for row in rows:
                if row.startswith(("Pss_Anon:", "Pss_Shmem:")):
                    allocated += int(row.split()[1]) * 1024
        except (OSError, ValueError, IndexError):
            # A disappeared process or unavailable counter restores no budget.
            continue
    return allocated


def _parallel_resources(args, record, active_records, active_jobs=None):
    records = [record, *active_records]
    try:
        extra_reserve_gib = sum(_disk_budget(row) for row in active_records) / 1024 ** 3
        _check_disk(args.dataset_root, record, args.min_free_gib + extra_reserve_gib)
    except RuntimeError as error:
        raise ResourcesUnavailable(str(error)) from error
    allocated = ([_process_group_memory(job["process"].pid) for job in active_jobs]
                 if active_jobs is not None else [0] * len(active_records))
    remaining = [max(0, _case_memory_budget(row) - used) for row, used in zip(active_records, allocated)]
    required = 2 * 1024 ** 3 + _case_memory_budget(record) + sum(remaining)
    available = _memory_available()
    if available < required:
        raise ResourcesUnavailable("host_memory_reserve: Not enough available RAM for another case")
    return {"host_memory_available_bytes": available, "host_memory_required_bytes": required,
            "candidate_memory_budget_bytes": _case_memory_budget(record), "active_allocated_memory_bytes": allocated,
            "active_remaining_memory_reserve_bytes": remaining,
            "active_disk_budget_bytes": sum(_disk_budget(row) for row in active_records),
            "candidate_disk_budget_bytes": _disk_budget(record), "disk_floor_gib": args.min_free_gib}


def _launch_parallel_case(row, args, root, identity, *, admission=None):
    _preserve_incomplete_inputs(args, row)
    attempt = len(row["attempts"]) + 1
    child_id = f"{args.run_id}_{row['case']}_a{attempt:03d}"
    while (args.output_root / child_id).exists() or (args.child_report_root / child_id).exists():
        attempt += 1
        child_id = f"{args.run_id}_{row['case']}_a{attempt:03d}"
    directory = root / "attempts" / row["case"] / child_id
    request_path = directory / "request.json"
    inventory_path = root / "inventory_snapshot.json"
    options = {name: str(getattr(args, name)) for name in
               ("dataset_root", "input_root", "output_root", "child_report_root", "prediction_root", "state_root")}
    options.update(scope=args.scope, device=args.device, run_id=args.run_id,
                   inference_workers=getattr(args, "inference_workers", 1),
                   audit_masks=getattr(args, "audit_masks", False))
    request = {"schema_version": SCHEMA_VERSION, "request_id": uuid.uuid4().hex, "case": row["case"],
               "child_run_id": child_id, "cohort_identity": identity, "options": options,
               "inventory_snapshot": str(inventory_path), "inventory_sha256": preparation.sha256(inventory_path),
               "result_path": str(directory / "result.json"), "progress_path": str(directory / "progress.json")}
    _write(request_path, request)
    attempt_record = {"run_id": child_id, "status": "running", "stage": "conversion", "started_utc": _now(),
                      "log_dir": str(directory), "request_sha256": _digest(request), "resource_admission": admission}
    row["attempts"].append(attempt_record)
    row.update(status="running", current_stage="conversion")
    handle = (directory / "worker.log").open("wb")
    try:
        process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--case-worker-request", str(request_path)],
                                   stdin=subprocess.DEVNULL, stdout=handle, stderr=subprocess.STDOUT,
                                   env=dict(inference_environment(), PYTHONUNBUFFERED="1"), start_new_session=True)
    except BaseException:
        handle.close()
        raise
    return {"process": process, "log_handle": handle, "request": request, "row": row,
            "attempt": attempt_record, "started": time.monotonic(), "last_progress": None}


def _parallel_result(job, args, identity):
    result = _read(job["request"]["result_path"])
    if (result.get("request_id") != job["request"]["request_id"]
            or result.get("request_sha256") != _digest(job["request"])
            or result.get("case") != job["row"]["case"]):
        raise ValueError("Case worker result identity or request checksum differs")
    if job["process"].returncode != 0 or result.get("status") != "complete" or result.get("validated") is not True:
        error = result.get("error", {})
        raise RuntimeError(f"Case worker failed: {error.get('type', 'error')}: {error.get('message', '')[:700]}")
    publication = result["publication"]
    expected_root = args.output_root / job["request"]["child_run_id"]
    if (publication.get("case") != job["row"]["case"] or publication.get("status") != "complete"
            or publication.get("run_dir") != str(expected_root)
            or publication.get("run_id") != job["request"]["child_run_id"]
            or publication.get("validation_path") != str(expected_root / job["row"]["case"] / "validation.json")):
        raise ValueError("Case worker publication identity differs")
    if preparation.sha256(Path(publication["validation_path"])) != publication["validation_sha256"]:
        raise ValueError("Case worker published validation changed")
    if _lightweight(identity):
        from build_airrc_cohort_report import DICE_IOU_METRICS, derive_overlap

        record = _read(publication["validation_path"])
        if (record.get("case") != job["row"]["case"] or record.get("status") != "segmentation_ready"
                or record.get("prediction_status") != "complete"
                or set(record.get("overlap", {})) != {"airways", "airways_wall"}):
            raise ValueError("Case worker saved result identity or metric scope differs")
        for value in record["overlap"].values():
            scores = derive_overlap(value, DICE_IOU_METRICS)
            if scores["status"] != value.get("status"):
                raise ValueError("Case worker saved overlap availability differs")
        return publication
    _verify(publication["artifact_sha256"])
    _verify(publication["source_sha256"])
    _verify(identity["code_sha256"])
    if _dependency_versions() != identity["dependency_versions"] or validation._model_identity() != identity["model_identity"]:
        raise ValueError("Runtime or installed model changed during case execution")
    return publication


def _selected(identity, case):
    return identity.get("selected_cases") is None or case in identity["selected_cases"]


def _finished_status(state, identity, *, limited=False):
    if all(row["status"] == "complete" for row in state["cases"]):
        return "complete"
    if limited:
        return "paused_after_case_limit"
    if (identity.get("selected_cases") is not None and
            all(row["status"] == "complete" for row in state["cases"] if _selected(identity, row["case"]))):
        return "paused_after_case_selection"
    return "partial"


def _run_parallel_cases(args, root, state_path, state, identity, inventory):
    """Schedule at most eight isolated cases; only this parent writes cohort state."""
    pending = [row for row in state["cases"] if _selected(identity, row["case"]) and row["status"] != "complete"
               and (row["status"] != "failed" or args.retry_failed)]
    records = {row["case"]: row for row in inventory["records"]}
    jobs, launched, finished = {}, 0, 0
    previous_descendants = _descendant_pids()
    evidence = state.setdefault("parallel_execution", {"requested_case_workers": args.workers,
        "inference_workers": getattr(args, "inference_workers", 1), "peak_active_case_workers": 0,
        "resource_waits": []})
    waiting, wait_started = None, None
    def active_count():
        active = [job for job in jobs.values() if job["process"].poll() is None]
        evidence["current_active_case_workers"] = len(active)
        if len(active) > evidence["peak_active_case_workers"]:
            evidence.update(peak_active_case_workers=len(active), peak_observed_utc=_now(),
                            peak_case_ids=[job["row"]["case"] for job in active],
                            peak_worker_pids=[job["process"].pid for job in active])
        return len(active)
    def end_wait(status):
        nonlocal waiting, wait_started
        if waiting is not None:
            waiting.update(status=status, ended_utc=_now(), elapsed_seconds=time.monotonic() - wait_started)
            waiting, wait_started = None, None
    state["status"] = "running"
    _save_state(state_path, state)
    try:
        while pending or jobs:
            while pending and len(jobs) < args.workers and (args.max_cases is None or launched < args.max_cases):
                row = pending[0]
                try:
                    admission = _parallel_resources(args, records[row["case"]], [records[name] for name in jobs], list(jobs.values()))
                except ResourcesUnavailable as error:
                    if jobs:
                        if waiting is None:
                            wait_started = time.monotonic()
                            waiting = {"case": row["case"], "status": "waiting", "started_utc": _now(),
                                       "active_case_workers": active_count(), "requested_case_workers": args.workers,
                                       "reasons": [str(error)[:500]]}
                            evidence["resource_waits"].append(waiting)
                            _save_state(state_path, state)
                            print(f"{row['case']}: waiting for resources; {waiting['active_case_workers']}/{args.workers} case workers active", flush=True)
                        elif str(error)[:500] not in waiting["reasons"]:
                            waiting["reasons"].append(str(error)[:500])
                            _save_state(state_path, state)
                        break
                    raise
                end_wait("admitted")
                _active_identity(args, identity, records[row["case"]])
                pending.pop(0)
                try:
                    job = _launch_parallel_case(row, args, root, identity, admission=admission)
                except (OSError, ValueError, RuntimeError) as error:
                    row.update(status="failed", error={"type": type(error).__name__, "message": str(error)[:1000]})
                    if row["attempts"] and row["attempts"][-1]["status"] == "running":
                        row["attempts"][-1].update(status="failed", finished_utc=_now(), error=row["error"])
                    finished += 1
                else:
                    jobs[row["case"]] = job
                    observed = active_count()
                    print(f"{row['case']}: case worker started ({observed}/{args.workers} active)", flush=True)
                launched += 1
                _save_state(state_path, state)
            for name, job in list(jobs.items()):
                process = job["process"]
                progress_path = Path(job["request"]["progress_path"])
                if progress_path.exists():
                    progress = _read(progress_path)
                    if (progress.get("request_id") == job["request"]["request_id"]
                            and progress.get("stage") in ("conversion", "validation", "verification")
                            and progress.get("stage") != job["last_progress"]):
                        job["last_progress"] = progress["stage"]
                        job["row"]["current_stage"] = progress["stage"]
                        job["attempt"]["stage"] = progress["stage"]
                        _save_state(state_path, state)
                timed_out = process.poll() is None and time.monotonic() - job["started"] > args.case_timeout_seconds
                if process.poll() is None and not timed_out:
                    continue
                _terminate_workers([job], grace=1.)
                row = job["row"]
                try:
                    if timed_out:
                        raise RuntimeError("Case worker exceeded its case timeout")
                    _active_identity(args, identity, records[name])
                    publication = _parallel_result(job, args, identity)
                    row.update(publication, batch_verified_dependency_versions=identity["dependency_versions"])
                    row.pop("error", None)
                    row.pop("current_stage", None)
                    job["attempt"].update(status="complete", finished_utc=_now())
                except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
                    row.update(status="failed", error={"type": type(error).__name__, "message": str(error)[:1000]})
                    job["attempt"].update(status="failed", finished_utc=_now(), error=row["error"])
                del jobs[name]
                active_count()
                finished += 1
                _save_state(state_path, state)
                print(f"{name}: {row['status']}; {state['counts'].get('complete', 0)}/{len(state['cases'])} complete", flush=True)
                if args.report_every and finished % args.report_every == 0:
                    _refresh_report(args, root, state_path, state)
            if not jobs and (not pending or (args.max_cases is not None and launched >= args.max_cases)):
                break
            if jobs:
                time.sleep(.05)
        end_wait("stopped")
        state["status"] = _finished_status(state, identity, limited=bool(pending))
    except BaseException as error:
        # Keep ownership until all worker groups and their inference children stop.
        try:
            _terminate_workers(list(jobs.values()))
            _terminate_serial_descendants(previous_descendants)
        finally:
            end_wait("interrupted" if isinstance(error, KeyboardInterrupt) else "stopped_error")
            evidence["current_active_case_workers"] = 0
            for job in jobs.values():
                job["row"]["status"] = "pending"
                job["attempt"].update(status="interrupted", finished_utc=_now())
            state.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "paused_error",
                         stop_reason={"type": type(error).__name__, "message": str(error)[:1000]})
            _save_state(state_path, state)
            _refresh_report(args, root, state_path, state)
        raise
    _save_state(state_path, state)
    _refresh_report(args, root, state_path, state)
    return state


def _case_worker(request_path):
    """Run conversion and validation in one group; inference retains the shared lock."""
    request = _read(request_path)
    result_path = request_path.parent / "result.json"
    case = request["case"]
    result = {"request_id": request["request_id"], "request_sha256": _digest(request), "case": case,
              "status": "failed", "validated": False}
    try:
        with _interrupt_handlers():
            identity = request["cohort_identity"]
            if (request.get("schema_version") != SCHEMA_VERSION or not CASE_RE.fullmatch(case)
                    or case not in identity.get("cases", []) or not _selected(identity, case)
                    or identity.get("scope") != "segmentation_overlap"):
                raise ValueError("Invalid segmentation case worker request")
            options = dict(request["options"])
            for key in ("dataset_root", "input_root", "output_root", "child_report_root", "prediction_root", "state_root"):
                options[key] = Path(options[key])
            args = SimpleNamespace(**options)
            if args.scope != "segmentation_overlap" or request["child_run_id"].split(f"_{case}_")[0] != args.run_id:
                raise ValueError("Case worker run or scope identity differs")
            if Path(request["result_path"]) != result_path or Path(request["progress_path"]) != request_path.parent / "progress.json":
                raise ValueError("Case worker result paths differ from its request")
            _verify({request["inventory_snapshot"]: request["inventory_sha256"]})
            if not _lightweight(identity):
                _verify(identity["code_sha256"])
                if _dependency_versions() != identity["dependency_versions"]:
                    raise ValueError("Case worker dependency versions differ")
            def stage(value):
                _write(Path(request["progress_path"]), {"request_id": request["request_id"], "case": case,
                                                        "stage": value, "updated_utc": _now()})
            inventory = _read(request["inventory_snapshot"])
            source = next(row for row in inventory["records"] if row["case"] == case)
            _active_identity(args, identity, source)
            stage("conversion")
            preparation.prepare_case(case, args.dataset_root, args.input_root, inventory=inventory)
            stage("validation")
            status = validation.main(_child_arguments(args, case, request["child_run_id"]))
            if status:
                raise RuntimeError(f"Shared segmentation validator exited with {status}")
            stage("verification")
            publication = _validate_publication(args.output_root / request["child_run_id"], case, identity, identity["runtime_identity"])
            _active_identity(args, identity, source)
            result.update(status="complete", validated=True, publication=publication)
    except (Exception, KeyboardInterrupt) as error:
        result["error"] = {"type": type(error).__name__, "message": str(error)[:1000]}
    _write(result_path, result)
    return 0 if result["status"] == "complete" else 2


def _reuse_completed_states(args, state, identity):
    """Carry forward completed immutable publications; keep old state untouched."""
    by_case = {row["case"]: row for row in state["cases"]}
    for name in getattr(args, "reuse_state_json", None) or []:
        path = _safe_path(name, args.dataset_root, (args.input_root,))
        previous = _read(path)
        manifest_path = _safe_path(Path(previous["cohort_manifest"]), args.dataset_root, (args.input_root,))
        _verify({str(manifest_path): previous["cohort_manifest_sha256"]})
        prior_manifest = _read(manifest_path)
        rows = previous.get("cases", [])
        if (str(previous.get("dataset", "")).lower() != "airrc"
                or [row.get("case") for row in rows] != identity["cases"]
                or prior_manifest.get("cases") != identity["cases"]):
            raise ValueError("Reusable state does not match the formal AirRC cohort roster")
        for field in ("scope", "input_root", "prediction_root", "annotation_package_sha256"):
            if prior_manifest.get(field) != identity.get(field):
                raise ValueError(f"Reusable cohort identity differs: {field}")
        old_inventory, new_inventory = prior_manifest.get("inventory"), identity.get("inventory")
        if (old_inventory or new_inventory) and (
                not old_inventory or not new_inventory
                or prior_manifest.get("source_sha256", {}).get(old_inventory)
                    != identity.get("source_sha256", {}).get(new_inventory)):
            raise ValueError("Reusable cohort inventory mapping differs")
        imported = 0
        for previous_row in rows:
            row = by_case[previous_row["case"]]
            if previous_row.get("status") != "complete" or row["status"] == "complete":
                continue
            publication = _safe_path(Path(previous_row["run_dir"]), args.dataset_root, (args.input_root,))
            try:
                if not _lightweight(identity):
                    _verify(previous_row["artifact_sha256"])
                    _verify(previous_row["source_sha256"])
                checked = _validate_publication(publication, row["case"], identity, identity["runtime_identity"],
                                                published_entry=previous_row)
                if checked["validation_sha256"] != previous_row["validation_sha256"]:
                    raise ValueError("Reusable publication validation changed")
            except (OSError, ValueError, KeyError, TypeError) as error:
                row.setdefault("reuse_rejections", []).append({"run_dir": str(publication),
                    "source_state": str(path), "error_type": type(error).__name__, "error": str(error)[:1000]})
            else:
                # Old attempts stay in their original state, linked here by origin.
                checked.pop("attempts", None)
                row.update(checked, reused_publication=True, reused_from_state_json=str(path))
                imported += 1
        state.setdefault("reused_state_sources", []).append({"path": str(path), "sha256": preparation.sha256(path),
            "cohort_run_id": previous.get("cohort_run_id"), "cases_imported": imported})


def run(args):
    args = _normalize(args)
    root = args.state_root / args.run_id
    state_path = root / "publications.json"
    with _lock(args.dataset_root), _child_subreaper(), _interrupt_handlers():
        inventory = _read(args.inventory)
        identity = _identity(args, inventory)
        if args.resume:
            saved = _read(root / "cohort_manifest.json")
            if saved.get("identity_sha256") != _digest(identity) or saved.get("identity") != identity:
                raise ValueError("Resume sources, calculation code or runtime changed; use a new cohort run ID")
            _verify(saved["code_snapshot_sha256"])
            state = _read(state_path)
            if (state.get("cohort_manifest_sha256") != preparation.sha256(root / "cohort_manifest.json")
                    or [row["case"] for row in state.get("cases", [])] != identity["cases"]):
                raise ValueError("Resume cohort roster or manifest identity changed")
            for row in state["cases"]:
                if row["status"] == "complete":
                    if not _lightweight(identity):
                        _verify(row["artifact_sha256"])
                        _verify(row["source_sha256"])
                    checked = _validate_publication(row["run_dir"], row["case"], identity, identity["runtime_identity"],
                                                    published_entry=row)
                    if checked["validation_sha256"] != row["validation_sha256"]:
                        raise ValueError("Resume completed publication changed")
                elif row["status"] == "running":
                    row["status"] = "pending"
                    row["attempts"][-1].update(status="interrupted", finished_utc=_now())
        else:
            if root.exists():
                raise ValueError("Cohort exists; use --resume or a new cohort run ID")
            root.mkdir(parents=True)
            snapshot = root / "source_code"
            snapshot.mkdir()
            for name in identity["code_sha256"]:
                shutil.copyfile(name, snapshot / Path(name).name)
            snapshot_hashes = {str(snapshot / Path(name).name): digest for name, digest in identity["code_sha256"].items()}
            _verify(snapshot_hashes)
            manifest = {**identity, "identity": identity, "identity_sha256": _digest(identity), "created_utc": _now(),
                        "code_snapshot_sha256": snapshot_hashes}
            _write(root / "cohort_manifest.json", manifest)
            _write(root / "inventory_snapshot.json", inventory)
            state = {"schema_version": SCHEMA_VERSION, "dataset": "AIRRC", "cohort_run_id": args.run_id,
                     "cohort_manifest": str(root / "cohort_manifest.json"),
                     "cohort_manifest_sha256": preparation.sha256(root / "cohort_manifest.json"),
                     "status": "initialized", "cases": [{"case": case, "status": "pending", "run_dir": None,
                                                            "attempts": []} for case in identity["cases"]]}
            for publication in args.reuse_run_dir or []:
                publication = _safe_path(publication, args.dataset_root, (args.input_root,))
                candidate_manifest = _read(publication / "run_manifest.json")
                for row in state["cases"]:
                    if row["status"] == "complete" or row["case"] not in candidate_manifest.get("prediction_cases", []):
                        continue
                    try:
                        checked = _validate_publication(publication, row["case"], identity, identity["runtime_identity"])
                    except (OSError, ValueError, KeyError, TypeError) as error:
                        row.setdefault("reuse_rejections", []).append({"run_dir": str(publication), "error_type": type(error).__name__,
                                                                      "error": str(error)[:1000]})
                    else:
                        row.update(checked, reused_publication=True)
            _reuse_completed_states(args, state, identity)
        _save_state(state_path, state)
        _refresh_report(args, root, state_path, state)
        if args.init_only:
            print(f"Initialized {len(identity['cases'])} cases; {state['counts'].get('complete', 0)} verified publications reused.", flush=True)
            return state
        if args.device == "gpu":
            import torch
            if not torch.cuda.is_available():
                raise RuntimeError("GPU inference requested but CUDA is unavailable")
        if getattr(args, "workers", 1) > 1:
            return _run_parallel_cases(args, root, state_path, state, identity, inventory)
        by_case = {row["case"]: row for row in inventory["records"]}
        state["status"] = "running"
        processed = 0
        active_row = None
        try:
            for row in state["cases"]:
                if not _selected(identity, row["case"]):
                    continue
                if row["status"] == "complete" or (row["status"] == "failed" and not args.retry_failed):
                    continue
                if args.max_cases is not None and processed >= args.max_cases:
                    state["status"] = "paused_after_case_limit"
                    break
                _check_disk(args.dataset_root, by_case[row["case"]], args.min_free_gib)
                _active_identity(args, identity, by_case[row["case"]])
                _preserve_incomplete_inputs(args, row)
                active_row = row
                attempt = len(row["attempts"]) + 1
                child_id = f"{args.run_id}_{row['case']}_a{attempt:03d}"
                while (args.output_root / child_id).exists() or (args.child_report_root / child_id).exists():
                    attempt += 1
                    child_id = f"{args.run_id}_{row['case']}_a{attempt:03d}"
                logs = root / "attempts" / row["case"] / child_id
                record = {"run_id": child_id, "status": "running", "stage": "conversion",
                          "started_utc": _now(), "log_dir": str(logs)}
                row["attempts"].append(record)
                row.update(status="running", current_stage="conversion")
                _save_state(state_path, state)
                print(f"{row['case']}: conversion; {state['counts'].get('complete', 0)}/{len(state['cases'])} complete", flush=True)
                try:
                    converted = _subprocess([sys.executable, str(Path(__file__).resolve()), "--prepare-worker",
                                             "--dataset-root", str(args.dataset_root), "--input-root", str(args.input_root),
                                             "--inventory", str(root / "inventory_snapshot.json"), "--worker-case", row["case"]],
                                            logs / "conversion.log", args.case_timeout_seconds)
                    if converted:
                        raise RuntimeError(f"Conversion worker exited with {converted}; see conversion.log")
                    row["current_stage"] = "validation"
                    record["stage"] = "validation"
                    _save_state(state_path, state)
                    command = [sys.executable, str(Path(__file__).with_name("run_airrc_airway_validation.py")),
                               *_child_arguments(args, row["case"], child_id)]
                    result = _subprocess(command, logs / "validation.log", args.case_timeout_seconds)
                    if result:
                        raise RuntimeError(f"Validation worker exited with {result}; see validation.log")
                    checked = _validate_publication(args.output_root / child_id, row["case"], identity, identity["runtime_identity"])
                    _active_identity(args, identity, by_case[row["case"]])
                    row.update(checked, batch_verified_dependency_versions=identity["dependency_versions"])
                    row.pop("current_stage", None)
                    row.pop("error", None)
                    record.update(status="complete", finished_utc=_now())
                except (OSError, ValueError, KeyError, TypeError, RuntimeError, subprocess.TimeoutExpired) as error:
                    row.update(status="failed", error={"type": type(error).__name__, "message": str(error)[:1000]})
                    record.update(status="failed", finished_utc=_now(), error=row["error"])
                processed += 1
                active_row = None
                _save_state(state_path, state)
                print(f"{row['case']}: {row['status']}; {state['counts'].get('complete', 0)}/{len(state['cases'])} complete", flush=True)
                if args.report_every and processed % args.report_every == 0:
                    _refresh_report(args, root, state_path, state)
            else:
                state["status"] = _finished_status(state, identity)
        except BaseException as error:
            if active_row is not None:
                active_row["status"] = "pending"
                active_row["attempts"][-1].update(status="interrupted", finished_utc=_now())
            state.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "paused_error",
                         stop_reason={"type": type(error).__name__, "message": str(error)[:1000]})
            _save_state(state_path, state)
            _refresh_report(args, root, state_path, state)
            raise
        _save_state(state_path, state)
        _refresh_report(args, root, state_path, state)
        return state


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("data/airrc"))
    parser.add_argument("--inventory", type=Path)
    for option in ("input-root", "output-root", "child-report-root", "prediction-root", "state-root", "report-root"):
        parser.add_argument(f"--{option}", type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--case", action="append", help="Restrict execution to these cases while retaining the formal 254-case roster.")
    parser.add_argument("--device", choices=("gpu", "cpu"), default="gpu")
    parser.add_argument("--scope", choices=("segmentation_overlap",), default="segmentation_overlap")
    parser.add_argument("--workers", type=int, choices=tuple(range(1, 9)), default=1, help="Bounded case concurrency (default: 1).")
    parser.add_argument("--inference-workers", type=int, choices=(1, 2, 4), default=1, help="Bounded inference processes on one shared GPU (default: 1).")
    parser.add_argument("--case-timeout-seconds", type=float, default=7200.)
    parser.add_argument("--min-free-gib", type=float, default=20.)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--reuse-run-dir", action="append", type=Path)
    parser.add_argument("--reuse-state-json", action="append", type=Path,
                        help="Reuse completed publications from a previous cohort without rerunning its cases.")
    parser.add_argument("--audit-masks", action="store_true",
                        help="Explicitly rehash source data and independently remeasure masks; overlap batches use saved metrics by default.")
    parser.add_argument("--init-only", action="store_true")
    parser.add_argument("--max-cases", type=int, help="Stop after this many new attempts, retaining the full cohort roster.")
    parser.add_argument("--build-report", action="store_true", help="Refresh the cohort report initially, periodically and when this invocation stops.")
    parser.add_argument("--report-every", type=int, default=0,
                        help="Refresh summaries after this many attempts; 0 disables periodic refresh (default: 0).")
    parser.add_argument("--case-worker-request", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--prepare-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--worker-case", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.case_worker_request is not None:
        return _case_worker(args.case_worker_request)
    if args.prepare_worker:
        try:
            if args.worker_case is None or args.inventory is None or args.input_root is None:
                raise ValueError("A preparation request must identify its case, inventory and output")
            preparation.prepare_case(args.worker_case, args.dataset_root, args.input_root, inventory=_read(args.inventory))
            return 0
        except Exception as error:
            print(f"Preparation failed: {type(error).__name__}", file=sys.stderr)
            return 1
    if args.run_id is None:
        parser.error("--run-id is required")
    try:
        state = run(args)
    except BatchInterrupted as error:
        print(str(error), file=sys.stderr)
        return 128 + error.signum
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as error:
        print(f"AirRC batch failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    return 0 if state["status"] in ("complete", "initialized", "paused_after_case_limit", "paused_after_case_selection") else 2


if __name__ == "__main__":
    raise SystemExit(main())
