"""Reuse fully validated original lung unions within a PARSE derived scope."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import uuid

import nibabel as nib

from dataset_conversion import verify_case_conversion
from measurement_scope import load_profile, source_results_root
import run_lung_volume_parse as lungs


def _digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _identity() -> dict:
    files = (Path(__file__), Path(lungs.__file__), Path(__file__).with_name("dataset_conversion.py"))
    return {"lung_schema_version": lungs.SCHEMA_VERSION,
            "lung_parameters": json.loads(json.dumps(lungs.PARAMETERS)),
            "code_sha256": {str(path.resolve()): lungs._sha256(path) for path in files}}


def _read(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _snapshot(ct: Path, directory: Path, profile_path: Path) -> dict:
    paths = [ct, profile_path, directory / "lung_volume_report.json",
             directory / "totalsegmentator_report.json",
             *(directory / f"{name}.nii.gz" for name in lungs.REGIONS)]
    conversion = verify_case_conversion(ct.parent.parent, groups=("ct",))
    return {**conversion["source_sha256"],
            **{str(path): lungs._sha256(path) for path in paths}}


def load_lungs(case_dir: Path, derived_root: Path) -> tuple[dict, dict]:
    """Validate raw lung unions once, then verify a complete hash-bound receipt.

    Only the receipt is written under the derived PARSE root. The original CT,
    lobe masks, bilateral unions and reports remain unchanged. A stale receipt
    is an explicit error rather than an automatic change of measurement scope.
    """
    case_dir, root = Path(case_dir).resolve(), Path(derived_root).resolve()
    profile = load_profile(root)
    if (profile is None or not re.fullmatch(r"PA[0-9]{6}", case_dir.name)
            or case_dir.parent != Path(profile["input_root"]).resolve()):
        raise ValueError("Lung validation receipts require a matching PARSE largest_26 profile")
    ct = lungs._case_file(case_dir).resolve()
    directory = source_results_root(root) / "lung_volume" / case_dir.name
    receipt_path = root / "lung_reuse" / case_dir.name / "validation.json"
    before = _snapshot(ct, directory, root / "measurement_scope.json")
    identity = _identity()
    report_path = directory / "lung_volume_report.json"
    if receipt_path.exists():
        receipt = _read(receipt_path)
        payload = {key: value for key, value in receipt.items() if key != "payload_sha256"}
        if receipt.get("payload_sha256") != _digest(payload):
            raise ValueError("Lung validation receipt payload was changed")
        if (receipt.get("schema_version") != 1 or receipt.get("case") != case_dir.name
                or receipt.get("measurement_scope") != "largest_26"
                or receipt.get("validation") != "full_raw_lung_union_validation_passed"
                or receipt.get("source_sha256") != before
                or receipt.get("validation_identity") != identity
                or receipt.get("raw_report_path") != str(report_path)):
            raise ValueError("Lung validation receipt sources, code, parameters, or identity are stale")
        report = _read(report_path)
        if receipt.get("report_payload_sha256") != _digest(report):
            raise ValueError("Validated lung report payload differs from its receipt")
        images = {name: nib.load(directory / f"{name}.nii.gz") for name in lungs.REGIONS}
    else:
        images, report = lungs._load_validated_result(ct, directory)
        if report.get("case") != case_dir.name or report != _read(report_path):
            raise ValueError("Validated lung report differs from the requested saved case report")
        if _snapshot(ct, directory, root / "measurement_scope.json") != before or _identity() != identity:
            raise ValueError("Lung validation inputs or code changed during full validation")
        receipt = {"schema_version": 1, "case": case_dir.name, "measurement_scope": "largest_26",
                   "validation": "full_raw_lung_union_validation_passed",
                   "lung_policy": "Unchanged original lobe unions; no component filtering",
                   "source_sha256": before, "validation_identity": identity,
                   "raw_report_path": str(report_path), "report_payload_sha256": _digest(report)}
        receipt["payload_sha256"] = _digest(receipt)
        receipt_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = receipt_path.with_name(".validation_" + uuid.uuid4().hex + ".tmp")
        try:
            temporary.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n", encoding="utf-8")
            temporary.replace(receipt_path)
        finally:
            temporary.unlink(missing_ok=True)
    if _snapshot(ct, directory, root / "measurement_scope.json") != before or _identity() != identity:
        raise ValueError("Lung sources or validation code changed during receipt verification")
    return images, report
