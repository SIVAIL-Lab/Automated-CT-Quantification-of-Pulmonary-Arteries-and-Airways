"""Validate original PARSE inputs and the audited AirRC conversion layout."""

from __future__ import annotations

import hashlib
from pathlib import Path

import nibabel as nib


GROUP_DIRECTORIES = {"ct": "image", "artery": "label", "vein": "label_vein"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_case_conversion(
    case_dir: Path, *, groups: tuple[str, ...] = ("ct",),
) -> dict[str, dict[str, str]]:
    """Verify AirRC provenance; original PARSE volumes need no conversion manifest."""
    if not groups or any(group not in GROUP_DIRECTORIES for group in groups):
        raise ValueError(f"Unknown or empty conversion groups: {groups!r}")
    case_dir = Path(case_dir).expanduser().resolve()
    legacy_manifest = case_dir / "conversion_report.json"
    if legacy_manifest.exists() or legacy_manifest.is_symlink():
        raise ValueError("Unsupported conversion layout; only original PARSE and audited AirRC inputs are supported")
    airrc_manifest = case_dir / "airrc_conversion.json"
    if airrc_manifest.exists() or airrc_manifest.is_symlink():
        from prepare_airrc import verify_airrc_conversion

        return verify_airrc_conversion(case_dir, groups=groups)
    for group in dict.fromkeys(groups):
        path = case_dir / GROUP_DIRECTORIES[group] / f"{case_dir.name}.nii.gz"
        if not path.is_file():
            continue
        description = bytes(nib.load(path).header["descrip"])
        if description.startswith(b"AIRRC converted"):
            raise ValueError("Converted AIRRC case is missing its conversion report")
        if description.startswith(b"HIPAS converted"):
            raise ValueError("Unsupported converted dataset; only PARSE and AirRC are supported")
    return {"sources": {}, "source_sha256": {}}
