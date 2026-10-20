"""Validate CT and airway-lumen inputs without changing their geometry.

The outer surface encloses lumen plus wall. A wall-only mask is derived only
when the independently thresholded inner and outer exports are nested.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np

from run_lung_volume_parse import _binary_data, _check_geometry, _sha256
from run_lung_volume_parse import _voxel_volume_mm3


def _metadata(path: Path, image: nib.Nifti1Image, data: np.ndarray) -> dict[str, Any]:
    qform, qcode = image.get_qform(coded=True)
    sform, scode = image.get_sform(coded=True)
    finite = bool(np.all(np.isfinite(data)))
    spatial_unit, temporal_unit = image.header.get_xyzt_units()
    return {
        "path": str(path.resolve()), "sha256": _sha256(path),
        "shape": list(image.shape), "affine": image.affine.tolist(),
        "affine_mm": image.affine.tolist() if spatial_unit == "mm" else None,
        "qform": None if qform is None else qform.tolist(), "qform_code": int(qcode),
        "sform": None if sform is None else sform.tolist(), "sform_code": int(scode),
        "spacing": [float(value) for value in image.header.get_zooms()[:3]],
        "spatial_unit": spatial_unit, "temporal_unit": temporal_unit,
        "stored_dtype": str(image.get_data_dtype()), "finite": finite,
        "intensity_min": float(data.min()) if finite and data.size else None,
        "intensity_max": float(data.max()) if finite and data.size else None,
        "stored_scaling_slope": float(image.dataobj.slope),
        "stored_scaling_intercept": float(image.dataobj.inter),
        "voxel_volume_mm3": None,
    }


def _load(path: Path) -> tuple[nib.Nifti1Image, np.ndarray, dict[str, Any]]:
    image = nib.load(str(path))
    # Apply only the scaling stored in the NIfTI; do not window/normalize CT HU.
    data = image.get_fdata(dtype=np.float32)
    return image, data, _metadata(path, image, data)


def _validate_image(image: nib.Nifti1Image, metadata: dict[str, Any]) -> None:
    if len(image.shape) != 3 or any(size <= 0 for size in image.shape):
        raise ValueError("Expected a nonempty 3-D NIfTI volume")
    if image.header.get_xyzt_units()[0] != "mm":
        raise ValueError("NIfTI spatial units must be mm; no implicit unit conversion is applied")
    metadata["voxel_volume_mm3"] = _voxel_volume_mm3(image)
    if not metadata["finite"]:
        raise ValueError("NIfTI voxel values must be finite")
    for name in ("qform", "sform"):
        form, code = getattr(image, f"get_{name}")(coded=True)
        if code and (not np.all(np.isfinite(form)) or abs(np.linalg.det(form[:3, :3])) <= 0):
            raise ValueError(f"NIfTI {name} must be finite and nonsingular")


def _aligned(reference: nib.Nifti1Image, image: nib.Nifti1Image) -> None:
    _check_geometry(reference, image)
    # Check the encoded transforms too: matching best affines alone can conceal
    # a conflicting qform used by a different NIfTI reader in the legacy runtime.
    for name in ("qform", "sform"):
        expected, expected_code = getattr(reference, f"get_{name}")(coded=True)
        actual, actual_code = getattr(image, f"get_{name}")(coded=True)
        if expected_code != actual_code or (
            expected_code and not np.allclose(expected, actual, rtol=1e-5, atol=1e-4)
        ):
            raise ValueError(f"Volume {name} or its code differs from the CT")


def _base_report() -> dict[str, Any]:
    return {
        "status": "needs_review", "geometry_usable": False, "reasons": [],
        "anatomical_validation": "not_performed",
    }


def inspect_inputs(ct_path: Path, mask_path: Path) -> dict[str, Any]:
    """Report CT/lumen geometry, stored scaled intensities and source identities.

    Unusable inputs return ``needs_review`` with a reason; they are never silently
    resampled, relabelled or assigned new units. No image arrays are returned.
    """
    report = _base_report()
    images = {}
    for name, source in (("ct", Path(ct_path)), ("mask", Path(mask_path))):
        report[name] = {"path": str(source.resolve())}
        try:
            image, data, metadata = _load(source)
            report[name] = metadata
            _validate_image(image, metadata)
            if name == "mask":
                binary = _binary_data(image)
                count = int(np.count_nonzero(binary))
                metadata.update(voxel_count=count,
                                volume_mm3=count * metadata["voxel_volume_mm3"],
                                volume_ml=count * metadata["voxel_volume_mm3"] / 1000.0,
                                observed_values=[float(value) for value in np.unique(data)])
                if not count:
                    raise ValueError("Initial airway lumen mask must be nonempty")
            images[name] = image
        except (OSError, ValueError, nib.filebasedimages.ImageFileError) as exc:
            report["reasons"].append(f"{name}: {exc}")
    if len(images) == 2:
        try:
            _aligned(images["ct"], images["mask"])
        except ValueError as exc:
            report["reasons"].append(str(exc))
    if not report["reasons"]:
        report.update(status="ok", geometry_usable=True)
    report["intensity_handling"] = "Stored NIfTI scaling applied; no CT normalization or windowing"
    return report
