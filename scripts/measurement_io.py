"""Shared calibrated image, table and input-integrity helpers."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage

from run_edt_experiment import geometry_mm, sha256


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_csv(path: Path, records: list[dict]) -> None:
    fields = list(dict.fromkeys(key for row in records for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in records:
            writer.writerow({key: json.dumps(value, allow_nan=False) if isinstance(value, (list, dict))
                             else value for key, value in row.items()})


def verify_hashes(hashes: dict[str, str]) -> None:
    for name, expected in hashes.items():
        if sha256(Path(name)) != expected:
            raise ValueError(f"Source changed or hash mismatch: {name}")


def largest_component(mask: np.ndarray, voxel_volume_mm3: float) -> tuple[np.ndarray, dict]:
    labels, count = ndimage.label(mask, structure=np.ones((3, 3, 3), dtype=np.uint8))
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    if count == 0:
        raise ValueError("A nonempty artery mask is required")
    chosen = int(np.argmax(sizes))
    selected = labels == chosen
    total, retained = int(mask.sum()), int(sizes[chosen])
    return selected, {
        "connectivity": 26, "component_count": int(count), "selected_component_id": chosen,
        "input_voxels": total, "selected_voxels": retained, "excluded_voxels": total - retained,
        "input_volume_ml": total * voxel_volume_mm3 / 1000.,
        "selected_volume_ml": retained * voxel_volume_mm3 / 1000.,
        "excluded_volume_ml": (total - retained) * voxel_volume_mm3 / 1000.,
        "retained_fraction": retained / total,
    }


def matching_image(path: Path, reference, description: str):
    image = nib.load(str(path))
    spacing, affine = geometry_mm(image)
    ref_spacing, ref_affine = geometry_mm(reference)
    if (image.shape != reference.shape or not np.allclose(affine, ref_affine, atol=1e-4, rtol=0)
            or not np.allclose(spacing, ref_spacing, atol=1e-6, rtol=0)):
        raise ValueError(f"{description} geometry differs from mask")
    return image


def aligned_image(path: Path, reference, description: str):
    return matching_image(path, reference, description)
