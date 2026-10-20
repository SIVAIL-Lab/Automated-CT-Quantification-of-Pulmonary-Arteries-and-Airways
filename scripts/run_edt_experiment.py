#!/usr/bin/env python3
"""Calibrated Euclidean distance transforms and weighted voxel paths.

Coordinates are zero-based NIfTI array indices (i,j,k), NOT DICOM LPS / world XYZ.
EDT measures distance to background voxel CENTRES. It is not a validated
cross-sectional radius. Path count is NOT anatomical branch count.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy import ndimage as ndi

VERSION = "0.2.0"


def binary_mask(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    if values.ndim != 3:
        raise ValueError("Expected a 3-D binary mask")
    unique = np.unique(values)
    if not (np.all(np.isin(unique, [0, 1])) or np.all(np.isin(unique, [0, 255]))):
        raise ValueError("Expected a finite binary mask: 0/1 or 0/255, not probabilities/multilabels")
    mask = values != 0
    if not mask.any():
        raise ValueError("Mask is empty")
    return mask


def validate_spacing(spacing: Sequence[float]) -> np.ndarray:
    spacing = np.asarray(spacing, dtype=float)
    if spacing.shape != (3,) or not np.isfinite(spacing).all() or (spacing <= 0).any():
        raise ValueError("Spacing must contain three finite positive values in mm")
    return spacing


def mask_bbox(mask: np.ndarray) -> tuple[slice, slice, slice]:
    slices = []
    for axis in range(3):
        occupied = np.flatnonzero(mask.any(axis=tuple(a for a in range(3) if a != axis)))
        if not len(occupied):
            raise ValueError("Mask is empty")
        slices.append(slice(int(occupied[0]), int(occupied[-1]) + 1))
    return tuple(slices)


def boundary_contact_summary(mask: np.ndarray) -> dict[str, Any]:
    """Count unique foreground boundary voxels and contacts with each FOV face."""
    mask = binary_mask(mask)
    boundary = np.zeros(mask.shape, dtype=bool)
    faces = {}
    for axis, name in enumerate(("i", "j", "k")):
        for index, side in ((0, "min"), (-1, "max")):
            selection = [slice(None)] * 3
            selection[axis] = index
            selection = tuple(selection)
            faces[f"{name}_{side}"] = int(np.count_nonzero(mask[selection]))
            boundary[selection] |= mask[selection]
    count = int(np.count_nonzero(boundary))
    return {"touches_boundary": bool(count), "boundary_voxel_count": count,
            "per_face_counts": faces}


def compute_edt(mask: np.ndarray, spacing_mm: Sequence[float],
                max_crop_voxels: int = 150_000_000) -> np.ndarray:
    """Compute cropped/padded EDT, then restore the input shape (float32).

One zero-valued layer is added outside the foreground bounding box. If the
mask touches the image edge this treats unknown exterior as background; record
and review this truncation. No resampling, pruning, closing or gap bridging.
"""
    mask = binary_mask(mask)
    spacing = validate_spacing(spacing_mm)
    bbox = mask_bbox(mask)
    padded_shape = np.array([s.stop - s.start + 2 for s in bbox], dtype=np.int64)
    if max_crop_voxels < 1 or int(np.prod(padded_shape)) > max_crop_voxels:
        raise ValueError("EDT crop exceeds --max-crop-voxels; review memory before raising the limit")
    padded = np.pad(mask[bbox], 1, mode="constant", constant_values=False)
    cropped = ndi.distance_transform_edt(padded, sampling=spacing)[1:-1, 1:-1, 1:-1]
    result = np.zeros(mask.shape, dtype=np.float32)
    result[bbox] = cropped
    return result


def parse_point(value: Any) -> tuple[int, int, int]:
    if not isinstance(value, (list, tuple, np.ndarray)) or len(value) != 3:
        raise ValueError("Each point must be an integer [i,j,k] triplet")
    if any(isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)) for v in value):
        raise ValueError("Voxel coordinates must be integers; no implicit rounding or snapping")
    return tuple(int(v) for v in value)


def inside(point: tuple[int, ...], shape: tuple[int, ...]) -> bool:
    return all(0 <= p < n for p, n in zip(point, shape))


def bounded_traceback(root: tuple[int, int, int], target: tuple[int, int, int],
                      cost: np.ndarray, predecessor: np.ndarray, offsets: np.ndarray,
                      domain: np.ndarray, max_steps: int) -> list[tuple[int, int, int]]:
    """Follow stored predecessors with an explicit loop/bounds/monotonicity guard."""
    current = target
    reverse_path = []
    seen = set()
    for _ in range(max_steps + 1):
        if not inside(current, domain.shape) or not domain[current]:
            raise RuntimeError("Predecessor left the foreground domain")
        if current in seen:
            raise RuntimeError("Predecessor cycle detected")
        seen.add(current)
        reverse_path.append(current)
        if current == root:
            return reverse_path[::-1]
        direction = int(predecessor[current])
        if direction < 0 or direction >= len(offsets):
            raise RuntimeError("Missing predecessor before reaching root")
        offset = np.asarray(offsets[direction])
        if offset.shape != (3,) or not np.isfinite(offset).all() or not np.equal(offset, np.floor(offset)).all():
            raise RuntimeError("Invalid predecessor offset")
        if np.max(np.abs(offset)) != 1:
            raise RuntimeError("Invalid predecessor adjacency")
        previous = tuple(int(c - d) for c, d in zip(current, offsets[direction]))
        if not inside(previous, domain.shape):
            raise RuntimeError("Predecessor out of bounds")
        if not (np.isfinite(cost[previous]) and cost[previous] < cost[current]):
            raise RuntimeError("Predecessor cost is not strictly decreasing")
        current = previous
    raise RuntimeError("Predecessor step limit exceeded")


def audit_path(points: np.ndarray, root: tuple[int, int, int],
               target: tuple[int, int, int], domain: np.ndarray,
               density: np.ndarray, cumulative: np.ndarray,
               spacing_mm: Sequence[float], connectivity: int,
               max_steps: int) -> dict[str, Any]:
    """Independently check a recovered path and its physical trapezoidal cost."""
    points = np.asarray(points)
    if (points.ndim != 2 or points.shape[1:] != (3,) or len(points) < 2
            or not np.issubdtype(points.dtype, np.integer)):
        raise RuntimeError("Path must contain at least two integer voxel coordinates")
    if tuple(points[0]) != root or tuple(points[-1]) != target:
        raise RuntimeError("Path endpoints differ from the original seeds")
    if np.any(points < 0) or np.any(points >= np.asarray(domain.shape)):
        raise RuntimeError("Path contains out-of-bounds points")
    indices = tuple(points.T)
    if not domain[indices].all():
        raise RuntimeError("Path left the foreground domain")
    delta = np.diff(points, axis=0)
    changed_axes = np.count_nonzero(delta, axis=1)
    if np.any(np.abs(delta) > 1) or np.any(changed_axes == 0):
        raise RuntimeError("Path has invalid adjacency")
    if connectivity == 6 and np.any(changed_axes != 1):
        raise RuntimeError("Path has diagonal adjacency under six-connectivity")
    if len(np.unique(points, axis=0)) != len(points):
        raise RuntimeError("Path contains repeated voxels")
    if len(points) - 1 > max_steps:
        raise RuntimeError("Path step limit exceeded")
    costs = cumulative[indices]
    increments = np.diff(costs)
    if not np.isfinite(costs).all() or np.any(increments <= 0):
        raise RuntimeError("Path cost is not strictly increasing from the root")
    if costs[0] != 0:
        raise RuntimeError("Root accumulated cost must be zero")
    lengths = np.linalg.norm(delta * validate_spacing(spacing_mm), axis=1)
    weights = density[indices]
    if not np.isfinite(weights).all() or np.any(weights <= 0):
        raise RuntimeError("Path contains invalid cost density")
    recomputed = float(np.sum(lengths * (weights[:-1] + weights[1:]) * 0.5))
    accumulated = float(costs[-1])
    if not np.isclose(recomputed, accumulated, rtol=1e-9, atol=1e-10):
        raise RuntimeError(f"Recomputed geometric cost differs from MCP cost: {recomputed} vs {accumulated}")
    return {"audit_passed": True, "accumulated_cost": accumulated,
            "recomputed_cost": recomputed, "cost_abs_error": abs(recomputed - accumulated),
            "minimum_cost_decrement": float(increments.min()),
            "diagonal_step_count": int(np.count_nonzero(changed_axes > 1)),
            "audit": {"endpoints_match": True, "foreground_contained": True,
                      "adjacency_valid": True, "no_repeated_voxels": True,
                      "step_limit_passed": True, "strictly_monotonic_cost": True,
                      "geometric_cost_consistent": True}}


def trace_edt_paths(mask: np.ndarray, edt: np.ndarray, spacing_mm: Sequence[float],
                    root: Sequence[int], targets: Sequence[Sequence[int]], *,
                    connectivity: int = 6, power: float = 2.0,
                    epsilon_mm: float = 0.1) -> dict[str, Any]:
    """EDT-weighted Dijkstra/MCP_Geometric prototype, NOT Fast Marching.

Returns paths only to explicitly provided targets; does not extract a complete
anatomical tree or infer missing connections. Components use the same grid
connectivity as the path solver. 26-connectivity permits corner contacts.
"""
    from skimage.graph import MCP_Geometric

    mask = binary_mask(mask)
    spacing = validate_spacing(spacing_mm)
    if connectivity not in (6, 26):
        raise ValueError("Connectivity must be 6 or 26")
    if not np.isfinite(power) or not 0 < power <= 8:
        raise ValueError("power must be finite and in (0,8]")
    if not np.isfinite(epsilon_mm) or epsilon_mm <= 0:
        raise ValueError("epsilon_mm must be finite and positive")
    if edt.shape != mask.shape or not np.isfinite(edt).all() or np.any(edt[mask] <= 0):
        raise ValueError("EDT must match mask and be finite/positive in the foreground")
    root = parse_point(root)
    targets = [parse_point(p) for p in targets]
    if not targets:
        raise ValueError("At least one target is required")
    if len(set(targets)) != len(targets):
        raise ValueError("Targets contain duplicates")
    if root in targets:
        raise ValueError("Root cannot also be a target")
    if not inside(root, mask.shape) or not mask[root]:
        raise ValueError("Root is outside the mask; explicit correction is required")

    bbox = mask_bbox(mask)
    origin = np.array([s.start for s in bbox], dtype=int)
    crop = mask[bbox]
    labels, n_components = ndi.label(crop, ndi.generate_binary_structure(3, 1 if connectivity == 6 else 3))
    labels_six = labels if connectivity == 6 else ndi.label(crop, ndi.generate_binary_structure(3, 1))[0]
    local_root = tuple(np.array(root) - origin)
    root_label = int(labels[local_root])
    root_label_six = int(labels_six[local_root])
    records = []
    reachable = []
    for index, target in enumerate(targets):
        local_target = tuple(np.array(target) - origin)
        if not inside(target, mask.shape):
            status = "outside_image"
        elif not mask[target]:
            status = "outside_mask"
        elif int(labels[local_target]) != root_label:
            status = "disconnected"
        else:
            status = "pending"
            reachable.append(local_target)
        reachable_six = bool(inside(target, mask.shape) and mask[target]
                             and int(labels_six[local_target]) == root_label_six)
        records.append({"target_id": index + 1, "target_ijk": list(target), "status": status,
                        "topologically_reachable": status == "pending",
                        "reachable_under_6": reachable_six,
                        "audit_passed": False, "six_component_ids": [], "six_component_count": 0})

    path_results = []
    if reachable:
        domain = labels == root_label
        costs = np.full(crop.shape, np.inf, dtype=float)
        # Density is (EDT_mm + epsilon_mm)^(-power); MCP edge lengths are mm.
        costs[domain] = np.power(1.0 / (edt[bbox][domain].astype(float) + epsilon_mm), power)
        if not np.isfinite(costs[domain]).all() or np.any(costs[domain] <= 0):
            raise ValueError("Nonpositive/nonfinite path weights; review spacing and parameters")
        solver = MCP_Geometric(costs, fully_connected=(connectivity == 26), sampling=tuple(spacing))
        cumulative, predecessor = solver.find_costs([local_root], ends=reachable, find_all_ends=True)
        max_steps = int(domain.sum())
        for record in records:
            if record["status"] != "pending":
                continue
            endpoint = tuple(np.array(record["target_ijk"]) - origin)
            try:
                if not np.isfinite(cumulative[endpoint]):
                    raise RuntimeError("Topologically reachable target has nonfinite path cost")
                local_path = bounded_traceback(local_root, endpoint, cumulative, predecessor,
                                              solver.offsets, domain, max_steps)
                local_points = np.asarray(local_path, dtype=int)
                audit = audit_path(local_points, local_root, endpoint, domain, costs,
                                   cumulative, spacing, connectivity, max_steps)
                components_six = np.unique(labels_six[tuple(local_points.T)]).astype(int).tolist()
                ijk = local_points + origin
                lengths = np.linalg.norm(np.diff(ijk, axis=0) * spacing, axis=1)
                radii = edt[tuple(ijk.T)]
                record.update(status="ok", path_length_mm=float(lengths.sum()),
                              point_count=int(len(ijk)), mean_edt_sample_mm=float(radii.mean()),
                              six_component_ids=components_six, six_component_count=len(components_six),
                              **audit)
                path_results.append({"target_id": record["target_id"], "points_ijk": ijk.tolist(),
                                     "edt_mm": radii.astype(float).tolist()})
            except RuntimeError as exc:
                record.update(status="traceback_failed", error=str(exc))

    n_reachable = len(reachable)
    n_success = sum(r["status"] == "ok" for r in records)
    return {"method": "EDT-weighted MCP_Geometric (Dijkstra), not Fast Marching",
            "root_ijk": list(root), "connectivity": connectivity,
            "component_count": int(n_components), "power": power, "epsilon_mm": epsilon_mm,
            "target_count": len(targets), "reachable_target_count": n_reachable,
            "successful_target_count": n_success,
            "success_fraction_all_targets": n_success / len(targets),
            "success_fraction_among_reachable": n_success / n_reachable if n_reachable else None,
            "technical_validation": "passed" if n_success == n_reachable else "failed",
            "six_component_count": int(labels_six.max()),
            "cost_definition": "integral of (EDT_mm + epsilon_mm)^(-power) along voxel-centre edges",
            "targets": records, "paths": path_results,
            "branch_count": None, "anatomical_validation": "not_performed"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def geometry_mm(image: Any) -> tuple[np.ndarray, np.ndarray]:
    unit = image.header.get_xyzt_units()[0]
    scale = {"mm": 1.0, "meter": 1000.0, "micron": 0.001}.get(unit)
    if scale is None:
        raise ValueError(f"NIfTI spatial unit must be known, got {unit!r}")
    affine = np.asarray(image.affine, dtype=float)
    spacing = validate_spacing(image.header.get_zooms()[:3])
    if affine.shape != (4, 4) or not np.isfinite(affine).all():
        raise ValueError("Invalid NIfTI affine")
    if not np.allclose(affine[3], [0, 0, 0, 1]):
        raise ValueError("Invalid homogeneous affine")
    directions = affine[:3, :3] / spacing
    if not np.allclose(directions.T @ directions, np.eye(3), atol=1e-5, rtol=1e-5):
        raise ValueError("Affine must have orthogonal axes matching header spacing; shear is unsupported")
    world_mm = affine.copy()
    world_mm[:3, :] *= scale
    return spacing * scale, world_mm


def save_like(data: np.ndarray, reference: Any, path: Path) -> None:
    import nibabel as nib
    header = reference.header.copy()
    header.set_data_dtype(data.dtype)
    header.set_slope_inter(1.0, 0.0)
    header.set_intent("none")
    header["cal_min"] = 0
    header["cal_max"] = float(np.max(data))
    output = reference.__class__(data, reference.affine, header=header)
    # Preserve both transform matrices/codes even when one code is zero.
    output.set_qform(reference.get_qform(), int(reference.header["qform_code"]))
    output.set_sform(reference.get_sform(), int(reference.header["sform_code"]))
    nib.save(output, str(path))
