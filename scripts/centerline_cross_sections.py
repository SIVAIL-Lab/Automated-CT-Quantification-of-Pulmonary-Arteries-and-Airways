"""Orthogonal mask sections on immutable, physically calibrated tree segments.

These measurements describe the sampled binary mask. They do not establish
anatomical vessel identity or replace the existing EDT segment radius.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from math import fsum, pi, sqrt
from typing import Callable

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class Config:
    """Physical sampling settings; linearity is PCA lambda_max / sum(lambda)."""

    step_mm: float = 1.0
    tangent_window_mm: float = 4.0
    tangent_support_step_mm: float = 0.25
    min_tangent_span_mm: float = 2.0
    min_linearity: float = 0.8
    plane_spacing_mm: float = 0.25
    window_half_width_mm: float = 8.0
    max_window_half_width_mm: float = 64.0
    junction_margin_mm: float = 2.0
    endpoint_margin_mm: float = 1.0

    def validate(self) -> None:
        for name, value in asdict(self).items():
            if isinstance(value, bool) or not np.isfinite(value):
                raise ValueError(f"{name} must be a finite number")
            if name in {"junction_margin_mm", "endpoint_margin_mm"}:
                if value < 0:
                    raise ValueError(f"{name} must be nonnegative")
            elif value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.min_linearity > 1:
            raise ValueError("min_linearity must be at most one")
        if self.min_tangent_span_mm > self.tangent_window_mm:
            raise ValueError("Minimum tangent span exceeds the tangent window")
        if self.window_half_width_mm > self.max_window_half_width_mm:
            raise ValueError("Initial plane window exceeds its maximum")
        if self.plane_spacing_mm > self.window_half_width_mm:
            raise ValueError("Plane spacing exceeds the initial window half width")


def _polyline(points_mm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points_mm, dtype=float)
    if (points.ndim != 2 or points.shape[1] != 3 or len(points) < 2
            or not np.isfinite(points).all()):
        raise ValueError("A segment requires at least two finite world-mm points")
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    if np.any(lengths <= 0):
        raise ValueError("Consecutive segment points must be distinct")
    return points, np.r_[0., np.cumsum(lengths)]


def _interpolate(points: np.ndarray, arc: np.ndarray, positions: np.ndarray) -> np.ndarray:
    return np.column_stack([np.interp(positions, arc, points[:, axis]) for axis in range(3)])


def _interval_samples(start: float, end: float, step: float) -> np.ndarray:
    values = np.arange(start, end, step, dtype=float)
    values = values[values < end - 1e-10]
    return np.r_[values, end]


def estimate_tangent(points_mm: np.ndarray, s_mm: float, config: Config = Config()) -> dict:
    """Fit PCA to uniformly spaced support within this segment alone."""
    config.validate()
    points, arc = _polyline(points_mm)
    if not np.isfinite(s_mm) or s_mm < 0 or s_mm > arc[-1] + 1e-9:
        raise ValueError("Tangent sample arc position lies outside its segment")
    start = max(0., float(s_mm) - config.tangent_window_mm / 2)
    end = min(float(arc[-1]), float(s_mm) + config.tangent_window_mm / 2)
    result = {"fit_start_s_mm": start, "fit_end_s_mm": end, "linearity": None,
              "tangent": None, "status": "unavailable", "reasons": []}
    positions = _interval_samples(start, end, config.tangent_support_step_mm)
    if end - start < config.min_tangent_span_mm - 1e-9 or len(positions) < 3:
        result["reasons"] = ["insufficient_tangent_support"]
        return result
    support = _interpolate(points, arc, positions)
    if len(np.unique(support, axis=0)) < 3:
        result["reasons"] = ["insufficient_tangent_support"]
        return result
    centered = support - support.mean(axis=0)
    eigenvalues, eigenvectors = np.linalg.eigh(centered.T @ centered)
    largest = float(eigenvalues[-1])
    if largest <= 1e-12:
        result["reasons"] = ["degenerate_tangent"]
        return result
    linearity = float(np.clip(largest / max(largest, float(eigenvalues.sum())), 0., 1.))
    result["linearity"] = linearity
    if linearity < config.min_linearity:
        result["reasons"] = ["low_linearity"]
        return result
    tangent = eigenvectors[:, -1]
    forward = support[-1] - support[0]
    if np.linalg.norm(forward) <= 1e-12 or abs(float(tangent @ forward)) <= 1e-12:
        result["reasons"] = ["degenerate_tangent"]
        return result
    if tangent @ forward < 0:
        tangent = -tangent
    result.update(tangent=tangent.tolist(), status="ok")
    return result


def _unit_vector(value: np.ndarray, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=float)
    if vector.shape != (3,) or not np.isfinite(vector).all() or np.linalg.norm(vector) <= 1e-12:
        raise ValueError(f"{name} must be a finite nonzero three-vector")
    return vector / np.linalg.norm(vector)


def orthogonal_basis(tangent: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return a deterministic right-handed orthonormal basis of the plane."""
    normal = _unit_vector(tangent, "Tangent")
    axis = np.eye(3)[int(np.argmin(np.abs(normal)))]
    first = _unit_vector(np.cross(normal, axis), "First plane basis")
    return first, np.cross(normal, first)


def _geometry(mask: np.ndarray, affine_mm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(mask)
    if values.ndim != 3 or any(size < 1 for size in values.shape):
        raise ValueError("A nonempty three-dimensional mask grid is required")
    if values.dtype != np.bool_:
        unique = np.unique(values)
        if not (np.isin(unique, [0, 1]).all() or np.isin(unique, [0, 255]).all()):
            raise ValueError("Mask must contain binary values")
        values = values != 0
    affine = np.asarray(affine_mm, dtype=float)
    if (affine.shape != (4, 4) or not np.isfinite(affine).all()
            or not np.allclose(affine[3], [0, 0, 0, 1], rtol=0, atol=1e-12)
            or abs(np.linalg.det(affine[:3, :3])) <= 1e-12):
        raise ValueError("A finite nonsingular homogeneous affine in mm is required")
    directions = affine[:3, :3] / np.linalg.norm(affine[:3, :3], axis=0)
    if not np.allclose(directions.T @ directions, np.eye(3), rtol=1e-5, atol=1e-5):
        raise ValueError("Sheared image grids are unsupported")
    return values, affine


def _null_measurements() -> dict:
    return {"pixel_count": None, "area_mm2": None, "area_equiv_radius_mm": None,
            "area_equiv_diameter_mm": None}


def _sample_plane(mask: np.ndarray, inverse: np.ndarray, center: np.ndarray,
                  tangent: np.ndarray, config: Config, first: np.ndarray,
                  second: np.ndarray) -> dict:
    spacing = config.plane_spacing_mm
    # Integer half-grid counts ensure the center is an actual pixel. Flooring
    # enforces the configured physical cap even for nonintegral half widths.
    max_half = int(np.floor(config.max_window_half_width_mm / spacing + 1e-10))
    half = min(max_half, int(np.ceil(config.window_half_width_mm / spacing - 1e-10)))
    while True:
        axis = np.arange(-half, half + 1, dtype=float) * spacing
        xx, yy = np.meshgrid(axis, axis, indexing="xy")
        world = center[:, None, None] + first[:, None, None] * xx + second[:, None, None] * yy
        coordinates = np.einsum("ij,jkl->ikl", inverse[:3, :3], world) + inverse[:3, 3, None, None]
        upper = np.asarray(mask.shape, dtype=float)[:, None, None] - 1
        valid = np.all((coordinates >= -1e-9) & (coordinates <= upper + 1e-9), axis=0)
        # Remove floating-point noise at exact image-grid boundaries only.
        coordinates = np.where((coordinates < 0) & (coordinates >= -1e-9), 0., coordinates)
        coordinates = np.where((coordinates > upper) & (coordinates <= upper + 1e-9), upper, coordinates)
        foreground = ndimage.map_coordinates(mask, coordinates, order=0, mode="constant",
                                              cval=0, prefilter=False).astype(bool)
        foreground &= valid
        selected = np.zeros(foreground.shape, dtype=bool)
        result = {**_null_measurements(), "status": "unavailable", "reasons": [],
                  "plane_spacing_mm": float(spacing), "window_half_width_mm": float(half * spacing),
                  "foreground": foreground, "selected": selected, "valid": valid, "axis_mm": axis,
                  "tangent": tangent.copy(), "e1": first.copy(), "e2": second.copy()}
        if not foreground[half, half]:
            result["reasons"] = ["center_background"]
            return result
        labels, _ = ndimage.label(foreground, ndimage.generate_binary_structure(2, 1))
        selected = labels == labels[half, half]
        result["selected"] = selected
        border = bool(selected[0].any() or selected[-1].any()
                      or selected[:, 0].any() or selected[:, -1].any())
        collar = ndimage.binary_dilation(selected, structure=np.ones((3, 3), dtype=bool))
        fov = bool(np.any(collar & ~valid))
        if border and half < max_half:
            half = min(max_half, half * 2)
            continue
        if border or fov:
            result["reasons"] = (["window_truncated"] if border else []) + (["fov_truncated"] if fov else [])
            return result
        count = int(selected.sum())
        area = count * spacing * spacing
        radius = sqrt(area / pi)
        result.update(pixel_count=count, area_mm2=area, area_equiv_radius_mm=radius,
                      area_equiv_diameter_mm=2 * radius, status="ok")
        return result


def _prepare_tree(tree: dict, mask: np.ndarray, affine: np.ndarray) -> tuple[dict, list[dict]]:
    saved = np.asarray(tree.get("affine_mm"), dtype=float)
    if saved.shape != (4, 4) or not np.allclose(saved, affine, rtol=0, atol=1e-9):
        raise ValueError("Tree affine differs from the mask affine in mm")
    if "shape" in tree and list(tree["shape"]) != list(mask.shape):
        raise ValueError("Tree shape differs from the mask")
    voxels = {}
    occupied = set()
    for row in tree.get("voxels", []):
        identity = row.get("voxel_id")
        point = np.asarray(row.get("ijk"))
        if (isinstance(identity, bool) or not isinstance(identity, int) or identity in voxels
                or point.shape != (3,) or not np.issubdtype(point.dtype, np.integer)
                or np.any(point < 0) or np.any(point >= mask.shape) or not mask[tuple(point)]):
            raise ValueError("Invalid, duplicate, or out-of-mask tree voxel")
        if tuple(point) in occupied:
            raise ValueError("Repeated tree voxel coordinate")
        occupied.add(tuple(point))
        world = affine[:3, :3] @ point + affine[:3, 3]
        if not np.allclose(row.get("world_mm"), world, rtol=0, atol=1e-8):
            raise ValueError("Tree world coordinates disagree with its affine")
        if not np.isfinite(row.get("edt_mm", np.nan)) or row["edt_mm"] <= 0:
            raise ValueError("Tree EDT samples must be finite and positive")
        voxels[identity] = row
    if not voxels:
        raise ValueError("Tree must contain voxels")
    branches, identities, pairs = [], set(), set()
    for branch in tree.get("branches", []):
        identity = branch.get("branch_id")
        chain = branch.get("voxel_ids")
        if (isinstance(identity, bool) or not isinstance(identity, int) or identity < 1
                or identity in identities or not isinstance(chain, list) or len(chain) < 2
                or len(set(chain)) != len(chain) or any(v not in voxels for v in chain)):
            raise ValueError("Invalid segment identity or ordered voxel chain")
        identities.add(identity)
        if branch.get("start_node_id") != chain[0] or branch.get("end_node_id") != chain[-1]:
            raise ValueError("Segment endpoints disagree with its ordered chain")
        coordinates = np.asarray([voxels[v]["ijk"] for v in chain], dtype=int)
        if "points_ijk" in branch and branch["points_ijk"] != coordinates.tolist():
            raise ValueError("Segment coordinates disagree with its voxel IDs")
        points, arc = _polyline(coordinates @ affine[:3, :3].T + affine[:3, 3])
        if not np.isclose(branch.get("length_mm", np.nan), arc[-1], rtol=1e-9, atol=1e-8):
            raise ValueError("Segment length differs from its physical path")
        if branch.get("edge_count", len(chain) - 1) != len(chain) - 1:
            raise ValueError("Segment edge count differs from its path")
        for first, second in zip(chain, chain[1:]):
            pair = first, second
            if (pair in pairs or voxels[second].get("parent_id") != first
                    or second not in voxels[first].get("child_ids", [])
                    or np.max(np.abs(np.asarray(voxels[second]["ijk"]) - voxels[first]["ijk"])) > 1):
                raise ValueError("Segment changes directed tree edges")
            pairs.add(pair)
        radii = np.asarray([voxels[v]["edt_mm"] for v in chain], dtype=float)
        representative = fsum((np.diff(arc) * (radii[:-1] + radii[1:]) / 2).tolist()) / fsum(np.diff(arc).tolist())
        branches.append({"record": branch, "points": points, "arc": arc,
                         "edt_segment_radius_mm": float(representative)})
    if not branches:
        raise ValueError("Tree must contain segments")
    expected_pairs = {(row["parent_id"], row["child_id"]) for row in tree.get("edges", [])}
    if len(expected_pairs) != len(tree.get("edges", [])) or pairs != expected_pairs:
        raise ValueError("Segments do not partition the unique tree edges")
    return voxels, sorted(branches, key=lambda row: row["record"]["branch_id"])


class _CrossingIndex:
    def __init__(self, branches: list[dict]):
        self.first = np.concatenate([row["points"][:-1] for row in branches])
        self.last = np.concatenate([row["points"][1:] for row in branches])
        self.branch_ids = np.concatenate([np.full(len(row["arc"]) - 1, row["record"]["branch_id"], dtype=int)
                                          for row in branches])
        self.start_s = np.concatenate([row["arc"][:-1] for row in branches])
        self.end_s = np.concatenate([row["arc"][1:] for row in branches])
        self.half_lengths = np.linalg.norm(self.last - self.first, axis=1) / 2
        self.max_half_length = float(self.half_lengths.max())
        self.index = cKDTree((self.first + self.last) / 2)

    def crossings(self, sample: dict, plane: dict) -> tuple[list[int], bool]:
        rows, cols = np.nonzero(plane["selected"])
        axis = plane["axis_mm"]
        radius = float(np.hypot(axis[cols], axis[rows]).max()) + sample["plane_spacing_mm"]
        center = np.asarray(sample["center_mm"])
        normal, e1, e2 = (np.asarray(sample[key]) for key in ("tangent", "e1", "e2"))
        candidates = self.index.query_ball_point(center, radius + self.max_half_length)
        other, nonlocal_same = set(), False
        spacing = sample["plane_spacing_mm"]
        half = len(axis) // 2
        for index in candidates:
            same = self.branch_ids[index] == sample["branch_id"]
            low, high = sample["fit_start_s_mm"], sample["fit_end_s_mm"]
            if same and self.start_s[index] >= low - 1e-9 and self.end_s[index] <= high + 1e-9:
                continue
            first, last = self.first[index], self.last[index]
            da, db = float((first - center) @ normal), float((last - center) @ normal)
            if abs(da) <= 1e-8 and abs(db) <= 1e-8:
                count = max(1, int(np.ceil(2 * self.half_lengths[index] / (spacing / 2))))
                weights = np.linspace(0., 1., count + 1)
            elif da * db > 0 and abs(da) > 1e-8 and abs(db) > 1e-8:
                continue
            else:
                if abs(da - db) <= 1e-12:
                    continue
                weight = da / (da - db)
                if weight < -1e-8 or weight > 1 + 1e-8:
                    continue
                weights = np.array([np.clip(weight, 0., 1.)])
            if same:
                arc = self.start_s[index] + weights * (self.end_s[index] - self.start_s[index])
                weights = weights[(arc < low - 1e-9) | (arc > high + 1e-9)]
            if not len(weights):
                continue
            offsets = first + weights[:, None] * (last - first) - center
            columns = np.floor(offsets @ e1 / spacing + half + .5).astype(int)
            lines = np.floor(offsets @ e2 / spacing + half + .5).astype(int)
            inside = (columns >= 0) & (columns < len(axis)) & (lines >= 0) & (lines < len(axis))
            if np.any(plane["selected"][lines[inside], columns[inside]]):
                if same:
                    nonlocal_same = True
                else:
                    other.add(int(self.branch_ids[index]))
        return sorted(other), nonlocal_same


def measure_tree_cross_sections(tree: dict, mask: np.ndarray, affine_mm: np.ndarray,
                                config: Config = Config(), progress: Callable | None = None) -> dict:
    """Measure each saved segment independently without moving its centerline."""
    config.validate()
    values, affine = _geometry(mask, affine_mm)
    inverse = np.linalg.inv(affine)
    voxels, branches = _prepare_tree(tree, values, affine)
    crossing_index = _CrossingIndex(branches)
    samples, segments = [], []
    status_counts = Counter({"ok": 0, "needs_review": 0, "unavailable": 0})
    reason_counts = Counter()
    for completed, source in enumerate(branches, start=1):
        branch, points, arc = source["record"], source["points"], source["arc"]
        identity, length = branch["branch_id"], float(arc[-1])
        short = length < config.min_tangent_span_mm - 1e-9
        positions = np.array([length / 2]) if short else _interval_samples(0., length, config.step_mm)
        centers = _interpolate(points, arc, positions)
        branch_samples = []
        for sample_index, (position, center) in enumerate(zip(positions, centers)):
            bracket = min(int(np.searchsorted(arc, position, side="right")) - 1, len(arc) - 2)
            fraction = float((position - arc[bracket]) / (arc[bracket + 1] - arc[bracket]))
            fit = estimate_tangent(points, float(position), config)
            sample = {"sample_id": f"B{identity:06d}_S{sample_index:06d}", "branch_id": identity,
                      "sample_index": sample_index, "s_mm": float(position), "center_mm": center.tolist(),
                      "ijk": (inverse[:3, :3] @ center + inverse[:3, 3]).tolist(),
                      "bracket_voxel_ids": branch["voxel_ids"][bracket:bracket + 2],
                      "bracket_fraction": fraction, "e1": None, "e2": None,
                      "plane_spacing_mm": config.plane_spacing_mm,
                      "window_half_width_mm": config.window_half_width_mm,
                      "crossing_branch_ids": [], **_null_measurements(), **fit}
            if short:
                sample.update(status="unavailable", reasons=["short_segment"], tangent=None)
            elif fit["status"] == "ok":
                normal = np.asarray(fit["tangent"])
                first, second = orthogonal_basis(normal)
                plane = _sample_plane(values, inverse, center, normal, config, first, second)
                sample.update({key: plane[key] for key in (*_null_measurements(), "status", "reasons",
                                                          "plane_spacing_mm", "window_half_width_mm")})
                sample.update(e1=first.tolist(), e2=second.tolist())
                if sample["status"] == "ok":
                    other, same = crossing_index.crossings(sample, plane)
                    sample["crossing_branch_ids"] = other
                    if other:
                        sample["reasons"].append("other_branch_crossing")
                    if same:
                        sample["reasons"].append("nonlocal_same_branch_crossing")
                    for vertex, distance in ((branch["voxel_ids"][0], position),
                                             (branch["voxel_ids"][-1], length - position)):
                        node = voxels[vertex]
                        junction = node.get("node_kind") == "junction" or len(node.get("child_ids", [])) > 1
                        endpoint = node.get("node_kind") in {"root", "leaf"} and not junction
                        if junction and distance <= config.junction_margin_mm + 1e-9:
                            sample["reasons"].append("near_junction")
                        if endpoint and distance <= config.endpoint_margin_mm + 1e-9:
                            sample["reasons"].append("near_endpoint")
                    if sample["reasons"]:
                        sample["status"] = "needs_review"
            sample["reasons"] = sorted(set(sample["reasons"]))
            status_counts[sample["status"]] += 1
            reason_counts.update(sample["reasons"])
            branch_samples.append(sample)
        span = fsum(last["s_mm"] - first["s_mm"] for first, last in zip(branch_samples, branch_samples[1:])
                    if first["status"] == last["status"] == "ok")
        counts = Counter(row["status"] for row in branch_samples)
        segments.append({"branch_id": identity, "length_mm": float(branch["length_mm"]),
                         "edt_segment_radius_mm": source["edt_segment_radius_mm"],
                         "sample_count": len(branch_samples), "ok_count": counts["ok"],
                         "needs_review_count": counts["needs_review"], "unavailable_count": counts["unavailable"],
                         "ok_span_mm": span, "ok_span_fraction": span / length,
                         "inherited_tree_flag_ids": [row["flag_id"] for row in tree.get("flags", [])
                                                     if identity in row.get("branch_ids", [])]})
        samples.extend(branch_samples)
        if progress is not None and (completed % 50 == 0 or completed == len(branches)):
            progress({"completed_segments": completed, "total_segments": len(branches),
                      "sample_count": len(samples)})
    return {"samples": samples, "segments": segments, "parameters": asdict(config),
            "summary": {"segment_count": len(segments), "sample_count": len(samples),
                        "status_counts": dict(status_counts), "reason_counts": dict(sorted(reason_counts.items())),
                        "total_length_mm": fsum(row["length_mm"] for row in segments),
                        "ok_span_mm": fsum(row["ok_span_mm"] for row in segments)}}
