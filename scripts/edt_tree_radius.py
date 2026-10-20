"""Measure local EDT radii on an existing, unchanged centerline tree.

The distance map is already calibrated in millimetres. This module neither
recomputes EDT nor changes connectivity, target identities, or the saved tree.
"""
from __future__ import annotations

from collections import defaultdict
from math import fsum
from typing import Any

import numpy as np

from run_edt_experiment import binary_mask, inside, parse_point


def _identity(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def _records(tree: dict, name: str, identity: str) -> dict[int, dict]:
    records = tree.get(name)
    if not isinstance(records, list) or not records:
        raise ValueError(f"A nonempty {name} table is required")
    result = {}
    for record in records:
        if not isinstance(record, dict):
            raise ValueError(f"Every {name} record must be an object")
        key = _identity(record.get(identity), identity)
        if key in result:
            raise ValueError(f"Duplicate {identity}")
        result[key] = record
    return result


def _length(value: Any, expected: float, name: str) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a positive physical length") from None
    if (not np.isfinite(value) or value <= 0
            or not np.isclose(value, expected, rtol=1e-9, atol=1e-8)):
        raise ValueError(f"{name} differs from its physical edge lengths")
    return value


def measure_tree_radius(tree: dict, edt_data: np.ndarray, mask_data: np.ndarray,
                        affine_mm: np.ndarray) -> dict[str, Any]:
    """Return unique point samples and length-weighted segment/tree summaries.

    ``r_edt_mm`` is an exact alias for the saved EDT sample at each integer tree
    voxel. For every unique directed edge, the radius integral is its physical
    length times the mean of its endpoint radii (the trapezoidal rule). Shared
    trunks are counted once; shared junctions remain endpoints of all incident
    segments. Existing unweighted ``edt_mean_mm`` values are never overwritten.

    Inputs must describe the same physical grid and a coherent rooted tree whose
    maximal segments partition its edges. Corrupt cached samples, geometry, or
    connectivity raise ``ValueError``. The caller verifies file hashes and NIfTI
    grid metadata before passing the arrays. No input is mutated.
    """
    mask = binary_mask(mask_data)
    edt = np.asarray(edt_data)
    affine = np.asarray(affine_mm, dtype=float)
    if edt.shape != mask.shape:
        raise ValueError("EDT and mask shapes differ")
    if (affine.shape != (4, 4) or not np.isfinite(affine).all()
            or not np.allclose(affine[3], [0, 0, 0, 1], rtol=0, atol=1e-12)
            or abs(np.linalg.det(affine[:3, :3])) < 1e-12):
        raise ValueError("A finite nonsingular homogeneous affine in mm is required")
    saved_affine = np.asarray(tree.get("affine_mm"), dtype=float)
    if (saved_affine.shape != (4, 4)
            or not np.allclose(saved_affine, affine, rtol=0, atol=1e-9)):
        raise ValueError("Tree affine differs from the supplied affine in mm")
    if "shape" in tree and tuple(tree["shape"]) != mask.shape:
        raise ValueError("Tree shape differs from the supplied image grid")

    voxels = _records(tree, "voxels", "voxel_id")
    edges = _records(tree, "edges", "edge_id")
    branches = _records(tree, "branches", "branch_id")
    root = _identity(tree.get("root_voxel_id"), "root_voxel_id")
    if root not in voxels:
        raise ValueError("Root voxel is missing")
    coordinates, worlds, radii = {}, {}, {}
    occupied = set()
    for identity in sorted(voxels):
        voxel = voxels[identity]
        point = parse_point(voxel.get("ijk"))
        if point in occupied:
            raise ValueError("Duplicate centerline voxel coordinates")
        occupied.add(point)
        if not inside(point, mask.shape) or not mask[point]:
            raise ValueError("A centerline point is outside the foreground mask")
        radius = float(edt[point])
        if not np.isfinite(radius) or radius <= 0:
            raise ValueError("Centerline EDT samples must be finite and positive")
        if voxel.get("edt_mm") != radius:
            raise ValueError("Cached voxel EDT sample differs from the supplied EDT")
        world = affine[:3, :3] @ point + affine[:3, 3]
        saved_world = np.asarray(voxel.get("world_mm"), dtype=float)
        if (saved_world.shape != (3,)
                or not np.allclose(saved_world, world, rtol=0, atol=1e-8)):
            raise ValueError("World coordinates differ from the supplied affine")
        coordinates[identity], worlds[identity], radii[identity] = point, world, radius
    if parse_point(tree.get("root_ijk")) != coordinates[root]:
        raise ValueError("Root coordinates differ from its voxel record")

    pairs, parents, children = {}, {}, defaultdict(list)
    for identity in sorted(edges):
        edge = edges[identity]
        first = _identity(edge.get("parent_id"), "edge parent_id")
        second = _identity(edge.get("child_id"), "edge child_id")
        branch = _identity(edge.get("branch_id"), "edge branch_id")
        if first not in voxels or second not in voxels or branch not in branches:
            raise ValueError("Edge references an unknown voxel or segment")
        if first == second or (first, second) in pairs:
            raise ValueError("Repeated edge or self-edge")
        if second == root or second in parents:
            raise ValueError("Tree has a repeated parent or an incoming root edge")
        delta = np.asarray(coordinates[second]) - coordinates[first]
        if np.any(np.abs(delta) > 1):
            raise ValueError("Edge is not a saved 26-neighbour step")
        _length(edge.get("length_mm"), float(np.linalg.norm(affine[:3, :3] @ delta)),
                "Edge length")
        pairs[first, second] = edge
        parents[second] = first
        children[first].append(second)
    if len(edges) != len(voxels) - 1 or set(parents) != set(voxels) - {root}:
        raise ValueError("Tree must have one parent per nonroot voxel and E = V - 1")
    visited, stack = set(), [root]
    while stack:
        current = stack.pop()
        if current in visited:
            raise ValueError("Tree contains a cycle")
        visited.add(current)
        stack.extend(children[current])
    if visited != set(voxels):
        raise ValueError("Every voxel must be reachable from the root")

    kinds = {}
    for identity, voxel in voxels.items():
        kind = ("root" if identity == root else "leaf" if not children[identity]
                else "junction" if len(children[identity]) > 1 else "chain")
        kinds[identity] = kind
        saved_parent = voxel.get("parent_id")
        if saved_parent is not None:
            saved_parent = _identity(saved_parent, "voxel parent_id")
        saved_children = voxel.get("child_ids")
        if not isinstance(saved_children, list):
            raise ValueError("Voxel child_ids must be a list")
        saved_children = [_identity(child, "voxel child_id") for child in saved_children]
        if (saved_parent != parents.get(identity)
                or sorted(saved_children) != sorted(children[identity])
                or voxel.get("node_kind") != kind):
            raise ValueError("Voxel topology metadata differs from the edge graph")

    covered, incident, segments = set(), defaultdict(set), []
    for identity in sorted(branches):
        branch = branches[identity]
        raw_chain = branch.get("voxel_ids")
        if not isinstance(raw_chain, list) or len(raw_chain) < 2:
            raise ValueError("A segment requires at least two voxels")
        chain = [_identity(vertex, "segment voxel_id") for vertex in raw_chain]
        if (len(set(chain)) != len(chain) or any(v not in voxels for v in chain)):
            raise ValueError("Segment contains repeated or unknown voxels")
        if (branch.get("start_node_id") != chain[0]
                or branch.get("end_node_id") != chain[-1]
                or kinds[chain[0]] == "chain" or kinds[chain[-1]] == "chain"
                or any(kinds[v] != "chain" for v in chain[1:-1])):
            raise ValueError("Segments must be maximal chains between graph nodes")
        if branch.get("edge_count") != len(chain) - 1:
            raise ValueError("Segment edge count differs from its voxel chain")
        if "points_ijk" in branch and branch["points_ijk"] != [list(coordinates[v]) for v in chain]:
            raise ValueError("Segment coordinates differ from its voxel chain")
        lengths, integrals = [], []
        for first, second in zip(chain, chain[1:]):
            pair = first, second
            edge = pairs.get(pair)
            if pair in covered:
                raise ValueError("Segments overlap a unique edge")
            if edge is None or edge["branch_id"] != identity:
                raise ValueError("Segment chain differs from its assigned edges")
            covered.add(pair)
            length = float(edge["length_mm"])
            lengths.append(length)
            integrals.append(length * (radii[first] + radii[second]) / 2)
        length = _length(branch.get("length_mm"), fsum(lengths), "Segment length")
        for vertex in chain:
            incident[vertex].add(identity)
        values = [radii[vertex] for vertex in chain]
        mean_radius = fsum(integrals) / fsum(lengths)
        segments.append({"branch_id": identity, "start_node_id": chain[0],
                         "end_node_id": chain[-1], "point_count": len(chain),
                         "edge_count": len(chain) - 1, "length_mm": length,
                         "segment_radius_mm": mean_radius,
                         "r_edt_min_mm": min(values), "r_edt_max_mm": max(values),
                         "r_edt_length_weighted_mean_mm": mean_radius})
    if covered != set(pairs):
        raise ValueError("Segments do not partition all unique edges")

    ordered_edges = [edges[identity] for identity in sorted(edges)]
    unique_length = fsum(float(edge["length_mm"]) for edge in ordered_edges)
    old_summary = tree.get("summary", {})
    for name, count in (("voxel_count", len(voxels)), ("edge_count", len(edges)),
                        ("segment_count", len(branches))):
        if name in old_summary and old_summary[name] != count:
            raise ValueError(f"Tree summary {name} differs from its graph")
    if "unique_length_mm" in old_summary:
        unique_length = _length(old_summary["unique_length_mm"], unique_length,
                                "Tree unique length")
    points = []
    for identity in sorted(voxels):
        i, j, k = coordinates[identity]
        x, y, z = worlds[identity]
        points.append({"voxel_id": identity, "i": i, "j": j, "k": k,
                       "x_mm": float(x), "y_mm": float(y), "z_mm": float(z),
                       "node_kind": kinds[identity],
                       "incident_branch_ids": sorted(incident[identity]),
                       "r_edt_mm": radii[identity]})
    integral = fsum(float(edge["length_mm"]) *
                    (radii[edge["parent_id"]] + radii[edge["child_id"]]) / 2
                    for edge in ordered_edges)
    return {"points": points, "segments": segments,
            "summary": {"point_count": len(points), "edge_count": len(edges),
                        "segment_count": len(segments), "unique_length_mm": unique_length,
                        "r_edt_min_mm": min(radii.values()),
                        "r_edt_max_mm": max(radii.values()),
                        "r_edt_length_weighted_mean_mm": integral / fsum(
                            float(edge["length_mm"]) for edge in ordered_edges)},
            "definitions": {
                "r_edt_mm": "Saved EDT at the centerline voxel, in mm; distance to the nearest background voxel centre, not a validated cross-sectional radius",
                "segment_radius_mm": "Representative radius of this graph segment in mm; exactly its physical-length-weighted mean EDT radius (r_edt_length_weighted_mean_mm), not an independently validated anatomical radius",
                "r_edt_min_mm": "Minimum EDT radius among the unique tree or segment voxels",
                "r_edt_max_mm": "Maximum EDT radius among the unique tree or segment voxels",
                "r_edt_length_weighted_mean_mm": "Sum over unique edges of length_mm * (endpoint_radius_0 + endpoint_radius_1) / 2, divided by total edge length_mm",
                "segment": "Existing maximal graph chain between root, junctions and leaves; not an anatomical branch count",
                "point_count": "Unique voxels for the tree; segment endpoints may be shared by incident segments",
                "incident_branch_ids": "IDs of all existing graph segments containing this voxel",
                "reference_radius": "EDT derived from the reference mask; not a manual radius annotation",
                "legacy_edt_mean_mm": "The source tree's unweighted segment EDT mean remains unchanged; it is distinct from the length-weighted radius mean"}}
