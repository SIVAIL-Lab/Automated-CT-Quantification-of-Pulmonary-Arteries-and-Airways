"""Losslessly deduplicate saved EDT paths into an inspectable directed tree.

Only consecutive saved path edges define connectivity. Spatial contacts and
crossings are review flags, never additional edges or automatic repairs.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
from itertools import combinations, permutations, product
from typing import Any

import numpy as np
from scipy import ndimage as ndi

from run_edt_experiment import binary_mask, inside, mask_bbox, parse_point


FLAG_KINDS = (
    "six_component_crossing", "no_local_face_route", "short_segment",
    "high_fanout", "non_tree_contact", "geometric_crossing",
)


def local_face_route(first: tuple[int, int, int], second: tuple[int, int, int],
                     mask: np.ndarray) -> bool:
    """Whether the step has a foreground face route in its minimal voxel block."""
    delta = np.asarray(second) - first
    axes = np.flatnonzero(delta)
    for order in permutations(axes):
        current = list(first)
        for axis in order:
            current[axis] += int(delta[axis])
            if not mask[tuple(current)]:
                break
        else:
            return True
    return False


def _integer_id(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{label} must be an integer")
    return int(value)


def _validate_paths(trace: dict, mask: np.ndarray, edt: np.ndarray,
                    affine: np.ndarray) -> tuple[tuple, dict, dict]:
    if trace.get("connectivity") != 26:
        raise ValueError("Tree construction requires the saved 26-neighbour run")
    if trace.get("technical_validation") != "passed":
        raise ValueError("The source path run has not passed technical validation")
    root = parse_point(trace.get("root_ijk"))
    if not inside(root, mask.shape) or not mask[root]:
        raise ValueError("Root is outside the foreground mask")
    targets = trace.get("targets")
    paths = trace.get("paths")
    if not isinstance(targets, list) or not targets or not isinstance(paths, list):
        raise ValueError("Saved target records and paths are required")
    if any(trace.get(key) != len(targets) for key in
           ("target_count", "successful_target_count", "reachable_target_count")):
        raise ValueError("Saved target counts do not describe complete successful paths")
    records, target_voxels = {}, set()
    for target in targets:
        identity = _integer_id(target.get("target_id"), "target_id")
        if identity in records:
            raise ValueError("Duplicate target identity")
        point = parse_point(target.get("target_ijk"))
        if point == root or point in target_voxels:
            raise ValueError("Duplicate target voxel or target equals root")
        target_voxels.add(point)
        if target.get("status") != "ok" or target.get("audit_passed") is not True:
            raise ValueError("Every target must have successful status and path audit")
        if target.get("audit") and not all(v is True for v in target["audit"].values()):
            raise ValueError("A saved target audit contains a failed check")
        records[identity] = target
    saved_paths = {}
    for path in paths:
        identity = _integer_id(path.get("target_id"), "path target_id")
        if identity in saved_paths or identity not in records:
            raise ValueError("Duplicate or unknown path target identity")
        raw = path.get("points_ijk")
        if not isinstance(raw, list) or len(raw) < 2:
            raise ValueError("Path must contain at least two integer points")
        points = [parse_point(point) for point in raw]
        if points[0] != root or points[-1] != parse_point(records[identity]["target_ijk"]):
            raise ValueError("Path endpoints differ from saved root/target seeds")
        if len(set(points)) != len(points):
            raise ValueError("Path contains a cycle or repeated voxel")
        if any(not inside(point, mask.shape) or not mask[point] for point in points):
            raise ValueError("Path contains a voxel outside the foreground mask")
        array = np.asarray(points, dtype=int)
        delta = np.diff(array, axis=0)
        if np.any(np.abs(delta) > 1) or np.any(np.count_nonzero(delta, axis=1) == 0):
            raise ValueError("Path has invalid 26-neighbour adjacency")
        sampled = edt[tuple(array.T)]
        saved = np.asarray(path.get("edt_mm", []), dtype=float)
        if (saved.shape != sampled.shape or not np.isfinite(saved).all()
                or not np.array_equal(saved, sampled.astype(float))):
            raise ValueError("Saved path EDT samples differ from the supplied EDT")
        length = float(np.linalg.norm(delta @ affine[:3, :3].T, axis=1).sum())
        target = records[identity]
        if target.get("point_count") != len(points):
            raise ValueError("Saved point count differs from the path")
        if not np.isclose(target.get("path_length_mm", np.nan), length,
                          rtol=1e-9, atol=1e-8):
            raise ValueError("Saved path length differs from the physical polyline")
        saved_paths[identity] = points
    if set(saved_paths) != set(records):
        raise ValueError("Missing path for a saved target")
    return root, records, saved_paths


def build_tree(trace: dict, mask: np.ndarray, edt: np.ndarray,
               affine_mm: np.ndarray) -> dict[str, Any]:
    """Build a deterministic, JSON-serializable tree or raise on data corruption.

    ``mask`` is the selected component used by the source path run. The supplied
    distance map and affine are in mm. Geometry and all requested target IDs are
    preserved, including targets located inside a compressed segment.
    """
    mask = binary_mask(mask)
    edt = np.asarray(edt)
    affine = np.asarray(affine_mm, dtype=float)
    if (edt.shape != mask.shape or not np.isfinite(edt).all()
            or np.any(edt[mask] <= 0)):
        raise ValueError("EDT must match the mask and be finite/positive in foreground")
    if (affine.shape != (4, 4) or not np.isfinite(affine).all()
            or not np.allclose(affine[3], [0, 0, 0, 1])
            or abs(np.linalg.det(affine[:3, :3])) < 1e-12):
        raise ValueError("A finite nonsingular homogeneous affine in mm is required")
    root, target_records, path_points = _validate_paths(trace, mask, edt, affine)
    coordinates = sorted({point for path in path_points.values() for point in path})
    voxel_ids = {point: index for index, point in enumerate(coordinates)}
    root_id = voxel_ids[root]
    edge_pairs = sorted({(voxel_ids[first], voxel_ids[second])
                         for path in path_points.values()
                         for first, second in zip(path, path[1:])})
    parents, children = {}, defaultdict(list)
    for first, second in edge_pairs:
        if second == root_id:
            raise ValueError("Root has an incoming edge; cycle detected")
        if second in parents and parents[second] != first:
            raise ValueError("Conflicting parents: saved paths do not form a tree")
        parents[second] = first
        children[first].append(second)
    if len(edge_pairs) != len(coordinates) - 1:
        raise ValueError("Tree must satisfy E = V - 1")
    visited, stack = set(), [root_id]
    while stack:
        vertex = stack.pop()
        if vertex in visited:
            raise ValueError("Cycle detected in saved path edges")
        visited.add(vertex)
        stack.extend(children[vertex])
    if len(visited) != len(coordinates) or set(parents) != set(range(len(coordinates))) - {root_id}:
        raise ValueError("Not every vertex is reachable from the unique root")

    def kind(vertex: int) -> str:
        if vertex == root_id:
            return "root"
        if not children[vertex]:
            return "leaf"
        return "junction" if len(children[vertex]) > 1 else "chain"

    nodes = [{"node_id": vertex, "voxel_id": vertex, "ijk": list(point),
              "kind": kind(vertex), "child_count": len(children[vertex])}
             for vertex, point in enumerate(coordinates) if kind(vertex) != "chain"]
    branch_for_edge, locations, branches = {}, {}, []
    for node in nodes:
        start = node["node_id"]
        for first in sorted(children[start]):
            chain = [start, first]
            while kind(chain[-1]) == "chain":
                chain.append(children[chain[-1]][0])
            branch_id = len(branches) + 1
            points = np.asarray([coordinates[v] for v in chain], dtype=int)
            values = edt[tuple(points.T)].astype(float)
            length = float(np.linalg.norm(np.diff(points, axis=0) @ affine[:3, :3].T, axis=1).sum())
            branches.append({"branch_id": branch_id, "start_node_id": start,
                             "end_node_id": chain[-1], "voxel_ids": chain,
                             "points_ijk": points.tolist(), "length_mm": length,
                             "edge_count": len(chain) - 1,
                             "terminal": not bool(children[chain[-1]]),
                             "edt_min_mm": float(values.min()),
                             "edt_mean_mm": float(values.mean())})
            for first_id, second_id in zip(chain, chain[1:]):
                pair = (first_id, second_id)
                if pair in branch_for_edge:
                    raise ValueError("Compressed segments overlap an edge")
                branch_for_edge[pair] = branch_id
            for index, vertex in enumerate(chain[1:-1], start=1):
                locations[vertex] = (branch_id, index)
    if set(branch_for_edge) != set(edge_pairs):
        raise ValueError("Compressed graph expansion differs from saved path edges")

    bbox = mask_bbox(mask)
    origin = np.array([part.start for part in bbox], dtype=int)
    component_labels, six_count = ndi.label(mask[bbox], ndi.generate_binary_structure(3, 1))
    component_by_vertex = [int(component_labels[tuple(np.asarray(point) - origin)])
                           for point in coordinates]
    del component_labels
    voxels = [{"voxel_id": vertex, "ijk": list(point),
               "world_mm": (affine[:3, :3] @ point + affine[:3, 3]).tolist(),
               "parent_id": parents.get(vertex), "child_ids": sorted(children[vertex]),
               "edt_mm": float(edt[point]), "node_kind": kind(vertex)}
              for vertex, point in enumerate(coordinates)]
    edges = []
    for edge_id, (first, second) in enumerate(edge_pairs):
        p, q = coordinates[first], coordinates[second]
        delta = np.asarray(q) - p
        edges.append({"edge_id": edge_id, "parent_id": first, "child_id": second,
                      "branch_id": branch_for_edge[(first, second)],
                      "length_mm": float(np.linalg.norm(affine[:3, :3] @ delta)),
                      "diagonal_axes": int(np.count_nonzero(delta)),
                      "from_six_component": component_by_vertex[first],
                      "to_six_component": component_by_vertex[second],
                      "local_face_route": local_face_route(p, q, mask)})
    targets = []
    for identity in sorted(target_records):
        target = target_records[identity]
        path = [voxel_ids[point] for point in path_points[identity]]
        vertex = path[-1]
        recovered = [vertex]
        while recovered[-1] != root_id:
            recovered.append(parents[recovered[-1]])
        if recovered[::-1] != path:
            raise ValueError("Tree does not exactly reconstruct an original target path")
        path_branches = []
        for pair in zip(path, path[1:]):
            branch_id = branch_for_edge[pair]
            if not path_branches or path_branches[-1] != branch_id:
                path_branches.append(branch_id)
        expanded = [root_id]
        for branch_id in path_branches:
            branch_path = branches[branch_id - 1]["voxel_ids"]
            if branch_path[0] != expanded[-1]:
                raise ValueError("Compressed target path has a disconnected segment")
            if vertex in branch_path:
                branch_path = branch_path[:branch_path.index(vertex) + 1]
            expanded.extend(branch_path[1:])
        if expanded != path:
            raise ValueError("Expanded compressed graph does not recover target path")
        record = deepcopy(target)
        record["original_target"] = deepcopy(target)
        if "node_id" in record:
            record["historical_node_id"] = record.pop("node_id")
        record.update(voxel_id=vertex, path_branch_ids=path_branches)
        if vertex in locations:
            record.update(branch_id=locations[vertex][0], point_index=locations[vertex][1])
        else:
            record["node_id"] = vertex
        targets.append(record)

    flags = []

    def flag(name: str, vertices: list[int], edge_ids: list[int], branch_ids: list[int]) -> None:
        flags.append({"flag_id": f"F{len(flags) + 1:06d}", "kind": name,
                      "severity": "review", "voxel_ids": sorted(set(vertices)),
                      "edge_ids": sorted(set(edge_ids)), "branch_ids": sorted(set(branch_ids)),
                      "focus_ijk": np.mean([coordinates[v] for v in vertices], axis=0).tolist(),
                      "review_status": "pending_visual_review"})

    edge_id_for_pair = {(e["parent_id"], e["child_id"]): e["edge_id"] for e in edges}
    branches_by_vertex = defaultdict(set)
    for branch in branches:
        for vertex in branch["voxel_ids"]:
            branches_by_vertex[vertex].add(branch["branch_id"])
    for edge in edges:
        vertices = [edge["parent_id"], edge["child_id"]]
        args = (vertices, [edge["edge_id"]], [edge["branch_id"]])
        if edge["from_six_component"] != edge["to_six_component"]:
            flag("six_component_crossing", *args)
        if edge["diagonal_axes"] > 1 and not edge["local_face_route"]:
            flag("no_local_face_route", *args)
    for branch in branches:
        if branch["edge_count"] <= 2:
            pairs = zip(branch["voxel_ids"], branch["voxel_ids"][1:])
            flag("short_segment", branch["voxel_ids"], [edge_id_for_pair[pair] for pair in pairs],
                 [branch["branch_id"]])
    for vertex in range(len(coordinates)):
        if len(children[vertex]) >= 3:
            flag("high_fanout", [vertex], [edge_id_for_pair[(vertex, child)]
                                         for child in children[vertex]],
                 list(branches_by_vertex[vertex]))
    undirected_edges = {tuple(sorted(pair)) for pair in edge_pairs}
    offsets = [offset for offset in product((-1, 0, 1), repeat=3) if offset > (0, 0, 0)]
    for vertex, point in enumerate(coordinates):
        for offset in offsets:
            neighbour = tuple(p + d for p, d in zip(point, offset))
            other = voxel_ids.get(neighbour)
            if other is not None and (vertex, other) not in undirected_edges:
                flag("non_tree_contact", [vertex, other], [],
                     list(branches_by_vertex[vertex] | branches_by_vertex[other]))
    by_midpoint = defaultdict(list)
    for edge in edges:
        p, q = coordinates[edge["parent_id"]], coordinates[edge["child_id"]]
        by_midpoint[tuple(a + b for a, b in zip(p, q))].append(edge)
    for midpoint in sorted(by_midpoint):
        for first, second in combinations(by_midpoint[midpoint], 2):
            vertices = [first["parent_id"], first["child_id"], second["parent_id"], second["child_id"]]
            if len(set(vertices)) == 4:
                flag("geometric_crossing", vertices, [first["edge_id"], second["edge_id"]],
                     [first["branch_id"], second["branch_id"]])

    edge_length = sum(edge["length_mm"] for edge in edges)
    branch_length = sum(branch["length_mm"] for branch in branches)
    if not np.isclose(edge_length, branch_length, rtol=1e-12, atol=1e-10):
        raise ValueError("Unique edge lengths and segment lengths differ")
    flag_counts = Counter(item["kind"] for item in flags)
    summary = {"voxel_count": len(voxels), "edge_count": len(edges),
               "leaf_count": sum(node["kind"] == "leaf" for node in nodes),
               "junction_count": sum(node["kind"] == "junction" for node in nodes),
               "segment_count": len(branches), "target_count": len(targets),
               "internal_target_count": sum(bool(children[t["voxel_id"]]) for t in targets),
               "unique_length_mm": edge_length, "six_component_count": int(six_count),
               "flag_counts": {key: flag_counts[key] for key in FLAG_KINDS},
               "technical_validation": "passed", "anatomical_validation": "not_performed"}
    checks = {name: True for name in (
        "all_targets_successful", "path_identity_and_seed_endpoints", "foreground_contained",
        "edt_samples_match", "adjacency_valid", "single_root", "unique_parent",
        "all_vertices_reachable", "acyclic", "edge_count_equals_vertex_count_minus_one",
        "compression_lossless", "all_target_paths_reconstructed", "physical_length_consistent",
    )}
    return {"schema_version": "1.0", "root_voxel_id": root_id, "root_ijk": list(root),
            "affine_mm": affine.tolist(), "voxels": voxels, "nodes": nodes, "edges": edges,
            "branches": branches, "targets": targets, "flags": flags,
            "summary": summary, "checks": checks,
            "definitions": {"connectivity": "Consecutive directed edges in saved root-to-target paths only",
                            "segment": "Maximal chain between root, junctions and leaves; not an anatomical branch count",
                            "edt_mm": "Distance to background voxel centres; not a validated cross-sectional radius",
                            "edt_mean_mm": "Unweighted mean of saved EDT samples at segment voxels",
                            "internal_target": "Requested target that has children in the deduplicated tree",
                            "node_id": "Lexicographic zero-based voxel ID; original target node_id is historical_node_id",
                            "edge_id": "Zero-based order of directed (parent voxel ID, child voxel ID) pairs",
                            "branch_id": "One-based order of start voxel ID then first successor voxel ID",
                            "local_face_route": "Foreground face-neighbour route within the diagonal step's minimal voxel block",
                            "geometric_crossing": "Nonincident unit voxel edges sharing their doubled midpoint",
                            "review_flags": "Preserved for visual review; no pruning, smoothing, connections or repairs"}}
