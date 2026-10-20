"""Exploratory tracheal rooting and physical sampling of saved airway graphs.

Source voxel paths and junction clusters are immutable. For cyclic skeletons a
physical shortest-path tree provides provisional generations; the omitted cycle
edges remain in the output with unavailable generations and their original IDs.
This topology-derived labeling is not expert-confirmed anatomical generation.
"""
from __future__ import annotations

from collections import Counter, defaultdict, deque
import heapq
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage

from airway_graph_inputs import _binary, _branch_path, _read, _sha256, verify_saved_graph
from centerline_cross_sections import (
    Config as TangentConfig, _interpolate, _polyline, estimate_tangent, orthogonal_basis,
)


METHOD = "tracheal_inscribed_ball_spur_normalized_generations_v2"


def _bridge(voxels, start, end, affine):
    """Join edge attachment points inside the existing junction voxel cluster."""
    start, end = tuple(start), tuple(end)
    if start == end:
        return np.asarray([start])
    allowed = {tuple(point) for point in voxels}
    if start not in allowed or end not in allowed:
        raise ValueError("junction_attachment_outside_cluster")
    distance, previous, queue = {start: 0.}, {}, [(0., start)]
    offsets = [np.asarray(offset) - 1 for offset in np.ndindex(3, 3, 3)
               if offset != (1, 1, 1)]
    while queue:
        cost, point = heapq.heappop(queue)
        if cost != distance[point]:
            continue
        if point == end:
            path = [end]
            while path[-1] != start:
                path.append(previous[path[-1]])
            return np.asarray(path[::-1])
        for offset in offsets:
            other = tuple(np.asarray(point) + offset)
            candidate = cost + float(np.linalg.norm(affine[:3, :3] @ offset))
            if other in allowed and candidate < distance.get(other, np.inf):
                distance[other], previous[other] = candidate, point
                heapq.heappush(queue, (candidate, other))
    raise ValueError("disconnected_junction_cluster")


def _prepare_graph(graph, affine, radius_at, reviewed_spurs=()):
    components = []
    for component in graph["components"]:
        nodes = {node["node_id"]: np.asarray(node["voxels"], dtype=int)
                 for node in component["nodes"]}
        adjacency, edges = defaultdict(list), {}
        for source in component["branches"]:
            flags = []
            try:
                path = _branch_path(source, nodes)
                world = nib.affines.apply_affine(affine, path)
                length = float(np.linalg.norm(np.diff(world, axis=0), axis=1).sum())
                if length <= 0:
                    flags.append("zero_length_source_branch")
            except ValueError as error:
                path, world, length = np.empty((0, 3), int), np.empty((0, 3)), 0.
                flags = ["source_path_break", str(error)]
            edge = {**source, "path": path, "world": world,
                    "physical_length_mm": length, "flags": flags}
            edges[source["branch_id"]] = edge
            adjacency[source["v1"]].append((source["v2"], source["branch_id"]))
            adjacency[source["v2"]].append((source["v1"], source["branch_id"]))
        unseen, connected_count = set(nodes), 0
        while unseen:
            connected_count += 1
            queue = [min(unseen)]
            while queue:
                node = queue.pop()
                if node not in unseen:
                    continue
                unseen.remove(node)
                queue.extend(other for other, _ in adjacency[node] if other in unseen)
        prepared = {"component_id": component["component_id"], "nodes": nodes,
                    "edges": edges, "adjacency": adjacency,
                    "source_connected_part_count": connected_count,
                    "source_cycle_rank": len(edges) - len(nodes) + connected_count}
        prepared["intraluminal_terminal_spurs"] = _intraluminal_terminal_spurs(prepared, affine, radius_at)
        prepared["reviewed_terminal_spurs"] = []
        for review in reviewed_spurs:
            if review["component_id"] != component["component_id"]:
                continue
            edge = edges[review["branch_id"]]
            terminals = [node for node in (edge["v1"], edge["v2"]) if len(adjacency[node]) == 1]
            if len(terminals) != 1 or edge["flags"]:
                raise ValueError("Engineering fold review must identify a valid terminal source branch")
            terminal = terminals[0]
            junction = edge["v2"] if terminal == edge["v1"] else edge["v1"]
            if len(adjacency[junction]) < 3:
                raise ValueError("Engineering fold review must attach to a source branching node")
            record = {**review, "terminal_node_id": terminal, "junction_node_id": junction,
                      "source_branch_ids": [review["branch_id"]], "engineering_review": True,
                      "anatomical_validation": "not_performed",
                      "effect": "Retain source path and ID; omit reviewed terminal spur from generation divisions"}
            prepared["reviewed_terminal_spurs"].append(record)
            edges[review["branch_id"]]["folded_into_junction"] = record
        components.append(prepared)
    return components, _root_candidates(components, radius_at)


def _root_candidates(components, radius_at):
    candidates = []
    for component in components:
        edges, adjacency = component["edges"], component["adjacency"]
        topology_adjacency = {node: [(other, key) for other, key in neighbors
                                    if not edges[key].get("folded_into_junction")]
                              for node, neighbors in adjacency.items()}
        for node, neighbors in topology_adjacency.items():
            if len(neighbors) != 1:
                continue
            edge = edges[neighbors[0][1]]
            if edge["flags"] or len(edge["path"]) < 2:
                continue
            position = 0 if edge["v1"] == node else -1
            radii = np.asarray(radius_at(edge["path"]), dtype=float)
            candidates.append({"component_id": component["component_id"],
                               "node_id": node, "source_branch_id": edge["branch_id"],
                               "point_mm": edge["world"][position].tolist(),
                               "superior_z_mm": float(edge["world"][position, 2]),
                               "mean_lumen_edt_radius_mm": float(np.mean(radii)),
                               "branch_length_mm": edge["physical_length_mm"]})
    return candidates


def _apply_reviewed_subtrees(components, reviewed_subtrees, root):
    """Fold explicit pendant source trees without deleting any source geometry."""
    by_component = {component["component_id"]: component for component in components}
    for component in components:
        component["reviewed_terminal_subtrees"] = []
    for review in reviewed_subtrees:
        component = by_component[review["component_id"]]
        edges, adjacency = component["edges"], component["adjacency"]
        identifiers = set(review["source_branch_ids"])
        selected = [edges[key] for key in identifiers]
        if any(edge["flags"] for edge in selected):
            raise ValueError("Engineering subtree review requires valid source paths")
        nodes = {edge[name] for edge in selected for name in ("v1", "v2")}
        reached, queue = set(), [min(nodes)]
        while queue:
            node = queue.pop()
            if node in reached:
                continue
            reached.add(node)
            queue.extend(other for other, key in adjacency[node]
                         if key in identifiers and other not in reached)
        if reached != nodes or len(identifiers) != len(nodes) - 1:
            raise ValueError("Engineering subtree review must identify one connected source tree")
        attachments = {node for node in nodes
                       if any(key not in identifiers for _, key in adjacency[node])}
        if len(attachments) != 1:
            raise ValueError("Engineering subtree review requires exactly one external attachment")
        attachment = next(iter(attachments))
        if review.get("attachment_node_id") != attachment or len(adjacency[attachment]) < 3:
            raise ValueError("Engineering subtree review must name its source branching attachment")
        if not any(key not in identifiers and not edges[key].get("folded_into_junction")
                   for _, key in adjacency[attachment]):
            raise ValueError("Engineering subtree review must retain an attached source edge")
        if (root is not None and root["component_id"] == component["component_id"]
                and (root["source_branch_id"] in identifiers or root["node_id"] in nodes - {attachment})):
            raise ValueError("Engineering subtree review cannot remove the selected tracheal root")
        record = {**review, "source_branch_ids": sorted(identifiers),
                  "junction_node_id": attachment,
                  "terminal_node_ids": sorted(node for node in nodes if len(adjacency[node]) == 1),
                  "engineering_review": True, "reviewed_subtree": True,
                  "anatomical_validation": "not_performed",
                  "effect": "Retain every source path and ID; omit the reviewed terminal subtree from generation divisions"}
        component["reviewed_terminal_subtrees"].append(record)
        for key in identifiers:
            edges[key]["folded_into_junction"] = record


def _reachable_source_nodes(component, start, excluded=()):
    excluded, reached, queue = set(excluded), set(), [start]
    while queue:
        node = queue.pop()
        if node in reached:
            continue
        reached.add(node)
        for other, key in component["adjacency"][node]:
            edge = component["edges"][key]
            if key not in excluded and not edge["flags"] and not edge.get("folded_into_junction"):
                queue.append(other)
    return reached


def _apply_reviewed_redundant_paths(components, reviewed_paths, root):
    """Omit only explicitly reviewed groups with a retained alternative route."""
    by_component = {component["component_id"]: component for component in components}
    for component in components:
        component["reviewed_redundant_paths"] = []
    for review in reviewed_paths:
        component = by_component[review["component_id"]]
        edges, adjacency = component["edges"], component["adjacency"]
        identifiers = set(review["source_branch_ids"])
        selected = [edges[key] for key in identifiers]
        if any(edge["flags"] for edge in selected):
            raise ValueError("Engineering redundant-path review requires valid source paths")
        nodes = {edge[name] for edge in selected for name in ("v1", "v2")}
        reached, queue = set(), [min(nodes)]
        while queue:
            node = queue.pop()
            if node in reached:
                continue
            reached.add(node)
            queue.extend(other for other, key in adjacency[node]
                         if key in identifiers and other not in reached)
        if reached != nodes:
            raise ValueError("Engineering redundant-path review must identify one connected source group")
        boundaries = {node for node in nodes
                      if any(key not in identifiers for _, key in adjacency[node])}
        declared = review.get("boundary_node_ids", [])
        if len(boundaries) != 2 or len(declared) != 2 or set(declared) != boundaries:
            raise ValueError("Engineering redundant-path review requires exactly two explicit external boundaries")
        if (root is not None and root["component_id"] == component["component_id"]
                and (root["source_branch_id"] in identifiers or root["node_id"] in nodes - boundaries)):
            raise ValueError("Engineering redundant-path review cannot remove the selected tracheal root")
        first, second = sorted(boundaries)
        reachable_before = _reachable_source_nodes(component, first)
        reachable_after = _reachable_source_nodes(component, first, identifiers)
        if second not in reachable_after:
            raise ValueError("Engineering redundant-path review requires a retained alternative route")
        internal_nodes = nodes - boundaries
        if not reachable_before - internal_nodes <= reachable_after:
            raise ValueError("Engineering redundant-path review cannot disconnect external source nodes")
        record = {**review, "source_branch_ids": sorted(identifiers),
                  "boundary_node_ids": [first, second], "internal_node_ids": sorted(internal_nodes),
                  "engineering_review": True, "reviewed_redundant_path": True,
                  "anatomical_validation": "not_performed",
                  "retained_alternative_route_verified": True,
                  "effect": "Retain every source path and ID; omit the explicitly reviewed redundant group while retaining external connectivity"}
        component["reviewed_redundant_paths"].append(record)
        for key in identifiers:
            edges[key]["folded_into_junction"] = record


def _intraluminal_terminal_spurs(component, affine, radius_at):
    """Fold only terminal paths wholly inside an original junction's EDT ball.

    Such voxel paths remain inside the local EDT sphere represented by that node.
    EDT is distance to background voxel centers, not an exact continuous wall.
    This is an engineering topology approximation, not a whole-tree length cutoff.
    Source edges and voxels remain intact and individually mapped in the result.
    """
    evidence = []
    adjacency, edges, nodes = (component[name] for name in ("adjacency", "edges", "nodes"))
    for terminal in sorted(node for node, neighbors in adjacency.items() if len(neighbors) == 1):
        node, edge_id = terminal, adjacency[terminal][0][1]
        path, source_ids = [], []
        while True:
            edge = edges[edge_id]
            if edge["flags"]:
                path = []
                break
            ordered = edge["path"] if edge["v1"] == node else edge["path"][::-1]
            if path:
                try:
                    path.extend(_bridge(nodes[node], path[-1], ordered[0], affine)[1:].tolist())
                except ValueError:
                    path = []
                    break
                path.extend(ordered[1:].tolist())
            else:
                path.extend(ordered.tolist())
            source_ids.append(edge_id)
            other = edge["v2"] if edge["v1"] == node else edge["v1"]
            if len(adjacency[other]) != 2:
                junction = other
                break
            next_edges = [key for _, key in adjacency[other] if key != edge_id]
            node, edge_id = other, next_edges[0]
        if not path or len(adjacency[junction]) < 3:
            continue
        if np.any(np.asarray(radius_at(np.asarray(path)), dtype=float) <= 0):
            continue
        world = nib.affines.apply_affine(affine, path)
        centers = nib.affines.apply_affine(affine, nodes[junction])
        radii = np.asarray(radius_at(nodes[junction]), dtype=float)
        extents = np.linalg.norm(world[:, None, :] - centers[None, :, :], axis=2).max(axis=0)
        clearance = radii - extents
        chosen = int(np.argmax(clearance))
        if clearance[chosen] <= 1e-6:
            continue
        record = {"component_id": component["component_id"], "terminal_node_id": terminal,
                  "junction_node_id": junction, "source_branch_ids": source_ids,
                  "ball_center_ijk": nodes[junction][chosen].tolist(),
                  "ball_center_mm": centers[chosen].tolist(), "ball_radius_mm": float(radii[chosen]),
                  "maximum_path_distance_mm": float(extents[chosen]),
                  "interior_clearance_mm": float(clearance[chosen]),
                  "strict_interior_tolerance_mm": 1e-6,
                  "radius_definition": "EDT distance to background voxel centers; not an exact continuous anatomical wall",
                  "reason": "terminal_path_wholly_inside_existing_junction_lumen_inscribed_ball",
                  "effect": "Retain source path and ID; omit this intraluminal spur from generation divisions"}
        evidence.append(record)
        for key in source_ids:
            edges[key]["folded_into_junction"] = record
    return evidence


def _choose_root(candidates):
    if not candidates:
        return None, [], ["no_terminal_branch_for_tracheal_root"]
    candidates = [dict(row) for row in candidates]
    superior = np.asarray([row["superior_z_mm"] for row in candidates])
    radius = np.asarray([row["mean_lumen_edt_radius_mm"] for row in candidates])
    # Unitless equal-weight rank within this image; retain the physical evidence.
    z_score = (superior - superior.min()) / max(float(np.ptp(superior)), 1e-9)
    r_score = radius / max(float(radius.max()), 1e-9)
    for row, z, r in zip(candidates, z_score, r_score):
        row["selection_score"] = float(.5 * z + .5 * r)
    candidates.sort(key=lambda row: (-row["selection_score"], -row["superior_z_mm"],
                                    row["component_id"], row["node_id"]))
    chosen, flags = candidates[0], []
    if chosen["superior_z_mm"] < float(superior.max()) - 1e-6:
        flags.append("root_not_most_superior_endpoint")
    if chosen["mean_lumen_edt_radius_mm"] < float(radius.max()) - 1e-6:
        flags.append("root_not_largest_endpoint_branch")
    if len(candidates) > 1 and chosen["selection_score"] - candidates[1]["selection_score"] < .1:
        flags.append("ambiguous_root_score")
    return chosen, candidates, flags


def _spanning_tree(component, root_node):
    """Distance tree is an explicit cycle-resolution approximation, not anatomy."""
    distance, parent_edge = {root_node: 0.}, {}
    queue, equal_cost = [(0., root_node)], set()
    while queue:
        cost, node = heapq.heappop(queue)
        if cost != distance[node]:
            continue
        for other, edge_id in sorted(component["adjacency"][node]):
            edge = component["edges"][edge_id]
            if edge["flags"] or edge.get("folded_into_junction") or other == node:
                continue
            candidate = cost + max(edge["physical_length_mm"], 1e-9)
            current = distance.get(other, np.inf)
            if candidate < current - 1e-9:
                distance[other], parent_edge[other] = candidate, edge_id
                heapq.heappush(queue, (candidate, other))
            elif abs(candidate - current) <= 1e-9 and parent_edge.get(other) != edge_id:
                equal_cost.add(other)
    return set(parent_edge.values()), distance, equal_cost


def _compress(component, selected, affine):
    """Maximal degree-two chains, including paths inside original node clusters."""
    adjacency = defaultdict(list)
    for edge_id in selected:
        edge = component["edges"][edge_id]
        adjacency[edge["v1"]].append((edge["v2"], edge_id))
        adjacency[edge["v2"]].append((edge["v1"], edge_id))
    visited, result = set(), []
    critical = sorted(node for node in adjacency if len(adjacency[node]) != 2)
    for initial in critical:
        for _, initial_edge in sorted(adjacency[initial]):
            if initial_edge in visited:
                continue
            node, edge_id, path, mappings, flags = initial, initial_edge, [], [], []
            while True:
                visited.add(edge_id)
                edge = component["edges"][edge_id]
                reverse = edge["v2"] == node
                ordered = edge["path"][::-1] if reverse else edge["path"]
                mappings.append({"component_id": component["component_id"],
                                 "branch_id": edge_id, "reversed": bool(reverse)})
                flags.extend(edge["flags"])
                if len(ordered):
                    if len(path):
                        try:
                            bridge = _bridge(component["nodes"][node], path[-1], ordered[0], affine)
                            path.extend(bridge[1:].tolist())
                        except ValueError as error:
                            flags.append(str(error))
                        path.extend(ordered[1:].tolist())
                    else:
                        path.extend(ordered.tolist())
                other = edge["v1"] if reverse else edge["v2"]
                if len(adjacency[other]) != 2:
                    terminal = other
                    break
                remaining = [(neighbor, candidate) for neighbor, candidate in adjacency[other]
                             if candidate != edge_id]
                node, edge_id = other, remaining[0][1]
            source_ids = sorted(row["branch_id"] for row in mappings)
            world = nib.affines.apply_affine(affine, np.asarray(path).reshape(-1, 3))
            length = float(np.linalg.norm(np.diff(world, axis=0), axis=1).sum())
            result.append({"branch_id": f"AG_C{component['component_id']}_B" + "_".join(map(str, source_ids)),
                           "component_id": component["component_id"], "source_branches": mappings,
                           "proximal_node_id": initial, "distal_node_id": terminal,
                           "path_ijk": path, "points_mm": world.tolist(), "length_mm": length,
                           "source_length_mm": sum(component["edges"][key]["length_mm"] for key in source_ids),
                           "flags": sorted(set(flags)), "generation": None})
    return result


def generations_from_graph(graph, affine_mm, radius_at, *, max_generation=6, reviewed_spurs=(),
                           reviewed_subtrees=(), reviewed_redundant_paths=()):
    """Pure graph entry point; ``radius_at`` returns EDT mm at integer XYZ paths."""
    if isinstance(max_generation, bool) or not isinstance(max_generation, int) or max_generation < 0:
        raise ValueError("max_generation must be a nonnegative integer")
    affine = np.asarray(affine_mm, dtype=float)
    if affine.shape != (4, 4) or not np.isfinite(affine).all() or abs(np.linalg.det(affine[:3, :3])) < 1e-12:
        raise ValueError("A finite nonsingular world-mm affine is required")
    identities = {(component["component_id"], branch["branch_id"])
                  for component in graph["components"] for branch in component["branches"]}
    if any((review["component_id"], review["branch_id"]) not in identities for review in reviewed_spurs):
        raise ValueError("Engineering fold review refers to an unknown source branch")
    subtree_identities = []
    for review in reviewed_subtrees:
        if not review.get("source_branch_ids"):
            raise ValueError("Engineering subtree review requires explicit source branches")
        subtree_identities.extend((review["component_id"], key) for key in review["source_branch_ids"])
    if any(identity not in identities for identity in subtree_identities):
        raise ValueError("Engineering subtree review refers to an unknown source branch")
    spur_identities = {(review["component_id"], review["branch_id"]) for review in reviewed_spurs}
    if (len(set(subtree_identities)) != len(subtree_identities)
            or set(subtree_identities) & spur_identities):
        raise ValueError("Engineering subtree review contains duplicate or overlapping source branches")
    redundant_identities = []
    for review in reviewed_redundant_paths:
        if not review.get("source_branch_ids"):
            raise ValueError("Engineering redundant-path review requires explicit source branches")
        redundant_identities.extend((review["component_id"], key) for key in review["source_branch_ids"])
    if any(identity not in identities for identity in redundant_identities):
        raise ValueError("Engineering redundant-path review refers to an unknown source branch")
    if (len(set(redundant_identities)) != len(redundant_identities)
            or set(redundant_identities) & (spur_identities | set(subtree_identities))):
        raise ValueError("Engineering redundant-path review contains duplicate or overlapping source branches")
    components, candidates = _prepare_graph(graph, affine, radius_at, reviewed_spurs)
    baseline_root = _choose_root(candidates)[0]
    _apply_reviewed_subtrees(components, reviewed_subtrees, baseline_root)
    _apply_reviewed_redundant_paths(components, reviewed_redundant_paths, baseline_root)
    if reviewed_subtrees or reviewed_redundant_paths:
        candidates = _root_candidates(components, radius_at)
    root, ranked, root_flags = _choose_root(candidates)
    if ((reviewed_subtrees or reviewed_redundant_paths) and baseline_root is not None
            and (root is None or (root["component_id"], root["node_id"]) !=
                 (baseline_root["component_id"], baseline_root["node_id"]))):
        kind = "redundant-path" if reviewed_redundant_paths else "subtree"
        raise ValueError(f"Engineering {kind} review cannot change the selected tracheal root")
    branches, cycle_count = [], 0
    for component in components:
        rooted = root is not None and component["component_id"] == root["component_id"]
        if rooted:
            selected, distance, equal_cost = _spanning_tree(component, root["node_id"])
        else:
            selected, distance, equal_cost = set(), {}, set()
        excluded = set(component["edges"]) - selected
        cycle_edges = {key for key in excluded if rooted and not component["edges"][key]["flags"]
                       and not component["edges"][key].get("folded_into_junction")
                       and component["edges"][key]["v1"] in distance
                       and component["edges"][key]["v2"] in distance}
        cycle_nodes = {component["edges"][key][name] for key in cycle_edges for name in ("v1", "v2")}
        reviewed_nodes = {row["junction_node_id"] for row in
                          component["reviewed_terminal_spurs"] + component["reviewed_terminal_subtrees"]}
        reviewed_nodes.update(node for row in component["reviewed_redundant_paths"]
                              for node in row["boundary_node_ids"])
        cycle_count += len(cycle_edges)
        mapped = _compress(component, selected, affine)
        for row in mapped:
            if distance[row["proximal_node_id"]] > distance[row["distal_node_id"]]:
                row["proximal_node_id"], row["distal_node_id"] = row["distal_node_id"], row["proximal_node_id"]
                row["path_ijk"].reverse()
                row["points_mm"].reverse()
                row["source_branches"].reverse()
                for source in row["source_branches"]:
                    source["reversed"] = not source["reversed"]
            source_nodes = {component["edges"][source["branch_id"]][name]
                            for source in row["source_branches"] for name in ("v1", "v2")}
            if source_nodes & cycle_nodes:
                row["flags"].append("cycle_adjacent_generation_needs_review")
            if source_nodes & equal_cost:
                row["flags"].append("equal_cost_parent_needs_review")
            if source_nodes & reviewed_nodes:
                row["flags"].append("engineering_reviewed_generation_topology_correction")
        by_start = defaultdict(list)
        for row in mapped:
            by_start[row["proximal_node_id"]].append(row)
        queue = deque([(root["node_id"], 0, bool(root_flags))]) if rooted else deque()
        while queue:
            node, generation, inherited_ambiguity = queue.popleft()
            short_side_branch = len(by_start[node]) > 1 and any(
                child["length_mm"] < 2. and not by_start[child["distal_node_id"]]
                for child in by_start[node])
            for row in sorted(by_start[node], key=lambda value: value["branch_id"]):
                row["generation"] = generation
                if short_side_branch:
                    # A terminal stub shorter than the tangent support could be
                    # a skeleton artifact. Preserve it, but flag this division.
                    row["flags"].append("short_terminal_side_branch_generation_needs_review")
                if inherited_ambiguity:
                    row["flags"].append("upstream_generation_needs_review")
                ambiguous = inherited_ambiguity or any("generation" in flag or "parent" in flag for flag in row["flags"])
                queue.append((row["distal_node_id"], generation + 1, ambiguous))
        branches.extend(mapped)
        for edge_id in sorted(excluded):
            edge = component["edges"][edge_id]
            reason = ("engineering_reviewed_redundant_path" if (edge.get("folded_into_junction") or {}).get("reviewed_redundant_path") else
                      "engineering_reviewed_terminal_subtree" if (edge.get("folded_into_junction") or {}).get("reviewed_subtree") else
                      "engineering_reviewed_terminal_spur" if (edge.get("folded_into_junction") or {}).get("engineering_review") else
                      "intraluminal_terminal_spur" if edge.get("folded_into_junction") else
                      "cycle_cross_edge" if edge_id in cycle_edges else "disconnected_from_tracheal_root")
            flags = edge["flags"] + [reason]
            if not rooted and component["source_cycle_rank"]:
                flags.append("unrooted_component_contains_cycle")
            branches.append({"branch_id": f"AG_C{component['component_id']}_B{edge_id}",
                             "component_id": component["component_id"],
                             "source_branches": [{"component_id": component["component_id"],
                                                   "branch_id": edge_id, "reversed": False}],
                             "proximal_node_id": edge["v1"], "distal_node_id": edge["v2"],
                             "path_ijk": edge["path"].tolist(), "points_mm": edge["world"].tolist(),
                             "length_mm": edge["physical_length_mm"], "source_length_mm": edge["length_mm"],
                             "generation": None, "flags": sorted(set(flags)),
                             "topology_normalization": edge.get("folded_into_junction")})
    for row in branches:
        row["in_requested_range"] = row["generation"] is not None and row["generation"] <= max_generation
        geometry_bad = any(flag in row["flags"] for flag in (
            "source_path_break", "zero_length_source_branch", "disconnected_junction_cluster",
            "junction_attachment_outside_cluster"))
        row["status"] = ("unavailable" if row["generation"] is None or geometry_bad else
                         "needs_review" if row["flags"] else "estimated")
        row["sampling_margin_mm"] = .5
        row["sampling_unavailable_reason"] = ("branch_unavailable" if row["status"] == "unavailable" else
                                              "insufficient_branch_length" if row["length_mm"] < 2. else None)
        row["flags"] = sorted(set(row["flags"]))
    counts = Counter(row["generation"] for row in branches if row["generation"] is not None)
    return {"schema_version": 1, "method": METHOD, "includes_trachea": True,
            "manual_anatomical_confirmation": "not_performed", "max_generation": max_generation,
            "root": {"status": "unavailable" if root is None else "needs_review" if root_flags else "estimated",
                     "selected": root, "flags": root_flags, "candidate_count": len(ranked),
                     "candidates": ranked, "coordinate_convention": "NIfTI RAS world mm; +Z superior",
                     "policy": "Equal-weight normalized superior position and mean branch EDT caliber; terminal branches only"},
            "branches": sorted(branches, key=lambda row: (row["generation"] is None,
                               row["generation"] if row["generation"] is not None else 0, row["branch_id"])),
            "source_branch_count": sum(len(c["edges"]) for c in components),
            "source_cycle_rank": sum(c["source_cycle_rank"] for c in components),
            "intraluminal_terminal_spurs": [spur for c in components for spur in c["intraluminal_terminal_spurs"]],
            "intraluminal_terminal_spur_count": sum(len(c["intraluminal_terminal_spurs"]) for c in components),
            "intraluminal_terminal_spur_source_edge_count": sum(len(spur["source_branch_ids"])
                                                               for c in components for spur in c["intraluminal_terminal_spurs"]),
            "engineering_reviewed_terminal_spurs": [spur for c in components for spur in c["reviewed_terminal_spurs"]],
            "engineering_reviewed_terminal_subtrees": [subtree for c in components
                                                        for subtree in c["reviewed_terminal_subtrees"]],
            "engineering_reviewed_redundant_paths": [path for c in components
                                                      for path in c["reviewed_redundant_paths"]],
            "source_components": [{"component_id": c["component_id"], "node_count": len(c["nodes"]),
                                   "edge_count": len(c["edges"]),
                                   "connected_part_count": c["source_connected_part_count"],
                                   "cycle_rank": c["source_cycle_rank"]} for c in components],
            "mapped_branch_count": len(branches), "cycle_cross_edge_count": cycle_count,
            "generation_counts": {str(generation): counts[generation] for generation in range(max_generation + 1)},
            "missing_generations": [generation for generation in range(max_generation + 1) if not counts[generation]],
            "ambiguous_generation_branch_count": sum(row["generation"] is not None and bool(row["flags"]) for row in branches),
            "unassigned_branch_count": sum(row["generation"] is None for row in branches),
            "beyond_requested_generation_count": sum(row["generation"] is not None and row["generation"] > max_generation for row in branches),
            "topology_policy": "Fold only terminal paths wholly inside an original junction EDT-inscribed ball; retain source IDs and geometric evidence; physical shortest-path tree from tracheal endpoint; retain non-tree edges unassigned; compress degree-two chains through source junction voxels before generation increments" +
                               ("; explicit engineering review additionally folds verified pendant source trees with one external attachment" if reviewed_subtrees else "") +
                               ("; explicit engineering review additionally omits connected redundant groups with two external boundaries and a retained alternative route" if reviewed_redundant_paths else ""),
            "generation_interpretation": "Exploratory topology-derived generations; cycles and upstream ambiguity remain flagged"}


def build_generations(root: Path, case: str, input_record: dict, max_generation=6,
                      generation_review=None) -> dict:
    """Verify the existing airway scope, then root and map without changing it."""
    root = Path(root).resolve()
    verified = verify_saved_graph(root, case, input_record)
    directory = root / "airway" / case
    graph = _read(directory / "skeleton_graph.json")
    reviewed_spurs, reviewed_subtrees, reviewed_redundant_paths = [], [], []
    if generation_review is not None:
        if (generation_review.get("case") != case or generation_review.get("source_graph_sha256") !=
                _sha256(directory / "skeleton_graph.json")):
            raise ValueError("Generation review must match the case and exact source graph hash")
        if (generation_review.get("configuration_type") != "final_engineering_topology_adjustment"
                or generation_review.get("review_type") != "engineering_visual_topology_correction"
                or generation_review.get("anatomical_validation") != "not_performed"
                or generation_review.get("evidence_rechecked") is not False):
            raise ValueError("Final adjustments require declared engineering provenance and evidence_rechecked=false")
        reviewed_spurs = generation_review.get("folded_branches", [])
        reviewed_subtrees = generation_review.get("folded_subtrees", [])
        reviewed_redundant_paths = generation_review.get("reviewed_redundant_paths", [])
        identifiers = [(row["component_id"], row["branch_id"]) for row in reviewed_spurs]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("Generation adjustment contains duplicate source branches")
        for row in reviewed_spurs + reviewed_subtrees + reviewed_redundant_paths:
            digests = row.get("evidence_sha256", [])
            if (not row.get("reason") or not digests or row.get("evidence_rechecked") is not False
                    or any(not isinstance(value, str) or len(value) != 64
                           or any(letter not in "0123456789abcdef" for letter in value) for value in digests)):
                raise ValueError("Final adjustments require a reason and historical evidence digests")
            if "evidence" in row:
                raise ValueError("Final configuration accepts evidence digests only, not historical file paths")
    image = nib.load(input_record["mask"])
    mask = _binary(image, "Airway lumen")
    spacing = np.linalg.norm(image.affine[:3, :3], axis=0)
    directions = image.affine[:3, :3] / spacing
    if not np.allclose(directions.T @ directions, np.eye(3), rtol=0, atol=1e-5):
        raise ValueError("EDT root-caliber evidence requires orthogonal voxel axes")
    voxels = np.argwhere(mask)
    low, high = voxels.min(axis=0), voxels.max(axis=0) + 1
    crop = tuple(slice(int(a), int(b)) for a, b in zip(low, high))
    distance = ndimage.distance_transform_edt(np.pad(mask[crop], 1), sampling=spacing)

    def radius_at(path):
        return ndimage.map_coordinates(distance, (np.asarray(path) - low + 1).T,
                                       order=1, mode="constant", cval=0.)

    result = generations_from_graph(graph, image.affine, radius_at, max_generation=max_generation,
                                    reviewed_spurs=reviewed_spurs, reviewed_subtrees=reviewed_subtrees,
                                    reviewed_redundant_paths=reviewed_redundant_paths)
    return {**result, "case": case, "source_sha256": verified["source_sha256"],
            "generation_adjustment": generation_review,
            "evidence_rechecked": False if generation_review is not None else None,
            "mask": str(Path(input_record["mask"]).resolve()), "affine_mm": image.affine.tolist(),
            "source_graph": str((directory / "skeleton_graph.json").resolve()),
            "verification": "Saved graph, skeleton, mask and source hash verification"}


def sample_branch(branch: dict, step_mm=.5, tangent_window_mm=4.) -> list[dict]:
    """Sample at physical intervals using same-branch PCA and continuous frames."""
    if not np.isfinite(step_mm) or step_mm <= 0:
        raise ValueError("step_mm must be finite and positive")
    config = TangentConfig(tangent_window_mm=tangent_window_mm)
    config.validate()
    if branch.get("status") == "unavailable" or branch.get("generation") is None:
        return []
    points, arc = _polyline(np.asarray(branch["points_mm"], dtype=float))
    margin = float(branch.get("sampling_margin_mm", .5))
    if not np.isfinite(margin) or margin < 0:
        raise ValueError("sampling_margin_mm must be finite and nonnegative")
    if arc[-1] < max(config.min_tangent_span_mm, 2 * margin) - 1e-9:
        return []
    positions = np.arange(margin, arc[-1] - margin + 1e-9, step_mm)
    result, previous = [], None
    for index, position in enumerate(positions):
        fit = estimate_tangent(points, float(position), config)
        row = {"section_id": f"{branch['branch_id']}_S{index + 1:05d}",
               "branch_id": branch["branch_id"], "component_id": branch["component_id"],
               "generation": branch["generation"], "s_mm": float(position),
               "center_mm": _interpolate(points, arc, np.asarray([position]))[0].tolist(),
               "tangent_fit": fit, "status": "unusable", "tangent_mm": None,
               "e1": None, "e2": None, "flags": list(branch.get("flags", [])),
               "reasons": fit["reasons"]}
        if fit["status"] == "ok":
            tangent = np.asarray(fit["tangent"])
            first, second = orthogonal_basis(tangent)
            if previous is not None:
                projected = previous - np.dot(previous, tangent) * tangent
                if np.linalg.norm(projected) > 1e-8:
                    first = projected / np.linalg.norm(projected)
                    second = np.cross(tangent, first)
            previous = first
            row.update(status="ready", tangent_mm=tangent.tolist(), e1=first.tolist(), e2=second.tolist())
        else:
            previous = None
        result.append(row)
    return result
