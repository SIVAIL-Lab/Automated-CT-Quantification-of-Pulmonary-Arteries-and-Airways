"""Build one PARSE prediction EDT tree from verified saved pipeline artifacts."""
from __future__ import annotations

from collections import Counter
import argparse
import importlib.metadata
import json
from pathlib import Path
import re
import resource
import time
import sys

import nibabel as nib
import numpy as np

from measurement_io import write_csv, write_json
from edt_centerline_tree import build_tree
from edt_tree_radius import measure_tree_radius
import run_edt_experiment as edt
from measurement_io import largest_component, matching_image, verify_hashes


PARAMETERS = {
    "component_connectivity": 26, "path_connectivity": 26,
    "power": 2.0, "epsilon_mm": 0.1, "max_crop_voxels": 150_000_000,
    "endpoint_voxel_selection": "maximum_edt_within_degree_one_graph_node_then_lexicographic_ijk",
    "root_selection": "maximum_endpoint_edt_then_lexicographic_ijk_component_node",
    "target_order": "lexicographic_ijk_component_node",
    "segment_radius_method": "length_weighted_mean_edt", "radius_units": "mm",
    "shared_edges": "count_once", "geometry_changed": False,
}


def computation_identity() -> dict:
    """Return the exact reusable computation identity, excluding presentation."""
    names = ("parse_radius_case.py", "run_edt_experiment.py", "measurement_io.py",
             "edt_centerline_tree.py", "edt_tree_radius.py")
    return {
        "parameters": dict(PARAMETERS),
        "code_sha256": {str(Path(__file__).with_name(name).resolve()):
                        edt.sha256(Path(__file__).with_name(name)) for name in names},
        "versions": {name: importlib.metadata.version(name)
                     for name in ("numpy", "scipy", "nibabel", "scikit-image")},
    }


def _paths(case: str, input_root: Path, results_root: Path) -> dict[str, Path]:
    if not re.fullmatch(r"PA[0-9]{6}", case):
        raise ValueError("Expected a PARSE case ID such as PA000005")
    input_root, results_root = Path(input_root).resolve(), Path(results_root).resolve()
    from measurement_scope import load_profile, get_mask, source_results_root
    raw = source_results_root(results_root)
    raw_mask = raw / "totalsegmentator" / case / "lung_arteries.nii.gz"
    mask = (get_mask(input_root / case, results_root)[0]
            if load_profile(results_root) else raw_mask)
    skeleton = results_root / "skeletonize3d" / case / "lung_arteries_skeleton.nii.gz"
    graph = results_root / "analyzeskeleton" / case / "skeleton_graph.json"
    return {"ct": input_root / case / "image" / f"{case}.nii.gz", "mask": mask,
            "provenance": raw_mask.with_name("segmentation_provenance.json"),
            "skeleton": skeleton, "skeleton_report": skeleton.with_name("skeleton_report.json"),
            "graph": graph, "analysis_report": graph.with_name("analysis_report.json")}


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def source_snapshot(case: str, input_root: Path, results_root: Path) -> dict[str, str]:
    """Verify the saved mask/skeleton/graph chain and hash all required inputs.

    The saved plugin identities are retained without requiring Fiji to be
    installed. A legacy CT association remains explicitly unverified.
    """
    paths = _paths(case, input_root, results_root)
    hashes = {str(path): edt.sha256(path) for path in paths.values()}
    from measurement_scope import load_profile, get_mask, source_results_root
    profile = load_profile(results_root)
    raw_mask = source_results_root(results_root) / "totalsegmentator" / case / "lung_arteries.nii.gz"
    if profile:
        _, selection = get_mask(Path(input_root) / case, results_root)
        hashes.update(selection["source_sha256"])
    provenance = _read_json(paths["provenance"])
    skeleton_report = _read_json(paths["skeleton_report"])
    analysis_report = _read_json(paths["analysis_report"])
    for report in (provenance, skeleton_report, analysis_report):
        if report.get("case") != case:
            raise ValueError("Saved source report case identity differs from requested case")
    for report in (skeleton_report, analysis_report):
        if report.get("status") not in ("ok", "skipped") or report.get("error"):
            raise ValueError("Saved skeleton and analysis must have completed successfully")
    state = provenance.get("provenance_status")
    if (state not in ("inference_verified", "legacy_adopted")
            or provenance.get("model_input_verified") is not (state == "inference_verified")):
        raise ValueError("Invalid segmentation provenance status or original CT association")
    if (provenance.get("input_sha256") != hashes[str(paths["ct"])]
            or provenance.get("artifact_sha256", {}).get(paths["mask"].name)
            != edt.sha256(raw_mask)):
        raise ValueError("Segmentation provenance does not match the live CT and artery mask")
    if (skeleton_report.get("input_sha256") != hashes[str(paths["mask"])]
            or skeleton_report.get("output_sha256") != hashes[str(paths["skeleton"])]):
        raise ValueError("Skeleton report does not bind the live mask and skeleton")
    analysis_provenance = analysis_report.get("provenance", {})
    if analysis_provenance.get("input_sha256") != hashes[str(paths["skeleton"])]:
        raise ValueError("Graph analysis does not bind the live skeleton")
    if analysis_provenance.get("parameters") != {
            "prune_cycles": False, "prune_ends": False, "shortest_paths": False}:
        raise ValueError("Graph analysis must use the unchanged, unpruned skeleton")
    for digest in (skeleton_report.get("plugin_sha256"),
                   analysis_provenance.get("plugin", {}).get("sha256"),
                   analysis_provenance.get("bridge_sha256")):
        if not isinstance(digest, str) or not re.fullmatch("[0-9a-f]{64}", digest):
            raise ValueError("Saved skeleton/graph plugin and bridge identities are required")
    artifacts = analysis_report.get("artifact_sha256", {})
    if artifacts.get(paths["graph"].name) != hashes[str(paths["graph"])]:
        raise ValueError("Graph analysis does not bind its saved graph artifact")
    for name, expected in artifacts.items():
        path = (paths["graph"].parent / name).resolve()
        if not path.is_relative_to(paths["graph"].parent):
            raise ValueError("Analysis artifact path leaves its source directory")
        digest = hashes.get(str(path)) or edt.sha256(path)
        if digest != expected:
            raise ValueError(f"Analysis artifact hash mismatch: {path}")
        hashes[str(path)] = digest
    image = nib.load(paths["mask"])
    spacing, _ = edt.geometry_mm(image)
    matching_image(paths["skeleton"], image, "Existing skeleton")
    matching_image(paths["ct"], image, "Associated CT")
    for record in (skeleton_report, analysis_report, analysis_provenance):
        saved_spacing = np.asarray(record.get("spacing_mm"), dtype=float)
        if saved_spacing.shape != (3,) or not np.allclose(saved_spacing, spacing, atol=1e-6, rtol=0):
            raise ValueError("Saved skeleton/graph calibration differs from the mask")
    graph = _read_json(paths["graph"])
    if (graph.get("shape") != list(image.shape)
            or np.asarray(graph.get("affine")).shape != (4, 4)
            or not np.allclose(graph["affine"], image.affine, atol=1e-4, rtol=0)):
        raise ValueError("Saved graph geometry differs from the mask")
    return hashes


def select_seeds(graph: dict, selected: np.ndarray, skeleton: np.ndarray,
                 distances: np.ndarray) -> dict:
    """Select one original voxel per degree-one node, never snap endpoints.

    A degree-one Fiji graph node may be a junction voxel cluster. Choose its
    maximum-EDT original voxel with lexicographic IJK tie-breaking, and retain
    the full node voxel identity. Single-voxel endpoints are unchanged.
    """
    coordinates = graph.get("coordinate_system", {})
    if (coordinates.get("space"), coordinates.get("axes"), coordinates.get("index_base")) != (
            "voxel", "XYZ", 0):
        raise ValueError("Saved graph must use zero-based XYZ voxel coordinates")
    if list(selected.shape) != graph.get("shape") or skeleton.shape != selected.shape:
        raise ValueError("Graph and skeleton shapes differ from the selected component")
    endpoints, component_ids, occupied = [], set(), set()

    def integer(value, description):
        if type(value) is not int:
            raise ValueError(f"{description} must be an original integer")
        return value

    def point(value):
        ijk = edt.parse_point(value)
        if not edt.inside(ijk, skeleton.shape) or not skeleton[ijk]:
            raise ValueError("Saved graph voxel lies outside the original skeleton")
        return ijk

    for component in graph["components"]:
        component_id = integer(component["component_id"], "Component ID")
        if component_id in component_ids:
            raise ValueError("Duplicate skeleton graph component ID")
        component_ids.add(component_id)
        nodes, degree, branch_ids = {}, Counter(), set()
        for node in component["nodes"]:
            node_id = integer(node["node_id"], "Node ID")
            if node_id in nodes:
                raise ValueError("Duplicate skeleton graph node ID")
            nodes[node_id] = [point(value) for value in node["voxels"]]
            if not nodes[node_id]:
                raise ValueError("A skeleton graph node has no original voxel")
        for branch in component["branches"]:
            branch_id = integer(branch["branch_id"], "Branch ID")
            if branch_id in branch_ids:
                raise ValueError("Duplicate skeleton graph branch ID")
            branch_ids.add(branch_id)
            for key in ("v1", "v2"):
                node_id = integer(branch[key], "Branch endpoint ID")
                if node_id not in nodes:
                    raise ValueError("Graph branch references an unknown node")
                degree[node_id] += 1
            for voxel in branch["slabs"]:
                point(voxel)
        for node_id, voxels in nodes.items():
            if degree[node_id] != 1:
                continue
            membership = {bool(selected[ijk]) for ijk in voxels}
            if len(membership) != 1:
                raise ValueError("Degree-one graph node voxels have mixed selected-component membership")
            if membership == {False}:
                continue
            radii = {ijk: float(distances[ijk]) for ijk in voxels}
            if any(not np.isfinite(radius) or radius <= 0 for radius in radii.values()):
                raise ValueError("Endpoint EDT must be finite and positive")
            ijk = min(voxels, key=lambda point: (-radii[point], point))
            if ijk in occupied:
                raise ValueError("Duplicate selected endpoint voxel")
            occupied.add(ijk)
            endpoints.append({"component_id": component_id, "node_id": node_id,
                              "ijk": list(ijk), "edt_mm": radii[ijk],
                              "node_voxel_count": len(voxels),
                              "node_voxels_ijk": [list(point) for point in sorted(voxels)]})
    if len(endpoints) < 2:
        raise ValueError("At least two degree-one endpoints in the selected component are required")
    order = lambda record: (tuple(record["ijk"]), record["component_id"], record["node_id"])
    root = min(endpoints, key=lambda record: (-record["edt_mm"], *order(record)))
    targets = sorted((record for record in endpoints if record is not root), key=order)
    return {"root_ijk": root["ijk"], "root_metadata": root,
            "targets_ijk": [record["ijk"] for record in targets],
            "target_metadata": [dict(target_id=index, **record)
                                for index, record in enumerate(targets, start=1)],
            "endpoint_count": len(endpoints), "source_is_anatomical_inlet": False,
            "coordinate_convention": "Zero-based original NIfTI indices; no endpoint snapping",
            "endpoint_voxel_selection": PARAMETERS["endpoint_voxel_selection"],
            "root_selection": PARAMETERS["root_selection"], "target_order": PARAMETERS["target_order"]}


def compute_case(case: str, input_root: Path, results_root: Path, output_dir: Path) -> dict:
    """Compute in a new case directory; preserve failures and all input files."""
    started = time.monotonic()
    paths = _paths(case, input_root, results_root)
    output = Path(output_dir).resolve()
    if output.exists():
        raise ValueError("Each case computation requires a new output directory")
    protected = [Path(input_root).resolve(), *(path.parent for path in paths.values())]
    if any(output == path or path in output.parents or output in path.parents for path in protected):
        raise ValueError("Case output must be separate from original source directories")
    output.mkdir(parents=True)
    result = {"schema_version": 1, "case": case, "source": "prediction", "status": "running",
              "technical_validation": "not_completed", "anatomical_validation": "not_performed",
              "visual_review": "not_performed", **computation_identity()}

    def progress(stage):
        result.update(stage=stage, elapsed_seconds=time.monotonic() - started,
                      peak_memory_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0)
        write_json(output / "progress.json", result)

    try:
        progress("verify_sources")
        hashes = source_snapshot(case, input_root, results_root)
        result["source_sha256"] = hashes
        provenance = _read_json(paths["provenance"])
        result.update(segmentation_provenance_status=provenance["provenance_status"],
                      model_input_verified=provenance["model_input_verified"],
                      input_mask_path=str(paths["mask"]), ct_path=str(paths["ct"]))
        image = nib.load(paths["mask"])
        spacing, affine = edt.geometry_mm(image)
        mask = edt.binary_mask(np.asarray(image.dataobj))
        skeleton = edt.binary_mask(np.asarray(nib.load(paths["skeleton"]).dataobj))
        if np.any(skeleton & ~mask):
            raise ValueError("Existing skeleton contains voxels outside the original artery mask")
        selected, inclusion = largest_component(mask, float(np.prod(spacing)))
        del mask
        result.update(selected_component=inclusion, shape=list(image.shape),
                      affine_mm=affine.tolist(), spacing_mm=spacing.tolist(),
                      boundary_contact=edt.boundary_contact_summary(selected))
        selected_path, edt_path = output / "selected_component.nii.gz", output / "edt_mm.nii.gz"
        edt.save_like(selected.astype(np.uint8), image, selected_path)
        progress("compute_edt")
        distances = edt.compute_edt(selected, spacing, PARAMETERS["max_crop_voxels"])
        edt.save_like(distances, image, edt_path)
        seeds = select_seeds(_read_json(paths["graph"]), selected, skeleton, distances)
        del skeleton
        write_json(output / "seeds.json", seeds)
        result.update(root_ijk=seeds["root_ijk"], target_count=len(seeds["targets_ijk"]),
                      source_is_anatomical_inlet=False)
        progress("trace_26")
        trace = edt.trace_edt_paths(selected, distances, spacing, seeds["root_ijk"],
                                    seeds["targets_ijk"], connectivity=26,
                                    power=PARAMETERS["power"], epsilon_mm=PARAMETERS["epsilon_mm"])
        if (trace["technical_validation"] != "passed"
                or trace["successful_target_count"] != len(seeds["targets_ijk"])):
            raise ValueError("Every selected endpoint must have an audited successful path")
        for row, metadata in zip(trace["targets"], seeds["target_metadata"], strict=True):
            row.update(skeleton_component_id=metadata["component_id"],
                       skeleton_node_id=metadata["node_id"])
        write_json(output / "paths.json", trace)
        progress("build_tree")
        tree = build_tree(trace, selected, distances, affine)
        tree.update(case=case, source="prediction", shape=list(image.shape), spacing_mm=spacing.tolist(),
                    anatomical_branch_count=None, geometry_policy="Original audited directed voxel paths only")
        tree_path, tree_mask_path = output / "centerline_tree.json", output / "tree_mask.nii.gz"
        write_json(tree_path, tree)
        tree_mask = np.zeros(image.shape, dtype=np.uint8)
        points = np.asarray([row["ijk"] for row in tree["voxels"]], dtype=int)
        tree_mask[tuple(points.T)] = 1
        edt.save_like(tree_mask, image, tree_mask_path)
        del tree_mask, trace
        progress("measure_radius")
        measured = measure_tree_radius(tree, distances, selected, affine)
        points_path, segments_path = output / "radius_points.csv", output / "radius_segments.csv"
        write_csv(points_path, [dict(source="prediction", **row) for row in measured["points"]])
        write_csv(segments_path, [dict(source="prediction", **row) for row in measured["segments"]])
        result.update(summary=measured["summary"], definitions=measured["definitions"],
                      tree_summary=tree["summary"], checks=tree["checks"], flag_counts=tree["summary"]["flag_counts"],
                      tree_path=str(tree_path), tree_mask_path=str(tree_mask_path),
                      points_path=str(points_path), segments_path=str(segments_path),
                      mask_path=str(selected_path), edt_path=str(edt_path))
        verify_hashes(hashes)
        verify_hashes(result["code_sha256"])
        result.update(status="ok", technical_validation="passed", input_integrity="verified_after_computation")
        result["artifact_sha256"] = {str(path): edt.sha256(path) for path in sorted(output.iterdir())
                                     if path.is_file() and path.name not in ("progress.json", "report.json")}
        progress("complete")
        write_json(output / "report.json", result)
        return result
    except Exception as error:
        result.update(status="error", technical_validation="failed", error=f"{type(error).__name__}: {error}")
        progress("failed")
        write_json(output / "failure.json", result)
        raise


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--memory-limit-gb", type=float, default=10.0)
    args = parser.parse_args(argv)
    if not np.isfinite(args.memory_limit_gb) or args.memory_limit_gb <= 0:
        parser.error("--memory-limit-gb must be positive and finite")
    limit = int(args.memory_limit_gb * 1024 ** 3)
    resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
    try:
        result = compute_case(args.case, args.input_root, args.results_root, args.output_dir)
    except Exception as error:
        print(f"PARSE radius computation failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    print(json.dumps({"case": args.case, "status": result["status"], "summary": result["summary"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
