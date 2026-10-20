"""Verify the source identities and physical geometry of final airway graphs."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any
import nibabel as nib
import numpy as np


def _branch_path(branch: dict[str, Any], nodes: dict[int, np.ndarray]) -> np.ndarray:
    """Attach ordered slabs to adjacent node voxels, never to node centroids."""
    first, last = nodes[branch["v1"]], nodes[branch["v2"]]
    slabs = np.asarray(branch["slabs"], dtype=int).reshape(-1, 3)

    def neighbor(node: np.ndarray, point: np.ndarray) -> np.ndarray | None:
        delta = node - point
        nearby = np.flatnonzero(np.max(np.abs(delta), axis=1) <= 1)
        if not len(nearby):
            return None
        return node[nearby[np.argmin(np.sum(delta[nearby] ** 2, axis=1))]]

    if len(slabs):
        # Exports use traversal order. Accommodate the reverse direction only
        # if it is still adjacent to the correct endpoint vertex regions.
        for ordered in (slabs, slabs[::-1]):
            start, end = neighbor(first, ordered[0]), neighbor(last, ordered[-1])
            if start is not None and end is not None:
                path = np.vstack((start, ordered, end))
                keep = np.r_[True, np.any(np.diff(path, axis=0) != 0, axis=1)]
                path = path[keep]
                if len(path) > 1 and np.any(np.max(np.abs(np.diff(path, axis=0)), axis=1) != 1):
                    raise ValueError(f"Discontinuous slabs in branch {branch['branch_id']}")
                return path
    else:
        # A direct endpoint/junction connection can have no slab voxels.
        for point in first:
            end = neighbor(last, point)
            if end is not None:
                return np.asarray([point] if np.array_equal(point, end) else [point, end])
    raise ValueError(f"Branch {branch['branch_id']} does not touch both endpoint nodes")


def _load_analysis(
    skeleton: nib.Nifti1Image, directory: Path,
    reference_image: str = "lung_arteries_skeleton.nii.gz",
) -> dict[str, Any]:
    from run_analyzeskeleton_parse import (
        SCHEMA_VERSION, _artifact_hashes, _sha256, _validate_spatial_outputs,
    )

    try:
        report = json.loads((directory / "analysis_report.json").read_text(encoding="utf-8"))
        if (
            report["status"] not in ("ok", "skipped")
            or report["provenance"]["schema_version"] != SCHEMA_VERSION
            or report["provenance"]["input_sha256"] != _sha256(Path(skeleton.get_filename()))
            or report["artifact_sha256"] != _artifact_hashes(directory)
        ):
            raise ValueError("Analysis report or artifact provenance is stale")
        _validate_spatial_outputs(skeleton, directory, reference_image=reference_image)
        graph = json.loads((directory / "skeleton_graph.json").read_text(encoding="utf-8"))
        data: dict[str, Any] = {"graph": graph, "branches": []}
        for key, filename in (
            ("types", "skeleton_voxel_types.nii.gz"),
            ("components", "skeleton_components.nii.gz"),
        ):
            image = nib.load(directory / filename)
            data[key] = {"image": image, "data": np.asarray(image.dataobj)}
        for component in graph["components"]:
            nodes = {node["node_id"]: np.asarray(node["voxels"]) for node in component["nodes"]}
            for branch in component["branches"]:
                data["branches"].append({
                    **branch, "component_id": component["component_id"],
                    "path": _branch_path(branch, nodes),
                })
        return data
    except Exception as error:
        command = ("run_airway_parse.py" if reference_image == "lung_airways_skeleton.nii.gz"
                   else "run_analyzeskeleton_parse.py")
        raise ValueError(
            f"Invalid or missing analysis outputs in {directory}: {error}. "
            f"Run scripts/{command} --case {directory.name} --overwrite "
            "to export spatial results."
        ) from error


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))

def _aligned(image: nib.Nifti1Image, reference: nib.Nifti1Image, label: str) -> None:
    if (image.header.get_xyzt_units()[0] != "mm"
            or len(image.shape) != 3 or image.shape != reference.shape
            or not np.allclose(image.affine, reference.affine, rtol=0, atol=1e-5)):
        raise ValueError(f"{label} must have known mm units and align with the source airway grid")

def _binary(image: nib.Nifti1Image, label: str) -> np.ndarray:
    values = np.asanyarray(image.dataobj)
    if not np.isfinite(values).all() or not np.isin(values, (0, 1, 255)).all():
        raise ValueError(f"{label} must be finite and binary")
    return values != 0

def verify_saved_graph(root: Path, case: str, input_record: dict) -> dict:
    """Verify final graph, mask and skeleton identities without site selection."""
    root = Path(root).resolve()
    directory = root / "airway" / case
    source_report = directory / "airway_report.json"
    saved = _read(source_report)
    if saved.get("case") != case or saved.get("status") not in ("ok", "skipped"):
        raise ValueError("A completed matching airway report is required")
    hashes = dict(input_record.get("source_sha256", {}))
    for name, digest in hashes.items():
        if _sha256(Path(name)) != digest:
            raise ValueError(f"Changed source hash: {name}")
    mask_path = Path(input_record["mask"]).resolve()
    skeleton_path = directory / "lung_airways_skeleton.nii.gz"
    if mask_path != Path(saved["mask"]).resolve():
        raise ValueError("Airway source mask differs from the saved airway analysis scope")
    if input_record.get("skeleton_path") and Path(input_record["skeleton_path"]).resolve() != skeleton_path:
        raise ValueError("Airway source skeleton differs from the saved airway analysis")
    skeleton_stage = saved["stages"]["skeletonize3d"]
    analysis_stage = saved["stages"]["analyzeskeleton"]
    expected = {mask_path: saved["mask_sha256"], skeleton_path: skeleton_stage["output_sha256"]}
    if skeleton_stage["input_sha256"] != saved["mask_sha256"]:
        raise ValueError("Airway skeleton was not created from the selected lumen mask")
    if analysis_stage["provenance"]["input_sha256"] != skeleton_stage["output_sha256"]:
        raise ValueError("Airway graph was not created from the saved airway skeleton")
    for name, digest in analysis_stage["artifact_sha256"].items():
        expected[directory / name] = digest
    for path, digest in expected.items():
        if _sha256(path) != digest:
            raise ValueError(f"Changed airway graph/skeleton/mask hash: {path}")
        hashes[str(path.resolve())] = digest
    for path in (source_report, directory / "analysis_report.json", directory / "skeleton_report.json"):
        hashes[str(path.resolve())] = _sha256(path)
    skeleton, mask_image = nib.load(skeleton_path), nib.load(mask_path)
    if skeleton.header.get_xyzt_units()[0] != "mm":
        raise ValueError("Airway skeleton requires known mm units")
    _aligned(mask_image, skeleton, "Airway lumen")
    lumen = _binary(mask_image, "Airway lumen")
    analysis = _load_analysis(skeleton, directory, reference_image="lung_airways_skeleton.nii.gz")
    graph = analysis["graph"]
    if graph["coordinate_system"]["reference_image"] != "lung_airways_skeleton.nii.gz":
        raise ValueError("Airway graph measurements cannot use an arterial graph")
    return {"case": case, "source_sha256": hashes, "graph": graph}
