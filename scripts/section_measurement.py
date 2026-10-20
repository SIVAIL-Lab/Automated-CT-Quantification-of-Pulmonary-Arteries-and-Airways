"""Measure orthogonal sections of one calibrated, saved centerline tree."""
from dataclasses import asdict
from pathlib import Path

import nibabel as nib
import numpy as np

from centerline_cross_sections import Config, measure_tree_cross_sections
from edt_tree_radius import measure_tree_radius
from measurement_io import aligned_image, write_csv, write_json
from run_edt_experiment import binary_mask, geometry_mm, sha256


def _hash_files(directory: Path) -> dict:
    return {str(path): sha256(path) for path in sorted(directory.rglob("*")) if path.is_file()}


def _measure_source(name: str, source: dict, tree: dict, review: dict,
                    output: Path, config: Config, progress) -> dict:
    image = nib.load(source["mask_path"])
    spacing, affine = geometry_mm(image)
    for geometry in (source, tree):
        if (list(image.shape) != geometry["shape"]
                or not np.allclose(affine, geometry["affine_mm"], atol=1e-5, rtol=0)
                or not np.allclose(spacing, geometry["spacing_mm"], atol=1e-6, rtol=0)):
            raise ValueError(f"{name}: saved mask and tree geometry disagree")
    mask = binary_mask(np.asarray(image.dataobj))
    edt_image = aligned_image(Path(source["edt_path"]), image, "EDT")
    tree_image = aligned_image(Path(source["tree_mask_path"]), image, "Tree mask")
    tree_mask = binary_mask(np.asarray(tree_image.dataobj))
    points = np.asarray([voxel["ijk"] for voxel in tree["voxels"]], dtype=int)
    if (np.any(points < 0) or np.any(points >= np.asarray(mask.shape))
            or np.count_nonzero(tree_mask) != len(points)
            or not np.all(tree_mask[tuple(points.T)])):
        raise ValueError(f"{name}: saved tree mask differs from the original tree voxels")
    # This audits the saved tree and EDT; it does not extract or modify either.
    original = measure_tree_radius(tree, np.asarray(edt_image.dataobj), mask, affine)
    del edt_image, tree_image, tree_mask
    measured = measure_tree_cross_sections(tree, mask, affine, config, progress=progress)
    old_segments = {row["branch_id"]: row for row in original["segments"]}
    if {row["branch_id"] for row in measured["segments"]} != set(old_segments):
        raise ValueError(f"{name}: cross-section analysis changes the segment identities")
    for row in measured["segments"]:
        old = old_segments[row["branch_id"]]
        if (not np.isclose(row["length_mm"], old["length_mm"], rtol=1e-12, atol=1e-8)
                or not np.isclose(row["edt_segment_radius_mm"], old["segment_radius_mm"], rtol=1e-12, atol=1e-8)):
            raise ValueError(f"{name}: cross-section analysis changes saved segment geometry or EDT radius")
    if (measured["summary"]["segment_count"] != tree["summary"]["segment_count"]
            or not np.isclose(measured["summary"]["total_length_mm"],
                              tree["summary"]["unique_length_mm"], rtol=1e-12, atol=1e-8)):
        raise ValueError(f"{name}: cross-section analysis changes the tree length or segment count")
    for row in measured["samples"]:
        row["source"] = name
    for row in measured["segments"]:
        row["source"] = name
    output.mkdir()
    samples_path, segments_path = output / "cross_section_samples.csv", output / "segment_coverage.csv"
    measurement_path = output / "measurements.json"
    write_csv(samples_path, measured["samples"])
    write_csv(segments_path, measured["segments"])
    write_json(measurement_path, measured)
    result = {
        "source": name, "status": "ok", "technical_validation": "passed", "summary": measured["summary"],
        "samples_path": str(samples_path), "segments_path": str(segments_path),
        "measurement_path": str(measurement_path), "parameters": asdict(config),
        **{key: source[key] for key in ("tree_path", "tree_mask_path", "mask_path", "edt_path",
                                       "spacing_mm", "affine_mm", "shape",
                                       "segmentation_provenance_status", "model_input_verified")},
        "visual_review": review["status"], "review_observations": review.get("observations", []),
        "anatomical_validation": "not_performed", "inherited_tree_flag_counts": tree["summary"]["flag_counts"],
        "checks": {"segment_ids_preserved": True, "geometry_and_length_preserved": True,
                   "saved_edt_verified": True, "tree_mask_matches": True},
        "artifact_sha256": _hash_files(output),
    }
    write_json(output / "summary.json", result)
    result["artifact_sha256"][str(output / "summary.json")] = sha256(output / "summary.json")
    return result
