#!/usr/bin/env python3
"""Measure pulmonary-artery skeleton components and branches with Fiji."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import tifffile

from run_skeletonize3d_parse import (
    _convert_to_tiff, _discover_cases, _find_fiji, _load_binary_image, _spacing_mm,
)


TARGET_MASK = "lung_arteries_skeleton.nii.gz"
BRIDGE = Path(__file__).resolve().parent / "fiji/analyze_skeleton.groovy"
TIMEOUT_SEC = 600
PARAMETERS = {"prune_cycles": False, "prune_ends": False, "shortest_paths": False}
SCHEMA_VERSION = 2
VOXEL_TYPES = "skeleton_voxel_types.nii.gz"
COMPONENT_LABELS = "skeleton_components.nii.gz"
GRAPH = "skeleton_graph.json"
ARTIFACTS = ("branches.csv", "components.csv", VOXEL_TYPES, COMPONENT_LABELS, GRAPH)
COORDINATE_SYSTEM = {
    "space": "voxel", "axes": "XYZ", "index_base": 0, "reference_image": TARGET_MASK,
}
BRANCH_FIELDS = ["component_id", "branch_id", "length_mm"]
COMPONENT_FIELDS = [
    "component_id", "branch_count", "endpoint_count", "junction_count",
    "junction_voxel_count", "mean_branch_length_mm", "max_branch_length_mm",
]
METRIC_FIELDS = [
    "component_count", "branch_count", "endpoint_count", "junction_count",
    "junction_voxel_count", "total_branch_length_mm", "mean_branch_length_mm",
    "max_branch_length_mm",
]


def _sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _plugin_identity(fiji: Path) -> dict[str, str]:
    plugins = sorted((fiji.resolve().parent / "plugins").glob("AnalyzeSkeleton_*.jar"))
    if len(plugins) != 1:
        raise FileNotFoundError(
            f"Expected one AnalyzeSkeleton plugin in {fiji.parent / 'plugins'}, found {len(plugins)}"
        )
    plugin = plugins[0]
    return {"jar": plugin.name, "sha256": _sha256(plugin)}


def _load_skeleton(path: Path) -> tuple[nib.Nifti1Image, int, list[float]]:
    image = _load_binary_image(path)
    return image, int(np.count_nonzero(np.asarray(image.dataobj))), list(_spacing_mm(image))


def _read_csv(path: Path, fields: list[str]) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != fields:
            raise ValueError(f"Unexpected columns in {path.name}")
        rows = list(reader)
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ValueError(f"Malformed rows in {path.name}")
    return rows


def _nonnegative_int(value: str) -> int:
    result = int(value)
    if result < 0:
        raise ValueError("Counts must be nonnegative")
    return result


def _length(value: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError("Branch lengths must be finite and nonnegative")
    return result


def _table_metrics(branches_path: Path, components_path: Path) -> dict[str, Any]:
    branches = _read_csv(branches_path, BRANCH_FIELDS)
    components = _read_csv(components_path, COMPONENT_FIELDS)
    component_ids = [int(component["component_id"]) for component in components]
    if component_ids != list(range(1, len(components) + 1)):
        raise ValueError("Component IDs must be consecutive and start at 1")
    lengths_by_component: dict[int, list[float]] = defaultdict(list)
    for branch in branches:
        component_id = int(branch["component_id"])
        if component_id not in component_ids:
            raise ValueError("Branch references an unknown component")
        lengths = lengths_by_component[component_id]
        if int(branch["branch_id"]) != len(lengths) + 1:
            raise ValueError("Branch IDs must be consecutive within each component")
        lengths.append(_length(branch["length_mm"]))
    totals = {field: 0 for field in (
        "branch_count", "endpoint_count", "junction_count", "junction_voxel_count"
    )}
    for component in components:
        lengths = lengths_by_component[int(component["component_id"])]
        for field in totals:
            totals[field] += _nonnegative_int(component[field])
        if int(component["branch_count"]) != len(lengths):
            raise ValueError("Component branch count does not match branch table")
        mean = math.fsum(lengths) / len(lengths) if lengths else 0.0
        maximum = max(lengths, default=0.0)
        for field, expected in (
            ("mean_branch_length_mm", mean), ("max_branch_length_mm", maximum)
        ):
            if not math.isclose(_length(component[field]), expected, rel_tol=1e-8, abs_tol=1e-8):
                raise ValueError(f"Inconsistent {field} in component table")
    all_lengths = [value for lengths in lengths_by_component.values() for value in lengths]
    total_length = math.fsum(all_lengths)
    return {
        "component_count": len(components),
        **totals,
        "total_branch_length_mm": total_length,
        "mean_branch_length_mm": total_length / len(all_lengths) if all_lengths else 0.0,
        "max_branch_length_mm": max(all_lengths, default=0.0),
    }


def _write_empty_tables(branches: Path, components: Path) -> None:
    for path, fields in ((branches, BRANCH_FIELDS), (components, COMPONENT_FIELDS)):
        with path.open("w", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerow(fields)


def _save_labels(image: nib.Nifti1Image, labels: np.ndarray, path: Path) -> None:
    """Keep categorical values and the original NIfTI grid and spatial forms."""
    if labels.shape != image.shape:
        raise ValueError("Label image shape does not match the input skeleton")
    header = image.header.copy()
    header.set_data_dtype(labels.dtype)
    result = nib.Nifti1Image(labels, image.affine, header)
    result.set_qform(*image.get_qform(coded=True))
    result.set_sform(*image.get_sform(coded=True))
    nib.save(result, path)


def _labels_from_tiff(
    image: nib.Nifti1Image, source: Path, destination: Path, dtype: type,
) -> None:
    stack = np.asarray(tifffile.imread(source))
    if stack.ndim == 2 and image.shape[2] == 1:
        stack = stack[None, ...]
    if stack.shape != image.shape[::-1]:
        raise ValueError("Fiji label stack dimensions do not match the input skeleton")
    if (
        not np.all(np.isfinite(stack)) or np.any(stack < 0)
        or np.any(stack > np.iinfo(dtype).max) or np.any(stack != np.floor(stack))
    ):
        raise ValueError("Fiji label stack must contain representable nonnegative integers")
    _save_labels(image, stack.transpose(2, 1, 0).astype(dtype), destination)


def _empty_spatial_outputs(
    image: nib.Nifti1Image, directory: Path, reference_image: str = TARGET_MASK,
) -> None:
    _save_labels(image, np.zeros(image.shape, dtype=np.uint8), directory / VOXEL_TYPES)
    _save_labels(image, np.zeros(image.shape, dtype=np.uint32), directory / COMPONENT_LABELS)
    (directory / GRAPH).write_text(json.dumps({
        "schema_version": SCHEMA_VERSION,
        "coordinate_system": {**COORDINATE_SYSTEM, "reference_image": reference_image},
        "shape": list(image.shape), "affine": image.affine.tolist(), "components": [],
    }) + "\n", encoding="utf-8")


def _validate_spatial_outputs(
    image: nib.Nifti1Image, directory: Path, reference_image: str = TARGET_MASK,
) -> None:
    """Cross-check exported geometry, voxel labels, graph membership, and CSV IDs."""
    skeleton = np.asarray(image.dataobj) > 0
    skeleton_voxels = int(np.count_nonzero(skeleton))
    arrays = []
    for name, dtype in ((VOXEL_TYPES, np.uint8), (COMPONENT_LABELS, np.uint32)):
        output = nib.load(directory / name)
        if (
            output.shape != image.shape or output.get_data_dtype() != np.dtype(dtype)
            or not np.allclose(output.affine, image.affine, rtol=0, atol=1e-5)
        ):
            raise ValueError(f"Incorrect geometry or dtype in {name}")
        for form in ("qform", "sform"):
            expected, expected_code = getattr(image, f"get_{form}")(coded=True)
            actual, actual_code = getattr(output, f"get_{form}")(coded=True)
            if expected_code != actual_code or (
                expected_code and not np.allclose(expected, actual, rtol=0, atol=1e-5)
            ):
                raise ValueError(f"Incorrect {form} in {name}")
        data = np.asarray(output.dataobj)
        if np.count_nonzero(data) != skeleton_voxels or np.any(data[skeleton] == 0):
            raise ValueError(f"Foreground in {name} does not match the skeleton")
        arrays.append(data)
    voxel_types, labels = arrays
    if not np.all(np.isin(voxel_types[skeleton], [30, 70, 127])):
        raise ValueError("Unexpected skeleton voxel type; labels must retain plugin values")
    components = _read_csv(directory / "components.csv", COMPONENT_FIELDS)
    csv_branches = _read_csv(directory / "branches.csv", BRANCH_FIELDS)
    graph = json.loads((directory / GRAPH).read_text(encoding="utf-8"))
    if (
        graph["schema_version"] != SCHEMA_VERSION
        or graph["coordinate_system"] != {**COORDINATE_SYSTEM, "reference_image": reference_image}
        or graph["shape"] != list(image.shape)
        or np.asarray(graph["affine"]).shape != (4, 4)
        or not np.allclose(graph["affine"], image.affine, rtol=0, atol=1e-5)
    ):
        raise ValueError("Graph coordinate metadata does not match the input skeleton")
    expected_ids = list(range(1, len(components) + 1))
    if (
        np.unique(labels[skeleton]).tolist() != expected_ids
        or [item["component_id"] for item in graph["components"]] != expected_ids
    ):
        raise ValueError("Graph, component labels, and CSV component IDs disagree")
    covered = set()
    graph_branches = []
    foreground_types = voxel_types[skeleton]
    foreground_components = labels[skeleton]
    for component, table in zip(graph["components"], components):
        component_id = component["component_id"]
        nodes = component["nodes"]
        node_ids = [node["node_id"] for node in nodes]
        if node_ids != list(range(1, len(nodes) + 1)) or not nodes:
            raise ValueError("Graph node IDs must be consecutive and start at 1")

        def check_points(points: list, allow_empty: bool = False) -> np.ndarray:
            if not points:
                if allow_empty:
                    return np.empty((0, 3), dtype=int)
                raise ValueError("Graph nodes must contain voxel coordinates")
            coords = np.asarray(points)
            if (
                coords.ndim != 2 or coords.shape[1] != 3 or coords.dtype.kind not in "iu"
                or np.any(coords < 0) or np.any(coords >= np.asarray(image.shape))
            ):
                raise ValueError("Graph coordinates must be in-bounds integer XYZ indices")
            if np.any(labels[tuple(coords.T)] != component_id):
                raise ValueError("Graph coordinates lie outside their labeled component")
            covered.update(map(tuple, coords.tolist()))
            return coords

        for node in nodes:
            check_points(node["voxels"])
        for branch in component["branches"]:
            if branch["v1"] not in node_ids or branch["v2"] not in node_ids:
                raise ValueError("Graph branch refers to an unknown node")
            slabs = check_points(branch["slabs"], allow_empty=True)
            if len(slabs) > 1 and np.any(np.max(np.abs(np.diff(slabs, axis=0)), axis=1) != 1):
                raise ValueError("Graph slab coordinates must follow a continuous voxel path")
            graph_branches.append((component_id, branch["branch_id"], _length(branch["length_mm"])))
        types = foreground_types[foreground_components == component_id]
        if (
            int(np.count_nonzero(types == 30)) != int(table["endpoint_count"])
            or int(np.count_nonzero(types == 70)) != int(table["junction_voxel_count"])
        ):
            raise ValueError("Voxel classifications disagree with component statistics")
    if len(covered) != skeleton_voxels:
        raise ValueError("Exported graph does not cover the input skeleton")
    expected_branches = [
        (int(row["component_id"]), int(row["branch_id"]), float(row["length_mm"]))
        for row in csv_branches
    ]
    if graph_branches != expected_branches:
        raise ValueError("Graph branch IDs or lengths disagree with the CSV")


def _artifact_hashes(directory: Path) -> dict[str, str]:
    return {name: _sha256(directory / name) for name in ARTIFACTS}


def _cached_metrics(
    report_path: Path, provenance: dict[str, Any], image: nib.Nifti1Image, skeleton_voxels: int,
    reference_image: str = TARGET_MASK,
) -> dict[str, Any] | None:
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if (
            report["status"] not in {"ok", "skipped"}
            or report["provenance"] != provenance
            or report["skeleton_voxels"] != skeleton_voxels
            or report["artifact_sha256"] != _artifact_hashes(report_path.parent)
        ):
            return None
        metrics = _table_metrics(report_path.parent / "branches.csv", report_path.parent / "components.csv")
        _validate_spatial_outputs(image, report_path.parent, reference_image=reference_image)
        if any(report[field] != value for field, value in metrics.items()):
            return None
        return metrics
    except (OSError, ValueError, KeyError, TypeError, IndexError, nib.filebasedimages.ImageFileError):
        return None


def _run_case(
    case_dir: Path, output_root: Path, fiji: Path, overwrite: bool,
    *, input_path: Path | None = None,
) -> dict[str, Any]:
    """Analyze the default artery skeleton or an explicitly supplied skeleton."""
    started = time.perf_counter()
    input_path = (Path(input_path) if input_path is not None else case_dir / TARGET_MASK).resolve()
    reference_image = input_path.name
    output_dir = output_root / case_dir.name
    branches = output_dir / "branches.csv"
    components = output_dir / "components.csv"
    report_path = output_dir / "analysis_report.json"
    row: dict[str, Any] = {
        "case": case_dir.name, "status": "error", "error": None,
        "input": str(input_path), "fiji": str(fiji), "runtime_sec": None,
        "skeleton_voxels": None, **{field: None for field in METRIC_FIELDS},
        "branches": str(branches), "components": str(components),
        "voxel_types": str(output_dir / VOXEL_TYPES),
        "component_labels": str(output_dir / COMPONENT_LABELS),
        "graph": str(output_dir / GRAPH),
        "report": str(report_path),
    }
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        image, skeleton_voxels, spacing_mm = _load_skeleton(input_path)
        provenance = {
            "input_sha256": _sha256(input_path),
            "plugin": _plugin_identity(fiji),
            "bridge_sha256": _sha256(BRIDGE),
            "spacing_mm": spacing_mm, "parameters": PARAMETERS,
            "schema_version": SCHEMA_VERSION,
        }
        row.update(skeleton_voxels=skeleton_voxels, spacing_mm=spacing_mm, provenance=provenance)
        cached = None if overwrite else _cached_metrics(
            report_path, provenance, image, skeleton_voxels, reference_image=reference_image,
        )
        if cached is not None:
            row.update(cached, status="skipped")
        else:
            # Temporary stacks and bridge outputs stay within the case directory.
            with tempfile.TemporaryDirectory(prefix=".analyzeskeleton_", dir=output_dir) as directory:
                temporary = Path(directory).resolve()
                temp_branches = temporary / "branches.csv"
                temp_components = temporary / "components.csv"
                if skeleton_voxels == 0:
                    _write_empty_tables(temp_branches, temp_components)
                    _empty_spatial_outputs(image, temporary, reference_image=reference_image)
                else:
                    input_tiff = temporary / "skeleton.tif"
                    _convert_to_tiff(image, input_tiff)
                    config = temporary / "config.json"
                    completion_path = temporary / "completed.json"
                    types_tiff = temporary / "voxel_types.tif"
                    components_tiff = temporary / "components.tif"
                    config.write_text(json.dumps({
                        "input_tiff": str(input_tiff), "spacing_mm": spacing_mm,
                        "shape": list(image.shape), "affine": image.affine.tolist(),
                        "branches_csv": str(temp_branches), "components_csv": str(temp_components),
                        "voxel_types_tiff": str(types_tiff), "components_tiff": str(components_tiff),
                        "graph_json": str(temporary / GRAPH),
                        "completion_json": str(completion_path),
                    }), encoding="utf-8")
                    environment = os.environ.copy()
                    environment["PARSE_ANALYZE_CONFIG"] = str(config)
                    completed = subprocess.run(
                        [str(fiji.resolve()), "--headless", "--console", "--run", str(BRIDGE)],
                        check=False, capture_output=True, text=True,
                        timeout=TIMEOUT_SEC, env=environment,
                    )
                    if (
                        completed.returncode != 0 or not completion_path.is_file()
                        or not temp_branches.is_file() or not temp_components.is_file()
                        or not types_tiff.is_file() or not components_tiff.is_file()
                        or not (temporary / GRAPH).is_file()
                    ):
                        details = (completed.stderr + "\n" + completed.stdout).strip().splitlines()
                        raise RuntimeError(
                            f"Fiji analysis failed (exit {completed.returncode}): " + " | ".join(details[-12:])
                        )
                    _labels_from_tiff(image, types_tiff, temporary / VOXEL_TYPES, np.uint8)
                    _labels_from_tiff(image, components_tiff, temporary / COMPONENT_LABELS, np.uint32)
                    if reference_image != TARGET_MASK:
                        # The bridge knows only its temporary TIFF. Name the original
                        # NIfTI here without changing the bridge identity and invalidating
                        # existing arterial analysis caches.
                        graph_path = temporary / GRAPH
                        graph = json.loads(graph_path.read_text(encoding="utf-8"))
                        if graph["coordinate_system"] != COORDINATE_SYSTEM:
                            raise ValueError("Unexpected bridge graph coordinate metadata")
                        graph["coordinate_system"] = {
                            **COORDINATE_SYSTEM, "reference_image": reference_image,
                        }
                        graph_path.write_text(json.dumps(graph) + "\n", encoding="utf-8")
                metrics = _table_metrics(temp_branches, temp_components)
                _validate_spatial_outputs(image, temporary, reference_image=reference_image)
                if skeleton_voxels:
                    completion = json.loads(completion_path.read_text(encoding="utf-8"))
                    if completion != {
                        "status": "ok", "component_count": metrics["component_count"],
                        "branch_count": metrics["branch_count"],
                    }:
                        raise ValueError("Fiji completion record does not match the branch tables")
                if skeleton_voxels > 0 and metrics["component_count"] == 0:
                    raise ValueError("Fiji returned no components for a nonempty skeleton")
                if _sha256(input_path) != provenance["input_sha256"]:
                    raise ValueError("Input skeleton changed during analysis")
                for name in ARTIFACTS:
                    (temporary / name).replace(output_dir / name)
                row.update(metrics, status="ok")
        row["artifact_sha256"] = _artifact_hashes(output_dir)
    except Exception as error:
        row["status"] = "error"
        row["error"] = f"{type(error).__name__}: {error}"
    row["runtime_sec"] = round(time.perf_counter() - started, 3)
    try:
        report_path.write_text(json.dumps(row, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    except OSError as error:
        row["status"] = "error"
        row["error"] = f"Could not write analysis report: {error}"
    return row


def _write_summary(output_root: Path, rows: list[dict[str, Any]]) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "summary.json").write_text(json.dumps({
        "plugin": "AnalyzeSkeleton", "target_mask": TARGET_MASK,
        "total_cases": len(rows),
        "successful_cases": sum(row["status"] in {"ok", "skipped"} for row in rows),
        "failed_cases": sum(row["status"] == "error" for row in rows),
        "cases": rows,
    }, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=Path("data/parse/results/skeletonize3d"))
    parser.add_argument("--output-root", type=Path, default=Path("data/parse/results/analyzeskeleton"))
    parser.add_argument("--case", dest="cases", action="append", help="Case ID; repeat for multiple cases.")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--fiji", type=Path, help="Fiji executable; otherwise use FIJI_BIN or common local paths.")
    parser.add_argument("--overwrite", action="store_true", help="Recompute existing branch statistics.")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be a positive integer")
    return args


def main() -> int:
    args = _parse_args()
    try:
        fiji = _find_fiji(args.fiji)
        cases = _discover_cases(args.input_root, args.cases, args.limit, mask_name=TARGET_MASK)
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    rows = []
    for case in cases:
        row = _run_case(case, args.output_root, fiji, args.overwrite)
        rows.append(row)
        if row["status"] == "error":
            print(f"{row['case']}: {row['error']}", file=sys.stderr)
        elif not args.quiet:
            print(f"{row['case']}: {row['status']}, components={row['component_count']}, "
                  f"branches={row['branch_count']}, total_length_mm={row['total_branch_length_mm']:.3f}")
    _write_summary(args.output_root, rows)
    return 1 if any(row["status"] == "error" for row in rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())
