#!/usr/bin/env python3
"""Skeletonize TotalSegmentator pulmonary-artery masks with Fiji.

Skeletonize3D is an ImageJ plugin and accepts 8-bit binary image stacks. This
wrapper converts each NIfTI mask to a temporary TIFF stack, invokes the Fiji
installation in headless mode, and converts the result back to NIfTI while
preserving the input shape and affine.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import tifffile

from dataset_cases import discover_mask_cases


TARGET_MASK = "lung_arteries.nii.gz"
OUTPUT_MASK = "lung_arteries_skeleton.nii.gz"
FIJI_TIMEOUT = 600


def _sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _load_binary_image(path: Path) -> nib.Nifti1Image:
    image = nib.load(str(path))
    if len(image.shape) != 3:
        raise ValueError(f"Expected a 3-D mask: {path}")
    values = np.unique(np.asarray(image.dataobj))
    if not (np.all(np.isin(values, (0, 1))) or np.all(np.isin(values, (0, 255)))):
        raise ValueError(f"Expected a finite binary mask (0/1 or 0/255): {path}")
    return image


def _spacing_mm(image: nib.Nifti1Image) -> tuple[float, float, float]:
    unit = image.header.get_xyzt_units()[0]
    scale = {"mm": 1.0, "meter": 1000.0, "micron": 0.001}.get(unit)
    if scale is None:
        raise ValueError(f"A known NIfTI spatial unit is required; got {unit!r}")
    spacing = np.asarray(image.header.get_zooms()[:3], dtype=float)
    if not np.all(np.isfinite(spacing)) or np.any(spacing <= 0):
        raise ValueError("Voxel spacing must be finite and positive")
    # ImageJ calibration can represent unequal spacing and rotation, but not shear.
    directions = image.affine[:3, :3] / spacing
    if not np.all(np.isfinite(image.affine)) or not np.allclose(
        directions.T @ directions, np.eye(3), rtol=1e-5, atol=1e-5
    ):
        raise ValueError("NIfTI affine must have orthogonal voxel axes matching spacing")
    return tuple(float(value) for value in spacing * scale)


def _validate_skeleton(reference: nib.Nifti1Image, skeleton: nib.Nifti1Image) -> int:
    if reference.shape != skeleton.shape or not np.allclose(
        reference.affine, skeleton.affine, rtol=1e-5, atol=1e-3
    ):
        raise ValueError("Skeleton shape or affine differs from its input mask")
    if not np.allclose(_spacing_mm(reference), _spacing_mm(skeleton)):
        raise ValueError("Skeleton physical spacing differs from its input mask")
    mask = np.asarray(reference.dataobj) > 0
    points = np.asarray(skeleton.dataobj) > 0
    if np.any(points & ~mask):
        raise ValueError("Skeleton contains voxels outside its input mask")
    count = int(np.count_nonzero(points))
    if np.any(mask) and count == 0:
        raise ValueError("Nonempty mask produced an empty skeleton")
    return count


def _default_fiji_candidates() -> list[Path]:
    return [
        Path.home() / ".local/share/Fiji.app/ImageJ-linux64",
        Path("/opt/Fiji.app/ImageJ-linux64"),
        Path("/usr/local/Fiji.app/ImageJ-linux64"),
    ]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Skeletonize TotalSegmentator lung-artery masks with Fiji Skeletonize3D."
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("data/parse/results/totalsegmentator"),
        help="Directory containing case outputs (default: data/parse/results/totalsegmentator).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/parse/results/skeletonize3d"),
        help="Directory for skeleton NIfTI files and reports (default: data/parse/results/skeletonize3d).",
    )
    parser.add_argument(
        "--fiji",
        type=Path,
        help="Path to Fiji ImageJ-linux64; otherwise FIJI_BIN, standard installations and PATH are checked.",
    )
    parser.add_argument(
        "--case",
        dest="cases",
        action="append",
        help="Case ID to process; repeat for multiple cases (default: all matching mask outputs).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Process at most this many cases after case selection.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rerun Skeletonize3D even when the skeleton output already exists.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress successful case messages.",
    )
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be a positive integer")
    return args


def _find_fiji(requested: Path | None) -> Path:
    candidates: list[Path] = []
    configured = os.environ.get("FIJI_BIN")
    explicit = requested if requested is not None else Path(configured) if configured else None
    if explicit is not None:
        explicit = explicit.expanduser().resolve()
        if not explicit.is_file() or not os.access(explicit, os.X_OK):
            raise FileNotFoundError(f"Configured Fiji executable is not executable: {explicit}")
        return explicit
    candidates.extend(_default_fiji_candidates())
    path_from_path = shutil.which("ImageJ-linux64")
    if path_from_path:
        candidates.append(Path(path_from_path))

    seen: set[Path] = set()
    for candidate in candidates:
        candidate = candidate.expanduser()
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate.resolve()
    searched = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Fiji ImageJ-linux64 not found; searched: {searched}")


def _discover_cases(
    input_root: Path, selected: list[str] | None, limit: int | None,
    *, mask_name: str = TARGET_MASK,
) -> list[Path]:
    return discover_mask_cases(input_root, selected, limit, mask_name=mask_name)


def _write_macro(input_tiff: Path, output_tiff: Path, directory: Path) -> Path:
    # json.dumps supplies ImageJ macro string escaping for paths with spaces.
    macro = directory / "skeletonize3d.ijm"
    macro.write_text(
        "\n".join(
            [
                "setBatchMode(true);",
                f"open({json.dumps(str(input_tiff))});",
                'run("8-bit");',
                'run("Skeletonize (2D/3D)");',
                f"saveAs(\"Tiff\", {json.dumps(str(output_tiff))});",
                "close();",
                "setBatchMode(false);",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return macro


def _convert_to_tiff(image: nib.Nifti1Image, path: Path) -> int:
    data = np.asarray(image.dataobj) > 0
    # ImageJ interprets the first dimension of a stack as Z. Keep X/Y/Z
    # orientation recoverable by transposing back after Skeletonize3D.
    stack = (data.transpose(2, 1, 0).astype(np.uint8) * 255)
    sx, sy, sz = _spacing_mm(image)
    tifffile.imwrite(
        path, stack, imagej=True, resolution=(1 / sx, 1 / sy),
        metadata={"axes": "ZYX", "spacing": sz, "unit": "mm"},
    )
    return int(np.count_nonzero(data))


def _save_skeleton_nifti(
    input_image: nib.Nifti1Image, output_tiff: Path, output_path: Path
) -> int:
    stack = np.asarray(tifffile.imread(output_tiff))
    expected_shape = (input_image.shape[2], input_image.shape[1], input_image.shape[0])
    if stack.ndim == 2 and expected_shape[0] == 1:
        stack = stack[None, ...]
    if stack.shape != expected_shape:
        raise ValueError(f"Fiji output shape {stack.shape} != expected stack shape {expected_shape}")
    skeleton = (stack > 0).astype(np.uint8).transpose(2, 1, 0)
    header = input_image.header.copy()
    header.set_data_dtype(np.uint8)
    result = nib.Nifti1Image(skeleton, input_image.affine, header)
    result.set_qform(*input_image.get_qform(coded=True))
    result.set_sform(*input_image.get_sform(coded=True))
    count = _validate_skeleton(input_image, result)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(result, str(output_path))
    return count


def _run_case(
    case_dir: Path, output_root: Path, fiji: Path, overwrite: bool,
    *, input_path: Path | None = None, output_name: str = OUTPUT_MASK,
) -> dict[str, Any]:
    """Skeletonize the default prediction or a named, explicitly supplied mask."""
    started = time.perf_counter()
    case = case_dir.name
    explicit_input = input_path is not None
    input_path = Path(input_path) if explicit_input else case_dir / TARGET_MASK
    output_path = output_root / case / output_name
    report_path = output_path.parent / "skeleton_report.json"
    row: dict[str, Any] = {
        "case": case,
        "status": "error",
        "input": str(input_path),
        "output": str(output_path),
        "fiji": str(fiji),
        "input_voxels": None,
        "skeleton_voxels": None,
        "runtime_sec": None,
        "error": None,
        "report": str(report_path),
    }
    try:
        if not output_name or Path(output_name).name != output_name or output_name in {".", ".."}:
            raise ValueError("Skeleton output name must be a filename without directories")
        if not input_path.exists():
            raise FileNotFoundError(f"Missing input mask: {input_path}")
        input_image = _load_binary_image(input_path)
        row["spacing_mm"] = list(_spacing_mm(input_image))
        row["input_sha256"] = _sha256(input_path)
        row["input_voxels"] = int(np.count_nonzero(np.asarray(input_image.dataobj)))
        plugins = sorted((fiji.parent / "plugins").glob("Skeletonize3D_*.jar"))
        if len(plugins) != 1:
            raise FileNotFoundError("Expected exactly one Skeletonize3D plugin in Fiji/plugins")
        row["plugin"] = plugins[0].name
        row["plugin_sha256"] = _sha256(plugins[0])
        if output_path.exists() and not overwrite:
            try:
                skeleton_image = _load_binary_image(output_path)
                count = _validate_skeleton(input_image, skeleton_image)
                output_hash = _sha256(output_path)
                cached = json.loads(report_path.read_text()) if report_path.exists() else None
                if (cached is None and not explicit_input and output_name == OUTPUT_MASK) or (
                    cached is not None
                    and cached.get("status") in {"ok", "skipped"}
                    and cached.get("input_sha256") == row["input_sha256"]
                    and cached.get("plugin_sha256") == row["plugin_sha256"]
                    and cached.get("output_sha256") == output_hash
                ):
                    row.update(status="skipped", skeleton_voxels=count, output_sha256=output_hash)
                    return row
            except (ValueError, TypeError, AttributeError, OSError, nib.filebasedimages.ImageFileError):
                pass  # Recompute incomplete, stale or invalid cached skeletons.

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".skeletonize3d_", dir=output_path.parent.resolve()) as temp_dir_name:
            temp_dir = Path(temp_dir_name)
            input_tiff = temp_dir / "input.tif"
            output_tiff = temp_dir / "output.tif"
            input_voxels = _convert_to_tiff(input_image, input_tiff)
            macro = _write_macro(input_tiff, output_tiff, temp_dir)
            completed = subprocess.run(
                [str(fiji), "--headless", "--console", "-macro", str(macro)],
                check=False,
                capture_output=True,
                text=True,
                timeout=FIJI_TIMEOUT,
            )
            if completed.returncode != 0:
                details = (completed.stderr or completed.stdout).strip().splitlines()
                tail = " | ".join(details[-5:])
                raise RuntimeError(f"Fiji exited with code {completed.returncode}: {tail}")
            if not output_tiff.exists():
                raise FileNotFoundError("Fiji completed without creating output.tif")
            candidate = temp_dir / output_name
            skeleton_voxels = _save_skeleton_nifti(input_image, output_tiff, candidate)
            if _sha256(input_path) != row["input_sha256"]:
                raise ValueError("Input mask changed during skeletonization")
            candidate.replace(output_path)

        row["input_voxels"] = input_voxels
        row["skeleton_voxels"] = skeleton_voxels
        row["output_sha256"] = _sha256(output_path)
        row["status"] = "ok"
    except Exception as error:
        row["error"] = f"{type(error).__name__}: {error}"
    finally:
        row["runtime_sec"] = round(time.perf_counter() - started, 3)
        if row["status"] in {"ok", "skipped"}:
            try:
                report_path.write_text(json.dumps(row, indent=2) + "\n", encoding="utf-8")
            except OSError as error:
                row.update(status="error", error=f"Cannot save skeleton report: {error}")
    return row


def _write_summary(output_root: Path, rows: list[dict[str, Any]]) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    summary = {
        "plugin": "Skeletonize3D",
        "target_mask": TARGET_MASK,
        "output_mask": OUTPUT_MASK,
        "total_cases": len(rows),
        "successful_cases": sum(row["status"] in {"ok", "skipped"} for row in rows),
        "failed_cases": sum(row["status"] == "error" for row in rows),
        "cases": rows,
    }
    (output_root / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )


def main() -> int:
    args = _parse_args()
    try:
        fiji = _find_fiji(args.fiji)
        cases = _discover_cases(args.input_root, args.cases, args.limit)
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    rows = []
    for case in cases:
        row = _run_case(case, args.output_root, fiji, args.overwrite)
        rows.append(row)
        if row["status"] == "error":
            print(f"{case.name}: error: {row['error']}", file=sys.stderr)
        elif not args.quiet:
            print(
                f"{case.name}: {row['status']} "
                f"({row['skeleton_voxels']} skeleton voxels, {row['runtime_sec']:.3f}s)"
            )

    _write_summary(args.output_root, rows)
    return 1 if any(row["status"] == "error" for row in rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())
