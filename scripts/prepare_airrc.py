#!/usr/bin/env python3
"""Prepare de-identified AirRC cases on the reference annotation grid."""

from __future__ import annotations

import argparse
import gzip
import itertools
import json
import os
import re
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import SimpleITK as sitk

from dataset_conversion import sha256


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_NAME = "airrc_conversion.json"
CONVERSION_NIFTI_MARKER = b"AIRRC converted"
GROUP_DIRECTORIES = {
    "ct": "image", "native_ct": "source_image", "artery": "label",
    "vein": "label_vein", "airway": "label_airway", "airway_wall": "label_airway_wall",
    "airway_outer": "label_airway_outer", "multiclass": "label_multiclass",
}
LABEL_VALUES = {"airway": (1,), "airway_wall": (2,), "airway_outer": (1, 2),
                "artery": (3,), "vein": (4,)}


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False,
                                     encoding="utf-8") as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _geometry(image: nib.spatialimages.SpatialImage) -> dict[str, Any]:
    return {"shape": list(image.shape), "spacing_mm": list(map(float, image.header.get_zooms()[:3])),
            "affine": image.affine.tolist(), "spatial_unit": image.header.get_xyzt_units()[0],
            "qform_code": int(image.header["qform_code"]),
            "sform_code": int(image.header["sform_code"])}


def _safe_output(output_root: Path, protected: tuple[Path, ...]) -> None:
    if output_root.resolve() != output_root or output_root.is_symlink():
        raise ValueError("AirRC output must not contain symlinks")
    for path in protected:
        if path.is_dir() and (output_root.is_relative_to(path) or path.is_relative_to(output_root)):
            raise ValueError("AirRC output must not overlap original input directories")
        if not path.is_dir() and output_root == path:
            raise ValueError("AirRC output must not overwrite an original input")


def _write_nifti(path: Path, image: nib.Nifti1Image) -> None:
    with path.open("wb") as handle:
        with gzip.GzipFile(filename="", mode="wb", fileobj=handle, mtime=0, compresslevel=1) as zipped:
            image.to_file_map({"image": nib.FileHolder(fileobj=zipped)})


def _derived_image(array: np.ndarray, reference: nib.Nifti1Image) -> nib.Nifti1Image:
    image = nib.Nifti1Image(array, reference.affine, reference.header.copy())
    image.set_data_dtype(array.dtype)
    image.header.set_xyzt_units("mm")
    image.header["descrip"] = CONVERSION_NIFTI_MARKER + b"; see airrc_conversion.json"
    image.set_qform(reference.get_qform(), code=int(reference.header["qform_code"]))
    image.set_sform(reference.get_sform(), code=int(reference.header["sform_code"]))
    return image


def _parse_mhd(text: str) -> dict[str, str]:
    result = {}
    for line in text.splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            key, value = line.split("=", 1)
            result[key.strip()] = value.strip()
    if result.get("NDims") != "3" or result.get("ElementType") != "MET_SHORT":
        raise ValueError("Expected an AirRC source CT with 3D int16 voxels")
    raw = result.get("ElementDataFile", "")
    if not raw or Path(raw).name != raw or raw == "LOCAL":
        raise ValueError("Source CT must refer to one adjacent voxel file")
    return result


def _native_geometry_from_mhd(header: dict[str, str]) -> tuple[list[int], np.ndarray]:
    """Read physical directions independently of the converted NIfTI header."""
    shape = [int(value) for value in header["DimSize"].split()]
    spacing = np.array([float(value) for value in header["ElementSpacing"].split()])
    direction = np.array([float(value) for value in header["TransformMatrix"].split()]).reshape(3, 3)
    origin = np.array([float(value) for value in header["Offset"].split()])
    if len(shape) != 3 or any(value <= 0 for value in shape) or spacing.shape != (3,) or origin.shape != (3,):
        raise ValueError("Invalid native CT geometry")
    if not np.isfinite(spacing).all() or np.any(spacing <= 0) or not np.isfinite(origin).all():
        raise ValueError("Invalid native CT spacing or origin")
    if not np.isfinite(direction).all() or not np.allclose(direction.T @ direction, np.eye(3), atol=1e-5):
        raise ValueError("Invalid native CT direction matrix")
    affine = np.eye(4)
    lps_to_ras = np.diag([-1, -1, 1])
    affine[:3, :3] = lps_to_ras @ direction @ np.diag(spacing)
    affine[:3, 3] = lps_to_ras @ origin
    return shape, affine


def extract_annotations(input_root: Path) -> Path:
    """Join the split archive once; preserve the source archive and original labels."""
    input_root = input_root.expanduser().resolve()
    package = input_root / "26878867.zip"
    package_hash = sha256(package)
    destination = input_root / "source_annotations"
    complete = destination / "extraction.json"
    if complete.is_file():
        saved = json.loads(complete.read_text())
        if saved.get("annotation_package_sha256") != package_hash:
            raise ValueError("AirRC annotation archive changed after extraction")
        for relative, digest in saved["extracted_sha256"].items():
            if sha256(destination / relative) != digest:
                raise ValueError("An extracted AirRC annotation source changed")
        return destination
    if destination.exists():
        raise ValueError("Incomplete AirRC annotation extraction exists")
    with tempfile.TemporaryDirectory(dir=input_root, prefix=".airrc_annotations.") as temporary:
        staged = Path(temporary) / "source_annotations"
        staged.mkdir()
        parts = Path(temporary) / "parts"
        parts.mkdir()
        with zipfile.ZipFile(package) as archive:
            expected = {"metadata.xlsx", "labelsTr.zip", *(f"labelsTr.z{i:02d}" for i in range(1, 8))}
            if set(archive.namelist()) != expected:
                raise ValueError("Unexpected AirRC annotation package contents")
            for name in sorted(expected):
                target = staged / name if name == "metadata.xlsx" else parts / name
                with archive.open(name) as source, target.open("wb") as output:
                    shutil.copyfileobj(source, output)
        joined = Path(temporary) / "labels_joined.zip"
        subprocess.run(["zip", "-s", "0", str(parts / "labelsTr.zip"), "--out", str(joined)],
                       check=True, capture_output=True, text=True)
        labels = staged / "labels"
        labels.mkdir()
        with zipfile.ZipFile(joined) as archive:
            label_members = [info for info in archive.infolist() if info.filename.endswith(".nii.gz")]
            if len(label_members) != 254:
                raise ValueError("Expected 254 formal AirRC annotation volumes")
            names = [Path(info.filename).name for info in label_members]
            if len(set(names)) != len(names):
                raise ValueError("Duplicate AirRC annotation volume names")
            for info, name in zip(label_members, names):
                if not re.fullmatch(r"[0-9.]+\.nii\.gz", name):
                    raise ValueError("Invalid annotation series identifier")
                with archive.open(info) as source, (labels / name).open("wb") as output:
                    shutil.copyfileobj(source, output)
        extracted = {str(path.relative_to(staged)): sha256(path)
                     for path in sorted(staged.rglob("*")) if path.is_file()}
        _write_json(staged / "extraction.json", {
            "schema_version": 1, "dataset": "AIRRC", "annotation_package_sha256": package_hash,
            "extracted_sha256": extracted, "label_count": len(label_members),
        })
        if sha256(package) != package_hash:
            raise ValueError("AirRC annotation package changed during extraction")
        staged.replace(destination)
    return destination


def discover_inventory(input_root: Path) -> dict[str, Any]:
    """Build a local-only source mapping; printed summaries omit original identifiers."""
    input_root = input_root.expanduser().resolve()
    annotations = extract_annotations(input_root)
    label_paths = sorted((annotations / "labels").glob("*.nii.gz"))
    scans = {}
    for archive_path in sorted(input_root.glob("subset*.zip")):
        with zipfile.ZipFile(archive_path) as archive:
            for info in archive.infolist():
                if not info.filename.endswith(".mhd"):
                    continue
                uid = Path(info.filename).stem
                if uid in scans:
                    raise ValueError("Duplicate CT series in AirRC source archives")
                header = _parse_mhd(archive.read(info).decode("ascii"))
                raw_member = str(Path(info.filename).parent / header["ElementDataFile"])
                raw_info = archive.getinfo(raw_member)
                scans[uid] = {"ct_archive": str(archive_path), "ct_header_member": info.filename,
                              "ct_voxel_member": raw_member,
                              "ct_archive_entries": {"header_crc32": info.CRC, "voxel_crc32": raw_info.CRC,
                                                     "voxel_size_bytes": raw_info.file_size},
                              "native_spacing_mm": [float(v) for v in header["ElementSpacing"].split()],
                              "native_shape": [int(v) for v in header["DimSize"].split()],
                              "native_direction_lps": [float(v) for v in header["TransformMatrix"].split()],
                              "native_origin_lps_mm": [float(v) for v in header["Offset"].split()]}
    records = []
    for index, label in enumerate(label_paths, 1):
        uid = label.name.removesuffix(".nii.gz")
        if uid not in scans:
            raise ValueError("A formal AirRC annotation has no matching source CT")
        records.append({"case": f"AIRRC{index:06d}", "series_uid": uid, **scans[uid],
                        "label_path": str(label), "reference_geometry": _geometry(nib.load(label))})
    inventory = {"schema_version": 1, "dataset": "AIRRC", "case_id_rule": "Series UID lexicographic order",
                 "annotation_package_sha256": sha256(input_root / "26878867.zip"), "records": records}
    _write_json(input_root / "audit" / "airrc_case_inventory.json", inventory)
    return inventory


def select_pilot_cases(inventory: dict[str, Any]) -> list[str]:
    """Choose orientation and slice-spacing diversity without looking at prediction quality."""
    rows = inventory["records"]
    reflected = [row for row in rows if not np.allclose(row["native_direction_lps"], np.eye(3).ravel())]
    choices = [min(rows, key=lambda row: (row["native_spacing_mm"][2], row["case"])),
               max(rows, key=lambda row: (row["native_spacing_mm"][2], row["case"]))]
    choices.append(reflected[0] if reflected else rows[len(rows) // 2])
    return list(dict.fromkeys(row["case"] for row in choices))


def _extract_ct(record: dict[str, Any], input_root: Path) -> Path:
    directory = input_root / "source_ct" / record["case"]
    directory.mkdir(parents=True, exist_ok=True)
    archive_path = Path(record["ct_archive"])
    with zipfile.ZipFile(archive_path) as archive:
        for key in ("ct_header_member", "ct_voxel_member"):
            member = record[key]
            info = archive.getinfo(member)
            destination = directory / Path(member).name
            if destination.exists():
                import zlib
                crc = 0
                with destination.open("rb") as handle:
                    for block in iter(lambda: handle.read(1024 * 1024), b""):
                        crc = zlib.crc32(block, crc)
                if destination.stat().st_size != info.file_size or crc != info.CRC:
                    raise ValueError("Extracted AirRC CT source differs from its archive entry")
            else:
                with tempfile.NamedTemporaryFile(dir=directory, delete=False) as output:
                    temporary = Path(output.name)
                    try:
                        with archive.open(info) as source:
                            shutil.copyfileobj(source, output)
                    except BaseException:
                        temporary.unlink(missing_ok=True)
                        raise
                temporary.replace(destination)
    return directory / Path(record["ct_header_member"]).name


def convert_pair(case: str, ct_path: Path, label_path: Path, output_root: Path, *,
                 source_package: Path | None = None, source_context: dict[str, Any] | None = None,
                 overwrite: bool = False) -> dict[str, Any]:
    """Convert one native CT and preserve the original annotation without resampling labels."""
    if not re.fullmatch(r"AIRRC\d{6}", case):
        raise ValueError("Invalid de-identified AirRC case ID")
    ct_path, label_path = ct_path.resolve(), label_path.resolve()
    output_root = Path(os.path.abspath(output_root.expanduser()))
    _safe_output(output_root, (ct_path.parent, label_path.parent))
    header = _parse_mhd(ct_path.read_text(encoding="ascii"))
    _native_geometry_from_mhd(header)
    voxel_path = ct_path.parent / header["ElementDataFile"]
    source_files = {"ct_header": str(ct_path), "ct_voxels": str(voxel_path), "multiclass": str(label_path)}
    if source_package is not None:
        source_files["annotation_package"] = str(source_package.resolve())
    case_dir = output_root / case
    parameters = {"ct_resampling": "linear", "ct_output_dtype": "float32", "outside_value_hu": -1024,
                  "analysis_grid": "exact_reference_annotation_grid", "labels_resampled": False,
                  "coordinate_conversion": "SimpleITK LPS to NIfTI RAS", "origin_is_synthetic": False}
    if case_dir.exists() and not overwrite:
        verify_airrc_conversion(case_dir, groups=tuple(GROUP_DIRECTORIES))
        saved = json.loads((case_dir / REPORT_NAME).read_text())
        if saved["source_files"] != source_files or saved["parameters"] != parameters:
            raise ValueError("AirRC conversion inputs or parameters changed; use --overwrite")
        return {"case": case, "status": "skipped", "report": str(case_dir / REPORT_NAME)}
    if case_dir.is_symlink() or case_dir.resolve() != case_dir:
        raise ValueError("AirRC case output must not contain symlinks")
    if case_dir.exists() and any(path.is_symlink() for path in case_dir.rglob("*")):
        raise ValueError("AirRC case output must not contain symlinks")
    source_hashes = {path: sha256(Path(path)) for path in source_files.values()}
    reference = nib.load(label_path)
    if len(reference.shape) != 3 or reference.header.get_xyzt_units()[0] != "mm":
        raise ValueError("AirRC reference must be a 3D NIfTI in millimeters")
    qform, qcode = reference.get_qform(coded=True)
    sform, scode = reference.get_sform(coded=True)
    if not qcode or not scode or not np.allclose(qform, sform, atol=1e-4, rtol=1e-6):
        raise ValueError("AirRC reference requires consistent real-world qform and sform")
    labels = np.asarray(reference.dataobj)
    if labels.dtype.kind not in "ui" or not np.isin(labels, range(5)).all():
        raise ValueError("AirRC reference must preserve labels 0, 1, 2, 3, 4")
    native = sitk.ReadImage(str(ct_path))
    reference_sitk = sitk.ReadImage(str(label_path))
    native_shape = np.array(native.GetSize())
    corners = np.array(list(itertools.product(*[(0, n - 1) for n in reference.shape])))
    physical_ras = nib.affines.apply_affine(reference.affine, corners)
    continuous = np.array([native.TransformPhysicalPointToContinuousIndex(tuple(p * [-1, -1, 1]))
                           for p in physical_ras])
    outside = np.maximum(np.maximum(-continuous, continuous - (native_shape - 1)), 0)
    max_outside_mm = float(np.max(outside * np.array(native.GetSpacing())))
    if max_outside_mm > max(3.0, 2 * max(native.GetSpacing())):
        raise ValueError("AirRC reference grid is not physically aligned with the native CT")
    resampled = sitk.Resample(native, reference_sitk, sitk.Transform(), sitk.sitkLinear,
                             -1024.0, sitk.sitkFloat32)
    values = sitk.GetArrayFromImage(resampled).transpose(2, 1, 0)
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite intensities after CT resampling")
    output_root.mkdir(parents=True, exist_ok=True)
    outputs = {group: str(case_dir / relative / f"{case}.nii.gz")
               for group, relative in GROUP_DIRECTORIES.items()}
    with tempfile.TemporaryDirectory(dir=output_root, prefix=f".{case}.") as temporary:
        stage = Path(temporary)
        staged = {}
        native_temp = stage / "native_ct.nii.gz"
        sitk.WriteImage(native, str(native_temp))
        native_image = nib.load(native_temp)
        native_array = np.asarray(native_image.dataobj)
        if not np.array_equal(native_array, sitk.GetArrayFromImage(native).transpose(2, 1, 0)):
            raise ValueError("Native CT values changed during LPS-to-RAS conversion")
        native_image.header["descrip"] = CONVERSION_NIFTI_MARKER + b"; see airrc_conversion.json"
        _write_nifti(stage / "native.nii.gz", native_image)
        staged["native_ct"] = stage / "native.nii.gz"
        _write_nifti(stage / "ct.nii.gz", _derived_image(values.astype(np.float32), reference))
        staged["ct"] = stage / "ct.nii.gz"
        for group, label_values in LABEL_VALUES.items():
            path = stage / f"{group}.nii.gz"
            _write_nifti(path, _derived_image(np.isin(labels, label_values).astype(np.uint8), reference))
            staged[group] = path
        shutil.copyfile(label_path, stage / "multiclass.nii.gz")
        staged["multiclass"] = stage / "multiclass.nii.gz"
        report = {"schema_version": 1, "dataset": "AIRRC", "case": case, "parameters": parameters,
                  "source_files": source_files, "source_sha256": source_hashes, "source_context": source_context or {},
                  "outputs": outputs, "output_sha256": {outputs[g]: sha256(p) for g, p in staged.items()},
                  "geometry": _geometry(reference), "native_geometry": _geometry(native_image),
                  "alignment": {"reference_corners_max_outside_native_center_extent_mm": max_outside_mm,
                                "checked_in_physical_coordinates": True},
                  "label_definitions": {"0": "Background", "1": "Airway lumen", "2": "Airway wall",
                                        "3": "Pulmonary artery", "4": "Pulmonary vein"}}
        for path, digest in source_hashes.items():
            if sha256(Path(path)) != digest:
                raise ValueError("An AirRC source changed during conversion")
        for group, path in staged.items():
            target = Path(outputs[group])
            target.parent.mkdir(parents=True, exist_ok=True)
            path.replace(target)
        _write_json(case_dir / REPORT_NAME, report)
    verify_airrc_conversion(case_dir, groups=tuple(GROUP_DIRECTORIES))
    return {"case": case, "status": "ok", "report": str(case_dir / REPORT_NAME)}


def verify_airrc_conversion(case_dir: Path, *, groups: tuple[str, ...] = ("ct",)) -> dict[str, Any]:
    """Verify selected artifacts and original sources without trusting a moved manifest."""
    case_dir = Path(case_dir).resolve()
    manifest = case_dir / REPORT_NAME
    if manifest.is_symlink():
        raise ValueError("AirRC conversion manifest must not be a symlink")
    if not groups or any(group not in GROUP_DIRECTORIES for group in groups):
        raise ValueError("Unknown or empty AirRC conversion groups")
    manifest_hash = sha256(manifest)
    report = json.loads(manifest.read_text())
    if report.get("schema_version") != 1 or report.get("dataset") != "AIRRC" or report.get("case") != case_dir.name:
        raise ValueError("Invalid AirRC conversion manifest identity")
    parameters = report.get("parameters", {})
    if (parameters.get("ct_resampling") != "linear" or parameters.get("labels_resampled") is not False
            or parameters.get("origin_is_synthetic") is not False
            or parameters.get("analysis_grid") != "exact_reference_annotation_grid"
            or parameters.get("ct_output_dtype") != "float32" or parameters.get("outside_value_hu") != -1024):
        raise ValueError("Invalid AirRC conversion parameters")
    hashes = {str(manifest): manifest_hash}
    sources = {"conversion_report": str(manifest)}
    required_sources = {"multiclass"}
    if any(group in ("ct", "native_ct") for group in groups):
        required_sources.update(("ct_header", "ct_voxels"))
    if "annotation_package" in report["source_files"]:
        required_sources.add("annotation_package")
    for key in sorted(required_sources):
        path = Path(report["source_files"][key])
        if not path.is_absolute() or path.is_symlink() or path.resolve() != path:
            raise ValueError("AirRC source path changed identity")
        actual = sha256(path)
        if actual != report["source_sha256"].get(str(path)):
            raise ValueError("Stale AirRC conversion source")
        hashes[str(path)] = actual
        sources[f"conversion_{key}_source"] = str(path)
    original_label = nib.load(report["source_files"]["multiclass"])
    if _geometry(original_label) != report["geometry"]:
        raise ValueError("AirRC reference geometry differs from conversion manifest")
    if any(group in ("ct", "native_ct") for group in groups):
        header = _parse_mhd(Path(report["source_files"]["ct_header"]).read_text(encoding="ascii"))
        native_shape, native_affine = _native_geometry_from_mhd(header)
        if (report["native_geometry"]["shape"] != native_shape
                or not np.allclose(report["native_geometry"]["affine"], native_affine, atol=1e-4, rtol=1e-6)):
            raise ValueError("AirRC native geometry differs from the original CT")
    for group in dict.fromkeys(groups):
        expected = case_dir / GROUP_DIRECTORIES[group] / f"{case_dir.name}.nii.gz"
        path = Path(report["outputs"][group])
        if path != expected or path.is_symlink() or path.resolve() != path:
            raise ValueError("AirRC output points to a different case or changed identity")
        actual = sha256(path)
        if actual != report["output_sha256"].get(str(path)):
            raise ValueError("Stale AirRC conversion output")
        image = nib.load(path)
        geometry = report["native_geometry"] if group == "native_ct" else report["geometry"]
        if _geometry(image) != geometry:
            raise ValueError("AirRC output NIfTI geometry differs from conversion manifest")
        hashes[str(path)] = actual
        sources[f"conversion_{group}_output"] = str(path)
    if sha256(manifest) != manifest_hash:
        raise ValueError("AirRC conversion manifest changed during verification")
    return {"sources": sources, "source_sha256": hashes}


def prepare_case(case: str, input_root: Path, output_root: Path, *, inventory: dict[str, Any] | None = None,
                 overwrite: bool = False) -> dict[str, Any]:
    input_root = input_root.expanduser().resolve()
    inventory = inventory or discover_inventory(input_root)
    records = {row["case"]: row for row in inventory["records"]}
    if case not in records:
        raise ValueError("Unknown AirRC case")
    record = records[case]
    ct_path = _extract_ct(record, input_root)
    return convert_pair(case, ct_path, Path(record["label_path"]), output_root,
                        source_package=input_root / "26878867.zip", source_context=record, overwrite=overwrite)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=PROJECT_ROOT / "data/airrc")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "data/airrc/cases")
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--case", action="append")
    selection.add_argument("--limit", type=int, help="Prepare the first N de-identified cases")
    selection.add_argument("--list", action="store_true", help="List de-identified IDs and scan geometry")
    selection.add_argument("--pilot", action="store_true", help="Prepare three cases spanning orientation and slice spacing")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    inventory = discover_inventory(args.input_root)
    if args.list:
        for row in inventory["records"]:
            print(json.dumps({"case": row["case"], "native_spacing_mm": row["native_spacing_mm"],
                              "native_direction_lps": row["native_direction_lps"],
                              "reference_shape": row["reference_geometry"]["shape"]}), flush=True)
        return 0
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be positive")
    cases = args.case or (select_pilot_cases(inventory) if args.pilot else
                          [row["case"] for row in inventory["records"][:args.limit]])
    failed = False
    for case in dict.fromkeys(cases):
        try:
            row = prepare_case(case, args.input_root, args.output_root, inventory=inventory, overwrite=args.overwrite)
        except Exception as error:
            # Do not print source paths or patient metadata in public CLI summaries.
            row = {"case": case, "status": "error", "error_type": type(error).__name__}
            failed = True
        print(json.dumps(row, allow_nan=False), flush=True)
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
