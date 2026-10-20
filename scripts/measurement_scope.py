"""Resolve immutable raw masks and PARSE largest-component measurement masks."""
from __future__ import annotations

import json
from pathlib import Path
import re
import uuid

import nibabel as nib
import numpy as np
from scipy import ndimage

from run_edt_experiment import sha256
from run_lung_volume_parse import _check_geometry, _case_file, _voxel_volume_mm3


POLICY = "largest_26"
SELECTION = {"connectivity": 26, "criterion": "foreground_voxel_count",
             "tie_break": "lexicographically_smallest_original_ijk", "algorithm_version": 1}
FILENAMES = {"artery": "lung_arteries.nii.gz", "airway": "lung_airways.nii.gz"}


def _read(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def load_profile(results_root: Path) -> dict | None:
    root = Path(results_root).resolve()
    path = root / "measurement_scope.json"
    if not path.exists():
        return None
    profile = _read(path)
    if (profile.get("schema_version") != 1 or profile.get("dataset") != "PARSE"
            or profile.get("measurement_scope") != POLICY):
        raise ValueError("Only PARSE largest_26 measurement profiles are supported")
    for key in ("input_root", "source_results_root"):
        if not isinstance(profile.get(key), str) or not Path(profile[key]).is_absolute():
            raise ValueError(f"Measurement profile requires an absolute {key}")
    source = Path(profile["source_results_root"]).resolve()
    if source == root or (source / "measurement_scope.json").exists():
        raise ValueError("Measurement source must be a separate original results root")
    if not isinstance(profile.get("run_id"), str) or not profile["run_id"]:
        raise ValueError("Measurement profile requires a run ID")
    return profile


def source_results_root(results_root: Path) -> Path:
    profile = load_profile(results_root)
    return Path(profile["source_results_root"]).resolve() if profile else Path(results_root).resolve()


def scope_for_root(results_root: Path) -> str:
    return POLICY if load_profile(results_root) else "all"


def select_largest_component(values: np.ndarray, voxel_volume_mm3: float) -> tuple[np.ndarray, dict]:
    """Keep one 26-component and original label values; permit empty masks."""
    data = np.asarray(values)
    if data.ndim != 3 or not np.isfinite(data).all() or not (
            np.isin(data, [0, 1]).all() or np.isin(data, [0, 255]).all()):
        raise ValueError("A finite binary 3D mask with labels 0/1 or 0/255 is required")
    if not np.isfinite(voxel_volume_mm3) or voxel_volume_mm3 <= 0:
        raise ValueError("Voxel volume must be positive and finite")
    labels, count = ndimage.label(data != 0, structure=np.ones((3, 3, 3), dtype=np.uint8))
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    # ndimage labels in lexicographic array-index order; choosing the first
    # tied label therefore chooses the component with the smallest first IJK.
    chosen = int(np.argmax(sizes)) if count else None
    retained = int(sizes[chosen]) if chosen is not None else 0
    total = int(sizes.sum())
    selected = np.where(labels == chosen, data, 0) if chosen is not None else np.zeros_like(data)
    first = np.argwhere(labels == chosen)[0].tolist() if chosen is not None else None
    return selected, {**SELECTION, "component_count": int(count), "selected_component_id": chosen,
                      "selected_component_first_ijk": first, "input_voxels": total,
                      "selected_voxels": retained, "excluded_voxels": total - retained,
                      "input_volume_ml": total * voxel_volume_mm3 / 1000.,
                      "selected_volume_ml": retained * voxel_volume_mm3 / 1000.,
                      "excluded_volume_ml": (total - retained) * voxel_volume_mm3 / 1000.,
                      "retained_fraction": retained / total if total else None}


def _raw_source(case_dir: Path, results_root: Path, organ: str, source: str) -> tuple[Path, dict]:
    if organ not in FILENAMES or source not in ("prediction", "reference") or (
            source == "reference" and organ != "artery"):
        raise ValueError("Expected predicted artery/airway or reference artery")
    case_dir, root = Path(case_dir).resolve(), Path(results_root).resolve()
    profile = load_profile(root)
    if profile and (not re.fullmatch(r"PA[0-9]{6}", case_dir.name)
                    or case_dir.parent != Path(profile["input_root"]).resolve()):
        raise ValueError("Largest-component measurement is restricted to the profile's PARSE cases")
    original_root = source_results_root(root)
    if source == "prediction":
        mask = original_root / "totalsegmentator" / case_dir.name / FILENAMES[organ]
    else:
        from run_totalsegmentator_parse import _case_file as source_case_file
        mask = source_case_file(case_dir, "label").resolve()
    ct = _case_file(case_dir).resolve()
    hashes = {str(path): sha256(path) for path in (mask, ct)}
    if source == "prediction":
        provenance_path = mask.with_name("segmentation_provenance.json")
        provenance = _read(provenance_path)
        state = provenance.get("provenance_status")
        if (provenance.get("schema_version") != 1 or provenance.get("case") != case_dir.name
                or provenance.get("input_sha256") != hashes[str(ct)]
                or provenance.get("artifact_sha256", {}).get(mask.name) != hashes[str(mask)]
                or state not in ("inference_verified", "legacy_adopted")
                or provenance.get("model_input_verified") is not (state == "inference_verified")):
            raise ValueError("Original segmentation provenance is missing, invalid, or stale")
        hashes[str(provenance_path)] = sha256(provenance_path)
    if profile:
        hashes[str(root / "measurement_scope.json")] = sha256(root / "measurement_scope.json")
    return mask, hashes


def _locations(case_dir: Path, root: Path, organ: str, source: str) -> tuple[Path, Path]:
    directory = root / "measurement_masks" / source / case_dir.name
    filename = FILENAMES[organ]
    selected = directory / filename
    return selected, directory / (filename.removesuffix(".nii.gz") + "_selection.json")


def get_mask(case_dir: Path, results_root: Path, organ: str = "artery",
             source: str = "prediction") -> tuple[Path, dict]:
    """Read and verify a prepared selected mask; never create or update files."""
    root, case_dir = Path(results_root).resolve(), Path(case_dir).resolve()
    original, hashes = _raw_source(case_dir, root, organ, source)
    scope = scope_for_root(root)
    metadata = {"measurement_scope": scope, "scope": scope, "organ": organ, "source": source,
                "original_mask_path": str(original), "original_mask_sha256": hashes[str(original)]}
    if scope == "all":
        return original, {**metadata, "selected_mask_path": str(original),
                          "selected_mask_sha256": hashes[str(original)], "mask_path": str(original),
                          "mask_sha256": hashes[str(original)], "selection": None,
                          "manifest_path": None, "manifest_sha256": None, "source_sha256": hashes}
    selected, manifest = _locations(case_dir, root, organ, source)
    saved = _read(manifest)
    if any(saved.get(key) != value for key, value in metadata.items()):
        raise ValueError("Selected mask identity differs from the requested original mask or scope")
    if (saved.get("schema_version") != 1 or saved.get("case") != case_dir.name
            or saved.get("source_sha256") != hashes or saved.get("selection_parameters") != SELECTION
            or saved.get("selected_mask_path") != str(selected)
            or saved.get("selection_code_sha256") != sha256(Path(__file__))):
        raise ValueError("Selected mask inputs, parameters, or selection code are stale")
    selected_hash = sha256(selected)
    if selected_hash != saved.get("selected_mask_sha256"):
        raise ValueError("Selected mask artifact hash is stale")
    original_image, selected_image = nib.load(original), nib.load(selected)
    _check_geometry(original_image, selected_image)
    for name in ("qform", "sform"):
        old, old_code = getattr(original_image, f"get_{name}")(coded=True)
        new, new_code = getattr(selected_image, f"get_{name}")(coded=True)
        if old_code != new_code or (old_code and not np.array_equal(old, new)):
            raise ValueError(f"Selected mask changed {name}")
    manifest_hash = sha256(manifest)
    return selected, {**metadata, "selected_mask_path": str(selected), "selected_mask_sha256": selected_hash,
                       "mask_path": str(selected), "mask_sha256": selected_hash,
                       "selection": saved["selection"], "manifest_path": str(manifest),
                       "manifest_sha256": manifest_hash,
                       "source_sha256": {**hashes, str(selected): selected_hash, str(manifest): manifest_hash}}


def prepare_case_masks(case_dir: Path, results_root: Path) -> dict:
    """Prepare three independently selected masks; retain raw inputs untouched."""
    case_dir, root = Path(case_dir).resolve(), Path(results_root).resolve()
    if load_profile(root) is None:
        return {}
    results = {}
    for source, organ in (("prediction", "artery"), ("prediction", "airway"), ("reference", "artery")):
        original, hashes = _raw_source(case_dir, root, organ, source)
        selected, manifest = _locations(case_dir, root, organ, source)
        if manifest.exists():
            # Never repair a stale selected mask silently inside an existing run.
            _, metadata = get_mask(case_dir, root, organ, source)
            results[f"{source}_{organ}"] = metadata
            continue
        # A killed writer may have renamed the selected NIfTI before committing
        # its manifest. Preserve incomplete artifacts, then restart this mask.
        incomplete = [path for path in (selected, selected.with_name("." + selected.name),
                                        manifest.with_suffix(".tmp")) if path.exists()]
        orphaned = []
        if incomplete:
            quarantine = selected.parent / "orphaned" / uuid.uuid4().hex
            quarantine.mkdir(parents=True)
            for path in incomplete:
                destination = quarantine / path.name
                path.replace(destination)
                orphaned.append(str(destination))
        image = nib.load(original)
        _check_geometry(nib.load(_case_file(case_dir)), image)
        data = np.asarray(image.dataobj)
        values, audit = select_largest_component(data, _voxel_volume_mm3(image))
        if orphaned:
            audit["orphaned_incomplete_artifacts"] = orphaned
        header = image.header.copy()
        candidate = nib.Nifti1Image(values, image.affine, header)
        candidate.set_qform(*image.get_qform(coded=True))
        candidate.set_sform(*image.get_sform(coded=True))
        selected.parent.mkdir(parents=True, exist_ok=True)
        temporary = selected.with_name("." + selected.name)
        try:
            nib.save(candidate, temporary)
            for path, digest in hashes.items():
                if sha256(Path(path)) != digest:
                    raise ValueError("Original sources changed during component selection")
            temporary.replace(selected)
            payload = {"schema_version": 1, "case": case_dir.name, "measurement_scope": POLICY,
                       "scope": POLICY, "source": source, "organ": organ,
                       "original_mask_path": str(original), "original_mask_sha256": hashes[str(original)],
                       "selected_mask_path": str(selected), "selected_mask_sha256": sha256(selected),
                       "selection": audit, "selection_parameters": SELECTION,
                       "selection_code_sha256": sha256(Path(__file__)), "source_sha256": hashes}
            staging = manifest.with_suffix(".tmp")
            staging.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
            staging.replace(manifest)
        except BaseException:
            if not manifest.exists():
                selected.unlink(missing_ok=True)
            raise
        finally:
            temporary.unlink(missing_ok=True)
        _, results[f"{source}_{organ}"] = get_mask(case_dir, root, organ, source)
    return results
