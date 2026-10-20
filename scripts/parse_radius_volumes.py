"""Group existing PARSE tube-volume estimates by the saved segment EDT radius.

This module reads no images or geometry. It checks saved measurements and sums
their existing estimates; it never recalculates a radius or cross section.
"""
from __future__ import annotations

from copy import deepcopy
import math
from numbers import Integral, Real

from parse_radius_statistics import BIN_DEFINITIONS, BIN_IDS, classify_segments


METHOD = "segment_mean_area_times_measured_length"
DEFINITIONS = {
    "segment_volume": "Reuse saved estimated_volume_mm3 = unrounded segment_area_mm2 * measured_length_mm; mL = mm3 / 1000",
    "classification": "Assign the entire measured segment estimate by its unrounded physical-length-weighted mean EDT radius over the original whole segment: (0, 2], (2, 4], (4, 6], or >6 mm",
    "volume_fraction": "Radius-bin estimated volume divided by the total estimated volume of all valid-area, measured segments in the same case; the four fractions sum to one when the denominator is positive",
    "pooled_fraction": "Summed radius-bin estimated volumes divided by summed total estimated volumes across cases; separate from equally weighted patient distributions",
    "cohort_observation": "One equally weighted patient volume fraction per radius bin",
    "missing_values": "Unavailable area segments are excluded before classification; partial segments contribute only their original measured lengths without gap filling, gap bridging, or endpoint extrapolation",
    "empty_denominator": "Zero estimated volumes and null fractions; omit such cases from patient fraction distributions",
    "scope": "PARSE prediction only, existing graph segments in the largest 26-connected arterial component",
    "limitations": "Estimated tube volumes, not actual segmentation-mask volumes or physiological blood volumes; measured spans may overlap near branches, and whole-segment radius classes differ from local-radius classes",
    "review": "Preserve original coverage, area status, needs_review evidence, and inherited tree flags; complete and partial numerical coverage is not anatomical validation",
}


def _positive(value, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite positive number")
    try:
        number = float(value)
    except (OverflowError, ValueError):
        raise ValueError(f"{name} must be a finite positive number") from None
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{name} must be a finite positive number")
    return number


def _positive_integer(value, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _sum(values, name: str) -> float:
    try:
        value = math.fsum(values)
    except OverflowError:
        raise ValueError(f"{name} is nonfinite") from None
    if not math.isfinite(value):
        raise ValueError(f"{name} is nonfinite")
    return value


def summarize_radius_volumes(segments: list[dict]) -> dict:
    """Return copied segment rows, four volume bins, and the shared denominator.

    Input rows must already have available segment areas and saved tube-volume
    estimates. The caller verifies their publication and source-table identity.
    Empty input represents no measurable segments, not a zero-valued fraction.
    """
    if not isinstance(segments, list):
        raise ValueError("A segment table list is required")
    rows, radii, seen = [], [], set()
    for original in segments:
        if not isinstance(original, dict):
            raise ValueError("Each segment row must be a mapping")
        branch = _positive_integer(original.get("branch_id"), "branch_id")
        if branch in seen:
            raise ValueError(f"Duplicate branch_id: {branch}")
        seen.add(branch)
        radius = _positive(original.get("edt_segment_radius_mm"), "edt_segment_radius_mm")
        if "segment_radius_mm" in original:
            alias = _positive(original["segment_radius_mm"], "segment_radius_mm")
            if alias != radius:
                raise ValueError("segment_radius_mm differs from edt_segment_radius_mm")
        length = _positive(original.get("length_mm"), "length_mm")
        measured = _positive(original.get("measured_length_mm"), "measured_length_mm")
        coverage = _positive(original.get("area_coverage_fraction"), "area_coverage_fraction")
        _positive_integer(original.get("usable_interval_count"), "usable_interval_count")
        if (coverage > 1 or measured > length + max(1e-8, length * 1e-10)
                or not math.isclose(coverage, measured / length, rel_tol=1e-10, abs_tol=1e-8)):
            raise ValueError("Measured length and saved area coverage disagree")
        status = original.get("area_status")
        if status not in ("complete", "partial"):
            raise ValueError("Only available complete or partial segment areas are accepted")
        complete = math.isclose(measured, length, rel_tol=1e-10, abs_tol=1e-8)
        if (status == "complete") != complete:
            raise ValueError("Saved area status differs from measured length coverage")
        area = _positive(original.get("segment_area_mm2"), "segment_area_mm2")
        volume = _positive(original.get("estimated_volume_mm3"), "estimated_volume_mm3")
        volume_ml = _positive(original.get("estimated_volume_ml"), "estimated_volume_ml")
        expected = _positive(area * measured, "area-times-measured-length volume")
        expected_ml = _positive(volume / 1000, "estimated volume in mL")
        if not math.isclose(volume, expected, rel_tol=1e-10, abs_tol=0):
            raise ValueError("Saved estimated_volume_mm3 differs from area times measured length")
        if not math.isclose(volume_ml, expected_ml, rel_tol=1e-10, abs_tol=0):
            raise ValueError("Saved estimated_volume_ml differs from estimated_volume_mm3 / 1000")
        rows.append(deepcopy(original))
        radii.append({"branch_id": branch, "segment_radius_mm": radius})
    classified = classify_segments(radii)["segments"] if radii else []
    groups = {key: [] for key in BIN_IDS}
    for row, radius_row in zip(rows, classified):
        key = radius_row["radius_bin"]
        if "radius_bin" in row and row["radius_bin"] != key:
            raise ValueError("Existing radius_bin differs from the unrounded EDT radius")
        row["radius_bin"] = key
        groups[key].append(row["estimated_volume_mm3"])
    volumes = {key: _sum(values, f"{key} estimated volume") for key, values in groups.items()}
    total = _sum(volumes.values(), "total estimated volume")
    total_ml = total / 1000
    measured_total = _sum((row["measured_length_mm"] for row in rows), "total measured length")
    bins = [{"radius_bin": definition["radius_bin"], "label": definition["label"],
             "segment_count": len(groups[definition["radius_bin"]]),
             "volume_mm3": volumes[definition["radius_bin"]],
             "volume_ml": volumes[definition["radius_bin"]] / 1000,
             "total_volume_mm3": total, "total_volume_ml": total_ml,
             "volume_fraction": volumes[definition["radius_bin"]] / total if total else None}
            for definition in BIN_DEFINITIONS]
    return {"segments": rows, "bins": bins, "summary": {
        "classified_segment_count": len(rows), "total_volume_mm3": total,
        "total_volume_ml": total_ml, "measured_length_mm": measured_total,
    }}
