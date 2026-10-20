"""Estimate PARSE area-class tube volumes from saved measured cross sections."""
from __future__ import annotations

import math

from parse_cross_section_bins import BIN_IDS, classify_segments


METHOD = "segment_mean_area_times_measured_length"
DEFINITIONS = {
    "segment_volume": "Estimated volume (mm3) = unrounded segment_area_mm2 * measured_length_mm; mL = mm3 / 1000",
    "equivalence": "Equivalent to the trapezoidal area integral over original consecutive sample pairs with two finite endpoint areas",
    "classification": "Assign the entire measured segment volume using its original mean area: BV5 < 5 mm2; BV5-10 includes 5 and 10 mm2; BV10 > 10 mm2",
    "volume_fraction": "Class estimated volume divided by the total estimated volume of all valid-area segments in the same case",
    "pooled_fraction": "Summed class estimated volumes divided by summed total estimated volumes across cases",
    "cohort_observation": "One equally weighted patient volume fraction per area class; pooled volume fractions are separate",
    "missing_values": "Exclude unavailable means; partial segments contribute only measured lengths, without gap bridging, zero filling or endpoint extrapolation",
    "empty_denominator": "Zero estimated volumes and null fractions; omit such cases from patient distribution plots",
    "scope": "PARSE prediction, existing graph segments in the largest 26-connected arterial component",
    "limitations": "Estimated tube volumes, not segmentation-mask volumes or measured physiological blood volumes. Nearby paths and branching cross sections may overlap; segment-mean classes differ from local-area classes",
    "review": "Preserve original coverage and needs_review evidence; complete and partial means contribute without promoting anatomical validation",
}
VOLUME_FIELDS = ("estimated_volume_mm3", "estimated_volume_ml")


def _finite(value, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Expected finite {'positive' if positive else 'nonnegative'} {name}")
    try:
        number = float(value)
    except (OverflowError, ValueError) as error:
        raise ValueError(f"Nonfinite {name}") from error
    if not math.isfinite(number) or number < 0 or (positive and number == 0):
        raise ValueError(f"Expected finite {'positive' if positive else 'nonnegative'} {name}")
    return number


def _sum(values, name: str) -> float:
    try:
        result = math.fsum(values)
    except OverflowError as error:
        raise ValueError(f"Nonfinite {name}") from error
    return _finite(result, name)


def estimate_segment_volumes(segments: list[dict]) -> dict:
    """Return valid-area segments with derived estimates and volume-weighted bins.

    The caller verifies saved means against the original local sample tables.
    This function retains those means and each original field verbatim. An
    unavailable mean is never given a zero-valued segment-volume measurement;
    its original segment count remains in the internal summary only.
    """
    classified = classify_segments(segments)
    groups = {key: [] for key in BIN_IDS}
    rows, full_lengths, measured_lengths = [], [], []
    for original in classified["segments"]:
        if any(key in original for key in VOLUME_FIELDS):
            raise ValueError("Existing segment volume fields cannot be overwritten")
        length = _finite(original.get("length_mm"), "length_mm", positive=True)
        measured = _finite(original.get("measured_length_mm"), "measured_length_mm")
        coverage = _finite(original.get("area_coverage_fraction"), "area_coverage_fraction")
        intervals = original.get("usable_interval_count")
        if isinstance(intervals, bool) or not isinstance(intervals, int) or intervals < 0:
            raise ValueError("Expected nonnegative integer usable_interval_count")
        if (coverage > 1 or measured > length + max(1e-8, length * 1e-10)
                or not math.isclose(coverage, measured / length, rel_tol=1e-10, abs_tol=1e-8)):
            raise ValueError("Measured length and saved area coverage disagree")
        full_lengths.append(length)
        measured_lengths.append(measured)
        if original["area_bin"] is None:
            if measured != 0 or coverage != 0 or intervals != 0:
                raise ValueError("Unavailable area must have no measured intervals")
            continue
        if measured <= 0 or intervals <= 0:
            raise ValueError("A valid mean area needs a positive measured interval")
        complete = math.isclose(measured, length, rel_tol=1e-10, abs_tol=1e-8)
        if (original["area_status"] == "complete") != complete:
            raise ValueError("Saved area status differs from measured length coverage")
        area = _finite(original["segment_area_mm2"], "segment_area_mm2", positive=True)
        volume = _finite(area * measured,
                         "estimated segment volume", positive=True)
        volume_ml = _finite(volume / 1000, "estimated segment volume in mL", positive=True)
        row = {**original, "estimated_volume_mm3": volume, "estimated_volume_ml": volume_ml}
        rows.append(row)
        groups[row["area_bin"]].append(volume)
    totals = {key: _sum(values, f"{key} estimated volume") for key, values in groups.items()}
    total = _sum(totals.values(), "total estimated volume")
    full = _sum(full_lengths, "total segment length")
    measured = _sum(measured_lengths, "total measured length")
    bins = [{"area_bin": key, "label": key, "segment_count": len(groups[key]),
             "volume_mm3": totals[key], "volume_ml": totals[key] / 1000,
             "total_volume_mm3": total, "total_volume_ml": total / 1000,
             "volume_fraction": totals[key] / total if total else None} for key in BIN_IDS]
    return {"segments": rows, "bins": bins, "summary": {
        "total_segment_count": classified["summary"]["total_segment_count"],
        "classified_segment_count": classified["summary"]["classified_segment_count"],
        "unavailable_segment_count": classified["summary"]["unavailable_segment_count"],
        "total_volume_mm3": total, "total_volume_ml": total / 1000,
        "total_length_mm": full, "measured_length_mm": measured,
        "area_coverage_fraction": min(measured / full, 1.),
    }}
