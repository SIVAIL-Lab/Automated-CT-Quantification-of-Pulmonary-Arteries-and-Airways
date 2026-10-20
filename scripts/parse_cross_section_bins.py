"""Classify saved segment mean cross-sectional areas without changing measurements."""
from __future__ import annotations

from copy import deepcopy
import math


BIN_DEFINITIONS = (
    {"area_bin": "BV5", "label": "BV5", "lower_mm2": 0., "upper_mm2": 5.,
     "lower_inclusive": False, "upper_inclusive": False},
    {"area_bin": "BV5-10", "label": "BV5-10", "lower_mm2": 5., "upper_mm2": 10.,
     "lower_inclusive": True, "upper_inclusive": True},
    {"area_bin": "BV10", "label": "BV10", "lower_mm2": 10., "upper_mm2": None,
     "lower_inclusive": False, "upper_inclusive": False},
)
BIN_IDS = tuple(definition["area_bin"] for definition in BIN_DEFINITIONS)


def classify_segments(segments: list[dict]) -> dict:
    """Add area_bin to each original segment; fractions use classified segments.

    The input is one case/source segment table. Finite mean areas from complete
    and partial coverage are classified at their saved precision. Unavailable
    means remain null for auditing and contribute to no area bin or bin-fraction
    denominator. Bin fractions are null when no segment has an available area.
    Every existing field is copied unchanged, including coverage and review
    evidence. These categories describe area, without estimating blood volume.
    """
    if not isinstance(segments, list) or not segments:
        raise ValueError("A nonempty segment list is required")
    counts = dict.fromkeys(BIN_IDS, 0)
    seen, output = set(), []
    for row in segments:
        if not isinstance(row, dict):
            raise ValueError("Each segment must be a dictionary")
        branch = row.get("branch_id")
        if isinstance(branch, bool) or not isinstance(branch, int) or branch <= 0:
            raise ValueError("Invalid positive integer branch_id")
        if branch in seen:
            raise ValueError("Duplicate branch_id in segment table")
        seen.add(branch)
        if "area_bin" in row:
            raise ValueError("Existing area_bin cannot be overwritten")
        if "segment_area_mm2" not in row:
            raise ValueError("Missing segment_area_mm2")
        status = row.get("area_status")
        if status not in ("complete", "partial", "unavailable"):
            raise ValueError("Invalid segment area_status")
        area = row["segment_area_mm2"]
        if area is None:
            if status != "unavailable":
                raise ValueError("Null segment area requires unavailable area_status")
            area_bin = None
        else:
            if status == "unavailable":
                raise ValueError("Unavailable segment must have null area")
            if (isinstance(area, bool) or not isinstance(area, (int, float))
                    or (isinstance(area, float) and not math.isfinite(area)) or area <= 0):
                raise ValueError("Segment area must be a finite positive number")
            area_bin = "BV5" if area < 5 else "BV5-10" if area <= 10 else "BV10"
            counts[area_bin] += 1
        output.append({**deepcopy(row), "area_bin": area_bin})

    total = len(output)
    classified = sum(counts.values())
    unavailable = total - classified
    bins = [{"area_bin": definition["area_bin"], "label": definition["label"],
             "segment_count": counts[definition["area_bin"]],
             "total_segment_count": total,
             "classified_segment_count": classified,
             "fraction": counts[definition["area_bin"]] / classified if classified else None}
            for definition in BIN_DEFINITIONS]
    return {"segments": output, "bins": bins, "summary": {
        "total_segment_count": total,
        "classified_segment_count": classified,
        "unavailable_segment_count": unavailable,
        "classified_fraction": classified / total,
        "unavailable_fraction": unavailable / total}}
