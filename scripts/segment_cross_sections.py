"""Integrate finite adjacent cross sections into representative segment areas."""
from collections import Counter, defaultdict
from copy import deepcopy
import csv
import json
import math
from pathlib import Path

SOURCE_LABELS = {"prediction": "Prediction", "reference": "Manual-label-derived reference"}

STATUSES = ("ok", "needs_review", "unavailable")
NEW_FIELDS = ("segment_area_mm2", "measured_length_mm", "area_coverage_fraction",
              "usable_interval_count", "area_status", "includes_needs_review")
METHOD = "length_weighted_trapezoidal_finite_adjacent"
DEFINITIONS = {
    "segment_area_mm2": "Representative segment cross-sectional area: sum(ds * (A_i + A_(i+1)) / 2) / sum(ds), using original consecutive finite sample pairs",
    "measured_length_mm": "Sum of original arc intervals with two finite endpoint areas; includes ok and needs_review samples",
    "area_coverage_fraction": "measured_length_mm / original length_mm",
    "usable_interval_count": "Number of original consecutive sample pairs contributing to the mean",
    "area_status": "complete, partial, or unavailable numerical interval coverage; not anatomical validation",
    "includes_needs_review": "At least one contributing interval endpoint has the original needs_review status",
    "missing_values": "No gap bridging, zero filling or endpoint extrapolation; no eligible interval means null area",
    "review": "Original sample quality counts, strict ok coverage and inherited review evidence remain unchanged",
    "source_pairing": "Prediction and manual-label-derived reference have independent branch IDs; no paired errors are computed",
}


def _number(value, name: str, *, positive: bool = False) -> float:
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or (positive and value <= 0)):
        raise ValueError(f"Invalid finite {'positive ' if positive else ''}number: {name}")
    return float(value)


def _same_number(actual, expected, name: str) -> None:
    if not math.isclose(_number(actual, name), expected, rel_tol=1e-10, abs_tol=1e-8):
        raise ValueError(f"Saved {name} disagrees with the sample records")


def _identity(value, name: str, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"Invalid {name}")
    return value


def summarize_measurement(measurement: dict, source: str) -> dict:
    """Integrate only original adjacent finite pairs, retaining every segment."""
    if source not in SOURCE_LABELS:
        raise ValueError("Expected prediction or reference source")
    segments, samples = measurement["segments"], measurement["samples"]
    if not isinstance(segments, list) or not segments or not isinstance(samples, list):
        raise ValueError("Nonempty segment and sample tables are required")
    indexed, by_branch, seen_samples = {}, defaultdict(list), set()
    counts = Counter({status: 0 for status in STATUSES})
    reasons = Counter()
    for segment in segments:
        branch = _identity(segment["branch_id"], "branch_id")
        if branch in indexed or any(key in segment for key in NEW_FIELDS):
            raise ValueError("Duplicate segment identity or existing area-summary fields")
        if segment.get("source", source) != source:
            raise ValueError("Segment source differs from its saved source")
        _number(segment["length_mm"], "segment length", positive=True)
        indexed[branch] = segment
    for sample in samples:
        branch = _identity(sample["branch_id"], "sample branch_id")
        identity = sample["sample_id"]
        if (branch not in indexed or not isinstance(identity, str) or not identity
                or identity in seen_samples):
            raise ValueError("Unknown sample branch or duplicate/invalid sample identity")
        if sample.get("source", source) != source:
            raise ValueError("Sample source differs from its saved source")
        seen_samples.add(identity)
        _identity(sample["sample_index"], "sample_index", 0)
        position = _number(sample["s_mm"], "sample arc position")
        if position < 0 or position > indexed[branch]["length_mm"] + 1e-8:
            raise ValueError("Sample arc position lies outside its segment length")
        status = sample["status"]
        if status not in STATUSES:
            raise ValueError("Unknown sample status")
        if status == "unavailable":
            if sample["area_mm2"] is not None:
                raise ValueError("Unavailable samples must have null area")
        else:
            _number(sample["area_mm2"], "sample area", positive=True)
        if not isinstance(sample.get("reasons", []), list):
            raise ValueError("Sample reasons must be a list")
        counts[status] += 1
        reasons.update(sample.get("reasons", []))
        by_branch[branch].append(sample)

    output, strict_spans = [], []
    for segment in segments:
        branch, length = segment["branch_id"], float(segment["length_mm"])
        ordered = sorted(by_branch[branch], key=lambda row: row["sample_index"])
        if not ordered or [row["sample_index"] for row in ordered] != list(range(len(ordered))):
            raise ValueError("Each segment requires a complete consecutive sample_index sequence")
        if any(last["s_mm"] <= first["s_mm"] for first, last in zip(ordered, ordered[1:])):
            raise ValueError("Sample arc positions must be strictly increasing in sample order")
        local_counts = Counter(row["status"] for row in ordered)
        if segment["sample_count"] != len(ordered):
            raise ValueError("Saved segment sample_count disagrees with its samples")
        for status in STATUSES:
            if segment[f"{status}_count"] != local_counts[status]:
                raise ValueError(f"Saved segment {status}_count disagrees with its samples")
        distances, integrals, strict, review = [], [], [], False
        for first, last in zip(ordered, ordered[1:]):
            delta = float(last["s_mm"] - first["s_mm"])
            if first["status"] == last["status"] == "ok":
                strict.append(delta)
            if first["area_mm2"] is not None and last["area_mm2"] is not None:
                distances.append(delta)
                integrals.append(delta * (first["area_mm2"] / 2 + last["area_mm2"] / 2))
                review |= "needs_review" in (first["status"], last["status"])
        strict_span = math.fsum(strict)
        _same_number(segment["ok_span_mm"], strict_span, "segment ok_span_mm")
        _same_number(segment["ok_span_fraction"], strict_span / length, "segment ok_span_fraction")
        strict_spans.append(strict_span)
        measured = math.fsum(distances)
        mean = math.fsum(integrals) / measured if measured else None
        if mean is not None:
            _number(mean, "segment mean area", positive=True)
        coverage = measured / length
        if coverage > 1 + 1e-9:
            raise ValueError("Measured interval length exceeds original segment length")
        complete = math.isclose(measured, length, rel_tol=1e-10, abs_tol=1e-8)
        output.append({**deepcopy(segment), "segment_area_mm2": mean,
                       "measured_length_mm": measured, "area_coverage_fraction": min(coverage, 1.),
                       "usable_interval_count": len(distances),
                       "area_status": "unavailable" if not distances else "complete" if complete else "partial",
                       "includes_needs_review": review})

    original = measurement["summary"]
    if original["segment_count"] != len(segments) or original["sample_count"] != len(samples):
        raise ValueError("Saved summary counts disagree with the measurement tables")
    if original["status_counts"] != dict(counts):
        raise ValueError("Saved summary status counts disagree with the sample records")
    if original.get("reason_counts", dict(reasons)) != dict(reasons):
        raise ValueError("Saved summary reason counts disagree with the sample records")
    total = math.fsum(row["length_mm"] for row in output)
    _same_number(original["total_length_mm"], total, "total_length_mm")
    _same_number(original["ok_span_mm"], math.fsum(strict_spans), "total ok_span_mm")
    measured = math.fsum(row["measured_length_mm"] for row in output)
    coverage_counts = Counter(row["area_status"] for row in output)
    return {"segments": output, "summary": {
        "segment_count": len(output), "sample_count": len(samples),
        "segment_area_count": sum(row["segment_area_mm2"] is not None for row in output),
        **{f"{status}_count": coverage_counts[status] for status in ("complete", "partial", "unavailable")},
        "measured_length_mm": measured, "total_length_mm": total, "area_coverage_fraction": measured / total,
        "includes_needs_review_count": sum(row["includes_needs_review"] for row in output),
        "status_counts": dict(counts)}}


def _validate_csv(path: Path, records: list[dict]) -> None:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
        fields = list(dict.fromkeys(key for row in records for key in row))
        if reader.fieldnames != fields or len(rows) != len(records):
            raise ValueError(f"CSV/JSON table structure differs: {path}")
    for row, expected in zip(rows, records):
        serialized = {key: "" if value is None else json.dumps(value, allow_nan=False)
                      if isinstance(value, (list, dict)) else str(value) for key, value in expected.items()}
        if row != serialized:
            raise ValueError(f"CSV/JSON table content differs: {path}")
