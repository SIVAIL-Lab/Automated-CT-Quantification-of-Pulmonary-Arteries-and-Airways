#!/usr/bin/env python3
"""Report PARSE patient percentiles of area-class count and estimated-volume fractions."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import platform
import re
import sys
import tempfile
import uuid

import numpy as np

from area_volume_charts import _validated_record
from parse_area_volumes import DEFINITIONS, METHOD
from parse_cross_section_bins import BIN_DEFINITIONS, BIN_IDS
from parse_cross_section_statistics import _csv_value, _publish_verified, _relative, _sha256, _verify, _write_csv
from report_parse_area_volumes import BIN_FIELDS, SUMMARY_FIELDS
from report_parse_cross_section_bins import _merge_hashes


PERCENTILES = (10, 25, 50, 75)
PERCENTILE_FIELDS = tuple(f"p{value}" for value in PERCENTILES)
METRICS = ("fraction", "volume_fraction")
FIELDS = ("area_bin", "label", "metric", "unit", "n", *PERCENTILE_FIELDS)


def _count(value) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("Expected a nonnegative integer segment or case count")
    return value


def _number(value) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("Expected a finite nonnegative numerical measurement")
    try:
        number = float(value)
    except (ValueError, OverflowError) as error:
        raise ValueError("Expected a finite numerical measurement") from error
    if not math.isfinite(number) or number < 0:
        raise ValueError("Expected a finite nonnegative numerical measurement")
    return number


def _load_publication(root: Path) -> tuple[list[dict], dict, dict]:
    """Read both metrics from the same verified volume publication and patient set."""
    path = root / "sources.json"
    raw = path.read_bytes()
    source = json.loads(raw)
    if (not isinstance(source, dict) or source.get("schema_version") != 1
            or source.get("dataset") != "PARSE" or source.get("source") != "prediction"
            or source.get("status") != "complete"
            or source.get("input_integrity") != "verified_after_publication"
            or source.get("volume_estimation_method") != METHOD
            or source.get("bins") != list(BIN_DEFINITIONS)
            or any(source.get("definitions", {}).get(key) != DEFINITIONS[key]
                   for key in ("classification", "volume_fraction", "scope", "missing_values"))):
        raise ValueError("Expected a complete PARSE estimated-volume publication with compatible definitions")
    cases = source.get("source_cases")
    if (not isinstance(cases, list) or not cases
            or any(not isinstance(case, str) or not re.fullmatch(r"PA[0-9]{6}", case) for case in cases)
            or len(cases) != len(set(cases)) or source.get("case_count") != len(cases)):
        raise ValueError("Source must declare unique, complete PARSE case membership")
    hashes = {str(path): hashlib.sha256(raw).hexdigest()}
    for field in ("source_sha256", "code_sha256", "artifact_sha256"):
        _merge_hashes(hashes, source.get(field, {}))
    _verify(hashes)

    def index(items, keys):
        if not isinstance(items, list) or any(not isinstance(row, dict) for row in items):
            raise ValueError("Expected a source record list")
        indexed = {tuple(row[key] for key in keys): row for row in items}
        if len(indexed) != len(items):
            raise ValueError("Duplicate source table identity")
        return indexed

    def read_csv(name, fields):
        input_path = root / name
        content = input_path.read_bytes()
        if source.get("artifact_sha256", {}).get(str(input_path)) != hashlib.sha256(content).hexdigest():
            raise ValueError(f"Unverified percentile source table: {name}")
        reader = csv.DictReader(io.StringIO(content.decode("utf-8-sig")))
        if reader.fieldnames != list(fields):
            raise ValueError(f"Unexpected percentile source columns: {name}")
        return list(reader)

    def agrees(actual, expected, fields):
        serialized = {key: "" if expected[key] is None else str(_csv_value(expected[key])) for key in fields}
        if actual != serialized:
            raise ValueError("Saved percentile input CSV differs from its publication metadata")

    summaries = index(source.get("case_summaries"), ("case",))
    summary_fields = ("source", "case", *SUMMARY_FIELDS)
    summary_csv = index(read_csv("case_summary.csv", summary_fields), ("case",))
    if set(summaries) != {(case,) for case in cases} or set(summary_csv) != set(summaries):
        raise ValueError("Case denominator membership differs from the source cohort")
    positive, empty = [], []
    for case in cases:
        row = summaries[(case,)]
        if row.get("source") != "prediction":
            raise ValueError("Expected prediction case summaries")
        classified = _count(row["classified_segment_count"])
        total = _count(row["total_segment_count"])
        unavailable = _count(row["unavailable_segment_count"])
        volume, volume_ml = _number(row["total_volume_mm3"]), _number(row["total_volume_ml"])
        if (classified + unavailable != total or not total
                or (classified > 0) != (volume > 0)
                or not math.isclose(volume / 1000, volume_ml, rel_tol=1e-12, abs_tol=0.)):
            raise ValueError("Invalid source segment or estimated-volume denominator")
        agrees(summary_csv[(case,)], row, summary_fields)
        (positive if classified else empty).append(case)
    exclusions = [{"case": case, "reason": "no_positive_estimated_volume"} for case in empty]
    if (source.get("included_cases") != positive or source.get("excluded_cases") != exclusions
            or source.get("included_case_count") != len(positive)):
        raise ValueError("Source eligibility differs from its case denominators")
    declared = index(source.get("cases"), ("case",))
    if set(declared) != {(case,) for case in positive}:
        raise ValueError("Source case records differ from eligible membership")
    bin_fields = ("source", "case", *BIN_FIELDS)
    published_bins = index(read_csv("per_case_area_volumes.csv", bin_fields), ("case", "area_bin"))
    if set(published_bins) != {(case, key) for case in cases for key in BIN_IDS}:
        raise ValueError("Missing, duplicate or unknown source area classes")
    records = []
    for case in cases:
        summary = summaries[(case,)]
        denominator = summary["classified_segment_count"]
        if denominator:
            record = _validated_record(declared[(case,)])
            if record.get("source") != "prediction" or any(record[key] != value for key, value in summary.items()):
                raise ValueError("Case plot record differs from its denominator summary")
            bins = record["bins"]
        else:
            bins = [{"area_bin": key, "label": key, "segment_count": 0,
                     "volume_mm3": 0., "volume_ml": 0., "total_volume_mm3": 0.,
                     "total_volume_ml": 0., "volume_fraction": None} for key in BIN_IDS]
        if sum(_count(row["segment_count"]) for row in bins) != denominator:
            raise ValueError("Area class counts do not conserve valid-area segments")
        combined = []
        for row in bins:
            key = row["area_bin"]
            if (row["label"] != key
                    or (row["segment_count"] > 0) != (row["volume_mm3"] > 0)
                    or not math.isclose(_number(row["volume_ml"]), row["volume_mm3"] / 1000,
                                        rel_tol=1e-12, abs_tol=0.)
                    or row["total_volume_ml"] != summary["total_volume_ml"]):
                raise ValueError("Area class count, label or volume units disagree")
            agrees(published_bins[(case, key)], {"source": "prediction", "case": case, **row}, bin_fields)
            combined.append({"area_bin": key, "fraction": row["segment_count"] / denominator if denominator else None,
                             "volume_fraction": row["volume_fraction"]})
        if denominator:
            records.append({"case": case, "bins": combined})
    return sorted(records, key=lambda row: row["case"]), hashes, source


def _summarize(records: list[dict]) -> list[dict]:
    """Use one unrounded observation per patient, including genuine zero classes."""
    if len({row["case"] for row in records}) != len(records):
        raise ValueError("Duplicate patient percentile observations")
    rows = []
    for key in BIN_IDS:
        bins = []
        for record in records:
            local = [row for row in record["bins"] if row["area_bin"] == key]
            if len(local) != 1:
                raise ValueError("Each patient requires one observation per area class")
            bins.append(local[0])
        for metric in METRICS:
            values = [_number(row[metric]) for row in bins]
            if any(value > 1 for value in values):
                raise ValueError("Patient fractions must be between zero and one")
            quantiles = np.quantile(values, np.asarray(PERCENTILES) / 100., method="linear") if values else [None] * 4
            rows.append({"area_bin": key, "label": key, "metric": metric, "unit": "fraction", "n": len(values),
                         **{field: float(value) if value is not None else None
                            for field, value in zip(PERCENTILE_FIELDS, quantiles)}})
    return rows


def _markdown(rows: list[dict], case_count: int, source_link: str) -> str:
    lines = ["# PARSE cross-sectional area fraction percentiles", "",
             f"Included patients: **{case_count}**. Each patient contributes one equally weighted observation per class and metric.", "",
             "Classes use the original segment mean cross-sectional area: BV5 <5 mm²; BV5-10 includes 5 and 10 mm²; BV10 >10 mm². "
             "Segment proportions divide each class count by the patient's number of segments with a valid mean area. "
             "Estimated volume fractions divide class estimates by the patient's total estimate from those measurable segments.", "",
             "Segment volume estimates equal saved mean area × measured length. Partial segments contribute only measured intervals. "
             "These estimates can overlap and are not segmentation-mask volumes. Existing review flags remain unchanged.", "",
             "Percentiles use linear interpolation, with P50 equal to the median. Values below are percentages rounded to two decimal places; "
             "CSV values retain full-precision fractions between zero and one. Zero class values remain observations. "
             "Cases with no measurable segments are excluded from both metrics, with exclusions recorded in the provenance.", "",
             "| Area class | Metric | N | P10 | P25 | P50 | P75 |", "|---|---|---:|---:|---:|---:|---:|"]
    for row in rows:
        label = "Segment proportion (%)" if row["metric"] == "fraction" else "Estimated volume fraction (%)"
        values = " | ".join("—" if row[field] is None else f"{100 * row[field]:.2f}" for field in PERCENTILE_FIELDS)
        lines.append(f"| {row['label']} | {label} | {row['n']} | {values} |")
    lines += ["", "[Full-precision table](percentiles.csv) · [Sources and calculation details](sources.json) · "
              f"[Source cohort report](<{source_link}>)", ""]
    return "\n".join(lines)


def generate(publication_root: Path, output_root: Path) -> dict:
    publication_root, output_root = Path(publication_root).resolve(), Path(output_root).resolve()
    if publication_root.is_relative_to(output_root) or output_root.is_relative_to(publication_root):
        raise ValueError("Percentile outputs must be separate from the source publication")
    records, hashes, source = _load_publication(publication_root)
    if any(output_root == Path(name).parent or output_root in Path(name).parents
           or Path(name).parent in output_root.parents for name in hashes):
        raise ValueError("Percentile outputs must be separate from original input directories")
    names = ("report_parse_area_percentiles.py", "area_volume_charts.py", "parse_area_volumes.py",
             "parse_cross_section_bins.py", "parse_cross_section_statistics.py", "report_parse_area_volumes.py",
             "report_parse_cross_section_bins.py")
    code_hashes = {str(Path(__file__).with_name(name).resolve()): _sha256(Path(__file__).with_name(name)) for name in names}
    rows = _summarize(records)
    identifier = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ_") + uuid.uuid4().hex[:8]
    final = output_root / identifier
    output_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".area_percentiles_", dir=output_root) as temporary:
        staging = Path(temporary)
        _write_csv(staging / "percentiles.csv", rows, FIELDS)
        (staging / "report.md").write_text(_markdown(rows, len(records), _relative(publication_root / "report.md", final)), encoding="utf-8")
        result = {"schema_version": 1, "status": "complete", "dataset": "PARSE", "source": "prediction",
                  "run_id": source["run_id"], "publication_id": identifier, "output_dir": str(final),
                  "source_publication": str(publication_root), "source_case_count": source["case_count"],
                  "case_count": len(records), "included_cases": [row["case"] for row in records],
                  "excluded_cases": source["excluded_cases"], "rows": rows, "bins": list(BIN_DEFINITIONS),
                  "parameters": {"percentiles": list(PERCENTILES), "method": "linear", "display_decimals": 2},
                  "definitions": {**DEFINITIONS, "fraction": "Class segment count divided by the same patient's valid-area segment count",
                                  "csv_fraction_unit": "Fraction between zero and one", "report_fraction_unit": "Percent",
                                  "observation": "One equally weighted patient per class and metric; not pooled counts or volumes"},
                  "source_sha256": hashes, "code_sha256": code_hashes,
                  "versions": {"python": platform.python_version(), "numpy": np.__version__},
                  "areas_recomputed": False, "volumes_recomputed": False, "review_status_promotion": False,
                  "artifacts": {key: str(final / name) for key, name in
                                (("report", "report.md"), ("csv", "percentiles.csv"), ("sources", "sources.json"))},
                  "artifact_sha256": {str(final / path.name): _sha256(path) for path in staging.iterdir()},
                  "input_integrity": "verified_after_publication"}
        _verify(hashes)
        _verify(code_hashes)
        (staging / "sources.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        return _publish_verified(staging, final, result, hashes, code_hashes)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--publication-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = generate(args.publication_root, args.output_root)
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"Area percentile report failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    print(f"Completed PARSE area percentiles for {result['case_count']} patients.")
    print(result["artifacts"]["report"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
