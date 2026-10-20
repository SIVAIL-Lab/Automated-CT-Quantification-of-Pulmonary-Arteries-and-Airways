#!/usr/bin/env python3
"""Publish PARSE estimated pulmonary arterial volume fractions by segment radius."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import html
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

from area_volume_charts import render_case, render_cohort
from parse_area_volumes import DEFINITIONS as AREA_DEFINITIONS
from parse_cross_section_statistics import _write_csv
from parse_radius_statistics import BIN_DEFINITIONS, BIN_IDS
from parse_radius_volumes import DEFINITIONS, METHOD, summarize_radius_volumes
from plot_parse_radius_case import TICK_LABELS


BIN_FIELDS = ("case", "radius_bin", "label", "segment_count", "volume_mm3", "volume_ml",
              "total_volume_mm3", "total_volume_ml", "volume_fraction")
SUMMARY_FIELDS = ("case", "total_segment_count", "classified_segment_count", "unavailable_segment_count",
                  "total_volume_mm3", "total_volume_ml", "measured_length_mm", "area_coverage_fraction")
STATISTIC_FIELDS = ("radius_bin", "label", "metric", "unit", "n", "mean", "std",
                    "min", "p10", "p25", "p50", "p75", "max")
HISTOGRAM_FIELDS = ("radius_bin", "metric", "unit", "bin_index", "lower_bound", "upper_bound",
                    "upper_inclusive", "case_count", "included_case_count")
METHOD_TEXT = (
    "Each segment retains its saved estimated volume: mean cross-sectional area × measured length. "
    "Its unrounded, physical-length-weighted mean EDT radius assigns the entire estimate to "
    "(0, 2], (2, 4], (4, 6], or >6 mm. A group's estimated volume is divided by the same patient's "
    "total estimated volume of measurable segments. The four fractions sum to 100% for a positive "
    "denominator. Unavailable-area segments are excluded; partially measurable segments contribute "
    "only their measured intervals. Measurements use the largest 26-connected predicted arterial "
    "component in PARSE."
)
LIMITATION = (
    "These are estimated tube-volume fractions, not segmentation-mask volume fractions or measured "
    "physiological blood volumes. Cross sections and nearby paths can overlap, so estimated totals "
    "may exceed mask volumes. Original coverage, segmentation provenance and review flags are retained; "
    "this regrouping does not establish anatomical validity."
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _number(value, field: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"Invalid numerical field: {field}")
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"Expected finite nonnegative {field}")
    return number


def _count(value, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"Expected nonnegative integer {field}")
    return value


def _load_publication(root: Path) -> tuple[list[dict], dict, dict]:
    """Read hash-bound saved numeric tables without opening clinical source files."""
    path = root / "sources.json"
    raw = path.read_bytes()
    source = json.loads(raw)
    if (not isinstance(source, dict) or source.get("schema_version") != 1
            or source.get("dataset") != "PARSE" or source.get("source") != "prediction"
            or source.get("status") != "complete"
            or source.get("input_integrity") != "verified_after_publication"
            or source.get("volume_estimation_method") != METHOD
            or any(source.get("definitions", {}).get(key) != AREA_DEFINITIONS[key]
                   for key in ("scope", "missing_values", "volume_fraction"))):
        raise ValueError("Expected a complete compatible PARSE predicted area-volume publication")
    cases = source.get("source_cases")
    if (not isinstance(cases, list) or not cases
            or any(not isinstance(case, str) or not re.fullmatch(r"PA[0-9]{6}", case) for case in cases)
            or len(set(cases)) != len(cases) or source.get("case_count") != len(cases)):
        raise ValueError("Source must declare unique PARSE case membership")
    hashes = {str(path): hashlib.sha256(raw).hexdigest()}

    def read_table(name):
        table = root / name
        content = table.read_bytes()
        digest = hashlib.sha256(content).hexdigest()
        if source.get("artifact_sha256", {}).get(str(table)) != digest:
            raise ValueError(f"Saved numeric table checksum mismatch: {name}")
        hashes[str(table)] = digest
        return list(csv.DictReader(io.StringIO(content.decode("utf-8-sig"))))

    saved_summaries = source.get("case_summaries", [])
    if (not isinstance(saved_summaries, list) or any(not isinstance(row, dict) for row in saved_summaries)
            or len(saved_summaries) != len(cases)
            or {row.get("case") for row in saved_summaries} != set(cases)):
        raise ValueError("Source case summaries differ from the declared cohort")
    summaries = {row["case"]: row for row in saved_summaries}
    totals = read_table("case_summary.csv")
    if len(totals) != len(cases) or {row.get("case") for row in totals} != set(cases):
        raise ValueError("Saved denominator table differs from the declared cohort")
    for row in totals:
        saved = summaries[row["case"]]
        if row.get("source") != "prediction" or saved.get("source") != "prediction":
            raise ValueError("Only prediction measurements are supported")
        for field in ("classified_segment_count", "total_volume_mm3", "total_volume_ml", "measured_length_mm"):
            if _number(row[field], field) != _number(saved[field], field):
                raise ValueError("Saved case denominator CSV differs from its publication metadata")

    groups = {case: [] for case in cases}
    float_fields = ("length_mm", "edt_segment_radius_mm", "segment_area_mm2", "measured_length_mm",
                    "area_coverage_fraction", "ok_span_mm", "ok_span_fraction",
                    "estimated_volume_mm3", "estimated_volume_ml")
    int_fields = ("branch_id", "usable_interval_count", "sample_count", "ok_count",
                  "needs_review_count", "unavailable_count")
    for original in read_table("all_segment_area_volumes.csv"):
        if original.get("source") != "prediction" or original.get("case") not in groups:
            raise ValueError("Segment table contains an unknown case or nonprediction source")
        row = dict(original)
        for field in float_fields:
            if field in row:
                row[field] = _number(row[field], field)
        for field in int_fields:
            if field in row:
                row[field] = int(row[field])
        if "includes_needs_review" in row:
            if row["includes_needs_review"] not in ("True", "False"):
                raise ValueError("Invalid saved segment review flag")
            row["includes_needs_review"] = row["includes_needs_review"] == "True"
        if "inherited_tree_flag_ids" in row:
            row["inherited_tree_flag_ids"] = json.loads(row["inherited_tree_flag_ids"])
        groups[row["case"]].append(row)

    records = []
    for case in cases:
        result = summarize_radius_volumes(groups[case])
        saved = summaries[case]
        total_count = _count(saved["total_segment_count"], "total_segment_count")
        classified = _count(saved["classified_segment_count"], "classified_segment_count")
        unavailable = _count(saved["unavailable_segment_count"], "unavailable_segment_count")
        if total_count <= 0 or classified + unavailable != total_count:
            raise ValueError("Source segment counts do not conserve the original tree")
        for field in ("classified_segment_count", "total_volume_mm3", "total_volume_ml", "measured_length_mm"):
            if not math.isclose(result["summary"][field], _number(saved[field], field), rel_tol=1e-12, abs_tol=1e-8):
                raise ValueError(f"Radius regrouping changed the source case denominator: {case}/{field}")
        records.append({"case": case, **result["summary"], "bins": result["bins"],
                        "segments": result["segments"], "total_segment_count": total_count,
                        "unavailable_segment_count": unavailable,
                        "area_coverage_fraction": _number(saved["area_coverage_fraction"], "area_coverage_fraction")})
    positive = [row["case"] for row in records if row["total_volume_mm3"] > 0]
    if source.get("included_cases") != positive or source.get("included_case_count") != len(positive):
        raise ValueError("Radius regrouping changed source eligible membership")
    return records, hashes, source


def _statistics(records: list[dict]) -> list[dict]:
    rows = []
    for index, definition in enumerate(BIN_DEFINITIONS):
        values = np.asarray([row["bins"][index]["volume_fraction"] for row in records], dtype=float)
        n = len(values)
        quantiles = np.percentile(values, (0, 10, 25, 50, 75, 100), method="linear") if n else [None] * 6
        rows.append({"radius_bin": definition["radius_bin"], "label": definition["label"],
                     "metric": "volume_fraction", "unit": "fraction", "n": n,
                     "mean": float(values.mean()) if n else None,
                     "std": float(values.std(ddof=1)) if n > 1 else None,
                     **dict(zip(("min", "p10", "p25", "p50", "p75", "max"),
                                [float(value) if value is not None else None for value in quantiles]))})
    return rows


def _table(bins: list[dict]) -> list[str]:
    lines = ["| Radius (mm) | Measurable segments | Estimated volume (mL) | Estimated volume fraction (%) |",
             "|---|---:|---:|---:|"]
    for row in bins:
        fraction = "Unavailable" if row["volume_fraction"] is None else f"{100 * row['volume_fraction']:.3f}"
        lines.append(f"| {row['label']} | {row['segment_count']:,} | {row['volume_ml']:.6f} | {fraction} |")
    return lines


def _markdown(records, pooled, statistics, figures):
    included = [row for row in records if row["total_volume_mm3"] > 0]
    lines = ["# PARSE estimated pulmonary arterial volume fractions by segment radius", "",
             f"Source cases: **{len(records)}**. Cases with a positive estimated-volume denominator: **{len(included)}**.",
             "", METHOD_TEXT, "", LIMITATION, "", "[Image gallery](index.html) · "
             "[Per-case values](per_case_radius_volumes.csv) · [Case totals](case_summary.csv) · "
             "[Measurable segment estimates](all_segment_radius_volumes.csv) · [Sources](sources.json)",
             "", "## Patient distributions", "",
             "Each histogram contains one equally weighted patient fraction per radius interval, including "
             "genuine zero class volumes. Patients are not weighted by vessel volume.", ""]
    for start in (0, 2):
        selected = figures[start:start + 2]
        if selected:
            lines += ["| " + " | ".join(figure["radius_bin"] for figure in selected) + " |",
                      "| " + " | ".join("---" for figure in selected) + " |",
                      "| " + " | ".join(f"[![Estimated volume fraction]({figure['png']})]({figure['png']}) "
                                           f"[PDF]({figure['pdf']})" for figure in selected) + " |", ""]
    lines += ["## Patient percentiles", "",
              "Values below are percentages; full-precision CSV values use a 0–1 scale. "
              "Percentiles use linear interpolation; P50 is the median.", "",
              "| Radius (mm) | N | P10 (%) | P25 (%) | P50 (%) | P75 (%) |",
              "|---|---:|---:|---:|---:|---:|"]
    for row in statistics:
        values = ["Unavailable" if row[key] is None else f"{100 * row[key]:.3f}"
                  for key in ("p10", "p25", "p50", "p75")]
        lines.append(f"| {row['label']} | {row['n']} | " + " | ".join(values) + " |")
    lines += ["", "[Patient statistics](cohort_volume_statistics.csv) · [Histogram frequencies](cohort_histogram_bins.csv)",
              "", "## Pooled estimated volumes", "",
              "Pooled fractions divide summed class estimates by the summed total estimate across patients. "
              "These fractions differ from the equally weighted patient statistics above.", "", *_table(pooled),
              "", "## Individual cases", "", "| Case | Total estimated volume (mL) | Plot |", "|---|---:|---|"]
    for row in included:
        figure = row["figures"][0]
        lines.append(f"| [{row['case']}](cases/{row['case']}/report.md) | {row['total_volume_ml']:.6f} | "
                     f"[PNG]({figure['png']}) · [PDF]({figure['pdf']}) |")
    return "\n".join(lines) + "\n"


def _gallery(records, figures):
    def chart(figure, label):
        png, pdf = html.escape(figure["png"], quote=True), html.escape(figure["pdf"], quote=True)
        return (f'<section><h2>{html.escape(label)}</h2><a href="{png}"><img src="{png}" '
                f'alt="{html.escape(label, quote=True)}" loading="lazy"></a>'
                f'<p><a href="{png}">PNG</a> · <a href="{pdf}">PDF</a></p></section>')
    content = "".join(chart(figure, figure["radius_bin"]) for figure in figures)
    content += "".join(chart(row["figures"][0], row["case"]) for row in records if row.get("figures"))
    return ('<!doctype html><html lang="en"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            '<title>PARSE estimated arterial volume fractions by radius</title><style>'
            'body{margin:0;background:white;color:#303030;font:14px/1.45 Arial,sans-serif}'
            'main{max-width:1200px;margin:auto;padding:16px}h1{font-size:23px}h2{font-size:16px}'
            'a{color:#345f84}.charts{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px}'
            'section{min-width:0}img{display:block;width:100%;height:auto}'
            '@media(max-width:640px){.charts{grid-template-columns:1fr}}'
            '</style></head><body><main><h1>PARSE estimated arterial volume fractions by radius</h1>'
            f'<p>{html.escape(METHOD_TEXT)}</p><p>{html.escape(LIMITATION)}</p>'
            '<p><a href="report.md">Report and percentiles</a> · '
            '<a href="per_case_radius_volumes.csv">Patient values</a> · '
            '<a href="sources.json">Sources and definitions</a></p>'
            f'<div class="charts">{content}</div></main></body></html>\n')


def generate(publication_root: Path, output_root: Path) -> dict:
    source_root, output_root = Path(publication_root).resolve(), Path(output_root).resolve()
    if source_root.is_relative_to(output_root) or output_root.is_relative_to(source_root):
        raise ValueError("Radius-volume outputs must be separate from source measurements")
    dataset_root = next((path for path in source_root.parents
                         if path.name == "parse" and path.parent.name == "data"), None)
    report_roots = [dataset_root / "reports"] if dataset_root is not None else []
    statistics_root = next((path for path in source_root.parents
                            if path.name == "statistics" and dataset_root is not None
                            and path.is_relative_to(dataset_root)), None)
    if statistics_root is not None:
        report_roots.append(statistics_root)
    if not any(output_root != root and output_root.is_relative_to(root) for root in report_roots):
        raise ValueError("Radius-volume reports must remain under data/parse/reports "
                         "or the source run's statistics directory")
    records, hashes, source = _load_publication(source_root)
    publication_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ_") + uuid.uuid4().hex[:8]
    final = output_root / publication_id
    included = [row for row in records if row["total_volume_mm3"] > 0]
    total = math.fsum(row["total_volume_mm3"] for row in records)
    pooled = []
    for index, definition in enumerate(BIN_DEFINITIONS):
        volume = math.fsum(row["bins"][index]["volume_mm3"] for row in records)
        pooled.append({"radius_bin": definition["radius_bin"], "label": definition["label"],
                       "segment_count": sum(row["bins"][index]["segment_count"] for row in records),
                       "volume_mm3": volume, "volume_ml": volume / 1000,
                       "total_volume_mm3": total, "total_volume_ml": total / 1000,
                       "volume_fraction": volume / total if total else None})
    statistics = _statistics(included)
    chart_options = {"bin_key": "radius_bin", "bin_ids": BIN_IDS, "tick_labels": TICK_LABELS,
                     "x_label": "Radius (mm)", "stem": "radius_bin_volume_fractions", "method": METHOD}
    output_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".radius_volumes_", dir=output_root) as temporary:
        staging = Path(temporary)
        all_segments = [segment for row in records for segment in row["segments"]]
        fields = list(all_segments[0]) if all_segments else ["source", "case", "branch_id", "radius_bin"]
        _write_csv(staging / "all_segment_radius_volumes.csv", all_segments, fields)
        _write_csv(staging / "per_case_radius_volumes.csv",
                   [{"case": row["case"], **group} for row in records for group in row["bins"]], BIN_FIELDS)
        _write_csv(staging / "case_summary.csv", records, SUMMARY_FIELDS)
        _write_csv(staging / "cohort_radius_volumes.csv", pooled, BIN_FIELDS[1:])
        _write_csv(staging / "cohort_volume_statistics.csv", statistics, STATISTIC_FIELDS)
        for row in records:
            case_dir = staging / "cases" / row["case"]
            case_dir.mkdir(parents=True)
            _write_csv(case_dir / "radius_volumes.csv", row["bins"], BIN_FIELDS[1:])
            row["figures"] = render_case(row, case_dir, **chart_options) if row["total_volume_mm3"] > 0 else []
            lines = [f"# {row['case']}: estimated arterial volume fractions by radius", "", METHOD_TEXT, "",
                     f"Total estimated volume: **{row['total_volume_ml']:.6f} mL**. "
                     f"Measurable segments: **{row['classified_segment_count']} / {row['total_segment_count']}**. "
                     f"Unavailable-area segments excluded: **{row['unavailable_segment_count']}**.", "",
                     *_table(row["bins"]), ""]
            for figure in row["figures"]:
                lines += [f"![Estimated volume fractions]({Path(figure['png']).name})", "",
                          f"[PDF]({Path(figure['pdf']).name})", ""]
                for key in ("png", "pdf"):
                    figure[key] = f"cases/{row['case']}/{Path(figure[key]).name}"
            lines += [LIMITATION, "", "[Class values](radius_volumes.csv)", ""]
            (case_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
            (case_dir / "plot_report.json").write_text(json.dumps(
                {key: value for key, value in row.items() if key != "segments"}, indent=2, allow_nan=False) + "\n")
        figures, histogram_rows = render_cohort(included, staging / "cohort", **chart_options)
        for figure in figures:
            for key in ("png", "pdf"):
                figure[key] = f"cohort/{Path(figure[key]).name}"
        _write_csv(staging / "cohort_histogram_bins.csv", histogram_rows, HISTOGRAM_FIELDS)
        (staging / "report.md").write_text(_markdown(records, pooled, statistics, figures), encoding="utf-8")
        (staging / "index.html").write_text(_gallery(records, figures), encoding="utf-8")
        implementation = (Path(__file__).resolve(), *[Path(__file__).with_name(name).resolve() for name in (
            "parse_radius_volumes.py", "parse_radius_statistics.py", "area_volume_charts.py",
            "area_bin_charts.py", "plot_parse_distributions.py", "plot_parse_radius_case.py",
            "parse_cross_section_statistics.py")])
        metadata = {"schema_version": 1, "status": "complete", "dataset": "PARSE", "source": "prediction",
                    "publication_id": publication_id, "source_publication": str(source_root), "output_dir": str(final),
                    "case_count": len(records), "included_case_count": len(included),
                    "source_cases": [row["case"] for row in records], "included_cases": [row["case"] for row in included],
                    "excluded_cases": [{"case": row["case"], "reason": "no_positive_estimated_volume"}
                                       for row in records if row["total_volume_mm3"] <= 0],
                    "bins": list(BIN_DEFINITIONS), "volume_estimation_method": METHOD,
                    "definitions": dict(DEFINITIONS), "case_provenance": source.get("case_provenance", []),
                    "case_summaries": [{key: row[key] for key in SUMMARY_FIELDS} for row in records],
                    "cases": [{key: value for key, value in row.items() if key != "segments"} for row in records],
                    "cohort_bins": pooled, "cohort_figures": figures, "cohort_statistics": statistics,
                    "source_sha256": hashes, "recorded_source_sha256": source.get("source_sha256", {}),
                    "recorded_source_code_sha256": source.get("code_sha256", {}),
                    "code_sha256": {str(path): _sha256(path) for path in implementation},
                    "geometry_changed": False, "radii_recomputed": False, "areas_recomputed": False,
                    "review_status_promotion": False,
                    "verification": {"mode": "saved_measurements", "numeric_receipts_verified": True,
                                     "clinical_sources_rehashed": False, "source_case_denominators_preserved": True},
                    "versions": {"python": platform.python_version(), "numpy": np.__version__},
                    "report_path": str(final / "report.md"), "index_path": str(final / "index.html"),
                    "artifact_sha256": {str(final / path.relative_to(staging)): _sha256(path)
                                        for path in sorted(staging.rglob("*")) if path.is_file()}}
        (staging / "sources.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
        staging.rename(final)
    latest = {"publication_id": publication_id, "publication_dir": str(final),
              "report_path": metadata["report_path"], "index_path": metadata["index_path"]}
    with tempfile.NamedTemporaryFile(mode="w", dir=output_root, prefix=".latest_", delete=False) as handle:
        json.dump(latest, handle, indent=2)
        handle.write("\n")
        latest_path = Path(handle.name)
    latest_path.replace(output_root / "latest.json")
    return metadata


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--publication-root", type=Path, required=True,
                        help="Existing complete PARSE cross-sectional area-volume publication")
    parser.add_argument("--output-root", type=Path, default=Path("data/parse/reports/artery_radius_volumes"))
    args = parser.parse_args(argv)
    try:
        result = generate(args.publication_root, args.output_root)
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"Radius-volume report failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    print(f"Completed PARSE estimated radius-volume fractions for {result['included_case_count']}/{result['case_count']} cases.")
    print(result["report_path"])
    print(result["index_path"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
