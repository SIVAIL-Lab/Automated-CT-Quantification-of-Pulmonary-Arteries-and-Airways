#!/usr/bin/env python3
"""Publish estimated PARSE vessel-volume fractions by mean cross-sectional area."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import html
import json
import math
from pathlib import Path
import platform
import sys
import tempfile
import uuid

import numpy as np

from area_volume_charts import render_case, render_cohort
from parse_area_volumes import DEFINITIONS, METHOD, estimate_segment_volumes
from parse_cross_section_bins import BIN_DEFINITIONS, BIN_IDS
from parse_cross_section_statistics import (
    REVIEW_FIELDS, _collect_inputs, _csv_value, _publish_verified, _relative,
    _sha256, _validated_case, _verify, _write_csv,
)
from report_parse_cross_section_bins import _load_run, _merge_hashes
from segment_cross_sections import METHOD as AREA_METHOD


BIN_FIELDS = ("area_bin", "label", "segment_count", "volume_mm3", "volume_ml",
              "total_volume_mm3", "total_volume_ml", "volume_fraction")
SUMMARY_FIELDS = ("classified_segment_count", "total_volume_mm3", "total_volume_ml", "measured_length_mm")
HISTOGRAM_FIELDS = ("area_bin", "metric", "unit", "bin_index", "lower_bound", "upper_bound",
                    "upper_inclusive", "case_count", "included_case_count")
STATISTIC_FIELDS = ("area_bin", "metric", "unit", "n", "mean", "std", "min", "p10", "p25", "p50", "p75", "max")
LIMITATION = (
    "These are cross-sectional integral estimates, not voxel-mask volumes or a spatial partition of the "
    "segmentation. Contributions can overlap near branches or nearby centerline paths; estimates may "
    "substantially exceed the segmentation-mask volume. Existing review flags remain in effect, and "
    "this calculation does not establish anatomical validity."
)
METHOD_TEXT = (
    "Each segment estimate is its saved, unrounded mean area multiplied by its measured length "
    "(mm² × mm = mm³; 1,000 mm³ = 1 mL). Partial segments contribute only their measured intervals, "
    "without filling gaps or extrapolating to the full segment length. The original mean area selects "
    "BV5 (<5 mm²), BV5-10 (5–10 mm², inclusive), or BV10 (>10 mm²). Each class fraction divides its "
    "estimated volume by the total estimated volume of measurable segments in the same case; the "
    "three fractions sum to 100% for a positive denominator."
)


def _statistics(records: list[dict]) -> list[dict]:
    """Summarize equally weighted patient fractions; save fractions on a 0–1 scale."""
    rows = []
    for index, key in enumerate(BIN_IDS):
        values = np.asarray([record["bins"][index]["volume_fraction"] for record in records], dtype=float)
        n = len(values)
        percentiles = np.percentile(values, [0, 10, 25, 50, 75, 100], method="linear") if n else [None] * 6
        rows.append({"area_bin": key, "metric": "volume_fraction", "unit": "fraction", "n": n,
                     "mean": float(np.mean(values)) if n else None,
                     "std": float(np.std(values, ddof=1)) if n > 1 else None,
                     **dict(zip(("min", "p10", "p25", "p50", "p75", "max"),
                                [float(value) if value is not None else None for value in percentiles]))})
    return rows


def _table(bins: list[dict]) -> list[str]:
    lines = ["| Area class | Measurable segments | Estimated volume (mL) | Estimated volume fraction (%) |",
             "|---|---:|---:|---:|"]
    for row in bins:
        fraction = "—" if row["volume_fraction"] is None else f"{100 * row['volume_fraction']:.3f}"
        lines.append(f"| {row['label']} | {row['segment_count']:,} | {row['volume_ml']:.6f} | {fraction} |")
    return lines


def _case_report(record: dict, item: dict, directory: Path) -> str:
    lines = [f"# {record['case']}: estimated vessel-volume fractions", "", METHOD_TEXT, "",
             f"Total estimated volume of measurable segments: **{item['total_volume_ml']:.6f} mL**. "
             f"Measured segment length: **{item['measured_length_mm']:.6f} mm**.", "", *_table(item["bins"]), ""]
    for figure in item["figures"]:
        png, pdf = Path(figure["png"]).name, Path(figure["pdf"]).name
        lines += [f"![Estimated volume fractions]({png})", "", f"[PDF]({pdf})", ""]
    lines += [LIMITATION, "", "[Measurable segment estimates](segment_area_volumes.csv) · "
              "[Class estimates](area_volumes.csv)", "",
              f"[Original mean-area table](<{_relative(record['segment_areas_path'], directory)}>) · "
              f"[Original local areas](<{_relative(record['samples_path'], directory)}>)", "",
              "Inherited review and segmentation provenance:", "", "```json",
              json.dumps({field: record.get(field, "not_recorded") for field in REVIEW_FIELDS}, indent=2),
              "```", ""]
    return "\n".join(lines)


def _markdown(summary: dict, pooled: list[dict], cases: list[dict], figures: list[dict],
              statistics: list[dict]) -> str:
    lines = ["# PARSE estimated vessel-volume fractions by cross-sectional area", "",
             f"Source prediction cases: **{summary['case_count']}**. "
             f"Cases with a positive estimated-volume denominator: **{len(cases)}**.", "", METHOD_TEXT, "",
             LIMITATION, "", "[Image gallery](index.html) · [Patient values](plot_values.csv) · "
             "[All measurable segment estimates](all_segment_area_volumes.csv) · "
             "[Per-case class estimates](per_case_area_volumes.csv) · [Case totals](case_summary.csv) · "
             "[Sources and definitions](sources.json)", "", "## Patient distributions", "",
             "Each histogram contains one equally weighted patient observation per area class. "
             "A true zero class volume remains a zero observation when that patient has a positive "
             "total estimate. The histograms do not pool segments or weight patients by vessel volume.", ""]
    if figures:
        lines += ["| BV5 | BV5-10 | BV10 |", "|---|---|---|", "| " + " | ".join(
            f"[![{figure['area_bin']} estimated volume fractions]({figure['png']})]({figure['png']}) "
            f"[PDF]({figure['pdf']})" for figure in figures) + " |", ""]
    lines += ["[Histogram bins](cohort_histogram_bins.csv) · "
              "[Patient summary statistics](cohort_volume_statistics.csv)", "",
              "Patient statistics below are percentages; CSV values retain fractions on a 0–1 scale. "
              "SD is the sample standard deviation (n−1 denominator). Percentiles use linear interpolation; "
              "Min and Max correspond to the 0th and 100th percentiles.", "",
              "| Area class | Patients | Mean (%) | SD (%) | Min (%) | P10 (%) | P25 (%) | P50 (%) | P75 (%) | Max (%) |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in statistics:
        values = ["—" if row[key] is None else f"{100 * row[key]:.3f}"
                  for key in ("mean", "std", "min", "p10", "p25", "p50", "p75", "max")]
        lines.append(f"| {row['area_bin']} | {row['n']} | " + " | ".join(values) + " |")
    lines += ["", "## Pooled estimated volumes", "",
              "This table sums estimated volumes across cases. Its fractions divide summed class estimates "
              "by the summed total estimate; these pooled fractions differ from the equally weighted "
              "patient distributions above.", "", *_table(pooled), "",
              "[Pooled class estimates](cohort_area_volumes.csv)", "", "## Individual cases", "",
              "| Case | Estimated measurable volume (mL) | Estimated volume fractions |", "|---|---:|---|"]
    for item in cases:
        links = " · ".join(f"[PNG]({figure['png']}) / [PDF]({figure['pdf']})" for figure in item["figures"])
        lines.append(f"| [{item['case']}](cases/{item['case']}/report.md) | {item['total_volume_ml']:.6f} | {links} |")
    return "\n".join(lines) + "\n"


def _chart(figure: dict, label: str) -> str:
    png, pdf = html.escape(figure["png"], quote=True), html.escape(figure["pdf"], quote=True)
    return (f'<section><h3>{html.escape(label)}</h3><a href="{png}"><img src="{png}" '
            f'loading="lazy" alt="{html.escape(label, quote=True)}"></a>'
            f'<p><a href="{png}">PNG</a> · <a href="{pdf}">PDF</a></p></section>')


def _gallery(cases: list[dict], figures: list[dict]) -> str:
    patients = []
    for item in cases:
        case = item["case"]
        patients.append(f'<article id="{case}"><h2><a href="cases/{case}/report.md">{case}</a></h2>'
                        f'<p>Estimated measurable volume: {item["total_volume_ml"]:.6f} mL</p>' +
                        "".join(_chart(figure, "Estimated volume fraction by area class")
                                for figure in item["figures"]) +
                        f'<p><a href="cases/{case}/plot_report.json">Plot values and provenance</a></p></article>')
    return ('<!doctype html><html lang="en"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            '<title>PARSE estimated vessel-volume fractions</title><style>'
            'body{margin:0;background:white;color:#303030;font:14px/1.45 Arial,sans-serif}'
            'main{max-width:1400px;margin:auto;padding:20px}h1{font-size:24px}h2{font-size:18px}'
            'h3{font-size:14px;font-weight:normal;margin:8px 0}a{color:#345f84}'
            '.charts{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:16px}'
            'article,section{min-width:0}article{border:1px solid #ddd;padding:12px}'
            'img{display:block;width:100%;height:auto}'
            '@media(max-width:1000px){.charts{grid-template-columns:repeat(2,minmax(0,1fr))}}'
            '@media(max-width:640px){.charts{grid-template-columns:1fr}}'
            '</style></head><body><main><h1>PARSE estimated vessel-volume fractions</h1>'
            f'<p>{html.escape(METHOD_TEXT)}</p><p>{html.escape(LIMITATION)}</p>'
            '<p><a href="report.md">Markdown report</a> · <a href="plot_values.csv">Patient values</a> · '
            '<a href="cohort_volume_statistics.csv">Patient summary statistics</a> · '
            '<a href="cohort_histogram_bins.csv">Histogram bins</a> · <a href="sources.json">Provenance</a></p>'
            '<h2>Across patients</h2><p>One equally weighted patient observation per area class; '
            'true zero class estimates are retained for patients with a positive total estimate.</p><div class="charts">' +
            "".join(_chart(figure, f"{figure['area_bin']}: patient estimated volume fractions") for figure in figures) +
            f'</div><h2>Individual cases ({len(cases)})</h2><div class="charts">' + "".join(patients) +
            '</div></main></body></html>\n')


def _check_summary(summary: dict, original: dict) -> None:
    for new, old in (("total_segment_count", "segment_count"),
                     ("classified_segment_count", "segment_area_count"),
                     ("unavailable_segment_count", "unavailable_count")):
        if summary[new] != original[old]:
            raise ValueError("Estimated-volume summaries must conserve original segment and area counts")
    for field in ("total_length_mm", "measured_length_mm", "area_coverage_fraction"):
        if not math.isclose(summary[field], original[field], rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("Estimated-volume summaries must preserve original length coverage")


def generate(source_run: Path, output_root: Path) -> dict:
    """Publish estimates and patient distributions from immutable saved area integrals."""
    source_run, output_root = Path(source_run).resolve(), Path(output_root).resolve()
    if output_root.is_relative_to(source_run) or source_run.is_relative_to(output_root):
        raise ValueError("Estimated-volume outputs must be separate from source computations")
    source_report, records, hashes = _load_run(source_run)
    publication_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ_") + uuid.uuid4().hex[:8]
    final = output_root / publication_id
    _merge_hashes(hashes, _collect_inputs(records, final))
    for name in hashes:
        parent = Path(name).parent
        if parent == output_root or parent in output_root.parents or output_root in parent.parents:
            raise ValueError("Estimated-volume outputs must be separate from original input directories")
    names = ("report_parse_area_volumes.py", "parse_area_volumes.py", "area_volume_charts.py", "area_bin_charts.py",
             "parse_cross_section_bins.py", "report_parse_cross_section_bins.py",
             "parse_cross_section_statistics.py", "segment_cross_sections.py",
             "section_measurement.py", "measurement_io.py", "run_edt_experiment.py",
             "parse_radius_statistics.py", "plot_parse_distributions.py")
    code_hashes = {str(Path(__file__).with_name(name).resolve()): _sha256(Path(__file__).with_name(name))
                   for name in names}
    import matplotlib

    output_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".area_volumes_", dir=output_root) as temporary:
        staging = Path(temporary)
        case_rows, case_bins, plotted, exclusions, plot_values = [], [], [], [], []
        fields = None
        with (staging / "all_segment_area_volumes.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = None
            for index, record in enumerate(records, 1):
                original, original_summary = _validated_case(record)
                grouped = estimate_segment_volumes(original)
                summary = grouped["summary"]
                _check_summary(summary, original_summary)
                current_fields = list(dict.fromkeys(["source", "case", *[
                    key for row in original for key in row], "area_bin", "estimated_volume_mm3", "estimated_volume_ml"]))
                if writer is None:
                    fields = current_fields
                    writer = csv.DictWriter(stream, fieldnames=fields)
                    writer.writeheader()
                elif fields != current_fields:
                    raise ValueError("Segment table fields differ across source cases")
                rows = [{"source": "prediction", "case": record["case"], **row} for row in grouped["segments"]]
                if len(rows) != summary["classified_segment_count"]:
                    raise ValueError("Exported segments differ from the measurable segment count")
                for row in rows:
                    writer.writerow({key: _csv_value(row.get(key)) for key in fields})
                directory = staging / "cases" / record["case"]
                directory.mkdir(parents=True)
                _write_csv(directory / "segment_area_volumes.csv", rows, fields)
                bins = [{"source": "prediction", "case": record["case"], **row} for row in grouped["bins"]]
                _write_csv(directory / "area_volumes.csv", bins, ("source", "case", *BIN_FIELDS))
                item = {"source": "prediction", "case": record["case"], **summary, "bins": grouped["bins"]}
                figures = render_case(item, directory) if summary["total_volume_mm3"] > 0 else []
                for figure in figures:
                    for extension in ("png", "pdf"):
                        figure[extension] = f"cases/{record['case']}/{figure[extension]}"
                item["figures"] = figures
                if summary["total_volume_mm3"] > 0:
                    plotted.append(item)
                    plot_values.extend(bins)
                    (directory / "plot_report.json").write_text(json.dumps({**item,
                        "volume_estimation_method": METHOD, "definitions": DEFINITIONS, "limitation": LIMITATION,
                        "publication_root": str(final), "sources_path": str(final / "sources.json")},
                        indent=2, allow_nan=False) + "\n", encoding="utf-8")
                else:
                    exclusions.append({"case": record["case"], "reason": "no_positive_estimated_volume"})
                (directory / "report.md").write_text(
                    _case_report(record, item, final / "cases" / record["case"]), encoding="utf-8")
                case_rows.append({"source": "prediction", "case": record["case"], **summary})
                case_bins.extend(bins)
                if index % 10 == 0 or index == len(records):
                    print(f"Validated estimates and rendered {index}/{len(records)} PARSE cases", flush=True)
        total_volume = math.fsum(row["total_volume_mm3"] for row in case_rows)
        summary = {"case_count": len(records), "included_case_count": len(plotted),
                   **{field: sum(row[field] for row in case_rows)
                      for field in ("total_segment_count", "classified_segment_count", "unavailable_segment_count")},
                   "total_volume_mm3": total_volume, "total_volume_ml": total_volume / 1000.,
                   **{field: math.fsum(row[field] for row in case_rows)
                      for field in ("total_length_mm", "measured_length_mm")}}
        summary["area_coverage_fraction"] = summary["measured_length_mm"] / summary["total_length_mm"]
        pooled = []
        for key in BIN_IDS:
            local = [row for row in case_bins if row["area_bin"] == key]
            volume = math.fsum(row["volume_mm3"] for row in local)
            pooled.append({"source": "prediction", "case_count": len(records), "area_bin": key, "label": key,
                           "segment_count": sum(row["segment_count"] for row in local),
                           "volume_mm3": volume, "volume_ml": volume / 1000.,
                           "total_volume_mm3": total_volume, "total_volume_ml": total_volume / 1000.,
                           "volume_fraction": volume / total_volume if total_volume else None})
        if (sum(row["segment_count"] for row in pooled) != summary["classified_segment_count"]
                or not math.isclose(math.fsum(row["volume_mm3"] for row in pooled), total_volume,
                                    rel_tol=1e-12, abs_tol=1e-12)):
            raise ValueError("Pooled estimates do not conserve measurable segments and estimated volume")
        cohort_dir = staging / "cohort"
        cohort_dir.mkdir()
        figures, histogram_rows = render_cohort(plotted, cohort_dir) if plotted else ([], [])
        for figure in figures:
            for extension in ("png", "pdf"):
                figure[extension] = f"cohort/{figure[extension]}"
        statistics = _statistics(plotted)
        _write_csv(staging / "per_case_area_volumes.csv", case_bins, ("source", "case", *BIN_FIELDS))
        _write_csv(staging / "cohort_area_volumes.csv", pooled, ("source", "case_count", *BIN_FIELDS))
        _write_csv(staging / "case_summary.csv", case_rows, ("source", "case", *SUMMARY_FIELDS))
        _write_csv(staging / "plot_values.csv", plot_values, ("source", "case", *BIN_FIELDS))
        _write_csv(staging / "cohort_histogram_bins.csv", histogram_rows, HISTOGRAM_FIELDS)
        _write_csv(staging / "cohort_volume_statistics.csv", statistics, STATISTIC_FIELDS)
        (staging / "report.md").write_text(_markdown(summary, pooled, plotted, figures, statistics), encoding="utf-8")
        (staging / "index.html").write_text(_gallery(plotted, figures), encoding="utf-8")
        result = {
            "schema_version": 1, "status": "complete", "dataset": "PARSE", "source": "prediction",
            "run_id": source_report["run_id"], "publication_id": publication_id,
            "source_run": str(source_run), "output_dir": str(final), "case_count": len(records),
            "source_cases": [record["case"] for record in records],
            "included_case_count": len(plotted), "included_cases": [item["case"] for item in plotted],
            "excluded_cases": exclusions, "summary": summary, "case_summaries": case_rows,
            "cases": plotted, "bins": list(BIN_DEFINITIONS), "cohort_bins": pooled,
            "cohort_figures": figures, "cohort_statistics": statistics,
            "definitions": {**DEFINITIONS,
                "cohort_observation": "One equally weighted patient estimated-volume fraction per area class",
                "pooled_fraction": "Summed class estimated volume divided by summed total estimated volume across cases",
                "statistical_units": "CSV fractions use a 0-1 scale; plot axes and Markdown statistics use percent",
                "standard_deviation": "Sample standard deviation with n-1 denominator; null for n < 2",
                "percentiles": "Linear interpolation at 0, 10, 25, 50, 75 and 100 percent; 0 is min and 100 is max"},
            "limitation": LIMITATION, "segment_area_method": AREA_METHOD, "volume_estimation_method": METHOD,
            "geometry_changed": False, "areas_recomputed": False, "review_status_promotion": False,
            "case_provenance": [{"case": record["case"], **{field: record.get(field, "not_recorded")
                                 for field in REVIEW_FIELDS}} for record in records],
            "index_path": str(final / "index.html"), "report_path": str(final / "report.md"),
            "sources_path": str(final / "sources.json"), "source_sha256": hashes, "code_sha256": code_hashes,
            "versions": {"python": platform.python_version(), "numpy": np.__version__,
                         "matplotlib": matplotlib.__version__},
            "input_integrity": "verified_after_publication",
            "artifact_sha256": {str(final / path.relative_to(staging)): _sha256(path)
                                for path in staging.rglob("*") if path.is_file()},
        }
        _verify(hashes)
        _verify(code_hashes)
        (staging / "sources.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        return _publish_verified(staging, final, result, hashes, code_hashes)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run-dir", type=Path, required=True,
                        help="Completed cross-section computation directory from this reproduction run.")
    parser.add_argument("--output-root", type=Path, default=Path("data/parse/reports/cross_section_area_volumes"))
    args = parser.parse_args(argv)
    try:
        result = generate(args.source_run_dir, args.output_root)
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"Estimated-volume publication failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    print(f"Completed estimated-volume plots for {result['included_case_count']} PARSE cases.")
    print(result["index_path"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
