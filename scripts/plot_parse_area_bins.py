#!/usr/bin/env python3
"""Plot PARSE area-class counts and proportions for each case and across patients."""
from __future__ import annotations

import argparse
from collections import Counter
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

from area_bin_charts import render_case, render_cohort
from parse_cross_section_bins import BIN_DEFINITIONS, BIN_IDS
from parse_cross_section_statistics import _publish_verified, _relative, _sha256, _verify, _write_csv
from report_parse_cross_section_bins import BIN_FIELDS, DEFINITIONS, _merge_hashes


HISTOGRAM_FIELDS = ("area_bin", "metric", "unit", "bin_index", "lower_bound", "upper_bound",
                    "upper_inclusive", "case_count", "included_case_count")


def _integer(value, name):
    if isinstance(value, bool) or not re.fullmatch(r"[0-9]+", str(value)):
        raise ValueError(f"Expected a nonnegative integer: {name}")
    return int(value)


def _load_publication(publication_root: Path) -> tuple[list[dict], dict, dict]:
    """Read the saved classes without remeasuring or regrouping the vessels."""
    manifest_path = publication_root / "sources.json"
    raw = manifest_path.read_bytes()
    manifest = json.loads(raw)
    if (not isinstance(manifest, dict) or manifest.get("schema_version") != 2 or manifest.get("dataset") != "PARSE"
            or manifest.get("source") != "prediction" or manifest.get("status") != "complete"
            or manifest.get("input_integrity") != "verified_after_publication"
            or manifest.get("bins") != list(BIN_DEFINITIONS)
            or manifest.get("definitions", {}).get("fraction") != DEFINITIONS["fraction"]):
        raise ValueError("Expected a complete PARSE area-class publication using valid-area denominators")
    cases = manifest.get("included_cases")
    if (not isinstance(cases, list) or not cases
            or any(not isinstance(case, str) or not re.fullmatch(r"PA[0-9]{6}", case) for case in cases)
            or len(set(cases)) != len(cases) or manifest.get("case_count") != len(cases)):
        raise ValueError("Area publication must contain unique, complete PARSE case membership")
    hashes = {str(manifest_path): hashlib.sha256(raw).hexdigest()}
    for field in ("source_sha256", "code_sha256", "artifact_sha256"):
        _merge_hashes(hashes, manifest.get(field, {}))
    _verify(hashes)

    def read_csv(name, required):
        path = publication_root / name
        content = path.read_bytes()
        if manifest.get("artifact_sha256", {}).get(str(path)) != hashlib.sha256(content).hexdigest():
            raise ValueError(f"Plot input is not bound to its source publication: {name}")
        reader = csv.DictReader(io.StringIO(content.decode("utf-8-sig")))
        if not set(required) <= set(reader.fieldnames or []):
            raise ValueError(f"Plot input is missing required columns: {name}")
        return list(reader)

    summary_rows = read_csv("case_summary.csv", ("source", "case", "classified_segment_count"))
    denominators = {}
    declared = {row["case"]: row for row in manifest["case_summaries"]}
    if len(declared) != len(manifest["case_summaries"]) or set(declared) != set(cases):
        raise ValueError("Case summary metadata differs from the source membership")
    for row in summary_rows:
        case = row["case"]
        if row["source"] != "prediction" or case not in cases or case in denominators:
            raise ValueError("Unexpected source, unknown case or duplicate case summary")
        denominator = _integer(row["classified_segment_count"], "classified_segment_count")
        if denominator != declared[case]["classified_segment_count"]:
            raise ValueError("Published case denominator differs from its source summary")
        denominators[case] = denominator
    if set(denominators) != set(cases):
        raise ValueError("Missing case denominator")
    grouped = {case: {} for case in cases}
    for row in read_csv("per_case_area_bins.csv", ("source", "case", *BIN_FIELDS)):
        case, key = row["case"], row["area_bin"]
        if (row["source"] != "prediction" or case not in grouped or key not in BIN_IDS
                or key in grouped[case] or row["label"] != key):
            raise ValueError("Unexpected source, case, class label or duplicate area-bin row")
        count = _integer(row["segment_count"], "segment_count")
        denominator = _integer(row["classified_segment_count"], "classified_segment_count")
        if denominator != denominators[case] or count > denominator:
            raise ValueError("Area-bin denominator or count disagrees with its case")
        fraction = float(row["fraction"]) if row["fraction"] else None
        if denominator:
            if (fraction is None or not math.isfinite(fraction)
                    or not math.isclose(fraction, count / denominator, rel_tol=1e-12, abs_tol=1e-15)):
                raise ValueError("Area-bin fraction disagrees with its valid-area denominator")
        elif count != 0 or fraction is not None:
            raise ValueError("Zero-denominator cases require zero counts and null fractions")
        grouped[case][key] = {"area_bin": key, "label": key, "segment_count": count,
                              "classified_segment_count": denominator, "fraction": fraction}
    records, counts = [], Counter()
    for case in sorted(cases):
        if set(grouped[case]) != set(BIN_IDS):
            raise ValueError("Each case requires exactly three area classes")
        bins = [grouped[case][key] for key in BIN_IDS]
        if sum(row["segment_count"] for row in bins) != denominators[case]:
            raise ValueError("Area class counts do not sum to the valid-area denominator")
        counts.update({row["area_bin"]: row["segment_count"] for row in bins})
        records.append({"case": case, "source": "prediction",
                        "classified_segment_count": denominators[case], "bins": bins})
    if sum(denominators.values()) != manifest["summary"]["classified_segment_count"]:
        raise ValueError("Cohort valid-area denominator differs from its cases")
    pooled = manifest.get("cohort_bins", [])
    if len(pooled) != 3 or {row["area_bin"] for row in pooled} != set(BIN_IDS):
        raise ValueError("Missing or duplicate pooled area classes")
    for row in pooled:
        denominator = sum(denominators.values())
        expected = counts[row["area_bin"]] / denominator if denominator else None
        if (row["segment_count"] != counts[row["area_bin"]]
                or row["classified_segment_count"] != denominator
                or (expected is None and row["fraction"] is not None)
                or (expected is not None and (row["fraction"] is None or not math.isclose(
                    row["fraction"], expected, rel_tol=1e-12, abs_tol=1e-15)))):
            raise ValueError("Pooled class values differ from the per-case observations")
    _verify(hashes)
    return records, hashes, manifest


def _chart(figure: dict, label: str) -> str:
    png, pdf = html.escape(figure["png"], quote=True), html.escape(figure["pdf"], quote=True)
    return (f'<section><h3>{html.escape(label)}</h3><a href="{png}"><img src="{png}" '
            f'loading="lazy" width="2100" height="1680" alt="{html.escape(label, quote=True)}"></a>'
            f'<nav><a href="{png}">PNG</a><a href="{pdf}">PDF</a></nav></section>')


def _gallery(cases: list[dict], cohort: list[dict], source_link: str) -> str:
    groups = []
    for key in BIN_IDS:
        figures = [figure for figure in cohort if figure["area_bin"] == key]
        if figures:
            groups.append(f'<article><h2>{key}</h2><div class="charts">' + "".join(
                _chart(figure, "Patient segment counts" if figure["metric"] == "segment_count"
                       else "Patient segment proportions") for figure in figures) + '</div></article>')
    patients = []
    for record in cases:
        case = record["case"]
        patients.append(f'<article id="{case}"><h2>{case}</h2>'
                        f'<p>{record["classified_segment_count"]:,} segments with valid mean area</p>'
                        '<div class="charts">' + "".join(_chart(figure, "Segment counts"
                            if figure["metric"] == "segment_count" else "Segment proportions")
                            for figure in record["figures"]) + '</div>'
                        f'<p><a href="cases/{case}/plot_report.json">Plot values and details</a></p></article>')
    return ('<!doctype html><html lang="en"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            '<title>PARSE area-class distributions</title><style>'
            'body{margin:0;background:white;color:#303030;font:14px/1.45 Arial,sans-serif}'
            'main{max-width:1400px;margin:auto;padding:20px}h1{font-size:24px}h2{font-size:18px;margin:0}'
            'h3{font-size:14px;font-weight:normal;margin:8px 0 0}a{color:#345f84}'
            'article{border:1px solid #ddd;padding:12px;min-width:0}article p{margin:4px 0}'
            '.charts,.cases{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px}'
            '.cohort{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:16px}'
            '.cohort .charts{grid-template-columns:1fr}img{display:block;width:100%;height:auto}'
            'nav{display:flex;gap:14px}main>h2{margin:24px 0 12px}'
            '@media(max-width:1000px){.cohort,.cases{grid-template-columns:1fr}.cohort .charts{grid-template-columns:repeat(2,minmax(0,1fr))}}'
            '@media(max-width:640px){.charts,.cohort .charts{grid-template-columns:1fr}}'
            '</style></head><body><main><h1>PARSE vessel-segment area classes</h1>'
            f'<p>{len(cases)} cases with valid segment mean areas. BV5: &lt;5 mm²; '
            'BV5-10: 5–10 mm², inclusive; BV10: &gt;10 mm².</p>'
            '<p>Each cohort histogram contains one observation per patient. Counts are group segment counts; '
            'proportions divide by that patient’s valid-area segment count. Case proportion plots share a 0–100% scale.</p>'
            f'<p><a href="{html.escape(source_link, quote=True)}">Source classification report</a> · '
            '<a href="report.md">Markdown report</a> · <a href="plot_values.csv">Patient values</a> · '
            '<a href="cohort_histogram_bins.csv">Histogram bins</a> · <a href="sources.json">Provenance</a></p>'
            '<h2>Across patients</h2><div class="cohort">' + "".join(groups) + '</div>'
            '<h2>Individual cases</h2><div class="cases">' + "".join(patients) + '</div></main></body></html>\n')


def _markdown(cases: list[dict], cohort: list[dict], source_link: str) -> str:
    lines = ['# PARSE area-class counts and proportions', '',
             f'Included cases: **{len(cases)}**. BV5 <5 mm²; BV5-10 includes 5–10 mm²; BV10 >10 mm².', '',
             'The cohort histograms use one equally weighted observation per patient for each class and metric. '
             'Each proportion divides a class count by that patient’s number of segments with valid mean areas.', '',
             f'[Image gallery](index.html) · [Source classification](<{source_link}>) · '
             '[Patient values](plot_values.csv) · [Histogram bins](cohort_histogram_bins.csv)', '',
             '## Across patients', '', '| Area class | Segment count | Segment proportion |', '|---|---|---|']
    for key in BIN_IDS:
        figures = [figure for figure in cohort if figure['area_bin'] == key]
        if figures:
            cells = [f"[![{key}]({figure['png']})]({figure['png']}) [PDF]({figure['pdf']})" for figure in figures]
            lines.append(f'| {key} | ' + ' | '.join(cells) + ' |')
    lines += ['', '## Individual cases', '', '| Case | Segment count | Segment proportion |', '|---|---|---|']
    for record in cases:
        cells = [f"[PNG]({figure['png']}) / [PDF]({figure['pdf']})" for figure in record['figures']]
        lines.append(f"| {record['case']} | " + ' | '.join(cells) + ' |')
    return '\n'.join(lines) + '\n'


def generate(publication_root: Path, output_root: Path) -> dict:
    publication_root, output_root = Path(publication_root).resolve(), Path(output_root).resolve()
    if publication_root.is_relative_to(output_root) or output_root.is_relative_to(publication_root):
        raise ValueError("Plot outputs must be separate from source classifications")
    records, source_hashes, source = _load_publication(publication_root)
    for name in source_hashes:
        parent = Path(name).parent
        if parent == output_root or parent in output_root.parents or output_root in parent.parents:
            raise ValueError("Plot outputs must be separate from source input directories")
    eligible = [record for record in records if record["classified_segment_count"] > 0]
    exclusions = [{"case": record["case"], "reason": "no_valid_mean_areas"}
                  for record in records if record["classified_segment_count"] == 0]
    publication_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ_") + uuid.uuid4().hex[:8]
    final = output_root / publication_id
    names = ("plot_parse_area_bins.py", "area_bin_charts.py", "parse_cross_section_bins.py",
             "report_parse_cross_section_bins.py", "parse_cross_section_statistics.py",
             "parse_radius_statistics.py", "plot_parse_distributions.py")
    code_hashes = {str(Path(__file__).with_name(name).resolve()): _sha256(Path(__file__).with_name(name))
                   for name in names}
    import matplotlib
    import numpy
    output_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".area_plots_", dir=output_root) as temporary:
        staging = Path(temporary)
        plotted, values = [], []
        for index, record in enumerate(eligible, 1):
            directory = staging / "cases" / record["case"]
            directory.mkdir(parents=True)
            figures = render_case(record, directory)
            for figure in figures:
                for extension in ("png", "pdf"):
                    figure[extension] = f"cases/{record['case']}/{figure[extension]}"
            item = {**record, "figures": figures}
            case_report = {**item, "publication_root": str(final), "sources_path": str(final / "sources.json")}
            (directory / "plot_report.json").write_text(
                json.dumps(case_report, indent=2, allow_nan=False) + '\n', encoding="utf-8")
            plotted.append(item)
            values.extend({"source": "prediction", "case": record["case"], **row} for row in record["bins"])
            if index % 10 == 0 or index == len(eligible):
                print(f"Rendered {index}/{len(eligible)} PARSE cases", flush=True)
        cohort_dir = staging / "cohort"
        cohort_dir.mkdir()
        figures, histogram_rows = render_cohort(eligible, cohort_dir) if eligible else ([], [])
        for figure in figures:
            for extension in ("png", "pdf"):
                figure[extension] = f"cohort/{figure[extension]}"
        _write_csv(staging / "plot_values.csv", values, ("source", "case", *BIN_FIELDS))
        _write_csv(staging / "cohort_histogram_bins.csv", histogram_rows, HISTOGRAM_FIELDS)
        source_link = _relative(publication_root / "report.md", final)
        (staging / "index.html").write_text(_gallery(plotted, figures, source_link), encoding="utf-8")
        (staging / "report.md").write_text(_markdown(plotted, figures, source_link), encoding="utf-8")
        result = {
            "schema_version": 1, "status": "complete", "dataset": "PARSE", "source": "prediction",
            "source_publication": str(publication_root), "run_id": source["run_id"],
            "publication_id": publication_id, "output_dir": str(final), "case_count": len(records),
            "included_case_count": len(plotted), "included_cases": [record["case"] for record in plotted],
            "excluded_cases": exclusions, "cases": plotted, "cohort_figures": figures,
            "bins": list(BIN_DEFINITIONS), "definitions": {**DEFINITIONS,
                "cohort_observation": "One equally weighted patient value per area class and metric",
                "count_zero": "Zero class counts are retained for patients with a positive valid-area denominator",
                "plot_scope": "Cases without valid mean areas are excluded from both plotted metrics"},
            "index_path": str(final / "index.html"), "report_path": str(final / "report.md"),
            "sources_path": str(final / "sources.json"), "source_sha256": source_hashes,
            "code_sha256": code_hashes, "areas_recomputed": False,
            "versions": {"python": platform.python_version(), "numpy": numpy.__version__,
                         "matplotlib": matplotlib.__version__},
            "input_integrity": "verified_after_publication",
            "artifact_sha256": {str(final / path.relative_to(staging)): _sha256(path)
                                for path in staging.rglob('*') if path.is_file()},
        }
        _verify(source_hashes)
        _verify(code_hashes)
        (staging / "sources.json").write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding="utf-8")
        return _publish_verified(staging, final, result, source_hashes, code_hashes)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--publication-root", type=Path, required=True,
                        help="Area-class statistics directory generated from this reproduction run.")
    parser.add_argument("--output-root", type=Path, default=Path("data/parse/reports/cross_section_area_plots"))
    args = parser.parse_args(argv)
    try:
        result = generate(args.publication_root, args.output_root)
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"Area-class plotting failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    print(f"Completed area-class plots for {result['included_case_count']} PARSE cases.")
    print(result["index_path"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
