#!/usr/bin/env python3
"""Plot individual PARSE radius-bin counts and proportions from a published cohort CSV."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import html
import io
import json
from pathlib import Path
import platform
import re
import sys
import tempfile
import uuid

from parse_radius_statistics import BIN_DEFINITIONS, BIN_FIELDS, _validated_bins
from plot_parse_distributions import COLORS, FIGURE_SIZE, PNG_DPI, STYLE, _tight_layout


STEM = "radius_bin_proportions"
COUNT_STEM = "radius_bin_counts"
TICK_LABELS = ("(0, 2]", "(2, 4]", "(4, 6]", ">6")


def _sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _verify(hashes: dict[str, str]) -> None:
    for path, expected in hashes.items():
        if _sha256(Path(path)) != expected:
            raise ValueError(f"Source changed or has an invalid hash: {path}")


def _validate_case_ids(cases: list[str]) -> None:
    if (not isinstance(cases, list) or not cases
            or any(not isinstance(case, str) or not re.fullmatch(r"PA[0-9]{6}", case) for case in cases)
            or len(set(cases)) != len(cases)):
        raise ValueError("Case lists must contain nonempty, unique PARSE case IDs")


def _load_cases(publication_root: Path, cases: list[str] | None) -> tuple[list[dict], dict[str, str], dict]:
    manifest_path = publication_root / "sources.json"
    csv_path = publication_root / "per_case_radius_bins.csv"
    manifest_bytes, csv_bytes = manifest_path.read_bytes(), csv_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    if (manifest.get("dataset") != "PARSE" or manifest.get("source") != "prediction"
            or manifest.get("status") not in {"complete", "partial"}
            or manifest.get("input_integrity") != "verified_after_publication"
            or manifest.get("bins") != list(BIN_DEFINITIONS)):
        raise ValueError("Expected a verified PARSE prediction radius publication with the current bins")
    included = manifest.get("included_cases")
    _validate_case_ids(included)
    if cases is None:
        if manifest["status"] != "complete":
            raise ValueError("--all-cases requires a complete source publication")
        cases = sorted(included)
    _validate_case_ids(cases)
    if not set(cases) <= set(included):
        raise ValueError("Requested case is not included in the source publication")
    hashes = {str(manifest_path): hashlib.sha256(manifest_bytes).hexdigest(),
              str(csv_path): hashlib.sha256(csv_bytes).hexdigest()}
    if manifest.get("artifact_sha256", {}).get(str(csv_path)) != hashes[str(csv_path)]:
        raise ValueError("Published radius CSV hash does not match sources.json")
    reader = csv.DictReader(io.StringIO(csv_bytes.decode("utf-8-sig")))
    if not {"case", *BIN_FIELDS} <= set(reader.fieldnames or []):
        raise ValueError("Published radius CSV is missing required columns")
    selected = {case: [] for case in cases}
    for row in reader:
        if row["case"] in selected:
            selected[row["case"]].append({
                "radius_bin": row["radius_bin"], "label": row["label"],
                "segment_count": int(row["segment_count"]),
                "total_segment_count": int(row["total_segment_count"]),
                "fraction": float(row["fraction"]),
            })
    records = []
    for case, rows in selected.items():
        if not rows:
            raise ValueError(f"No published radius rows for {case}")
        bins = _validated_bins({"case": case, "bins": rows,
                                "summary": {"segment_count": rows[0]["total_segment_count"]}})
        records.append({"case": case, "run_id": manifest["run_id"], "bins": bins,
                        "total_segment_count": bins[0]["total_segment_count"]})
    _verify(hashes)
    return records, hashes, {"source_status": manifest["status"], "source_case_count": len(included)}


def _common_render_parameters() -> dict:
    return {
        "style": "academic-v2", "figure_size_inches": list(FIGURE_SIZE), "png_dpi": PNG_DPI,
        "render_size": [round(value * PNG_DPI) for value in FIGURE_SIZE],
        "font": STYLE["font.family"], "color": COLORS["arterial"], "background": "white",
        "title": False, "legend": False, "grid": False, "mean_median_markers": False,
        "x_axis": "Radius (mm)", "x_tick_labels": list(TICK_LABELS), "bar_width": .85,
    }


def _render_plot(bins: list[dict], directory: Path, *, counts: bool) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    values = [row["segment_count"] if counts else row["fraction"] * 100 for row in bins]
    stem = COUNT_STEM if counts else STEM
    with plt.rc_context(STYLE):
        figure, axis = plt.subplots(figsize=FIGURE_SIZE, dpi=PNG_DPI)
        try:
            # Equal categorical widths also accommodate the open-ended >6 mm bin.
            axis.bar(range(4), values, width=.85,
                     color=COLORS["arterial"], edgecolor="white", linewidth=.45)
            axis.set_xticks(range(4), TICK_LABELS)
            axis.set_xlim(-.55, 3.55)
            if counts:
                axis.set_ylim(0, max(1, max(values)) * 1.04)
                axis.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=True, min_n_ticks=2))
            else:
                axis.set_ylim(0, 100)
                axis.set_yticks(range(0, 101, 20))
            axis.set_xlabel("Radius (mm)", labelpad=3)
            axis.set_ylabel("Number of segments" if counts else "Segment proportion (%)", labelpad=3)
            axis.grid(False)
            _tight_layout(figure)
            for extension in ("png", "pdf"):
                figure.savefig(directory / f"{stem}.{extension}", format=extension, dpi=PNG_DPI,
                               facecolor="white", metadata={"Creator": "PARSE EDT case radius plot"}
                               if extension == "pdf" else None)
            lower, upper = axis.get_ylim()
            return {
                **_common_render_parameters(), "y_axis": axis.get_ylabel(),
                "y_limits": [float(lower), float(upper)],
                # Locators can propose ticks outside the displayed limits.
                "y_ticks": [int(tick) for tick in axis.get_yticks() if lower <= tick <= upper],
            }
        finally:
            plt.close(figure)


def _render(bins: list[dict], directory: Path) -> dict:
    _render_plot(bins, directory, counts=False)
    return _render_plot(bins, directory, counts=True)


def _index_html(reports: list[dict]) -> str:
    cards = []
    for report in sorted(reports, key=lambda row: row["case"]):
        case = html.escape(report["case"])
        charts = []
        for stem, label in ((COUNT_STEM, "Segment counts"), (STEM, "Segment proportions")):
            png, pdf = (f"{case}/{stem}.{extension}" for extension in ("png", "pdf"))
            charts.append(
                f'<section><h3>{label}</h3><a href="{png}"><img src="{png}" loading="lazy" '
                f'width="2100" height="1680" alt="{case} arterial radius group {label.lower()}"></a>'
                f'<nav><a href="{png}">PNG</a><a href="{pdf}">PDF</a></nav></section>')
        cards.append(
            f'<article><h2>{case}</h2><p>{report["total_segment_count"]:,} segments</p>'
            '<div class="charts">' + "".join(charts) + '</div>'
            f'<p><a href="{case}/plot_report.json">Data and provenance</a></p></article>')
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        '<title>PARSE arterial radius counts and proportions</title><style>'
        'body{margin:0;background:#fff;color:#303030;font-family:Arial,sans-serif;line-height:1.45}'
        'main{max-width:1400px;margin:auto;padding:20px}h1{font-size:24px;margin:0 0 8px}'
        'header{margin-bottom:20px}header p{margin:6px 0}a{color:#345f84}'
        '.cases{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px}'
        '.charts{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px}'
        'article{border:1px solid #ddd;padding:12px;min-width:0}'
        'h2{font-size:17px;margin:0}article p{font-size:13px;margin:4px 0;color:#555}'
        'h3{font-size:14px;font-weight:normal;margin:8px 0 0}'
        'img{display:block;width:100%;height:auto}nav{display:flex;gap:14px;font-size:13px}'
        '@media(max-width:1000px){.cases{grid-template-columns:1fr}}'
        '@media(max-width:640px){.charts{grid-template-columns:1fr}}'
        '</style></head><body><main><header><h1>PARSE arterial radius counts and proportions</h1>'
        f'<p>Completed requested cases: {len(reports)} / {len(reports)}.</p>'
        '<p>Counts use a separate y-axis scale for each case. '
        'Proportions use each case\'s total EDT graph segment count as the denominator '
        'and share the same 0–100% scale. '
        'Radii are physical-length-weighted EDT means from the largest arterial mask component. '
        'Within each case, counts appear first, followed by proportions.</p>'
        '<p><a href="plot_summary.json">Publication summary</a></p></header>'
        '<div class="cases">' + "".join(cards) + '</div></main></body></html>\n')


def generate(publication_root: Path, output_root: Path, cases: list[str] | None) -> dict:
    """Publish new case plots without modifying or recomputing their source data."""
    publication_root, output_root = Path(publication_root).resolve(), Path(output_root).resolve()
    if (output_root.is_relative_to(publication_root)
            or publication_root.is_relative_to(output_root)):
        raise ValueError("Case plot outputs must be separate from the source publication")
    records, source_hashes, source_info = _load_cases(publication_root, cases)
    implementation = [Path(__file__).resolve(), Path(__file__).with_name("parse_radius_statistics.py").resolve(),
                      Path(__file__).with_name("plot_parse_distributions.py").resolve()]
    code_hashes = {str(path): _sha256(path) for path in implementation}
    publication_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ_") + uuid.uuid4().hex[:8]
    final = output_root / publication_id
    output_root.mkdir(parents=True, exist_ok=True)
    published = []
    import matplotlib

    # Only publish the complete directory after rendering and rechecking sources.
    with tempfile.TemporaryDirectory(prefix=".case_plots_", dir=output_root) as temporary:
        staging = Path(temporary)
        for record in records:
            directory = staging / record["case"]
            directory.mkdir()
            count_render_parameters = _render(record["bins"], directory)
            destinations = {extension: final / record["case"] / f"{STEM}.{extension}"
                            for extension in ("png", "pdf")}
            destinations.update({f"count_{extension}": final / record["case"] / f"{COUNT_STEM}.{extension}"
                                 for extension in ("png", "pdf")})
            report = {
                "schema_version": 1, "status": "ok", "dataset": "PARSE", "source": "prediction",
                "publication_id": publication_id, "source_publication": str(publication_root), **record,
                "definitions": {
                    "segment": "EDT graph segment; distinct from Fiji branch count",
                    "radius": "Physical-length-weighted mean EDT radius in millimetres",
                    "denominator": "Total EDT graph segment count in this case's largest 26-connected artery mask component",
                    "proportion": "Radius-group segment count / this case's total segment count",
                },
                "intervals": list(BIN_DEFINITIONS),
                "plot_values_percent": [row["fraction"] * 100 for row in record["bins"]],
                "plot_values_count": [row["segment_count"] for row in record["bins"]],
                "render_parameters": {
                    **_common_render_parameters(),
                    "y_axis": "Segment proportion (%)", "y_limits": [0, 100], "y_ticks": list(range(0, 101, 20)),
                },
                "count_render_parameters": count_render_parameters,
                "source_sha256": source_hashes, "code_sha256": code_hashes,
                "versions": {"python": platform.python_version(), "matplotlib": matplotlib.__version__},
                "artifacts": {extension: str(path) for extension, path in destinations.items()},
                "artifact_sha256": {str(path): _sha256(directory / path.name)
                                    for path in destinations.values()},
            }
            (directory / "plot_report.json").write_text(
                json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
            published.append(report)
        (staging / "index.html").write_text(_index_html(published), encoding="utf-8")
        result = {
            "schema_version": 1, "status": "complete", "dataset": "PARSE", "source": "prediction",
            "publication_id": publication_id, "output_root": str(final),
            "run_id": records[0]["run_id"], "source_publication": str(publication_root),
            "selection_mode": "all_cases" if cases is None else "explicit_cases", **source_info,
            "case_count": len(published), "completed_count": len(published),
            "selected_cases": [record["case"] for record in published],
            "index_path": str(final / "index.html"), "summary_path": str(final / "plot_summary.json"),
            "source_sha256": source_hashes, "code_sha256": code_hashes,
            "cases": published,
            "artifact_sha256": {str(final / path.relative_to(staging)): _sha256(path)
                                for path in sorted(staging.rglob("*")) if path.is_file()},
        }
        (staging / "plot_summary.json").write_text(
            json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        _verify({**source_hashes, **code_hashes})
        staging.rename(final)
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--case", dest="cases", action="append")
    selection.add_argument("--all-cases", action="store_true",
                           help="Plot every included case from a complete source publication")
    parser.add_argument("--publication-root", type=Path, required=True,
                        help="Existing radius publication containing sources.json and per_case_radius_bins.csv")
    parser.add_argument("--output-root", type=Path, required=True,
                        help="Separate parent directory for new versioned case plots")
    args = parser.parse_args(argv)
    try:
        result = generate(args.publication_root, args.output_root, None if args.all_cases else args.cases)
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"Radius case plot failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    print(f"Completed {result['completed_count']} / {result['case_count']} requested PARSE case plots.")
    print(result["index_path"])
    print(result["summary_path"])
    if not args.all_cases:
        for report in result["cases"]:
            print(f"{report['case']}: {report['total_segment_count']} segments")
            for path in report["artifacts"].values():
                print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
