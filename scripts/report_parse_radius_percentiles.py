#!/usr/bin/env python3
"""Report patient-level PARSE radius-bin count and proportion percentiles."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import sys
import tempfile
import uuid

import numpy as np

from parse_radius_statistics import BIN_DEFINITIONS
from plot_parse_radius_case import TICK_LABELS, _load_cases, _sha256, _verify


PERCENTILES = (10, 25, 50, 75)
PERCENTILE_FIELDS = tuple(f"p{value}" for value in PERCENTILES)
FIELDS = ("radius_bin", "label", "metric", "unit", "n", *PERCENTILE_FIELDS)


def _summarize(records: list[dict]) -> list[dict]:
    """Compute equal-patient quantiles from validated, complete case records."""
    rows = []
    for definition in BIN_DEFINITIONS:
        bins = [next(row for row in record["bins"] if row["radius_bin"] == definition["radius_bin"])
                for record in records]
        for metric in ("segment_count", "fraction"):
            quantiles = np.quantile([row[metric] for row in bins],
                                    np.asarray(PERCENTILES) / 100., method="linear")
            rows.append({
                "radius_bin": definition["radius_bin"], "label": definition["label"],
                "metric": metric, "unit": "segments" if metric == "segment_count" else "fraction",
                "n": len(bins),
                **{field: float(value) for field, value in zip(PERCENTILE_FIELDS, quantiles)},
            })
    return rows


def _markdown(rows: list[dict], case_count: int) -> str:
    intervals = {definition["radius_bin"]: label
                 for definition, label in zip(BIN_DEFINITIONS, TICK_LABELS)}
    lines = [
        "# PARSE arterial radius percentiles", "",
        f"Included patients: **{case_count}**. Each patient contributes one observation per metric and radius interval.",
        "",
        "Measurements use EDT graph segments from each patient's largest 26-connected predicted arterial mask component. "
        "Segment radius is the physical-length-weighted mean EDT radius. Each proportion divides a patient's "
        "radius-group segment count by that same patient's total segment count.",
        "",
        "Percentiles use linear interpolation. P50 is the median; count percentiles can be fractional. "
        "Values below are rounded to two decimal places. Proportions are displayed as percentages; "
        "the full-precision CSV retains fractions between zero and one.",
        "",
        "| Radius (mm) | Metric | N | P10 | P25 | P50 | P75 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        proportion = row["metric"] == "fraction"
        label = "Segment proportion (%)" if proportion else "Segment count"
        scale = 100. if proportion else 1.
        values = " | ".join(f"{row[field] * scale:.2f}" for field in PERCENTILE_FIELDS)
        lines.append(f"| {intervals[row['radius_bin']]} | {label} | {row['n']} | {values} |")
    lines += ["", "[Full-precision table](percentiles.csv) | [Sources and calculation details](sources.json)", ""]
    return "\n".join(lines)


def generate(publication_root: Path, output_root: Path) -> dict:
    """Publish an independent percentile report from an existing complete cohort."""
    publication_root, output_root = Path(publication_root).resolve(), Path(output_root).resolve()
    if (output_root.is_relative_to(publication_root)
            or publication_root.is_relative_to(output_root)):
        raise ValueError("Percentile outputs must be separate from the source publication")
    records, source_hashes, source_info = _load_cases(publication_root, None)
    implementation = [Path(__file__).resolve(), *[Path(__file__).with_name(name).resolve() for name in (
        "plot_parse_radius_case.py", "parse_radius_statistics.py", "plot_parse_distributions.py")]]
    code_hashes = {str(path): _sha256(path) for path in implementation}
    rows = _summarize(records)
    publication_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ_") + uuid.uuid4().hex[:8]
    final = output_root / publication_id
    artifacts = {"report": final / "report.md", "csv": final / "percentiles.csv", "sources": final / "sources.json"}
    output_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".percentiles_", dir=output_root) as temporary:
        staging = Path(temporary)
        with (staging / "percentiles.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        (staging / "report.md").write_text(_markdown(rows, len(records)), encoding="utf-8")
        manifest = {
            "schema_version": 1, "status": "complete", "dataset": "PARSE", "source": "prediction",
            "publication_id": publication_id, "run_id": records[0]["run_id"],
            "source_publication": str(publication_root), "output_root": str(final), **source_info,
            "case_count": len(records), "included_cases": [record["case"] for record in records],
            "intervals": list(BIN_DEFINITIONS), "rows": rows,
            "parameters": {"percentiles": list(PERCENTILES), "method": "linear", "display_decimals": 2},
            "definitions": {
                "segment": "EDT graph segment in the largest 26-connected predicted arterial mask component",
                "radius": "Physical-length-weighted mean EDT radius in millimetres",
                "fraction": "Radius-group segment count / the same patient's total segment count",
                "sample": "One equally weighted observation per included patient for each radius bin and metric",
                "csv_fraction_unit": "Fraction between zero and one",
                "report_fraction_unit": "Percent",
            },
            "source_sha256": source_hashes, "code_sha256": code_hashes,
            "versions": {"python": platform.python_version(), "numpy": np.__version__},
            "artifacts": {key: str(path) for key, path in artifacts.items()},
            "artifact_sha256": {str(path): _sha256(staging / path.name)
                                for key, path in artifacts.items() if key != "sources"},
        }
        (staging / "sources.json").write_text(
            json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        _verify({**source_hashes, **code_hashes})
        staging.rename(final)
    return manifest


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--publication-root", type=Path, required=True,
                        help="Complete radius publication containing sources.json and per_case_radius_bins.csv")
    parser.add_argument("--output-root", type=Path, required=True,
                        help="Separate parent directory for new versioned percentile reports")
    args = parser.parse_args(argv)
    try:
        result = generate(args.publication_root, args.output_root)
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"Radius percentile report failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    print(f"Completed PARSE percentile report for {result['case_count']} patients.")
    for path in result["artifacts"].values():
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
