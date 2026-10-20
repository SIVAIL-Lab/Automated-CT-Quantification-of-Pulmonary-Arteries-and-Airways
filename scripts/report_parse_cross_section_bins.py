#!/usr/bin/env python3
"""Classify saved PARSE segment mean areas into BV5, BV5-10 and BV10."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import re
import sys
import tempfile
import uuid

from parse_cross_section_bins import BIN_DEFINITIONS, BIN_IDS, classify_segments
from parse_cross_section_statistics import (
    REVIEW_FIELDS, _collect_inputs, _csv_value, _publish_verified, _relative,
    _sha256, _validated_case, _verify, _write_csv,
)
from segment_cross_sections import METHOD


BIN_FIELDS = ("area_bin", "label", "segment_count", "classified_segment_count", "fraction")
SUMMARY_FIELDS = ("classified_segment_count",)
DEFINITIONS = {
    "area": "Unrounded segment_area_mm2: physical-length-weighted mean of original consecutive finite local area pairs",
    "segment": "Existing EDT graph segment in the largest 26-connected predicted arterial component",
    "classification": "BV5: area < 5 mm2; BV5-10: 5 <= area <= 10 mm2; BV10: area > 10 mm2",
    "fraction": "Group segment count divided by segments with a valid mean area in the same case",
    "pooled_fraction": "Summed group segment count divided by summed valid-area segment count across all included cases",
    "export_scope": "Only segments with a valid mean area appear in classification tables and denominators",
    "empty_denominator": "A case with no valid mean areas has zero group counts and null fractions",
    "review": "Complete and partial means both contribute; inherited needs_review flags and numerical coverage remain unchanged",
}


def _merge_hashes(target: dict, incoming: dict) -> None:
    for name, digest in incoming.items():
        if (not Path(name).is_absolute() or not isinstance(digest, str)
                or not re.fullmatch(r"[0-9a-f]{64}", digest)):
            raise ValueError("Expected absolute source paths and SHA256 digests")
        if name in target and target[name] != digest:
            raise ValueError(f"Conflicting source hashes: {name}")
        target[name] = digest


def _load_run(source_run: Path) -> tuple[dict, list[dict], dict]:
    hashes = {}

    def read(path):
        raw = path.read_bytes()
        hashes[str(path)] = hashlib.sha256(raw).hexdigest()
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise ValueError(f"Expected a source JSON object: {path}")
        return value

    manifest, report = read(source_run / "manifest.json"), read(source_run / "report.json")
    cases = manifest.get("cases")
    if (not isinstance(cases, list) or not cases
            or any(not isinstance(case, str) or not re.fullmatch(r"PA[0-9]{6}", case) for case in cases)
            or len(set(cases)) != len(cases) or manifest.get("dataset") != "PARSE"
            or manifest.get("source") != "prediction"
            or manifest.get("component_policy") != "largest_26_connected"):
        raise ValueError("Expected a fixed PARSE prediction cohort using the largest 26-connected component")
    publication = report.get("publication", {})
    if (report.get("status") != "complete" or report.get("dataset") != "PARSE"
            or report.get("source") != "prediction" or report.get("run_id") != manifest.get("run_id")
            or report.get("case_count") != len(cases) or report.get("completed_count") != len(cases)
            or report.get("failed_count") != 0 or report.get("pending_count") != 0
            or publication.get("status") != "complete" or publication.get("segment_area_method") != METHOD):
        raise ValueError("Source must be a complete, successful cross-section area cohort")
    records = report.get("records", [])
    indexed = {row["case"]: row for row in records}
    if len(indexed) != len(records) or set(indexed) != set(cases):
        raise ValueError("Source report membership differs from its manifest")
    for case in cases:
        record = indexed[case]
        if (record.get("source") != "prediction" or record.get("status") != "ok"
                or record.get("dataset") != "PARSE" or record.get("technical_validation") != "passed"
                or record.get("input_integrity") != "verified_after_computation"
                or read(source_run / "checkpoints" / f"{case}.json") != record):
            raise ValueError(f"Source checkpoint differs from its verified prediction record: {case}")
    for owner in (report, publication):
        for field in ("source_sha256", "code_sha256", "orchestrator_sha256", "artifact_sha256"):
            _merge_hashes(hashes, owner.get(field, {}))
    _verify(hashes)
    return report, [indexed[case] for case in cases], hashes


def _table(bins: list[dict]) -> list[str]:
    lines = ["| Class | Segment count | Fraction of valid-area segments (%) |",
             "|---|---:|---:|"]
    for row in bins:
        fraction = "—" if row["fraction"] is None else f"{100 * row['fraction']:.3f}"
        lines.append(f"| {row['label']} | {row['segment_count']:,} | {fraction} |")
    return lines


def _case_report(record: dict, grouped: dict, final: Path) -> str:
    summary = grouped["summary"]
    lines = [f"# {record['case']}: vessel-segment area classes", "",
             "Classes use the original segment mean cross-sectional area (mm²): "
             "BV5 < 5; BV5-10 includes both 5 and 10; BV10 > 10. Classification uses full precision.", "",
             f"Segments with valid mean area: **{summary['classified_segment_count']:,}**.", "",
             *_table(grouped["bins"]), "",
             "Fractions use only segments with a valid mean area as the denominator. "
             "The three class fractions sum to 100% when this denominator is positive. "
             "Complete and partial segment means retain their original coverage and review flags.", "",
             "[Classified segment table](segment_area_bins.csv) · [Class counts](area_bins.csv)", "",
             f"[Original mean-area table](<{_relative(record['segment_areas_path'], final)}>) · "
             f"[Original local areas](<{_relative(record['samples_path'], final)}>)", "",
             "Inherited review and segmentation provenance:", "", "```json",
             json.dumps({field: record.get(field, "not_recorded") for field in REVIEW_FIELDS}, indent=2),
             "```", ""]
    return "\n".join(lines)


def _report(summary: dict, pooled: list[dict], cases: list[dict]) -> str:
    lines = ["# PARSE vessel-segment cross-sectional area classes", "",
             f"Included prediction cases: **{summary['case_count']}**. "
             "Each segment with a valid saved mean area receives one class.", "",
             "| Class | Mean cross-sectional area (mm²) |",
             "|---|---|", "| BV5 | < 5 |", "| BV5-10 | 5–10, inclusive |", "| BV10 | > 10 |", "",
             "Classification uses unrounded, physical-length-weighted segment mean areas. "
             "Partial measurements and needs_review flags are retained.", "",
             f"Segments with valid mean area: **{summary['classified_segment_count']:,}**.", "",
             "## Pooled segment counts", "", *_table(pooled), "",
             "Pooled counts sum segments across patients. Fractions divide these counts by segments with "
             "valid mean areas. Per-case fractions use each patient's own valid-area denominator. "
             "The three class fractions sum to 100% when this denominator is positive.", "",
             "[All classified segments](all_segment_area_bins.csv) · [Per-case class counts](per_case_area_bins.csv) · "
             "[Pooled class counts](cohort_area_bins.csv) · [Case denominators](case_summary.csv) · "
             "[Sources and definitions](sources.json)", "",
             "## Cases", "", "| Case | Segments with valid mean area |", "|---|---:|"]
    lines.extend(f"| [{row['case']}](cases/{row['case']}/report.md) | {row['classified_segment_count']:,} |"
                 for row in cases)
    return "\n".join(lines) + "\n"


def generate(source_run: Path, output_root: Path) -> dict:
    """Publish independent area-class tables without modifying measured areas."""
    source_run, output_root = Path(source_run).resolve(), Path(output_root).resolve()
    if output_root.is_relative_to(source_run) or source_run.is_relative_to(output_root):
        raise ValueError("Classification outputs must be separate from source computations")
    source_report, records, hashes = _load_run(source_run)
    publication_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ_") + uuid.uuid4().hex[:8]
    final = output_root / publication_id
    _merge_hashes(hashes, _collect_inputs(records, final))
    for name in hashes:
        parent = Path(name).parent
        if parent == output_root or parent in output_root.parents or output_root in parent.parents:
            raise ValueError("Classification outputs must be separate from original input directories")
    names = ("report_parse_cross_section_bins.py", "parse_cross_section_bins.py",
             "parse_cross_section_statistics.py", "segment_cross_sections.py",
             "section_measurement.py", "measurement_io.py", "run_edt_experiment.py")
    code_hashes = {str(Path(__file__).with_name(name).resolve()): _sha256(Path(__file__).with_name(name))
                   for name in names}
    case_rows, case_bins, counts, fields = [], [], dict.fromkeys(BIN_IDS, 0), None
    output_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".area_bins_", dir=output_root) as temporary:
        staging = Path(temporary)
        with (staging / "all_segment_area_bins.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = None
            for index, record in enumerate(records, 1):
                segments, original_summary = _validated_case(record)
                grouped = classify_segments(segments)
                summary = grouped["summary"]
                if (summary["total_segment_count"] != original_summary["segment_count"]
                        or summary["classified_segment_count"] != original_summary["segment_area_count"]
                        or summary["unavailable_segment_count"] != original_summary["unavailable_count"]):
                    raise ValueError("Classification does not conserve the original segment and area counts")
                # Derive the schema before filtering so even a case with no
                # valid means gets a complete, header-only segment table.
                current_fields = list(dict.fromkeys(["source", "case", *[
                    key for row in grouped["segments"] for key in row]]))
                rows = [{"source": "prediction", "case": record["case"], **row}
                        for row in grouped["segments"] if row["area_bin"] is not None]
                if len(rows) != summary["classified_segment_count"]:
                    raise ValueError("Exported segment count differs from the valid-area denominator")
                if writer is None:
                    fields = current_fields
                    writer = csv.DictWriter(stream, fieldnames=fields)
                    writer.writeheader()
                elif current_fields != fields:
                    raise ValueError("Segment table fields differ across source cases")
                for row in rows:
                    writer.writerow({key: _csv_value(row[key]) for key in fields})
                case_dir = staging / "cases" / record["case"]
                case_dir.mkdir(parents=True)
                _write_csv(case_dir / "segment_area_bins.csv", rows, fields)
                bins = [{"source": "prediction", "case": record["case"], **row} for row in grouped["bins"]]
                _write_csv(case_dir / "area_bins.csv", bins, ("source", "case", *BIN_FIELDS))
                (case_dir / "report.md").write_text(
                    _case_report(record, grouped, final / "cases" / record["case"]), encoding="utf-8")
                case_rows.append({"source": "prediction", "case": record["case"], **summary})
                case_bins.extend(bins)
                for row in grouped["bins"]:
                    counts[row["area_bin"]] += row["segment_count"]
                if index % 10 == 0 or index == len(records):
                    print(f"Validated and classified {index}/{len(records)} PARSE cases", flush=True)
        total = sum(row["total_segment_count"] for row in case_rows)
        classified = sum(row["classified_segment_count"] for row in case_rows)
        unavailable = sum(row["unavailable_segment_count"] for row in case_rows)
        if sum(counts.values()) != classified or classified + unavailable != total:
            raise ValueError("Cohort area classes do not conserve the segment denominator")
        summary = {"case_count": len(records), "total_segment_count": total,
                   "classified_segment_count": classified, "unavailable_segment_count": unavailable,
                   "classified_fraction": classified / total, "unavailable_fraction": unavailable / total}
        pooled = [{"source": "prediction", "case_count": len(records), "area_bin": definition["area_bin"],
                   "label": definition["label"], "segment_count": counts[definition["area_bin"]],
                   "classified_segment_count": classified,
                   "fraction": counts[definition["area_bin"]] / classified if classified else None}
                  for definition in BIN_DEFINITIONS]
        _write_csv(staging / "per_case_area_bins.csv", case_bins, ("source", "case", *BIN_FIELDS))
        _write_csv(staging / "cohort_area_bins.csv", pooled, ("source", "case_count", *BIN_FIELDS))
        _write_csv(staging / "case_summary.csv", case_rows, ("source", "case", *SUMMARY_FIELDS))
        (staging / "report.md").write_text(_report(summary, pooled, case_rows), encoding="utf-8")
        publication = {
            "schema_version": 2, "status": "complete", "dataset": "PARSE", "source": "prediction",
            "run_id": source_report["run_id"], "publication_id": publication_id,
            "source_run": str(source_run), "output_dir": str(final), "case_count": len(records),
            "included_cases": [record["case"] for record in records], "summary": summary,
            "bins": list(BIN_DEFINITIONS), "cohort_bins": pooled, "case_summaries": case_rows,
            "definitions": DEFINITIONS, "segment_area_method": METHOD,
            "geometry_changed": False, "areas_recomputed": False, "review_status_promotion": False,
            "report_path": str(final / "report.md"), "sources_path": str(final / "sources.json"),
            "source_sha256": hashes, "code_sha256": code_hashes,
            "versions": {"python": platform.python_version()},
            "input_integrity": "verified_after_publication",
            "artifact_sha256": {str(final / path.relative_to(staging)): _sha256(path)
                                for path in staging.rglob("*") if path.is_file()},
        }
        _verify(hashes)
        _verify(code_hashes)
        (staging / "sources.json").write_text(json.dumps(publication, indent=2, allow_nan=False) + "\n",
                                               encoding="utf-8")
        return _publish_verified(staging, final, publication, hashes, code_hashes)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run-dir", type=Path, required=True,
                        help="Completed cross-section computation directory from this reproduction run.")
    parser.add_argument("--output-root", type=Path, default=Path("data/parse/reports/cross_section_area_bins"))
    args = parser.parse_args(argv)
    try:
        result = generate(args.source_run_dir, args.output_root)
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"Cross-section classification failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    print(f"Completed area classification for {result['case_count']} PARSE cases.")
    print(result["report_path"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
