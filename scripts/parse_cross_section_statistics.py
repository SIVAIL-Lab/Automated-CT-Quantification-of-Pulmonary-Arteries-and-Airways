"""Publish PARSE cohort segment-area tables from verified saved case results."""
from __future__ import annotations

from collections import Counter
import csv
import hashlib
import html
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile

from segment_cross_sections import DEFINITIONS, METHOD, _validate_csv, summarize_measurement


SUMMARY_FIELDS = ("segment_count", "sample_count", "segment_area_count", "complete_count", "partial_count",
                  "unavailable_count", "includes_needs_review_count", "total_length_mm", "measured_length_mm",
                  "area_coverage_fraction")
CASE_FIELDS = ("source", "case", "status", *SUMMARY_FIELDS, "ok_sample_count", "needs_review_sample_count",
               "unavailable_sample_count", "strict_ok_span_mm", "reason_counts", "visual_review",
               "anatomical_validation", "reason")
REQUIRED_PATHS = ("samples_path", "measurement_path", "segments_path", "segment_areas_path", "segment_areas_json_path")
OPTIONAL_LINKS = (("radius_segments_path", "Original EDT segment radii"),
                  ("source_report_path", "Original EDT computation record"),
                  ("source_checkpoint_path", "Source checkpoint"),
                  ("worker_report_path", "Cross-section computation record"))
SOURCES = {"prediction": "Prediction", "reference": "Manual-label-derived reference"}
REVIEW_FIELDS = ("visual_review", "review_observations", "anatomical_validation",
                 "segmentation_provenance_status", "model_input_verified")


def _sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _verify(hashes: dict[str, str]) -> None:
    for raw_path, expected in hashes.items():
        path = Path(raw_path)
        if not path.is_file() or _sha256(path) != expected:
            raise ValueError(f"Cross-section cohort input hash mismatch: {path}")


def _publish_verified(staging: Path, report_dir: Path, publication: dict,
                      source_hashes: dict, code_hashes: dict) -> dict:
    """Validate the published bytes and roll back only this invocation's directory."""
    own_stat = staging.stat()
    artifacts = {**publication["artifact_sha256"],
                 str(report_dir / "sources.json"): _sha256(staging / "sources.json")}
    try:
        if report_dir.exists():
            raise ValueError("Publication directory appeared during rendering")
        staging.rename(report_dir)
        _verify(source_hashes)
        _verify(code_hashes)
        _verify(artifacts)
    except BaseException:
        # An interruption can arrive immediately after rename, before a Python
        # ownership flag could be assigned. The inode identifies our directory
        # and prevents deleting an unrelated destination in a publication race.
        if report_dir.is_dir():
            current = report_dir.stat()
            if (current.st_dev, current.st_ino) == (own_stat.st_dev, own_stat.st_ino):
                shutil.rmtree(report_dir)
        raise
    return {**publication, "artifact_sha256": artifacts}


def _collect_inputs(records: list[dict], report_dir: Path) -> dict[str, str]:
    hashes = {}

    def collect(mapping):
        if not isinstance(mapping, dict):
            raise ValueError("Input hashes must map absolute paths to SHA256 digests")
        for raw_path, expected in mapping.items():
            path = Path(raw_path)
            if (not path.is_absolute() or not isinstance(expected, str)
                    or not re.fullmatch(r"[0-9a-f]{64}", expected)):
                raise ValueError("Input hashes require absolute paths and complete SHA256 values")
            path = path.resolve()
            if report_dir == path or report_dir in path.parents or path.parent in report_dir.parents:
                raise ValueError("Cohort reports must be separate from their source inputs")
            key = str(path)
            if key in hashes and hashes[key] != expected:
                raise ValueError(f"Conflicting source digests: {path}")
            hashes[key] = expected

    for record in records:
        for field in ("source_sha256", "code_sha256", "artifact_sha256", "source_checkpoint_sha256"):
            collect(record.get(field, {}))
        worker_path, worker_digest = record.get("worker_report_path"), record.get("worker_report_sha256")
        if bool(worker_path) != bool(worker_digest) or (record.get("status") == "ok" and not worker_path):
            raise ValueError("Completed cases require a worker report and its SHA256")
        if worker_path:
            collect({worker_path: worker_digest})
            _verify({str(Path(worker_path).resolve()): worker_digest})
            worker = json.loads(Path(worker_path).read_text(encoding="utf-8"))
            if not isinstance(worker, dict):
                raise ValueError("Worker report must be a JSON object")
            for key, value in worker.items():
                if key == "artifact_sha256":
                    if any(record.get(key, {}).get(path) != digest for path, digest in value.items()):
                        raise ValueError("Case artifact hashes differ from the worker report")
                elif key != "cache_status" and record.get(key) != value:
                    raise ValueError(f"Case {key} differs from its saved worker report")
        if record.get("status") == "ok":
            for field in (*REQUIRED_PATHS, *[key for key, _ in OPTIONAL_LINKS if record.get(key)]):
                if str(Path(record[field]).resolve()) not in hashes:
                    raise ValueError(f"Case report input has no recorded SHA256: {field}")
    _verify(hashes)
    return hashes


def _validated_case(record: dict) -> tuple[list[dict], dict]:
    measurement = json.loads(Path(record["measurement_path"]).read_text(encoding="utf-8"))
    areas = json.loads(Path(record["segment_areas_json_path"]).read_text(encoding="utf-8"))
    if measurement["summary"] != record["sample_summary"] or areas["summary"] != record["summary"]:
        raise ValueError("Saved case summaries differ from their numerical tables")
    calculated = summarize_measurement(measurement, record["source"])
    if calculated["summary"] != areas["summary"] or calculated["segments"] != areas["segments"]:
        raise ValueError("Segment-area JSON differs from the original adjacent sample integration")
    for rows in (measurement["samples"], measurement["segments"], areas["segments"]):
        for row in rows:
            if row.get("case", record["case"]) != record["case"] or row.get("source", record["source"]) != record["source"]:
                raise ValueError("Numerical table case/source identity differs from its record")
    _validate_csv(Path(record["samples_path"]), measurement["samples"])
    _validate_csv(Path(record["segments_path"]), measurement["segments"])
    _validate_csv(Path(record["segment_areas_path"]), areas["segments"])
    summary, sample_summary = record["summary"], record["sample_summary"]
    row = {"case": record["case"], "source": record["source"], "status": "ok",
           **{field: summary[field] for field in SUMMARY_FIELDS},
           **{f"{status}_sample_count": sample_summary["status_counts"].get(status, 0)
              for status in ("ok", "needs_review", "unavailable")},
           "strict_ok_span_mm": sample_summary["ok_span_mm"],
           "reason_counts": sample_summary["reason_counts"],
           "visual_review": record.get("visual_review", "not_recorded"),
           "anatomical_validation": record.get("anatomical_validation", "not_recorded"), "reason": ""}
    return areas["segments"], row


def _csv_value(value):
    return "" if value is None else json.dumps(value, allow_nan=False, sort_keys=True) if isinstance(value, (dict, list)) else value


def _write_csv(path: Path, rows: list[dict], fields) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field)) for field in fields})


def _relative(path: str | Path, destination: Path) -> str:
    return Path(os.path.relpath(Path(path).resolve(), destination)).as_posix()


def _source_summary(rows: list[dict], source: str, requested_count: int) -> dict:
    usable = [row for row in rows if row["source"] == source and row["status"] == "ok"]
    result = {"source": source, "requested_case_count": requested_count,
              "included_case_count": len(usable), "excluded_case_count": requested_count - len(usable)}
    for field in SUMMARY_FIELDS:
        if field != "area_coverage_fraction":
            result[field] = (sum(row[field] for row in usable) if field.endswith("_count")
                             else math.fsum(row[field] for row in usable)) if usable else None
    total, measured = result["total_length_mm"], result["measured_length_mm"]
    result["area_coverage_fraction"] = measured / total if total else None
    result["reason_counts"] = dict(sum((Counter(row["reason_counts"]) for row in usable), Counter()))
    return result


def _case_markdown(case: str, rows: list[dict], records: dict, directory: Path) -> str:
    lines = [f"# {case}: segment cross-sectional areas", "",
             "Each segment mean uses physical-length-weighted trapezoids over original adjacent samples with "
             "finite ok or needs_review areas. Unavailable gaps are not bridged, filled with zero or extrapolated. "
             "No usable interval means a null area. Coverage is numerical, and does not establish anatomical validity.", "",
             "| Source | Status | Segments | Complete | Partial | Unavailable | Using needs_review | Length coverage (%) |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        if row["status"] != "ok":
            lines.append(f"| {SOURCES[row['source']]} | {row['status']} | — | — | — | — | — | — |")
        else:
            lines.append(f"| {SOURCES[row['source']]} | ok | {row['segment_count']} | {row['complete_count']} | "
                         f"{row['partial_count']} | {row['unavailable_count']} | {row['includes_needs_review_count']} | "
                         f"{100 * row['area_coverage_fraction']:.3f} |")
    for row in rows:
        source = row["source"]
        lines += ["", f"## {SOURCES[source]}", ""]
        record = records.get((source, case), {})
        if row["status"] != "ok":
            lines += [f"Status: `{row['status']}`. {str(row['reason']).replace(chr(10), ' ')}", ""]
        else:
            lines += [f"Original sample status counts: ok={row['ok_sample_count']}, "
                      f"needs_review={row['needs_review_sample_count']}, unavailable={row['unavailable_sample_count']}. "
                      f"Strict consecutive-ok span: {row['strict_ok_span_mm']:.3f} mm.", "",
                      f"Sample reason counts: `{json.dumps(row['reason_counts'], sort_keys=True)}`.", ""]
            for key, label in (("segment_areas_path", "Segment mean areas — CSV"),
                               ("segment_areas_json_path", "Segment mean areas — JSON"),
                               ("samples_path", "All local sample measurements"),
                               ("measurement_path", "Full local measurement record"),
                               ("segments_path", "Original segment quality and coverage"), *OPTIONAL_LINKS):
                if record.get(key):
                    lines.append(f"- [{label}](<{_relative(record[key], directory)}>)")
        provenance = {field: record.get(field, "not_recorded") for field in REVIEW_FIELDS}
        lines += ["", "Inherited review and segmentation provenance:", "", "```json",
                  json.dumps(provenance, indent=2, ensure_ascii=True), "```", ""]
    lines += ["Prediction and reference IDs are independent; no paired errors are inferred. "
              "Existing EDT radii, centerline geometry and review status remain unchanged.", ""]
    return "\n".join(lines)


def _documents(rows: list[dict], source_summaries: list[dict], requested_cases: list[str], status: str) -> tuple[str, str]:
    intro = ("Segment area (mm²) is the physical-length-weighted mean of saved local cross-sectional areas "
             "over original consecutive finite sample pairs. Both ok and needs_review endpoints contribute. "
             "Missing spans are never bridged, zero-filled or extrapolated. Null segment means remain unavailable. "
             "Complete/partial/unavailable describe interval coverage, not anatomical validation. "
             "Prediction and manual-label-derived reference records, when present, have independent branch identities.")
    lines = ["# PARSE cohort segment cross-sectional areas", "", f"Publication status: **{status}**. "
             f"Requested cohort: **{len(requested_cases)} cases** per source.", "", intro, "",
             "| Source | Included / requested cases | Segments | Complete | Partial | Unavailable | Using needs_review | Length coverage (%) |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    body = ['<h1>PARSE cohort segment cross-sectional areas</h1>', f'<p>{html.escape(intro)}</p>',
            f'<p>Publication status: <strong>{status}</strong>. Requested cohort: {len(requested_cases)} cases per source.</p>']
    for row in source_summaries:
        values = ["—" if row[field] is None else f"{row[field]:,.0f}" for field in (
            "segment_count", "complete_count", "partial_count", "unavailable_count", "includes_needs_review_count")]
        coverage = "—" if row["area_coverage_fraction"] is None else f"{100 * row['area_coverage_fraction']:.3f}"
        lines.append(f"| {SOURCES[row['source']]} | {row['included_case_count']} / {row['requested_case_count']} | "
                     + " | ".join([*values, coverage]) + " |")
        body.append(f"<p>{SOURCES[row['source']]}: {row['included_case_count']} / {row['requested_case_count']} "
                    f"cases complete; total segments {values[0]}; complete / partial / unavailable "
                    f"{values[1]} / {values[2]} / {values[3]}; using needs_review {values[4]}; "
                    f"length coverage {coverage}%.</p>")
    lines += ["", "Failed, pending or missing cases are shown explicitly and do not contribute artificial zeros. "
              "Coverage totals include only successful cases; the requested cohort denominator remains unchanged. "
              "No new plots are produced.", "",
              "[Per-case summaries](per_case_summary.csv) · [All segment mean areas](all_segment_areas.csv) · "
              "[Exclusions](exclusions.csv) · [Provenance](sources.json)", "", "## Cases", "",
              "| Case | Source | Status | Segments | Complete | Partial | Unavailable | Using needs_review | Coverage (%) | Reason |",
              "|---|---|---|---:|---:|---:|---:|---:|---:|---|"]
    body += ['<p><a href="report.md">Markdown report</a> · <a href="per_case_summary.csv">Per-case summaries</a> · '
             '<a href="all_segment_areas.csv">All segment mean areas</a> · <a href="exclusions.csv">Exclusions</a> · '
             '<a href="sources.json">Provenance</a></p>',
             '<table><thead><tr><th>Case</th><th>Source</th><th>Status</th><th>Segments</th><th>Complete</th>'
             '<th>Partial</th><th>Unavailable</th><th>Using needs_review</th><th>Coverage (%)</th><th>Reason</th>'
             '</tr></thead><tbody>']
    for row in rows:
        link = f"cases/{row['case']}/report.md"
        values = ["—" if row[field] is None else str(row[field]) for field in (
            "segment_count", "complete_count", "partial_count", "unavailable_count", "includes_needs_review_count")]
        coverage = "—" if row["area_coverage_fraction"] is None else f"{100 * row['area_coverage_fraction']:.3f}"
        reason = str(row["reason"]).replace("\n", " ")
        lines.append(f"| [{row['case']}]({link}) | {SOURCES[row['source']]} | {row['status']} | "
                     + " | ".join([*values, coverage, reason.replace("|", "\\|")]) + " |")
        cells = [SOURCES[row["source"]], row["status"], *values, coverage, reason]
        body.append(f'<tr><td><a href="{link}">{row["case"]}</a></td>'
                    + "".join(f"<td>{html.escape(str(cell))}</td>" for cell in cells) + "</tr>")
    body.append('</tbody></table>')
    page = ('<!doctype html><html lang="en"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            '<title>PARSE segment cross-sectional areas</title><style>'
            'body{background:white;color:#303030;font:14px/1.5 Arial,sans-serif;margin:24px}'
            'table{border-collapse:collapse}th,td{padding:6px 9px;border-bottom:1px solid #ddd;text-align:left}'
            'a{color:#345f84}th{white-space:nowrap}</style></head><body>' + "".join(body) + '</body></html>\n')
    return "\n".join(lines) + "\n", page


def build_report(records: list[dict], requested_cases: list[str], report_dir: Path, *,
                 run_id: str, computation_dir: Path) -> dict:
    """Validate one case at a time and atomically publish streamed cohort tables."""
    if (not isinstance(requested_cases, list) or not requested_cases
            or any(not isinstance(case, str) or not re.fullmatch(r"PA[0-9]{6}", case) for case in requested_cases)
            or len(set(requested_cases)) != len(requested_cases)):
        raise ValueError("Requested cases must be unique PARSE case IDs")
    report_dir, computation_dir = Path(report_dir).resolve(), Path(computation_dir).resolve()
    if (report_dir.exists() or report_dir == computation_dir
            or report_dir in computation_dir.parents or computation_dir in report_dir.parents):
        raise ValueError("Publication must use a new directory separate from computation results")
    indexed = {}
    for record in records:
        source, case = record.get("source", "prediction"), record.get("case")
        if source not in SOURCES or case not in requested_cases or (source, case) in indexed:
            raise ValueError("Unknown, unrequested or duplicate source/case record")
        if record.get("source") != source:
            raise ValueError("Each case record must declare its source")
        indexed[source, case] = record
    source_names = sorted({key[0] for key in indexed} or {"prediction"})
    hashes = _collect_inputs(records, report_dir)
    code_paths = [Path(__file__).resolve(), *[Path(__file__).with_name(name).resolve() for name in (
        "segment_cross_sections.py", "section_measurement.py", "measurement_io.py",
        "run_edt_experiment.py")]]
    code_hashes = {str(path): _sha256(path) for path in code_paths}
    case_rows, exclusions, fields = [], [], ["source", "case", "branch_id"]
    for case in requested_cases:
        for source in source_names:
            record = indexed.get((source, case))
            if record is not None and record.get("status") == "ok":
                segments, summary = _validated_case(record)
                for segment in segments:
                    for field in segment:
                        if field not in fields:
                            fields.append(field)
                case_rows.append(summary)
                del segments
            else:
                state = record.get("status", "missing") if record else "missing"
                reason = str(record.get("error") or record.get("reason") or "Computation has not completed successfully") if record else "No case result record"
                row = {field: None for field in CASE_FIELDS}
                row.update(source=source, case=case, status=state, reason=reason)
                case_rows.append(row)
                exclusions.append({"source": source, "case": case, "status": state, "reason": reason})
    summaries = [_source_summary(case_rows, source, len(requested_cases)) for source in source_names]
    status = "partial" if exclusions else "complete"
    report_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".cross_section_cohort_", dir=report_dir.parent) as temporary:
        staging = Path(temporary)
        _write_csv(staging / "per_case_summary.csv", case_rows, CASE_FIELDS)
        _write_csv(staging / "exclusions.csv", exclusions, ("source", "case", "status", "reason"))
        segment_count = 0
        with (staging / "all_segment_areas.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for summary in case_rows:
                if summary["status"] != "ok":
                    continue
                source, case = summary["source"], summary["case"]
                count = 0
                with Path(indexed[source, case]["segment_areas_path"]).open(newline="", encoding="utf-8") as source_handle:
                    for row in csv.DictReader(source_handle):
                        writer.writerow({**row, "source": source, "case": case})
                        count += 1
                if count != summary["segment_count"]:
                    raise ValueError("Segment CSV row count changed during report publication")
                segment_count += count
        for case in requested_cases:
            directory = staging / "cases" / case
            directory.mkdir(parents=True)
            text = _case_markdown(case, [row for row in case_rows if row["case"] == case], indexed,
                                  report_dir / "cases" / case)
            (directory / "report.md").write_text(text, encoding="utf-8")
        markdown, page = _documents(case_rows, summaries, requested_cases, status)
        (staging / "report.md").write_text(markdown, encoding="utf-8")
        (staging / "index.html").write_text(page, encoding="utf-8")
        publication = {
            "schema_version": 1, "dataset": "PARSE", "status": status, "run_id": run_id,
            "sources": source_names, "computation_dir": str(computation_dir),
            "report_path": str(report_dir / "report.md"), "index_path": str(report_dir / "index.html"),
            "sources_path": str(report_dir / "sources.json"), "requested_cases": requested_cases,
            "requested_case_count": len(requested_cases), "requested_source_case_count": len(case_rows),
            "included_source_case_count": len(case_rows) - len(exclusions), "excluded_source_case_count": len(exclusions),
            "included_cases": [case for case in requested_cases if any(row["case"] == case and row["status"] == "ok" for row in case_rows)],
            "case_summaries": case_rows, "source_summaries": summaries, "exclusions": exclusions,
            "segment_count": segment_count, "segment_fields": fields, "segment_area_method": METHOD,
            "definitions": DEFINITIONS, "anatomical_validation": "not_performed", "review_status_promotion": False,
            "case_provenance": [{key: record[key] for key in ("case", "source", "status", *REVIEW_FIELDS,
                "worker_report_path", "worker_report_sha256", "parameters", "versions", "computation_identity",
                "provenance", "historical_review_code", "baseline_reused", "source_report_path", "radius_segments_path") if key in record}
                                for record in records],
            "source_sha256": hashes, "code_sha256": code_hashes,
            "record_payload_sha256": hashlib.sha256(json.dumps(records, sort_keys=True, allow_nan=False).encode()).hexdigest(),
            "input_integrity": "verified_before_publication", "new_plots": False,
            "artifact_sha256": {str(report_dir / path.relative_to(staging)): _sha256(path)
                                for path in sorted(staging.rglob("*")) if path.is_file()},
        }
        _verify(hashes)
        _verify(code_hashes)
        publication["input_integrity"] = "verified_after_publication"
        (staging / "sources.json").write_text(json.dumps(publication, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        return _publish_verified(staging, report_dir, publication, hashes, code_hashes)
