#!/usr/bin/env python3
"""Measure local and segment cross-sectional areas on saved PARSE prediction trees."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, fields
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import threading
import time

from measurement_io import write_json
from centerline_cross_sections import Config
from run_edt_experiment import sha256
from run_parse_radius_statistics import _lock, _read, _stamp, _stop_process


WORKER = Path(__file__).with_name("parse_cross_section_case.py")
_PROCESS_LOCK = threading.Lock()
_PROCESSES = set()
_STOP = threading.Event()


def _verify(hashes: dict) -> None:
    if not hashes:
        raise ValueError("Required cross-section hash mapping is empty")
    for name, digest in hashes.items():
        if (not Path(name).is_absolute() or not re.fullmatch(r"[0-9a-f]{64}", digest)
                or sha256(Path(name)) != digest):
            raise ValueError(f"Cross-section SHA-256 mismatch: {name}")


def _merge_hashes(*mappings: dict) -> dict:
    output = {}
    for mapping in mappings:
        for name, digest in mapping.items():
            if name in output and output[name] != digest:
                raise ValueError(f"Conflicting cross-section source hashes: {name}")
            output[name] = digest
    return output


def _runtime_identity(config: Config) -> dict:
    from parse_cross_section_case import computation_identity
    return computation_identity(config)


def _source_manifest(source_run_dir: Path) -> dict:
    """Bind the complete upstream membership and all fixed checkpoint records."""
    source_run_dir = Path(source_run_dir).resolve()
    manifest_path, report_path = source_run_dir / "manifest.json", source_run_dir / "report.json"
    manifest, report = _read(manifest_path), _read(report_path)
    cases = manifest.get("cases", [])
    if (manifest.get("dataset") != "PARSE" or manifest.get("source") != "prediction"
            or manifest.get("component_policy") != "largest_26_connected"
            or not cases or len(set(cases)) != len(cases)
            or any(not isinstance(case, str) or not re.fullmatch(r"PA[0-9]{6}", case) for case in cases)):
        raise ValueError("Expected a fixed PARSE prediction cohort using the largest 26-connected component")
    if (report.get("status") != "complete" or report.get("dataset") != "PARSE"
            or report.get("source") != "prediction" or report.get("run_id") != manifest.get("run_id")
            or report.get("case_count") != len(cases) or report.get("completed_count") != len(cases)
            or report.get("failed_count") != 0 or report.get("pending_count") != 0):
        raise ValueError("Source radius cohort must be complete and successful")
    records = {row["case"]: row for row in report["records"]}
    if len(records) != len(report["records"]) or set(records) != set(cases):
        raise ValueError("Source cohort report membership differs from its manifest")
    hashes = {str(path): sha256(path) for path in (manifest_path, report_path)}
    checkpoints = {}
    for case in cases:
        path = source_run_dir / "checkpoints" / f"{case}.json"
        value = _read(path)
        if value != records[case] or value.get("case") != case or value.get("status") != "ok":
            raise ValueError(f"Source checkpoint differs from its successful cohort report: {case}")
        hashes[str(path)] = sha256(path)
        checkpoints[case] = str(path)
    return {"cases": cases, "source_sha256": hashes, "checkpoints": checkpoints,
            "source_run_id": manifest["run_id"], "component_policy": manifest["component_policy"]}


def _cached_record(case: str, checkpoint: Path, source_checkpoint: Path, identity: dict) -> dict:
    if not checkpoint.is_file():
        return {"case": case, "source": "prediction", "status": "pending", "error": "Not computed"}
    try:
        record = _read(checkpoint)
        if record.get("case") != case or record.get("source") != "prediction":
            raise ValueError("Checkpoint case/source identity differs")
        if record.get("status") not in {"ok", "error", "pending"}:
            raise ValueError("Checkpoint status is missing or unsupported")
        if record.get("status") != "ok":
            return record
        if (record.get("technical_validation") != "passed"
                or record.get("input_integrity") != "verified_after_computation"
                or record.get("computation_identity") != identity):
            raise ValueError("Cached scientific computation identity or validation changed")
        if record.get("source_sha256", {}).get(str(source_checkpoint)) != sha256(source_checkpoint):
            raise ValueError("Cached result does not bind the current source checkpoint")
        _verify(_merge_hashes(record["source_sha256"], record["code_sha256"], record["artifact_sha256"],
                              {record["worker_report_path"]: record["worker_report_sha256"]}))
        saved = _read(Path(record["worker_report_path"]))
        for key, value in saved.items():
            if key != "cache_status" and record.get(key) != value:
                raise ValueError(f"Checkpoint differs from its verified worker record: {key}")
        return dict(record, cache_status="reused")
    except (OSError, ValueError, KeyError, TypeError) as error:
        return {"case": case, "source": "prediction", "status": "pending",
                "error": f"Invalidated cache: {error}"}


def _compute_case(case: str, args: argparse.Namespace, attempt_dir: Path) -> dict:
    command = [sys.executable, str(WORKER), "--case", case,
               "--source-checkpoint", str(args.source_run_dir / "checkpoints" / f"{case}.json"),
               "--output-dir", str(attempt_dir), "--parameters-json", json.dumps(asdict(args.config)),
               "--memory-limit-gb", str(args.memory_limit_gb)]
    environment = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    log_path = attempt_dir.with_suffix(".log")
    with log_path.open("w", encoding="utf-8") as log:
        with _PROCESS_LOCK:
            if _STOP.is_set():
                raise RuntimeError("Cross-section batch was interrupted")
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                       env=environment, start_new_session=True)
            _PROCESSES.add(process)
        try:
            try:
                code = process.wait(timeout=args.timeout_seconds)
            except subprocess.TimeoutExpired as error:
                _stop_process(process)
                raise RuntimeError(f"Case exceeded {args.timeout_seconds:g} seconds; see {log_path}") from error
            if code:
                failure = attempt_dir / "failure.json"
                detail = _read(failure).get("error", "") if failure.is_file() else ""
                raise RuntimeError(f"Worker exited {code}: {detail}; log: {log_path}")
            return _read(attempt_dir / "report.json")
        finally:
            with _PROCESS_LOCK:
                _PROCESSES.discard(process)


def _execute(case: str, args: argparse.Namespace, run_dir: Path, identity: dict) -> dict:
    attempt_dir = run_dir / "cases" / case / ("attempt_" + _stamp())
    attempt_dir.parent.mkdir(parents=True, exist_ok=True)
    try:
        record = _compute_case(case, args, attempt_dir)
        if (record.get("case") != case or record.get("source") != "prediction"
                or record.get("status") != "ok" or record.get("technical_validation") != "passed"
                or record.get("input_integrity") != "verified_after_computation"
                or record.get("computation_identity") != identity or _runtime_identity(args.config) != identity):
            raise ValueError("Worker result differs from the requested scientific computation")
        source_path = args.source_run_dir / "checkpoints" / f"{case}.json"
        if record.get("source_sha256", {}).get(str(source_path)) != sha256(source_path):
            raise ValueError("Worker result does not bind the fixed source checkpoint")
        _verify(_merge_hashes(record["source_sha256"], record["code_sha256"], record["artifact_sha256"]))
        return dict(record, cache_status="computed", worker_report_path=str(attempt_dir / "report.json"),
                    worker_report_sha256=sha256(attempt_dir / "report.json"))
    except Exception as error:
        return {"case": case, "source": "prediction", "status": "error",
                "error": f"{type(error).__name__}: {error}", "attempt_dir": str(attempt_dir)}


def _progress(run_dir: Path, run_id: str, records: dict, *, status: str = "running") -> dict:
    rows = [records[case] for case in sorted(records)]
    progress = {"schema_version": 1, "dataset": "PARSE", "source": "prediction", "run_id": run_id,
                "status": status, "case_count": len(rows),
                "completed_count": sum(row["status"] == "ok" for row in rows),
                "failed_count": sum(row["status"] == "error" for row in rows),
                "pending_count": sum(row["status"] == "pending" for row in rows), "records": rows}
    write_json(run_dir / "progress.json", progress)
    return progress


def run_batch(args: argparse.Namespace) -> dict:
    from parse_cross_section_statistics import build_report

    started = time.monotonic()
    for field in ("source_run_dir", "output_root", "report_root"):
        setattr(args, field, Path(getattr(args, field)).resolve())
    args.config.validate()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.run_id):
        raise ValueError("Run ID may contain only letters, digits, underscores and hyphens")
    if (args.workers < 1 or not math.isfinite(args.timeout_seconds) or args.timeout_seconds <= 0
            or not math.isfinite(args.memory_limit_gb) or args.memory_limit_gb <= 0):
        raise ValueError("Worker count, timeout and memory limit must be positive")
    source = _source_manifest(args.source_run_dir)
    cases = source["cases"]
    selected = cases if not args.cases else list(args.cases)
    if len(set(selected)) != len(selected) or set(selected) - set(cases):
        raise ValueError("Selected cases must be unique members of the source PARSE cohort")
    run_dir, publication_root = args.output_root / args.run_id, args.report_root / args.run_id
    protected = [args.source_run_dir]
    for destination in (run_dir, publication_root):
        if any(destination == folder or folder in destination.parents or destination in folder.parents
               for folder in protected):
            raise ValueError("Cross-section outputs must be separate from original input directories")
    if run_dir == publication_root or run_dir in publication_root.parents or publication_root in run_dir.parents:
        raise ValueError("Computation and report roots must be separate")
    manifest = {"schema_version": 1, "dataset": "PARSE", "source": "prediction", "run_id": args.run_id,
                "cases": cases, "source_run_dir": str(args.source_run_dir),
                "source_manifest": source, "parameters": asdict(args.config),
                "component_policy": "largest_26_connected", "included_area_statuses": ["ok", "needs_review"]}
    orchestrator = {str(Path(__file__).resolve()): sha256(Path(__file__))}
    run_dir.mkdir(parents=True, exist_ok=True)
    _STOP.clear()
    with _lock(run_dir):
        manifest_path = run_dir / "manifest.json"
        if manifest_path.exists():
            if _read(manifest_path) != manifest:
                raise ValueError("Saved cohort inputs or measurement parameters changed; use a new run ID")
        else:
            if any(path.name != ".batch.lock" for path in run_dir.iterdir()):
                raise ValueError("Existing output directory has no recognized cross-section manifest")
            write_json(manifest_path, manifest)
        checkpoint_dir = run_dir / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)
        identity = _runtime_identity(args.config)
        records = {case: _cached_record(case, checkpoint_dir / f"{case}.json",
                                       Path(source["checkpoints"][case]), identity) for case in cases}
        _progress(run_dir, args.run_id, records)
        jobs = [case for case in selected if records[case]["status"] != "ok"]
        print(f"PARSE cross sections: {len(cases)} cases; selected {len(selected)}; compute {len(jobs)}; "
              f"reuse {len(selected) - len(jobs)}; workers {args.workers}", flush=True)
        executor = ThreadPoolExecutor(max_workers=args.workers)
        try:
            futures = {executor.submit(_execute, case, args, run_dir, identity): case for case in jobs}
            for future in as_completed(futures):
                case = futures[future]
                records[case] = future.result()
                write_json(checkpoint_dir / f"{case}.json", records[case])
                progress = _progress(run_dir, args.run_id, records)
                print(json.dumps({"case": case, "status": records[case]["status"],
                                  "completed": progress["completed_count"], "total": len(cases),
                                  "error": records[case].get("error")}), flush=True)
        except BaseException:
            _STOP.set()
            with _PROCESS_LOCK:
                for process in list(_PROCESSES):
                    _stop_process(process)
            executor.shutdown(wait=True, cancel_futures=True)
            _progress(run_dir, args.run_id, records, status="interrupted")
            raise
        finally:
            executor.shutdown(wait=True, cancel_futures=True)
        for case, record in list(records.items()):
            if record["status"] == "ok":
                checked = _cached_record(case, checkpoint_dir / f"{case}.json",
                                         Path(source["checkpoints"][case]), identity)
                if checked["status"] != "ok":
                    checked["status"] = "error"
                else:
                    checked["cache_status"] = record.get("cache_status", "reused")
                records[case] = checked
                write_json(checkpoint_dir / f"{case}.json", checked)
        publication_dir = publication_root / "publications" / _stamp()
        if publication_dir.exists():
            raise ValueError("A new publication directory is required")
        try:
            _verify(_merge_hashes(source["source_sha256"], orchestrator))
            publication = build_report([records[case] for case in cases], cases, publication_dir,
                                       run_id=args.run_id, computation_dir=run_dir)
            _verify(_merge_hashes(source["source_sha256"], orchestrator))
            complete = all(row["status"] == "ok" for row in records.values())
            if publication.get("status") != ("complete" if complete else "partial"):
                raise ValueError("Publication status disagrees with actual case completion")
        except BaseException as error:
            # This invocation owns this previously absent publication directory.
            # A late input change or interrupt must not leave a valid-looking
            # report behind; older publications and latest.json remain intact.
            if publication_dir.is_dir():
                shutil.rmtree(publication_dir)
            failure = _progress(run_dir, args.run_id, records,
                                status="interrupted" if isinstance(error, KeyboardInterrupt) else "publication_failed")
            failure.update(error=f"{type(error).__name__}: {error}", publication_dir=str(publication_dir),
                           elapsed_seconds=time.monotonic() - started)
            write_json(run_dir / "failure.json", failure)
            write_json(run_dir / "report.json", failure)
            raise
        result = _progress(run_dir, args.run_id, records, status="complete" if complete else "partial")
        result.update(selected_cases=selected, elapsed_seconds=time.monotonic() - started,
                      workers=args.workers, publication=publication, computation_identity=identity,
                      orchestrator_sha256=orchestrator, source_sha256=source["source_sha256"])
        write_json(run_dir / "report.json", result)
        if complete:
            write_json(publication_root / "latest.json", {"run_id": args.run_id, "status": "complete",
                       "report_path": publication["report_path"], "sources_path": publication["sources_path"],
                       "sources_sha256": sha256(Path(publication["sources_path"]))})
        print(json.dumps({"status": result["status"], "completed": result["completed_count"],
                          "total": len(cases), "report": publication["report_path"]}), flush=True)
        return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run-dir", type=Path,
                        required=True)
    parser.add_argument("--output-root", type=Path, default=Path("data/parse/results/cross_section_cohort"))
    parser.add_argument("--report-root", type=Path, default=Path("data/parse/reports/cross_section_cohort"))
    parser.add_argument("--run-id", default=_stamp(), help="Use the same run ID to resume the fixed cohort")
    parser.add_argument("--case", dest="cases", action="append")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout-seconds", type=float, default=900.)
    parser.add_argument("--memory-limit-gb", type=float, default=6.)
    for field in fields(Config):
        parser.add_argument("--" + field.name.replace("_", "-"), type=float, default=field.default)
    args = parser.parse_args(argv)
    args.config = Config(**{field.name: getattr(args, field.name) for field in fields(Config)})
    try:
        args.config.validate()
    except ValueError as error:
        parser.error(str(error))
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    def interrupt(signum, frame):
        raise KeyboardInterrupt

    previous_handler = signal.signal(signal.SIGTERM, interrupt)
    try:
        result = run_batch(args)
    except KeyboardInterrupt:
        print("Cross-section batch interrupted; completed checkpoints are retained.", file=sys.stderr)
        return 130
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as error:
        print(f"Cross-section batch failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    finally:
        signal.signal(signal.SIGTERM, previous_handler)
    selected = [row for row in result["records"] if row["case"] in result["selected_cases"]]
    return 0 if all(row["status"] == "ok" for row in selected) else 1


if __name__ == "__main__":
    raise SystemExit(main())
