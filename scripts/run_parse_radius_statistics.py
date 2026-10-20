#!/usr/bin/env python3
"""Measure prediction-only PARSE EDT segment-radius bins with resumable cases."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
import csv
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import threading
import time
import uuid

from measurement_io import write_csv, write_json
from dataset_cases import discover_ct_cases
from run_edt_experiment import sha256


WORKER = Path(__file__).with_name("parse_radius_case.py")
_PROCESS_LOCK = threading.Lock()
_PROCESSES = set()
_STOP = threading.Event()


def _read(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ_") + uuid.uuid4().hex[:8]


def _runtime_identity() -> dict:
    from parse_radius_case import computation_identity
    return computation_identity()


def _source_snapshot(case: str, args: argparse.Namespace) -> dict:
    from parse_radius_case import source_snapshot
    return source_snapshot(case, args.input_root, args.results_root)


def _verify_hashes(hashes: dict) -> None:
    if not hashes:
        raise ValueError("Required artifact hash mapping is empty")
    for name, expected in hashes.items():
        if not Path(name).is_absolute() or sha256(Path(name)) != expected:
            raise ValueError(f"Artifact SHA-256 mismatch: {name}")


@contextmanager
def _lock(run_dir: Path):
    with (run_dir / ".batch.lock").open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another radius batch is already writing this run") from error
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _stop_process(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()
    except ProcessLookupError:
        process.wait()


def _compute_case(case: str, args: argparse.Namespace, attempt_dir: Path) -> dict:
    """Run each case in a bounded subprocess; worker logs stay with the attempt."""
    command = [sys.executable, str(WORKER), "--case", case,
               "--input-root", str(args.input_root), "--results-root", str(args.results_root),
               "--output-dir", str(attempt_dir), "--memory-limit-gb", str(args.memory_limit_gb)]
    environment = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    log_path = attempt_dir.with_suffix(".log")
    with log_path.open("w", encoding="utf-8") as log:
        with _PROCESS_LOCK:
            if _STOP.is_set():
                raise RuntimeError("Radius batch was interrupted")
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
                detail = _read(failure).get("error", "") if failure.exists() else ""
                raise RuntimeError(f"Worker exited {code}: {detail}; log: {log_path}")
            return _read(attempt_dir / "report.json")
        finally:
            with _PROCESS_LOCK:
                _PROCESSES.discard(process)


def _load_segments(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        records = list(csv.DictReader(handle))
    for record in records:
        for key in ("branch_id", "start_node_id", "end_node_id", "point_count", "edge_count"):
            if key in record:
                record[key] = int(record[key])
        for key in ("length_mm", "segment_radius_mm", "r_edt_length_weighted_mean_mm",
                    "r_edt_min_mm", "r_edt_max_mm"):
            if key in record:
                record[key] = float(record[key])
    return records


def _finalize_case(case: str, record: dict, attempt_dir: Path, args: argparse.Namespace,
                   identity: dict) -> dict:
    from parse_radius_statistics import classify_segments
    if (record.get("case") != case or record.get("source") != "prediction"
            or record.get("status") != "ok" or record.get("technical_validation") != "passed"):
        raise ValueError("A successful, technically validated prediction worker result is required")
    if record.get("source_sha256") != _source_snapshot(case, args):
        raise ValueError("Case inputs changed during computation")
    if _runtime_identity() != identity:
        raise ValueError("Radius computation code or parameters changed during the run")
    _verify_hashes(record["artifact_sha256"])
    classified = classify_segments(_load_segments(Path(record["segments_path"])))
    if len(classified["segments"]) != record["summary"]["segment_count"]:
        raise ValueError("Radius table and tree segment counts disagree")
    bins_path, table_path = attempt_dir / "radius_bins.json", attempt_dir / "radius_segment_bins.csv"
    write_csv(table_path, classified["segments"])
    write_json(bins_path, {"case": case, "bins": classified["bins"]})
    record = dict(record, bins=classified["bins"], segment_bins_path=str(table_path),
                  computation_identity=identity, cache_status="computed",
                  worker_report_path=str(attempt_dir / "report.json"),
                  worker_report_sha256=sha256(attempt_dir / "report.json"))
    record["artifact_sha256"] = {**record["artifact_sha256"], str(table_path): sha256(table_path),
                                  str(bins_path): sha256(bins_path)}
    record["grouping_code_sha256"] = sha256(Path(__file__).with_name("parse_radius_statistics.py"))
    return record


def _cached_record(case: str, checkpoint: Path, args: argparse.Namespace, identity: dict) -> dict:
    from parse_radius_statistics import classify_segments
    if not checkpoint.exists():
        return {"case": case, "status": "pending", "error": "Not computed"}
    try:
        previous = _read(checkpoint)
        if previous.get("case") != case:
            raise ValueError(f"Checkpoint case mismatch: {case}")
        if previous.get("status") != "ok":
            return previous
        if previous.get("computation_identity") != identity:
            raise ValueError("Computation identity changed")
        if previous.get("source_sha256") != _source_snapshot(case, args):
            raise ValueError("Source identity changed")
        _verify_hashes(previous["artifact_sha256"])
        _verify_hashes({previous["worker_report_path"]: previous["worker_report_sha256"]})
        classified = classify_segments(_load_segments(Path(previous["segments_path"])))
        if (classified["bins"] != previous["bins"]
                or len(classified["segments"]) != previous["summary"]["segment_count"]):
            raise ValueError("Saved radius-bin summary changed")
        return dict(previous, cache_status="reused")
    except (ValueError, OSError, KeyError) as error:
        return {"case": case, "status": "pending", "error": f"Invalidated cache: {error}"}


def _progress(run_dir: Path, run_id: str, records: dict, *, status: str = "running") -> dict:
    rows = [records[case] for case in sorted(records)]
    result = {"schema_version": 1, "dataset": "PARSE", "source": "prediction", "run_id": run_id,
              "status": status, "case_count": len(rows),
              "completed_count": sum(row["status"] == "ok" for row in rows),
              "failed_count": sum(row["status"] == "error" for row in rows),
              "pending_count": sum(row["status"] == "pending" for row in rows), "records": rows}
    write_json(run_dir / "progress.json", result)
    return result


def _execute(case: str, args: argparse.Namespace, run_dir: Path, identity: dict) -> dict:
    attempt_dir = run_dir / "cases" / case / ("attempt_" + _stamp())
    attempt_dir.parent.mkdir(parents=True, exist_ok=True)
    print(f"Starting {case}", flush=True)
    try:
        return _finalize_case(case, _compute_case(case, args, attempt_dir), attempt_dir, args, identity)
    except Exception as error:
        record = {"case": case, "status": "error", "error": f"{type(error).__name__}: {error}",
                  "attempt_dir": str(attempt_dir), "computation_identity": identity}
        write_json(attempt_dir.with_suffix(".failure.json"), record)
        return record


def run_batch(args: argparse.Namespace) -> dict:
    from parse_radius_statistics import build_report

    started = time.monotonic()
    for name in ("input_root", "results_root", "output_root", "report_root"):
        setattr(args, name, Path(getattr(args, name)).resolve())
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.run_id):
        raise ValueError("Run ID may contain only letters, digits, underscores and hyphens")
    if args.workers < 1 or args.timeout_seconds <= 0 or args.memory_limit_gb <= 0:
        raise ValueError("Workers, timeout and memory limit must be positive")
    cases = [path.name for path in discover_ct_cases(args.input_root)]
    if any(not re.fullmatch(r"PA[0-9]{6}", case) for case in cases):
        raise ValueError("This command supports PARSE case IDs only")
    selected = cases if not args.cases else list(args.cases)
    if len(set(selected)) != len(selected) or set(selected) - set(cases):
        raise ValueError("Selected cases must be unique members of the PARSE cohort")
    if getattr(args, "selected_cohort", False):
        cases = selected
    run_dir = args.output_root / args.run_id
    publication_root = args.report_root / args.run_id
    protected = [args.input_root, *(args.results_root / stage for stage in
                                   ("totalsegmentator", "skeletonize3d", "analyzeskeleton"))]
    for destination in (run_dir, publication_root):
        if any(destination == folder or folder in destination.parents or destination in folder.parents
               for folder in protected):
            raise ValueError("Radius outputs must be separate from original input directories")
    if (run_dir == publication_root or run_dir in publication_root.parents
            or publication_root in run_dir.parents):
        raise ValueError("Computation and report directories must be separate")
    run_dir.mkdir(parents=True, exist_ok=True)
    _STOP.clear()
    with _lock(run_dir):
        manifest_path = run_dir / "manifest.json"
        manifest = {"schema_version": 1, "dataset": "PARSE", "source": "prediction",
                    "run_id": args.run_id, "cases": cases, "input_root": str(args.input_root),
                    "results_root": str(args.results_root), "component_policy": "largest_26_connected",
                    "radius_bins": {"boundaries_mm": [0, 2, 4, 6], "upper_bounds_inclusive": True,
                                    "last_bin": "radius > 6 mm", "invalid_radius": "case_failure"}}
        if manifest_path.exists():
            if _read(manifest_path) != manifest:
                raise ValueError("Saved radius cohort membership or input roots changed; use a new run ID")
        else:
            if any(path.name != ".batch.lock" for path in run_dir.iterdir()):
                raise ValueError("Existing output directory has no recognized radius cohort manifest")
            write_json(manifest_path, manifest)
        checkpoint_dir = run_dir / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)
        identity = _runtime_identity()
        records = {case: _cached_record(case, checkpoint_dir / f"{case}.json", args, identity) for case in cases}
        _progress(run_dir, args.run_id, records)
        jobs = [case for case in selected if records[case]["status"] != "ok"]
        print(f"PARSE cohort: {len(cases)} cases; selected {len(selected)}; compute {len(jobs)}; "
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
        # Recheck before publication, including reused cases outside --case.
        for case, record in list(records.items()):
            if record["status"] == "ok":
                checked = _cached_record(case, checkpoint_dir / f"{case}.json", args, identity)
                if checked["status"] != "ok":
                    checked.update(status="error")
                else:
                    checked["cache_status"] = record.get("cache_status", "reused")
                records[case] = checked
                write_json(checkpoint_dir / f"{case}.json", checked)
        publication_dir = publication_root / "publications" / _stamp()
        try:
            publication = build_report([records[case] for case in cases], cases, publication_dir,
                                       run_id=args.run_id, computation_dir=run_dir)
        except Exception as error:
            failure = _progress(run_dir, args.run_id, records, status="publication_failed")
            failure.update(error=f"{type(error).__name__}: {error}",
                           elapsed_seconds=time.monotonic() - started,
                           publication_dir=str(publication_dir), computation_identity=identity)
            write_json(run_dir / "failure.json", failure)
            write_json(run_dir / "report.json", failure)
            raise
        result = _progress(run_dir, args.run_id, records, status=publication["status"])
        result.update(elapsed_seconds=time.monotonic() - started, selected_cases=selected,
                      workers=args.workers, publication=publication, computation_identity=identity,
                      orchestrator_sha256=sha256(Path(__file__)))
        write_json(run_dir / "report.json", result)
        write_json(publication_root / "latest.json", {"run_id": args.run_id,
                   "status": result["status"], "case_count": len(cases),
                   "completed_count": result["completed_count"], "report_path": publication["report_path"]})
        return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=Path("data/parse/train"))
    parser.add_argument("--results-root", type=Path, default=Path("data/parse/results"))
    parser.add_argument("--output-root", type=Path, default=Path("data/parse/results/edt_radius_cohort"))
    parser.add_argument("--report-root", type=Path, default=Path("data/parse/statistics/artery_radius"))
    parser.add_argument("--run-id", default=_stamp(), help="Use the same ID to resume the fixed cohort")
    parser.add_argument("--case", dest="cases", action="append", help="Schedule selected cases within the full saved cohort")
    parser.add_argument("--selected-cohort", action="store_true",
                        help="Treat --case selections as the entire cohort, for single-case reproduction.")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--timeout-seconds", type=float, default=600)
    parser.add_argument("--memory-limit-gb", type=float, default=10)
    args = parser.parse_args(argv)
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        result = run_batch(args)
    except Exception as error:
        print(f"PARSE radius batch failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    print(json.dumps({"status": result["status"], "completed_count": result["completed_count"],
                      "case_count": result["case_count"], "report": result["publication"]["report_path"]}))
    selected_records = [row for row in result["records"] if row["case"] in result["selected_cases"]]
    return 0 if all(row["status"] == "ok" for row in selected_records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
