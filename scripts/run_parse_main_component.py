#!/usr/bin/env python3
"""Measure PARSE artery and airway largest components from saved predictions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess
import sys

from dataset_cases import discover_ct_cases
from measurement_scope import load_profile


def initialize_profile(input_root: Path, source_results_root: Path, output_root: Path) -> dict:
    inputs, source, output = (Path(path).resolve() for path in
                             (input_root, source_results_root, output_root))
    if (output == source or source in output.parents or output in source.parents
            or output == inputs or inputs in output.parents or output in inputs.parents):
        raise ValueError("Measurement outputs must be separate from original inputs and predictions")
    profile = {"schema_version": 1, "dataset": "PARSE", "measurement_scope": "largest_26",
               "input_root": str(inputs), "source_results_root": str(source), "run_id": "study"}
    path = output / "measurement_scope.json"
    if path.exists():
        if load_profile(output) != profile:
            raise ValueError("Existing measurements belong to different source roots")
    else:
        if output.exists() and any(output.iterdir()):
            raise ValueError("Existing output directory has no measurement profile")
        output.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(profile, indent=2) + "\n", encoding="utf-8")
    return profile


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--source-results-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--case", dest="cases", action="append")
    parser.add_argument("--fiji", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    cases = discover_ct_cases(args.input_root, args.cases)
    if any(not re.fullmatch(r"PA[0-9]{6}", case.name) for case in cases):
        raise ValueError("This command accepts PARSE cases only")
    initialize_profile(args.input_root, args.source_results_root, args.output_root)
    command = [sys.executable, str(Path(__file__).with_name("run_parse_pipeline.py")),
               "--input-root", str(args.input_root.resolve()), "--output-root", str(args.output_root.resolve()),
               "--measurement-scope", "largest_26", "--with-lung-volume", "--with-airway", "--device", "cpu"]
    for case in cases:
        command += ["--case", case.name]
    if args.fiji:
        command += ["--fiji", str(args.fiji)]
    if args.overwrite:
        command += ["--overwrite"]
    return subprocess.run(command, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
