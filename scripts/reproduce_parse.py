#!/usr/bin/env python3
"""Reproduce PARSE measurements from public scans, without historical outputs."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
import subprocess
import sys

from dataset_cases import discover_ct_cases


STAGES = ("segmentation", "metrics", "airway", "radius", "cross-sections", "statistics")
SCRIPTS = Path(__file__).resolve().parent


def paths(output_root: Path) -> dict[str, Path]:
    output = Path(output_root).resolve()
    return {"raw": output / "raw", "measurements": output / "measurements",
            "radius": output / "radius" / "cohort", "sections": output / "cross_sections" / "cohort",
            "statistics": output / "statistics"}


def command(script: str, *arguments) -> None:
    subprocess.run([sys.executable, str(SCRIPTS / script), *map(str, arguments)], check=True)


def _publication(root: Path) -> Path:
    result = json.loads((root / "report.json").read_text(encoding="utf-8"))
    if result.get("status") != "complete":
        raise ValueError(f"The requested upstream cohort is incomplete: {root}")
    return Path(result["publication"]["sources_path"]).parent


def run(args) -> dict:
    inputs, output = args.input_root.resolve(), args.output_root.resolve()
    if output == inputs or inputs in output.parents or output in inputs.parents:
        raise ValueError("Output root must be separate from original inputs")
    selected = discover_ct_cases(inputs, args.cases)
    if any(not re.fullmatch(r"PA[0-9]{6}", case.name) for case in selected):
        raise ValueError("The PARSE workflow requires case IDs such as PA000005")
    roots = paths(output)
    flags = [item for case in selected for item in ("--case", case.name)]
    stages = STAGES if args.stage == "all" else (args.stage,)
    fiji = ["--fiji", args.fiji] if args.fiji else []
    if "segmentation" in stages:
        command("run_totalsegmentator_parse.py", "--input-root", inputs, "--output-root",
                roots["raw"] / "totalsegmentator", "--device", args.device, *flags)
        command("run_lung_volume_parse.py", "--input-root", inputs, "--output-root",
                roots["raw"] / "lung_volume", "--device", args.device, *flags)
    if "metrics" in stages:
        command("run_parse_main_component.py", "--input-root", inputs,
                "--source-results-root", roots["raw"], "--output-root", roots["measurements"],
                *fiji, *flags)
    if "airway" in stages:
        command("run_parse_airway.py", "--measurement-root", roots["measurements"],
                "--output-root", output / "airway", "--workers", args.workers, *flags)
    if "radius" in stages:
        command("run_parse_radius_statistics.py", "--input-root", inputs,
                "--results-root", roots["measurements"], "--output-root", output / "radius",
                "--report-root", roots["statistics"] / "radius", "--run-id", "cohort",
                "--workers", args.workers, "--selected-cohort", *flags)
    if "cross-sections" in stages:
        command("run_parse_cross_section_statistics.py", "--source-run-dir", roots["radius"],
                "--output-root", output / "cross_sections", "--report-root",
                roots["statistics"] / "cross_sections", "--run-id", "cohort", "--workers", args.workers)
    if "statistics" in stages:
        from report_parse_area_volumes import generate as area_volumes
        from report_parse_cross_section_bins import generate as area_bins
        from report_parse_radius_volumes import generate as radius_volumes
        from report_parse_radius_percentiles import generate as radius_percentiles
        from report_parse_area_percentiles import generate as area_percentiles
        from plot_parse_radius_case import generate as radius_case_figures
        from plot_parse_area_bins import generate as area_count_figures

        radius_publication = _publication(roots["radius"])
        radius_percentiles(radius_publication, roots["statistics"] / "radius_percentiles")
        radius_case_figures(radius_publication, roots["statistics"] / "radius_case_figures", None)
        area_result = area_bins(roots["sections"], roots["statistics"] / "area_bins")
        area_count_figures(Path(area_result["sources_path"]).parent, roots["statistics"] / "area_count_figures")
        volume_result = area_volumes(roots["sections"], roots["statistics"] / "area_volumes")
        area_publication = Path(volume_result["sources_path"]).parent
        area_percentiles(area_publication, roots["statistics"] / "area_percentiles")
        radius_volumes(area_publication, roots["statistics"] / "radius_volumes")
        command("plot_parse_distributions.py", "--metrics", roots["measurements"] / "pipeline_metrics.csv",
                "--output-root", roots["statistics"] / "metrics", "--dataset-name", "PARSE")
    return {"cases": [case.name for case in selected], "stages": list(stages),
            "paths": {key: str(value) for key, value in roots.items()}}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("data/parse/study"))
    parser.add_argument("--case", dest="cases", action="append", help="Repeat to select cases; default: all.")
    parser.add_argument("--device", default="gpu")
    parser.add_argument("--fiji", type=Path)
    parser.add_argument("--workers", type=int, default=2, help="Concurrent CPU radius/cross-section cases.")
    parser.add_argument("--stage", choices=("all", *STAGES), default="all",
                        help="all computes every measurement and statistics stage.")
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("--workers must be positive")
    return args


def main(argv=None) -> int:
    try:
        result = run(parse_args(argv))
    except (ValueError, OSError, KeyError, subprocess.CalledProcessError) as error:
        print(f"PARSE reproduction failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
