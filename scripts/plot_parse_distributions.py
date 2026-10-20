#!/usr/bin/env python3
"""Plot the 18 reported measurements independently from a saved pipeline CSV."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import tempfile

import numpy as np


METRIC_DEFINITIONS = (
    {"metric": "lung_total_volume_l", "label": "Total lung volume",
     "unit": "L", "places": 3, "organ": "Lungs / lobes", "group": "ordinary"},
    {"metric": "lung_upper_lobe_left_volume_ml", "label": "Left upper lobe volume",
     "unit": "mL", "places": 1, "organ": "Lungs / lobes", "group": "ordinary"},
    {"metric": "lung_lower_lobe_left_volume_ml", "label": "Left lower lobe volume",
     "unit": "mL", "places": 1, "organ": "Lungs / lobes", "group": "ordinary"},
    {"metric": "lung_upper_lobe_right_volume_ml", "label": "Right upper lobe volume",
     "unit": "mL", "places": 1, "organ": "Lungs / lobes", "group": "ordinary"},
    {"metric": "lung_middle_lobe_right_volume_ml", "label": "Right middle lobe volume",
     "unit": "mL", "places": 1, "organ": "Lungs / lobes", "group": "ordinary"},
    {"metric": "lung_lower_lobe_right_volume_ml", "label": "Right lower lobe volume",
     "unit": "mL", "places": 1, "organ": "Lungs / lobes", "group": "ordinary"},
    {"metric": "arterial_volume_ml", "label": "Arterial lumen volume",
     "unit": "mL", "places": 3, "organ": "Arteries", "group": "ordinary"},
    {"metric": "arterial_volume_fraction", "label": "Arterial volume / lung volume",
     "unit": "mL/mL", "places": 6, "organ": "Arteries", "group": "composite"},
    {"metric": "total_branch_length_mm", "label": "Arterial total length",
     "unit": "mm", "places": 1, "organ": "Arteries", "group": "ordinary"},
    {"metric": "branch_count", "label": "Arterial branch count",
     "unit": "branches", "places": 0, "organ": "Arteries", "group": "ordinary"},
    {"metric": "arterial_length_m_per_lung_l", "label": "Arterial length / lung volume",
     "unit": "m/L", "places": 3, "organ": "Arteries", "group": "composite"},
    {"metric": "branches_per_lung_l", "label": "Arterial branches / lung volume",
     "unit": "branches/L", "places": 2, "organ": "Arteries", "group": "composite"},
    {"metric": "airway_volume_ml", "label": "Airway lumen volume",
     "unit": "mL", "places": 3, "organ": "Airways", "group": "ordinary"},
    {"metric": "airway_volume_fraction", "label": "Airway volume / lung volume",
     "unit": "mL/mL", "places": 6, "organ": "Airways", "group": "composite"},
    {"metric": "airway_total_branch_length_mm", "label": "Airway total length",
     "unit": "mm", "places": 1, "organ": "Airways", "group": "ordinary"},
    {"metric": "airway_branch_count", "label": "Airway branch count",
     "unit": "branches", "places": 0, "organ": "Airways", "group": "ordinary"},
    {"metric": "airway_length_m_per_lung_l", "label": "Airway length / lung volume",
     "unit": "m/L", "places": 3, "organ": "Airways", "group": "composite"},
    {"metric": "airway_branches_per_lung_l", "label": "Airway branches / lung volume",
     "unit": "branches/L", "places": 2, "organ": "Airways", "group": "composite"},
)
DEFAULT_WIDTHS = {
    "lung_total_volume_l": 0.5,
    "lung_upper_lobe_left_volume_ml": 100.0, "lung_lower_lobe_left_volume_ml": 100.0,
    "lung_upper_lobe_right_volume_ml": 100.0, "lung_middle_lobe_right_volume_ml": 100.0,
    "lung_lower_lobe_right_volume_ml": 100.0,
    "arterial_volume_ml": 25.0, "total_branch_length_mm": 2000.0, "branch_count": 200.0,
    "airway_volume_ml": 10.0, "airway_total_branch_length_mm": 1000.0, "airway_branch_count": 50.0,
    "arterial_volume_fraction": 0.005, "arterial_length_m_per_lung_l": 0.25,
    "branches_per_lung_l": 50.0, "airway_volume_fraction": 0.001,
    "airway_length_m_per_lung_l": 0.25, "airway_branches_per_lung_l": 20.0,
}
NORMALIZED = {
    "arterial_volume_fraction": ("arterial_volume_ml", "lung_total_volume_ml", 1.0),
    "arterial_length_m_per_lung_l": ("total_branch_length_mm", "lung_total_volume_l", 0.001),
    "branches_per_lung_l": ("branch_count", "lung_total_volume_l", 1.0),
    "airway_volume_fraction": ("airway_volume_ml", "lung_total_volume_ml", 1.0),
    "airway_length_m_per_lung_l": ("airway_total_branch_length_mm", "lung_total_volume_l", 0.001),
    "airway_branches_per_lung_l": ("airway_branch_count", "lung_total_volume_l", 1.0),
}
COUNT_METRICS = {"branch_count", "airway_branch_count"}
SUCCESS = {"ok", "skipped"}
FIGURE_SIZE = (3.5, 2.8)
PNG_DPI = 600
COLORS = {"lung": "#76A69A", "arterial": "#7299BA", "airway": "#C59673"}
STYLE = {
    "font.family": "DejaVu Sans", "font.size": 8, "axes.unicode_minus": False,
    "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42,
    "axes.edgecolor": "#303030", "axes.labelcolor": "#303030", "text.color": "#303030",
    "axes.linewidth": 0.6, "axes.labelsize": 9, "axes.grid": False,
    "xtick.color": "#303030", "ytick.color": "#303030",
    "xtick.direction": "out", "ytick.direction": "out",
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 3, "ytick.major.size": 3,
    "xtick.major.pad": 3, "ytick.major.pad": 3,
    "figure.facecolor": "white", "axes.facecolor": "white",
}


def _render_parameters() -> dict:
    size = FIGURE_SIZE
    return {"style_version": "academic-v2", "figure_size_inches": list(size), "dpi": PNG_DPI,
            "render_size": [round(value * PNG_DPI) for value in size],
            "font": "DejaVu Sans", "background": "white"}


def _color(metric: str) -> str:
    group = "lung" if metric.startswith("lung_") else "airway" if metric.startswith("airway_") else "arterial"
    return COLORS[group]


def _wrap_xlabel(axis) -> None:
    """Fit the full measurement label to the fixed axes width, without cropping."""
    label = axis.xaxis.label
    words = label.get_text().split()
    renderer = axis.figure.canvas.get_renderer()
    available = axis.get_window_extent(renderer).width
    lines, line = [], ""
    for word in words:
        candidate = f"{line} {word}".strip()
        width = renderer.get_text_width_height_descent(candidate, label.get_fontproperties(), False)[0]
        if line and width > available:
            lines.append(line)
            line = word
        else:
            line = candidate
    if line:
        lines.append(line)
    label.set_text("\n".join(lines))


def _tight_layout(figure) -> None:
    """Keep a fixed page size while fitting margins to the actual axis labels."""
    for _ in range(2):
        for axis in figure.axes:
            _wrap_xlabel(axis)
        figure.tight_layout(pad=0.45, w_pad=1.2)


def _definitions() -> list[dict]:
    return [dict(definition) for definition in METRIC_DEFINITIONS]


def _stages(metric: str) -> tuple[str, ...]:
    if metric in NORMALIZED:
        return (*_stages(NORMALIZED[metric][0]), "lung_volume_status")
    if metric.startswith("lung_"):
        return ("lung_volume_status",)
    if metric == "arterial_volume_ml":
        return ("arterial_volume_status",)
    if metric == "airway_volume_ml":
        return ("airway_volume_status",)
    if metric.startswith("airway_"):
        return ("airway_analyzeskeleton_status",)
    return ("analyzeskeleton_status",)


def _number(row: dict, key: str) -> float | None:
    raw = row.get(key)
    if raw is None or not raw.strip():
        return None
    try:
        value = float(raw)
    except (ValueError, TypeError) as error:
        raise ValueError(f"{row['case']}: {key} is not numeric") from error
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{row['case']}: {key} must be finite and nonnegative")
    if key in COUNT_METRICS and not value.is_integer():
        raise ValueError(f"{row['case']}: {key} must be an integer")
    return value


def _load_metrics(path: Path, definitions: list[dict] | None = None, *,
                  measurement_root: Path | None = None) -> dict:
    """Read once, validate before writes, and retain independent metric cohorts."""
    definitions = definitions if definitions is not None else _definitions()
    from measurement_scope import scope_for_root
    root = Path(measurement_root).resolve() if measurement_root is not None else path.resolve().parent
    scope = scope_for_root(root)
    source_hashes = {}
    profile_path = root / "measurement_scope.json"
    if scope == "largest_26":
        if path.resolve() != root / "pipeline_metrics.csv":
            raise ValueError("Scoped histogram input must be the profile result root pipeline_metrics.csv")
        source_hashes[str(profile_path)] = hashlib.sha256(profile_path.read_bytes()).hexdigest()
    raw = path.read_bytes()
    reader = csv.DictReader(io.StringIO(raw.decode("utf-8-sig")))
    keys = [definition["metric"] for definition in definitions]
    required = {"case", "status", *keys}
    required.update(stage for key in keys for stage in _stages(key))
    required.update(("lung_total_volume_ml", "normalization_status"))
    if not required.issubset(reader.fieldnames or []):
        raise ValueError(f"Missing CSV columns: {sorted(required - set(reader.fieldnames or []))}")
    cohorts = {key: {"values": [], "included_cases": [], "excluded_cases": []} for key in keys}
    seen = set()
    for row in reader:
        if (row.get("measurement_scope") or "all") != scope:
            raise ValueError("Pipeline CSV contains mixed or incompatible measurement scopes")
        case = row["case"]
        if not case or case in seen:
            raise ValueError(f"Missing or duplicate case ID: {case!r}")
        seen.add(case)
        numeric = {}

        def number(key):
            if key not in numeric:
                numeric[key] = _number(row, key)
            return numeric[key]

        # mL/L consistency is shared by lung volume and all six normalizations.
        lung_ready = row["lung_volume_status"] in SUCCESS
        if lung_ready:
            ml, liters = number("lung_total_volume_ml"), number("lung_total_volume_l")
            if ml is None or liters is None:
                raise ValueError(f"{case}: successful lung stage has missing volume")
            if not math.isclose(ml / 1000, liters, rel_tol=1e-9, abs_tol=1e-12):
                raise ValueError(f"{case}: lung_total_volume_ml and lung_total_volume_l disagree")
        for key in keys:
            cohort = cohorts[key]
            stages = _stages(key)
            bad = {stage: row[stage] for stage in stages if row[stage] not in SUCCESS}
            reason = f"Unavailable stages: {bad}" if bad else None
            if not reason:
                value = number(key)
                if value is None:
                    if key in NORMALIZED and row["normalization_status"] not in SUCCESS:
                        reason = "normalization_unavailable"
                    else:
                        raise ValueError(f"{case}: successful stage has missing {key}")
                elif key in NORMALIZED:
                    numerator_key, denominator_key, scale = NORMALIZED[key]
                    numerator, denominator = number(numerator_key), number(denominator_key)
                    if numerator is None or denominator is None:
                        reason = "Missing normalization source"
                    elif denominator <= 0:
                        raise ValueError(f"{case}: {key} requires a positive lung denominator")
                    elif not math.isclose(value, numerator * scale / denominator, rel_tol=1e-8, abs_tol=1e-10):
                        raise ValueError(f"{case}: {key} disagrees with {numerator_key} * {scale:g} / {denominator_key}")
            if reason:
                cohort["excluded_cases"].append({"case": case, "reason": reason, "status": row["status"]})
            else:
                cohort["included_cases"].append(case)
                cohort["values"].append(value)
    for cohort in cohorts.values():
        cohort["values"] = np.asarray(cohort["values"], dtype=float)
    digest = hashlib.sha256(raw).hexdigest()
    return {"cohorts": cohorts, "case_count": len(seen), "input_sha256": digest,
            "input_csv": str(path.resolve()), "measurement_scope": scope,
            "source_sha256": {**source_hashes, str(path.resolve()): digest}}


def _bin_edges(values: np.ndarray, width: float) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if not math.isfinite(width) or width <= 0:
        raise ValueError("Bin width must be finite and positive")
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("Bins require nonempty finite values")
    first = math.floor(float(values.min()) / width)
    last = max(first + 1, math.ceil(float(values.max()) / width))
    edges = np.arange(first, last + 1, dtype=float) * width
    if edges[0] > values.min():
        edges = np.insert(edges, 0, (first - 1) * width)
    if edges[-1] < values.max():
        edges = np.append(edges, (last + 1) * width)
    return edges


def _summarize(loaded: dict, definitions: list[dict], widths: dict[str, float]) -> tuple[list[dict], list[dict]]:
    summaries, bins = [], []
    for definition in definitions:
        key = definition["metric"]
        cohort = loaded["cohorts"][key]
        data, width = cohort["values"], widths[key]
        summary = {**definition, "status": "ok" if len(data) else "not_generated", "n": len(data),
                   "excluded_count": len(cohort["excluded_cases"]), "bin_width": width,
                   "included_cases": cohort["included_cases"], "excluded_cases": cohort["excluded_cases"],
                   "mean": None, "median": None, "min": None, "max": None, "bin_edges": [], "counts": []}
        summaries.append(summary)
        if not len(data):
            continue
        edges = _bin_edges(data, width)
        counts, _ = np.histogram(data, bins=edges)
        if int(counts.sum()) != len(data):
            raise ValueError(f"{key}: bins do not cover every case")
        summary.update(mean=float(data.mean()), median=float(np.median(data)), min=float(data.min()),
                       max=float(data.max()), bin_edges=edges.tolist(), counts=counts.tolist())
        for index, count in enumerate(counts):
            final = index == len(counts) - 1
            lower, upper = float(edges[index]), float(edges[index + 1])
            bins.append({"metric": key, "unit": definition["unit"], "bin_width": width, "n": len(data),
                         "bin_index": index + 1, "interval": f"[{lower:g}, {upper:g}{']' if final else ')'}",
                         "lower_bound": lower, "upper_bound": upper, "upper_inclusive": final,
                         "case_count": int(count), "percentage": float(100 * count / len(data))})
    return summaries, bins


def _draw_histogram(axis, summary: dict, color: str) -> None:
    from matplotlib import ticker

    edges, counts = np.asarray(summary["bin_edges"]), np.asarray(summary["counts"])
    axis.bar(edges[:-1], counts, width=np.diff(edges), align="edge", color=color,
             edgecolor="white", linewidth=0.45)
    axis.set_xlabel(f"{summary['label']} ({summary['unit']})", labelpad=3)
    axis.set_ylabel("Number of cases", labelpad=3)
    max_ticks = 5
    if len(edges) <= max_ticks:
        axis.set_xticks(edges)
    else:
        axis.xaxis.set_major_locator(ticker.MaxNLocator(nbins=max_ticks - 1, integer=summary["metric"] in COUNT_METRICS))
    axis.xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:g}"))
    axis.set(xlim=(edges[0], edges[-1]), ylim=(0, max(1, float(counts.max())) * 1.04))
    axis.yaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=5, min_n_ticks=2))
    axis.grid(False)


def _save_figure(figure, output: Path, stem: str, *, dpi: int = PNG_DPI) -> None:
    for extension in ("png", "pdf"):
        path = output / f"{stem}.{extension}"
        with tempfile.NamedTemporaryFile(dir=output, suffix=f".{extension}", delete=False) as handle:
            temporary = Path(handle.name)
        try:
            figure.savefig(temporary, format=extension, dpi=dpi, facecolor="white")
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def _plot(summaries: list[dict], output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with plt.rc_context(STYLE):
        for summary in summaries:
            key = summary["metric"]
            if summary["status"] != "ok":
                for extension in ("png", "pdf"):
                    (output / f"{key}.{extension}").unlink(missing_ok=True)
                continue
            figure, axis = plt.subplots(figsize=FIGURE_SIZE, dpi=PNG_DPI)
            try:
                _draw_histogram(axis, summary, _color(key))
                _tight_layout(figure)
                _save_figure(figure, output, key)
            finally:
                plt.close(figure)


def _write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with tempfile.NamedTemporaryFile("w", dir=path.parent, suffix=".csv", newline="", encoding="utf-8", delete=False) as handle:
        temporary = Path(handle.name)
        try:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, default=Path("data/parse/results/pipeline_metrics.csv"))
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--measurement-root", type=Path)
    parser.add_argument("--measurement-scope", choices=("all", "largest_26"), default=None)
    parser.add_argument("--dataset-name", default="PARSE")
    parser.add_argument("--branch-bin-width", type=float, default=200)
    parser.add_argument("--volume-bin-width", type=float, default=0.5, help="CT lung volume bin width in liters")
    parser.add_argument("--normalized-bin-width", type=float, default=50, help="Arterial branches per liter bin width")
    parser.add_argument("--bin-width", action="append", default=[], metavar="METRIC=VALUE",
                        help="Repeat to override any selected metric's default width")
    args = parser.parse_args(argv)
    if not args.dataset_name.strip():
        parser.error("--dataset-name must not be empty")
    args.output_root = args.output_root or Path("data/parse/statistics/metrics")
    args.widths = {**DEFAULT_WIDTHS, "branch_count": args.branch_bin_width,
                   "lung_total_volume_l": args.volume_bin_width, "branches_per_lung_l": args.normalized_bin_width}
    selected = {definition["metric"] for definition in _definitions()}
    for override in args.bin_width:
        try:
            key, raw = override.split("=", 1)
            if key not in selected:
                raise ValueError(f"Unknown selected metric: {key}")
            args.widths[key] = float(raw)
        except ValueError as error:
            parser.error(f"Invalid --bin-width {override!r}: {error}")
    if any(not math.isfinite(width) or width <= 0 for width in args.widths.values()):
        parser.error("All bin widths must be finite and positive")
    return args


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    definitions = _definitions()
    loaded = _load_metrics(args.metrics, definitions,
                           measurement_root=args.measurement_root)
    if args.measurement_scope is not None and args.measurement_scope != loaded["measurement_scope"]:
        raise ValueError("Requested histogram scope differs from the result profile")
    if loaded["measurement_scope"] == "largest_26" and args.dataset_name != "PARSE":
        raise ValueError("Largest-component profiles are only available for PARSE")
    summaries, bins = _summarize(loaded, definitions, args.widths)
    # Everything above is read-only: malformed input never replaces valid plots.
    output = args.output_root.expanduser().resolve()
    owned = ["histogram_bins.csv", "histogram_statistics.csv", "histogram_report.json"]
    stems = [definition["metric"] for definition in definitions]
    owned.extend(f"{stem}.{suffix}" for stem in stems for suffix in ("png", "pdf"))
    if any((output / name).resolve() == args.metrics.resolve() for name in owned):
        raise ValueError("Histogram outputs must not overwrite the source CSV")
    output.mkdir(parents=True, exist_ok=True)
    _plot(summaries, output)
    if loaded["measurement_scope"] == "largest_26":
        for source, digest in loaded["source_sha256"].items():
            if hashlib.sha256(Path(source).read_bytes()).hexdigest() != digest:
                raise ValueError(f"Histogram source changed while rendering: {source}")
    common = {"dataset": args.dataset_name, "input_csv": loaded["input_csv"], "input_sha256": loaded["input_sha256"],
              "measurement_scope": loaded["measurement_scope"]}
    bin_fields = ["metric", "unit", "bin_width", "n", "bin_index", "interval", "lower_bound", "upper_bound",
                  "upper_inclusive", "case_count", "percentage", *common]
    _write_csv(output / "histogram_bins.csv", [{**row, **common} for row in bins], bin_fields)
    summary_rows = [{**summary, **common, "included_cases": json.dumps(summary["included_cases"]),
                     "excluded_cases": json.dumps(summary["excluded_cases"])} for summary in summaries]
    summary_fields = ["metric", "label", "unit", "group", "status", "n", "excluded_count", "bin_width",
                      "mean", "median", "min", "max", "included_cases", "excluded_cases", *common]
    _write_csv(output / "histogram_statistics.csv", summary_rows, summary_fields)
    report = {
        "schema_version": 2, **common, "case_count": loaded["case_count"], "statistics": summaries,
        "parameters": {"legacy_three_panel": False,
                       "bin_widths": {definition["metric"]: args.widths[definition["metric"]] for definition in definitions},
                       **_render_parameters()},
        "interval_convention": "[lower, upper), except the final bin includes upper",
        "eligibility": "Each measurement uses its own ok/skipped prerequisite stages; overall case status is not a filter",
        "source_sha256": loaded["source_sha256"],
        "branch_definition": ("AnalyzeSkeleton edge count from the largest 26-connected mask component"
                              if loaded["measurement_scope"] == "largest_26"
                              else "AnalyzeSkeleton edge count across all connected components"),
        "volume_definition": ("Selected artery/airway mask voxel count times calibrated voxel volume; lung/lobe masks and their union are unchanged"
                              if loaded["measurement_scope"] == "largest_26"
                              else "Saved mask voxel count times calibrated voxel volume; lung volume is the five-lobe union"),
        "airway_definition": ("Largest 26-connected airway lumen component, including the trachea as segmented"
                              if loaded["measurement_scope"] == "largest_26"
                              else "Complete saved airway lumen, including the trachea as segmented"),
    }
    with tempfile.NamedTemporaryFile("w", dir=output, suffix=".json", encoding="utf-8", delete=False) as handle:
        temporary = Path(handle.name)
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")
    temporary.replace(output / "histogram_report.json")
    missing = [summary["metric"] for summary in summaries if summary["status"] != "ok"]
    print(f"{args.dataset_name}: generated {len(summaries) - len(missing)}/{len(summaries)} metric distributions from {loaded['case_count']} cases; output: {output}")
    return int(bool(missing))


if __name__ == "__main__":
    raise SystemExit(main())
