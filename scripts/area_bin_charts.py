"""Render case and patient-distribution charts for classified PARSE segment areas."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from parse_cross_section_bins import BIN_IDS
from parse_radius_statistics import _histogram
from plot_parse_distributions import COLORS, FIGURE_SIZE, PNG_DPI, STYLE, _tight_layout


def _validated_record(record: dict) -> dict:
    """Require the three complete classes and their valid-area denominator."""
    case = record.get("case")
    total = record.get("classified_segment_count")
    if not isinstance(case, str) or not case:
        raise ValueError("Area chart records need a case ID")
    if isinstance(total, bool) or not isinstance(total, int) or total <= 0:
        raise ValueError("Area charts require a positive classified segment count")
    bins = record.get("bins")
    if (not isinstance(bins, list) or len(bins) != len(BIN_IDS)
            or any(not isinstance(row, dict) for row in bins)
            or {row.get("area_bin") for row in bins} != set(BIN_IDS)):
        raise ValueError("Area charts require each area class exactly once")
    ordered = []
    for key in BIN_IDS:
        row = next(row for row in bins if row["area_bin"] == key)
        count, fraction = row.get("segment_count"), row.get("fraction")
        if (isinstance(count, bool) or not isinstance(count, int) or count < 0
                or row.get("classified_segment_count") != total
                or isinstance(fraction, bool) or not isinstance(fraction, (int, float))
                or not math.isfinite(fraction)
                or not math.isclose(fraction, count / total, rel_tol=1e-12, abs_tol=1e-12)):
            raise ValueError("Area counts and fractions must use the classified segment count")
        ordered.append(dict(row))
    if sum(row["segment_count"] for row in ordered) != total:
        raise ValueError("Area classes do not conserve the classified segment count")
    return {**record, "bins": ordered}


def _render_parameters(axis) -> dict:
    lower, upper = axis.get_ylim()
    return {
        "style": "academic-v2", "figure_size_inches": list(FIGURE_SIZE),
        "png_dpi": PNG_DPI, "render_size": [round(value * PNG_DPI) for value in FIGURE_SIZE],
        "font": STYLE["font.family"], "color": COLORS["arterial"], "background": "white",
        "title": False, "legend": False, "grid": False, "mean_median_markers": False,
        "x_axis": axis.get_xlabel(), "y_axis": axis.get_ylabel(),
        "x_limits": [float(value) for value in axis.get_xlim()],
        "y_limits": [float(lower), float(upper)],
        "y_ticks": [float(tick) for tick in axis.get_yticks() if lower <= tick <= upper],
    }


def _save(figure, directory: Path, stem: str) -> dict:
    _tight_layout(figure)
    for extension in ("png", "pdf"):
        figure.savefig(directory / f"{stem}.{extension}", format=extension, dpi=PNG_DPI,
                       facecolor="white", metadata={"Creator": "PARSE cross-sectional area charts"}
                       if extension == "pdf" else None)
    return {"png": f"{stem}.png", "pdf": f"{stem}.pdf"}


def render_case(record: dict, directory: Path) -> list[dict]:
    """Write class counts and proportions using this patient's measurable segments."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    record = _validated_record(record)
    directory.mkdir(parents=True, exist_ok=True)
    figures = []
    with plt.rc_context(STYLE):
        for metric, stem in (("segment_count", "area_bin_counts"),
                             ("fraction", "area_bin_proportions")):
            counts = metric == "segment_count"
            values = [row[metric] for row in record["bins"]]
            plot_values = values if counts else [100 * value for value in values]
            figure, axis = plt.subplots(figsize=FIGURE_SIZE, dpi=PNG_DPI)
            try:
                axis.bar(range(len(BIN_IDS)), plot_values, width=.85,
                         color=COLORS["arterial"], edgecolor="white", linewidth=.45)
                axis.set_xticks(range(len(BIN_IDS)), BIN_IDS)
                axis.set_xlim(-.55, len(BIN_IDS) - .45)
                if counts:
                    axis.set_ylim(0, max(1, max(plot_values)) * 1.04)
                    axis.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=True, min_n_ticks=2))
                else:
                    axis.set_ylim(0, 100)
                    axis.set_yticks(range(0, 101, 20))
                axis.set_xlabel("Area class", labelpad=3)
                axis.set_ylabel("Number of segments" if counts else "Segment proportion (%)", labelpad=3)
                axis.grid(False)
                files = _save(figure, directory, stem)
                figures.append({
                    "scope": "case", "case": record["case"], "metric": metric,
                    "classified_segment_count": record["classified_segment_count"],
                    "area_bins": list(BIN_IDS), "values": values, "plot_values": plot_values,
                    "unit": "segments" if counts else "percent", "y_axis": axis.get_ylabel(),
                    "render_parameters": {**_render_parameters(axis),
                                          "x_tick_labels": list(BIN_IDS), "bar_width": .85},
                    **files,
                })
            finally:
                plt.close(figure)
    return figures


def render_cohort(records: list[dict], directory: Path) -> tuple[list[dict], list[dict]]:
    """Write six distributions, with one observation per included patient."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator, StrMethodFormatter

    records = [_validated_record(record) for record in records]
    cases = [record["case"] for record in records]
    if len(cases) != len(set(cases)):
        raise ValueError("Cohort charts cannot contain duplicate cases")
    if not records:
        return [], []
    directory.mkdir(parents=True, exist_ok=True)
    figures, histogram_rows = [], []
    with plt.rc_context(STYLE):
        for bin_index, key in enumerate(BIN_IDS):
            for metric in ("segment_count", "fraction"):
                counts = metric == "segment_count"
                values = [record["bins"][bin_index][metric] for record in records]
                histogram = _histogram(values, counts=counts)
                edges = np.asarray(histogram["edges"])
                frequencies = np.asarray(histogram["frequencies"])
                stem = f"{key}_{metric}"
                figure, axis = plt.subplots(figsize=FIGURE_SIZE, dpi=PNG_DPI)
                try:
                    axis.bar(edges[:-1], frequencies, width=np.diff(edges), align="edge",
                             color=COLORS["arterial"], edgecolor="white", linewidth=.45)
                    measure = "Segment count" if counts else "Segment proportion (%)"
                    axis.set_xlabel(f"{measure}, {key}", labelpad=3)
                    axis.set_ylabel("Number of cases", labelpad=3)
                    axis.set_xlim(edges[0], edges[-1])
                    axis.set_ylim(0, max(1, int(frequencies.max())) * 1.04)
                    axis.xaxis.set_major_locator(MaxNLocator(nbins=4, integer=counts))
                    axis.xaxis.set_major_formatter(StrMethodFormatter("{x:g}"))
                    axis.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=5, min_n_ticks=2))
                    axis.grid(False)
                    files = _save(figure, directory, stem)
                    figures.append({
                        "scope": "cohort", "area_bin": key, "metric": metric,
                        "included_cases": list(cases), "values": values,
                        "plot_values": values if counts else [100 * value for value in values],
                        "y_axis": axis.get_ylabel(), "render_parameters": _render_parameters(axis),
                        **histogram, **files,
                    })
                finally:
                    plt.close(figure)
                for index, frequency in enumerate(frequencies):
                    histogram_rows.append({
                        "area_bin": key, "metric": metric, "unit": histogram["unit"],
                        "bin_index": index, "lower_bound": float(edges[index]),
                        "upper_bound": float(edges[index + 1]),
                        "upper_inclusive": index == len(frequencies) - 1,
                        "case_count": int(frequency), "included_case_count": len(cases),
                    })
    return figures, histogram_rows
