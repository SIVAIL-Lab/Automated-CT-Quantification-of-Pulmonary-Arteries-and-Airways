"""Render PARSE class fractions of estimated valid-segment vessel volume."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from area_bin_charts import _render_parameters, _save
from parse_cross_section_bins import BIN_IDS
from parse_radius_statistics import _histogram
from plot_parse_distributions import COLORS, FIGURE_SIZE, PNG_DPI, STYLE

METHOD = "segment_mean_area_times_measured_length"


def _number(value, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite nonnegative number")
    try:
        number = float(value)
    except (OverflowError, ValueError):
        raise ValueError(f"{name} must be a finite nonnegative number") from None
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"{name} must be a finite nonnegative number")
    return number


def _bin_parameters(bin_key: str, bin_ids, tick_labels=None) -> tuple[tuple, tuple]:
    if not isinstance(bin_key, str) or not bin_key:
        raise ValueError("Volume chart bin keys must be nonempty strings")
    if not isinstance(bin_ids, (tuple, list)) or not bin_ids:
        raise ValueError("Volume charts require nonempty bin identities")
    identities = tuple(bin_ids)
    if (any(not isinstance(key, str) or not key for key in identities)
            or len(set(identities)) != len(identities)):
        raise ValueError("Volume chart bin identities must be nonempty and unique")
    if tick_labels is not None and not isinstance(tick_labels, (tuple, list)):
        raise ValueError("Volume chart tick labels must match the bin identities")
    labels = identities if tick_labels is None else tuple(tick_labels)
    if (len(labels) != len(identities)
            or any(not isinstance(label, str) or not label for label in labels)):
        raise ValueError("Volume chart tick labels must match the bin identities")
    return identities, labels


def _validated_record(record: dict, *, bin_key: str = "area_bin", bin_ids=BIN_IDS) -> dict:
    """Require all configured classes to conserve the valid-segment estimate."""
    bin_ids, _ = _bin_parameters(bin_key, bin_ids)
    kind = "Area" if bin_key == "area_bin" else "Radius" if bin_key == "radius_bin" else "Class"
    if not isinstance(record, dict):
        raise ValueError("Volume chart records must be dictionaries")
    case = record.get("case")
    if not isinstance(case, str) or not case.strip():
        raise ValueError("Volume chart records need a case ID")
    total = _number(record.get("total_volume_mm3"), "total_volume_mm3")
    if total <= 0:
        raise ValueError("Volume charts require a positive total vessel volume")
    bins = record.get("bins")
    if (not isinstance(bins, list) or len(bins) != len(bin_ids)
            or any(not isinstance(row, dict)
                   or not isinstance(row.get(bin_key), str) for row in bins)
            or {row[bin_key] for row in bins} != set(bin_ids)):
        raise ValueError(f"Volume charts require each {kind.lower()} class exactly once")
    ordered = []
    tolerance = total * 1e-12
    for key in bin_ids:
        row = next(row for row in bins if row[bin_key] == key)
        volume = _number(row.get("volume_mm3"), "volume_mm3")
        denominator = _number(row.get("total_volume_mm3"), "total_volume_mm3")
        fraction = _number(row.get("volume_fraction"), "volume_fraction")
        if (denominator <= 0 or not math.isclose(denominator, total, rel_tol=1e-12, abs_tol=0.)
                or volume > total + tolerance or fraction > 1
                or not math.isclose(fraction, volume / total, rel_tol=1e-12, abs_tol=1e-12)):
            raise ValueError("Volume fractions must use the total vessel volume")
        ordered.append(dict(row))
    try:
        classified = math.fsum(row["volume_mm3"] for row in ordered)
    except OverflowError:
        raise ValueError(f"{kind}-class volumes exceed the total vessel volume") from None
    if (not math.isclose(classified, total, rel_tol=1e-12, abs_tol=0.)
            or not math.isclose(math.fsum(row["volume_fraction"] for row in ordered),
                                1., rel_tol=0., abs_tol=1e-12)):
        raise ValueError(f"{kind}-class volumes do not conserve the total estimated vessel volume")
    if "unclassified_volume_mm3" in record:
        unclassified = _number(record["unclassified_volume_mm3"], "unclassified_volume_mm3")
        if unclassified != 0:
            raise ValueError("The estimated volume denominator must exclude unclassified segments")
    return {**record, "bins": ordered}


def render_case(record: dict, directory: Path, *, bin_key: str = "area_bin", bin_ids=BIN_IDS,
                tick_labels=None, x_label: str = "Area class",
                stem: str = "area_bin_volume_fractions", method: str = METHOD) -> list[dict]:
    """Write a bar chart relative to this patient's valid-segment volume estimate."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bin_ids, tick_labels = _bin_parameters(bin_key, bin_ids, tick_labels)
    record = _validated_record(record, bin_key=bin_key, bin_ids=bin_ids)
    directory.mkdir(parents=True, exist_ok=True)
    values = [row["volume_fraction"] for row in record["bins"]]
    plot_values = [100 * value for value in values]
    with plt.rc_context(STYLE):
        figure, axis = plt.subplots(figsize=FIGURE_SIZE, dpi=PNG_DPI)
        try:
            axis.bar(range(len(bin_ids)), plot_values, width=.85,
                     color=COLORS["arterial"], edgecolor="white", linewidth=.45)
            axis.set_xticks(range(len(bin_ids)), tick_labels)
            axis.set_xlim(-.55, len(bin_ids) - .45)
            axis.set_ylim(0, 100)
            axis.set_yticks(range(0, 101, 20))
            axis.set_xlabel(x_label, labelpad=3)
            axis.set_ylabel("Estimated volume fraction (%)", labelpad=3)
            axis.grid(False)
            files = _save(figure, directory, stem)
            return [{
                "scope": "case", "case": record["case"], "metric": "volume_fraction",
                "method": method,
                "total_volume_mm3": record["total_volume_mm3"],
                "volumes_mm3": [row["volume_mm3"] for row in record["bins"]],
                f"{bin_key}s": list(bin_ids), "values": values, "plot_values": plot_values,
                "unit": "percent", "y_axis": axis.get_ylabel(),
                "render_parameters": {**_render_parameters(axis),
                                      "x_tick_labels": list(tick_labels), "bar_width": .85},
                **files,
            }]
        finally:
            plt.close(figure)


def render_cohort(records: list[dict], directory: Path, *, bin_key: str = "area_bin", bin_ids=BIN_IDS,
                  tick_labels=None, x_label: str = "Area class",
                  stem: str = "area_bin_volume_fractions", method: str = METHOD
                  ) -> tuple[list[dict], list[dict]]:
    """Write one histogram per class, weighting each patient equally.

    The case-chart stem supplies the measure suffix after ``_bin_``; each
    histogram uses its own stable bin identity as the filename prefix.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator, StrMethodFormatter

    bin_ids, tick_labels = _bin_parameters(bin_key, bin_ids, tick_labels)
    suffix = stem.split("_bin_", 1)[-1].removesuffix("s")
    records = [_validated_record(record, bin_key=bin_key, bin_ids=bin_ids) for record in records]
    cases = [record["case"] for record in records]
    if len(cases) != len(set(cases)):
        raise ValueError("Cohort volume charts cannot contain duplicate cases")
    if not records:
        return [], []
    directory.mkdir(parents=True, exist_ok=True)
    figures, histogram_rows = [], []
    with plt.rc_context(STYLE):
        for bin_index, key in enumerate(bin_ids):
            values = [record["bins"][bin_index]["volume_fraction"] for record in records]
            histogram = _histogram(values, counts=False)
            edges = np.asarray(histogram["edges"])
            frequencies = np.asarray(histogram["frequencies"])
            figure, axis = plt.subplots(figsize=FIGURE_SIZE, dpi=PNG_DPI)
            try:
                axis.bar(edges[:-1], frequencies, width=np.diff(edges), align="edge",
                         color=COLORS["arterial"], edgecolor="white", linewidth=.45)
                label = tick_labels[bin_index]
                if x_label == "Radius (mm)":
                    label = f"{label} mm"
                elif x_label != "Area class":
                    label = f"{x_label}, {label}"
                axis.set_xlabel(f"Estimated volume fraction (%), {label}", labelpad=3)
                axis.set_ylabel("Number of cases", labelpad=3)
                axis.set_xlim(edges[0], edges[-1])
                axis.set_ylim(0, max(1, int(frequencies.max())) * 1.04)
                axis.xaxis.set_major_locator(MaxNLocator(nbins=4))
                axis.xaxis.set_major_formatter(StrMethodFormatter("{x:g}"))
                axis.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=5, min_n_ticks=2))
                axis.grid(False)
                files = _save(figure, directory, f"{key}_{suffix}")
                figures.append({
                    "scope": "cohort", bin_key: key, "metric": "volume_fraction",
                    "method": method,
                    "included_cases": list(cases), "values": values,
                    "plot_values": [100 * value for value in values],
                    "volumes_mm3": [record["bins"][bin_index]["volume_mm3"] for record in records],
                    "total_volumes_mm3": [record["total_volume_mm3"] for record in records],
                    "y_axis": axis.get_ylabel(), "render_parameters": _render_parameters(axis),
                    **histogram, **files,
                })
            finally:
                plt.close(figure)
            for index, frequency in enumerate(frequencies):
                histogram_rows.append({
                    bin_key: key, "metric": "volume_fraction", "unit": histogram["unit"],
                    "bin_index": index, "lower_bound": float(edges[index]),
                    "upper_bound": float(edges[index + 1]),
                    "upper_inclusive": index == len(frequencies) - 1,
                    "case_count": int(frequency), "included_case_count": len(cases),
                })
    return figures, histogram_rows
