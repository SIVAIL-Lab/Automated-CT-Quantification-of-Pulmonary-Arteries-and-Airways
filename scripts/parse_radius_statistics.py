"""Group predicted EDT graph segments and publish PARSE cohort distributions.

A segment radius is its physical-length-weighted mean EDT radius. Grouping uses
unrounded millimetres and never mixes the EDT graph denominator with Fiji branch
counts. Clinical inputs and centerline generation are managed by the batch CLI.
"""
from __future__ import annotations

from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import platform

import numpy as np

from plot_parse_distributions import COLORS, FIGURE_SIZE, PNG_DPI, STYLE, _tight_layout


BIN_DEFINITIONS = (
    {"radius_bin": "radius_0_2", "label": "0-2 mm", "lower_mm": 0., "upper_mm": 2., "interval": "(0, 2]"},
    {"radius_bin": "radius_2_4", "label": "2-4 mm", "lower_mm": 2., "upper_mm": 4., "interval": "(2, 4]"},
    {"radius_bin": "radius_4_6", "label": "4-6 mm", "lower_mm": 4., "upper_mm": 6., "interval": "(4, 6]"},
    {"radius_bin": "radius_gt_6", "label": ">6 mm", "lower_mm": 6., "upper_mm": None, "interval": "(6, infinity)"},
)
BIN_IDS = tuple(row["radius_bin"] for row in BIN_DEFINITIONS)
BIN_FIELDS = ["radius_bin", "label", "segment_count", "total_segment_count", "fraction"]


def _integer(value, name: str, *, positive: bool = False) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < int(positive):
        raise ValueError(f"{name} must be {'positive' if positive else 'nonnegative'}")
    return value


def _finite(value, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a finite number")
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a finite number") from None
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite number")
    return number


def classify_segments(segments: list[dict]) -> dict:
    """Return copied segment rows and all four bins, rejecting invalid radii."""
    if not isinstance(segments, list) or not segments:
        raise ValueError("A nonempty segment table is required")
    rows, seen = [], set()
    counts = dict.fromkeys(BIN_IDS, 0)
    for segment in segments:
        branch = _integer(segment.get("branch_id"), "branch_id", positive=True)
        if branch in seen:
            raise ValueError(f"Duplicate branch_id: {branch}")
        seen.add(branch)
        radius = _finite(segment.get("segment_radius_mm"), "segment_radius_mm")
        if radius <= 0:
            raise ValueError("segment_radius_mm must be positive")
        if "r_edt_length_weighted_mean_mm" in segment:
            alias = _finite(segment["r_edt_length_weighted_mean_mm"], "r_edt_length_weighted_mean_mm")
            if alias != radius:
                raise ValueError("segment_radius_mm differs from its length-weighted EDT mean")
        index = 0 if radius <= 2 else 1 if radius <= 4 else 2 if radius <= 6 else 3
        key = BIN_IDS[index]
        counts[key] += 1
        rows.append({**segment, "radius_bin": key})
    total = len(rows)
    bins = [{"radius_bin": definition["radius_bin"], "label": definition["label"],
             "segment_count": counts[definition["radius_bin"]], "total_segment_count": total,
             "fraction": counts[definition["radius_bin"]] / total}
            for definition in BIN_DEFINITIONS]
    return {"segments": rows, "bins": bins}


def _validated_bins(record: dict) -> list[dict]:
    case = record["case"]
    total = _integer(record.get("summary", {}).get("segment_count"),
                     f"{case}: summary.segment_count", positive=True)
    bins = record.get("bins")
    if not isinstance(bins, list) or len(bins) != 4:
        raise ValueError(f"{case}: all four radius bins are required")
    by_key = {row.get("radius_bin"): row for row in bins}
    if set(by_key) != set(BIN_IDS):
        raise ValueError(f"{case}: radius bin identities are missing, repeated, or unknown")
    checked = []
    for definition in BIN_DEFINITIONS:
        row = by_key[definition["radius_bin"]]
        count = _integer(row.get("segment_count"), f"{case}: segment_count")
        denominator = _integer(row.get("total_segment_count"), f"{case}: total_segment_count", positive=True)
        fraction = _finite(row.get("fraction"), f"{case}: fraction")
        if denominator != total or count > total:
            raise ValueError(f"{case}: radius bin denominator disagrees with the tree segment count")
        if not math.isclose(fraction, count / total, rel_tol=1e-12, abs_tol=1e-15):
            raise ValueError(f"{case}: radius bin fraction disagrees with its count")
        checked.append({"case": case, "radius_bin": definition["radius_bin"], "label": definition["label"],
                        "segment_count": count, "total_segment_count": total, "fraction": fraction})
    if sum(row["segment_count"] for row in checked) != total:
        raise ValueError(f"{case}: radius bin counts do not sum to the tree segment count")
    if not math.isclose(math.fsum(row["fraction"] for row in checked), 1., rel_tol=0, abs_tol=1e-12):
        raise ValueError(f"{case}: radius bin fractions do not sum to one")
    return checked


def aggregate_cases(records: list[dict], requested_cases: list[str]) -> dict:
    """Aggregate each successful case once; unavailable cases are never zeros.

    Descriptive standard deviation uses ddof=1 (sample standard deviation), and
    is unavailable for fewer than two cases. Quantiles use linear interpolation.
    A complete selected cohort does not imply a complete 100-case PARSE cohort.
    """
    if (not requested_cases or any(not isinstance(case, str) or not case for case in requested_cases)
            or len(set(requested_cases)) != len(requested_cases)):
        raise ValueError("Requested case IDs must be nonempty and unique")
    requested = set(requested_cases)
    by_case = {}
    for record in records:
        case = record.get("case")
        if case not in requested or case in by_case:
            raise ValueError(f"Unrequested or duplicate case record: {case!r}")
        by_case[case] = record
    case_bins, exclusions, included = [], [], []
    for case in requested_cases:
        record = by_case.get(case)
        if record is None:
            exclusions.append({"case": case, "status": "missing", "reason": "No result record available"})
        elif record.get("status") != "ok":
            exclusions.append({"case": case, "status": record.get("status", "missing"),
                               "reason": str(record.get("error") or "Computation did not complete successfully")})
        else:
            case_bins.extend(_validated_bins(record))
            included.append(case)
    total = sum(row["total_segment_count"] for row in case_bins if row["radius_bin"] == BIN_IDS[0])
    pooled, descriptives = [], []
    for definition in BIN_DEFINITIONS:
        key = definition["radius_bin"]
        rows = [row for row in case_bins if row["radius_bin"] == key]
        count = sum(row["segment_count"] for row in rows)
        pooled.append({"radius_bin": key, "label": definition["label"], "case_count": len(rows),
                       "segment_count": count, "total_segment_count": total,
                       "fraction": count / total if total else None})
        for metric in ("segment_count", "fraction"):
            data = np.asarray([row[metric] for row in rows], dtype=float)
            summary = {"radius_bin": key, "label": definition["label"], "metric": metric, "n": len(data),
                       **dict.fromkeys(("mean", "std", "median", "q25", "q75", "min", "max"))}
            if len(data):
                summary.update(mean=float(data.mean()), std=float(data.std(ddof=1)) if len(data) > 1 else None,
                               median=float(np.median(data)), q25=float(np.quantile(data, .25)),
                               q75=float(np.quantile(data, .75)), min=float(data.min()), max=float(data.max()))
            descriptives.append(summary)
    if sum(row["segment_count"] for row in pooled) != total:
        raise ValueError("Pooled radius bin counts do not conserve the segment total")
    return {"status": "partial" if exclusions else "complete", "requested_cases": list(requested_cases),
            "included_cases": included, "requested_case_count": len(requested_cases),
            "included_case_count": len(included), "excluded_case_count": len(exclusions),
            "pooled_segment_count": total, "case_bins": case_bins, "cohort_bins": pooled,
            "descriptives": descriptives, "exclusions": exclusions}


def _histogram(values: list[float], *, counts: bool) -> dict:
    """Use FD widths; counts have integer widths and fractions are percentages."""
    data = np.asarray(values, dtype=float)
    if not len(data) or not np.isfinite(data).all():
        raise ValueError("Histogram data must be nonempty and finite")
    if counts:
        if (data < 0).any() or not np.equal(data, np.floor(data)).all():
            raise ValueError("Histogram segment counts must be nonnegative integers")
    elif (data < 0).any() or (data > 1).any():
        raise ValueError("Histogram fractions must lie between zero and one")
    else:
        data = data * 100
    spread = float(np.ptp(data))
    iqr = float(np.quantile(data, .75) - np.quantile(data, .25))
    width = 2. * iqr * len(data) ** (-1. / 3.)
    if width <= 0:
        width = spread / max(1, math.ceil(math.log2(len(data)) + 1)) if spread else 1.
    if counts:
        width = float(max(1, math.ceil(width)))
        # Center integer observations in integer-width bins, including constants.
        first = math.floor(float(data.min()) / width) * width - .5
    else:
        first = math.floor(float(data.min()) / width) * width
        first = min(first, float(data.min()))
    bin_count = max(1, math.ceil((float(data.max()) - first) / width))
    edges = first + np.arange(bin_count + 1) * width
    if edges[-1] < data.max():
        edges = np.append(edges, edges[-1] + width)
    if not counts:
        edges = np.clip(edges, 0., 100.)
        edges = np.unique(edges)
        if len(edges) < 2:  # A constant 100% cohort still has a visible one-bin plot.
            edges = np.array([99., 100.])
    frequencies, edges = np.histogram(data, bins=edges)
    if int(frequencies.sum()) != len(data):
        raise ValueError("Histogram bins do not cover every included case")
    return {"edges": edges.tolist(), "frequencies": frequencies.tolist(), "fd_width": width,
            "sample_count": len(data), "unit": "segments" if counts else "percent"}


def _plot_histograms(aggregate: dict, assets: Path) -> tuple[list[dict], list[dict]]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import ticker

    figures, histogram_rows = [], []
    if not aggregate["included_cases"]:
        return figures, histogram_rows
    with plt.rc_context(STYLE):
        for definition in BIN_DEFINITIONS:
            key = definition["radius_bin"]
            rows = [row for row in aggregate["case_bins"] if row["radius_bin"] == key]
            for metric in ("segment_count", "fraction"):
                histogram = _histogram([row[metric] for row in rows], counts=metric == "segment_count")
                edges, frequencies = np.asarray(histogram["edges"]), np.asarray(histogram["frequencies"])
                stem = f"{key}_{metric}"
                figure, axis = plt.subplots(figsize=FIGURE_SIZE, dpi=PNG_DPI)
                try:
                    axis.bar(edges[:-1], frequencies, width=np.diff(edges), align="edge",
                             color=COLORS["arterial"], edgecolor="white", linewidth=.45)
                    measure = "Segment count" if metric == "segment_count" else "Segment proportion (%)"
                    axis.set_xlabel(f"{measure}, radius {definition['label']}", labelpad=3)
                    axis.set_ylabel("Number of cases", labelpad=3)
                    axis.set_xlim(edges[0], edges[-1])
                    axis.set_ylim(0, max(1, int(frequencies.max())) * 1.04)
                    axis.xaxis.set_major_locator(ticker.MaxNLocator(nbins=4, integer=metric == "segment_count"))
                    axis.xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:g}"))
                    axis.yaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=5, min_n_ticks=2))
                    axis.grid(False)
                    _tight_layout(figure)
                    for extension in ("png", "pdf"):
                        figure.savefig(assets / f"{stem}.{extension}", format=extension, dpi=PNG_DPI,
                                       facecolor="white", metadata={"Creator": "PARSE EDT radius statistics"}
                                       if extension == "pdf" else None)
                finally:
                    plt.close(figure)
                figures.append({"radius_bin": key, "metric": metric, "png": f"assets/{stem}.png",
                                "pdf": f"assets/{stem}.pdf", **histogram})
                for index, frequency in enumerate(frequencies):
                    histogram_rows.append({"radius_bin": key, "metric": metric, "unit": histogram["unit"],
                                           "bin_index": index, "lower_bound": float(edges[index]),
                                           "upper_bound": float(edges[index + 1]),
                                           "upper_inclusive": index == len(frequencies) - 1,
                                           "case_count": int(frequency), "included_case_count": len(rows)})
    return figures, histogram_rows


def _write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _markdown(aggregate: dict, figures: list[dict], records: list[dict], run_id: str) -> str:
    lines = ["# PARSE predicted arterial radius statistics", "", f"Run: `{run_id}`.", "",
             f"Status: **{aggregate['status']}**. Included **{aggregate['included_case_count']} / "
             f"{aggregate['requested_case_count']} requested cases**; excluded **{aggregate['excluded_case_count']}**.", "",
             "## Measurement and population", "",
             "Only PARSE predictions are included. Each case uses its largest 26-connected arterial mask component. "
             "Excluded components do not contribute segments or the denominator. Component retention and excluded "
             "volume are retained in the individual computation records.", "",
             "A segment is a maximal graph path between the root, junctions, or terminal points of the deduplicated "
             "EDT centerline tree. Its representative radius (`segment_radius_mm`) is the physical-length-weighted "
             "mean of local EDT samples, integrated by the trapezoidal rule over its unique edges. "
             "This is a mask-derived scale estimate, not a validated cross-sectional anatomical radius. "
             "Graph segments are not established anatomical branches and their count is not the existing Fiji branch count.", "",
             "Classification uses unrounded radii in millimetres: **(0, 2], (2, 4], (4, 6], and (6, infinity)**. "
             "Zero, negative, or nonfinite radii invalidate a case rather than being assigned to a group.", "",
             "For each case, the denominator is that case's total EDT graph segment count: fraction = group count / "
             "total segment count. The four fractions sum to one. Missing or failed cases are excluded, never imputed as zeros.", "",
             "## Pooled counts and proportions", "",
             "Pooled proportions divide the sum of group counts by the sum of segment counts across included cases. "
             "They weight cases by their segment counts and differ from the unweighted mean of case proportions.", "",
             "| Radius interval | Segment count | Pooled proportion |", "|---|---:|---:|"]
    for row in aggregate["cohort_bins"]:
        proportion = f"{row['fraction']:.2%}" if row["fraction"] is not None else "Unavailable"
        lines.append(f"| {row['label']} | {row['segment_count']:,} | {proportion} |")
    lines += [f"| Total | {aggregate['pooled_segment_count']:,} | "
              f"{'100.00%' if aggregate['pooled_segment_count'] else 'Unavailable'} |", "",
              "## Distributions across cases", "",
              "Each included case contributes one observation to each histogram. Proportion axes use percentages; "
              "the CSV fraction fields remain between zero and one. Histogram widths follow the Freedman-Diaconis "
              "rule, rounded upward to at least one for segment counts; zero-IQR samples use a range/Sturges fallback "
              "and constant samples use one bin. Exact histogram edges and frequencies are saved in `histogram_bins.csv`.", "",
              "Descriptive statistics in `distribution_summary.csv` use each case once: mean, sample standard deviation "
              "(ddof=1), median, linearly interpolated quartiles, minimum, and maximum. Standard deviation is unavailable "
              "for a single case. No mean or median markers are drawn on the figures.", ""]
    if figures:
        lines += ["| Radius interval | Count distribution | Proportion distribution |", "|---|---|---|"]
        for definition in BIN_DEFINITIONS:
            selected = {row["metric"]: row for row in figures if row["radius_bin"] == definition["radius_bin"]}
            cells = []
            for metric in ("segment_count", "fraction"):
                row = selected[metric]
                cells.append(f"![{definition['label']} {metric}]({row['png']}) [PDF]({row['pdf']})")
            lines.append(f"| {definition['label']} | {' | '.join(cells)} |")
    else:
        lines.append("No distribution figures were generated because no cases completed successfully.")
    lines += ["", "## Provenance and review status", ""]
    included = set(aggregate["included_cases"])
    selected_records = [row for row in records if row.get("case") in included]
    for key, label in (("segmentation_provenance_status", "Segmentation provenance"),
                       ("visual_review", "Visual review"), ("anatomical_validation", "Anatomical validation")):
        counts = Counter(str(row.get(key, "not_recorded")) for row in selected_records)
        detail = "; ".join(f"`{value}`: {count}" for value, count in sorted(counts.items())) or "Unavailable"
        lines.append(f"- {label}: {detail}.")
    inherited = [row for row in selected_records if row.get("inherited_review_report")]
    if inherited:
        lines += ["", "| Case | Baseline validation | Inherited review scope | Review report |", "|---|---|---|---|"]
        for row in inherited:
            scope = str(row.get("review_scope", "Not recorded")).replace("|", "\\|").replace("\n", " ")
            lines.append(f"| {row['case']} | `{row.get('baseline_validation', 'not_recorded')}` | "
                         f"{scope} | [Saved review]({row['inherited_review_report']}) |")
    lines += ["", "Worker report hashes bind the reported case metadata to the saved computations. "
              "Scientific parameters, computation code hashes, software versions, and inherited baseline evidence "
              "are retained in `sources.json`. Inputs are verified before and after report generation."]
    lines += ["", "Existing segmentation provenance and review status are inherited without promotion. "
              "Legacy adopted predictions retain their unverified historical CT association. Successful numerical "
              "checks do not establish anatomical validity, and known centerline topology concerns remain unresolved.", "",
              "## Excluded cases", ""]
    if aggregate["exclusions"]:
        lines += ["| Case | Status | Reason |", "|---|---|---|"]
        for row in aggregate["exclusions"]:
            escaped = [str(row[key]).replace("|", "\\|").replace("\n", " ") for key in ("case", "status", "reason")]
            lines.append("| " + " | ".join(escaped) + " |")
    else:
        lines.append("All requested cases are included. This completion status refers to the requested cohort listed in `sources.json`.")
    lines += ["", "## Data files", "", "- [Per-case radius groups](per_case_radius_bins.csv)",
              "- [Pooled radius groups](cohort_radius_bins.csv)", "- [Distribution summaries](distribution_summary.csv)",
              "- [Histogram bins](histogram_bins.csv)", "- [Excluded cases](exclusions.csv)",
              "- [Sources, parameters, software versions, and artifact hashes](sources.json)", ""]
    return "\n".join(lines)


def _verify_input_hashes(hashes: dict[Path, str]) -> None:
    for path, expected in hashes.items():
        if not path.is_file() or _sha256(path) != expected:
            raise ValueError(f"Input hash mismatch or missing source: {path}")


def _collect_report_inputs(records: list[dict], report_dir: Path) -> tuple[dict, list[dict]]:
    """Bind checkpoint metadata to immutable worker results and review evidence."""
    source_hashes, provenance = {}, []

    def collect(hashes, name):
        if not isinstance(hashes, dict):
            raise ValueError(f"{name} must map input paths to SHA256 digests")
        for raw_path, expected in hashes.items():
            path = Path(raw_path).resolve()
            if path in source_hashes and source_hashes[path] != expected:
                raise ValueError(f"Conflicting input hashes: {path}")
            if path == report_dir or report_dir in path.parents:
                raise ValueError("Report outputs must not contain or replace source inputs")
            source_hashes[path] = expected

    for record in records:
        for name in ("source_sha256", "artifact_sha256", "code_sha256", "baseline_sha256"):
            collect(record.get(name, {}), name)
        identity = record.get("computation_identity", {})
        if not isinstance(identity, dict):
            raise ValueError("computation_identity must be an object")
        collect(identity.get("code_sha256", {}), "computation_identity.code_sha256")
        has_path = "worker_report_path" in record
        has_digest = "worker_report_sha256" in record
        if has_path != has_digest:
            raise ValueError("worker_report_path and worker_report_sha256 must be provided together")
        if has_path:
            if not record["worker_report_path"] or not record["worker_report_sha256"]:
                raise ValueError("Worker report path and SHA256 must be nonempty")
            path = Path(record["worker_report_path"]).resolve()
            expected = record["worker_report_sha256"]
            collect({str(path): expected}, "worker_report_sha256")
            _verify_input_hashes({path: expected})
            worker = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(worker, dict):
                raise ValueError("Worker report must be a JSON object")
            for key, value in worker.items():
                if key == "artifact_sha256":
                    if any(record.get(key, {}).get(name) != digest for name, digest in value.items()):
                        raise ValueError("Worker report artifact hashes differ from the case record")
                elif key not in ("visual_review", "review_observations") and record.get(key) != value:
                    raise ValueError(f"Worker report {key} differs from the case record")
            for key in ("parameters", "versions", "code_sha256"):
                if key in identity and identity[key] != worker.get(key):
                    raise ValueError(f"Computation identity {key} differs from the worker report")
            review_changed = any(record.get(key) != worker.get(key)
                                 for key in ("visual_review", "review_observations"))
            if review_changed and not (record.get("inherited_review_report") and record.get("review_scope")
                                       and record.get("baseline_validation") and record.get("baseline_sha256")):
                raise ValueError("Changed worker review status requires explicit inherited baseline evidence")
        if record.get("inherited_review_report"):
            review_path = str(Path(record["inherited_review_report"]).resolve())
            baseline_paths = {str(Path(path).resolve()) for path in record.get("baseline_sha256", {})}
            if review_path not in baseline_paths:
                raise ValueError("Inherited review report must have a verified baseline hash")
        provenance.append({key: record[key] for key in (
            "case", "status", "segmentation_provenance_status", "model_input_verified", "visual_review",
            "anatomical_validation", "worker_report_path", "worker_report_sha256", "parameters", "versions",
            "code_sha256", "computation_identity", "grouping_code_sha256", "baseline_sha256",
            "baseline_validation", "inherited_review_report", "review_scope", "review_observations") if key in record})
    _verify_input_hashes(source_hashes)
    return source_hashes, provenance


def _publish_report(records, requested_cases, report_dir, *, run_id, computation_dir,
                    aggregate, source_hashes, case_provenance, publication_code_hashes, record_digest):
    report_dir.mkdir(parents=True, exist_ok=True)
    assets = report_dir / "assets"
    assets.mkdir()
    _write_csv(report_dir / "per_case_radius_bins.csv", aggregate["case_bins"], ["case", *BIN_FIELDS])
    _write_csv(report_dir / "cohort_radius_bins.csv", aggregate["cohort_bins"], [*BIN_FIELDS, "case_count"])
    _write_csv(report_dir / "distribution_summary.csv", aggregate["descriptives"],
               ["radius_bin", "label", "metric", "n", "mean", "std", "median", "q25", "q75", "min", "max"])
    _write_csv(report_dir / "exclusions.csv", aggregate["exclusions"], ["case", "status", "reason"])
    figures, histogram_rows = _plot_histograms(aggregate, assets)
    _write_csv(report_dir / "histogram_bins.csv", histogram_rows,
               ["radius_bin", "metric", "unit", "bin_index", "lower_bound", "upper_bound",
                "upper_inclusive", "case_count", "included_case_count"])
    (report_dir / "report.md").write_text(_markdown(aggregate, figures, records, run_id), encoding="utf-8")
    import matplotlib
    hashes = {str(path): _sha256(path) for path in sorted(report_dir.rglob("*")) if path.is_file()}
    manifest = {"schema_version": "1.0", "dataset": "PARSE", "source": "prediction", "run_id": run_id,
                "status": aggregate["status"], "computation_dir": str(computation_dir),
                "requested_cases": list(requested_cases), "included_cases": aggregate["included_cases"],
                "excluded_cases": aggregate["exclusions"], "bins": list(BIN_DEFINITIONS),
                "source_sha256": {str(path): value for path, value in sorted(source_hashes.items())},
                "record_payload_sha256": record_digest,
                "code_sha256": publication_code_hashes,
                "versions": {"python": platform.python_version(), "numpy": np.__version__,
                             "matplotlib": matplotlib.__version__},
                "render_parameters": {"style": "academic-v2", "figure_size_inches": list(FIGURE_SIZE),
                                      "png_dpi": PNG_DPI, "color": COLORS["arterial"],
                                      "background": "white", "title": False, "mean_median_markers": False},
                "definitions": {"segment_radius_mm": "Physical-length-weighted mean EDT radius in millimetres",
                                "denominator": "Total EDT graph segment count in the largest 26-connected mask component",
                                "pooled_fraction": "Sum of group counts / sum of total segment counts across included cases",
                                "distribution_fraction": "One group count / total segment count observation per case",
                                "standard_deviation": "Sample standard deviation, ddof=1; unavailable if n < 2",
                                "quantiles": "Linear interpolation", "zero_radius": "Invalid; excludes the case"},
                "case_provenance": case_provenance,
                "input_integrity": "pending_post_publication_verification",
                "figures": figures, "artifact_sha256": hashes}
    source_path = report_dir / "sources.json"
    source_path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    _verify_input_hashes({**source_hashes, **{Path(path): digest for path, digest in publication_code_hashes.items()}})
    manifest["input_integrity"] = "verified_after_publication"
    source_path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return {**aggregate, "report_path": str(report_dir / "report.md"), "sources_path": str(source_path),
            "artifact_sha256": {**hashes, str(source_path): _sha256(source_path)}}


def build_report(records: list[dict], requested_cases: list[str], report_dir: Path, *,
                 run_id: str, computation_dir: Path) -> dict:
    """Publish a new report, with worker binding and before/after input checks."""
    aggregate = aggregate_cases(records, requested_cases)
    report_dir, computation_dir = Path(report_dir).resolve(), Path(computation_dir).resolve()
    if report_dir.exists() and any(report_dir.iterdir()):
        raise ValueError("Report directory must be new or empty; existing reports are never overwritten")
    source_hashes, provenance = _collect_report_inputs(records, report_dir)
    style_path = Path(__file__).with_name("plot_parse_distributions.py").resolve()
    publication_code_hashes = {str(Path(__file__).resolve()): _sha256(Path(__file__)),
                              str(style_path): _sha256(style_path)}
    record_digest = hashlib.sha256(json.dumps(records, sort_keys=True, allow_nan=False).encode()).hexdigest()
    try:
        return _publish_report(records, requested_cases, report_dir, run_id=run_id,
                               computation_dir=computation_dir, aggregate=aggregate,
                               source_hashes=source_hashes, case_provenance=provenance,
                               publication_code_hashes=publication_code_hashes, record_digest=record_digest)
    except Exception as error:
        # A late mutation or rendering error must never leave apparently valid output.
        if report_dir.exists():
            failure = {"status": "error", "run_id": run_id, "dataset": "PARSE", "source": "prediction",
                       "error": f"{type(error).__name__}: {error}", "valid_for_analysis": False,
                       "input_integrity": "publication_failed", "requested_cases": list(requested_cases),
                       "source_sha256": {str(path): value for path, value in sorted(source_hashes.items())},
                       "code_sha256": publication_code_hashes, "case_provenance": provenance,
                       "record_payload_sha256": record_digest}
            (report_dir / "failure.json").write_text(json.dumps(failure, indent=2, allow_nan=False) + "\n",
                                                    encoding="utf-8")
            (report_dir / "report.md").write_text(
                "# PARSE predicted arterial radius statistics\n\n"
                "Status: **error**. Publication failed; the tables and figures in this directory are invalid "
                "and must not be used for analysis.\n\n"
                "See [failure details](failure.json) and [source evidence](sources.json).\n", encoding="utf-8")
            (report_dir / "sources.json").write_text(json.dumps(failure, indent=2, allow_nan=False) + "\n",
                                                    encoding="utf-8")
        raise
