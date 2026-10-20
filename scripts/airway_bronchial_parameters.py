"""Paper-aligned airway branch, case and cohort measurements.

This module does not segment or assign anatomical generations. ``summarize_branch``
accepts branch metadata (``case``, ``branch_id``, ``generation``) and sections with
``s_mm``, ``inner_radii_mm`` and ``outer_radii_mm`` on the same complete, uniformly
spaced angular grid. Smoothness-supported rays are estimates and are allowed;
missing rays, non-nested geometry and explicitly excluded sections are not.
Local radii are arithmetic means of ray radii. LR/TR are then arithmetic means
over exactly the same usable positions, BEFORE applying the area formulas.

Branch results contain flat LR_mm/TR_mm/LA_mm2/TA_mm2/WA_mm2/WT_mm/WAP_pct/P_mm,
``radius_samples``, counts, flags and review status. ``raw_parameters`` preserves
numerically valid estimates before explicit review exclusions; optional
``reviewed_parameters`` uses only explicitly accepted branches/positions.
Case results contain LA_g3_5_mm2/WT_g3_5_mm/WAP_g3_5_pct/Pi10_g0_6_mm,
generation counts/means and the complete ``pi10`` fit record. Missing generation
coverage does not suppress available-branch exploratory case means. Cohort
summaries give each case one vote, with linear-interpolated percentiles.

Interim choices (not recovered author settings): mean ray radius;
equal_branch_mean_g3_g5; Statsmodels RLM with HuberT(t=1.345), MAD scale and
coefficient convergence. Source paper: https://doi.org/10.1148/radiol.232677.
Fitting API: https://www.statsmodels.org/stable/rlm.html.
"""
from __future__ import annotations

import math
import warnings

import numpy as np


RADIUS_ESTIMATOR = "mean_ray_to_surface_radius_complete_contour"
CASE_AVERAGING_POLICY = "equal_branch_mean_g3_g5"
PARAMETER_KEYS = ("LR_mm", "TR_mm", "LA_mm2", "TA_mm2", "WA_mm2", "WT_mm",
                  "WAP_pct", "P_mm", "sqrt_WA_mm")
CASE_METRICS = {"LA_g3_5_mm2": "mm2", "WT_g3_5_mm": "mm",
                "WAP_g3_5_pct": "%", "Pi10_g0_6_mm": "mm"}
EXCLUDED_STATES = {"excluded", "unavailable", "unusable", "failed", "error"}
ACCEPTED_REVIEWS = {"accepted", "included", "reviewed_accepted"}
REJECTED_REVIEWS = {"rejected", "excluded", "reviewed_excluded"}


def _number(value):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _flags(record):
    value = record.get("flags", [])
    return [value] if isinstance(value, str) else list(value or [])


def _review(record):
    return str(record.get("review_status", "not_reviewed"))


def _radius_array(value):
    if value is None:
        return None
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return None
    if array.ndim != 1 or array.size < 3 or not np.all(np.isfinite(array)):
        return None
    return array if np.all(array > 0) else None


def _parameters(samples):
    values = {key: None for key in PARAMETER_KEYS}
    if not samples:
        return values
    lr = float(np.mean([sample["inner_radius_mm"] for sample in samples]))
    tr = float(np.mean([sample["outer_radius_mm"] for sample in samples]))
    la, ta = math.pi * lr**2, math.pi * tr**2
    wa = ta - la
    return dict(LR_mm=lr, TR_mm=tr, LA_mm2=la, TA_mm2=ta, WA_mm2=wa,
                WT_mm=tr-lr, WAP_pct=100 * wa / ta, P_mm=2 * math.pi * lr,
                sqrt_WA_mm=math.sqrt(wa))


def summarize_branch(branch: dict, sections: list[dict]) -> dict:
    """Reduce paired positions to a branch, keeping exclusions and raw estimates.

    A branch/section review accepts values only when ``review_status`` is one of
    ``accepted``, ``included`` or ``reviewed_accepted``. Explicit rejection is
    honored independently of numerical usability. Duplicate physical positions
    are excluded after the first occurrence to prevent accidental weighting.
    """
    samples, raw, included, reviewed = [], [], [], []
    seen = set()
    branch_flags = _flags(branch)
    branch_rejected = _review(branch) in REJECTED_REVIEWS
    branch_excluded = str(branch.get("status", "estimated")) in EXCLUDED_STATES
    branch_accepted = _review(branch) in ACCEPTED_REVIEWS
    for index, section in enumerate(sections):
        inner = _radius_array(section.get("inner_radii_mm"))
        outer = _radius_array(section.get("outer_radii_mm"))
        position = _number(section.get("s_mm"))
        flags, reasons = _flags(section), []
        if position is None:
            reasons.append("missing_physical_position")
        elif position in seen:
            reasons.append("duplicate_physical_position")
        else:
            seen.add(position)
        if inner is None:
            reasons.append("missing_or_incomplete_inner_contour")
        if outer is None:
            reasons.append("missing_or_incomplete_outer_contour")
        if inner is not None and outer is not None:
            if inner.shape != outer.shape:
                reasons.append("unpaired_angular_grid")
            elif not np.all(outer > inner):
                reasons.append("non_nested_contours")
        if section.get("contour_complete") is False or section.get("complete_contour") is False:
            reasons.append("incomplete_contour")
        theta = section.get("theta_rad")
        if theta is not None and inner is not None:
            angle = np.asarray(theta, dtype=float)
            if (angle.shape != inner.shape or not np.all(np.isfinite(angle)) or
                    not np.allclose(np.diff(np.r_[angle, angle[0] + 2 * math.pi]),
                                    2 * math.pi / len(angle), atol=1e-6, rtol=1e-6)):
                reasons.append("nonuniform_or_incomplete_angular_grid")
        sample = {
            "case": branch.get("case"), "branch_id": branch.get("branch_id"),
            "generation": branch.get("generation"),
            "section_id": section.get("section_id", index), "s_mm": position,
            "inner_radius_mm": float(np.mean(inner)) if inner is not None else None,
            "outer_radius_mm": float(np.mean(outer)) if outer is not None else None,
            "radius_estimator": RADIUS_ESTIMATOR,
            "status": section.get("status", "estimated"),
            "review_status": _review(section), "flags": flags,
            "paired_geometry_available": not reasons,
            "included": False, "reviewed_included": False,
            "exclusion_reasons": list(reasons),
        }
        if not reasons:
            raw.append(sample)
        if str(section.get("status", "estimated")) in EXCLUDED_STATES:
            reasons.append("section_" + str(section["status"]))
        if _review(section) in REJECTED_REVIEWS:
            reasons.append("section_excluded_after_review")
        if branch_excluded:
            reasons.append("branch_" + str(branch["status"]))
        if branch_rejected:
            reasons.append("branch_excluded_after_review")
        if not reasons:
            sample["included"] = True
            included.append(sample)
            if branch_accepted or _review(section) in ACCEPTED_REVIEWS:
                sample["reviewed_included"] = True
                reviewed.append(sample)
        sample["exclusion_reasons"] = reasons
        samples.append(sample)
    flags = sorted(set(branch_flags + [flag for sample in samples for flag in sample["flags"]]))
    if len(included) < len(samples):
        flags = sorted(set(flags + ["incomplete_paired_position_coverage"]))
    needs_review = bool(flags or branch.get("status") == "needs_review" or any(
        sample["status"] == "needs_review" for sample in included))
    has_review = branch_accepted or branch_rejected or any(
        _review(section) in ACCEPTED_REVIEWS | REJECTED_REVIEWS for section in sections)
    if branch_rejected:
        review_status = "rejected"
    elif included and len(reviewed) == len(included):
        review_status = "accepted"
    elif has_review:
        review_status = "partially_reviewed"
    else:
        review_status = "not_reviewed"
    result = {
        key: branch[key] for key in ("case", "branch_id", "generation", "source_branch_ids",
                                    "component_id", "length_mm", "method") if key in branch}
    result.update(_parameters(included))
    result.update(
        radius_estimator=RADIUS_ESTIMATOR,
        branch_averaging_policy="equal_paired_position_mean_then_radius_formulas",
        status=("needs_review" if needs_review else "estimated") if included else "unavailable",
        review_status=review_status, flags=flags,
        section_count=len(samples), paired_geometry_count=len(raw),
        included_sample_count=len(included), excluded_sample_count=len(samples)-len(included),
        inner_only_sample_count=sum(sample["inner_radius_mm"] is not None and
                                    sample["outer_radius_mm"] is None for sample in samples),
        reviewed_sample_count=len(reviewed),
        included_s_mm=[sample["s_mm"] for sample in included],
        included_section_ids=[sample["section_id"] for sample in included],
        raw_parameters=_parameters(raw),
        reviewed_parameters=_parameters(reviewed) if has_review else None,
        radius_samples=samples,
        unavailable_reason=None if included else "no_usable_paired_positions",
    )
    # Numerical support and search-limit diagnostics explain an estimate; they
    # are not acceptance thresholds or calibrated probabilities of correctness.
    paired_sections = [section for section, sample in zip(sections, samples) if sample["included"]]
    support = [section.get("support", {}).get("outer_fraction") for section in paired_sections]
    support = [value for value in support if _number(value) is not None]
    lower_hits = []
    for section in paired_sections:
        config = section.get("config", {})
        if "min_wall_search_mm" in config and "radial_step_mm" in config:
            separation = np.asarray(section["outer_radii_mm"]) - section["inner_radii_mm"]
            lower_hits.append(float(np.mean(separation <= config["min_wall_search_mm"] + .5 * config["radial_step_mm"])))
    result["mean_outer_direct_support_fraction"] = float(np.mean(support)) if support else None
    result["mean_minimum_separation_hit_fraction"] = float(np.mean(lower_hits)) if lower_hits else None
    return result


def fit_pi10(branches: list[dict], *, maxiter: int = 200, tol: float = 1e-8) -> dict:
    """Fit sqrt(WA) against P for available generations 0-6 using robust IRLS.

    At least three branches and two distinct perimeters are required so that a
    residual scale can be estimated. This minimum is an implementation choice.
    Unconverged or negative predictions remain unavailable, with fit diagnostics.
    """
    if maxiter < 2 or not math.isfinite(tol) or tol <= 0:
        raise ValueError("maxiter must be at least 2 and tol must be finite and positive")
    selected = [branch for branch in branches if branch.get("generation") in range(7)
                and branch.get("status") not in EXCLUDED_STATES
                and _review(branch) not in REJECTED_REVIEWS
                and _number(branch.get("P_mm")) is not None and branch["P_mm"] > 0
                and _number(branch.get("WA_mm2")) is not None and branch["WA_mm2"] > 0]
    x = np.array([branch["P_mm"] for branch in selected], dtype=float)
    y = np.sqrt([branch["WA_mm2"] for branch in selected])
    result = dict(
        method="statsmodels_rlm_huber_t", loss="HuberT", huber_t=1.345,
        scale_estimator="mad", convergence_criterion="coefs", maxiter=maxiter, tol=tol,
        generation_range=[0, 6], target_perimeter_mm=10.0,
        selected_branch_ids=[branch.get("branch_id") for branch in selected],
        selected_generations=[branch.get("generation") for branch in selected],
        P_mm=x.tolist(), sqrt_WA_mm=y.tolist(), n_branches=len(selected),
        perimeter_min_mm=float(x.min()) if len(x) else None,
        perimeter_max_mm=float(x.max()) if len(x) else None,
        extrapolation=bool(10 < x.min() or 10 > x.max()) if len(x) else None,
        status="unavailable", unavailable_reason=None, converged=False,
        intercept_mm=None, slope=None, Pi10_mm=None, raw_prediction_mm=None,
        iterations=None, coefficient_change=None, residual_scale=None,
        weights=[], residuals=[], warnings=[], flags=[], statsmodels_version=None,
    )
    if len(x) < 3:
        result["unavailable_reason"] = "insufficient_branches_for_robust_fit"
        return result
    design = np.column_stack((np.ones(len(x)), x))
    if np.ptp(x) <= 1e-8 * max(1., float(np.max(np.abs(x)))) or np.linalg.matrix_rank(design) < 2:
        result["unavailable_reason"] = "unidentifiable_perimeter_range"
        return result
    try:
        import statsmodels
        from statsmodels.robust.norms import HuberT
        from statsmodels.robust.robust_linear_model import RLM
    except ImportError:
        result["unavailable_reason"] = "statsmodels_not_installed"
        return result
    result["statsmodels_version"] = statsmodels.__version__
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fit = RLM(y, design, M=HuberT(t=1.345), missing="raise").fit(
                maxiter=maxiter, tol=tol, scale_est="mad", conv="coefs")
        result["warnings"] = sorted(set(str(warning.message) for warning in caught))
        params = np.asarray(fit.params)
        history = [np.asarray(item) for item in fit.fit_history.get("params", [])
                   if np.asarray(item).shape == (2,)]
        change = float(np.max(np.abs(history[-1] - history[-2]))) if len(history) >= 2 else None
        exact = bool(np.allclose(fit.resid, 0, atol=tol, rtol=0))
        converged = bool(np.all(np.isfinite(params)) and
                         ((change is not None and change <= tol) or exact))
        result.update(
            intercept_mm=_number(params[0]), slope=_number(params[1]),
            iterations=int(fit.fit_history.get("iteration", 0)), coefficient_change=_number(change),
            residual_scale=_number(fit.scale), converged=converged,
            weights=[_number(value) for value in np.asarray(fit.weights)],
            residuals=[_number(value) for value in np.asarray(fit.resid)],
        )
        if not np.all(np.isfinite(params)):
            result["unavailable_reason"] = "nonfinite_fit_coefficients"
            return result
        prediction = float(params @ [1., 10.])
        result["raw_prediction_mm"] = _number(prediction)
        if not converged:
            result["unavailable_reason"] = "fit_did_not_converge"
        elif not math.isfinite(prediction) or prediction < 0:
            result["unavailable_reason"] = "nonphysical_prediction"
        else:
            result.update(status="estimated", Pi10_mm=prediction)
        if result["extrapolation"]:
            result["flags"].append("target_perimeter_outside_observed_range")
        if params[1] < 0:
            result["flags"].append("negative_perimeter_slope")
        if exact:
            result["flags"].append("numerically_exact_linear_fit")
    except (ValueError, RuntimeError, ArithmeticError, np.linalg.LinAlgError) as error:
        result["unavailable_reason"] = "robust_fit_error"
        result["error"] = str(error)
    return result


def _usable(branch):
    return (branch.get("status") not in EXCLUDED_STATES and
            _review(branch) not in REJECTED_REVIEWS and
            all(_number(branch.get(key)) is not None for key in ("LA_mm2", "WT_mm", "WAP_pct")))


def _case_values(branches):
    included = [branch for branch in branches if branch.get("generation") in (3, 4, 5) and _usable(branch)]
    pi10 = fit_pi10(branches)
    values = {output: float(np.mean([branch[key] for branch in included])) if included else None
              for output, key in (("LA_g3_5_mm2", "LA_mm2"), ("WT_g3_5_mm", "WT_mm"),
                                  ("WAP_g3_5_pct", "WAP_pct"))}
    values.update(Pi10_g0_6_mm=pi10["Pi10_mm"], pi10=pi10,
                  included_branch_ids_g3_5=[branch.get("branch_id") for branch in included],
                  included_branch_count_g3_5=len(included),
                  included_generations_g3_5=sorted(set(branch["generation"] for branch in included)))
    return values


def summarize_case(case: str, branches: list[dict], *, method=None, configuration=None) -> dict:
    """One equal-branch g3-5 mean and g0-6 Pi10 per case; do not pool points."""
    ids = [str(branch.get("branch_id")) for branch in branches]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate branch IDs would weight a case more than once")
    if any(branch.get("case", case) != case for branch in branches):
        raise ValueError("Branches from different cases cannot be pooled")
    values = _case_values(branches)
    counts, means = {}, {}
    for generation in range(7):
        group = [branch for branch in branches if branch.get("generation") == generation]
        available = [branch for branch in group if _usable(branch)]
        counts[str(generation)] = dict(
            total=len(group), measurable=len(available), excluded=len(group)-len(available),
            needs_review=sum(branch.get("status") == "needs_review" for branch in available),
            reviewed=sum(_review(branch) in ACCEPTED_REVIEWS for branch in available),
        )
        means[str(generation)] = {
            key: float(np.mean([branch[key] for branch in available])) if available else None
            for key in ("LA_mm2", "WT_mm", "WAP_pct")}
    missing = [generation for generation in range(7) if not counts[str(generation)]["total"]]
    unmeasurable = [generation for generation in range(7) if not counts[str(generation)]["measurable"]]
    flags = sorted(set(flag for branch in branches for flag in _flags(branch)))
    if missing:
        flags.append("missing_generations")
    if any(value["excluded"] for value in counts.values()):
        flags.append("excluded_or_unavailable_branches")
    if any(branch.get("generation") is None for branch in branches):
        flags.append("unassigned_branch_generations")
    flags.extend(values["pi10"]["flags"])
    reviewed_branches = []
    has_review = any(_review(branch) in ACCEPTED_REVIEWS | REJECTED_REVIEWS | {"partially_reviewed"}
                     for branch in branches)
    if has_review:
        for branch in branches:
            if _review(branch) in REJECTED_REVIEWS:
                continue
            if branch.get("reviewed_parameters") is not None:
                reviewed_branches.append({**branch, **branch["reviewed_parameters"],
                                          "review_status": "accepted"})
            elif _review(branch) in ACCEPTED_REVIEWS:
                reviewed_branches.append(branch)
    result = dict(case=case, **values, case_averaging_policy=CASE_AVERAGING_POLICY,
                  radius_estimator=RADIUS_ESTIMATOR, method=method, configuration=configuration,
                  generation_counts=counts, generation_means=means,
                  missing_generations=missing, unmeasurable_generations=unmeasurable,
                  branch_count=len(branches), measurable_branch_count=sum(_usable(branch) for branch in branches),
                  unresolved_flagged_branch_count=sum(branch.get("status") == "needs_review" and
                                                       _review(branch) not in ACCEPTED_REVIEWS for branch in branches),
                  review_status="reviewed_only_available" if has_review else "not_reviewed",
                  review_policy="exploratory_estimates_including_unresolved_flags",
                  coverage_status="incomplete" if unmeasurable or any(value["excluded"] for value in counts.values())
                                  else "available_g0_g6",
                  flags=sorted(set(flags)),
                  reviewed_only=_case_values(reviewed_branches) if has_review else None)
    result["unavailable_reasons"] = {
        metric: (values["pi10"]["unavailable_reason"] if metric == "Pi10_g0_6_mm"
                 else "no_measurable_generation_3_to_5_branches")
        for metric in CASE_METRICS if values[metric] is None}
    result["status"] = ("needs_review" if result["flags"] or result["review_status"] == "not_reviewed"
                        else "estimated") if any(values[key] is not None for key in CASE_METRICS) else "unavailable"
    return result


def summarize_cohort(case_rows: list[dict]) -> list[dict]:
    """Return case-wise summaries, including reviewed-only rows only after review."""
    case_ids = [row["case"] for row in case_rows]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("Cohort summaries require one row per case")
    policies = ["exploratory"]
    if any(row.get("reviewed_only") is not None for row in case_rows):
        policies.append("reviewed_only")
    rows = []
    for policy in policies:
        for metric, unit in CASE_METRICS.items():
            selected = []
            for row in case_rows:
                if row.get("status") in EXCLUDED_STATES or _review(row) in REJECTED_REVIEWS:
                    continue
                source = row if policy == "exploratory" else (row.get("reviewed_only") or {})
                value = _number(source.get(metric))
                if value is not None:
                    selected.append((row, value))
            values = np.asarray([value for _, value in selected])
            result = dict(metric=metric, unit=unit, unit_of_analysis="case", review_policy=policy,
                          n=len(values), n_total=len(case_rows), n_missing=len(case_rows)-len(values),
                          case_ids=[row["case"] for row, _ in selected],
                          n_needs_review=sum(row.get("status") == "needs_review" for row, _ in selected),
                          n_incomplete=sum(row.get("coverage_status") == "incomplete" for row, _ in selected),
                          mean=float(values.mean()) if len(values) else None,
                          sd=float(values.std(ddof=1)) if len(values) > 1 else None,
                          median=float(np.median(values)) if len(values) else None,
                          minimum=float(values.min()) if len(values) else None,
                          maximum=float(values.max()) if len(values) else None,
                          percentile_method="linear")
            result.update({f"p{q}": float(np.percentile(values, q, method="linear")) if len(values) else None
                           for q in (10, 25, 50, 75, 90)})
            rows.append(result)
    return rows
