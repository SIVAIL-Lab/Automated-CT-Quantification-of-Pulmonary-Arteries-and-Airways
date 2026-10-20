"""Coupled branch-local airway surfaces, an independent Opfront approximation.

The pinned author ``src/segwall/main.cpp`` (e76cd3d06b567e8e5170a5a7b5603d6b4447f0af)
uses signed image derivatives, linear surface smoothness, and ordered interacting
surfaces without a preferred thickness. This module independently implements that
energy family on straight radial columns in transported centerline planes. It
does NOT implement the 2014 non-intersecting flow lines, the original training,
or the 2019 min-marginal continuous-label refinement. All settings are declared
engineering choices. A complete estimate is not a validated anatomical boundary.
Version 2 adds a robust angular CT bright-tissue enclosure cost to the outer
surface only. Its thresholds and weights are engineering choices, not recovered
or calibrated parameters from the paper.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import time

import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter1d, label, map_coordinates
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import breadth_first_order, maximum_flow

from airway_surface_geometry import METRICS, _area, _sample, build_local_surfaces, parallel_transport_frames


METHOD = "branch_local_coupled_surfaces_v2"
SOURCE_REFERENCE = {
    "author_source": "https://bitbucket.org/id-b3/opfront/src/e76cd3d06b567e8e5170a5a7b5603d6b4447f0af/src/segwall/main.cpp",
    "paper": "https://doi.org/10.1016/j.media.2014.02.004",
    "solver": "https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csgraph.maximum_flow.html",
    "relationship": "Independent energy-family implementation; no upstream code copied",
}


@dataclass(frozen=True)
class SurfaceConfig:
    step_mm: float = .5
    plane_spacing_mm: float = .25
    radial_step_mm: float = .2
    n_angles: int = 64
    tangent_window_mm: float = 4.
    smoothing_mm: float = .35
    max_profile_radius_mm: float = 24.
    max_center_shift_mm: float = 2.
    max_inner_shift_mm: float = 1.5
    min_wall_search_mm: float = .2
    max_wall_search_mm: float = 4.
    contrast_window_mm: float = .6
    noise_floor_hu: float = 5.
    min_contrast_snr: float = 3.
    angular_weight: float = .4
    longitudinal_weight: float = .4
    separation_weight: float = .05
    initialization_weight: float = .35
    contact_weight: float = .5
    bright_tissue_weight: float = .5
    bright_tissue_snr: float = 3.
    bright_reference_inner_mm: float = .2
    bright_reference_outer_mm: float = .6
    bright_excess_clip: float = 5.
    max_axial_radius_change_mm: float = 1.5
    max_graph_nodes: int = 400000
    capacity_scale: int = 1000

    def __post_init__(self):
        values = asdict(self)
        if not all(np.isfinite(value) and value > 0 for key, value in values.items()
                   if key != "bright_tissue_weight"):
            raise ValueError("Surface configuration values must be finite and positive")
        if not np.isfinite(self.bright_tissue_weight) or self.bright_tissue_weight < 0:
            raise ValueError("Bright-tissue weight must be finite and nonnegative")
        for key in ("n_angles", "max_graph_nodes", "capacity_scale"):
            if int(values[key]) != values[key]:
                raise ValueError(f"{key} must be an integer")
        if self.n_angles < 8 or self.max_graph_nodes < 1000:
            raise ValueError("At least eight angles and 1000 graph nodes are required")
        if not self.radial_step_mm <= self.min_wall_search_mm < self.max_wall_search_mm:
            raise ValueError("Wall bounds must be ordered and resolve at least one radial step")
        if self.bright_reference_inner_mm + self.bright_reference_outer_mm < self.radial_step_mm:
            raise ValueError("Bright-tissue reference window must contain a radial sample")


def _bright_tissue_enclosure_cost(smooth, initial, rho, noise, finite, config):
    """Return an outer-only soft cost and reproducible angular outlier evidence.

    Near-wall peak intensities yield a robust angular median and MAD. Bright
    excess beyond that reference is accumulated from the initial lumen surface
    outward. No inner cost is subtracted: enclosure avoidance must not receive
    an additional reward for moving the lumen boundary into bright tissue.
    This is not an artery classifier or an anatomical confidence probability.
    """
    reference = np.array([np.max(values[(rho >= radius - config.bright_reference_inner_mm)
                                       & (rho <= radius + config.bright_reference_outer_mm)])
                          for values, radius in zip(smooth, initial)])
    median = float(np.median(reference))
    mad = float(1.4826 * np.median(np.abs(reference - median)))
    noise_median = float(np.median(noise))
    scale = max(config.noise_floor_hu, mad, noise_median)
    threshold = median + config.bright_tissue_snr * scale
    excess = np.clip((smooth - threshold) / scale, 0., config.bright_excess_clip)
    excess *= rho[None] >= initial[:, None]
    excess *= finite
    cost = config.bright_tissue_weight * np.cumsum(excess, axis=1) * config.radial_step_mm
    diagnostic = {"reference_median_hu": median, "reference_angular_mad_hu": mad,
                  "reference_noise_hu": noise_median, "robust_scale_hu": scale,
                  "threshold_hu": threshold, "snr_multiplier": config.bright_tissue_snr,
                  "outer_only_cost_weight": config.bright_tissue_weight,
                  "reference_window_mm": [-config.bright_reference_inner_mm, config.bright_reference_outer_mm],
                  "excess_clip": config.bright_excess_clip,
                  "score": "clipped_positive_excess_over_robust_scale",
                  "interpretation": "Uncalibrated angular bright-tissue outlier evidence, not an anatomical label"}
    return cost, excess, diagnostic


def solve_ordered_surfaces(inner_cost, outer_cost, *, lower=None, upper=None,
                           radial_step_mm=.2, angular_weight=.4,
                           longitudinal_weight=.4, axial_spacing_mm=None,
                           min_separation_labels=1, max_separation_labels=None,
                           capacity_scale=1000, fixed_first=None):
    """Globally minimize the retained, quantized ordered-label surface energy.

    Inputs have shape (sections, angles, radial_labels). Each threshold node
    represents label >= k. Unary differences become terminal capacities;
    backward infinite arcs enforce a prefix; bidirectional equal-level arcs
    encode L1 angular (including the seam) and longitudinal regularization.
    Cross-surface implication arcs impose minimum/maximum label separation.
    ``fixed_first`` optionally fixes the first ring for a conditional block.
    No isolated gradient-peak pruning or post-hoc ordering repair occurs.
    """
    costs = np.stack([inner_cost, outer_cost]).astype(float)
    if costs.ndim != 4 or not np.all(np.isfinite(costs)):
        raise ValueError("Finite inner/outer costs with shape (sections,angles,labels) required")
    _, ns, na, nl = costs.shape
    if ns < 1 or na < 1 or nl < 2:
        raise ValueError("At least one section/angle and two labels are required")
    minimum = int(min_separation_labels)
    maximum = nl - 1 if max_separation_labels is None else int(max_separation_labels)
    if not 1 <= minimum <= maximum < nl:
        raise ValueError("Separation labels must satisfy 1 <= min <= max < label count")
    if not all(np.isfinite(x) and x >= 0 for x in (angular_weight, longitudinal_weight)):
        raise ValueError("Regularization weights must be finite and nonnegative")
    if radial_step_mm <= 0 or capacity_scale < 1:
        raise ValueError("Positive radial step and capacity scale required")
    lower = np.zeros((2, ns, na), int) if lower is None else np.array(lower, int, copy=True)
    upper = np.full((2, ns, na), nl - 1, int) if upper is None else np.array(upper, int, copy=True)
    if lower.shape != (2, ns, na) or upper.shape != lower.shape:
        raise ValueError("Bounds must have shape (2,sections,angles)")
    if fixed_first is not None:
        fixed = np.asarray(fixed_first, int)
        if fixed.shape != (2, na):
            raise ValueError("Fixed first ring must have shape (2,angles)")
        lower[:, 0], upper[:, 0] = fixed, fixed
    if np.any(lower < 0) or np.any(upper >= nl) or np.any(lower > upper):
        raise ValueError("Invalid radial label bounds")
    if np.any(lower[0] + minimum > upper[1]) or np.any(lower[1] - maximum > upper[0]):
        raise ValueError("No ordered pair exists within supplied bounds")
    spacing = np.ones(max(0, ns - 1)) if axial_spacing_mm is None else np.asarray(axial_spacing_mm, float)
    if spacing.shape != (ns - 1,) or np.any(~np.isfinite(spacing)) or np.any(spacing <= 0):
        raise ValueError("Positive axial spacing is required between neighboring sections")

    nk = nl - 1
    ids = np.arange(2 * ns * na * nk, dtype=np.int64).reshape(2, ns, na, nk)
    source, sink = ids.size, ids.size + 1
    finite_rows, finite_cols, finite_caps = [], [], []
    hard_rows, hard_cols = [], []

    def edge(a, b, capacity=None):
        a, b = np.broadcast_arrays(a, b)
        if capacity is None:
            hard_rows.append(a.ravel())
            hard_cols.append(b.ravel())
        else:
            capacities = np.broadcast_to(capacity, a.shape)
            rounded = np.rint(capacities * capacity_scale).astype(np.int64)
            positive = rounded > 0
            finite_rows.append(a[positive].ravel())
            finite_cols.append(b[positive].ravel())
            finite_caps.append(rounded[positive].ravel())

    delta = np.diff(costs, axis=-1)
    edge(source, ids, np.maximum(-delta, 0))
    edge(ids, sink, np.maximum(delta, 0))
    edge(ids[..., 1:], ids[..., :-1])
    flat_ids = ids.reshape(-1, nk)
    flat_low, flat_high = lower.ravel(), upper.ravel()
    lo = np.flatnonzero(flat_low > 0)
    hi = np.flatnonzero(flat_high < nl - 1)
    edge(source, flat_ids[lo, flat_low[lo] - 1])
    edge(flat_ids[hi, flat_high[hi]], sink)
    if na > 1:
        neighbor = np.roll(ids, -1, axis=2)
        # Two angular columns have one undirected edge, not two duplicate seams.
        before = ids if na > 2 else ids[:, :, :1]
        after = neighbor if na > 2 else neighbor[:, :, :1]
        edge(before, after, angular_weight * radial_step_mm)
        edge(after, before, angular_weight * radial_step_mm)
    if ns > 1:
        weight = longitudinal_weight * radial_step_mm / spacing[None, :, None, None]
        edge(ids[:, :-1], ids[:, 1:], weight)
        edge(ids[:, 1:], ids[:, :-1], weight)
    # inner label >= 0 is always true, therefore outer >= minimum is mandatory.
    edge(source, ids[1, :, :, minimum - 1])
    edge(ids[0, :, :, :nk - minimum], ids[1, :, :, minimum:])
    edge(ids[0, :, :, nk - minimum:], sink)
    if maximum < nl - 1:
        edge(ids[1, :, :, maximum:], ids[0, :, :, :nk - maximum])
    capacities = np.concatenate(finite_caps)
    infinity = int(capacities.sum(dtype=np.int64)) + 1
    if infinity >= np.iinfo(np.int32).max:
        raise ValueError("Graph capacity bound exceeds SciPy int32 flow range; reduce graph block size")
    hr, hc = np.concatenate(hard_rows), np.concatenate(hard_cols)
    rows = np.concatenate([*finite_rows, hr])
    cols = np.concatenate([*finite_cols, hc])
    values = np.concatenate([capacities, np.full(hr.size, infinity, np.int64)])
    graph = coo_matrix((values, (rows, cols)), shape=(sink + 1, sink + 1)).tocsr()
    # Duplicate hard arcs can otherwise overflow the solver's integer storage.
    graph.data = np.minimum(graph.data, infinity)
    solved = maximum_flow(graph, source, sink, method="dinic")
    residual = graph - solved.flow
    residual.data = (residual.data > 0).astype(np.int8)
    residual.eliminate_zeros()
    reachable = breadth_first_order(residual, source, directed=True, return_predecessors=False)
    selected = np.zeros(sink + 1, bool)
    selected[reachable] = True
    labels = np.sum(selected[ids], axis=-1).astype(int)
    if (np.any(labels < lower) or np.any(labels > upper)
            or np.any(labels[1] - labels[0] < minimum)
            or np.any(labels[1] - labels[0] > maximum)):
        raise RuntimeError("Minimum-cut solution violates ordered surface constraints")
    unary_energy = float(np.take_along_axis(costs, labels[..., None], axis=-1).sum())
    angular_energy = float(np.abs(labels - np.roll(labels, -1, axis=2)).sum()
                           * angular_weight * radial_step_mm) if na > 1 else 0.
    if na == 2:
        angular_energy /= 2
    longitudinal_energy = float((np.abs(np.diff(labels, axis=1))
                                * longitudinal_weight * radial_step_mm / spacing[None, :, None]).sum())
    return {"labels": labels, "diagnostics": {
        "solver": "scipy.sparse.csgraph.maximum_flow.dinic", "status": "ok",
        "nodes": int(sink + 1), "directed_edges": int(graph.nnz),
        "flow_value": int(solved.flow_value), "capacity_scale": int(capacity_scale),
        "exact_for_quantized_retained_graph": True, "capacity_quantization_unit": 1 / capacity_scale,
        "unary_energy": unary_energy, "angular_regularization_energy": angular_energy,
        "longitudinal_regularization_energy": longitudinal_energy,
        "total_energy": unary_energy + angular_energy + longitudinal_energy,
        "couples_angles": na > 1, "couples_sections": ns > 1,
        "fixed_previous_block_boundary": fixed_first is not None}}


def _empty_section(sample, frame, config):
    return {**sample, **(frame or {}), "method": METHOD, "status": "unavailable", "quality": "unavailable",
           "input_status": sample.get("status"), "flags": list(sample.get("flags") or []),
           "review_status": "not_reviewed", "anatomical_validation": "not_performed",
           "original_center_mm": list(sample["center_mm"]), "center_mm": list(sample["center_mm"]), "center_adjustment_mm": 0.,
           "theta_rad": (np.arange(config.n_angles) * (2 * np.pi / config.n_angles)).tolist(),
           "rho_mm": [], "profiles_hu": None,
           "smoothed_profiles_hu": None, "initial_radii_mm": None, "inner_radii_mm": None,
           "outer_radii_mm": None, "provisional_outer_radii_mm": None,
           "inner_points_mm": None, "outer_points_mm": None, "metrics": dict.fromkeys(METRICS),
           "config": asdict(config), "radius_estimator": "mean_complete_ray_radius",
           "coverage": {"inner_closed": False, "outer_closed": False},
           "support": {"inner_fraction": 0., "outer_fraction": 0.}}


def _prepare_section(ct, mask, inverse, sample, frame, config, artery_mask):
    center = np.asarray(sample["center_mm"], float)
    e1, e2 = np.asarray(frame["e1"]), np.asarray(frame["e2"])
    theta = np.arange(config.n_angles) * (2 * np.pi / config.n_angles)
    rho = np.arange(0, config.max_profile_radius_mm + config.radial_step_mm * .5, config.radial_step_mm)
    row = _empty_section(sample, frame, config)
    row["rho_mm"] = rho.tolist()
    half = config.max_profile_radius_mm
    axis = np.arange(-half, half + config.plane_spacing_mm * .5, config.plane_spacing_mm)
    xx, yy = np.meshgrid(axis, axis, indexing="xy")
    foreground = _sample(mask, inverse, center + xx[..., None] * e1 + yy[..., None] * e2,
                         order=0, outside=0) > 0
    components, _ = label(foreground)
    distance = np.hypot(xx, yy)
    near = foreground & (distance <= config.max_center_shift_mm)
    if not np.any(near):
        row["flags"].append("no_lumen_near_original_center")
        return row, None
    seed = np.argmin(np.where(near, distance, np.inf))
    component = components == components.flat[seed]
    if np.any(component[[0, -1]]) or np.any(component[:, [0, -1]]):
        row["flags"].append("lumen_touches_sampling_window")
        return row, None
    edt = distance_transform_edt(component, sampling=config.plane_spacing_mm)
    chosen = np.unravel_index(np.argmax(np.where(near, edt - .02 * distance, -np.inf)), edt.shape)
    shift_x, shift_y = float(xx[chosen]), float(yy[chosen])
    center = center + shift_x * e1 + shift_y * e2
    shift = float(np.hypot(shift_x, shift_y))
    row.update(center_mm=center.tolist(), center_adjustment_mm=shift)
    if shift > config.plane_spacing_mm / 2:
        row["flags"].append("sampling_center_adjusted_within_initial_lumen")
    directions = np.cos(theta)[:, None] * e1 + np.sin(theta)[:, None] * e2
    world = center + directions[:, None, :] * rho[None, :, None]
    coords = np.stack([(shift_y + np.sin(theta)[:, None] * rho + half) / config.plane_spacing_mm,
                       (shift_x + np.cos(theta)[:, None] * rho + half) / config.plane_spacing_mm])
    lumen = map_coordinates(component.astype(np.uint8), coords, order=0, mode="constant", cval=0) > 0
    initial, nonstar = [], []
    for ray in lumen:
        exits = np.flatnonzero(~ray)
        if not ray[0] or not len(exits):
            row["flags"].append("lumen_ray_has_no_bounded_exit")
            return row, None
        first = int(exits[0])
        initial.append((rho[first - 1] + rho[first]) / 2)
        nonstar.append(bool(np.any(ray[first:])))
    initial = np.asarray(initial)
    row["initial_radii_mm"] = initial.tolist()
    row["initial_points_mm"] = (center + initial[:, None] * directions).tolist()
    row["nonstar_fraction"] = float(np.mean(nonstar))
    if any(nonstar):
        row["flags"].append("initial_lumen_ray_reentry")
    raw = _sample(ct, inverse, world)
    finite = np.isfinite(raw)
    if not np.all(finite[:, 0]):
        row["flags"].append("sampling_center_outside_ct")
        return row, None
    filled = np.array([np.interp(rho, rho[valid], values[valid]) for values, valid in zip(raw, finite)])
    smooth = gaussian_filter1d(filled, config.smoothing_mm / config.radial_step_mm, axis=1, mode="nearest")
    gradient = np.gradient(smooth, config.radial_step_mm, axis=1)
    residual = filled - smooth
    noise = np.maximum(config.noise_floor_hu, 1.4826 * np.median(np.abs(np.diff(residual, axis=1)), axis=1) / np.sqrt(2))
    scale = np.maximum(3 * noise, np.percentile(smooth, 95, axis=1) - np.percentile(smooth, 5, axis=1))
    contact = np.zeros(raw.shape, bool) if artery_mask is None else _sample(artery_mask, inverse, world, order=0, outside=0) > 0
    inner = -2 * np.maximum(gradient, 0) * config.contrast_window_mm / scale[:, None]
    inner += config.initialization_weight * ((rho[None] - initial[:, None]) / config.max_inner_shift_mm)**2
    outer = 2 * np.minimum(gradient, 0) * config.contrast_window_mm / scale[:, None]
    # Linear separation does not set a preferred or fixed wall thickness.
    inner -= config.separation_weight * rho[None]
    outer += config.separation_weight * rho[None]
    # Cost for extending through known enhancing artery voxels; never a veto.
    cumulative_contact = np.cumsum(contact, axis=1) * config.radial_step_mm
    outer += config.contact_weight * cumulative_contact
    inner -= config.contact_weight * cumulative_contact
    bright_cost, bright_excess, bright_diagnostic = _bright_tissue_enclosure_cost(
        smooth, initial, rho, noise, finite, config)
    outer += bright_cost
    row["ct_bright_tissue_diagnostic"] = bright_diagnostic
    low_inner = np.maximum(1, np.ceil((initial - config.max_inner_shift_mm) / config.radial_step_mm).astype(int))
    high_inner = np.floor((initial + config.max_inner_shift_mm) / config.radial_step_mm).astype(int)
    minsep = int(np.ceil(config.min_wall_search_mm / config.radial_step_mm - 1e-8))
    maxsep = int(np.floor(config.max_wall_search_mm / config.radial_step_mm + 1e-8))
    last_finite = np.array([np.flatnonzero(valid)[-1] for valid in finite])
    high_outer = np.minimum(high_inner + maxsep, last_finite - int(np.ceil(config.contrast_window_mm / config.radial_step_mm)))
    high_inner = np.minimum(high_inner, high_outer - minsep)
    low_outer = low_inner + minsep
    if np.any(high_inner < low_inner) or np.any(high_outer < low_outer):
        row["flags"].append("no_ordered_surface_search_domain_within_ct")
        return row, None
    end = min(len(rho), int(np.max(high_outer)) + 1 + int(np.ceil(config.contrast_window_mm / config.radial_step_mm)))
    row["rho_mm"] = rho[:end].tolist()
    row["profiles_hu"] = [[float(value) if valid else None for value, valid in zip(values[:end], good[:end])]
                           for values, good in zip(raw, finite)]
    row["smoothed_profiles_hu"] = smooth[:, :end].tolist()
    return row, {"inner": inner, "outer": outer, "lower": np.stack([low_inner, low_outer]),
                 "upper": np.stack([high_inner, high_outer]), "rho": rho, "smooth": smooth,
                 "finite": finite, "noise": noise, "contact": contact, "directions": directions,
                 "gradient": gradient, "initial": initial, "nonstar": nonstar,
                 "bright_excess": bright_excess, "bright_outer_cost": bright_cost}


def _finish_section(row, prepared, labels, config, block_id, diagnostics):
    rho, smooth = prepared["rho"], prepared["smooth"]
    inner, outer = rho[labels[0]], rho[labels[1]]
    n = len(inner)
    angles = np.arange(n)
    window = config.contrast_window_mm
    noise = prepared["noise"]
    inner_contrast = np.array([np.interp(ri + window, rho, values) - np.interp(max(0, ri - window), rho, values)
                               for ri, values in zip(inner, smooth)])
    outer_contrast = np.array([np.interp(max(0, ro - window), rho, values) - np.interp(ro + window, rho, values)
                               for ro, values in zip(outer, smooth)])
    inner_support = inner_contrast >= config.min_contrast_snr * noise
    outer_support = outer_contrast >= config.min_contrast_snr * noise
    contact = np.array([np.any(ray[(rho >= ri) & (rho <= ro + window)])
                        for ri, ro, ray in zip(inner, outer, prepared["contact"])])
    bound = ((outer - inner <= config.min_wall_search_mm + config.radial_step_mm * .5)
             | (outer - inner >= config.max_wall_search_mm - config.radial_step_mm * .5))
    if not np.all(inner_support):
        row["flags"].append("weak_inner_arcs_completed_by_coupled_surface_model")
    if not np.all(outer_support):
        row["flags"].append("weak_outer_arcs_completed_by_coupled_surface_model")
    if np.any(contact):
        row["flags"].append("artery_contact_requires_review")
    if np.any(bound):
        row["flags"].append("surface_search_separation_bound_reached")
    if np.any(np.abs(inner - prepared["initial"]) >= config.max_inner_shift_mm - config.radial_step_mm):
        row["flags"].append("inner_search_bound_reached")
    enclosed_bright = np.array([np.any(score[(rho >= ri) & (rho <= ro)] > 0)
                               for ri, ro, score in zip(inner, outer, prepared["bright_excess"])])
    row["ct_bright_tissue_diagnostic"]["enclosed_bright_fraction"] = float(np.mean(enclosed_bright))
    row["ct_bright_tissue_diagnostic"]["selected_outer_enclosure_cost"] = float(
        prepared["bright_outer_cost"][angles, labels[1]].sum())
    if np.any(enclosed_bright):
        row["flags"].append("ct_bright_tissue_in_estimated_wall_requires_review")
    center = np.asarray(row["center_mm"])
    theta = np.asarray(row["theta_rad"])
    ai, ao = _area(inner, theta), _area(outer, theta)
    quality_flags = [value for value in row["flags"] if value != "sampling_center_adjusted_within_initial_lumen"]
    row.update(status="needs_review" if quality_flags else "estimated", quality="needs_review" if quality_flags else "estimated",
               inner_radii_mm=inner.tolist(), outer_radii_mm=outer.tolist(),
               inner_points_mm=(center + inner[:, None] * prepared["directions"]).tolist(),
               outer_points_mm=(center + outer[:, None] * prepared["directions"]).tolist(),
               inner_radius_mm=float(np.mean(inner)), outer_radius_mm=float(np.mean(outer)),
               graph_block_id=block_id,
               solver={"status": "ok", "method": diagnostics["solver"],
                       "couples_angles": diagnostics["couples_angles"], "couples_sections": diagnostics["couples_sections"],
                       "exact_for_quantized_retained_graph": True},
               support={"inner": inner_support.tolist(), "outer": outer_support.tolist(),
                        "inner_fraction": float(np.mean(inner_support)), "outer_fraction": float(np.mean(outer_support))},
               coverage={"inner_closed": True, "outer_closed": True, "inner_fraction": 1., "outer_fraction": 1.},
               edge_evidence=[{"inner_contrast_hu": float(inner_contrast[i]), "outer_contrast_hu": float(outer_contrast[i]),
                               "noise_hu": float(noise[i]), "artery_contact": bool(contact[i]),
                               "search_bound_hit": bool(bound[i]),
                               "ct_bright_tissue_enclosed": bool(enclosed_bright[i])} for i in angles])
    row["metrics"].update(LA_contour_mm2=ai, TA_contour_mm2=ao, WA_contour_mm2=ao - ai,
                          WAP_contour_pct=100 * (ao - ai) / ao, r_lumen_area_mm=math.sqrt(ai / np.pi),
                          r_outer_area_mm=math.sqrt(ao / np.pi), radial_wall_separation_mm=float(np.mean(outer - inner)),
                          area_equivalent_wall_thickness_mm=math.sqrt(ao / np.pi) - math.sqrt(ai / np.pi))
    row["local_cost_diagnostic"] = {
        "selected_unary": float(prepared["inner"][angles, labels[0]].sum() + prepared["outer"][angles, labels[1]].sum()),
        "unconstrained_column_minimum": float(prepared["inner"].min(axis=1).sum() + prepared["outer"].min(axis=1).sum()),
        "interpretation": "Energy diagnostic, not a calibrated confidence or anatomical accuracy probability"}


def estimate_branch(ct, lumen_mask, affine_mm, samples, config=SurfaceConfig(), artery_mask=None):
    """Estimate paired surfaces on every supplied branch-local physical sample.

    Invalid sections remain explicit and break longitudinal coupling. Large
    contiguous segments use bounded-memory blocks sharing one fixed previous
    ring; diagnostics disclose that these are conditional, not a global branch
    optimum. No sections are dropped, no masks changed, and weak/contact arcs
    receive explicit review flags.
    """
    started = time.monotonic()
    ct, mask = np.asarray(ct), np.asarray(lumen_mask)
    affine = np.asarray(affine_mm, float)
    if ct.ndim != 3 or ct.shape != mask.shape or affine.shape != (4, 4) or not np.all(np.isfinite(affine)):
        raise ValueError("Aligned 3-D CT/lumen mask and finite 4-by-4 affine_mm are required")
    if artery_mask is not None and np.shape(artery_mask) != ct.shape:
        raise ValueError("Artery mask must use the same CT grid")
    if not samples:
        return {"sections": [], "diagnostics": {"method": METHOD, "sections_requested": 0, "sections_returned": 0}}
    identities = {(str(item.get("component_id")), str(item.get("branch_id"))) for item in samples}
    if len(identities) > 1:
        raise ValueError("estimate_branch cannot couple different branch identities")
    positions = np.array([float(item.get("s_mm", i * config.step_mm)) for i, item in enumerate(samples)])
    if np.any(~np.isfinite(positions)) or np.any(np.diff(positions) <= 0):
        raise ValueError("Samples must be ordered at strictly increasing physical s_mm")
    centers = np.asarray([item["center_mm"] for item in samples], float)
    if centers.shape != (len(samples), 3) or not np.all(np.isfinite(centers)):
        raise ValueError("Sample centers must be finite 3-D physical coordinates")
    frames, invalid, valid_runs, current_run = [None] * len(samples), {}, [], []
    for index, item in enumerate(samples):
        tangent = np.asarray(item.get("tangent_mm"), float)
        reason = None
        if item.get("status") in ("unusable", "unavailable", "excluded"):
            reason = "upstream_unavailable_sample"
        elif tangent.shape != (3,) or not np.all(np.isfinite(tangent)) or np.linalg.norm(tangent) <= 0:
            reason = "unavailable_local_tangent"
        elif (item.get("e1") is None) != (item.get("e2") is None):
            reason = "incomplete_plane_frame"
        elif item.get("e1") is not None:
            basis = np.asarray([item["e1"], item["e2"], item["tangent_mm"]], float)
            if (basis.shape != (3, 3) or not np.all(np.isfinite(basis))
                    or not np.allclose(basis @ basis.T, np.eye(3), atol=1e-5) or np.linalg.det(basis) < .99):
                reason = "invalid_plane_frame"
        if reason:
            invalid[index] = reason
            if current_run:
                valid_runs.append(current_run)
                current_run = []
        else:
            current_run.append(index)
    if current_run:
        valid_runs.append(current_run)
    for run in valid_runs:
        transported = parallel_transport_frames([samples[i]["tangent_mm"] for i in run])
        for index, frame in zip(run, transported):
            item = samples[index]
            if item.get("e1") is not None:
                frame.update(e1=list(item["e1"]), e2=list(item["e2"]), tangent_mm=list(item["tangent_mm"]))
            frames[index] = frame
    inverse = np.linalg.inv(affine)
    rows, prepared = [], []
    for index, (item, frame, position) in enumerate(zip(samples, frames, positions)):
        sample = {**item, "s_mm": float(position), "section_id": item.get("section_id") or f"section_{index:06d}"}
        if index in invalid:
            row, data = _empty_section(sample, frame, config), None
            row["flags"].append(invalid[index])
        else:
            row, data = _prepare_section(ct, mask, inverse, sample, frame, config, artery_mask)
        rows.append(row)
        prepared.append(data)
    groups, current, breaks = [], [], []
    for index, data in enumerate(prepared):
        reason = None
        if data is None:
            reason = "unavailable_section"
        elif current:
            before = current[-1]
            if positions[index] - positions[before] > config.step_mm * 1.5:
                reason = "sampling_gap"
            elif (np.dot(frames[before]["e1"], frames[index]["e1"]) < .5
                  or np.dot(frames[before]["tangent_mm"], frames[index]["tangent_mm"]) < .5):
                reason = "inconsistent_neighboring_plane_frames"
        if reason:
            if current:
                groups.append(current)
                current = []
            breaks.append({"section_index": index, "reason": reason})
            if data is not None:
                rows[index]["flags"].append(reason)
        if data is not None:
            current.append(index)
    if current:
        groups.append(current)
    blocks = []
    minsep = int(np.ceil(config.min_wall_search_mm / config.radial_step_mm - 1e-8))
    maxsep = int(np.floor(config.max_wall_search_mm / config.radial_step_mm + 1e-8))
    for group_index, group in enumerate(groups):
        # A completely absent outer transition throughout a contiguous branch
        # cannot be rescued by a smoothness prior; retain no invented thickness.
        any_inner_evidence, any_outer_evidence = False, False
        for i in group:
            data = prepared[i]
            grid = np.arange(data["gradient"].shape[1])[None]
            domain = ((grid >= data["lower"][1, :, None]) & (grid <= data["upper"][1, :, None])
                      & data["finite"])
            evidence = -data["gradient"] * config.contrast_window_mm > config.min_contrast_snr * data["noise"][:, None]
            any_outer_evidence |= bool(np.any(domain & evidence))
            inner_domain = ((grid >= data["lower"][0, :, None]) & (grid <= data["upper"][0, :, None])
                            & data["finite"])
            inner_evidence = data["gradient"] * config.contrast_window_mm > config.min_contrast_snr * data["noise"][:, None]
            any_inner_evidence |= bool(np.any(inner_domain & inner_evidence))
        if not any_inner_evidence or not any_outer_evidence:
            for i in group:
                if not any_inner_evidence:
                    rows[i]["flags"].append("no_inner_transition_evidence_in_contiguous_branch")
                if not any_outer_evidence:
                    rows[i]["flags"].append("no_outer_transition_evidence_in_contiguous_branch")
            continue
        nl = max(int(np.max(prepared[i]["upper"])) + 1 for i in group)
        block_length = int(config.max_graph_nodes // (2 * config.n_angles * (nl - 1)))
        if block_length < 2 and len(group) > 1:
            raise ValueError("Graph budget cannot hold two adjacent rings; increase max_graph_nodes")
        block_length = max(1, block_length)
        offset, previous = 0, None
        while offset < len(group):
            first = offset - 1 if previous is not None else offset
            indices = group[first:min(len(group), first + block_length)]
            inner = np.stack([prepared[i]["inner"][:, :nl] for i in indices])
            outer = np.stack([prepared[i]["outer"][:, :nl] for i in indices])
            lower = np.stack([prepared[i]["lower"] for i in indices], axis=1)
            upper = np.stack([prepared[i]["upper"] for i in indices], axis=1)
            solution = solve_ordered_surfaces(inner, outer, lower=lower, upper=upper,
                radial_step_mm=config.radial_step_mm, angular_weight=config.angular_weight,
                longitudinal_weight=config.longitudinal_weight,
                axial_spacing_mm=np.diff(positions[indices]), min_separation_labels=minsep,
                max_separation_labels=min(maxsep, nl - 1), capacity_scale=config.capacity_scale,
                fixed_first=previous)
            block_id = len(blocks)
            diagnostic = {**solution["diagnostics"], "block_id": block_id, "group_index": group_index,
                          "section_indices": indices, "s_start_mm": float(positions[indices[0]]),
                          "s_end_mm": float(positions[indices[-1]])}
            blocks.append(diagnostic)
            selected = solution["labels"]
            for local, index in enumerate(indices):
                if previous is not None and local == 0:
                    continue
                _finish_section(rows[index], prepared[index], selected[:, local], config, block_id, diagnostic)
                if previous is not None:
                    rows[index]["flags"].append("conditional_graph_block_estimate")
            previous = selected[:, -1]
            offset = first + len(indices)
    for before, after in zip(rows, rows[1:]):
        if before["outer_radii_mm"] is not None and after["outer_radii_mm"] is not None:
            change = max(np.max(np.abs(np.asarray(after[key]) - before[key])) for key in ("inner_radii_mm", "outer_radii_mm"))
            if change > config.max_axial_radius_change_mm:
                after["flags"].append("abrupt_axial_surface_change")
                after["status"] = after["quality"] = "needs_review"
            projections = [(np.asarray(after["outer_points_mm"]) - before["center_mm"]) @ before["tangent_mm"],
                           (np.asarray(before["outer_points_mm"]) - after["center_mm"]) @ after["tangent_mm"]]
            if any(np.min(value) <= 1e-6 and np.max(value) >= -1e-6 for value in projections):
                after["flags"].append("adjacent_measurement_planes_intersect")
                after["status"] = after["quality"] = "needs_review"
    return {"sections": rows, "diagnostics": {
        "method": METHOD, "configuration": asdict(config), "source_reference": SOURCE_REFERENCE,
        "sections_requested": len(samples), "sections_returned": len(rows),
        "paired_sections": sum(row["outer_radii_mm"] is not None for row in rows),
        "blocks": blocks, "coupling_breaks": breaks, "elapsed_seconds": time.monotonic() - started,
        "search_domain": "Dense radial labels around the saved lumen; no top-3 peak pruning",
        "method_differences": ["Branch-local straight radial columns, not global non-intersecting flow lines",
                               "Discrete labels without 2019 min-marginal refinement",
                               "Long branches use conditional bounded-memory blocks with one fixed overlap ring",
                               "Independent CT costs and uncalibrated engineering parameters",
                               "Robust CT bright-tissue enclosure penalty applies only to outer-surface unary costs"],
        "radius_estimator": "mean_complete_ray_radius", "validation": "not_clinically_validated"}}


def build_estimated_surfaces(sections, max_gap_mm=.75):
    """Export open paired meshes without changing estimate/review identities.

    Rejects gaps, crossing planes, abrupt radii and branch changes. Only
    temporary geometry records receive measured status; exported metadata
    explicitly describe unvalidated estimates.
    """
    copies = [{**row, "section_id": row.get("section_id") or f"section_{index:06d}",
               "status": "measured" if row.get("status") in ("estimated", "needs_review")
               and row.get("inner_points_mm") and row.get("outer_points_mm") else "unusable"}
              for index, row in enumerate(sections)]
    output = build_local_surfaces(copies, max_gap_mm=max_gap_mm)
    by_id = {row["section_id"]: original for row, original in zip(copies, sections)}
    for group in output["groups"]:
        original = [by_id[identifier] for identifier in group["section_ids"]]
        group.update(status="estimated", review_status="not_reviewed",
                     section_quality=[row.get("quality", row.get("status")) for row in original],
                     anatomical_validation="not_performed")
        if all(row.get("initial_points_mm") for row in original):
            group["initial_lumen"] = {"vertices_mm": [point for row in original for point in row["initial_points_mm"]],
                                      "faces": group["inner"]["faces"], "open_ends": True}
    output.update(scope="estimated_local_branch_interiors_including_needs_review",
                  method=METHOD, validation="not_clinically_validated", outer_surface="estimated_lumen_plus_wall_enclosure")
    return output
