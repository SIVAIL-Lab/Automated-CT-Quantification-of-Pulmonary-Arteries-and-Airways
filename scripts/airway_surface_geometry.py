"""Physical frames, image samples and open meshes for coupled airway surfaces."""
from __future__ import annotations

import numpy as np
from scipy.ndimage import map_coordinates
from centerline_cross_sections import orthogonal_basis

METRICS = ("LA_contour_mm2", "TA_contour_mm2", "WA_contour_mm2", "WAP_contour_pct",
           "r_lumen_area_mm", "r_outer_area_mm", "radial_wall_separation_mm",
           "area_equivalent_wall_thickness_mm")

def parallel_transport_frames(tangents):
    """Transport angular correspondence along a single branch without twisting."""
    frames = []
    previous_tangent = previous_e1 = None
    for vector in tangents:
        tangent = np.asarray(vector, float)
        if tangent.shape != (3,) or not np.all(np.isfinite(tangent)) or np.linalg.norm(tangent) <= 0:
            raise ValueError("Tangents must be finite nonzero vectors")
        tangent = tangent / np.linalg.norm(tangent)
        if previous_tangent is None:
            e1, _ = orthogonal_basis(tangent)
        else:
            if tangent @ previous_tangent < 0:
                tangent = -tangent  # PCA sign does not define a different section.
            cross = np.cross(previous_tangent, tangent)
            sine, cosine = np.linalg.norm(cross), float(previous_tangent @ tangent)
            e1 = previous_e1.copy()
            if sine > 1e-10:
                axis = cross / sine
                e1 = (e1 * cosine + np.cross(axis, e1) * sine
                      + axis * (axis @ e1) * (1 - cosine))
            e1 = e1 - tangent * (e1 @ tangent)
            e1 /= np.linalg.norm(e1)
        e2 = np.cross(tangent, e1)
        frames.append({"tangent_mm": tangent.tolist(), "e1": e1.tolist(), "e2": e2.tolist()})
        previous_tangent, previous_e1 = tangent, e1
    return frames

def _sample(data, inverse, world, *, order=1, outside=np.nan):
    ijk = world @ inverse[:3, :3].T + inverse[:3, 3]
    return map_coordinates(data, np.moveaxis(ijk, -1, 0), order=order, mode="constant",
                           cval=outside, prefilter=False, output=np.float64)

def _area(radii, theta):
    points = np.c_[np.cos(theta) * radii, np.sin(theta) * radii]
    return float(abs(np.sum(points[:, 0] * np.roll(points[:, 1], -1)
                            - points[:, 1] * np.roll(points[:, 0], -1))) / 2)

def build_local_surfaces(sections, max_gap_mm=.75):
    """Open branch-interior meshes; invalid samples break rather than fill gaps."""
    if not np.isfinite(max_gap_mm) or max_gap_mm <= 0:
        raise ValueError("Mesh gap limit must be finite and positive")
    groups, current, breaks = [], [], []

    def finish():
        if len(current) < 2:
            current.clear()
            return
        n = len(current[0]["inner_points_mm"])
        faces = []
        for index in range(len(current) - 1):
            for angle in range(n):
                a, b = index * n + angle, index * n + (angle + 1) % n
                c, d = a + n, b + n
                faces.extend([[a, b, c], [b, d, c]])
        row = {"component_id": current[0].get("component_id"), "branch_id": current[0].get("branch_id"),
               "site_id": current[0].get("site_id"),
               "section_ids": [item.get("section_id", item.get("site_id")) for item in current],
               "s_start_mm": current[0].get("s_mm"), "s_end_mm": current[-1].get("s_mm")}
        for kind in ("inner", "outer"):
            row[kind] = {"vertices_mm": [point for item in current for point in item[f"{kind}_points_mm"]],
                         "faces": faces, "open_ends": True}
        groups.append(row)
        current.clear()

    for section in sections:
        valid = (section.get("status") == "measured" and section.get("inner_points_mm") and section.get("outer_points_mm")
                 and section.get("branch_id") is not None and not section.get("junction_excluded", False))
        if not valid:
            breaks.append({"section_id": section.get("section_id", section.get("site_id")), "reason": "invalid_or_unmeasured_section"})
            finish()
            continue
        if current:
            before = current[-1]
            same_branch = (before.get("component_id"), before.get("branch_id")) == (section.get("component_id"), section.get("branch_id"))
            gap = abs(float(section["s_mm"]) - float(before["s_mm"])) if "s_mm" in section and "s_mm" in before else float(np.linalg.norm(np.array(section["center_mm"]) - before["center_mm"]))
            frame_agreement = np.dot(section["e1"], before["e1"]) > .5 and np.dot(section["tangent_mm"], before["tangent_mm"]) > .5
            matching_angles = (len(section["inner_radii_mm"]) == len(before["inner_radii_mm"])
                               and len(section["outer_radii_mm"]) == len(before["outer_radii_mm"])
                               and np.allclose(section["theta_rad"], before["theta_rad"]))
            radius_change = max(np.max(np.abs(np.array(section[key]) - before[key])) for key in ("inner_radii_mm", "outer_radii_mm")) if matching_angles else np.inf
            radius_limit = min(float(item.get("config", {}).get("max_axial_radius_change_mm", 1.5)) for item in (section, before))
            # If a ring straddles its neighbour's plane, the local tube bends
            # too sharply to stitch safely with this simple ruled surface.
            projections = [
                (np.asarray(section["outer_points_mm"]) - before["center_mm"]) @ before["tangent_mm"],
                (np.asarray(before["outer_points_mm"]) - section["center_mm"]) @ section["tangent_mm"]]
            planes_cross = any(float(np.min(value)) <= 1e-6 and float(np.max(value)) >= -1e-6 for value in projections)
            displacement = np.asarray(section["center_mm"]) - before["center_mm"]
            direction_reversal = (displacement @ before["tangent_mm"]) * (displacement @ section["tangent_mm"]) <= 0
            reason = ("different_branch" if not same_branch else "coverage_gap" if not 0 < gap <= max_gap_mm
                      else "inconsistent_angular_frame" if not frame_agreement or not matching_angles
                      else "axial_radius_change" if radius_change > radius_limit
                      else "adjacent_planes_intersect_or_reverse" if planes_cross or direction_reversal else None)
            if reason:
                breaks.append({"section_id": section.get("section_id", section.get("site_id")), "reason": reason})
                finish()
        current.append(section)
    finish()
    return {"groups": groups, "open_ends": True, "caps": False, "breaks": breaks,
            "scope": "accepted_local_branch_interiors_only", "unmeasured_regions": "unknown"}
