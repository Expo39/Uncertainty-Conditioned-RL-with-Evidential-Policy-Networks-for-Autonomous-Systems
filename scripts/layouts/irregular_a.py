"""
@file irregular_a.py
@brief Irregular five-sided parking lot floor plan (OOD, 35x35 m base).

Rear-right corner cut diagonally (8x8 m triangle removed). Held out from
training to evaluate out-of-distribution generalisation.

Bay zones (intentionally different from rectangle and trapezoid):
  - Perpendicular row along the left wall (x=0), facing +X (yaw=0).
  - Angled bays: 3 on the diagonal cut wall + 2 on the rear wall (x=depth).
  - Parallel row along the bottom wall (y=0), facing +Y (yaw=90).

Used as an OOD evaluation layout (ood=True).
"""

import math
from typing import Any, Dict

from scripts.layouts.common import (
    BAY_DIMS,
    BAYS_PER_TYPE,
    ang_offset_from_wall,
    ang_x_margin,
    angled_bays_along_wall,
    par_centre_from_wall,
    parallel_bays,
    perp_centre_x_from_wall,
    perpendicular_bays,
    validate_bays_in_polygon,
    warn_narrow_corridors,
)

# World-frame origin used in multi-layout generation (Town05_Opt flat area).
ORIGIN_X = 100.0
ORIGIN_Y = 0.0
ORIGIN_Z = 0.3
HEADING_DEG = 0.0
OOD = True


def generate() -> Dict[str, Any]:
    """
    @brief Build the irregular_a layout in local frame.
    @return Layout dict with keys: corners, bays, spawn, patrol_waypoints,
            pedestrian_zones.
    """
    width = 35.0
    depth = 35.0
    cut = 8.0

    # Cut is at rear-top-right: from (depth, width-cut) to (depth-cut, width).
    corners = [
        {"x": 0.0, "y": 0.0},
        {"x": depth, "y": 0.0},
        {"x": depth, "y": width - cut},
        {"x": depth - cut, "y": width},
        {"x": 0.0, "y": width},
    ]

    dims_perp = BAY_DIMS["perpendicular"]
    dims_ang = BAY_DIMS["angled"]
    dims_par = BAY_DIMS["parallel"]

    # Perpendicular row: left wall (x=0), nose toward +X (yaw=0).
    perp_cx = perp_centre_x_from_wall(0.0, dims_perp["depth"], inward=False)
    perp_y_start = width / 2.0 - (BAYS_PER_TYPE * dims_perp["width"]) / 2.0
    perp_bays = perpendicular_bays(BAYS_PER_TYPE, perp_cx, perp_y_start, 0.0)

    # Angled bays: split across two walls.
    # Group A (3 bays): diagonal cut wall from (depth, width-cut) to (depth-cut, width).
    cut_dx = (depth - cut) - depth  # = -cut
    cut_dy = width - (width - cut)  # = cut
    cut_len = math.hypot(cut_dx, cut_dy)
    wdx_cut = cut_dx / cut_len
    wdy_cut = cut_dy / cut_len
    nx_cut = -wdy_cut
    ny_cut = wdx_cut
    wall_normal_deg_cut = math.degrees(math.atan2(ny_cut, nx_cut))
    ang_yaw_cut = (wall_normal_deg_cut + 45.0) % 360.0

    ang_offset = ang_offset_from_wall(dims_ang["depth"], dims_ang["width"])
    ang_start = ang_x_margin(dims_ang["depth"], dims_ang["width"])
    ang_bays_cut = angled_bays_along_wall(
        3,
        wall_x0=depth,
        wall_y0=width - cut,
        wall_dx=wdx_cut,
        wall_dy=wdy_cut,
        wall_len=cut_len,
        offset_from_wall=ang_offset,
        start_along_wall=ang_start,
        facing_yaw_deg=ang_yaw_cut,
    )

    # Group B (2 bays): rear wall (x=depth), lower section, facing left (yaw=225).
    ang_bays_rear = angled_bays_along_wall(
        2,
        wall_x0=depth,
        wall_y0=1.0,
        wall_dx=0.0,
        wall_dy=1.0,
        wall_len=width - cut - 2.0,
        offset_from_wall=ang_offset,
        start_along_wall=ang_start,
        facing_yaw_deg=180.0 + 45.0,
    )

    ang_bays = ang_bays_cut + ang_bays_rear

    # Parallel row: bottom wall (y=0), nose toward +Y (yaw=90).
    par_cy = par_centre_from_wall(0.0, dims_par["depth"], inward=False)
    par_x_start = dims_par["width"] / 2.0 + 0.5
    par_bays = parallel_bays(BAYS_PER_TYPE, par_cy, par_x_start, 90.0)

    all_bays = perp_bays + ang_bays + par_bays

    validate_bays_in_polygon(all_bays, corners, "irregular_a")
    warn_narrow_corridors(all_bays, "irregular_a")

    spawn = {"x": 1.5, "y": width / 2.0, "yaw_deg": 0.0}

    # Patrol path: loop around the aisle in front of the perp row.
    perp_nose_x_irr = perp_cx + dims_perp["depth"] / 2.0
    perp_aisle_cx_irr = perp_nose_x_irr + dims_perp["aisle"] / 2.0
    y_bot = par_cy + dims_par["depth"] / 2.0 + 1.0
    y_top = width - cut - 2.0
    patrol = [
        {"x": perp_aisle_cx_irr, "y": y_bot},
        {"x": perp_aisle_cx_irr, "y": y_top},
        {"x": depth - cut - 2.0, "y": y_top},
        {"x": depth - 2.0, "y": y_bot},
    ]

    PED_STRIP = 2.0
    ped_zones = [
        # Strip against perp row nose face (just to the right of perp bays, toward +X)
        {
            "x_min": perp_nose_x_irr,
            "x_max": perp_nose_x_irr + PED_STRIP,
            "y_min": perp_y_start,
            "y_max": perp_y_start + BAYS_PER_TYPE * dims_perp["width"],
        },
        # Strip against parallel bay nose face (just above the parallel row in Y)
        {
            "x_min": par_x_start - dims_par["width"] / 2.0,
            "x_max": par_x_start + BAYS_PER_TYPE * dims_par["width"],
            "y_min": par_cy + dims_par["depth"] / 2.0,
            "y_max": par_cy + dims_par["depth"] / 2.0 + PED_STRIP,
        },
    ]

    return {
        "corners": corners,
        "bays": all_bays,
        "spawn": spawn,
        "patrol_waypoints": patrol,
        "pedestrian_zones": ped_zones,
    }
