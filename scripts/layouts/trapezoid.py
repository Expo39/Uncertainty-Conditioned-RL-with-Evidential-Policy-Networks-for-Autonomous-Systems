"""
@file trapezoid.py
@brief Trapezoid parking lot floor plan (front=48, rear=30, depth=44 m).

Wider at the entrance end, narrower at the rear. The non-parallel sides give
a subtly different LiDAR wall profile compared to the rectangle layout.

Bay zones:
  - Two back-to-back perpendicular rows in the centre (8 bays per row).
  - Angled row following the diagonal bottom wall, 7 bays at 45 deg to the wall.
  - Parallel bays: 3 along the sloped top wall + 3 along the right wall.

Used as a training layout (ood=False).
"""

import math
from typing import Any, Dict

from scripts.layouts.common import (
    BAY_DIMS,
    BAYS_PER_TYPE,
    ang_offset_from_wall,
    ang_x_margin,
    angled_bays_along_wall,
    validate_bays_in_polygon,
    warn_narrow_corridors,
)

# World-frame origin used in multi-layout generation (Town05_Opt flat area).
ORIGIN_X = 0.0
ORIGIN_Y = -90.0
ORIGIN_Z = 0.3
HEADING_DEG = 0.0
OOD = False


def generate() -> Dict[str, Any]:
    """
    @brief Build the trapezoid layout in local frame.
    @return Layout dict with keys: corners, bays, spawn, extra_spawns,
            patrol_waypoints, pedestrian_zones.
    """
    width_front = 48.0
    width_rear = 30.0
    depth = 44.0

    y_offset = (width_front - width_rear) / 2.0  # taper per side

    corners = [
        {"x": 0.0, "y": 0.0},
        {"x": depth, "y": y_offset},
        {"x": depth, "y": width_front - y_offset},
        {"x": 0.0, "y": width_front},
    ]

    dims_perp = BAY_DIMS["perpendicular"]
    dims_ang = BAY_DIMS["angled"]
    dims_par = BAY_DIMS["parallel"]

    # Perpendicular bays: two back-to-back rows in the centre of the lot.
    # Row A: yaw=90  (nose toward +Y), Row B: yaw=270 (nose toward -Y).
    # 8 bays per row: base 5 + 2 left + 1 right.
    PERP_BAYS_PER_ROW = BAYS_PER_TYPE + 3
    perp_mid_y = 25.5
    perp_row_a_cy = perp_mid_y - dims_perp["aisle"] / 2.0 - dims_perp["depth"] / 2.0
    perp_row_b_cy = perp_mid_y + dims_perp["aisle"] / 2.0 + dims_perp["depth"] / 2.0
    perp_cx_start = (
        depth / 2.0
        - (PERP_BAYS_PER_ROW * dims_perp["width"]) / 2.0
        + dims_perp["width"] / 2.0
        - 1.5
        - 3.63
    )
    perp_bays = []
    for i in range(PERP_BAYS_PER_ROW):
        cx = perp_cx_start + i * dims_perp["width"]
        perp_bays.append(
            {
                "bay_type": "perpendicular",
                "local_x": cx,
                "local_y": perp_row_a_cy,
                "local_yaw_deg": 90.0,
                "width": dims_perp["width"],
                "depth": dims_perp["depth"],
            }
        )
        perp_bays.append(
            {
                "bay_type": "perpendicular",
                "local_x": cx,
                "local_y": perp_row_b_cy,
                "local_yaw_deg": 270.0,
                "width": dims_perp["width"],
                "depth": dims_perp["depth"],
            }
        )

    _WALL_GAP = 0.5  # minimum clearance between bay back face and perimeter wall/cones

    # Angled row: follow the diagonal bottom wall from (0,0) to (depth, y_offset).
    wall_len = math.hypot(depth, y_offset)
    wdx = depth / wall_len
    wdy = y_offset / wall_len
    ang_yaw = (math.degrees(math.atan2(wdx, -wdy)) + 45.0) % 360.0

    ang_offset = ang_offset_from_wall(dims_ang["depth"], dims_ang["width"]) + _WALL_GAP
    ang_start = ang_x_margin(dims_ang["depth"], dims_ang["width"]) + 6.0
    ang_bays = angled_bays_along_wall(
        7,
        wall_x0=0.0,
        wall_y0=0.0,
        wall_dx=wdx,
        wall_dy=wdy,
        wall_len=wall_len,
        offset_from_wall=ang_offset,
        start_along_wall=ang_start,
        facing_yaw_deg=ang_yaw,
    )

    # Parallel bays: short-side adjacent (8 m along wall, 2.5 m deep from wall).
    # Group A: 3 bays along the sloped top wall.
    top_wall_len = math.hypot(depth, y_offset)
    top_wdx = depth / top_wall_len
    top_wdy = -y_offset / top_wall_len
    # Inward normal (CW 90 from wall dir, pointing into lot).
    top_nx = top_wdy
    top_ny = -top_wdx
    top_par_yaw = math.degrees(math.atan2(top_wdy, top_wdx))
    par_top_spacing = dims_par["depth"]
    par_top_along_start = 9.0 + dims_par["depth"] / 2.0
    par_top_normal_offset = dims_par["width"] / 2.0 + _WALL_GAP
    par_bays_top = []
    for i in range(3):
        along = par_top_along_start + i * par_top_spacing
        wx = 0.0 + top_wdx * along
        wy = width_front + top_wdy * along
        cx = wx + top_nx * par_top_normal_offset
        cy = wy + top_ny * par_top_normal_offset
        par_bays_top.append(
            {
                "bay_type": "parallel",
                "local_x": cx,
                "local_y": cy,
                "local_yaw_deg": top_par_yaw,
                "width": dims_par["width"],
                "depth": dims_par["depth"],
            }
        )

    # Group B: 3 bays on right wall (x=depth), yaw=90 (nose +Y, depth=8 m along Y).
    par_cx_right = depth - dims_par["width"] / 2.0 - _WALL_GAP
    par_right_y_start = y_offset + 3.0
    par_bays_right = []
    for i in range(3):
        par_bays_right.append(
            {
                "bay_type": "parallel",
                "local_x": par_cx_right,
                "local_y": par_right_y_start
                + dims_par["depth"] / 2.0
                + i * dims_par["depth"],
                "local_yaw_deg": 90.0,
                "width": dims_par["width"],
                "depth": dims_par["depth"],
            }
        )

    par_bays = par_bays_top + par_bays_right
    all_bays = perp_bays + ang_bays + par_bays

    validate_bays_in_polygon(all_bays, corners, "trapezoid")
    warn_narrow_corridors(all_bays, "trapezoid")

    # Spawn 1: entrance in left wall (x=0).
    spawn = {"x": 0.0, "y": width_front / 2.0, "yaw_deg": 0.0}

    # Spawn 2: entrance in the diagonal bottom wall near x=38.
    _bw_len = math.hypot(depth, y_offset)
    _bwdx = depth / _bw_len
    _bwdy = y_offset / _bw_len
    _spawn2_yaw = math.degrees(math.atan2(_bwdx, -_bwdy))
    spawn2 = {
        "x": 38.0,
        "y": y_offset * (38.0 / depth),
        "yaw_deg": round(_spawn2_yaw, 1),
    }

    # Patrol path: rectangular loop through all three horizontal aisles.
    PED_STRIP = 3.0
    perp_cluster_x_min = perp_cx_start - dims_perp["width"] / 2.0
    perp_cluster_x_max = perp_cx_start + (PERP_BAYS_PER_ROW - 0.5) * dims_perp["width"]
    par_right_inner_x = par_cx_right - dims_par["width"] / 2.0
    aisle1_cy = perp_row_a_cy - dims_perp["depth"] / 2.0 - dims_perp["aisle"] / 2.0
    aisle3_cy = perp_row_b_cy + dims_perp["depth"] / 2.0 + dims_perp["aisle"] / 2.0
    x_enter = (PED_STRIP + perp_cluster_x_min) / 2.0
    x_exit = (perp_cluster_x_max + par_right_inner_x) / 2.0
    patrol = [
        {"x": x_enter, "y": aisle1_cy - 4},
        {"x": x_exit, "y": aisle1_cy},
        {"x": x_exit, "y": aisle3_cy},
        {"x": x_enter, "y": aisle3_cy + 4},
    ]

    # Pedestrian zones: 3 m strips along aisle faces and right-wall parallel bays.
    # _PED_MARGIN insets every edge that touches a bay face so pedestrians
    # cannot overlap a parked car's footprint.
    _PED_MARGIN = 0.5
    par_bay_x_min = par_cx_right - dims_par["width"] / 2.0
    ped_zones = [
        # Aisle 1: strip against row A nose face (below row A in Y)
        {
            "x_min": perp_cluster_x_min + _PED_MARGIN,
            "x_max": perp_cluster_x_max - _PED_MARGIN,
            "y_min": perp_row_a_cy - dims_perp["depth"] / 2.0 - PED_STRIP,
            "y_max": perp_row_a_cy - dims_perp["depth"] / 2.0 - _PED_MARGIN,
        },
        # Aisle 2: full back-to-back gap between row A back face and row B back face.
        {
            "x_min": perp_cluster_x_min + _PED_MARGIN,
            "x_max": perp_cluster_x_max - _PED_MARGIN,
            "y_min": perp_row_a_cy + dims_perp["depth"] / 2.0 + _PED_MARGIN,
            "y_max": perp_row_b_cy - dims_perp["depth"] / 2.0 - _PED_MARGIN,
        },
        # Aisle 3: strip against row B nose face (above row B in Y)
        {
            "x_min": perp_cluster_x_min + _PED_MARGIN,
            "x_max": perp_cluster_x_max - _PED_MARGIN,
            "y_min": perp_row_b_cy + dims_perp["depth"] / 2.0 + _PED_MARGIN,
            "y_max": perp_row_b_cy + dims_perp["depth"] / 2.0 + PED_STRIP,
        },
        # Aisle 4: vertical strip against left face of right-wall parallel bays
        {
            "x_min": par_bay_x_min - PED_STRIP,
            "x_max": par_bay_x_min - _PED_MARGIN,
            "y_min": par_right_y_start + _PED_MARGIN,
            "y_max": par_right_y_start + 3 * dims_par["depth"] - _PED_MARGIN,
        },
    ]

    return {
        "corners": corners,
        "bays": all_bays,
        "spawn": spawn,
        "extra_spawns": [spawn2],
        "patrol_waypoints": patrol,
        "pedestrian_zones": ped_zones,
    }
