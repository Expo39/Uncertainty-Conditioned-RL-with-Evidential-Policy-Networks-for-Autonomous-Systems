"""
@file trapezoid.py
@brief Trapezoid parking lot floor plan (front=48, rear=30, depth=44 m).

Wider at the entrance end (y=0..48) and narrower at the rear (y=9..39),
giving non-parallel top and bottom walls. The tapered profile produces a
subtly different LiDAR/sensor wall signature compared to the rectangle layout,
which helps the agent generalise to non-rectangular geometry.

Bay zones:
  - Centre cluster: 8 back-to-back perpendicular bays per row (yaw=90/270),
    left-shifted slightly from lot centre so the cluster clears the right-wall
    parallel group. Bay count = BAYS_PER_TYPE + 3 (geometry-driven).
  - Angled row: 7 bays following the diagonal bottom wall (P0->P1) at 45 deg
    to the wall slope, starting 6 m clear of the left corner.
  - Top-wall parallel group: 3 bays along the sloped top wall (P3->P2),
    starting 9 m from the left corner.
  - Right-wall parallel group: 3 bays against the right wall (x=44),
    nose facing +Y (yaw=90).

Two spawn transforms (3 m inside the perimeter along each heading):
  - Primary   (S1): left wall entry (x=3, y=24), facing +X into the lot.
  - Secondary (S2): diagonal bottom wall near x=38, 3 m inward perpendicular
    to the wall slope.

Used as a training layout (OOD=False).
"""

import math
from typing import Any, Dict

from scripts.layouts.common import (
    _PED_MARGIN,
    _WALL_GAP,
    BAY_DIMS,
    BAYS_PER_TYPE,
    PED_STRIP,
    ang_offset_from_wall,
    ang_x_margin,
    angled_bays_along_wall,
    validate_bays_in_polygon,
    warn_narrow_corridors,
)

# World-frame origin chosen so the primary spawn lands at CARLA (0,0),
# ensuring consistent coordinate frame alignment at the origin.
# to_world_frame() negates Y for CARLA's left-handed frame:
#   carla_y = -(local_y + (-ORIGIN_Y))  =>  carla_y = -(30.0 + (-30.0)) = 0.
# spawn_local = (-2.0, 30.0) -> CARLA (0.0, 0.0).
ORIGIN_X = 2.0
ORIGIN_Y = 30.0
ORIGIN_Z = 0.3
HEADING_DEG = 0.0
OOD = False


def generate() -> Dict[str, Any]:
    """
    @brief Build the trapezoid layout in local frame.
    @return Layout dict with keys: corners, bays, spawn, extra_spawns,
            patrol_waypoints, pedestrian_zones.
    """
    width_front = 60.0
    width_rear = 40.0
    depth = 44.0

    # y_offset: how much each sloped wall tapers inward from front to rear.
    y_offset = (width_front - width_rear) / 2.0  # 9.0

    corners = [
        {"x": -5.0, "y": 0.0},  # P0 bottom-left (front)
        {"x": depth, "y": y_offset},  # P1 bottom-right (rear)
        {"x": depth, "y": width_front - y_offset},  # P2 top-right (rear)
        {"x": -5.0, "y": width_front},  # P3 top-left (front)
    ]

    dims_perp = BAY_DIMS["perpendicular"]  # width=2.5, depth=5.0
    dims_ang = BAY_DIMS["angled"]  # width=2.5, depth=5.4
    dims_par = BAY_DIMS["parallel"]  # width=2.5, depth=8.0

    # ------------------------------------------------------------------
    # Centre cluster: back-to-back perpendicular rows along X.
    # Row A (lower): yaw=90  (nose +Y). Row B (upper): yaw=270 (nose -Y).
    # PERP_BAYS_PER_ROW = 8: base BAYS_PER_TYPE + 3 to fill the lot width.
    # Rows are offset slightly left from lot centre (_cx_shift) to keep
    # the right corridor clear of the right-wall parallel group.
    # _cx_shift = -1.5 (centre shift) - 3.63 (half an angled-bay pitch).
    # ------------------------------------------------------------------
    PERP_BAYS_PER_ROW = BAYS_PER_TYPE + 3  # 8 bays per row
    _cx_shift = -1.5 - 3.63
    perp_mid_y = width_front / 2.0
    perp_row_a_cy = perp_mid_y - dims_perp["aisle"] / 2.0 - dims_perp["depth"] / 2.0
    perp_row_b_cy = perp_mid_y + dims_perp["aisle"] / 2.0 + dims_perp["depth"] / 2.0
    perp_cx_start = (
        depth / 2.0
        - (PERP_BAYS_PER_ROW * dims_perp["width"]) / 2.0
        + dims_perp["width"] / 2.0
        + _cx_shift
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

    # ------------------------------------------------------------------
    # Angled row: 7 bays along the diagonal bottom wall P0(0,0)->P1(44,9).
    # Wall direction normalised; CCW inward normal points into lot interior.
    # ang_start: skip 6 m from the P0 corner so the first bay clears the
    # entrance gate and the diagonal wall cones.
    # ------------------------------------------------------------------
    # Bottom wall now runs from P0(-5, 0) to P1(depth, y_offset).
    # wall_len and direction recomputed from the extended P0.
    _p0_x = -5.0
    _p0_y = 0.0
    wall_len = math.hypot(depth - _p0_x, y_offset - _p0_y)
    wdx = (depth - _p0_x) / wall_len
    wdy = (y_offset - _p0_y) / wall_len
    ang_yaw = (math.degrees(math.atan2(wdx, -wdy)) + 45.0) % 360.0
    ang_offset = ang_offset_from_wall(dims_ang["depth"], dims_ang["width"]) + _WALL_GAP
    # Start 3 m from the new P0 (was 6 m from old P0=0) to fill the extra 5 m.
    ang_start = ang_x_margin(dims_ang["depth"], dims_ang["width"]) + 3.0
    ang_bays = angled_bays_along_wall(
        9,
        wall_x0=_p0_x,
        wall_y0=_p0_y,
        wall_dx=wdx,
        wall_dy=wdy,
        wall_len=wall_len,
        offset_from_wall=ang_offset,
        start_along_wall=ang_start,
        facing_yaw_deg=ang_yaw,
    )

    # ------------------------------------------------------------------
    # Top-wall parallel group: 3 bays along the sloped top wall P3(-5,48)->P2(44,39).
    # Wall direction derived from actual P3->P2 endpoints so the unit vector and
    # yaw match the real wall slope (dx=49, dy=-9, not the bottom-wall dx=44).
    # Inward normal: CW rotation of wall direction = (wdy, -wdx), pointing into lot.
    # Wall start uses actual P3 (-5, width_front) so par_top_normal_offset places
    # the back face exactly _WALL_GAP from the real wall line.
    # par_top_along_start: first bay centre 9 m + hd along the wall from P3.
    # par_top_spacing: depth + 1 m gap so bays do not touch end-to-end in CARLA.
    # ------------------------------------------------------------------
    _top_wall_dx = depth - (-5.0)  # = 49.0, actual P3->P2 x-span
    _top_wall_dy = -y_offset       # = -9.0, actual P3->P2 y-span
    _top_wall_len = math.hypot(_top_wall_dx, _top_wall_dy)
    top_wdx = _top_wall_dx / _top_wall_len
    top_wdy = _top_wall_dy / _top_wall_len
    top_nx = top_wdy   # CW inward normal x (points into lot)
    top_ny = -top_wdx  # CW inward normal y
    top_par_yaw = math.degrees(math.atan2(top_wdy, top_wdx))
    par_top_along_start = 9.0 + dims_par["depth"] / 2.0
    par_top_spacing = dims_par["depth"]  # end-to-end, matching the right-wall group
    par_top_normal_offset = dims_par["width"] / 2.0 + _WALL_GAP
    par_bays_top = []
    for i in range(4):
        along = par_top_along_start + i * par_top_spacing
        wx = -5.0 + top_wdx * along
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

    # ------------------------------------------------------------------
    # Right-wall parallel group: 3 bays against the right wall (x=depth).
    # yaw=90: depth (8 m) along Y, nose facing +Y. Back against x=depth.
    # par_right_y_start: centred on the rear wall span (y_offset..width_front-y_offset).
    # ------------------------------------------------------------------
    par_cx_right = depth - dims_par["width"] / 2.0 - _WALL_GAP
    _right_wall_span = (width_front - y_offset) - y_offset
    par_right_y_start = y_offset + (_right_wall_span - 3 * dims_par["depth"]) / 2.0
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

    # ------------------------------------------------------------------
    # Spawn transforms (3 m inside the perimeter along each heading).
    # S1: left wall entry, 3 m inward along +X from x=0.
    # S2: diagonal bottom wall near x=38, 3 m inward along the inward normal.
    #     y interpolated along P0(0,0)->P1(depth, y_offset).
    #     Inward normal yaw = atan2(wdx, -wdy) (CCW 90 from wall direction).
    # ------------------------------------------------------------------
    _spawn2_yaw = math.degrees(math.atan2(wdx, -wdy))
    _s2_x = round(38.0 + math.cos(math.radians(_spawn2_yaw)) * 3.0, 1)
    _s2_y = round(
        y_offset * (38.0 / depth) + math.sin(math.radians(_spawn2_yaw)) * 3.0, 1
    )
    spawn = {"x": -2.0, "y": width_front / 2.0, "yaw_deg": 0.0}
    spawn2 = {
        "x": _s2_x,
        "y": _s2_y,
        "yaw_deg": round(_spawn2_yaw, 1),
    }

    # ------------------------------------------------------------------
    # Patrol path: 4-waypoint loop through the lower and upper aisles.
    #
    # Corridor y values:
    #   aisle1_cy: midpoint below row A nose face (centre perp row, faces -Y).
    #   aisle3_cy: midpoint above row B nose face (centre perp row, faces +Y).
    # The x extents are offset by 4 m inside the lot boundary so the patrol
    # vehicle stays clear of the entrance gate (left) and right-wall bays.
    #   x_enter: between left lot boundary and left edge of perp cluster.
    #   x_exit:  between right edge of perp cluster and right-wall parallel bays.
    # ------------------------------------------------------------------
    perp_cluster_x_min = perp_cx_start - dims_perp["width"] / 2.0
    perp_cluster_x_max = perp_cx_start + (PERP_BAYS_PER_ROW - 0.5) * dims_perp["width"]
    par_right_inner_x = par_cx_right - dims_par["width"] / 2.0
    aisle1_cy = perp_row_a_cy - dims_perp["depth"] / 2.0 - dims_perp["aisle"] / 2.0
    aisle3_cy = perp_row_b_cy + dims_perp["depth"] / 2.0 + dims_perp["aisle"] / 2.0
    # Patrol y values: midpoint between the nearest parallel bay inner edge and
    # the closest perpendicular bay nose face, on each side.
    #
    # Upper: between row B nose face and the minimum-y corner of the top-wall bays.
    _row_b_nose_y = perp_row_b_cy + dims_perp["depth"] / 2.0
    _top_yaw_rad = math.radians(top_par_yaw)
    _top_cos = math.cos(_top_yaw_rad)
    _top_sin = math.sin(_top_yaw_rad)
    _hd = dims_par["depth"] / 2.0
    _hw = dims_par["width"] / 2.0
    _top_par_min_y = min(
        (width_front + top_wdy * (par_top_along_start + i * par_top_spacing) + top_ny * par_top_normal_offset)
        + _top_sin * lx + _top_cos * ly
        for i in range(4)
        for lx, ly in [(-_hd, -_hw), (_hd, -_hw), (_hd, _hw), (-_hd, _hw)]
    )
    _patrol_upper_cy = (_row_b_nose_y + _top_par_min_y) / 2.0
    #
    # Lower: between row A nose face and the max-y (inner) edge of the angled bays.
    _row_a_nose_y = perp_row_a_cy - dims_perp["depth"] / 2.0
    x_enter = perp_cluster_x_min / 2.0
    x_exit = (perp_cluster_x_max + par_right_inner_x) / 2.0
    _ang_yaw_rad = math.radians(ang_yaw)
    _ang_cos = math.cos(_ang_yaw_rad)
    _ang_sin = math.sin(_ang_yaw_rad)
    _ang_hd = dims_ang["depth"] / 2.0
    _ang_hw = dims_ang["width"] / 2.0
    _ang_max_y = max(
        b["local_y"] + _ang_sin * lx + _ang_cos * ly
        for b in ang_bays
        for lx, ly in [(-_ang_hd, -_ang_hw), (_ang_hd, -_ang_hw), (_ang_hd, _ang_hw), (-_ang_hd, _ang_hw)]
    )
    _patrol_lower_cy = (_row_a_nose_y + _ang_max_y) / 2.0
    patrol = [
        {"x": x_enter, "y": _patrol_lower_cy - 4.0},  # WP1: lower-left, clear of entrance
        {"x": x_exit, "y": _patrol_lower_cy},           # WP2: lower-right
        {"x": x_exit, "y": _patrol_upper_cy},           # WP3: upper-right
        {"x": x_enter, "y": _patrol_upper_cy + 4.0},   # WP4: upper-left
    ]

    # ------------------------------------------------------------------
    # Pedestrian zones -- one strip per distinct aisle face.
    # PED_STRIP and _PED_MARGIN are imported from common.py.
    # ------------------------------------------------------------------
    ped_zones = [
        # Zone 1: aisle below row A nose face (row A faces -Y, yaw=90).
        {
            "x_min": perp_cluster_x_min + _PED_MARGIN,
            "x_max": perp_cluster_x_max - _PED_MARGIN,
            "y_min": perp_row_a_cy - dims_perp["depth"] / 2.0 - PED_STRIP,
            "y_max": perp_row_a_cy - dims_perp["depth"] / 2.0 - _PED_MARGIN,
        },
        # Zone 2: back-to-back aisle between row A back face and row B back face.
        {
            "x_min": perp_cluster_x_min + _PED_MARGIN,
            "x_max": perp_cluster_x_max - _PED_MARGIN,
            "y_min": perp_row_a_cy + dims_perp["depth"] / 2.0 + _PED_MARGIN,
            "y_max": perp_row_b_cy - dims_perp["depth"] / 2.0 - _PED_MARGIN,
        },
        # Zone 3: aisle above row B nose face (row B faces +Y, yaw=270).
        {
            "x_min": perp_cluster_x_min + _PED_MARGIN,
            "x_max": perp_cluster_x_max - _PED_MARGIN,
            "y_min": perp_row_b_cy + dims_perp["depth"] / 2.0 + _PED_MARGIN,
            "y_max": perp_row_b_cy + dims_perp["depth"] / 2.0 + PED_STRIP,
        },
        # Zone 4: vertical strip against the left face of the right-wall parallel group.
        {
            "x_min": par_right_inner_x - PED_STRIP,
            "x_max": par_right_inner_x - _PED_MARGIN,
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
