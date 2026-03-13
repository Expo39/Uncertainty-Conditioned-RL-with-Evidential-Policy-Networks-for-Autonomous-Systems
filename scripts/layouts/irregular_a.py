"""
@file irregular_a.py
@brief Irregular nine-sided parking lot floor plan (OOD, ~80x50 m).

Perimeter inspired by a shed-style building footprint:
  - Left wall:  tall vertical side (x=0, y=0..50).
  - Top wall:   diagonal slope from top-right P6(80,37) down-left to P7(20,50),
                then flat from P7(20,50) to P8(0,50).
  - Right wall: shorter vertical side (x=80, y=0..37).
  - Bottom:     flat with a rectangular service notch cut in (x=53..65, y=0..8),
                splitting the bottom wall into two segments.

This combination of a sloped top wall, non-rectangular footprint, and bottom notch
creates a spatial layout the agent never sees during training (rectangle and trapezoid
are both convex and symmetric). Bay groups are placed in orientations absent from
the training layouts:

  - Obstacle cluster: back-to-back perpendicular rows on all four faces of the central
    rectangular obstacle (x=23.25..39.25, y=17..21): 6 bays/row on top/bottom faces,
    2 bays/row on left/right faces.
  - Top-left cluster: back-to-back perpendicular rows against the flat top wall
    (P8..P7, x=0..20): 7 bays/row, facing each other across a 6 m aisle.
  - Angled row:     11 bays along the diagonal top wall (P6->P7) at 45 deg to
                    the slope; group hugs the P7 end, spare gap at P6.
  - Left-wall angled group: 4 bays against the left wall (x=0), 45 deg,
                    starting at y=3 and stepping upward.
  - Notch cluster: 4 perpendicular bays against the notch top wall (y=8),
                    nose facing +Y.
  - Bottom parallel group: 5 bays along the bottom wall (x=0..53, y=0),
                    depth along X (yaw=0).
  - Right-wall parallel group: 3 bays against the right wall (x=80),
                    depth along Y (yaw=90).

Three spawn transforms:
  - Primary   (S1): left wall mid-height (x=0, y=25), facing +X into lot.
  - Secondary (S2): diagonal top wall at x=70, facing inward perpendicular to slope.
  - Tertiary  (S3): bottom wall right section (x=70, y=0), facing +Y.

Used as an OOD evaluation layout (OOD=True).
"""

import math
from typing import Any, Dict, List

from scripts.layouts.common import (
    BAY_DIMS,
    ang_offset_from_wall,
    ang_x_margin,
    angled_bays_along_wall,
    validate_bays_in_polygon,
    warn_narrow_corridors,
)

# World-frame origin used in multi-layout generation (Town05_Opt flat area).
ORIGIN_X = 100.0
ORIGIN_Y = 0.0
ORIGIN_Z = 0.3
HEADING_DEG = 0.0
OOD = True

_WALL_GAP = 0.5  # minimum clearance between bay back face and perimeter wall/cones


def generate() -> Dict[str, Any]:
    """
    @brief Build the irregular_a layout in local frame.
    @return Layout dict with keys: corners, bays, spawn, extra_spawns,
            patrol_waypoints, pedestrian_zones, obstacles.
    """
    # ------------------------------------------------------------------
    # Perimeter polygon (9 vertices, CCW).
    # The notch (x=53..65, y=0..8) is a service access recess on the bottom wall.
    # Diagonal top wall runs from P6(80,37) to P7(20,50), then flat to P8(0,50).
    # ------------------------------------------------------------------
    corners = [
        {"x":  0.0, "y":  0.0},   # P0 bottom-left
        {"x": 53.0, "y":  0.0},   # P1 notch-left bottom
        {"x": 53.0, "y":  8.0},   # P2 notch-left top
        {"x": 65.0, "y":  8.0},   # P3 notch-right top
        {"x": 65.0, "y":  0.0},   # P4 notch-right bottom
        {"x": 80.0, "y":  0.0},   # P5 bottom-right
        {"x": 80.0, "y": 37.0},   # P6 top-right (diagonal wall start)
        {"x": 20.0, "y": 50.0},   # P7 diagonal/flat wall junction
        {"x":  0.0, "y": 50.0},   # P8 left-wall top corner
    ]

    dims_perp = BAY_DIMS["perpendicular"]   # width=2.5, depth=5.0
    dims_ang  = BAY_DIMS["angled"]          # width=2.5, depth=5.4
    dims_par  = BAY_DIMS["parallel"]        # width=2.5, depth=8.0

    # ------------------------------------------------------------------
    # Perpendicular bays: back-to-back rows on all four faces of the central
    # obstacle rectangle (x=23.25..39.25, y=17..21, span 16x4 m).
    #   Row C (yaw=90,  nose +Y): back at obstacle y_min=17, 6 bays along X.
    #   Row D (yaw=270, nose -Y): back at obstacle y_max=21, 6 bays along X.
    #   Row E (yaw=0,   nose +X): back at obstacle x_min=23.25, 2 bays along Y.
    #   Row F (yaw=180, nose -X): back at obstacle x_max=39.25, 2 bays along Y.
    #   Bay counts are geometry-driven (16 m span fits 6*2.5 m; 4 m span fits 2*2.5 m).
    # ------------------------------------------------------------------
    OBSTACLE_X_MIN = 23.25
    OBSTACLE_X_MAX = 39.25
    OBSTACLE_Y_MIN = 17.0
    OBSTACLE_Y_MAX = 21.0
    PERP_OBS_TB = 6   # top/bottom rows: 6 * 2.5 m = 15 m, fits in 16 m span
    PERP_OBS_LR = 2   # left/right rows: 2 * 2.5 m =  5 m, fits in  4 m span

    perp_bays: List[Dict] = []

    # Rows C/D: top and bottom faces of obstacle, bays oriented nose along Y.
    obs_row_c_cy = OBSTACLE_Y_MIN - _WALL_GAP - dims_perp["depth"] / 2.0
    obs_row_d_cy = OBSTACLE_Y_MAX + _WALL_GAP + dims_perp["depth"] / 2.0
    obs_tb_cx_start = (
        (OBSTACLE_X_MIN + OBSTACLE_X_MAX) / 2.0
        - (PERP_OBS_TB - 1) / 2.0 * dims_perp["width"]
    )
    for i in range(PERP_OBS_TB):
        cx = obs_tb_cx_start + i * dims_perp["width"]
        perp_bays.append(
            {
                "bay_type": "perpendicular",
                "local_x": cx,
                "local_y": obs_row_c_cy,
                "local_yaw_deg": 90.0,
                "width": dims_perp["width"],
                "depth": dims_perp["depth"],
            }
        )
        perp_bays.append(
            {
                "bay_type": "perpendicular",
                "local_x": cx,
                "local_y": obs_row_d_cy,
                "local_yaw_deg": 270.0,
                "width": dims_perp["width"],
                "depth": dims_perp["depth"],
            }
        )

    # Rows E/F: left and right faces of obstacle, bays oriented nose along X.
    obs_row_e_cx = OBSTACLE_X_MIN - _WALL_GAP - dims_perp["depth"] / 2.0
    obs_row_f_cx = OBSTACLE_X_MAX + _WALL_GAP + dims_perp["depth"] / 2.0
    obs_lr_cy_start = (
        (OBSTACLE_Y_MIN + OBSTACLE_Y_MAX) / 2.0
        - (PERP_OBS_LR - 1) / 2.0 * dims_perp["width"]
    )
    for i in range(PERP_OBS_LR):
        cy = obs_lr_cy_start + i * dims_perp["width"]
        perp_bays.append(
            {
                "bay_type": "perpendicular",
                "local_x": obs_row_e_cx,
                "local_y": cy,
                "local_yaw_deg": 0.0,
                "width": dims_perp["width"],
                "depth": dims_perp["depth"],
            }
        )
        perp_bays.append(
            {
                "bay_type": "perpendicular",
                "local_x": obs_row_f_cx,
                "local_y": cy,
                "local_yaw_deg": 180.0,
                "width": dims_perp["width"],
                "depth": dims_perp["depth"],
            }
        )

    # ------------------------------------------------------------------
    # Angled bays: 4 bays against the left wall (x=0), starting at y=3.
    # Wall traversed downward: wall_dx=0, wall_dy=-1.
    # CCW inward normal = (+1, 0) -> into lot (+X). facing_yaw = 45 deg.
    # _left_wall_y0 computed so the bottom-most bay centre lands at y=3.
    # ------------------------------------------------------------------
    ANG_BAYS_LEFT = 4
    _left_spacing = dims_ang["width"] / math.sin(math.radians(45.0))
    _left_end_margin = ang_x_margin(dims_ang["depth"], dims_ang["width"]) + _WALL_GAP
    _left_offset = ang_offset_from_wall(dims_ang["depth"], dims_ang["width"]) + _WALL_GAP
    # wall_y0 chosen so that the first bay placed (bottom of group) centres at y=3.
    # First bay centre = wall_y0 - _left_end_margin, so wall_y0 = 3 + _left_end_margin
    # + (n-1)*spacing accounts for all n bays above the bottom one.
    _left_wall_y0 = 3.0 + (ANG_BAYS_LEFT - 1) * _left_spacing + 2.0 * _left_end_margin
    _left_yaw = (math.degrees(math.atan2(0.0, 1.0)) + 45.0) % 360.0  # 45 deg
    left_ang_bays = angled_bays_along_wall(
        ANG_BAYS_LEFT,
        wall_x0=0.0,
        wall_y0=_left_wall_y0,
        wall_dx=0.0,
        wall_dy=-1.0,
        wall_len=_left_wall_y0,
        offset_from_wall=_left_offset,
        start_along_wall=_left_end_margin,
        facing_yaw_deg=_left_yaw,
    )

    # ------------------------------------------------------------------
    # Angled bays: 11 bays along the diagonal top wall P6(80,37)->P7(20,50).
    # Wall traversed from P6 toward P7 so the CCW inward normal points into
    # the lot interior (down-left direction).
    # Wall direction: dx=-60, dy=+13 (normalised). Length ~61.4 m.
    # CCW inward normal: (-dy, dx)/len = (-13/len, -60/len) -> down-left.
    # facing_yaw = atan2(wdx, -wdy) + 45 deg (same convention as other angled groups).
    # Group hugs the P7 (top-left) end; the spare gap falls at the P6 (right) end.
    # ang_start = wall_len - end_margin - n*spacing drops the rightmost (P6-end) bay.
    # ------------------------------------------------------------------
    ANG_BAYS_DIAGONAL = 11

    top_wall_dx_raw = 20.0 - 80.0   # -60
    top_wall_dy_raw = 50.0 - 37.0   # +13
    top_wall_len = math.hypot(top_wall_dx_raw, top_wall_dy_raw)
    wdx_top = top_wall_dx_raw / top_wall_len
    wdy_top = top_wall_dy_raw / top_wall_len
    ang_yaw_top = (math.degrees(math.atan2(wdx_top, -wdy_top)) + 45.0) % 360.0

    ang_offset = ang_offset_from_wall(dims_ang["depth"], dims_ang["width"]) + _WALL_GAP
    _ang_spacing = dims_ang["width"] / math.sin(math.radians(45.0))
    _ang_end_margin = ang_x_margin(dims_ang["depth"], dims_ang["width"]) + _WALL_GAP
    # Shift start one full spacing beyond the P7-hugging position so the P6-end
    # bay is dropped and the group remains flush against P7.
    ang_start = top_wall_len - _ang_end_margin - ANG_BAYS_DIAGONAL * _ang_spacing
    ang_bays = angled_bays_along_wall(
        ANG_BAYS_DIAGONAL,
        wall_x0=80.0,
        wall_y0=37.0,
        wall_dx=wdx_top,
        wall_dy=wdy_top,
        wall_len=top_wall_len,
        offset_from_wall=ang_offset,
        start_along_wall=ang_start,
        facing_yaw_deg=ang_yaw_top,
    )

    # ------------------------------------------------------------------
    # Parallel bays: 5 bays centred along the bottom wall segment x=0..53, y=0.
    # yaw=0: depth (8 m) along X, width (2.5 m) along Y. Back against y=0.
    # 5 bays span 40 m; centred in the 53 m wall -> left edge at x=6.5.
    # ------------------------------------------------------------------
    PAR_BAYS_BOTTOM = 5
    par_cy = dims_par["width"] / 2.0 + _WALL_GAP   # 1.75
    par_x_start = (53.0 - PAR_BAYS_BOTTOM * dims_par["depth"]) / 2.0 + dims_par["depth"] / 2.0

    # ------------------------------------------------------------------
    # Parallel bays: 3 bays against the right wall (x=80), y=0..37.
    # yaw=90: depth (8 m) along Y, width (2.5 m) along X. Back against x=80.
    # 3 bays span 24 m; centred on the 37 m wall.
    # ------------------------------------------------------------------
    PAR_BAYS_RIGHT = 3
    par_right_cx = 80.0 - _WALL_GAP - dims_par["width"] / 2.0
    par_right_cy_start = 37.0 / 2.0 - (PAR_BAYS_RIGHT - 1) / 2.0 * dims_par["depth"]
    par_bays: List[Dict] = []
    for i in range(PAR_BAYS_RIGHT):
        par_bays.append(
            {
                "bay_type": "parallel",
                "local_x": par_right_cx,
                "local_y": par_right_cy_start + i * dims_par["depth"],
                "local_yaw_deg": 90.0,
                "width": dims_par["width"],
                "depth": dims_par["depth"],
            }
        )

    for i in range(PAR_BAYS_BOTTOM):
        par_bays.append(
            {
                "bay_type": "parallel",
                "local_x": par_x_start + i * dims_par["depth"],
                "local_y": par_cy,
                "local_yaw_deg": 0.0,
                "width": dims_par["width"],
                "depth": dims_par["depth"],
            }
        )

    # ------------------------------------------------------------------
    # Perpendicular bays: back-to-back rows against the flat top wall P8(0,50)->P7(20,50).
    # Row 1 (yaw=270, nose -Y): back against y=50, 7 bays centred over the 20 m span.
    # Row 2 (yaw=90,  nose +Y): facing row 1 across a 6 m aisle.
    # Centre-to-centre gap = aisle + depth = 6 + 5 = 11 m.
    # ------------------------------------------------------------------
    PERP_TOP_FLAT = 7
    top_flat_perp_cy = 50.0 - _WALL_GAP - dims_perp["depth"] / 2.0   # 44.5
    top_flat_perp_cx_start = (
        (0.0 + 20.0) / 2.0 - (PERP_TOP_FLAT - 1) / 2.0 * dims_perp["width"]
    )
    for i in range(PERP_TOP_FLAT):
        perp_bays.append(
            {
                "bay_type": "perpendicular",
                "local_x": top_flat_perp_cx_start + i * dims_perp["width"],
                "local_y": top_flat_perp_cy,
                "local_yaw_deg": 270.0,
                "width": dims_perp["width"],
                "depth": dims_perp["depth"],
            }
        )

    top_flat_perp2_cy = top_flat_perp_cy - (6.0 + dims_perp["depth"])   # 33.5
    for i in range(PERP_TOP_FLAT):
        perp_bays.append(
            {
                "bay_type": "perpendicular",
                "local_x": top_flat_perp_cx_start + i * dims_perp["width"],
                "local_y": top_flat_perp2_cy,
                "local_yaw_deg": 90.0,
                "width": dims_perp["width"],
                "depth": dims_perp["depth"],
            }
        )

    # ------------------------------------------------------------------
    # Perpendicular bays: 4 bays against the notch top wall P2(53,8)->P3(65,8).
    # Back against y=8, nose +Y (yaw=90). 4 * 2.5 m = 10 m, centred in 12 m span.
    # ------------------------------------------------------------------
    PERP_NOTCH = 4
    notch_perp_cy = 8.0 + _WALL_GAP + dims_perp["depth"] / 2.0   # 11.0
    notch_perp_cx_start = (
        (53.0 + 65.0) / 2.0 - (PERP_NOTCH - 1) / 2.0 * dims_perp["width"]
    )
    for i in range(PERP_NOTCH):
        perp_bays.append(
            {
                "bay_type": "perpendicular",
                "local_x": notch_perp_cx_start + i * dims_perp["width"],
                "local_y": notch_perp_cy,
                "local_yaw_deg": 90.0,
                "width": dims_perp["width"],
                "depth": dims_perp["depth"],
            }
        )

    all_bays = perp_bays + ang_bays + left_ang_bays + par_bays

    validate_bays_in_polygon(all_bays, corners, "irregular_a")
    warn_narrow_corridors(all_bays, "irregular_a")

    # ------------------------------------------------------------------
    # Spawn transforms.
    # S1: left wall mid-height, facing +X into lot.
    # S2: diagonal top wall at local x=70, facing inward perpendicular to the slope.
    #     y interpolated along P6(80,37)->P7(20,50).
    #     Inward normal direction: wall vec is (-60,13)/len, CCW normal = (-13/len, -60/len).
    # S3: bottom wall right section (right of notch), facing +Y.
    # ------------------------------------------------------------------
    _s2_y = 37.0 + (50.0 - 37.0) / (20.0 - 80.0) * (70.0 - 80.0)
    _top_wlen = math.hypot(60.0, 13.0)
    _s2_yaw = math.degrees(math.atan2(-60.0 / _top_wlen, -13.0 / _top_wlen)) % 360.0

    spawn  = {"x":  0.0, "y": 25.0, "yaw_deg":  0.0}
    spawn2 = {"x": 70.0, "y": round(_s2_y, 2), "yaw_deg": round(_s2_yaw, 1)}
    spawn3 = {"x": 70.0, "y":  0.0, "yaw_deg": 90.0}

    # ------------------------------------------------------------------
    # Patrol path: CCW orbit around the central obstacle (5 waypoints).
    #
    # Corridor x/y values are computed as midpoints between the nearest facing
    # bay nose/back surfaces on each side of each corridor segment:
    #   _left_x  : midpoint between left-wall angled bay right edges and
    #              obstacle row E left nose faces.
    #   _right_x : midpoint between obstacle row F right nose faces and
    #              notch perp bay left edges.
    #   _lower_y : midpoint between obstacle row C bottom nose faces and
    #              bottom parallel bay top faces.
    #   _upper_y : midpoint between obstacle row D top nose faces and the
    #              approximate lowest y of the diagonal top-wall bay footprints.
    # WP2b chamfers the top-left corner to avoid clipping the top-left perp cluster.
    # ------------------------------------------------------------------
    _left_ang_right_x = _left_offset + dims_ang["depth"] / 2.0 * math.cos(math.radians(45.0))
    _obs_row_e_left_x = obs_row_e_cx - dims_perp["depth"] / 2.0
    _obs_row_f_nose_x = obs_row_f_cx + dims_perp["depth"] / 2.0
    _notch_left_x = notch_perp_cx_start - dims_perp["width"] / 2.0
    _par_bottom_top_y = par_cy + dims_par["width"] / 2.0

    _obs_row_c_nose_y = obs_row_c_cy - dims_perp["depth"] / 2.0
    _obs_row_d_nose_y = obs_row_d_cy + dims_perp["depth"] / 2.0
    # Approximate lowest y of diagonal top-wall bay footprints (P6 end sits lowest).
    _diag_front_y = 37.0 - ang_offset - dims_ang["depth"] / 2.0

    _lower_y = (_obs_row_c_nose_y + _par_bottom_top_y) / 2.0
    _upper_y = (_diag_front_y + _obs_row_d_nose_y) / 2.0
    _right_x = (_obs_row_f_nose_x + _notch_left_x) / 2.0
    _left_x = (_left_ang_right_x + _obs_row_e_left_x) / 2.0

    patrol = [
        {"x": _left_x, "y": _lower_y},   # WP1: left corridor, lower level
        {"x": _left_x, "y": 28.0},        # WP2: left corridor, upper level
        {"x": 26.0,    "y": 34.0},        # WP3: chamfer cut toward top-right
        {"x": _right_x, "y": _upper_y},   # WP4: upper corridor, right end
        {"x": _right_x, "y": _lower_y},   # WP5: right corridor, drop to lower level
    ]

    # ------------------------------------------------------------------
    # Pedestrian zones -- one strip per distinct aisle face (PED_STRIP = 3.0 m).
    # ------------------------------------------------------------------
    PED_STRIP = 3.0
    _PED_MARGIN = 0.5

    obs_tb_cx_end = obs_tb_cx_start + (PERP_OBS_TB - 1) * dims_perp["width"]
    obs_row_c_nose_y = obs_row_c_cy - dims_perp["depth"] / 2.0   # 11.5
    obs_row_d_nose_y = obs_row_d_cy + dims_perp["depth"] / 2.0   # 26.5

    top_flat_perp_cx_end = top_flat_perp_cx_start + (PERP_TOP_FLAT - 1) * dims_perp["width"]
    top_row1_nose_y = top_flat_perp_cy - dims_perp["depth"] / 2.0    # row 1 nose faces -Y
    top_row2_nose_y = top_flat_perp2_cy + dims_perp["depth"] / 2.0   # row 2 nose faces +Y

    notch_perp_cx_end = notch_perp_cx_start + (PERP_NOTCH - 1) * dims_perp["width"]
    notch_nose_y = notch_perp_cy + dims_perp["depth"] / 2.0

    par_right_nose_x = par_right_cx - dims_par["width"] / 2.0
    par_right_y_end = par_right_cy_start + (PAR_BAYS_RIGHT - 1) * dims_par["depth"]

    ped_zones = [
        # Zone 1: aisle below obstacle row C nose faces (row C faces +Y, nose at y=11.5).
        {
            "x_min": obs_tb_cx_start - dims_perp["width"] / 2.0 + _PED_MARGIN,
            "x_max": obs_tb_cx_end + dims_perp["width"] / 2.0 - _PED_MARGIN,
            "y_min": obs_row_c_nose_y - PED_STRIP,
            "y_max": obs_row_c_nose_y - _PED_MARGIN,
        },
        # Zone 2: aisle above obstacle row D nose faces (row D faces -Y, nose at y=26.5).
        {
            "x_min": obs_tb_cx_start - dims_perp["width"] / 2.0 + _PED_MARGIN,
            "x_max": obs_tb_cx_end + dims_perp["width"] / 2.0 - _PED_MARGIN,
            "y_min": obs_row_d_nose_y + _PED_MARGIN,
            "y_max": obs_row_d_nose_y + PED_STRIP,
        },
        # Zone 3: aisle between the two top-left back-to-back perp rows
        # (row 2 nose at top_row2_nose_y, row 1 nose at top_row1_nose_y).
        {
            "x_min": top_flat_perp_cx_start - dims_perp["width"] / 2.0 + _PED_MARGIN,
            "x_max": top_flat_perp_cx_end + dims_perp["width"] / 2.0 - _PED_MARGIN,
            "y_min": top_row2_nose_y + _PED_MARGIN,
            "y_max": top_row1_nose_y - _PED_MARGIN,
        },
        # Zone 4: aisle above notch perp bay nose faces (bays face +Y).
        {
            "x_min": notch_perp_cx_start - dims_perp["width"] / 2.0 + _PED_MARGIN,
            "x_max": notch_perp_cx_end + dims_perp["width"] / 2.0 - _PED_MARGIN,
            "y_min": notch_nose_y + _PED_MARGIN,
            "y_max": notch_nose_y + PED_STRIP,
        },
        # Zone 5: strip to the left of right-wall parallel bay nose faces (bays face -X).
        {
            "x_min": par_right_nose_x - PED_STRIP,
            "x_max": par_right_nose_x - _PED_MARGIN,
            "y_min": par_right_cy_start - dims_par["depth"] / 2.0 + _PED_MARGIN,
            "y_max": par_right_y_end + dims_par["depth"] / 2.0 - _PED_MARGIN,
        },
    ]

    # ------------------------------------------------------------------
    # Interior obstacle: rectangular cone wall in the centre of the lot.
    # Extents match OBSTACLE_X_MIN/MAX, OBSTACLE_Y_MIN/MAX defined above.
    # ------------------------------------------------------------------
    obstacles = [
        {
            "x_min": OBSTACLE_X_MIN,
            "x_max": OBSTACLE_X_MAX,
            "y_min": OBSTACLE_Y_MIN,
            "y_max": OBSTACLE_Y_MAX,
        },
    ]

    return {
        "corners": corners,
        "bays": all_bays,
        "spawn": spawn,
        "extra_spawns": [spawn2, spawn3],
        "patrol_waypoints": patrol,
        "pedestrian_zones": ped_zones,
        "obstacles": obstacles,
    }
