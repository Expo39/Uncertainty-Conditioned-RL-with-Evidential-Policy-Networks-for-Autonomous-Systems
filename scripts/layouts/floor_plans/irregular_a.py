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
  - Top-flat cluster: back-to-back perpendicular rows against the flat top wall
    (P8..P7, x=0..20): 7 bays/row, facing each other across a 6 m aisle.
  - Top-diagonal angled row: 11 bays along the diagonal top wall (P6->P7) at 45 deg.
  - Left-wall angled group: 4 bays against the left wall (x=0), 45 deg.
  - Notch cluster: 4 perpendicular bays against the notch top wall (y=8).
  - Bottom parallel group: 5 bays along the bottom wall (x=0..53, y=0).
  - Right-wall parallel group: 3 bays against the right wall (x=80).

Three spawn transforms (3 m inside the perimeter along each heading):
  - Primary   (S1): left wall entry (x=3, y=25), facing +X into lot.
  - Secondary (S2): diagonal top wall at x=70, 3 m inward perpendicular to slope.
  - Tertiary  (S3): bottom wall right section (x=70, y=3), facing +Y.

Used as an OOD evaluation layout (OOD=True).
"""

import math
from typing import Any, Dict

from scripts.layouts.builder import LotBuilder, PatrolPath, PedestrianZone

# World-frame origin chosen so the primary spawn lands at CARLA (0,0).
ORIGIN_X = -3.0
ORIGIN_Y = 25.0
ORIGIN_Z = 0.3
HEADING_DEG = 0.0
OOD = True


def generate() -> Dict[str, Any]:
    """
    @brief Build the irregular_a layout using the LotBuilder DSL.
    @return Layout dict (corners, bays, spawn, extra_spawns, patrol_waypoints,
            pedestrian_zones, obstacles).
    """
    # Perimeter polygon (9 vertices, CCW).
    p0 = (0.0, 0.0)
    p1 = (53.0, 0.0)
    p2 = (53.0, 8.0)
    p3 = (65.0, 8.0)
    p4 = (65.0, 0.0)
    p5 = (80.0, 0.0)
    p6 = (80.0, 37.0)
    p7 = (20.0, 50.0)
    p8 = (0.0, 50.0)

    lot = LotBuilder(name="irregular_a", corners=[p0, p1, p2, p3, p4, p5, p6, p7, p8])
    dims_perp = lot.dims["perpendicular"]
    dims_ang = lot.dims["angled"]
    dims_par = lot.dims["parallel"]

    # ------------------------------------------------------------------
    # Obstacle perpendicular bays: back-to-back rows on all four faces of the
    # central obstacle rectangle (x=23.25..39.25, y=17..21).
    # ------------------------------------------------------------------
    OBSTACLE_X_MIN = 23.25
    OBSTACLE_X_MAX = 39.25
    OBSTACLE_Y_MIN = 17.0
    OBSTACLE_Y_MAX = 21.0
    PERP_OBS_TB = 6
    PERP_OBS_LR = 2

    obs_row_c_cy = OBSTACLE_Y_MIN - lot.wall_gap - dims_perp["depth"] / 2.0
    obs_row_d_cy = OBSTACLE_Y_MAX + lot.wall_gap + dims_perp["depth"] / 2.0
    obs_tb_cx_start = (
        (OBSTACLE_X_MIN + OBSTACLE_X_MAX) / 2.0
        - (PERP_OBS_TB - 1) / 2.0 * dims_perp["width"]
    )
    lot.row(
        bay_type="perpendicular",
        n=PERP_OBS_TB,
        anchor=(obs_tb_cx_start, obs_row_c_cy),
        direction="east",
        yaw_deg=90.0,
    )
    lot.row(
        bay_type="perpendicular",
        n=PERP_OBS_TB,
        anchor=(obs_tb_cx_start, obs_row_d_cy),
        direction="east",
        yaw_deg=270.0,
    )

    obs_row_e_cx = OBSTACLE_X_MIN - lot.wall_gap - dims_perp["depth"] / 2.0
    obs_row_f_cx = OBSTACLE_X_MAX + lot.wall_gap + dims_perp["depth"] / 2.0
    obs_lr_cy_start = (
        (OBSTACLE_Y_MIN + OBSTACLE_Y_MAX) / 2.0
        - (PERP_OBS_LR - 1) / 2.0 * dims_perp["width"]
    )
    lot.row(
        bay_type="perpendicular",
        n=PERP_OBS_LR,
        anchor=(obs_row_e_cx, obs_lr_cy_start),
        direction="north",
        yaw_deg=0.0,
    )
    lot.row(
        bay_type="perpendicular",
        n=PERP_OBS_LR,
        anchor=(obs_row_f_cx, obs_lr_cy_start),
        direction="north",
        yaw_deg=180.0,
    )

    # ------------------------------------------------------------------
    # Left-wall angled group: 4 bays against the left wall (x=0), 45 deg.
    # Wall traversed downward (P8->P0) so the inward normal points +X.
    # _left_wall_y0 chosen so the bottom bay lands close to y=3.
    # ------------------------------------------------------------------
    ANG_BAYS_LEFT = 4
    _left_spacing = dims_ang["width"] / math.sin(math.radians(45.0))
    _ang_diag_half = (dims_ang["depth"] + dims_ang["width"]) / 2.0
    _left_end_margin = _ang_diag_half * math.cos(math.radians(45.0)) + 0.5 + lot.wall_gap
    _left_wall_y0 = (
        3.0 + (ANG_BAYS_LEFT - 1) * _left_spacing + 2.0 * _left_end_margin
    )
    lot.row_along_wall(
        bay_type="angled",
        n=ANG_BAYS_LEFT,
        wall_p0=(0.0, _left_wall_y0),
        wall_p1=(0.0, 0.0),
        bay_angle_deg=45.0,
        side="ccw",
        start_along=_left_end_margin,
        pack_from="start",
    )

    # ------------------------------------------------------------------
    # Top-diagonal angled row: 11 bays along P6(80,37)->P7(20,50).
    # Group hugs the P7 (top-left) end; the spare gap falls at the P6 end.
    # ------------------------------------------------------------------
    ANG_BAYS_DIAGONAL = 11
    top_wall_len = math.hypot(p7[0] - p6[0], p7[1] - p6[1])
    _ang_spacing = dims_ang["width"] / math.sin(math.radians(45.0))
    _ang_end_margin = _ang_diag_half * math.cos(math.radians(45.0)) + 0.5 + lot.wall_gap
    ang_start_diag = top_wall_len - _ang_end_margin - ANG_BAYS_DIAGONAL * _ang_spacing
    lot.row_along_wall(
        bay_type="angled",
        n=ANG_BAYS_DIAGONAL,
        wall_p0=p6,
        wall_p1=p7,
        bay_angle_deg=45.0,
        side="ccw",
        start_along=ang_start_diag,
        pack_from="start",
    )

    # ------------------------------------------------------------------
    # Right-wall parallel group: 3 bays against the right wall (x=80, y=0..37).
    # Axis-aligned. yaw=90 (nose +Y), depth (8 m) along Y.
    # ------------------------------------------------------------------
    PAR_BAYS_RIGHT = 3
    par_right_cx = 80.0 - lot.wall_gap - dims_par["width"] / 2.0
    par_right_cy_start = 37.0 / 2.0 - (PAR_BAYS_RIGHT - 1) / 2.0 * dims_par["depth"]
    lot.row(
        bay_type="parallel",
        n=PAR_BAYS_RIGHT,
        anchor=(par_right_cx, par_right_cy_start),
        direction="north",
        yaw_deg=90.0,
        spacing=dims_par["depth"],
    )

    # ------------------------------------------------------------------
    # Bottom parallel group: 5 bays along x=0..53, y=0. yaw=0, depth along X.
    # 5 bays span 40 m; centred in the 53 m wall -> left edge at x=6.5.
    # ------------------------------------------------------------------
    PAR_BAYS_BOTTOM = 5
    par_cy = dims_par["width"] / 2.0 + lot.wall_gap
    par_x_start = (
        (53.0 - PAR_BAYS_BOTTOM * dims_par["depth"]) / 2.0 + dims_par["depth"] / 2.0
    )
    lot.row(
        bay_type="parallel",
        n=PAR_BAYS_BOTTOM,
        anchor=(par_x_start, par_cy),
        direction="east",
        yaw_deg=0.0,
        spacing=dims_par["depth"],
    )

    # ------------------------------------------------------------------
    # Top-flat cluster: back-to-back perpendicular rows against the flat top
    # wall (P8..P7, x=0..20). 7 bays/row, facing each other across a 6 m aisle.
    # ------------------------------------------------------------------
    PERP_TOP_FLAT = 7
    top_flat_perp_cy = 50.0 - lot.wall_gap - dims_perp["depth"] / 2.0
    top_flat_perp_cx_start = (
        (0.0 + 20.0) / 2.0 - (PERP_TOP_FLAT - 1) / 2.0 * dims_perp["width"]
    )
    lot.row(
        bay_type="perpendicular",
        n=PERP_TOP_FLAT,
        anchor=(top_flat_perp_cx_start, top_flat_perp_cy),
        direction="east",
        yaw_deg=270.0,
    )
    top_flat_perp2_cy = top_flat_perp_cy - (6.0 + dims_perp["depth"])
    lot.row(
        bay_type="perpendicular",
        n=PERP_TOP_FLAT,
        anchor=(top_flat_perp_cx_start, top_flat_perp2_cy),
        direction="east",
        yaw_deg=90.0,
    )

    # ------------------------------------------------------------------
    # Notch cluster: 4 perpendicular bays against the notch top wall (y=8).
    # 4 * 2.5 m = 10 m, centred in the 12 m notch top span (x=53..65).
    # ------------------------------------------------------------------
    PERP_NOTCH = 4
    notch_perp_cy = 8.0 + lot.wall_gap + dims_perp["depth"] / 2.0
    notch_perp_cx_start = (
        (53.0 + 65.0) / 2.0 - (PERP_NOTCH - 1) / 2.0 * dims_perp["width"]
    )
    lot.row(
        bay_type="perpendicular",
        n=PERP_NOTCH,
        anchor=(notch_perp_cx_start, notch_perp_cy),
        direction="east",
        yaw_deg=90.0,
    )

    # ------------------------------------------------------------------
    # Spawns (3 m inside the perimeter along each heading).
    # ------------------------------------------------------------------
    _s2_y_proj = 37.0 + (50.0 - 37.0) / (20.0 - 80.0) * (70.0 - 80.0)
    _top_wlen = math.hypot(60.0, 13.0)
    _s2_yaw = math.degrees(math.atan2(-60.0 / _top_wlen, -13.0 / _top_wlen)) % 360.0
    _s2_x_nudged = round(70.0 + math.cos(math.radians(_s2_yaw)) * 3.0, 1)
    _s2_y_nudged = round(_s2_y_proj + math.sin(math.radians(_s2_yaw)) * 3.0, 1)

    lot.spawn(x=3.0, y=25.0, yaw_deg=0.0, primary=True)
    lot.spawn(x=_s2_x_nudged, y=_s2_y_nudged, yaw_deg=round(_s2_yaw, 1))
    lot.spawn(x=70.0, y=3.0, yaw_deg=90.0)

    # ------------------------------------------------------------------
    # Patrol path: 5-waypoint CCW orbit around the central obstacle.
    # ------------------------------------------------------------------
    _left_offset = (
        _ang_diag_half * math.sin(math.radians(45.0)) + lot.wall_gap
    )
    _left_ang_right_x = _left_offset + dims_ang["depth"] / 2.0 * math.cos(
        math.radians(45.0)
    )
    _obs_row_e_left_x = obs_row_e_cx - dims_perp["depth"] / 2.0
    _obs_row_f_nose_x = obs_row_f_cx + dims_perp["depth"] / 2.0
    _notch_left_x = notch_perp_cx_start - dims_perp["width"] / 2.0
    _par_bottom_top_y = par_cy + dims_par["width"] / 2.0
    _obs_row_c_nose_y = obs_row_c_cy - dims_perp["depth"] / 2.0
    _obs_row_d_nose_y = obs_row_d_cy + dims_perp["depth"] / 2.0
    _ang_offset_diag = (
        _ang_diag_half * math.sin(math.radians(45.0)) + lot.wall_gap
    )
    _diag_front_y = 37.0 - _ang_offset_diag - dims_ang["depth"] / 2.0

    _lower_y = (_obs_row_c_nose_y + _par_bottom_top_y) / 2.0
    _upper_y = (_diag_front_y + _obs_row_d_nose_y) / 2.0
    _right_x = (_obs_row_f_nose_x + _notch_left_x) / 2.0
    _left_x = (_left_ang_right_x + _obs_row_e_left_x) / 2.0

    patrol = PatrolPath()
    patrol.add(_left_x, _lower_y)
    patrol.add(_left_x, 28.0)
    patrol.add(26.0, 34.0)
    patrol.add(_right_x, _upper_y)
    patrol.add(_right_x, _lower_y)
    lot.set_patrol(patrol)

    # ------------------------------------------------------------------
    # Pedestrian zones - one strip per distinct aisle face.
    # ------------------------------------------------------------------
    margin = 0.5
    strip = 3.0

    obs_tb_cx_end = obs_tb_cx_start + (PERP_OBS_TB - 1) * dims_perp["width"]
    obs_row_c_nose_y = obs_row_c_cy - dims_perp["depth"] / 2.0
    obs_row_d_nose_y = obs_row_d_cy + dims_perp["depth"] / 2.0
    top_flat_perp_cx_end = (
        top_flat_perp_cx_start + (PERP_TOP_FLAT - 1) * dims_perp["width"]
    )
    top_row1_nose_y = top_flat_perp_cy - dims_perp["depth"] / 2.0
    top_row2_nose_y = top_flat_perp2_cy + dims_perp["depth"] / 2.0
    notch_perp_cx_end = notch_perp_cx_start + (PERP_NOTCH - 1) * dims_perp["width"]
    notch_nose_y = notch_perp_cy + dims_perp["depth"] / 2.0
    par_right_nose_x = par_right_cx - dims_par["width"] / 2.0
    par_right_y_end = par_right_cy_start + (PAR_BAYS_RIGHT - 1) * dims_par["depth"]

    # Zone 1: aisle below obstacle row C nose faces (faces +Y, nose at y=11.5).
    lot.add_zone(
        PedestrianZone(
            x_min=obs_tb_cx_start - dims_perp["width"] / 2.0 + margin,
            x_max=obs_tb_cx_end + dims_perp["width"] / 2.0 - margin,
            y_min=obs_row_c_nose_y - strip,
            y_max=obs_row_c_nose_y - margin,
        )
    )
    # Zone 2: aisle above obstacle row D nose faces (faces -Y, nose at y=26.5).
    lot.add_zone(
        PedestrianZone(
            x_min=obs_tb_cx_start - dims_perp["width"] / 2.0 + margin,
            x_max=obs_tb_cx_end + dims_perp["width"] / 2.0 - margin,
            y_min=obs_row_d_nose_y + margin,
            y_max=obs_row_d_nose_y + strip,
        )
    )
    # Zone 3: aisle between the two top-flat back-to-back perp rows.
    lot.add_zone(
        PedestrianZone(
            x_min=top_flat_perp_cx_start - dims_perp["width"] / 2.0 + margin,
            x_max=top_flat_perp_cx_end + dims_perp["width"] / 2.0 - margin,
            y_min=top_row2_nose_y + margin,
            y_max=top_row1_nose_y - margin,
        )
    )
    # Zone 4: aisle above notch perp bay nose faces (bays face +Y).
    lot.add_zone(
        PedestrianZone(
            x_min=notch_perp_cx_start - dims_perp["width"] / 2.0 + margin,
            x_max=notch_perp_cx_end + dims_perp["width"] / 2.0 - margin,
            y_min=notch_nose_y + margin,
            y_max=notch_nose_y + strip,
        )
    )
    # Zone 5: strip left of right-wall parallel bay nose faces (bays face -X).
    lot.add_zone(
        PedestrianZone(
            x_min=par_right_nose_x - strip,
            x_max=par_right_nose_x - margin,
            y_min=par_right_cy_start - dims_par["depth"] / 2.0 + margin,
            y_max=par_right_y_end + dims_par["depth"] / 2.0 - margin,
        )
    )

    # ------------------------------------------------------------------
    # Interior obstacle: rectangular cone wall in the centre of the lot.
    # ------------------------------------------------------------------
    lot.add_obstacle(
        x_min=OBSTACLE_X_MIN,
        x_max=OBSTACLE_X_MAX,
        y_min=OBSTACLE_Y_MIN,
        y_max=OBSTACLE_Y_MAX,
    )

    return lot.build()
