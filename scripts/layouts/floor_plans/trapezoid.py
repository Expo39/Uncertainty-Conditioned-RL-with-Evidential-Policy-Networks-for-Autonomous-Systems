"""
@file trapezoid.py
@brief Trapezoid parking lot floor plan (front=60, rear=40, depth=44 m).

Wider at the entrance end (y=0..60) and narrower at the rear (y=10..50),
giving non-parallel top and bottom walls. The tapered profile produces a
subtly different LiDAR/sensor wall signature compared to the rectangle layout,
which helps the agent generalise to non-rectangular geometry.

Bay zones:
  - Centre cluster: 8 back-to-back perpendicular bays per row (yaw=90/270),
    left-shifted slightly from lot centre so the cluster clears the right-wall
    parallel group.
  - Angled row: 9 bays following the diagonal bottom wall (P0->P1) at 45 deg
    to the wall slope.
  - Top-wall parallel group: 4 bays along the sloped top wall (P3->P2).
  - Right-wall parallel group: 3 bays against the right wall (x=44),
    nose facing +Y (yaw=90).

Two spawn transforms (3 m inside the perimeter along each heading):
  - Primary   (S1): left wall entry (x=-2, y=30), facing +X into the lot.
  - Secondary (S2): diagonal bottom wall near x=38, 3 m inward perpendicular
    to the wall slope.

Used as a training layout (OOD=False).
"""

import math
from typing import Any, Dict

from scripts.layouts.builder import LotBuilder, PatrolPath, PedestrianZone

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
    @brief Build the trapezoid layout using the LotBuilder DSL.
    @return Layout dict (corners, bays, spawn, extra_spawns, patrol_waypoints,
            pedestrian_zones).
    """
    width_front = 60.0
    width_rear = 40.0
    depth = 44.0
    y_offset = (width_front - width_rear) / 2.0  # 10.0

    p0 = (-5.0, 0.0)                        # P0 bottom-left (front)
    p1 = (depth, y_offset)                  # P1 bottom-right (rear)
    p2 = (depth, width_front - y_offset)    # P2 top-right (rear)
    p3 = (-5.0, width_front)                # P3 top-left (front)

    lot = LotBuilder(name="trapezoid", corners=[p0, p1, p2, p3])
    dims_perp = lot.dims["perpendicular"]
    dims_ang = lot.dims["angled"]
    dims_par = lot.dims["parallel"]

    # ------------------------------------------------------------------
    # Centre cluster: back-to-back perpendicular rows along X.
    # Row A (lower): yaw=90  (nose +Y). Row B (upper): yaw=270 (nose -Y).
    # PERP_BAYS_PER_ROW = 8 to fill the lot width.
    # _cx_shift offsets the cluster left to keep the right corridor clear
    # of the right-wall parallel group.
    # ------------------------------------------------------------------
    PERP_BAYS_PER_ROW = 8
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
    lot.row(
        bay_type="perpendicular",
        n=PERP_BAYS_PER_ROW,
        anchor=(perp_cx_start, perp_row_a_cy),
        direction="east",
        yaw_deg=90.0,
    )
    lot.row(
        bay_type="perpendicular",
        n=PERP_BAYS_PER_ROW,
        anchor=(perp_cx_start, perp_row_b_cy),
        direction="east",
        yaw_deg=270.0,
    )

    # ------------------------------------------------------------------
    # Angled row: 9 bays along the diagonal bottom wall P0->P1.
    # bay_angle_deg=+45 leans the bay nose toward P1 (wall direction).
    # start_along=ang_x_margin+3 skips 3 m past the corner clearance so
    # the first bay clears the entrance gate area.
    # ------------------------------------------------------------------
    _ang_diag_half = (dims_ang["depth"] + dims_ang["width"]) / 2.0
    _ang_x_margin = _ang_diag_half * math.cos(math.radians(45.0)) + 0.5
    lot.row_along_wall(
        bay_type="angled",
        n=9,
        wall_p0=p0,
        wall_p1=p1,
        bay_angle_deg=45.0,
        side="ccw",
        start_along=_ang_x_margin + 3.0,
        pack_from="start",
    )

    # ------------------------------------------------------------------
    # Top-wall parallel group: 4 bays along the sloped top wall P3->P2.
    # Traversing P3->P2 is against CCW polygon order, so side="cw" gives
    # the correct inward normal (down-left into lot interior).
    # bay_angle_deg=+90 puts the bay nose along the wall direction (toward P2).
    # ------------------------------------------------------------------
    par_top_along_start = 9.0 + dims_par["depth"] / 2.0
    lot.row_along_wall(
        bay_type="parallel",
        n=4,
        wall_p0=p3,
        wall_p1=p2,
        bay_angle_deg=90.0,
        side="cw",
        start_along=par_top_along_start,
        pack_from="start",
    )

    # ------------------------------------------------------------------
    # Right-wall parallel group: 3 bays against the right wall (x=depth).
    # Axis-aligned -> use lot.row(). yaw=90: nose +Y, depth (8 m) along Y.
    # par_right_y_start centred on the rear wall span.
    # ------------------------------------------------------------------
    par_cx_right = depth - dims_par["width"] / 2.0 - lot.wall_gap
    _right_wall_span = (width_front - y_offset) - y_offset
    par_right_y_start = y_offset + (_right_wall_span - 3 * dims_par["depth"]) / 2.0
    par_right = lot.row(
        bay_type="parallel",
        n=3,
        anchor=(par_cx_right, par_right_y_start + dims_par["depth"] / 2.0),
        direction="north",
        yaw_deg=90.0,
        spacing=dims_par["depth"],
    )

    # ------------------------------------------------------------------
    # Spawns (3 m inside the perimeter along each heading).
    # S1: left wall entry, 3 m inward along +X from x=-5.
    # S2: diagonal bottom wall near x=38, 3 m inward along the inward normal.
    # ------------------------------------------------------------------
    wall_len = math.hypot(depth - p0[0], y_offset - p0[1])
    wdx = (depth - p0[0]) / wall_len
    wdy = (y_offset - p0[1]) / wall_len
    _spawn2_yaw = math.degrees(math.atan2(wdx, -wdy))
    _s2_x = round(38.0 + math.cos(math.radians(_spawn2_yaw)) * 3.0, 1)
    _s2_y = round(
        y_offset * (38.0 / depth) + math.sin(math.radians(_spawn2_yaw)) * 3.0, 1
    )
    lot.spawn(x=-2.0, y=width_front / 2.0, yaw_deg=0.0, primary=True)
    lot.spawn(x=_s2_x, y=_s2_y, yaw_deg=round(_spawn2_yaw, 1))

    # ------------------------------------------------------------------
    # Patrol path: 4-waypoint loop through the lower and upper aisles.
    # Corridor positions are midpoints between facing bay surfaces.
    # See documentation/detailed_notes/layout/patrol_paths.md for derivation.
    # ------------------------------------------------------------------
    perp_cluster_x_min = perp_cx_start - dims_perp["width"] / 2.0
    perp_cluster_x_max = perp_cx_start + (PERP_BAYS_PER_ROW - 0.5) * dims_perp["width"]
    par_right_inner_x = par_right.bbox[0]

    # Upper patrol y: between row B nose face and the min-y corner of the
    # top-wall bays (computed across all 4 top-wall bay rectangles).
    _row_b_nose_y = perp_row_b_cy + dims_perp["depth"] / 2.0
    _top_wall_dx = depth - p3[0]   # 49.0
    _top_wall_dy = -y_offset        # -10.0
    _top_wall_len = math.hypot(_top_wall_dx, _top_wall_dy)
    top_wdx = _top_wall_dx / _top_wall_len
    top_wdy = _top_wall_dy / _top_wall_len
    top_nx = top_wdy   # CW inward normal
    top_ny = -top_wdx
    top_par_yaw = math.degrees(math.atan2(top_wdy, top_wdx))
    par_top_normal_offset = dims_par["width"] / 2.0 + lot.wall_gap
    _top_yaw_rad = math.radians(top_par_yaw)
    _top_cos = math.cos(_top_yaw_rad)
    _top_sin = math.sin(_top_yaw_rad)
    _hd = dims_par["depth"] / 2.0
    _hw = dims_par["width"] / 2.0
    _top_par_min_y = min(
        (
            width_front
            + top_wdy * (par_top_along_start + i * dims_par["depth"])
            + top_ny * par_top_normal_offset
        )
        + _top_sin * lx + _top_cos * ly
        for i in range(4)
        for lx, ly in [(-_hd, -_hw), (_hd, -_hw), (_hd, _hw), (-_hd, _hw)]
    )
    _patrol_upper_cy = (_row_b_nose_y + _top_par_min_y) / 2.0

    # Lower patrol y: between row A nose face and the max-y (inner) edge of
    # the angled bays. The angled row is not axis-aligned so the max-y has to
    # be derived by enumerating the rotated bay corners.
    _row_a_nose_y = perp_row_a_cy - dims_perp["depth"] / 2.0
    x_enter = perp_cluster_x_min / 2.0
    x_exit = (perp_cluster_x_max + par_right_inner_x) / 2.0
    ang_yaw = (math.degrees(math.atan2(wdx, -wdy)) + 45.0) % 360.0
    ang_offset = _ang_diag_half * math.sin(math.radians(45.0)) + lot.wall_gap
    spacing = dims_ang["width"] / math.sin(math.radians(45.0))
    _ang_yaw_rad = math.radians(ang_yaw)
    _ang_cos = math.cos(_ang_yaw_rad)
    _ang_sin = math.sin(_ang_yaw_rad)
    _ang_hd = dims_ang["depth"] / 2.0
    _ang_hw = dims_ang["width"] / 2.0
    nx_b = -wdy
    ny_b = wdx
    _ang_max_y = max(
        (
            p0[1]
            + wdy * ((_ang_x_margin + 3.0) + i * spacing)
            + ny_b * ang_offset
        )
        + _ang_sin * lx + _ang_cos * ly
        for i in range(9)
        for lx, ly in [(-_ang_hd, -_ang_hw), (_ang_hd, -_ang_hw), (_ang_hd, _ang_hw), (-_ang_hd, _ang_hw)]
    )
    _patrol_lower_cy = (_row_a_nose_y + _ang_max_y) / 2.0

    patrol = PatrolPath()
    patrol.add(x_enter, _patrol_lower_cy - 4.0)
    patrol.add(x_exit, _patrol_lower_cy)
    patrol.add(x_exit, _patrol_upper_cy)
    patrol.add(x_enter, _patrol_upper_cy + 4.0)
    lot.set_patrol(patrol)

    # ------------------------------------------------------------------
    # Pedestrian zones - one strip per distinct aisle face.
    # ------------------------------------------------------------------
    margin = 0.5
    strip = 3.0

    # Zone 1: aisle below row A nose face (row A faces -Y, yaw=90 => nose +Y;
    # but the aisle "below" the cluster is on the -Y side of row A's nose).
    # Original keeps explicit bounds since the cluster spans both rows.
    lot.add_zone(
        PedestrianZone(
            x_min=perp_cluster_x_min + margin,
            x_max=perp_cluster_x_max - margin,
            y_min=perp_row_a_cy - dims_perp["depth"] / 2.0 - strip,
            y_max=perp_row_a_cy - dims_perp["depth"] / 2.0 - margin,
        )
    )

    # Zone 2: back-to-back aisle between row A back face and row B back face.
    lot.add_zone(
        PedestrianZone(
            x_min=perp_cluster_x_min + margin,
            x_max=perp_cluster_x_max - margin,
            y_min=perp_row_a_cy + dims_perp["depth"] / 2.0 + margin,
            y_max=perp_row_b_cy - dims_perp["depth"] / 2.0 - margin,
        )
    )

    # Zone 3: aisle above row B nose face (row B faces +Y, yaw=270 => nose -Y;
    # aisle on +Y side of row B back face).
    lot.add_zone(
        PedestrianZone(
            x_min=perp_cluster_x_min + margin,
            x_max=perp_cluster_x_max - margin,
            y_min=perp_row_b_cy + dims_perp["depth"] / 2.0 + margin,
            y_max=perp_row_b_cy + dims_perp["depth"] / 2.0 + strip,
        )
    )

    # Zone 4: vertical strip against the left face of the right-wall parallel group.
    lot.add_zone(
        PedestrianZone(
            x_min=par_right_inner_x - strip,
            x_max=par_right_inner_x - margin,
            y_min=par_right_y_start + margin,
            y_max=par_right_y_start + 3 * dims_par["depth"] - margin,
        )
    )

    return lot.build()
