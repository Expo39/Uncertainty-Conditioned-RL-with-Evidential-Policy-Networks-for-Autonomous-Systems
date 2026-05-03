"""
@file rectangle.py
@brief Rectangle parking lot floor plan (60x45 m).

Mixed bay layout to maximise training diversity across all three bay types:
  - Centre row A (lower): 12 perpendicular bays, yaw=90 (nose +Y).
  - Centre row B (upper):  8 angled bays (45 deg), yaw=225 (nose lower-left).
    Rows share a central aisle (6 m back-face gap); both are left-shifted so
    extra bays land 2 on the left, 1 on the right vs. a centred cluster.
  - Bottom wall left group  (y=0): 10 perpendicular bays, yaw=90.
  - Bottom wall right group (y=0):  7 angled bays (45 deg), yaw=135.
    Both groups are split by a ~12 m gap around x=30 for Spawn 2 entry.
  - Top wall parallel group:   4 bays, yaw=0  (nose +X), left side of top wall.
  - Right wall parallel group: 3 bays, yaw=270 (nose -Y), against x=60.
  - Inner parallel column:     3 bays, yaw=270 (nose -Y), 9 m aisle-to-aisle
    inward from the right-wall group, forming a back-to-back parallel pair.
  - Motorcycle bays: 2 narrow slots in the top-right corner (always_empty=True).

Three spawn transforms (3 m inside the perimeter along each heading):
  - Primary   (S1): left wall entry (x=3, y=22.5), facing +X into the lot.
  - Secondary (S2): bottom wall entry at x=30, y=3, facing +Y.
  - Tertiary  (S3): top wall entry at local x=45, facing -Y into the lot.

Bay dimensions follow German EAR 05 / EU harmonised practice (FGSV 2005).
Used as a training layout (OOD=False).
"""

import math
from typing import Any, Dict, List

from scripts.layouts.common import (
    _PED_MARGIN,
    _WALL_GAP,
    BAY_DIMS,
    BAYS_PER_TYPE,
    PED_STRIP,
    ang_offset_from_wall,
    validate_bays_in_polygon,
    warn_narrow_corridors,
)

# World-frame origin chosen so the primary spawn lands at CARLA (0,0),
# ensuring consistent coordinate frame alignment at the origin.
# to_world_frame() negates Y for CARLA's left-handed frame:
#   carla_y = -(local_y + (-ORIGIN_Y))  =>  carla_y = -(22.5 + (-22.5)) = 0.
# spawn_local = (-2.0, 22.5) -> CARLA (0.0, 0.0).
ORIGIN_X = 2.0
ORIGIN_Y = 22.5
ORIGIN_Z = 0.3
HEADING_DEG = 0.0
OOD = False


# Number of perp bays in the bottom-wall left group (extended to fill x=-5..wall).
_PERP_WALL_BAYS = 12
# Number of angled bays in the bottom-wall right group (7 fills the gap x~33..60).
_ANG_WALL_BAYS = 7


def generate() -> Dict[str, Any]:
    """
    @brief Build the rectangle layout in local frame.
    @return Layout dict with keys: corners, bays, spawn, extra_spawns,
            patrol_waypoints, pedestrian_zones.
    """
    width = 45.0
    depth = 60.0

    corners = [
        {"x": -5.0, "y": 0.0},
        {"x": depth, "y": 0.0},
        {"x": depth, "y": width},
        {"x": -5.0, "y": width},
    ]

    dims_perp = BAY_DIMS["perpendicular"]  # width=2.5, depth=5.0
    dims_ang = BAY_DIMS["angled"]  # width=2.5, depth=5.4
    dims_par = BAY_DIMS["parallel"]  # width=2.5, depth=8.0

    # ------------------------------------------------------------------
    # Centre: mixed back-to-back rows, axis-aligned along X.
    # Row A (lower): 12 perpendicular bays, yaw=90 (nose +Y).
    # Row B (upper):  8 angled bays (45 deg), yaw=225 (nose lower-left).
    # _CENTRE_BACK_GAP: extra separation between the two back faces so the
    # shared central aisle is wider than a standard EAR minimum.
    # Both rows are left-shifted by _CENTRE_X_OFFSET relative to lot centre
    # so the extra bays land 2 on the left, 1 on the right.
    # ------------------------------------------------------------------
    _CENTRE_BACK_GAP = 6.0
    _CENTRE_PERP_BAYS = BAYS_PER_TYPE + 7  # 12 bays
    _CENTRE_ANG_BAYS = BAYS_PER_TYPE + 3  # 8 bays
    _CENTRE_X_OFFSET = -7.25  # left-shift from lot centre

    centre_mid_y = width / 2.0
    centre_row_a_cy = centre_mid_y - _CENTRE_BACK_GAP / 2.0 - dims_perp["depth"] / 2.0
    centre_row_b_cy = (
        centre_mid_y + _CENTRE_BACK_GAP / 2.0 + dims_ang["depth"] / 2.0 + 1.0
    )

    _ang_spacing = dims_ang["width"] / math.sin(math.radians(45.0))

    perp_cluster_span = _CENTRE_PERP_BAYS * dims_perp["width"]
    perp_cx_start = (
        depth / 2.0
        + _CENTRE_X_OFFSET
        - perp_cluster_span / 2.0
        + dims_perp["width"] / 2.0
    )

    ang_cluster_span_c = _CENTRE_ANG_BAYS * _ang_spacing
    ang_cx_start_c = (
        depth / 2.0 + _CENTRE_X_OFFSET - ang_cluster_span_c / 2.0 + _ang_spacing / 2.0
    )

    centre_perp_bays: List[Dict] = []
    for i in range(_CENTRE_PERP_BAYS):
        centre_perp_bays.append(
            {
                "bay_type": "perpendicular",
                "local_x": perp_cx_start + i * dims_perp["width"],
                "local_y": centre_row_a_cy,
                "local_yaw_deg": 90.0,
                "width": dims_perp["width"],
                "depth": dims_perp["depth"],
            }
        )

    centre_ang_bays: List[Dict] = []
    for i in range(_CENTRE_ANG_BAYS):
        centre_ang_bays.append(
            {
                "bay_type": "angled",
                "local_x": ang_cx_start_c + i * _ang_spacing,
                "local_y": centre_row_b_cy,
                "local_yaw_deg": 225.0,
                "width": dims_ang["width"],
                "depth": dims_ang["depth"],
            }
        )

    # ------------------------------------------------------------------
    # Bottom wall (y=0): split by a ~12 m gap around x=30 for Spawn 2.
    # Left group:  10 perpendicular bays, yaw=90, packed from x=1.25 rightward.
    # Right group:  7 angled bays (45 deg), yaw=135, packed right-to-left
    #               flush against x=depth; leftmost bay clears the gap.
    # ------------------------------------------------------------------
    perp_cy = dims_perp["depth"] / 2.0 + _WALL_GAP
    # Start from the new left wall (x=-5 local) with standard clearance.
    perp_left_x_start = -5.0 + _WALL_GAP + dims_perp["width"] / 2.0

    perp_bays: List[Dict] = []
    for i in range(_PERP_WALL_BAYS):
        perp_bays.append(
            {
                "bay_type": "perpendicular",
                "local_x": perp_left_x_start + i * dims_perp["width"],
                "local_y": perp_cy,
                "local_yaw_deg": 90.0,
                "width": dims_perp["width"],
                "depth": dims_perp["depth"],
            }
        )

    # Right angled group packed right-to-left. At yaw=135 the rightmost bay
    # corner sits at cx + (hd + hw)*cos(45), so the rightmost centre is:
    #   cx = depth - (hd + hw)*cos(45) - _WALL_GAP
    _ang_off = ang_offset_from_wall(dims_ang["depth"], dims_ang["width"]) + _WALL_GAP
    _ang_spacing_b = dims_ang["width"] / math.sin(math.radians(45.0))
    _ang_cx_rightmost = (
        depth
        - (dims_ang["depth"] / 2.0 + dims_ang["width"] / 2.0)
        * math.cos(math.radians(45.0))
        - _WALL_GAP
    )
    bottom_ang_bays: List[Dict] = []
    for i in range(_ANG_WALL_BAYS):
        bottom_ang_bays.append(
            {
                "bay_type": "angled",
                "local_x": _ang_cx_rightmost - i * _ang_spacing_b,
                "local_y": _ang_off,
                "local_yaw_deg": 135.0,
                "width": dims_ang["width"],
                "depth": dims_ang["depth"],
            }
        )

    # ------------------------------------------------------------------
    # Parallel bays.
    # Top group (Group A):  4 bays along the top wall (y=width), yaw=0 (nose +X),
    #   placed at the left side so x=45 stays clear for Spawn 3.
    # Right column (Group B): 3 bays against the right wall (x=depth), yaw=270,
    #   shifted 3 m down from the top edge to leave clearance under Spawn 3.
    # Inner column (Group C): 3 bays 9 m (aisle-to-aisle) inward from Group B,
    #   same y-range and yaw=270, forming a back-to-back parallel pair.
    # ------------------------------------------------------------------
    par_cy_top = width - dims_par["width"] / 2.0 - _WALL_GAP
    par_top_x_start = 5.0
    par_bays_top: List[Dict] = []
    for i in range(4):
        par_bays_top.append(
            {
                "bay_type": "parallel",
                "local_x": par_top_x_start
                + dims_par["depth"] / 2.0
                + i * dims_par["depth"],
                "local_y": par_cy_top,
                "local_yaw_deg": 0.0,
                "width": dims_par["width"],
                "depth": dims_par["depth"],
            }
        )

    par_cx_right = depth - dims_par["width"] / 2.0 - _WALL_GAP
    par_right_y_start = width - 2.0 - 3 * dims_par["depth"] - 6.0
    par_bays_right: List[Dict] = []
    for i in range(3):
        par_bays_right.append(
            {
                "bay_type": "parallel",
                "local_x": par_cx_right,
                "local_y": par_right_y_start
                + dims_par["depth"] / 2.0
                + i * dims_par["depth"],
                "local_yaw_deg": 270.0,
                "width": dims_par["width"],
                "depth": dims_par["depth"],
            }
        )

    # Inner column: 9 m aisle-to-aisle gap from Group B.
    # Group B inner face = par_cx_right - width/2. Group C outer face = that - 9.0.
    par_cx_col2 = par_cx_right - dims_par["width"] - 9.0
    par_bays_col2: List[Dict] = []
    for i in range(3):
        par_bays_col2.append(
            {
                "bay_type": "parallel",
                "local_x": par_cx_col2,
                "local_y": par_right_y_start
                + dims_par["depth"] / 2.0
                + i * dims_par["depth"],
                "local_yaw_deg": 270.0,
                "width": dims_par["width"],
                "depth": dims_par["depth"],
            }
        )

    par_bays = par_bays_top + par_bays_right + par_bays_col2

    # ------------------------------------------------------------------
    # Motorcycle bays: 2 narrow slots in the top-right corner (always_empty=True).
    # yaw=0: depth (3.0 m) along X, width (1.5 m) along Y.
    # Back face 0.5 m clear of right wall (x=60): cx = 60 - 1.5 - 0.5 = 58.0.
    # Top bay top edge 0.5 m clear of top wall (y=45): cy = 45 - 0.75 - 0.5 = 43.75.
    # ------------------------------------------------------------------
    _MOTO_CX = depth - 3.0 / 2.0 - _WALL_GAP
    _MOTO_CY_TOP = width - 1.5 / 2.0 - _WALL_GAP
    motorcycle_bays: List[Dict] = [
        {
            "bay_type": "motorcycle",
            "local_x": _MOTO_CX,
            "local_y": _MOTO_CY_TOP,
            "local_yaw_deg": 0.0,
            "width": 1.5,
            "depth": 3.0,
            "always_empty": True,
            "occupant": "Kawasaki Ninja",
        },
        {
            "bay_type": "motorcycle",
            "local_x": _MOTO_CX,
            "local_y": _MOTO_CY_TOP - 1.5,
            "local_yaw_deg": 0.0,
            "width": 1.5,
            "depth": 3.0,
            "always_empty": True,
            "occupant": "Yamaha YZF-R",
        },
    ]

    all_bays = (
        centre_perp_bays
        + centre_ang_bays
        + perp_bays
        + bottom_ang_bays
        + par_bays
        + motorcycle_bays
    )

    validate_bays_in_polygon(all_bays, corners, "rectangle")
    warn_narrow_corridors(all_bays, "rectangle")

    # ------------------------------------------------------------------
    # Spawn transforms (3 m inside the perimeter along each heading).
    # S1: left wall entry, 3 m inward along +X from x=-5 (wall at x=-5).
    # S2: bottom wall entry at x=30, 3 m inward along +Y from y=0.
    # ------------------------------------------------------------------
    spawn = {"x": -2.0, "y": width / 2.0, "yaw_deg": 0.0}
    spawn2 = {"x": depth / 2.0, "y": 3.0, "yaw_deg": 90.0}

    # Patrol path: 6-waypoint CCW loop tracing the driving aisles.
    # Corridor positions are midpoints between facing bay surfaces.
    # See documentation/detailed_notes/layout/patrol_paths.md for derivation.
    centre_cluster_x_min = ang_cx_start_c - _ang_spacing / 2.0
    centre_cluster_x_max = ang_cx_start_c + (_CENTRE_ANG_BAYS - 0.5) * _ang_spacing
    perp_cluster_x_min = perp_cx_start - dims_perp["width"] / 2.0
    perp_cluster_x_max = perp_cx_start + (_CENTRE_PERP_BAYS - 0.5) * dims_perp["width"]

    centre_left_edge = min(perp_cluster_x_min, centre_cluster_x_min)
    patrol_x_left = centre_left_edge / 2.0

    centre_perp_front_y = centre_row_a_cy - dims_perp["depth"] / 2.0
    bottom_perp_nose_y = perp_cy + dims_perp["depth"] / 2.0
    bottom_ang_nose_y = _ang_off + dims_ang["depth"] / 2.0
    lower_aisle_max_nose = max(bottom_perp_nose_y, bottom_ang_nose_y)
    patrol_y_lower = (lower_aisle_max_nose + centre_perp_front_y) / 2.0

    par_col1_inner_x = par_cx_right - dims_par["width"] / 2.0
    par_col2_inner_x = par_cx_col2 + dims_par["width"] / 2.0
    par_col2_outer_x = par_cx_col2 - dims_par["width"] / 2.0
    patrol_x_par_aisle = (par_col1_inner_x + par_col2_inner_x) / 2.0

    par_top_front_y = par_cy_top - dims_par["width"] / 2.0
    centre_ang_top_y = centre_row_b_cy + dims_ang["depth"] / 2.0
    patrol_y_upper = (par_top_front_y + centre_ang_top_y) / 2.0

    par_top_x_end = par_top_x_start + 4 * dims_par["depth"]
    par_col_top_y = par_right_y_start + 3 * dims_par["depth"]
    patrol_y_top_aisle = (par_top_front_y + par_col_top_y) / 2.0

    patrol_diag_start_x = (par_top_x_end + par_col2_outer_x) / 2.0
    # 45-deg diagonal: dy = patrol_y_top_aisle - patrol_y_upper -> dx = same leftward.
    patrol_diag_end_x = patrol_diag_start_x - (patrol_y_top_aisle - patrol_y_upper)

    patrol = [
        {"x": patrol_x_left, "y": patrol_y_lower},  # WP1: left corridor, lower level
        {"x": patrol_x_par_aisle, "y": patrol_y_lower},  # WP2: lower aisle to right
        {
            "x": patrol_x_par_aisle,
            "y": patrol_y_top_aisle,
        },  # WP3: parallel column aisle, up
        {
            "x": patrol_diag_start_x,
            "y": patrol_y_top_aisle,
        },  # WP4: top aisle to diagonal start
        {"x": patrol_diag_end_x, "y": patrol_y_upper},  # WP5: 45-deg diagonal down-left
        {"x": patrol_x_left, "y": patrol_y_upper},  # WP6: upper aisle to left
        # WP1 is not repeated here: the cyclic modulo wrap in _spawn_npc_patrol /
        # _update_patrol_npcs closes the loop automatically.
    ]

    # ------------------------------------------------------------------
    # Pedestrian zones - one strip per distinct aisle face.
    # PED_STRIP and _PED_MARGIN are imported from common.py.
    # ------------------------------------------------------------------

    centre_x_min = min(perp_cluster_x_min, centre_cluster_x_min)
    centre_x_max = max(perp_cluster_x_max, centre_cluster_x_max)

    bottom_perp_x_min = perp_left_x_start - dims_perp["width"] / 2.0
    bottom_perp_x_max = perp_left_x_start + (_PERP_WALL_BAYS - 0.5) * dims_perp["width"]
    bottom_ang_x_min = (
        _ang_cx_rightmost
        - (_ANG_WALL_BAYS - 1) * _ang_spacing_b
        - (dims_ang["depth"] / 2.0 + dims_ang["width"] / 2.0)
        * math.cos(math.radians(45.0))
    )

    par_col_y_min = par_right_y_start
    par_col_y_max = par_right_y_start + 3 * dims_par["depth"]

    ped_zones = [
        # Zone 1: aisle in front of bottom-wall perp group (bays face +Y).
        {
            "x_min": bottom_perp_x_min + _PED_MARGIN,
            "x_max": bottom_perp_x_max - _PED_MARGIN,
            "y_min": dims_perp["depth"] + _WALL_GAP + _PED_MARGIN,
            "y_max": dims_perp["depth"] + _WALL_GAP + PED_STRIP,
        },
        # Zone 2: aisle in front of bottom-wall angled group (bays face upper-left).
        {
            "x_min": bottom_ang_x_min + _PED_MARGIN,
            "x_max": _ang_cx_rightmost + _PED_MARGIN,
            "y_min": _ang_off + dims_ang["depth"] / 2.0 + _PED_MARGIN,
            "y_max": _ang_off + dims_ang["depth"] / 2.0 + PED_STRIP,
        },
        # Zone 3: central aisle between row A back face and row B back face.
        {
            "x_min": centre_x_min + _PED_MARGIN,
            "x_max": centre_x_max - _PED_MARGIN,
            "y_min": centre_row_a_cy + dims_perp["depth"] / 2.0 + _PED_MARGIN,
            "y_max": centre_row_b_cy - dims_ang["depth"] / 2.0 - _PED_MARGIN,
        },
        # Zone 4: aisle above centre angled row B nose face (faces lower-left, yaw=225).
        {
            "x_min": centre_x_min + _PED_MARGIN,
            "x_max": centre_x_max - _PED_MARGIN,
            "y_min": centre_row_b_cy + dims_ang["depth"] / 2.0 + _PED_MARGIN,
            "y_max": centre_row_b_cy + dims_ang["depth"] / 2.0 + PED_STRIP,
        },
        # Zone 5: strip to the left of the inner parallel column (Group C outer face).
        {
            "x_min": par_col2_outer_x - PED_STRIP,
            "x_max": par_col2_outer_x - _PED_MARGIN,
            "y_min": par_col_y_min + _PED_MARGIN,
            "y_max": par_col_y_max - _PED_MARGIN,
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
