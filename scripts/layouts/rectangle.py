"""
@file rectangle.py
@brief Rectangle parking lot floor plan (60x45 m).

Mixed bay layout to maximise training diversity across all three bay types:
  - Centre row A (lower): perpendicular row (7 bays), yaw=90 (nose +Y).
  - Centre row B (upper): 45-deg angled row (5 bays), yaw=225 (nose lower-left).
    Rows face each other across a shared central aisle (6 m gap between back faces).
  - Bottom wall left group (y=0): perpendicular row (8 bays), yaw=90.
  - Bottom wall right group (y=0): 45-deg angled row (7 bays), yaw=135.
    Both groups split by a ~12 m gap at x=30 for Spawn 2 entry.
  - Parallel bays: 3 along top wall (yaw=0) + 3 along right wall (yaw=270).

Three spawn transforms:
  - Primary   (S1): left wall centre (x=0), facing +X into the lot.
  - Secondary (S2): bottom wall centre (y=0), facing +Y through the bottom-wall gap.
  - Tertiary  (S3): top wall at local x=45 (world x~-155), facing -Y into the lot.

Bay dimensions follow German EAR 05 / EU harmonised practice (FGSV 2005).
Used as a training layout (ood=False).
"""

import math
from typing import Any, Dict, List

from scripts.layouts.common import (
    BAY_DIMS,
    BAYS_PER_TYPE,
    ang_offset_from_wall,
    validate_bays_in_polygon,
    warn_narrow_corridors,
)

# World-frame origin used in multi-layout generation (Town05_Opt flat area).
ORIGIN_X = -200.0
ORIGIN_Y = 0.0
ORIGIN_Z = 0.3
HEADING_DEG = 0.0
OOD = False

# Number of perp bays in the left bottom-wall group.
_PERP_WALL_BAYS = 10
# Number of angled bays in the right bottom-wall group (7 fills the gap from x~33 to x=60).
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
        {"x": 0.0, "y": 0.0},
        {"x": depth, "y": 0.0},
        {"x": depth, "y": width},
        {"x": 0.0, "y": width},
    ]

    dims_perp = BAY_DIMS["perpendicular"]
    dims_ang = BAY_DIMS["angled"]
    dims_par = BAY_DIMS["parallel"]

    # ------------------------------------------------------------------
    # Centre: mixed back-to-back rows, axis-aligned along X.
    # Row A (lower): perpendicular, yaw=90  (nose toward +Y).
    # Row B (upper): 45-deg angled,  yaw=225 (nose toward lower-left).
    # Extra back-gap of 6 m between the two back faces for a wider central aisle.
    # ------------------------------------------------------------------
    _CENTRE_BACK_GAP = 6.0  # extra separation between back faces (wider than EAR aisle)
    centre_mid_y = width / 2.0

    # Row A: perp. Back face at mid_y - gap/2; centre_y = back_face - depth/2.
    centre_row_a_cy = centre_mid_y - _CENTRE_BACK_GAP / 2.0 - dims_perp["depth"] / 2.0
    # Row B: angled. Back face at mid_y + gap/2; centre_y = back_face + depth/2 + 1 m shift.
    centre_row_b_cy = centre_mid_y + _CENTRE_BACK_GAP / 2.0 + dims_ang["depth"] / 2.0 + 1.0

    # Row A uses perp spacing (2.5 m); Row B uses angled spacing (width/sin45 = 3.536 m).
    # Each row is centred independently on depth/2 so both clusters are visually centred.
    _ang_spacing = dims_ang["width"] / math.sin(math.radians(45.0))

    _CENTRE_PERP_BAYS = BAYS_PER_TYPE + 7  # 12 bays: +1 each side vs. previous 10
    _CENTRE_ANG_BAYS = BAYS_PER_TYPE + 3   # 8 bays:  +2 left, +1 right vs. original 5
    # Shift centre left so extra bays land 2 on the left, 1 on the right.
    # New perp cluster centre = old centre - 1.25 m (half a perp bay width).
    _CENTRE_X_OFFSET = -7.25

    perp_cluster_span = _CENTRE_PERP_BAYS * dims_perp["width"]
    perp_cx_start = (
        depth / 2.0 + _CENTRE_X_OFFSET
        - perp_cluster_span / 2.0
        + dims_perp["width"] / 2.0
    )

    ang_cluster_span_c = _CENTRE_ANG_BAYS * _ang_spacing
    ang_cx_start_c = (
        depth / 2.0 + _CENTRE_X_OFFSET
        - ang_cluster_span_c / 2.0
        + _ang_spacing / 2.0
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
    # Left group:  8 perpendicular bays, yaw=90, x = 1.25 .. 21.25.
    # Right group: 7 angled (45-deg) bays, yaw=135, packed right-to-left
    #              flush against x=depth, leftmost bay clears x=30.
    # ------------------------------------------------------------------
    _WALL_GAP = 0.5  # minimum clearance between bay back face and perimeter wall/cones
    perp_cy = dims_perp["depth"] / 2.0 + _WALL_GAP  # back edge 0.5 m clear of y=0 cones
    perp_left_x_start = dims_perp["width"] / 2.0 + 1.0

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

    # Right angled group: packed right-to-left so the rightmost bay corner
    # sits flush against the right wall (x=depth).  At yaw=135 the maximum-x
    # corner of a bay is at cx + (hd + hw) * cos(45), so:
    #   cx_rightmost = depth - (hd_ang + hw_ang) * cos(45)
    # Subsequent bays step left by width/sin(45) = 3.536 m.
    _ang_off = ang_offset_from_wall(dims_ang["depth"], dims_ang["width"]) + _WALL_GAP
    _ang_spacing_b = dims_ang["width"] / math.sin(math.radians(45.0))
    _ang_cx_rightmost = depth - (dims_ang["depth"] / 2.0 + dims_ang["width"] / 2.0) * math.cos(
        math.radians(45.0)
    ) - _WALL_GAP
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
    # Parallel bays: short-side adjacent (2.5 m deep from wall, 8 m along wall).
    # Group A: 3 bays along top wall (y=width), yaw=0 (nose +X).
    #          Placed at the left side (x=2..26) to keep x=45 clear for Spawn 3.
    # Group B: 3 bays along right wall (x=depth), yaw=270 (nose -Y).
    # Group C: 3 bays in a second column 6 m to the left of Group B (nose -Y),
    #          same y-range, forming a paired parallel aisle.
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
    # Shifted 3 m down from the top-wall edge to leave clearance under Spawn 3.
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

    # Group C: second parallel column, 9 m inner-to-inner gap from Group B.
    # inner_x of Group B = par_cx_right - dims_par["width"]/2 = depth - 2.5
    # inner_x of Group C = (depth - 2.5) - 9.0  ->  cx = inner_x_C + 1.25
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
    all_bays = centre_perp_bays + centre_ang_bays + perp_bays + bottom_ang_bays + par_bays

    validate_bays_in_polygon(all_bays, corners, "rectangle")
    warn_narrow_corridors(all_bays, "rectangle")

    # ------------------------------------------------------------------
    # Spawn transforms.
    # S1: left wall (x=0), centred vertically, facing +X.
    # S2: bottom wall (y=0), in the gap between the two bottom bay groups, facing +Y.
    # S3: top wall (y=width) at local x=45 (world x=-155), facing -Y into lot.
    # ------------------------------------------------------------------
    spawn = {"x": 0.0, "y": width / 2.0, "yaw_deg": 0.0}
    spawn2 = {"x": depth / 2.0, "y": 0.0, "yaw_deg": 90.0}
    spawn3 = {"x": 45.0, "y": width, "yaw_deg": 270.0}

    # ------------------------------------------------------------------
    # Patrol path: 8-waypoint loop tracing the driving aisles.
    #
    # Key geometry references (all local frame):
    #   centre_perp_x_min / _max : left/right extent of centre perp cluster
    #   centre_ang_x_min / _max  : left/right extent of centre angled cluster
    #   centre_perp_front_y      : nose face of row A (faces -Y, lower aisle)
    #   bottom_perp_nose_y       : nose face of bottom-wall perp bays (faces +Y)
    #   bottom_ang_nose_y        : nose face of bottom-wall angled bays (faces upper)
    #   par_top_front_y          : nose face of top parallel group (faces +X)
    #   centre_ang_top_y         : top face of centre angled row B
    #   par_col_aisle_x          : midpoint between the two right parallel columns
    #   par_top_back_y           : back face of top parallel bays (flush with top wall)
    # ------------------------------------------------------------------
    centre_cluster_x_min = ang_cx_start_c - _ang_spacing / 2.0
    centre_cluster_x_max = ang_cx_start_c + (_CENTRE_ANG_BAYS - 0.5) * _ang_spacing
    perp_cluster_x_min = perp_cx_start - dims_perp["width"] / 2.0
    perp_cluster_x_max = perp_cx_start + (_CENTRE_PERP_BAYS - 0.5) * dims_perp["width"]

    # Left wall to left edge of centre cluster: equidistant midpoint x.
    centre_left_edge = min(perp_cluster_x_min, centre_cluster_x_min)
    patrol_x_left = centre_left_edge / 2.0

    # Nose face of centre row A (perp, faces -Y into the lower aisle).
    centre_perp_front_y = centre_row_a_cy - dims_perp["depth"] / 2.0
    # Nose face of bottom-wall perp bays (faces +Y).
    bottom_perp_nose_y = perp_cy + dims_perp["depth"] / 2.0
    # Nose face of bottom-wall angled bays (centre_y + depth/2 projects upward).
    bottom_ang_nose_y = _ang_off + dims_ang["depth"] / 2.0
    # Lower aisle midpoint: between bottom-wall bay noses and centre row A nose.
    lower_aisle_max_nose_y = max(bottom_perp_nose_y, bottom_ang_nose_y)
    patrol_y_lower = (lower_aisle_max_nose_y + centre_perp_front_y) / 2.0

    # Midpoint aisle between the two right parallel columns (col2 and col1/right).
    par_col1_inner_x = par_cx_right - dims_par["width"] / 2.0
    par_col2_inner_x = par_cx_col2 + dims_par["width"] / 2.0
    patrol_x_par_aisle = (par_col1_inner_x + par_col2_inner_x) / 2.0

    # Top parallel bays back face (bays at par_cy_top, yaw=0, width=2.5 along Y).
    par_top_back_y = par_cy_top + dims_par["width"] / 2.0   # flush with top wall = width
    par_top_front_y = par_cy_top - dims_par["width"] / 2.0  # nose face (towards lot interior)
    # Top wall to top parallel back face: equidistant midpoint y (above bays).
    patrol_y_top = (par_top_back_y + width) / 2.0

    # Centre angled row B top face.
    centre_ang_top_y = centre_row_b_cy + dims_ang["depth"] / 2.0
    # Between top parallel nose face and centre angled row B top face.
    patrol_y_upper = (par_top_front_y + centre_ang_top_y) / 2.0

    # Right extent of the top parallel group.
    par_top_x_end = par_top_x_start + 4 * dims_par["depth"]
    # Right extent of centre cluster.
    centre_right_edge = max(perp_cluster_x_max, centre_cluster_x_max)

    # Top aisle y: equidistant between top parallel bays front face and top of right columns.
    par_col_top_y = par_right_y_start + 3 * dims_par["depth"]
    patrol_y_top_aisle = (par_top_front_y + par_col_top_y) / 2.0

    # Diagonal turn point: x equidistant between right end of top parallel bays
    # and left (outer) face of inner parallel column (col2).
    par_col2_outer_x = par_cx_col2 - dims_par["width"] / 2.0
    patrol_diag_start_x = (par_top_x_end + par_col2_outer_x) / 2.0

    # 45-deg diagonal down-left from (patrol_diag_start_x, patrol_y_top_aisle) until
    # y reaches patrol_y_upper (equidistant top parallel front face / centre angled top).
    # delta_y = patrol_y_top_aisle - patrol_y_upper, so delta_x = same value leftward.
    patrol_diag_end_x = patrol_diag_start_x - (patrol_y_top_aisle - patrol_y_upper)

    # Patrol waypoints (CCW loop):
    # WP1: left corridor at lower aisle height.
    # WP2: lower aisle sweeps right to parallel column aisle.
    # WP3: parallel column aisle climbs to top aisle height.
    # WP4: top aisle sweeps left to diagonal turn point.
    # WP5: 45-deg diagonal down-left to patrol_y_upper.
    # WP6: upper aisle sweeps left to left corridor x.
    # WP7: drop back to lower aisle height to close loop.
    patrol = [
        # WP1: left corridor at lower aisle height.
        {"x": patrol_x_left, "y": patrol_y_lower},
        # WP2: lower aisle sweeps right to parallel column aisle.
        {"x": patrol_x_par_aisle, "y": patrol_y_lower},
        # WP3: parallel column aisle climbs to top aisle height.
        {"x": patrol_x_par_aisle, "y": patrol_y_top_aisle},
        # WP4: top aisle sweeps left to the diagonal turn point.
        {"x": patrol_diag_start_x, "y": patrol_y_top_aisle},
        # WP5: 45-deg diagonal down-left to upper aisle height.
        {"x": patrol_diag_end_x, "y": patrol_y_upper},
        # WP6: upper aisle sweeps left to left corridor x.
        {"x": patrol_x_left, "y": patrol_y_upper},
        # WP7: drop back to lower aisle height to close loop.
        {"x": patrol_x_left, "y": patrol_y_lower},
    ]

    PED_STRIP = 3.0
    # Inset every zone edge that touches a bay face so pedestrians cannot
    # overlap the bay footprint and clip through parked cars.
    _PED_MARGIN = 0.5

    # Cluster extents already computed in patrol section above; derive combined bounds.
    centre_x_min = min(perp_cluster_x_min, centre_cluster_x_min)
    centre_x_max = max(perp_cluster_x_max, centre_cluster_x_max)

    # Bottom-wall group extents.
    bottom_perp_x_min = perp_left_x_start - dims_perp["width"] / 2.0
    bottom_perp_x_max = perp_left_x_start + (_PERP_WALL_BAYS - 0.5) * dims_perp["width"]
    bottom_ang_x_min = _ang_cx_rightmost - (_ANG_WALL_BAYS - 1) * _ang_spacing_b - (
        dims_ang["depth"] / 2.0 + dims_ang["width"] / 2.0
    ) * math.cos(math.radians(45.0))
    # Parallel column extents.
    par_col_y_min = par_right_y_start
    par_col_y_max = par_right_y_start + 3 * dims_par["depth"]
    # par_col2_inner_x already computed in patrol section above.
    par_col2_outer_x = par_cx_col2 - dims_par["width"] / 2.0

    ped_zones = [
        # Zone 1: aisle in front of bottom-wall perp group nose faces (bays face +Y).
        # Nose face = perp_cy + depth/2 = dims_perp["depth"] + _WALL_GAP.
        {
            "x_min": bottom_perp_x_min + _PED_MARGIN,
            "x_max": bottom_perp_x_max - _PED_MARGIN,
            "y_min": dims_perp["depth"] + _WALL_GAP + _PED_MARGIN,
            "y_max": dims_perp["depth"] + _WALL_GAP + PED_STRIP,
        },
        # Zone 2: aisle in front of bottom-wall angled group nose faces (bays face upper-left).
        {
            "x_min": bottom_ang_x_min + _PED_MARGIN,
            "x_max": _ang_cx_rightmost + _PED_MARGIN,
            "y_min": _ang_off + dims_ang["depth"] / 2.0 + _PED_MARGIN,
            "y_max": _ang_off + dims_ang["depth"] / 2.0 + PED_STRIP,
        },
        # Zone 3: central back-to-back aisle between row A back face and row B back face.
        {
            "x_min": centre_x_min + _PED_MARGIN,
            "x_max": centre_x_max - _PED_MARGIN,
            "y_min": centre_row_a_cy + dims_perp["depth"] / 2.0 + _PED_MARGIN,
            "y_max": centre_row_b_cy - dims_ang["depth"] / 2.0 - _PED_MARGIN,
        },
        # Zone 5: aisle above centre angled row nose face (row B faces lower-left at yaw=225).
        {
            "x_min": centre_x_min + _PED_MARGIN,
            "x_max": centre_x_max - _PED_MARGIN,
            "y_min": centre_row_b_cy + dims_ang["depth"] / 2.0 + _PED_MARGIN,
            "y_max": centre_row_b_cy + dims_ang["depth"] / 2.0 + PED_STRIP,
        },
        # Zone 6: strip to the left of the inner parallel column (col2 left face).
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
        "extra_spawns": [spawn2, spawn3],
        "patrol_waypoints": patrol,
        "pedestrian_zones": ped_zones,
    }
