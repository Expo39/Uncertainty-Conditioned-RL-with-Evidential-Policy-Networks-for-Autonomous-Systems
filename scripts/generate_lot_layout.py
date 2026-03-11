"""
@file generate_lot_layout.py
@brief Offline parking lot layout generator for the uncertainty-conditioned RL project.

Generates pre-computed world-frame lot geometry (corners, bay positions, spawn transform,
patrol waypoints, pedestrian zones) from shape dimensions and a world-frame origin.
Writes layout YAML files consumed directly by CARLAParkingEnv and optionally
produces bird's-eye PNG plots for visual inspection.

Three floor plan shapes are supported:

  rectangle   -- Standard axis-aligned rectangle, widened to hold 5-bay rows.
  trapezoid   -- Wider at the entrance end, narrower at the rear. One pair of
                 non-parallel sides gives a subtly different LiDAR wall profile.
  irregular_a -- Five-sided polygon (OOD floor plan). Held out from training.
                 One corner cut diagonally to break axis-aligned symmetry.

Bay dimensions follow German EAR 05 / EU harmonised practice (FGSV 2005):

  Perpendicular (90 deg): 2.5 m wide x 5.0 m deep, 6.0 m aisle
  Angled (45 deg):        2.5 m wide x 5.4 m deep, 3.6 m aisle
  Parallel (pull-in):     2.5 m wide x 8.0 m deep, 4.0 m aisle

Layout YAML format (consumed by CARLAParkingEnv):

  floor_plan: rectangle
  origin: {x: 0.0, y: 0.0, z: 0.3, heading_deg: 0.0}
  corners: [{x: ..., y: ...}, ...]
  spawn_transform: {x: ..., y: ..., z: 0.3, yaw_deg: ...}
  bays:
    - {id: perp_0, bay_type: perpendicular, x: ..., y: ..., yaw_deg: ...,
       width: 2.5, depth: 5.0}
    ...
  patrol_waypoints: [{x: ..., y: ...}, ...]
  pedestrian_zones: [{x_min: ..., x_max: ..., y_min: ..., y_max: ...}]

Usage:
  # Generate all three layouts with placeholder origins (no CARLA needed):
  make generate-layouts

  # Generate one layout explicitly:
  python scripts/generate_lot_layout.py \\
      --shape rectangle --origin 0 0 0.3 --heading 0 \\
      --output configs/layouts/rectangle.yaml \\
      --plot outputs/layouts/rectangle.png

  # After recording CARLA world origins via --mark mode:
  python scripts/generate_lot_layout.py \\
      --shape rectangle --origin -200.0 0.0 0.3 --heading 0 \\
      --output configs/layouts/rectangle.yaml \\
      --plot outputs/layouts/rectangle.png
"""

import argparse
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

# ---------------------------------------------------------------------------
# Bay dimension constants (German EAR 05)
# ---------------------------------------------------------------------------

BAY_DIMS: Dict[str, Dict[str, float]] = {
    "perpendicular": {"width": 2.5, "depth": 5.0, "aisle": 6.0},
    "angled": {"width": 2.5, "depth": 5.4, "aisle": 3.6},
    "parallel": {"width": 2.5, "depth": 8.0, "aisle": 4.0},
}

BAYS_PER_TYPE = 5  # Exactly 5 bays per type per floor plan
FLOOR_Z = 0.3  # CARLA ground z for all lots


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def _rotate(
    x: float,
    y: float,
    heading_rad: float,
) -> Tuple[float, float]:
    """
    @brief Rotate point (x, y) by heading_rad about the origin.
    @param x: Local x coordinate.
    @param y: Local y coordinate.
    @param heading_rad: Rotation angle in radians (CCW positive).
    @return Rotated (x, y) tuple.
    """
    cos_h = math.cos(heading_rad)
    sin_h = math.sin(heading_rad)
    return (cos_h * x - sin_h * y, sin_h * x + cos_h * y)


def _translate(
    x: float,
    y: float,
    origin_x: float,
    origin_y: float,
    heading_rad: float,
) -> Tuple[float, float]:
    """
    @brief Rotate then translate a local point to world frame.
    @param x: Local x coordinate.
    @param y: Local y coordinate.
    @param origin_x: World-frame x of the lot origin.
    @param origin_y: World-frame y of the lot origin.
    @param heading_rad: Lot heading in radians.
    @return World-frame (x, y) tuple.
    """
    rx, ry = _rotate(x, y, heading_rad)
    return (rx + origin_x, ry + origin_y)


def _world_yaw(local_yaw_deg: float, heading_deg: float) -> float:
    """
    @brief Convert a local yaw angle to world-frame yaw.
    @param local_yaw_deg: Yaw in the lot's local frame (degrees).
    @param heading_deg: Lot heading offset (degrees).
    @return World-frame yaw in degrees.
    """
    return (local_yaw_deg + heading_deg) % 360.0


# ---------------------------------------------------------------------------
# Bay layout generators (local frame, origin at lot corner)
# ---------------------------------------------------------------------------


def _bay_corners(
    cx: float,
    cy: float,
    yaw_deg: float,
    width: float,
    depth: float,
) -> List[Tuple[float, float]]:
    """
    @brief Return the four corner points of a bay rectangle in local frame.

    The bay rectangle has depth along the vehicle's heading axis and width
    perpendicular to it. Corners are ordered CCW starting from (-d/2, -w/2).

    @param cx: Bay centre x.
    @param cy: Bay centre y.
    @param yaw_deg: Bay heading in degrees (vehicle nose direction).
    @param width: Bay width (perpendicular to heading), metres.
    @param depth: Bay depth (along heading), metres.
    @return List of four (x, y) corner tuples.
    """
    yaw_rad = math.radians(yaw_deg)
    cos_y = math.cos(yaw_rad)
    sin_y = math.sin(yaw_rad)
    hw = width / 2.0
    hd = depth / 2.0
    local = [(-hd, -hw), (hd, -hw), (hd, hw), (-hd, hw)]
    return [
        (cx + cos_y * lx - sin_y * ly, cy + sin_y * lx + cos_y * ly)
        for lx, ly in local
    ]


def _point_in_polygon(px: float, py: float, polygon: List[Tuple[float, float]]) -> bool:
    """
    @brief Ray-casting point-in-polygon test (includes boundary).
    @param px: Point x.
    @param py: Point y.
    @param polygon: List of (x, y) vertices in order.
    @return True if the point is inside or on the polygon boundary.
    """
    n = len(polygon)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > py) != (yj > py)) and (
            px < (xj - xi) * (py - yi) / (yj - yi + 1e-12) + xi
        ):
            inside = not inside
        j = i
    return inside


def _validate_bays_in_polygon(
    bays: List[Dict[str, Any]],
    corners: List[Dict[str, float]],
    shape_name: str,
    margin: float = 0.05,
) -> None:
    """
    @brief Raise ValueError if any bay corner lies outside the lot polygon.

    Checks all four corners of every bay rectangle. A small margin is used
    to allow bays that are flush against a wall (floating-point tolerance).

    @param bays: List of bay dicts with local_x, local_y, local_yaw_deg, width, depth.
    @param corners: Lot perimeter as list of {x, y} dicts (local frame).
    @param shape_name: Floor plan name for error messages.
    @param margin: Outward expansion of polygon for boundary-touching bays.
    @raises ValueError: If any bay corner is outside the expanded polygon.
    """
    # Build slightly expanded polygon for tolerance
    poly = [(c["x"], c["y"]) for c in corners]
    cx_avg = sum(p[0] for p in poly) / len(poly)
    cy_avg = sum(p[1] for p in poly) / len(poly)
    expanded = [
        (
            cx_avg + (1.0 + margin) * (px - cx_avg),
            cy_avg + (1.0 + margin) * (py - cy_avg),
        )
        for px, py in poly
    ]

    for bay in bays:
        bay_corners_pts = _bay_corners(
            bay["local_x"],
            bay["local_y"],
            bay["local_yaw_deg"],
            bay["width"],
            bay["depth"],
        )
        for corner in bay_corners_pts:
            if not _point_in_polygon(corner[0], corner[1], expanded):
                raise ValueError(
                    f"[{shape_name}] Bay '{bay.get('bay_type','?')}' at "
                    f"({bay['local_x']:.2f}, {bay['local_y']:.2f}) has a corner "
                    f"at ({corner[0]:.2f}, {corner[1]:.2f}) outside the lot boundary."
                )


def _warn_narrow_corridors(
    bays: List[Dict[str, Any]],
    shape_name: str,
    min_width: float = 6.0,
) -> None:
    """
    @brief Warn if the minimum gap between any two facing bay clusters is narrower
           than min_width (EAR 05 minimum corridor width = 6.0 m).

    Computes the nearest-edge gap between every pair of bays. If the gap is less
    than min_width and the bays belong to different clusters (different bay_type
    or opposing yaw), a warning is printed. Generation is not blocked.

    @param bays: All bays in the layout (local frame), each with local_x, local_y,
                 local_yaw_deg, width, depth.
    @param shape_name: Floor plan name for warning messages.
    @param min_width: Minimum acceptable corridor width in metres (default 6.0 m).
    """
    for i in range(len(bays)):
        for j in range(i + 1, len(bays)):
            a = bays[i]
            b = bays[j]
            # Skip pairs of the same type with the same yaw (same cluster)
            same_type = a.get("bay_type") == b.get("bay_type")
            yaw_diff = abs(a["local_yaw_deg"] - b["local_yaw_deg"]) % 360.0
            same_yaw = yaw_diff < 1.0 or abs(yaw_diff - 360.0) < 1.0
            if same_type and same_yaw:
                continue
            # Compute axis-aligned nearest-edge gap (conservative, ignores rotation)
            a_half_x = a["depth"] / 2.0
            a_half_y = a["width"] / 2.0
            b_half_x = b["depth"] / 2.0
            b_half_y = b["width"] / 2.0
            gap_x = abs(a["local_x"] - b["local_x"]) - a_half_x - b_half_x
            gap_y = abs(a["local_y"] - b["local_y"]) - a_half_y - b_half_y
            # Nearest edge gap is the minimum positive axis separation
            gap = max(gap_x, gap_y, 0.0) if gap_x > 0 or gap_y > 0 else 0.0
            if gap < min_width:
                print(
                    f"  WARNING [{shape_name}]: corridor between "
                    f"'{a.get('bay_type','?')}' ({a['local_x']:.1f},{a['local_y']:.1f}) "
                    f"and '{b.get('bay_type','?')}' ({b['local_x']:.1f},{b['local_y']:.1f}) "
                    f"is {gap:.2f} m (min {min_width:.1f} m)."
                )


def _perpendicular_bays(
    n: int,
    row_x: float,
    row_y_start: float,
    facing_yaw_deg: float,
) -> List[Dict[str, Any]]:
    """
    @brief Generate n perpendicular bay centres in a row along the Y axis.

    Bay backs face the wall. row_x is the bay centre X, so the back edge
    sits at row_x + depth/2. facing_yaw_deg=180 means nose toward -X (aisle).

    @param n: Number of bays.
    @param row_x: X position of bay centre row.
    @param row_y_start: Y position of the first bay centre.
    @param facing_yaw_deg: Yaw of a parked vehicle (degrees, local frame).
    @return List of bay dicts.
    """
    dims = BAY_DIMS["perpendicular"]
    bays = []
    for i in range(n):
        bays.append(
            {
                "bay_type": "perpendicular",
                "local_x": row_x,
                "local_y": row_y_start + i * dims["width"],
                "local_yaw_deg": facing_yaw_deg,
                "width": dims["width"],
                "depth": dims["depth"],
            }
        )
    return bays


def _angled_bays_along_wall(
    n: int,
    wall_x0: float,
    wall_y0: float,
    wall_dx: float,
    wall_dy: float,
    wall_len: float,
    offset_from_wall: float,
    start_along_wall: float,
    facing_yaw_deg: float,
) -> List[Dict[str, Any]]:
    """
    @brief Generate n angled (45-deg) bay centres along an arbitrary wall.

    Bay backs touch the wall. Bays are spaced width/sin(45) apart along
    the wall direction vector.

    @param n: Number of bays.
    @param wall_x0: Wall start point x.
    @param wall_y0: Wall start point y.
    @param wall_dx: Wall unit direction x (normalised).
    @param wall_dy: Wall unit direction y (normalised).
    @param wall_len: Total wall length (unused but kept for documentation).
    @param offset_from_wall: Distance from wall to bay centre (inward).
    @param start_along_wall: Distance along wall to the first bay centre.
    @param facing_yaw_deg: Yaw of a parked vehicle (degrees, local frame).
    @return List of bay dicts.
    """
    dims = BAY_DIMS["angled"]
    spacing = dims["width"] / math.sin(math.radians(45.0))
    # Inward normal: rotate wall direction 90 deg CCW
    nx = -wall_dy
    ny = wall_dx
    bays = []
    for i in range(n):
        along = start_along_wall + i * spacing
        bx = wall_x0 + wall_dx * along + nx * offset_from_wall
        by = wall_y0 + wall_dy * along + ny * offset_from_wall
        bays.append(
            {
                "bay_type": "angled",
                "local_x": bx,
                "local_y": by,
                "local_yaw_deg": facing_yaw_deg,
                "width": dims["width"],
                "depth": dims["depth"],
            }
        )
    return bays


def _parallel_bays(
    n: int,
    row_y: float,
    row_x_start: float,
    facing_yaw_deg: float,
) -> List[Dict[str, Any]]:
    """
    @brief Generate n parallel (pull-in) bay centres along the X axis.

    Bays are side-by-side on their short side (width=2.5m spacing).

    @param n: Number of bays.
    @param row_y: Y position of bay centre row.
    @param row_x_start: X position of the first bay centre.
    @param facing_yaw_deg: Yaw of a parked vehicle (degrees, local frame).
    @return List of bay dicts.
    """
    dims = BAY_DIMS["parallel"]
    bays = []
    for i in range(n):
        bays.append(
            {
                "bay_type": "parallel",
                "local_x": row_x_start + i * dims["width"],
                "local_y": row_y,
                "local_yaw_deg": facing_yaw_deg,
                "width": dims["width"],
                "depth": dims["depth"],
            }
        )
    return bays


# ---------------------------------------------------------------------------
# Geometry: bay offset helpers
# ---------------------------------------------------------------------------


def _perp_centre_x_from_wall(wall_x: float, depth: float, inward: bool = True) -> float:
    """
    @brief Compute perpendicular bay centre x so back edge touches a vertical wall.
    @param wall_x: X coordinate of the wall.
    @param depth: Bay depth (metres).
    @param inward: True if bays face away from the wall (nose inward).
    @return Bay centre x.
    """
    if inward:
        # Wall is on the +x side, bays face -x: centre = wall - depth/2
        return wall_x - depth / 2.0
    else:
        # Wall is on the -x side, bays face +x: centre = wall + depth/2
        return wall_x + depth / 2.0


def _perp_centre_y_from_wall(wall_y: float, depth: float, inward: bool = True) -> float:
    """
    @brief Compute perpendicular bay centre y so back edge touches a horizontal wall.
    @param wall_y: Y coordinate of the wall.
    @param depth: Bay depth (metres).
    @param inward: True if nose faces away from wall (-y direction when wall is on +y side).
    @return Bay centre y.
    """
    if inward:
        return wall_y - depth / 2.0
    else:
        return wall_y + depth / 2.0


def _ang_offset_from_wall(bay_depth: float, bay_width: float) -> float:
    """
    @brief Inward offset from a flat wall to angled bay centre (45 deg parking).

    At 45 deg the farthest corner of the rotated bay rectangle is at distance
    (depth/2 + width/2) * sin(45) from the centre perpendicular to the wall.

    @param bay_depth: Bay depth (metres).
    @param bay_width: Bay width (metres).
    @return Perpendicular offset from wall to bay centre.
    """
    return (bay_depth / 2.0 + bay_width / 2.0) * math.sin(math.radians(45.0))


def _ang_x_margin(bay_depth: float, bay_width: float) -> float:
    """
    @brief Minimum along-wall start offset so leftmost bay corner sits at wall edge.
    @param bay_depth: Bay depth (metres).
    @param bay_width: Bay width (metres).
    @return Start offset along the wall direction.
    """
    return (bay_depth / 2.0 + bay_width / 2.0) * math.cos(math.radians(45.0)) + 0.5


def _par_centre_from_wall(wall_coord: float, bay_depth: float, inward: bool = True) -> float:
    """
    @brief Compute parallel bay centre so back edge touches a wall.

    For a parallel bay at yaw=270 (nose toward -Y), depth is along Y.

    @param wall_coord: Wall coordinate (y if top/bottom wall, x if side wall).
    @param bay_depth: Bay depth (metres).
    @param inward: True if nose faces away from the wall.
    @return Bay centre coordinate.
    """
    if inward:
        return wall_coord - bay_depth / 2.0
    else:
        return wall_coord + bay_depth / 2.0


# ---------------------------------------------------------------------------
# Floor plan geometry definitions
# ---------------------------------------------------------------------------


def _rectangle_layout(
    width: float,
    depth: float,
) -> Dict[str, Any]:
    """
    @brief Rectangle (35x35 m). Three distinct bay zones.

    Layout:
      - Perpendicular row along rear wall (x=depth), 5 bays, yaw=180.
        Bays adjacent on the short side (2.5 m): stacked along Y.
      - Angled row along bottom wall (y=0), 5 bays, yaw=135.
      - Parallel bays short-side adjacent (8 m along wall, 2.5 m deep):
          3 bays along top wall (y=width), yaw=0 (nose toward +X)
          2 bays along right wall (x=depth), yaw=90 (nose toward +Y)

    @param width: Lot width (Y axis), metres.
    @param depth: Lot depth (X axis), metres.
    @return Layout dict (local frame).
    """
    corners = [
        {"x": 0.0, "y": 0.0},
        {"x": depth, "y": 0.0},
        {"x": depth, "y": width},
        {"x": 0.0, "y": width},
    ]

    dims_perp = BAY_DIMS["perpendicular"]
    dims_ang = BAY_DIMS["angled"]
    dims_par = BAY_DIMS["parallel"]

    # Perpendicular row: rear wall (x=depth), nose toward -X (yaw=180).
    # Bays stacked along Y, adjacent on their short side (2.5 m spacing).
    perp_cx = _perp_centre_x_from_wall(depth, dims_perp["depth"], inward=True)
    perp_bays = _perpendicular_bays(BAYS_PER_TYPE, perp_cx, 1.0, 180.0)

    # Angled row: bottom wall (y=0), nose toward upper-left (yaw=135).
    ang_w_offset = _ang_offset_from_wall(dims_ang["depth"], dims_ang["width"])
    ang_x_start = _ang_x_margin(dims_ang["depth"], dims_ang["width"])
    ang_bays = _angled_bays_along_wall(
        BAYS_PER_TYPE,
        wall_x0=0.0, wall_y0=0.0,
        wall_dx=1.0, wall_dy=0.0,
        wall_len=depth,
        offset_from_wall=ang_w_offset,
        start_along_wall=ang_x_start,
        facing_yaw_deg=135.0,
    )

    # Parallel bays: short-side adjacent = each bay occupies 8 m along the wall.
    # Group A: 3 bays along top wall (y=width), yaw=0 (nose +X, depth=8 m along X).
    # Bay centre y = width - dims_par["width"]/2 (back edge at y=width).
    par_cy_top = width - dims_par["width"] / 2.0
    par_top_x_start = 2.0
    par_bays_top = []
    for i in range(3):
        par_bays_top.append({
            "bay_type": "parallel",
            "local_x": par_top_x_start + dims_par["depth"] / 2.0 + i * dims_par["depth"],
            "local_y": par_cy_top,
            "local_yaw_deg": 0.0,
            "width": dims_par["width"],
            "depth": dims_par["depth"],
        })

    # Group B: 2 bays along right wall (x=depth), yaw=90 (nose +Y, depth=8 m along Y).
    # Bay centre x = depth - dims_par["width"]/2 (back edge at x=depth).
    par_cx_right = depth - dims_par["width"] / 2.0
    par_right_y_start = 2.0
    par_bays_right = []
    for i in range(2):
        par_bays_right.append({
            "bay_type": "parallel",
            "local_x": par_cx_right,
            "local_y": par_right_y_start + dims_par["depth"] / 2.0 + i * dims_par["depth"],
            "local_yaw_deg": 90.0,
            "width": dims_par["width"],
            "depth": dims_par["depth"],
        })

    par_bays = par_bays_top + par_bays_right

    all_bays = perp_bays + ang_bays + par_bays
    _validate_bays_in_polygon(all_bays, corners, "rectangle")
    _warn_narrow_corridors(all_bays, "rectangle")

    spawn = {"x": 1.5, "y": width / 2.0, "yaw_deg": 0.0}

    # Patrol path threads through the centre of each aisle (3 m into each 6 m corridor).
    # Perp row nose faces -X (yaw=180); aisle in front of nose runs along x < perp_nose_x.
    perp_nose_x = perp_cx - dims_perp["depth"] / 2.0
    perp_aisle_cx = perp_nose_x - dims_perp["aisle"] / 2.0   # centre of 6 m aisle
    # Angled row nose faces upper-left; aisle runs above the bay cluster in Y.
    ang_aisle_cy = ang_w_offset + dims_ang["aisle"] / 2.0    # centre of angled-bay aisle
    y_enter = 1.5
    y_exit = width - 1.5
    patrol = [
        {"x": perp_aisle_cx, "y": y_enter},
        {"x": perp_aisle_cx, "y": y_exit},
        {"x": ang_aisle_cy,  "y": y_exit},
        {"x": ang_aisle_cy,  "y": y_enter},
    ]

    # Pedestrian zones: 2 m strips hugging bay faces.
    PED_STRIP = 2.0
    ped_zones = [
        # Strip against perp row nose face (just in front of perp bays, toward -X)
        {
            "x_min": perp_nose_x - PED_STRIP,
            "x_max": perp_nose_x,
            "y_min": 1.0,
            "y_max": 1.0 + BAYS_PER_TYPE * dims_perp["width"],
        },
        # Strip against angled bay nose face (just above the angled row cluster in Y)
        {
            "x_min": ang_x_start,
            "x_max": ang_x_start + BAYS_PER_TYPE * dims_ang["width"] / math.sin(math.radians(45.0)),
            "y_min": ang_w_offset,
            "y_max": ang_w_offset + PED_STRIP,
        },
    ]

    return {
        "corners": corners,
        "bays": all_bays,
        "spawn": spawn,
        "patrol_waypoints": patrol,
        "pedestrian_zones": ped_zones,
    }


def _trapezoid_layout(
    width_front: float,
    width_rear: float,
    depth: float,
) -> Dict[str, Any]:
    """
    @brief Trapezoid (front=35, rear=25, depth=35). Three distinct bay zones.

    Layout (intentionally different from rectangle):
      - Perpendicular 5 bays in the CENTRE of the lot, rotated 90 deg (yaw=90,
        nose toward +Y). Bays stacked along X, adjacent on short side (2.5 m).
      - Angled row following the diagonal BOTTOM wall, 5 bays at 45 deg to the wall.
      - Parallel bays, short-side adjacent (8 m along wall, 2.5 m deep):
          3 bays along top wall, rotated to follow the sloped wall
          3 bays along right wall (x=depth), yaw=90

    @param width_front: Width at entrance (y extent at x=0).
    @param width_rear: Width at rear (y extent at x=depth).
    @param depth: Lot depth (X axis), metres.
    @return Layout dict (local frame).
    """
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

    # Perpendicular bays: TWO back-to-back rows in the centre of the lot.
    # Row A: yaw=90  (nose toward +Y), backs at y = mid - aisle/2
    # Row B: yaw=270 (nose toward -Y), backs at y = mid + aisle/2
    # Gap between the two back edges = aisle (6 m) so a car can pass between them.
    # Rows extended asymmetrically: 2 extra bays on the left, 1 extra on the right
    # (8 bays per row total).
    #
    # Geometry target: >= 6 m corridor on all four sides of the perp cluster.
    # With width_front=45, depth=50:
    #   Angled bay top face ~ y=9.5; top parallel inner face ~ y=43.75.
    #   Row A nose = 18.5 (gap = 9.0 m above angled bays).
    #   Row B nose = 34.5 (gap = 9.25 m below top parallel bays).
    #   perp_mid_y = (21.0 + 32.0) / 2 = 26.5 centres the cluster vertically.
    PERP_BAYS_PER_ROW = BAYS_PER_TYPE + 3  # 8: base 5 + 2 left + 1 right
    perp_mid_y = 25.5   # chosen to give >= 6 m corridors above angled bays and below top parallel bays
    perp_row_a_cy = perp_mid_y - dims_perp["aisle"] / 2.0 - dims_perp["depth"] / 2.0
    perp_row_b_cy = perp_mid_y + dims_perp["aisle"] / 2.0 + dims_perp["depth"] / 2.0
    # X range: shifted 3 m left of mid-lot, extended 2 bays left and 1 bay right vs original 5
    perp_cx_start = depth / 2.0 - (PERP_BAYS_PER_ROW * dims_perp["width"]) / 2.0 + dims_perp["width"] / 2.0 - 1.5 - 3.63
    perp_bays = []
    for i in range(PERP_BAYS_PER_ROW):
        cx = perp_cx_start + i * dims_perp["width"]
        # Row A: nose toward +Y
        perp_bays.append({
            "bay_type": "perpendicular",
            "local_x": cx,
            "local_y": perp_row_a_cy,
            "local_yaw_deg": 90.0,
            "width": dims_perp["width"],
            "depth": dims_perp["depth"],
        })
        # Row B: nose toward -Y (opposite direction)
        perp_bays.append({
            "bay_type": "perpendicular",
            "local_x": cx,
            "local_y": perp_row_b_cy,
            "local_yaw_deg": 270.0,
            "width": dims_perp["width"],
            "depth": dims_perp["depth"],
        })

    # Angled row: follow the diagonal BOTTOM wall from (0,0) to (depth, y_offset).
    wall_len = math.hypot(depth, y_offset)
    wdx = depth / wall_len
    wdy = y_offset / wall_len
    wall_normal_deg = math.degrees(math.atan2(wdx, -wdy))  # inward normal CCW from wall dir
    ang_yaw = (math.degrees(math.atan2(wdx, -wdy)) + 45.0) % 360.0

    ang_offset = _ang_offset_from_wall(dims_ang["depth"], dims_ang["width"])
    ang_start = _ang_x_margin(dims_ang["depth"], dims_ang["width"]) + 6.0  # shift right up the wall
    ang_bays = _angled_bays_along_wall(
        7,  # 7 bays along the diagonal bottom wall
        wall_x0=0.0, wall_y0=0.0,
        wall_dx=wdx, wall_dy=wdy,
        wall_len=wall_len,
        offset_from_wall=ang_offset,
        start_along_wall=ang_start,
        facing_yaw_deg=ang_yaw,
    )

    # Parallel bays: short-side adjacent (8 m along wall, 2.5 m deep from wall).
    # Group A: 3 bays along the top wall, following its slope.
    # Top wall: from (0, width_front) to (depth, width_front - y_offset).
    # Wall direction unit vector: (depth, -y_offset) / wall_len_top.
    # Inward normal (CW 90 from wall dir, pointing into lot = downward): (wdy, -wdx).
    # Bay yaw = atan2(wall_dy, wall_dx) so depth axis lies along the wall.
    top_wall_len = math.hypot(depth, y_offset)
    top_wdx = depth / top_wall_len
    top_wdy = -y_offset / top_wall_len
    # Inward normal (CW 90): (top_wdy, -top_wdx) -- points down-left, into lot ✓
    top_nx = top_wdy
    top_ny = -top_wdx
    top_par_yaw = math.degrees(math.atan2(top_wdy, top_wdx))  # depth axis along wall
    par_bays_top = []
    par_top_spacing = dims_par["depth"]          # 8 m between bay centres along wall
    par_top_along_start = 9.0 + dims_par["depth"] / 2.0  # start along-wall distance (clear corner)
    par_top_normal_offset = dims_par["width"] / 2.0       # back edge flush with wall
    for i in range(3):
        along = par_top_along_start + i * par_top_spacing
        # Wall point at this distance from corner (0, width_front)
        wx = 0.0 + top_wdx * along
        wy = width_front + top_wdy * along
        # Bay centre offset inward from wall
        cx = wx + top_nx * par_top_normal_offset
        cy = wy + top_ny * par_top_normal_offset
        par_bays_top.append({
            "bay_type": "parallel",
            "local_x": cx,
            "local_y": cy,
            "local_yaw_deg": top_par_yaw,
            "width": dims_par["width"],
            "depth": dims_par["depth"],
        })

    # Group B: 3 bays on right wall (x=depth), yaw=90 (nose +Y, depth=8 m along Y).
    # Right wall runs from y=y_offset to y=width_front-y_offset (height=width_rear).
    # 3 bays need 3*8 + 2 gaps = 26 m; width_rear=30 m so this fits comfortably.
    par_cx_right = depth - dims_par["width"] / 2.0
    par_right_y_start = y_offset + 3.0  # clear the tapered bottom-right corner
    par_bays_right = []
    for i in range(3):
        par_bays_right.append({
            "bay_type": "parallel",
            "local_x": par_cx_right,
            "local_y": par_right_y_start + dims_par["depth"] / 2.0 + i * dims_par["depth"],
            "local_yaw_deg": 90.0,
            "width": dims_par["width"],
            "depth": dims_par["depth"],
        })

    par_bays = par_bays_top + par_bays_right

    all_bays = perp_bays + ang_bays + par_bays
    _validate_bays_in_polygon(all_bays, corners, "trapezoid")
    _warn_narrow_corridors(all_bays, "trapezoid")

    # Spawn 1: gap in the left wall (x=0), car faces into the lot (yaw=0).
    spawn = {"x": 0.0, "y": width_front / 2.0, "yaw_deg": 0.0}
    # Spawn 2: gap in the bottom diagonal wall at x=38.
    # Car faces perpendicular to that wall, pointing inward (CCW 90 from wall dir).
    _bw_len = math.hypot(depth, y_offset)
    _bwdx = depth / _bw_len
    _bwdy = y_offset / _bw_len
    # Inward normal = CCW 90 from wall dir: (-wdy, wdx). yaw = atan2(wdx, -wdy).
    _spawn2_yaw = math.degrees(math.atan2(_bwdx, -_bwdy))
    spawn2 = {
        "x": 38.0,
        "y": y_offset * (38.0 / depth),
        "yaw_deg": round(_spawn2_yaw, 1),
    }
    # Patrol path threads through the centre of the three horizontal aisles.
    # Row A nose faces +Y (yaw=90); aisle below nose = [row_a_nose_y - aisle, row_a_nose_y].
    # Row B nose faces -Y (yaw=270); aisle above nose = [row_b_nose_y, row_b_nose_y + aisle].
    # Back-to-back gap between row A back and row B back = [row_a_back_y, row_b_back_y].
    aisle1_cy = perp_row_a_cy - dims_perp["depth"] / 2.0 - dims_perp["aisle"] / 2.0
    aisle2_cy = perp_mid_y  # midpoint of the 6 m back-to-back gap
    aisle3_cy = perp_row_b_cy + dims_perp["depth"] / 2.0 + dims_perp["aisle"] / 2.0
    PED_STRIP = 3.0  # width of pedestrian zones in metres (must be < aisle width of 6.0 m)

    # Patrol path: rectangular loop equidistant between the perp cluster and its neighbours.
    # Left leg:   midpoint between ped zone right edge and left edge of perp cluster.
    # Right leg:  midpoint between right edge of perp cluster and inner face of right-wall par bays.
    # Bottom leg: aisle1_cy already = midpoint of corridor between Row A nose and angled bay top.
    # Top leg:    aisle3_cy already = midpoint of corridor between Row B nose and top parallel face.
    perp_cluster_x_min = perp_cx_start - dims_perp["width"] / 2.0
    perp_cluster_x_max = perp_cx_start + (PERP_BAYS_PER_ROW - 0.5) * dims_perp["width"]
    par_right_inner_x = par_cx_right - dims_par["width"] / 2.0  # inner face of right-wall par bays
    # Left leg: midpoint between ped zone right edge (PED_STRIP) and cluster left edge.
    # This ensures the patrol has a clear lane between the wall-hugging ped zones and the bays.
    x_enter = (PED_STRIP + perp_cluster_x_min) / 2.0
    x_exit = (perp_cluster_x_max + par_right_inner_x) / 2.0             # mid: cluster right <-> par right
    patrol = [
        {"x": x_enter, "y": aisle1_cy - 4},   # bottom-left
        {"x": x_exit,  "y": aisle1_cy},   # bottom-right
        {"x": x_exit,  "y": aisle3_cy},   # top-right
        {"x": x_enter, "y": aisle3_cy + 4},   # top-left
    ]

    # Pedestrian zones: 2 m strips hugging bay faces.
    # x extents match exactly the bay cluster left/right edges.
    perp_cluster_x_min = perp_cx_start - dims_perp["width"] / 2.0
    perp_cluster_x_max = perp_cx_start + (PERP_BAYS_PER_ROW - 0.5) * dims_perp["width"]
    par_bay_x_min = par_cx_right - dims_par["width"] / 2.0
    ped_zones = [
        # Aisle 1: 2 m strip against row A nose face (just below row A in Y)
        {
            "x_min": perp_cluster_x_min,
            "x_max": perp_cluster_x_max,
            "y_min": perp_row_a_cy - dims_perp["depth"] / 2.0 - PED_STRIP,
            "y_max": perp_row_a_cy - dims_perp["depth"] / 2.0,
        },
        # Aisle 2: 2 m strip against row A back face (inside the 6 m back-to-back gap)
        {
            "x_min": perp_cluster_x_min,
            "x_max": perp_cluster_x_max,
            "y_min": perp_row_a_cy + dims_perp["depth"] / 2.0,
            "y_max": perp_row_a_cy + dims_perp["depth"] / 2.0 + PED_STRIP,
        },
        # Aisle 3: 2 m strip against row B nose face (just above row B in Y)
        {
            "x_min": perp_cluster_x_min,
            "x_max": perp_cluster_x_max,
            "y_min": perp_row_b_cy + dims_perp["depth"] / 2.0,
            "y_max": perp_row_b_cy + dims_perp["depth"] / 2.0 + PED_STRIP,
        },
        # Aisle 4: 2 m vertical strip against left face of right-wall parallel bays
        {
            "x_min": par_bay_x_min - PED_STRIP,
            "x_max": par_bay_x_min,
            "y_min": par_right_y_start,
            "y_max": par_right_y_start + 3 * dims_par["depth"],
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


def _irregular_a_layout(
    width: float,
    depth: float,
) -> Dict[str, Any]:
    """
    @brief Five-sided irregular polygon (OOD, held out from training). Distinct bay layout.

    Rear-right corner cut diagonally (8x8 m triangle removed).

    Layout (intentionally different from both rectangle and trapezoid):
      - Perpendicular row along LEFT wall (x=0), bays face +X (yaw=0)
      - Angled row along the DIAGONAL CUT wall, bays face into lot
      - Parallel row along BOTTOM wall (y=0), bays face +Y (yaw=90)

    @param width: Nominal lot width (Y axis), metres.
    @param depth: Nominal lot depth (X axis), metres.
    @return Layout dict (local frame).
    """
    cut = 8.0
    # Cut is at rear-top-right: from (depth, width-cut) to (depth-cut, width)
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

    # Perpendicular row: LEFT wall (x=0), nose toward +X (yaw=0).
    # Bay centre x = depth/2 from left wall.
    # Bays run along Y, start near mid-lot to be clearly distinct from rectangle.
    perp_cx = _perp_centre_x_from_wall(0.0, dims_perp["depth"], inward=False)
    perp_y_start = width / 2.0 - (BAYS_PER_TYPE * dims_perp["width"]) / 2.0
    perp_bays = _perpendicular_bays(BAYS_PER_TYPE, perp_cx, perp_y_start, 0.0)

    # Angled bays: split across two walls for variety.
    #
    # Group A (3 bays): DIAGONAL CUT wall from (depth, width-cut) to (depth-cut, width).
    # The cut wall is 8*sqrt(2) ~ 11.3 m, fitting 3 bays comfortably.
    cut_dx = (depth - cut) - depth   # = -cut
    cut_dy = width - (width - cut)   # = cut
    cut_len = math.hypot(cut_dx, cut_dy)
    wdx_cut = cut_dx / cut_len  # = -1/sqrt(2)
    wdy_cut = cut_dy / cut_len  # = +1/sqrt(2)
    # Inward normal (90 deg CCW from wall direction):
    nx_cut = -wdy_cut   # points toward lower-left (into lot)
    ny_cut = wdx_cut
    wall_normal_deg_cut = math.degrees(math.atan2(ny_cut, nx_cut))
    ang_yaw_cut = (wall_normal_deg_cut + 45.0) % 360.0

    ang_offset = _ang_offset_from_wall(dims_ang["depth"], dims_ang["width"])
    ang_start = _ang_x_margin(dims_ang["depth"], dims_ang["width"])
    ang_bays_cut = _angled_bays_along_wall(
        3,
        wall_x0=depth, wall_y0=width - cut,
        wall_dx=wdx_cut, wall_dy=wdy_cut,
        wall_len=cut_len,
        offset_from_wall=ang_offset,
        start_along_wall=ang_start,
        facing_yaw_deg=ang_yaw_cut,
    )

    # Group B (2 bays): REAR wall (x=depth), lower section, facing left (yaw=180).
    # These bays are in the y=0 .. width-cut region, distinct from the perp row on x=0.
    ang_bays_rear = _angled_bays_along_wall(
        2,
        wall_x0=depth, wall_y0=1.0,
        wall_dx=0.0, wall_dy=1.0,
        wall_len=width - cut - 2.0,
        offset_from_wall=ang_offset,
        start_along_wall=ang_start,
        facing_yaw_deg=180.0 + 45.0,  # 225 deg: upper-left into lot
    )

    ang_bays = ang_bays_cut + ang_bays_rear

    # Parallel row: BOTTOM wall (y=0), nose toward +Y (yaw=90).
    # Bay centre y = depth_par/2 from bottom wall.
    # Bays run along X; at yaw=90 the bay extends +-width/2 in x.
    par_cy = _par_centre_from_wall(0.0, dims_par["depth"], inward=False)
    par_x_start = dims_par["width"] / 2.0 + 0.5
    par_bays = _parallel_bays(
        BAYS_PER_TYPE, par_cy, par_x_start, 90.0
    )

    all_bays = perp_bays + ang_bays + par_bays
    _validate_bays_in_polygon(all_bays, corners, "irregular_a")
    _warn_narrow_corridors(all_bays, "irregular_a")

    spawn = {"x": 1.5, "y": width / 2.0, "yaw_deg": 0.0}

    # Patrol path threads through the centre of the aisle in front of the perp row.
    # Perp row nose faces +X (yaw=0); aisle is to the right of the nose face.
    perp_nose_x_irr = perp_cx + dims_perp["depth"] / 2.0
    perp_aisle_cx_irr = perp_nose_x_irr + dims_perp["aisle"] / 2.0
    y_bot = par_cy + dims_par["depth"] / 2.0 + 1.0  # clear the parallel bay noses
    y_top = width - cut - 2.0
    patrol = [
        {"x": perp_aisle_cx_irr, "y": y_bot},
        {"x": perp_aisle_cx_irr, "y": y_top},
        {"x": depth - cut - 2.0, "y": y_top},
        {"x": depth - 2.0,       "y": y_bot},
    ]

    # Pedestrian zones: 2 m strips hugging bay faces.
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


# ---------------------------------------------------------------------------
# World-frame transformation
# ---------------------------------------------------------------------------


def _to_world_frame(
    local_layout: Dict[str, Any],
    origin_x: float,
    origin_y: float,
    origin_z: float,
    heading_deg: float,
) -> Dict[str, Any]:
    """
    @brief Transform a local-frame layout to CARLA world frame.

    All (x, y) coordinates are rotated by heading_deg and translated by
    (origin_x, origin_y). Yaw angles are offset by heading_deg.

    @param local_layout: Layout dict from one of the _*_layout() functions.
    @param origin_x: World-frame x of lot origin.
    @param origin_y: World-frame y of lot origin.
    @param origin_z: World-frame z of lot ground (CARLA z, typically 0.3).
    @param heading_deg: Heading of the lot in world frame (degrees, CCW positive).
    @return World-frame layout dict ready for YAML serialisation.
    """
    h_rad = math.radians(heading_deg)

    # Transform corners
    world_corners = []
    for c in local_layout["corners"]:
        wx, wy = _translate(c["x"], c["y"], origin_x, origin_y, h_rad)
        world_corners.append({"x": round(wx, 3), "y": round(wy, 3)})

    # Transform bays
    world_bays = []
    for i, b in enumerate(local_layout["bays"]):
        wx, wy = _translate(b["local_x"], b["local_y"], origin_x, origin_y, h_rad)
        world_yaw = _world_yaw(b["local_yaw_deg"], heading_deg)
        world_bays.append(
            {
                "id": f"{b['bay_type']}_{i}",
                "bay_type": b["bay_type"],
                "x": round(wx, 3),
                "y": round(wy, 3),
                "z": origin_z,
                "yaw_deg": round(world_yaw, 2),
                "width": b["width"],
                "depth": b["depth"],
            }
        )

    # Transform spawn
    sp = local_layout["spawn"]
    sx, sy = _translate(sp["x"], sp["y"], origin_x, origin_y, h_rad)
    world_spawn = {
        "x": round(sx, 3),
        "y": round(sy, 3),
        "z": origin_z,
        "yaw_deg": round(_world_yaw(sp["yaw_deg"], heading_deg), 2),
    }

    # Transform extra spawn points (optional)
    world_extra_spawns = []
    for esp in local_layout.get("extra_spawns", []):
        esx, esy = _translate(esp["x"], esp["y"], origin_x, origin_y, h_rad)
        world_extra_spawns.append({
            "x": round(esx, 3),
            "y": round(esy, 3),
            "z": origin_z,
            "yaw_deg": round(_world_yaw(esp["yaw_deg"], heading_deg), 2),
        })

    # Transform patrol waypoints
    world_patrol = []
    for wp in local_layout["patrol_waypoints"]:
        wx, wy = _translate(wp["x"], wp["y"], origin_x, origin_y, h_rad)
        world_patrol.append({"x": round(wx, 3), "y": round(wy, 3)})

    # Transform pedestrian zones (axis-aligned bounding boxes -> transform centre)
    world_ped_zones = []
    for zone in local_layout["pedestrian_zones"]:
        cx = (zone["x_min"] + zone["x_max"]) / 2.0
        cy = (zone["y_min"] + zone["y_max"]) / 2.0
        half_w = (zone["x_max"] - zone["x_min"]) / 2.0
        half_h = (zone["y_max"] - zone["y_min"]) / 2.0
        wcx, wcy = _translate(cx, cy, origin_x, origin_y, h_rad)
        world_ped_zones.append(
            {
                "centre_x": round(wcx, 3),
                "centre_y": round(wcy, 3),
                "half_width": round(half_w, 3),
                "half_height": round(half_h, 3),
            }
        )

    return {
        "corners": world_corners,
        "bays": world_bays,
        "spawn_transform": world_spawn,
        "extra_spawn_transforms": world_extra_spawns,
        "patrol_waypoints": world_patrol,
        "pedestrian_zones": world_ped_zones,
    }


# ---------------------------------------------------------------------------
# YAML output
# ---------------------------------------------------------------------------


def _write_layout_yaml(
    shape: str,
    origin_x: float,
    origin_y: float,
    origin_z: float,
    heading_deg: float,
    world_layout: Dict[str, Any],
    output_path: Path,
    ood: bool,
) -> None:
    """
    @brief Serialise a world-frame layout to YAML.

    @param shape: Floor plan shape name (rectangle, trapezoid, irregular_a).
    @param origin_x: World-frame x of lot origin.
    @param origin_y: World-frame y of lot origin.
    @param origin_z: World-frame z of lot origin.
    @param heading_deg: Lot heading in world frame (degrees).
    @param world_layout: World-frame layout dict.
    @param output_path: Path to write the YAML file.
    @param ood: Whether this floor plan is held out for OOD evaluation.
    """
    doc = {
        "# Parking lot layout - generated by scripts/generate_lot_layout.py": None,
        "# Edit origin and re-run make generate-layouts after measuring CARLA coordinates.": None,
        "floor_plan": shape,
        "ood": ood,
        "origin": {
            "x": origin_x,
            "y": origin_y,
            "z": origin_z,
            "heading_deg": heading_deg,
        },
        "spawn_transform": world_layout["spawn_transform"],
        "extra_spawn_transforms": world_layout["extra_spawn_transforms"],
        "corners": world_layout["corners"],
        "bays": world_layout["bays"],
        "patrol_waypoints": world_layout["patrol_waypoints"],
        "pedestrian_zones": world_layout["pedestrian_zones"],
    }

    # Remove sentinel comment keys (just used for readability above)
    doc_clean = {k: v for k, v in doc.items() if not k.startswith("#")}

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        # Write header comment manually since PyYAML strips comments
        f.write("# Parking lot layout - generated by scripts/generate_lot_layout.py\n")
        f.write("# Edit origin.x/y and re-run 'make generate-layouts' after\n")
        f.write("# measuring CARLA world-frame coordinates via:\n")
        f.write("#   make docker-inspect LAYOUT=<floor_plan>\n")
        f.write("#\n")
        yaml.dump(
            doc_clean, f, default_flow_style=False, sort_keys=False, allow_unicode=True
        )

    print(f"  Written: {output_path}")


# ---------------------------------------------------------------------------
# Bird's-eye PNG plot
# ---------------------------------------------------------------------------


def _plot_layout(
    shape: str,
    world_layout: Dict[str, Any],
    plot_path: Path,
) -> None:
    """
    @brief Render a bird's-eye PNG of the lot layout.

    Layers (back to front):
      - Grey lot polygon
      - Bay rectangles (colour-coded by type)
      - Yaw arrows on each bay
      - Spawn triangle
      - Patrol path

    @param shape: Floor plan shape name for title.
    @param world_layout: World-frame layout dict.
    @param plot_path: Path to save the PNG.
    """
    try:
        import matplotlib.patches as mpatches
        import matplotlib.pyplot as plt
        from matplotlib.patches import FancyArrowPatch, Polygon
    except ImportError:
        print("  WARNING: matplotlib not available, skipping plot.")
        return

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_aspect("equal")
    ax.set_title(f"Floor plan: {shape}", fontsize=14)
    ax.set_xlabel("x (m)", fontsize=12)
    ax.set_ylabel("y (m)", fontsize=12)

    # Lot boundary polygon (filled, no edge — edges drawn per-segment with gaps for spawns)
    corner_pts = [(c["x"], c["y"]) for c in world_layout["corners"]]
    lot_patch = Polygon(
        corner_pts, closed=True, facecolor="#DDDDDD", edgecolor="none", linewidth=0
    )
    ax.add_patch(lot_patch)

    # Draw perimeter with gaps at each spawn point (gap_half = 2.5 m each side -> 5 m gap)
    all_spawns = [world_layout["spawn_transform"]] + world_layout.get("extra_spawn_transforms", [])
    gap_half = 2.5  # metres each side of spawn centre

    def _draw_perimeter_with_gaps(
        ax: "plt.Axes",
        corners: List[Tuple[float, float]],
        spawns: List[Dict[str, Any]],
        g: float,
    ) -> None:
        """Draw lot perimeter as thick black segments, leaving a gap around each spawn."""
        n = len(corners)
        for i in range(n):
            p0 = corners[i]
            p1 = corners[(i + 1) % n]
            seg_dx = p1[0] - p0[0]
            seg_dy = p1[1] - p0[1]
            seg_len = math.hypot(seg_dx, seg_dy)
            if seg_len < 1e-6:
                continue
            udx = seg_dx / seg_len
            udy = seg_dy / seg_len

            # Collect gap intervals (t in [0, seg_len]) for spawns near this segment
            gaps: List[Tuple[float, float]] = []
            for sp in spawns:
                # Project spawn onto segment line
                t = (sp["x"] - p0[0]) * udx + (sp["y"] - p0[1]) * udy
                # Perpendicular distance from spawn to segment
                perp = abs((sp["x"] - p0[0]) * udy - (sp["y"] - p0[1]) * udx)
                if 0.0 <= t <= seg_len and perp < g + 0.5:
                    gaps.append((t - g, t + g))

            # Merge overlapping gaps
            gaps.sort()
            merged: List[Tuple[float, float]] = []
            for lo, hi in gaps:
                if merged and lo <= merged[-1][1]:
                    merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
                else:
                    merged.append([lo, hi])

            # Draw segments between gaps
            draw_intervals = []
            prev = 0.0
            for lo, hi in merged:
                if prev < lo:
                    draw_intervals.append((max(prev, 0.0), min(lo, seg_len)))
                prev = hi
            if prev < seg_len:
                draw_intervals.append((max(prev, 0.0), seg_len))

            for t0, t1 in draw_intervals:
                if t1 - t0 < 1e-3:
                    continue
                ax.plot(
                    [p0[0] + udx * t0, p0[0] + udx * t1],
                    [p0[1] + udy * t0, p0[1] + udy * t1],
                    color="black", linewidth=2, solid_capstyle="butt",
                )

    _draw_perimeter_with_gaps(ax, corner_pts, all_spawns, gap_half)

    # Bay rectangles
    bay_colours = {
        "perpendicular": "steelblue",
        "angled": "darkorange",
        "parallel": "forestgreen",
    }
    for bay in world_layout["bays"]:
        colour = bay_colours.get(bay["bay_type"], "grey")
        alpha = 0.7
        lw = 2
        bx, by = bay["x"], bay["y"]
        yaw_rad = math.radians(bay["yaw_deg"])
        w, d = bay["width"], bay["depth"]

        # Rectangle corners (local: -d/2..d/2 in X, -w/2..w/2 in Y)
        local_corners = [
            (-d / 2.0, -w / 2.0),
            (d / 2.0, -w / 2.0),
            (d / 2.0, w / 2.0),
            (-d / 2.0, w / 2.0),
        ]
        world_rect = []
        for lx, ly in local_corners:
            wx = bx + math.cos(yaw_rad) * lx - math.sin(yaw_rad) * ly
            wy = by + math.sin(yaw_rad) * lx + math.cos(yaw_rad) * ly
            world_rect.append((wx, wy))

        rect_patch = Polygon(
            world_rect,
            closed=True,
            facecolor=colour,
            edgecolor=colour,
            linewidth=lw,
            alpha=alpha,
        )
        ax.add_patch(rect_patch)

    # Pedestrian zone clouds: rounded semi-transparent yellow patches
    # World-frame zones use centre_x/y + half_width/height format.
    for zone in world_layout.get("pedestrian_zones", []):
        zw = zone["half_width"] * 2.0
        zh = zone["half_height"] * 2.0
        cloud = mpatches.FancyBboxPatch(
            (zone["centre_x"] - zone["half_width"], zone["centre_y"] - zone["half_height"]),
            zw,
            zh,
            boxstyle="round,pad=0.4",
            facecolor="yellow",
            edgecolor="goldenrod",
            alpha=0.40,
            linewidth=1.5,
            linestyle="--",
            zorder=4,
        )
        ax.add_patch(cloud)

    # Spawn triangles + detached direction arrows
    for idx, sp in enumerate([world_layout["spawn_transform"]] + world_layout.get("extra_spawn_transforms", [])):
        yaw_rad = math.radians(sp["yaw_deg"])
        cos_y, sin_y = math.cos(yaw_rad), math.sin(yaw_rad)
        # Triangle at spawn point, rotated to face yaw direction
        tri_size = 1.2
        tri_local = [(tri_size, 0.0), (-tri_size * 0.6, tri_size * 0.6), (-tri_size * 0.6, -tri_size * 0.6)]
        tri_world = [
            (sp["x"] + cos_y * lx - sin_y * ly, sp["y"] + sin_y * lx + cos_y * ly)
            for lx, ly in tri_local
        ]
        tri_patch = Polygon(tri_world, closed=True, facecolor="cyan", edgecolor="white", linewidth=1, zorder=6)
        ax.add_patch(tri_patch)
        label = f"SPAWN {idx + 1}"
        # Per-spawn label offsets: perpendicular offset and vertical nudge
        label_offset_perp = 1.5 if idx > 0 else 1.9
        label_y_nudge = 2.0 if idx > 0 else -0.5

        ax.text(
            sp["x"] + cos_y * 0.2 - sin_y * label_offset_perp,
            sp["y"] + sin_y * 0.2 + cos_y * label_offset_perp + label_y_nudge,
            label, fontsize=10, color="cyan", fontweight="bold", zorder=7,
            path_effects=[
                __import__("matplotlib.patheffects", fromlist=["withStroke"]).withStroke(
                    linewidth=2, foreground="black"
                )
            ],
        )

    # Patrol path
    patrol = world_layout["patrol_waypoints"]
    if patrol:
        px = [wp["x"] for wp in patrol] + [patrol[0]["x"]]
        py = [wp["y"] for wp in patrol] + [patrol[0]["y"]]
        ax.plot(px, py, "m--", linewidth=1.5, alpha=0.7, label="Patrol path")

    # Legend
    from matplotlib.lines import Line2D
    handles = [
        mpatches.Patch(color="steelblue", label="Perpendicular bays"),
        mpatches.Patch(color="darkorange", label="Angled (45 deg) bays"),
        mpatches.Patch(color="forestgreen", label="Parallel bays"),
        mpatches.Patch(color="#DDDDDD", edgecolor="black", label="Lot boundary"),
        Line2D([0], [0], color="magenta", linestyle="--", linewidth=1.5, label="Patrol path"),
        mpatches.FancyBboxPatch(
            (0, 0), 1, 1,
            boxstyle="round,pad=0.2",
            facecolor="yellow", edgecolor="goldenrod",
            alpha=0.5, linestyle="--",
            label="Pedestrian zones",
        ),
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)

    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot:    {plot_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    """
    @brief Parse command-line arguments.
    @return Parsed namespace.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Generate parking lot layout YAML + bird's-eye PNG.\n"
            "Run without --shape to generate all three layouts at once."
        )
    )
    parser.add_argument(
        "--shape",
        choices=["rectangle", "trapezoid", "irregular_a"],
        default=None,
        help="Floor plan shape. Omit to generate all three.",
    )
    parser.add_argument(
        "--origin",
        nargs=3,
        type=float,
        metavar=("X", "Y", "Z"),
        default=[0.0, 0.0, 0.3],
        help="World-frame origin of the lot (x y z). Default: 0 0 0.3.",
    )
    parser.add_argument(
        "--heading",
        type=float,
        default=0.0,
        help="Lot heading in world frame (degrees). Default: 0.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output YAML path (single shape mode only).",
    )
    parser.add_argument(
        "--plot",
        type=str,
        default=None,
        help="Output PNG path (single shape mode only).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="configs/layouts",
        help="Output directory for YAML files (multi-shape mode). Default: configs/layouts.",
    )
    parser.add_argument(
        "--plot-dir",
        type=str,
        default="outputs/layouts",
        help="Output directory for PNG files (multi-shape mode). Default: outputs/layouts.",
    )
    return parser.parse_args()


def _generate_one(
    shape: str,
    origin_x: float,
    origin_y: float,
    origin_z: float,
    heading_deg: float,
    output_path: Path,
    plot_path: Optional[Path],
) -> None:
    """
    @brief Generate one floor plan layout and write to disk.

    @param shape: One of 'rectangle', 'trapezoid', 'irregular_a'.
    @param origin_x: World x of lot origin.
    @param origin_y: World y of lot origin.
    @param origin_z: World z of lot origin.
    @param heading_deg: Lot heading in world frame (degrees).
    @param output_path: YAML output path.
    @param plot_path: PNG output path, or None to skip.
    """
    ood = shape == "irregular_a"

    if shape == "rectangle":
        local_layout = _rectangle_layout(width=35.0, depth=35.0)
    elif shape == "trapezoid":
        local_layout = _trapezoid_layout(width_front=48.0, width_rear=30.0, depth=44.0)
    elif shape == "irregular_a":
        local_layout = _irregular_a_layout(width=35.0, depth=35.0)
    else:
        raise ValueError(f"Unknown shape: {shape}")

    world_layout = _to_world_frame(
        local_layout, origin_x, origin_y, origin_z, heading_deg
    )

    _write_layout_yaml(
        shape, origin_x, origin_y, origin_z, heading_deg, world_layout, output_path, ood
    )

    if plot_path is not None:
        _plot_layout(shape, world_layout, plot_path)


def main() -> None:
    """
    @brief Entry point: generate one or all floor plan layouts.
    """
    args = _parse_args()
    ox, oy, oz = args.origin

    if args.shape is not None:
        # Single shape mode
        out = (
            Path(args.output)
            if args.output
            else Path(args.output_dir) / f"{args.shape}.yaml"
        )
        plot = (
            Path(args.plot) if args.plot else Path(args.plot_dir) / f"{args.shape}.png"
        )
        print(
            f"Generating {args.shape} layout (origin={ox},{oy},{oz}, heading={args.heading} deg)"
        )
        _generate_one(args.shape, ox, oy, oz, args.heading, out, plot)
    else:
        # Multi-shape mode: generate all three with well-separated origins on Town05_Opt
        # Placeholder origins: use actual CARLA coordinates from --mark mode
        # These are approximate positions in flat areas of Town05_Opt (z=0.30)
        configs_map = {
            "rectangle": {"ox": -200.0, "oy": 0.0, "oz": 0.3, "hdg": 0.0, "ood": False},
            "trapezoid": {"ox": 0.0, "oy": -90.0, "oz": 0.3, "hdg": 0.0, "ood": False},
            "irregular_a": {"ox": 100.0, "oy": 0.0, "oz": 0.3, "hdg": 0.0, "ood": True},
        }

        out_dir = Path(args.output_dir)
        plot_dir = Path(args.plot_dir)
        print("Generating all floor plan layouts...")
        print(f"  YAMLs -> {out_dir}/")
        print(f"  PNGs  -> {plot_dir}/")
        print()
        print("NOTE: Origins are approximate placeholders.")
        print("      Run 'make docker-inspect LAYOUT=<floor_plan>'")
        print(
            "      to verify coordinates visually, then re-run this script."
        )
        print()

        for shape, cfg in configs_map.items():
            _generate_one(
                shape=shape,
                origin_x=cfg["ox"],
                origin_y=cfg["oy"],
                origin_z=cfg["oz"],
                heading_deg=cfg["hdg"],
                output_path=out_dir / f"{shape}.yaml",
                plot_path=plot_dir / f"{shape}.png",
            )

    print("\nDone.")


if __name__ == "__main__":
    main()
