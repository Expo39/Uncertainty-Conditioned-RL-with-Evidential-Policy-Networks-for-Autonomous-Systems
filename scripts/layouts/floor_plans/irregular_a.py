"""
@file irregular_a.py
@brief Irregular nine-sided parking lot floor plan (OOD, ~80x50 m).
"""

import math
from typing import Any, Dict

from scripts.layouts.builder import (
    LotBuilder,
    PatrolPath,
    PedestrianZone,
    angled_corner_clearance,
)

ORIGIN_X = -3.0
ORIGIN_Y = 25.0
ORIGIN_Z = 0.3
HEADING_DEG = 0.0
OOD = True

# Perimeter corners (CCW polygon order).
P0 = (0.0, 0.0)    # bottom-left
P1 = (53.0, 0.0)   # notch bottom-left
P2 = (53.0, 8.0)   # notch top-left
P3 = (65.0, 8.0)   # notch top-right
P4 = (65.0, 0.0)   # notch bottom-right
P5 = (80.0, 0.0)   # bottom-right
P6 = (80.0, 37.0)  # diagonal start (top-right)
P7 = (20.0, 50.0)  # diagonal/flat junction
P8 = (0.0, 50.0)   # top-left

# Wall indices in CCW order.
WALL_BOTTOM_LEFT = 0    # P0 -> P1 (left of notch)
WALL_NOTCH_LEFT = 1     # P1 -> P2
WALL_NOTCH_TOP = 2      # P2 -> P3
WALL_NOTCH_RIGHT = 3    # P3 -> P4
WALL_BOTTOM_RIGHT = 4   # P4 -> P5
WALL_RIGHT = 5          # P5 -> P6
WALL_TOP_DIAGONAL = 6   # P6 -> P7
WALL_TOP_FLAT = 7       # P7 -> P8
WALL_LEFT = 8           # P8 -> P0

# Central obstacle (rectangular cone wall in lot interior).
OBSTACLE = (23.25, 39.25, 17.0, 21.0)   # (x_min, x_max, y_min, y_max)
TOP_FLAT_AISLE = 6.0     # Aisle between top-flat back-to-back perp rows.
LEFT_ANG_BOTTOM_Y = 3.0  # Local y of the bottom-most left-wall angled bay.


def generate() -> Dict[str, Any]:
    """
    @brief Build the irregular_a layout using the LotBuilder DSL.
    """
    lot = LotBuilder(
        name="irregular_a",
        corners=[P0, P1, P2, P3, P4, P5, P6, P7, P8],
    )
    dims_ang = lot.dims["angled"]

    # ---------- Bays around central obstacle ---------------------------
    obs_south = lot.row_along_obstacle_face("perpendicular", n=6, obstacle=OBSTACLE, face="south")
    obs_north = lot.row_along_obstacle_face("perpendicular", n=6, obstacle=OBSTACLE, face="north")
    obs_west = lot.row_along_obstacle_face("perpendicular", n=2, obstacle=OBSTACLE, face="west")
    obs_east = lot.row_along_obstacle_face("perpendicular", n=2, obstacle=OBSTACLE, face="east")

    # ---------- Left wall angled bays ----------------------------------
    # The bottom-most bay should land at y=LEFT_ANG_BOTTOM_Y. Walking the
    # left wall from P8 down to P0 (CCW order), placement starts at the
    # natural corner clearance from P8. We pack from the END (closer to P0)
    # so the first bay placed is the bottom one at LEFT_ANG_BOTTOM_Y.
    lot.row_along_perimeter(
        bay_type="angled", n=4, wall=WALL_LEFT, bay_angle_deg=45.0,
        start_along=LEFT_ANG_BOTTOM_Y - lot.wall_y(WALL_BOTTOM_LEFT)
        + angled_corner_clearance(lot) - lot.wall_gap,
        pack_from="end",
    )

    # ---------- Diagonal top wall: 11 angled bays hugging P7 end -------
    diag_wall_len = math.hypot(P7[0] - P6[0], P7[1] - P6[1])
    ang_spacing = dims_ang["width"] / math.sin(math.radians(45.0))
    end_clearance = angled_corner_clearance(lot)
    diag_ang = lot.row_along_perimeter(
        bay_type="angled", n=11, wall=WALL_TOP_DIAGONAL, bay_angle_deg=45.0,
        start_along=diag_wall_len - end_clearance - 11 * ang_spacing,
    )

    # ---------- Top-flat back-to-back perp rows ------------------------
    # Row 1 sits flush against the top-flat wall with backs to it; row 2
    # mirrors row 1 across a 6 m aisle.
    top_flat_back = lot.row_along_perimeter(
        bay_type="perpendicular", n=7, wall=WALL_TOP_FLAT, centred=True,
    )
    top_flat_facing = lot.facing_row(top_flat_back, gap=TOP_FLAT_AISLE)

    # ---------- Notch top wall: 4 perpendicular bays -------------------
    notch_perp = lot.row_along_perimeter(
        bay_type="perpendicular", n=4, wall=WALL_NOTCH_TOP, centred=True,
    )

    # ---------- Bottom + right parallel groups -------------------------
    bottom_par = lot.row_along_perimeter(
        bay_type="parallel", n=5, wall=WALL_BOTTOM_LEFT,
        bay_angle_deg=90.0, centred=True,
    )
    right_par = lot.row_along_perimeter(
        bay_type="parallel", n=4, wall=WALL_RIGHT,
        bay_angle_deg=90.0, centred=True,
    )

    # ---------- Spawns -------------------------------------------------
    lot.spawn(x=3.0, y=25.0, yaw_deg=0.0, primary=True)
    _diagonal_top_spawn(lot)
    lot.spawn(x=70.0, y=3.0, yaw_deg=90.0)

    # ---------- Patrol path (5-waypoint CCW orbit around obstacle) -----
    patrol = PatrolPath()
    y_lower = patrol.aisle_y(below=bottom_par, above=obs_south)
    y_upper = patrol.aisle_y(below=obs_north, above=diag_ang)
    x_left = patrol.aisle_x(left=0.0, right=obs_west)
    x_right = patrol.aisle_x(left=obs_east, right=notch_perp)
    patrol.add(x_left, y_lower)
    patrol.add(x_left, 28.0)
    patrol.add(26.0, 34.0)
    patrol.add(x_right, y_upper)
    patrol.add(x_right, y_lower)
    lot.set_patrol(patrol)

    # ---------- Pedestrian zones ---------------------------------------
    lot.add_zone(PedestrianZone.along_row(obs_south, side="south"))
    lot.add_zone(PedestrianZone.along_row(obs_north, side="north"))
    lot.add_zone(PedestrianZone.between_rows(top_flat_back, top_flat_facing))
    lot.add_zone(PedestrianZone.along_row(notch_perp, side="north"))
    lot.add_zone(PedestrianZone.along_row(right_par, side="west"))

    # ---------- Interior obstacle --------------------------------------
    lot.add_obstacle(*OBSTACLE)

    return lot.build()


def _diagonal_top_spawn(lot: LotBuilder) -> None:
    """@brief Add the diagonal top-wall spawn 3 m inward from the wall slope."""
    wall_dx = P7[0] - P6[0]    # -60
    wall_dy = P7[1] - P6[1]    # +13
    wall_len = math.hypot(wall_dx, wall_dy)
    s2_y = P6[1] + (wall_dy / wall_dx) * (70.0 - P6[0])
    yaw = math.degrees(math.atan2(wall_dx / wall_len, -wall_dy / wall_len)) % 360.0
    x = round(70.0 + math.cos(math.radians(yaw)) * 3.0, 1)
    y = round(s2_y + math.sin(math.radians(yaw)) * 3.0, 1)
    lot.spawn(x=x, y=y, yaw_deg=round(yaw, 1))
