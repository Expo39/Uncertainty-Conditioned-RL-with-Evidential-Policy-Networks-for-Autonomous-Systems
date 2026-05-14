"""
@file trapezoid.py
@brief Trapezoid parking lot floor plan (front=60, rear=40, depth=44 m).
"""

import math
from typing import Any, Dict

from scripts.layouts.builder import (
    LotBuilder,
    PatrolPath,
    PedestrianZone,
    angled_corner_clearance,
)

ORIGIN_X = 0.0
ORIGIN_Y = 30.0
ORIGIN_Z = 0.3
HEADING_DEG = 0.0
OOD = False

# Lot dimensions and structural offsets.
WIDTH_FRONT = 60.0
WIDTH_REAR = 40.0
DEPTH = 50.0
LEFT_X = -5.0
Y_OFFSET = (WIDTH_FRONT - WIDTH_REAR) / 2.0  # = 10 (top/bottom wall taper).
CENTRE_X_SHIFT = 18.0  # Cluster right-shift from depth/2 to touch right wall at x=50.
PERP_AISLE = 6.0  # Aisle between back-to-back perp rows.

# Walls (CCW polygon order: bottom -> right -> top -> left).
WALL_BOTTOM = 0
WALL_RIGHT = 1
WALL_TOP = 2
WALL_LEFT = 3


def generate() -> Dict[str, Any]:
    """
    @brief Build the trapezoid layout using the LotBuilder DSL.
    """
    p0 = (LEFT_X, 0.0)  # bottom-left (front)
    p1 = (DEPTH, Y_OFFSET)  # bottom-right (rear)
    p2 = (DEPTH, WIDTH_FRONT - Y_OFFSET)  # top-right (rear)
    p3 = (LEFT_X, WIDTH_FRONT)  # top-left (front)

    lot = LotBuilder(name="trapezoid", corners=[p0, p1, p2, p3])

    # ---------- Bays ---------------------------------------------------
    centre_low, centre_high = lot.row_pair_back_to_back(
        bay_type="perpendicular",
        n=5,
        centre=(DEPTH / 2.0 + CENTRE_X_SHIFT, WIDTH_FRONT / 2.0),
        direction="east",
        gap=PERP_AISLE + 2.0,
    )
    centre_mid_low, centre_mid_high = lot.row_pair_back_to_back(
        bay_type="perpendicular",
        n=7,
        centre=(DEPTH / 2.0 - 1.5 - 5.0 + 3.0, WIDTH_FRONT / 2.0),
        direction="east",
        gap=PERP_AISLE,
    )
    bottom_ang = lot.row_along_perimeter(
        "angled",
        n=7,
        wall=WALL_BOTTOM,
        bay_angle_deg=45.0,
        start_along=angled_corner_clearance(lot) + 10.0,
        pack_from="start",
    )
    top_ang = lot.row_along_perimeter(
        "angled",
        n=7,
        wall=WALL_TOP,
        bay_angle_deg=225.0,
        start_along=angled_corner_clearance(lot) + 15.0,
        pack_from="end",
    )
    # Left wall: angled bays at the top corner, leaning toward top wall.
    left_ang = lot.row_along_perimeter(
        "angled",
        n=6,
        wall=WALL_LEFT,
        bay_angle_deg=-45.0,
        start_along=angled_corner_clearance(lot, angle_deg=45.0) + 1.0,
    )
    # Left wall: perpendicular cluster below the primary spawn (y < 30).
    lot.row_along_perimeter(
        "perpendicular",
        n=5,
        wall=WALL_LEFT,
        bay_angle_deg=0.0,
        start_along=38.0,
    )

    # ---------- Spawns -------------------------------------------------
    lot.spawn(x=-2.0, y=WIDTH_FRONT / 2.0, yaw_deg=0.0, primary=True)
    _diagonal_bottom_spawn(lot, p0, p1)

    # ---------- Patrol path (4-waypoint loop) --------------------------
    patrol = PatrolPath()
    y_lower = patrol.aisle_y(below=bottom_ang, above=centre_low)
    y_upper = patrol.aisle_y(below=centre_high, above=top_ang)
    x_enter = patrol.aisle_x(left=0.0, right=centre_mid_low)
    x_exit = patrol.aisle_x(left=centre_mid_low, right=centre_low)
    patrol.add(x_enter, y_lower - 5.0)
    patrol.add(x_exit, y_lower - 1.0)
    patrol.add(x_exit, y_upper + 1.0)
    patrol.add(x_enter, y_upper + 5.0)
    lot.set_patrol(patrol)

    # ---------- Pedestrian zones ---------------------------------------
    lot.add_zone(
        PedestrianZone(
            x_min=left_ang.bbox[1] + 0.5,
            x_max=left_ang.bbox[1] + 3.5,
            y_min=left_ang.bbox[2] + 1.5,
            y_max=left_ang.bbox[3] - 3.5,
        )
    )
    lot.add_zone(PedestrianZone.along_row(centre_low, side="south"))
    lot.add_zone(PedestrianZone.along_row(centre_high, side="north"))
    lot.add_zone(PedestrianZone.along_row(centre_mid_low, side="south"))
    lot.add_zone(PedestrianZone.between_rows(centre_mid_low, centre_mid_high))
    lot.add_zone(PedestrianZone.along_row(centre_mid_high, side="north"))

    return lot.build()


def _diagonal_bottom_spawn(lot: LotBuilder, p0, p1) -> None:
    """@brief Add the diagonal-wall spawn 3 m inward from the bottom wall."""
    wall_len = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
    wdx = (p1[0] - p0[0]) / wall_len
    wdy = (p1[1] - p0[1]) / wall_len
    yaw = math.degrees(math.atan2(wdx, -wdy))
    x = round(38.0 + math.cos(math.radians(yaw)) * 3.0, 1)
    y = round(Y_OFFSET * (38.0 / DEPTH) + math.sin(math.radians(yaw)) * 3.0, 1)
    lot.spawn(x=x, y=y, yaw_deg=round(yaw, 1))
