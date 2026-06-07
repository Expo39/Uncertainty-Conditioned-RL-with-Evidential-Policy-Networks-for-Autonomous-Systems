"""
@file trapezoid.py
@brief Trapezoid parking lot floor plan (front=60, rear=40, depth=44 m).
"""

import math
from typing import Any, Dict

from scripts.layouts.builder import LotBuilder, PatrolPath, PedestrianZone

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
CENTRE_X_SHIFT = 17.0  # Cluster right-shift from depth/2 (right edge flush to wall).

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

    # Bays
    centre_low = lot.row_centred(
        bay_type="perpendicular",
        n=5,
        centre=(DEPTH / 2.0 + CENTRE_X_SHIFT, WIDTH_FRONT / 2.0),
        direction="east",
        yaw_deg=90.0,
    )
    # Vertical column: bays stacked along y, noses pointing east into the aisle.
    centre_mid_low = lot.row_centred(
        bay_type="perpendicular",
        n=6,
        centre=(DEPTH / 2.0 - 1.5 - 5.0 + 1.0, WIDTH_FRONT / 2.0),
        direction="north",
        yaw_deg=0.0,
    )
    bottom_perp = lot.row_along_perimeter(
        "perpendicular",
        n=7,
        wall=WALL_BOTTOM,
        centred=True,
    )
    # Top wall: span the wall starting two bay-widths in from the right (rear)
    # corner, stopping clear of the left-wall top column at the top-left corner.
    top_perp = lot.row_along_perimeter(
        "perpendicular",
        n=11,
        wall=WALL_TOP,
        start_along=8.0,
        pack_from="start",
    )
    # Left wall: perpendicular bays at the top corner, backs to the wall.
    lot.row_along_perimeter(
        "perpendicular",
        n=5,
        wall=WALL_LEFT,
        start_along=10.0,
    )
    # Left wall: perpendicular cluster below the primary spawn (y < 30).
    lot.row_along_perimeter(
        "perpendicular",
        n=5,
        wall=WALL_LEFT,
        bay_angle_deg=0.0,
        start_along=38.0,
    )

    # Spawns
    lot.spawn(x=-2.0, y=WIDTH_FRONT / 2.0, yaw_deg=0.0, primary=True)
    _diagonal_bottom_spawn(lot, p0, p1)

    # Patrol path (4-waypoint loop)
    patrol = PatrolPath()
    y_lower = patrol.aisle_y(below=bottom_perp, above=centre_low)
    y_upper = patrol.aisle_y(below=centre_low, above=top_perp)
    x_enter = patrol.aisle_x(left=0.0, right=centre_mid_low)
    x_exit = patrol.aisle_x(left=centre_mid_low, right=centre_low)
    patrol.add(x_enter, y_lower - 5.0)
    patrol.add(x_exit, y_lower - 1.0)
    patrol.add(x_exit, y_upper + 1.0)
    patrol.add(x_enter, y_upper + 5.0)
    lot.set_patrol(patrol)

    # Pedestrian zones
    lot.add_zone(PedestrianZone.along_row(centre_low, side="south"))
    lot.add_zone(PedestrianZone.along_row(centre_mid_low, side="nose"))

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
