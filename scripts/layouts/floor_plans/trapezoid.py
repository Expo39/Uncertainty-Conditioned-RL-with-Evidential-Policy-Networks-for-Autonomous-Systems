"""
@file trapezoid.py
@brief Trapezoid parking lot floor plan (front=60, rear=40, depth=44 m).
"""

import math
from typing import Any, Dict

from scripts.layouts.builder import LotBuilder, PatrolPath, PedestrianZone

ORIGIN_X = 2.0
ORIGIN_Y = 30.0
ORIGIN_Z = 0.3
HEADING_DEG = 0.0
OOD = False

# Lot dimensions and structural offsets.
WIDTH_FRONT = 60.0
WIDTH_REAR = 40.0
DEPTH = 44.0
LEFT_X = -5.0
Y_OFFSET = (WIDTH_FRONT - WIDTH_REAR) / 2.0   # = 10 (top/bottom wall taper).
CENTRE_X_SHIFT = -1.5 - 3.63                  # Cluster left-shift from depth/2.
PERP_AISLE = 6.0                              # Aisle between back-to-back perp rows.
ANGLED_GATE_OFFSET = 3.0                      # Skip 3 m past angled-clearance for entrance gate.
TOP_PAR_GATE_OFFSET = 9.0                     # Skip 9 m past p0 for top parallel group.

# Walls (CCW polygon order: bottom -> right -> top -> left).
WALL_BOTTOM = 0
WALL_RIGHT = 1
WALL_TOP = 2
WALL_LEFT = 3


def generate() -> Dict[str, Any]:
    """
    @brief Build the trapezoid layout using the LotBuilder DSL.
    """
    p0 = (LEFT_X, 0.0)                        # bottom-left (front)
    p1 = (DEPTH, Y_OFFSET)                    # bottom-right (rear)
    p2 = (DEPTH, WIDTH_FRONT - Y_OFFSET)      # top-right (rear)
    p3 = (LEFT_X, WIDTH_FRONT)                # top-left (front)

    lot = LotBuilder(name="trapezoid", corners=[p0, p1, p2, p3])
    dims_par = lot.dims["parallel"]

    # ---------- Bays ---------------------------------------------------
    centre_low, centre_high = lot.row_pair_back_to_back(
        bay_type="perpendicular",
        n=8,
        centre=(DEPTH / 2.0 + CENTRE_X_SHIFT, WIDTH_FRONT / 2.0),
        direction="east",
        gap=PERP_AISLE,
    )
    bottom_ang = lot.row_along_perimeter(
        "angled", n=9, wall=WALL_BOTTOM, bay_angle_deg=45.0,
        start_along=_angled_default_clearance(lot) + ANGLED_GATE_OFFSET,
    )
    par_top = lot.row_along_perimeter(
        "parallel", n=4, wall=WALL_TOP, bay_angle_deg=-90.0,
        start_along=TOP_PAR_GATE_OFFSET + dims_par["depth"] / 2.0,
    )
    par_right = lot.row_along_perimeter(
        "parallel", n=3, wall=WALL_RIGHT, bay_angle_deg=-90.0,
        centred=True,
    )

    # ---------- Spawns -------------------------------------------------
    lot.spawn(x=-2.0, y=WIDTH_FRONT / 2.0, yaw_deg=0.0, primary=True)
    _diagonal_bottom_spawn(lot, p0, p1)

    # ---------- Patrol path (4-waypoint loop) --------------------------
    patrol = PatrolPath()
    y_lower = patrol.aisle_y(below=bottom_ang, above=centre_low)
    y_upper = patrol.aisle_y(below=centre_high, above=par_top)
    x_enter = patrol.aisle_x(left=0.0, right=centre_low)
    x_exit = patrol.aisle_x(left=centre_low, right=par_right)
    patrol.add(x_enter, y_lower - 4.0)
    patrol.add(x_exit, y_lower)
    patrol.add(x_exit, y_upper)
    patrol.add(x_enter, y_upper + 4.0)
    lot.set_patrol(patrol)

    # ---------- Pedestrian zones ---------------------------------------
    lot.add_zone(PedestrianZone.along_row(centre_low, side="south"))
    lot.add_zone(PedestrianZone.between_rows(centre_low, centre_high))
    lot.add_zone(PedestrianZone.along_row(centre_high, side="north"))
    lot.add_zone(PedestrianZone.along_row(par_right, side="west"))

    return lot.build()


def _angled_default_clearance(lot: LotBuilder) -> float:
    """@brief Default along-wall clearance for a 45-deg angled bay's leftmost corner."""
    dims_ang = lot.dims["angled"]
    return (
        (dims_ang["depth"] / 2.0 + dims_ang["width"] / 2.0)
        * math.cos(math.radians(45.0))
        + lot.wall_gap
    )


def _diagonal_bottom_spawn(lot: LotBuilder, p0, p1) -> None:
    """@brief Add the diagonal-wall spawn 3 m inward from the bottom wall."""
    wall_len = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
    wdx = (p1[0] - p0[0]) / wall_len
    wdy = (p1[1] - p0[1]) / wall_len
    yaw = math.degrees(math.atan2(wdx, -wdy))
    x = round(38.0 + math.cos(math.radians(yaw)) * 3.0, 1)
    y = round(Y_OFFSET * (38.0 / DEPTH) + math.sin(math.radians(yaw)) * 3.0, 1)
    lot.spawn(x=x, y=y, yaw_deg=round(yaw, 1))
