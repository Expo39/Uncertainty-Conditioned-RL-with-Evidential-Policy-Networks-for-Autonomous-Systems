"""
@file irregular_a.py
@brief Irregular seven-sided parking lot floor plan (OOD, ~85x50 m).

The notch and central obstacle have been removed. The bottom boundary is now
a single straight wall from P0 to P1. A single cluster of perpendicular bays
runs along the bottom wall.
"""

import math
from typing import Any, Dict

from scripts.layouts.builder import LotBuilder, PatrolPath, PedestrianZone

ORIGIN_X = -3.0
ORIGIN_Y = 25.0
ORIGIN_Z = 0.3
HEADING_DEG = 0.0
OOD = True

# Perimeter corners (CCW polygon order).
P0 = (0.0, 0.0)   # bottom-left
P1 = (62.0, 0.0)  # bottom-right
P2 = (62.0, 20.0)  # right wall top
P3 = (20.0, 50.0)  # diagonal/flat junction
P4 = (0.0, 50.0)   # top-left

# Wall indices in CCW order.
WALL_BOTTOM = 0   # P0 -> P1
WALL_RIGHT = 1    # P1 -> P2
WALL_TOP_DIAGONAL = 2  # P2 -> P3
WALL_TOP_FLAT = 3  # P3 -> P4
WALL_LEFT = 4     # P4 -> P0

TOP_FLAT_AISLE = 6.0  # Aisle between top-flat back-to-back perp rows.
LEFT_ANG_BOTTOM_Y = 3.0  # Local y of the bottom-most left-wall bay.


def generate() -> Dict[str, Any]:
    """
    @brief Build the irregular_a layout using the LotBuilder DSL.
    """
    lot = LotBuilder(
        name="irregular_a",
        corners=[P0, P1, P2, P3, P4],
    )
    dims_perp = lot.dims["perpendicular"]

    # ---------- Bottom wall: single centred cluster --------------------
    bottom_perp = lot.row_along_perimeter(
        bay_type="perpendicular",
        n=8,
        wall=WALL_BOTTOM,
        pack_from="start",
        start_along=16.5,
    )

    # ---------- Left wall perpendicular bays ---------------------------
    perp_corner_clearance = dims_perp["width"] / 2.0 + lot.wall_gap
    lot.row_along_perimeter(
        bay_type="perpendicular",
        n=4,
        wall=WALL_LEFT,
        start_along=LEFT_ANG_BOTTOM_Y
        - lot.wall_y(WALL_BOTTOM)
        + perp_corner_clearance
        - lot.wall_gap,
        pack_from="end",
    )

    # ---------- Diagonal top wall: perpendicular bays hugging P3 end ---
    diag_wall_len = math.hypot(P3[0] - P2[0], P3[1] - P2[1])
    perp_spacing = dims_perp["width"]
    end_clearance = perp_corner_clearance
    diag_perp = lot.row_along_perimeter(
        bay_type="perpendicular",
        n=10,
        wall=WALL_TOP_DIAGONAL,
        start_along=diag_wall_len - end_clearance - 10 * perp_spacing - 5.0,
    )

    # ---------- Top-flat back-to-back perp rows ------------------------
    top_flat_back = lot.row_along_perimeter(
        bay_type="perpendicular",
        n=6,
        wall=WALL_TOP_FLAT,
        pack_from="end",
    )
    top_flat_facing = lot.facing_row(top_flat_back, gap=TOP_FLAT_AISLE, n=4)

    # ---------- Right wall perpendicular bays --------------------------
    right_perp = lot.row_along_perimeter(
        bay_type="perpendicular",
        n=4,
        wall=WALL_RIGHT,
        centred=True,
    )

    # ---------- Spawns -------------------------------------------------
    lot.spawn(x=3.0, y=25.0, yaw_deg=0.0, primary=True)
    _diagonal_top_spawn(lot)
    lot.spawn(x=45.0, y=5.0, yaw_deg=90.0)

    # ---------- Patrol path --------------------------------------------
    patrol = PatrolPath()
    y_lower = patrol.aisle_y(below=bottom_perp, above=0.0)
    y_upper = patrol.aisle_y(below=0.0, above=diag_perp)
    x_left = patrol.aisle_x(left=0.0, right=top_flat_back)
    x_right = patrol.aisle_x(left=top_flat_back, right=right_perp)
    patrol.add(x_left, y_lower + 4.0)
    patrol.add(x_left, y_upper)
    patrol.add(x_right, y_upper)
    patrol.add(x_right, y_lower + 4.0)
    lot.set_patrol(patrol)

    # ---------- Pedestrian zones ---------------------------------------
    lot.add_zone(PedestrianZone.along_row(bottom_perp, side="north"))
    lot.add_zone(PedestrianZone.along_row(top_flat_facing, side="south"))
    lot.add_zone(PedestrianZone.along_row(right_perp, side="west"))

    return lot.build()


def _diagonal_top_spawn(lot: LotBuilder) -> None:
    """@brief Add the diagonal top-wall spawn 3 m inward from the wall slope."""
    wall_dx = P3[0] - P2[0]
    wall_dy = P3[1] - P2[1]
    wall_len = math.hypot(wall_dx, wall_dy)
    s2_y = P2[1] + (wall_dy / wall_dx) * (57.0 - P2[0])
    yaw = math.degrees(math.atan2(wall_dx / wall_len, -wall_dy / wall_len)) % 360.0
    x = round(57.0 + math.cos(math.radians(yaw)) * 3.0, 1)
    y = round(s2_y + math.sin(math.radians(yaw)) * 3.0, 1)
    lot.spawn(x=x, y=y, yaw_deg=round(yaw, 1))
