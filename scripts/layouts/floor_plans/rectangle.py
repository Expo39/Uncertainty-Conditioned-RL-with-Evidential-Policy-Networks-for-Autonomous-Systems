"""
@file rectangle.py
@brief Rectangle parking lot floor plan (60x45 m).
"""

from typing import Any, Dict, Tuple

from scripts.layouts.builder import LotBuilder, PatrolPath, PedestrianZone

ORIGIN_X = 2.0
ORIGIN_Y = 22.5
ORIGIN_Z = 0.3
HEADING_DEG = 0.0
OOD = False

# Lot dimensions and structural offsets (the only numeric inputs).
WIDTH = 42.5
DEPTH = 60.0
LEFT_X = -5.0

# Walls (CCW polygon order: bottom -> right -> top -> left).
WALL_BOTTOM = 0
WALL_RIGHT = 1
WALL_TOP = 2
WALL_LEFT = 3


def generate() -> Dict[str, Any]:
    """
    @brief Build the rectangle layout using the LotBuilder DSL.
    """
    lot = LotBuilder(
        name="rectangle",
        corners=[(LEFT_X, 0.0), (DEPTH, 0.0), (DEPTH, WIDTH), (LEFT_X, WIDTH)],
    )
    # ---------- Bays ---------------------------------------------------
    centre_perp = _centre_perp_row(lot)
    top_perp = lot.row_along_perimeter(
        "perpendicular",
        n=8,
        wall=WALL_TOP,
        centred=True,
    )
    # Start 2 bay-widths from the left corner so the gap near spawn 1 is empty.
    # Spawn 1 is hardcoded below to its original position (midpoint of
    # left_perp.bays[-1] and the now-absent first bay) so it does not shift.
    bottom_perp = lot.row_along_perimeter(
        "perpendicular",
        n=8,
        wall=WALL_BOTTOM,
        start_along=8.25,
    )
    bottom_right_perp = lot.row_along_perimeter(
        "perpendicular",
        n=5,
        wall=WALL_BOTTOM,
        pack_from="end",
    )
    left_perp = lot.row_along_perimeter(
        "perpendicular",
        n=7,
        wall=WALL_LEFT,
        start_along=2.0,  # Start near top wall, respecting perimeter gap.
    )
    right_perp = lot.row_along_perimeter(
        "perpendicular",
        n=5,
        wall=WALL_RIGHT,
        # Clear the bottom-wall row's footprint at the corner.
        start_along=14.0,
    )
    moto_first, _ = _motorcycle_corner_bays(lot)

    # ---------- Spawns -------------------------------------------------
    # Spawn 2: midpoint of the last bottom-left bay and the last bottom-right bay.
    # Spawn 3: midpoint of the first top-wall bay and the upper motorcycle bay.
    # Spawn 1: original midpoint of left_perp.bays[-1] and the removed first
    # bottom bay; hardcoded so the gap left by the removed bays does not shift it.
    s1_x, s1_y = -2.3, 12.625
    s2_x, s2_y = _midpoint(bottom_perp.bays[-1], bottom_right_perp.bays[-1])
    s3_x, s3_y = _midpoint(top_perp.bays[0], moto_first)
    lot.spawn(x=s1_x, y=s1_y, yaw_deg=0.0, primary=True)
    lot.spawn(x=s2_x, y=s2_y, yaw_deg=90.0)
    lot.spawn(x=s3_x, y=s3_y, yaw_deg=270.0)

    # ---------- Patrol path (4-waypoint CCW loop) ----------------------
    # Loop the open aisles: lower aisle (between bottom rows and the perp
    # centre row) -> right aisle (in front of the right-wall row) -> upper
    # aisle (between the perp centre row and the top-wall row) -> left aisle
    # (in front of the left-wall row).
    patrol = PatrolPath()
    y_lower = patrol.aisle_y(below=[bottom_perp, bottom_right_perp], above=centre_perp)
    y_upper = patrol.aisle_y(below=centre_perp, above=top_perp)
    x_left = patrol.aisle_x(left=left_perp, right=centre_perp)
    x_right = patrol.aisle_x(left=centre_perp, right=right_perp)
    patrol.add(x_left, y_lower)
    patrol.add(x_right, y_lower)
    patrol.add(x_right, y_upper)
    patrol.add(x_left, y_upper)
    lot.set_patrol(patrol)

    # ---------- Pedestrian zones ---------------------------------------
    lot.add_zone(PedestrianZone.along_row(bottom_perp, side="north"))
    lot.add_zone(PedestrianZone.along_row(bottom_right_perp, side="north"))
    lot.add_zone(PedestrianZone.along_row(left_perp, side="east"))
    lot.add_zone(PedestrianZone.between_rows(centre_perp, top_perp))
    # Top row sits flush against the top wall, noses facing south into the lot;
    # the walkway hugs its nose face.
    lot.add_zone(PedestrianZone.along_row(top_perp, side="nose"))

    return lot.build()


def _centre_perp_row(lot: LotBuilder):
    """@brief Place the perpendicular centre row through the lot centre."""
    centre_x = (LEFT_X + DEPTH) / 2.0
    centre_y = WIDTH / 2.0
    return lot.row_centred(
        bay_type="perpendicular",
        n=12,
        centre=(centre_x, centre_y),
        direction="east",
        yaw_deg=90.0,
    )


def _midpoint(
    bay_a: Dict[str, Any], bay_b: Dict[str, Any]
) -> Tuple[float, float]:
    """@brief Local-frame midpoint between two bay centres."""
    return (
        (bay_a["local_x"] + bay_b["local_x"]) / 2.0,
        (bay_a["local_y"] + bay_b["local_y"]) / 2.0,
    )


def _motorcycle_corner_bays(
    lot: LotBuilder,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    @brief Place the two always-empty motorcycle bays in the top-right corner.
    @return The (upper, lower) motorcycle bay dicts in placement order.
    """
    cx = DEPTH - 3.0 / 2.0 - lot.wall_gap
    cy_top = WIDTH - 1.5 / 2.0 - lot.wall_gap - 1.0
    groups = [
        lot.place_bay(
            bay_type="motorcycle",
            x=cx,
            y=cy,
            yaw_deg=0.0,
            width=1.5,
            depth=3.0,
            bay_extras={"always_empty": True, "occupant": occupant},
        )
        for cy, occupant in (
            (cy_top, "Kawasaki Ninja"),
            (cy_top - 1.5, "Yamaha YZF-R"),
        )
    ]
    return groups[0].bays[0], groups[1].bays[0]
