"""
@file rectangle.py
@brief Rectangle parking lot floor plan (60x45 m).
"""

from typing import Any, Dict

from scripts.layouts.builder import LotBuilder, PatrolPath, PedestrianZone

ORIGIN_X = 2.0
ORIGIN_Y = 17.5
ORIGIN_Z = 0.3
HEADING_DEG = 0.0
OOD = False

# Lot dimensions and structural offsets (the only numeric inputs).
WIDTH = 42.5
DEPTH = 60.0
LEFT_X = -5.0
CENTRE_BACK_GAP = 6.0  # Aisle between the two centre rows' back faces.
CENTRE_X_OFFSET = 0.0  # Cluster x-shift from lot midline (centred).
CENTRE_ROW_B_EXTRA = 1.0  # Extra cy bump on the angled centre row.

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
    dims_perp = lot.dims["perpendicular"]
    dims_ang = lot.dims["angled"]

    # ---------- Bays ---------------------------------------------------
    centre_perp, centre_ang = _centre_rows(lot, dims_perp, dims_ang)
    bottom_perp = lot.row_along_perimeter("perpendicular", n=12, wall=WALL_BOTTOM)
    bottom_ang = lot.row_along_perimeter(
        "angled",
        n=7,
        wall=WALL_BOTTOM,
        bay_angle_deg=45.0,
        pack_from="end",
    )
    left_perp = lot.row_along_perimeter(
        "perpendicular",
        n=7,
        wall=WALL_LEFT,
        start_along=2.0,  # Start near top wall, respecting perimeter gap.
    )
    right_ang = lot.row_along_perimeter(
        "angled",
        n=6,
        wall=WALL_RIGHT,
        bay_angle_deg=45.0,
        # Clear the bottom-wall angled row's footprint at the corner.
        start_along=14.0,
    )
    _motorcycle_corner_bays(lot)

    # ---------- Spawns -------------------------------------------------
    lot.spawn(x=-2.0, y=WIDTH / 2.0 - 5.0, yaw_deg=0.0, primary=True)
    lot.spawn(x=DEPTH / 2.0, y=3.0, yaw_deg=90.0)
    lot.spawn(x=52.0, y=38.5, yaw_deg=270.0)

    # ---------- Patrol path (4-waypoint CCW loop) ----------------------
    # Loop the open aisles: lower aisle -> right aisle (in front of
    # right-wall angled row) -> upper aisle -> left aisle (in front of
    # new left-wall perp row). Top region is open driveway.
    patrol = PatrolPath()
    centre_cluster = [centre_perp, centre_ang]
    y_lower = patrol.aisle_y(below=[bottom_perp, bottom_ang], above=centre_perp)
    y_upper = patrol.aisle_y(below=centre_ang, above=WIDTH)
    x_left = patrol.aisle_x(left=left_perp, right=centre_cluster)
    x_right = patrol.aisle_x(left=centre_cluster, right=right_ang)
    patrol.add(x_left, y_lower)
    patrol.add(x_right, y_lower)
    patrol.add(x_right, y_upper)
    patrol.add(x_left, y_upper)
    lot.set_patrol(patrol)

    # ---------- Pedestrian zones ---------------------------------------
    lot.add_zone(PedestrianZone.along_row(bottom_perp, side="north"))
    lot.add_zone(PedestrianZone.along_row(bottom_ang, side="north"))
    lot.add_zone(PedestrianZone.along_row(left_perp, side="east"))
    lot.add_zone(PedestrianZone.between_rows(centre_perp, centre_ang))
    lot.add_zone(PedestrianZone.along_row(centre_ang, side="north"))

    return lot.build()


def _centre_rows(lot: LotBuilder, dims_perp, dims_ang):
    """@brief Place the two heterogeneous centre rows back-to-back."""
    centre_x = DEPTH / 2.0 + CENTRE_X_OFFSET - 3.0
    centre_y = WIDTH / 2.0 + 2.5
    row_a = lot.row_centred(
        bay_type="perpendicular",
        n=12,
        centre=(centre_x, centre_y - CENTRE_BACK_GAP / 2.0 - dims_perp["depth"] / 2.0),
        direction="east",
        yaw_deg=90.0,
    )
    row_b = lot.row_centred(
        bay_type="angled",
        n=8,
        centre=(
            centre_x,
            centre_y
            + CENTRE_BACK_GAP / 2.0
            + dims_ang["depth"] / 2.0
            + CENTRE_ROW_B_EXTRA,
        ),
        direction="east",
        yaw_deg=225.0,
        spacing=dims_ang["width"] / 0.7071067811865475,
    )
    return row_a, row_b


def _motorcycle_corner_bays(lot: LotBuilder):
    """@brief Place the two always-empty motorcycle bays in the top-right corner."""
    cx = DEPTH - 3.0 / 2.0 - lot.wall_gap
    cy_top = WIDTH - 1.5 / 2.0 - lot.wall_gap - 1.0
    for cy, occupant in ((cy_top, "Kawasaki Ninja"), (cy_top - 1.5, "Yamaha YZF-R")):
        lot.place_bay(
            bay_type="motorcycle",
            x=cx,
            y=cy,
            yaw_deg=0.0,
            width=1.5,
            depth=3.0,
            bay_extras={"always_empty": True, "occupant": occupant},
        )
