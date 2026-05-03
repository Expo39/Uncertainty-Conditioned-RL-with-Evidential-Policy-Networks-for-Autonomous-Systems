"""
@file rectangle.py
@brief Rectangle parking lot floor plan (60x45 m).

Mixed bay layout to maximise training diversity across all three bay types:
  - Centre row A (lower): 12 perpendicular bays, yaw=90 (nose +Y).
  - Centre row B (upper):  8 angled bays (45 deg), yaw=225 (nose lower-left).
    Rows share a central aisle (6 m back-face gap); both are left-shifted so
    extra bays land 2 on the left, 1 on the right vs. a centred cluster.
  - Bottom wall left group  (y=0): 12 perpendicular bays, yaw=90.
  - Bottom wall right group (y=0):  7 angled bays (45 deg), yaw=135.
    Both groups are split by a ~12 m gap around x=30 for Spawn 2 entry.
  - Top wall parallel group:   4 bays, yaw=0  (nose +X).
  - Right wall parallel group: 3 bays, yaw=270 (nose -Y), against x=60.
  - Inner parallel column:     3 bays, yaw=270, 9 m aisle-to-aisle inward
    from the right-wall group, forming a back-to-back parallel pair.
  - Motorcycle bays: 2 narrow slots in the top-right corner (always_empty=True).

Two spawn transforms (3 m inside the perimeter along each heading):
  - Primary   (S1): left wall entry (x=-2, y=22.5), facing +X into the lot.
  - Secondary (S2): bottom wall entry at x=30, y=3, facing +Y.

Bay dimensions follow German EAR 05 / EU harmonised practice (FGSV 2005).
Used as a training layout (OOD=False).
"""

import math
from typing import Any, Dict

from scripts.layouts.builder import LotBuilder, PatrolPath, PedestrianZone

ORIGIN_X = 2.0
ORIGIN_Y = 22.5
ORIGIN_Z = 0.3
HEADING_DEG = 0.0
OOD = False


def generate() -> Dict[str, Any]:
    """
    @brief Build the rectangle layout using the LotBuilder DSL.
    """
    width = 45.0
    depth = 60.0

    lot = LotBuilder(
        name="rectangle",
        corners=[(-5.0, 0.0), (depth, 0.0), (depth, width), (-5.0, width)],
    )
    dims_perp = lot.dims["perpendicular"]
    dims_ang = lot.dims["angled"]
    dims_par = lot.dims["parallel"]
    ang_spacing = dims_ang["width"] / math.sin(math.radians(45.0))

    # ------------------------------------------------------------------
    # Centre back-to-back rows (left-shifted by _CENTRE_X_OFFSET so the cluster
    # has 2 extra bays on the left, 1 on the right vs a centred cluster).
    # Row A is perpendicular, row B is angled - so we use two row_centred()
    # calls rather than row_pair_back_to_back() which assumes one bay type.
    # ------------------------------------------------------------------
    _CENTRE_BACK_GAP = 6.0
    _CENTRE_X_OFFSET = -7.25
    centre_x = depth / 2.0 + _CENTRE_X_OFFSET
    centre_mid_y = width / 2.0
    centre_row_a_y = centre_mid_y - _CENTRE_BACK_GAP / 2.0 - dims_perp["depth"] / 2.0
    centre_row_b_y = (
        centre_mid_y + _CENTRE_BACK_GAP / 2.0 + dims_ang["depth"] / 2.0 + 1.0
    )

    centre_perp = lot.row_centred(
        bay_type="perpendicular",
        n=12,
        centre=(centre_x, centre_row_a_y),
        direction="east",
        yaw_deg=90.0,
    )
    centre_ang = lot.row_centred(
        bay_type="angled",
        n=8,
        centre=(centre_x, centre_row_b_y),
        direction="east",
        yaw_deg=225.0,
        spacing=ang_spacing,
    )

    # ------------------------------------------------------------------
    # Bottom wall (y=0): 12 perpendicular bays packed left, 7 angled bays
    # packed right (yaw=135 leans toward upper-left). Both use row_along_wall
    # so wall-clearance maths stay inside the builder.
    # ------------------------------------------------------------------
    bottom_wall_p0 = (-5.0, 0.0)
    bottom_wall_p1 = (depth, 0.0)
    bottom_perp = lot.row_along_wall(
        bay_type="perpendicular",
        n=12,
        wall_p0=bottom_wall_p0,
        wall_p1=bottom_wall_p1,
        bay_angle_deg=0.0,
        side="ccw",
        pack_from="start",
    )
    bottom_ang = lot.row_along_wall(
        bay_type="angled",
        n=7,
        wall_p0=bottom_wall_p0,
        wall_p1=bottom_wall_p1,
        bay_angle_deg=45.0,
        side="ccw",
        pack_from="end",
    )

    # ------------------------------------------------------------------
    # Parallel groups:
    # - Top wall: 4 bays, yaw=0, depth (8 m) along X. Left-anchored.
    # - Right wall: 3 bays, yaw=270, depth along Y. Y-anchored.
    # - Inner column (Group C): 3 bays, yaw=270, 9 m aisle-to-aisle inward
    #   from the right-wall group.
    # ------------------------------------------------------------------
    par_cy_top = width - dims_par["width"] / 2.0 - lot.wall_gap
    par_top_x_start = 5.0
    lot.row(
        bay_type="parallel",
        n=4,
        anchor=(par_top_x_start + dims_par["depth"] / 2.0, par_cy_top),
        direction="east",
        yaw_deg=0.0,
        spacing=dims_par["depth"],
    )

    par_cx_right = depth - dims_par["width"] / 2.0 - lot.wall_gap
    par_right_y_start = width - 2.0 - 3 * dims_par["depth"] - 6.0
    lot.row(
        bay_type="parallel",
        n=3,
        anchor=(par_cx_right, par_right_y_start + dims_par["depth"] / 2.0),
        direction="north",
        yaw_deg=270.0,
        spacing=dims_par["depth"],
    )

    par_cx_col2 = par_cx_right - dims_par["width"] - 9.0
    par_col2 = lot.row(
        bay_type="parallel",
        n=3,
        anchor=(par_cx_col2, par_right_y_start + dims_par["depth"] / 2.0),
        direction="north",
        yaw_deg=270.0,
        spacing=dims_par["depth"],
    )

    # ------------------------------------------------------------------
    # Motorcycle bays: 2 narrow slots in the top-right corner (always_empty).
    # Different occupants per bay -> placed individually via place_bay().
    # ------------------------------------------------------------------
    moto_cx = depth - 3.0 / 2.0 - lot.wall_gap
    moto_cy_top = width - 1.5 / 2.0 - lot.wall_gap
    lot.place_bay(
        bay_type="motorcycle",
        x=moto_cx,
        y=moto_cy_top,
        yaw_deg=0.0,
        width=1.5,
        depth=3.0,
        bay_extras={"always_empty": True, "occupant": "Kawasaki Ninja"},
    )
    lot.place_bay(
        bay_type="motorcycle",
        x=moto_cx,
        y=moto_cy_top - 1.5,
        yaw_deg=0.0,
        width=1.5,
        depth=3.0,
        bay_extras={"always_empty": True, "occupant": "Yamaha YZF-R"},
    )

    # ------------------------------------------------------------------
    # Spawns (3 m inside the perimeter along each heading).
    # ------------------------------------------------------------------
    lot.spawn(x=-2.0, y=width / 2.0, yaw_deg=0.0, primary=True)
    lot.spawn(x=depth / 2.0, y=3.0, yaw_deg=90.0)

    # ------------------------------------------------------------------
    # Patrol path: 6-waypoint CCW loop tracing the driving aisles.
    # See documentation/detailed_notes/layout/patrol_paths.md for derivation.
    # ------------------------------------------------------------------
    centre_perp_xmin, centre_perp_xmax, _, _ = centre_perp.bbox
    centre_ang_xmin, centre_ang_xmax, _, centre_ang_ymax = centre_ang.bbox
    par_top_xmax = par_top_x_start + 4 * dims_par["depth"]
    par_top_front_y = par_cy_top - dims_par["width"] / 2.0
    par_col1_inner_x = par_cx_right - dims_par["width"] / 2.0
    par_col2_inner_x = par_cx_col2 + dims_par["width"] / 2.0
    par_col2_outer_x = par_cx_col2 - dims_par["width"] / 2.0
    par_col_top_y = par_right_y_start + 3 * dims_par["depth"]

    centre_left_edge = min(centre_perp_xmin, centre_ang_xmin)
    patrol_x_left = centre_left_edge / 2.0
    patrol_x_par_aisle = (par_col1_inner_x + par_col2_inner_x) / 2.0

    _, _, _, bottom_perp_ymax = bottom_perp.bbox
    _, _, _, bottom_ang_ymax = bottom_ang.bbox
    lower_aisle_max_nose = max(bottom_perp_ymax, bottom_ang_ymax)
    patrol_y_lower = (lower_aisle_max_nose + centre_perp.back_y) / 2.0

    patrol_y_upper = (par_top_front_y + centre_ang_ymax) / 2.0
    patrol_y_top_aisle = (par_top_front_y + par_col_top_y) / 2.0

    patrol_diag_start_x = (par_top_xmax + par_col2_outer_x) / 2.0
    patrol_diag_end_x = patrol_diag_start_x - (patrol_y_top_aisle - patrol_y_upper)

    patrol = PatrolPath()
    patrol.add(patrol_x_left, patrol_y_lower)
    patrol.add(patrol_x_par_aisle, patrol_y_lower)
    patrol.add(patrol_x_par_aisle, patrol_y_top_aisle)
    patrol.add(patrol_diag_start_x, patrol_y_top_aisle)
    patrol.add(patrol_diag_end_x, patrol_y_upper)
    patrol.add(patrol_x_left, patrol_y_upper)
    lot.set_patrol(patrol)

    # ------------------------------------------------------------------
    # Pedestrian zones - one strip per distinct aisle face.
    # Axis-aligned rows use along_row(); zones across angled rows or spanning
    # two clusters use between_rows() or explicit bounds.
    # ------------------------------------------------------------------
    centre_x_min = min(centre_perp_xmin, centre_ang_xmin)
    centre_x_max = max(centre_perp_xmax, centre_ang_xmax)

    # Zone 1: aisle north of bottom-wall perp group nose face.
    lot.add_zone(PedestrianZone.along_row(bottom_perp, side="north"))

    # Zone 2: aisle north of bottom-wall angled group bbox (yaw=135 - not
    # axis-aligned, so use the bbox-aligned cardinal "north").
    lot.add_zone(PedestrianZone.along_row(bottom_ang, side="north"))

    # Zone 3: central aisle between row A back face and row B back face.
    # Row B is angled (yaw=225) so its bbox y extends past depth/2; the
    # original code uses depth/2 as a tighter approximation, kept here for
    # byte-equivalent output.
    lot.add_zone(
        PedestrianZone(
            x_min=centre_x_min + 0.5,
            x_max=centre_x_max - 0.5,
            y_min=centre_perp.nose_y + 0.5,
            y_max=centre_row_b_y - dims_ang["depth"] / 2.0 - 0.5,
        )
    )

    # Zone 4: aisle north of centre angled row B (yaw=225 - bbox-aligned strip).
    lot.add_zone(PedestrianZone.along_row(centre_ang, side="north"))

    # Zone 5: strip west of the inner parallel column (Group C outer face).
    lot.add_zone(PedestrianZone.along_row(par_col2, side="west"))

    return lot.build()
