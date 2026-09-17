# Layout

Extracted from `scripts/layouts/` and `uncertainty_rl/envs/sim/helpers/`.

Section 3.2 of the dissertation (`docs/AntonioGaldes_Dissertation.pdf`) is canonical for
the three lots (`rectangle`, `trapezoid`, `irregular_a`), the target-bay selection and
the per-episode occupancy draw. `scripts/layouts/README.md` describes each floor plan and
`BUILDER.md` is the full `LotBuilder` DSL reference.

This note records how the patrol corridors and cone rings are derived from the bay
geometry, which neither covers.

## Source files

- `scripts/layouts/floor_plans/rectangle.py`, `trapezoid.py`, `irregular_a.py`;
  `scripts/layouts/common.py`, `scripts/layouts/builder.py`
- `uncertainty_rl/envs/sim/helpers/_npc_controller.py` (patrol lifecycle)
- `uncertainty_rl/envs/sim/helpers/_lot_spawner.py` (bay sampling, cone placement)
- `uncertainty_rl/utils/geometry.py` (`_interpolate_cone_positions`)

---

## Patrol path construction

Patrol corridors are not hand-computed. Each floor plan builds them through `PatrolPath`,
whose `aisle_y(below, above)` and `aisle_x(left, right)` helpers take the midpoint between
two facing bay-group edges, so a waypoint sits in the navigable gap by construction
rather than by a coordinate maintained alongside the bays. `BUILDER.md` documents both
helpers and the `Edge` type they accept, which may be a `BayGroup`, a list of groups
whose bounding boxes are unioned, or a raw float standing for a notional boundary.

All three layouts use the same four-waypoint loop over two aisle rows, differing only in
which groups bound each aisle and in the small manual offsets applied to a waypoint:

| Layout | Aisle bounds | Offsets |
|--------|--------------|---------|
| `rectangle` | Lower and upper aisles bounded by the bottom, centre and top perpendicular rows; the left aisle anchors to a float just inside the wall, as that wall carries no bay row | None |
| `trapezoid` | Aisles bounded by the bottom, centre-low and top perpendicular rows; x from the left boundary and the two centre clusters | Waypoints pushed out by 1-5 m so the loop clears the tapering walls |
| `irregular_a` | Aisles bounded by the bottom and diagonal perpendicular rows against the central obstacle at 0.0; x from the obstacle and the right row | Lower waypoints raised 4 m |

The offsets are the only hand-set numbers in the paths. Everything else follows from the
bay groups, so moving a row moves its aisle with it.

---

## Bay sampling

Source: `LotSpawner._spawn_static_vehicles()` in `envs/sim/helpers/_lot_spawner.py`.

Section 3.2 of the dissertation covers the per-episode occupancy rate, the per-bay
draw and the 180 deg flip. Two exclusions it does not mention are applied first, in
order:

1. `bay_id` matches the target bay - always excluded, so the agent can enter it.
2. `always_empty: true` in the layout YAML - a layout-level reservation, never occupied
   regardless of the occupancy rate.

A `motorcycle` bay bypasses both checks and the occupancy draw: its occupant blueprint
is fixed by the `occupant` key in the YAML and is always spawned.

## Cone interpolation

Source: `_interpolate_cone_positions()` in `uncertainty_rl/utils/geometry.py`,
called by `LotSpawner._spawn_perimeter_cones()` and `_spawn_obstacle_cones()`.

### Perimeter cones

`_interpolate_cone_positions(corners, spacing)` walks the closed polygon formed by the
layout `corners` list. For each directed edge it computes the number of cones that fit at
the requested spacing, then adjusts (adaptive spacing) so the final cone of each edge lands
exactly at the far corner rather than leaving a gap. Each cone gets a `yaw_deg` equal to
the edge direction so the marker faces along the wall.

Optionally, cones within `entrance_half_width` metres (default 4.0 m) of a named entrance
point are omitted, leaving a driveable gap. The perimeter cone set is layout-keyed and
cached across episodes: if the layout is unchanged and all actors are alive the set is
reused without re-spawning.

### Obstacle cones

Interior obstacle rectangles (from the `obstacles` key of the layout YAML, stored as
`centre_x`, `centre_y`, `half_width`, `half_height`) are outlined with a separate grid:

- Top and bottom edges: `np.arange(-hw, hw + eps, spacing)` steps along X at fixed Y.
- Left and right edges: `np.arange(-hh + spacing, hh - eps, spacing)` steps along Y.
  The vertical pass starts one spacing in from each corner to avoid double-placing a cone
  at each corner (the horizontal pass already covers them).

Obstacle cones are re-spawned every episode (not cached).
