# Layout

Geometry derivations and design rationale for the three parking-lot layouts used in simulation:
`rectangle`, `trapezoid`, and `irregular_a`.

## Source files

- `scripts/layouts/rectangle.py`, `trapezoid.py`, `irregular_a.py`, `common.py`
- `uncertainty_rl/envs/sim/_npc_controller.py` (patrol path geometry)
- `uncertainty_rl/envs/sim/_lot_spawner.py` (bay sampling, cone placement)

## See also

- `docs/detailed_notes/observation_space.md` - LiDAR sector derivation that depends
  on lot geometry.

---

## Patrol path derivation

Each layout computes patrol waypoint positions as midpoints between facing bay surfaces,
ensuring the NPC patrol vehicle always drives along navigable corridors.

### Rectangle

Corridor positions are derived from the centre and wall bay cluster geometry:

- `patrol_x_left`: midpoint between the left lot wall (x=-5) and the left edge of the centre cluster (min of perp and angled cluster x extents).
- `patrol_y_lower`: midpoint between the bottom-wall bay nose faces (perp and angled groups, whichever projects furthest) and the centre row A (perp) front face.
- `patrol_x_par_aisle`: midpoint between the inner faces of the right parallel column (Group B) and the inner parallel column (Group C).
- `patrol_y_upper`: midpoint between the top parallel bay nose faces and the top face of the centre angled row B.
- `patrol_y_top_aisle`: midpoint between the top parallel bay front faces and the top of the right parallel columns.
- `patrol_diag_start_x`: midpoint between the right end of the top parallel group and the outer face of the inner parallel column (Group C).
- `patrol_diag_end_x`: after a 45-degree diagonal from `patrol_diag_start_x`, dropping from `patrol_y_top_aisle` to `patrol_y_upper` (equal delta x and y).

### Trapezoid

Corridor y values:

- `_patrol_lower_cy`: midpoint between row A nose face and the maximum-y corner of the angled bay footprints (computed by rotating the angled bay half-extents through `ang_yaw`).
- `_patrol_upper_cy`: midpoint between row B nose face and the minimum-y corner of the top-wall parallel bay footprints (computed by rotating the parallel bay half-extents through `top_par_yaw`).

x extents use `x_enter` (midpoint between left boundary and left perp cluster edge) and `x_exit` (midpoint between right perp cluster edge and right-wall parallel inner face).

### Irregular-a

Five-waypoint CCW orbit around the central obstacle:

- `_lower_y`: midpoint between obstacle row C nose faces (bottom side, y=11.5) and the top faces of the bottom parallel bays.
- `_upper_y`: midpoint between obstacle row D nose faces (top side, y=26.5) and the approximate lowest y of the diagonal top-wall bay footprints (approximated as 37 - ang_offset - depth/2).
- `_left_x`: midpoint between the right edges of the left-wall angled bays and the left nose faces of obstacle row E.
- `_right_x`: midpoint between the right nose faces of obstacle row F and the left edge of the notch perpendicular bays.
- WP3 (26, 34): manual chamfer point to route the patrol around the top-left perpendicular cluster without clipping it.

---

## Bay sampling

Source: `_LotSpawner._spawn_static_vehicles()` in `envs/sim/helpers/_lot_spawner.py`.

Each episode a fresh occupancy rate is drawn:

    bay_occupancy_rate ~ Uniform(bay_occupancy_min, bay_occupancy_max)

For every bay in the layout YAML the spawner applies three exclusion checks in order:

1. `bay_id` matches the target bay - always excluded (agent must be able to enter it).
2. `always_empty: true` flag in the YAML - layout-level reservation. Never occupied
   regardless of occupancy rate.
3. `random.random() > bay_occupancy_rate` - probabilistic occupancy. At the minimum
   rate the lot is mostly empty; at the maximum it is nearly full.

Bays that pass all three checks receive a randomly selected CARLA car blueprint with a
randomly chosen colour. A 50 % coin flip then rotates the parked car 180 deg, simulating
both nose-in and nose-out orientations for the same bay.

`motorcycle` bay type bypasses all three checks: its occupant blueprint is fixed by the
`occupant` key in the YAML and is always spawned.

The per-episode `bay_occupancy_rate` is the primary source of difficulty variation for the
LiDAR obstacle clearance features. High occupancy forces the ego to navigate narrow gaps;
low occupancy produces wide clearances with little obstacle signal.

## Cone interpolation

Source: `_interpolate_cone_positions()` in `uncertainty_rl/utils/geometry.py`,
called by `_LotSpawner._spawn_perimeter_cones()` and `_spawn_obstacle_cones()`.

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
