# scripts/layouts/

Floor plan modules live in `floor_plans/`. Each module defines one lot geometry in **local frame** (origin at lot corner (0,0)). The orchestrator `generate_layouts.py` applies a world-frame transform and writes `configs/layouts/<name>.yaml` + `outputs/raw_derived/layouts/<name>.png`.

Never write world-frame coordinates in a layout module - always work in local frame.

## Quick reference

```bash
make generate-layouts                      # All three layouts
make generate-layouts LAYOUT=rectangle     # Single layout
make docker-inspect INSPECT_LAYOUT=rectangle  # Verify geometry in CARLA
```

## Floor plan summary

Every target bay is **perpendicular** - forward perpendicular parking is the task, and
angled, parallel, and motorcycle bays exist only as geometry, never as targets (bay
geometry is not the experimental variable). The two motorcycle bays in the rectangle carry
`always_empty: true`, so they are excluded from the target pool; the spawner still places
their fixed `occupant` motorcycle in each one every episode. Bay counts are taken from the
generated `configs/layouts/*.yaml`, so regenerate after any module change.

| Layout | File | Polygon | Bays | OOD | World origin (CARLA) |
|--------|------|---------|------|-----|----------------------|
| `rectangle` | `floor_plans/rectangle.py` | 65 x 42.5 m axis-aligned | 47 perpendicular + 2 motorcycle | No | x=2.0, y=22.5, z=0.3 |
| `trapezoid` | `floor_plans/trapezoid.py` | front=60, rear=40, 55 m front-to-rear | 39 perpendicular | Yes | x=0.0, y=30.0, z=0.3 |
| `irregular_a` | `floor_plans/irregular_a.py` | five-sided, 62 x 50 m | 30 perpendicular | Yes | x=-3.0, y=25.0, z=0.3 |

---

## `floor_plans/rectangle.py` - Standard Rectangle (Training)

Axis-aligned rectangle (corners `(-5,0)`, `(60,0)`, `(60,42.5)`, `(-5,42.5)` in local
frame). A centred perpendicular row through the lot centre, perpendicular perimeter rows on
the top, bottom, and right walls, and two always-empty motorcycle bays in the bottom-left
corner. Three spawns give varied approach angles during training.

**Bay groups:**

| Group | Type | Count | Notes |
|-------|------|-------|-------|
| Centre row | Perpendicular | 12 | Centred through the lot, noses +Y |
| Top wall | Perpendicular | 13 | Packed leftward from x=42.0, just short of the centred row's right edge |
| Bottom wall (west) | Perpendicular | 8 | Starts two bay-widths in from the left corner |
| Bottom wall (east cluster) | Perpendicular | 5 | Packed from the bottom-right corner, 8 m beyond the default corner clearance |
| Right wall | Perpendicular | 9 | Clears the bottom row's corner footprint |
| Motorcycle corner (`always_empty`) | Motorcycle | 2 | Bottom-left, backs to the left wall, never a target; each holds a fixed motorcycle occupant |

**Total: 47 perpendicular bays + 2 motorcycle.**

**Spawns:**
- S1 (primary): open left aisle, local (x=-2.3, y=12.625), facing +X (east)
- S2: midpoint of the last bay of each bottom-wall cluster, local (x=31.25, y=3.35), facing +Y (north)
- S3: midpoint of the top-wall row's right end and the right-wall row's top end, local (x=48.55, y=38.975), facing -Y (south)

**Patrol:** 4-waypoint CCW loop built from `aisle_y` / `aisle_x` midpoints - lower aisle ->
right aisle -> upper aisle -> left aisle. The left leg is anchored 1.5 m inside the left
wall, which carries no bay row.

**World origin (FlatPlane):** ORIGIN_X=2.0, ORIGIN_Y=22.5, ORIGIN_Z=0.3, HEADING_DEG=0.0

---

## `floor_plans/trapezoid.py` - Trapezoid (OOD evaluation only)

Wider at the entrance (front width=60 m) and narrower at the rear (rear width=40 m), so the
top and bottom walls taper. This gives a different LiDAR wall signature from the rectangle,
testing generalisation to non-rectangular geometry. Held out from training (the curriculum
trains on rectangle only). All bays are perpendicular.

**Polygon corners (local frame):**

| Vertex | x | y | Description |
|--------|---|---|-------------|
| P0 | -5.0 | 0.0 | Bottom-left (front) |
| P1 | 50.0 | 10.0 | Bottom-right (rear) |
| P2 | 50.0 | 50.0 | Top-right (rear) |
| P3 | -5.0 | 60.0 | Top-left (front) |

**Bay groups:**

| Group | Type | Count | Notes |
|-------|------|-------|-------|
| Centre row (right cluster) | Perpendicular | 5 | Centred at mid-height, shifted 17 m right of the depth midpoint, noses +Y |
| Centre column (mid cluster) | Perpendicular | 6 | Stacked along Y, noses +X |
| Bottom wall | Perpendicular | 7 | Centred along the tapered bottom wall, bays tilted with the slope |
| Top wall | Perpendicular | 11 | Packed 8 m in from the rear (right) corner, bays tilted with the slope |
| Left wall (top) | Perpendicular | 5 | Backs to the left wall, packed 10 m down from the top corner (y=50 down to 37.6) |
| Left wall (below spawn) | Perpendicular | 5 | Backs to the left wall, below the primary spawn (y=22 down to 9.6) |

**Total: 39 perpendicular bays.**

**Spawns:**
- S1 (primary): entrance gate, local (x=-2.0, y=30.0), facing +X
- S2: diagonal bottom wall, 3 m inward from the wall point at x=38, local (x=37.5, y=10.6),
  yaw 100.3 deg (inward perpendicular to the wall slope)

**Patrol:** 4-waypoint CCW loop built from `aisle_y` / `aisle_x` midpoints, entering left of
the mid cluster (x=8.325) and exiting between the mid and right clusters (x=28.3). The
corner waypoints are offset a few metres along Y so the loop tracks the tapering walls.

**World origin (FlatPlane):** ORIGIN_X=0.0, ORIGIN_Y=30.0, ORIGIN_Z=0.3, HEADING_DEG=0.0

---

## `floor_plans/irregular_a.py` - Five-sided Irregular Polygon (OOD)

Never sampled during training - held out for out-of-distribution evaluation only. The OOD
feature is the steep diagonal top wall (P2->P3): the agent has only seen axis-aligned and
gently tapered walls. The bottom boundary is a single straight wall, the lot has no interior
obstacle, and every bay is perpendicular.

**Perimeter vertices (local frame, CCW):**

| Vertex | x | y | Description |
|--------|---|---|-------------|
| P0 | 0.0 | 0.0 | Bottom-left |
| P1 | 62.0 | 0.0 | Bottom-right |
| P2 | 62.0 | 20.0 | Right-wall top |
| P3 | 20.0 | 50.0 | Diagonal / flat-wall junction |
| P4 | 0.0 | 50.0 | Top-left |

**Bay groups:**

| Group | Type | Count | Notes |
|-------|------|-------|-------|
| Bottom wall | Perpendicular | 8 | Single cluster packed from 16.5 m along the wall (x=16.5 to 38.2) |
| Left wall | Perpendicular | 4 | Packed from the bottom corner upward (y=14.05 to 23.35) |
| Diagonal top wall (P2->P3) | Perpendicular | 10 | Toward the P3 end, stopping ~10 m short of the P3 corner |
| Top-flat wall (P3->P4) | Perpendicular | 4 | Anchored two bay-widths in from the P4 end and running back toward P3 (x=8.25 to 17.55) |
| Right wall (P1->P2) | Perpendicular | 4 | Centred |

**Total: 30 perpendicular bays.**

**Spawns:**
- S1 (primary): left wall mid-height, local (x=3.0, y=35.0), facing +X
- S2: diagonal top wall, 3 m inward from the wall point at x=57, local (x=55.3, y=21.1),
  yaw 234.5 deg (inward perpendicular to the wall slope)
- S3: bottom wall right section, local (x=45.0, y=5.0), facing +Y

**Patrol:** 4-waypoint loop built from `aisle_y` / `aisle_x` midpoints, tracing the open
band in the lower half of the lot between the bottom-wall row and the left-wall / right-wall
rows (x=3.35 to 37.45, y=7.1 to 10.97). The loop runs clockwise in local frame: up the left
leg, across the top, then down the right.

**World origin (FlatPlane):** ORIGIN_X=-3.0, ORIGIN_Y=25.0, ORIGIN_Z=0.3, HEADING_DEG=0.0

---

## Shared modules

### `builder.py`

Declarative DSL for assembling floor plans, plus the `BAY_DIMS` table that fixes every bay's
footprint. See [BUILDER.md](BUILDER.md) for the full method reference.

| Symbol | Type | Description |
|--------|------|-------------|
| `BAY_DIMS` | dict | Per-type bay footprint and aisle width: perpendicular (3.1 x 5.7 m, aisle 6.0 m), angled (3.1 x 5.85 m, aisle 3.6 m). These two are the only registered types; `parallel` raises, and a custom type needs explicit `width` / `depth` via `place_bay()` |
| `WALL_GAP` | float | 0.5 m - default minimum clearance from any bay corner to the perimeter (`LotBuilder(wall_gap=...)` overrides it) |
| `LotBuilder` | class | Top-level builder: define polygon, place bays, add spawns/zones/patrol/obstacles, call `build()` |
| `BayGroup` | class | Returned by every row method, exposing `.bbox`, `.nose_y`, `.back_y`, `.nose_x`, `.back_x` for zone/patrol alignment |
| `PedestrianZone` | class | Axis-aligned zone, constructed via `along_row()`, `between_rows()`, `beside_wall()`, or explicit bounds |
| `PatrolPath` | class | Ordered waypoints with `aisle_y()`, `aisle_x()`, `add_diag_from_prev()` helpers |
| `validate_bays_in_polygon()` / `warn_narrow_corridors()` | function | Validators run automatically by `build()`; also importable directly |

### `common.py`

Low-level engine: bay-dimension constants, world-frame transform, YAML writer, PNG plotter, and validators. Floor plan modules do not import from `common.py` directly - use `builder.py`.

| Symbol | Type | Description |
|--------|------|-------------|
| `to_world_frame()` | function | Applies rotation and translation from local frame to CARLA world frame (negates Y and yaw for left-handed CARLA convention). Min/max boxes (pedestrian zones, obstacles) become centre + half-extent pairs |
| `write_layout_yaml()` | function | Serialises the layout dict to a YAML file consumed by `CARLAParkingEnv`, prefixed with a regenerate-me header and the `floor_plan` / `ood` / `origin` metadata block |
| `plot_layout()` | function | Renders a bird's-eye PNG of the layout using Matplotlib, via `scripts/figure_style.py` and `scripts/colours`. Y is inverted so the PNG reads north-up despite the left-handed CARLA frame. Motorcycle bays draw in neutral grey; the legend lists only the bay types present |

### `generate_layouts.py`

Orchestrator script. Imports each floor plan module's `generate()` function, calls `to_world_frame()`, writes YAML to `configs/layouts/`, and writes PNG to `outputs/raw_derived/layouts/`. Invoked via `make generate-layouts`.

Before plotting it reads `configs/deployment/sim/env_config.yaml` so the PNG mirrors the
active training environment: the patrol path, pedestrian zones, and extra spawns are drawn
only when `num_patrol_vehicles_max > 0`, `pedestrian_spawn_probability > 0`, and
`use_extra_spawns` respectively, and the soft out-of-bounds skirt uses
`oob_inflation_margin`. A missing config falls back to drawing everything.

CLI flags: `--layout` (omit for all three), `--origin X Y Z` and `--heading` (single-layout
mode only; ignored with a warning otherwise), `--output-dir`, `--plot-dir`, `--no-plot`.

### Package markers

`__init__.py` marks this directory as a package. `floor_plans/__init__.py` re-exports each
module's `generate()` as `generate_rectangle`, `generate_trapezoid`, and
`generate_irregular_a`; `generate_layouts.py` imports the modules themselves (it needs the
`ORIGIN_*` / `HEADING_DEG` / `OOD` constants too), so a new floor plan must be registered in
its `_LAYOUTS` dict.

`CLAUDE.md` holds the working contract for editing modules in this directory.

---

## Validation

`lot.build()` runs three checks on every generate. `validate_bays_in_polygon` raises if any
bay corner escapes the lot polygon; `warn_narrow_corridors` prints a non-fatal warning for
any facing pair closer than the EAR 05 minimum 6 m aisle; `_warn_bays_close_to_wall` warns
for any bay corner inside `wall_gap` of the perimeter. `build()` also raises unless exactly
one primary spawn and at least one extra spawn were registered.

Warnings are expected where the geometry is deliberately tight - the rectangle's motorcycle
corner sits close to the adjacent perpendicular row, the trapezoid's centre row runs
near-flush to the rear wall, and irregular_a's diagonal and top-flat rows converge at the P3
corner. Read them, do not silence them.

---

## Coordinate convention

All coordinates in layout modules are **local frame**: right-handed (Y-up, yaw CCW-positive), origin at lot corner (0,0). `to_world_frame()` in `common.py` applies heading rotation, then translation, then converts to CARLA's left-handed frame (Y rightward, yaw CW-positive) by negating Y and yaw. The resulting YAML files are in CARLA world coordinates.

- Yaw convention (local frame): 0 deg = facing +X, 90 deg = facing +Y, CCW positive
- `ORIGIN_X` / `ORIGIN_Y` in module constants are in CARLA world frame

## Regenerating layouts

```bash
make generate-layouts                          # Regenerate all three
make generate-layouts LAYOUT=irregular_a       # Single layout
make docker-inspect INSPECT_LAYOUT=rectangle   # Verify in CARLA after regenerating
```

After any change to a floor plan module, re-run `make generate-layouts LAYOUT=<name>` and inspect the PNG in `outputs/raw_derived/layouts/<name>.png` before committing.

## See also

- [BUILDER.md](BUILDER.md) - full `LotBuilder`, `BayGroup`, `PedestrianZone`, `PatrolPath` method reference
- [scripts/inspect/README.md](../inspect/README.md) - CARLA geometry verification
- [scripts/analysis/README.md](../analysis/README.md) - eval analysis tooling (covariance / gate / uncertainty claims)
- [configs/layouts/README.md](../../configs/layouts/README.md) - pre-computed YAML schema reference
- [uncertainty_rl/envs/README.md](../../uncertainty_rl/envs/README.md) - `CARLAParkingEnv` that consumes the YAML files
