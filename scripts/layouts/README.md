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

Every bay is **perpendicular** - angled, parallel, and motorcycle-drivable bays are out
of scope (bay geometry is not the experimental variable). The two motorcycle bays in the
rectangle are `always_empty` markers, never parking targets. Bay counts are taken from the
generated `configs/layouts/*.yaml`; regenerate after any module change.

| Layout | File | Polygon | Bays | OOD | World origin (CARLA) |
|--------|------|---------|------|-----|----------------------|
| `rectangle` | `floor_plans/rectangle.py` | 65 x 42.5 m axis-aligned | 47 perpendicular + 2 motorcycle | No | x=2.0, y=22.5, z=0.3 |
| `trapezoid` | `floor_plans/trapezoid.py` | front=60, rear=40, depth=50 m | 39 perpendicular | No | x=0.0, y=30.0, z=0.3 |
| `irregular_a` | `floor_plans/irregular_a.py` | five-sided, ~62 x 50 m | 36 perpendicular | Yes | x=-3.0, y=25.0, z=0.3 |

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
| Top wall | Perpendicular | 13 | Packed leftward from the centred cluster's right edge |
| Bottom wall (west) | Perpendicular | 8 | Starts two bay-widths in from the left corner |
| Bottom wall (east cluster) | Perpendicular | 5 | Inset 7 m from the bottom-right corner |
| Right wall | Perpendicular | 9 | Clears the bottom row's corner footprint |
| Motorcycle corner (`always_empty`) | Motorcycle | 2 | Bottom-left, backs to the left wall; never a target |

**Total: 47 perpendicular bays + 2 motorcycle.**

**Spawns:**
- S1 (primary): open left aisle, local (x=-2.3, y=12.625), facing +X (east)
- S2: midpoint of the two bottom-wall clusters, facing +Y (north)
- S3: open top-right corner, facing -Y (south)

**Patrol:** 4-waypoint CCW loop, lower aisle -> right aisle -> upper aisle -> left aisle.

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
| Centre row (right cluster) | Perpendicular | 5 | Centred, right-shifted, noses +Y |
| Centre column (mid cluster) | Perpendicular | 6 | Stacked along Y, noses +X |
| Bottom wall | Perpendicular | 7 | Centred along the tapered bottom wall |
| Top wall | Perpendicular | 11 | Packed from two bay-widths in at the rear corner |
| Left wall (top) | Perpendicular | 5 | Backs to the left wall, near the top corner |
| Left wall (below spawn) | Perpendicular | 5 | Below the primary spawn (y < 30) |

**Total: 39 perpendicular bays.**

**Spawns:**
- S1 (primary): entrance gate, local (x=-2.0, y=30.0), facing +X
- S2: diagonal bottom wall at x~38, facing inward perpendicular to the wall slope

**Patrol:** 4-waypoint loop entering left of the mid cluster and exiting between the mid and
right clusters.

**World origin (FlatPlane):** ORIGIN_X=0.0, ORIGIN_Y=30.0, ORIGIN_Z=0.3, HEADING_DEG=0.0

---

## `floor_plans/irregular_a.py` - Five-sided Irregular Polygon (OOD)

Never sampled during training - held out for out-of-distribution evaluation only. The OOD
feature is the diagonal top wall (P2->P3): the agent has only seen axis-aligned and tapered
walls. The earlier notch and central obstacle have been removed; the bottom boundary is a
single straight wall and every bay is perpendicular.

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
| Bottom wall | Perpendicular | 8 | Single centred cluster |
| Left wall | Perpendicular | 4 | Packed from the bottom corner upward |
| Diagonal top wall (P2->P3) | Perpendicular | 10 | Hugging the P3 end |
| Top-flat wall (P3->P4), back row | Perpendicular | 6 | Packed from the right |
| Top-flat wall, facing row | Perpendicular | 4 | Back-to-back across a 6 m aisle |
| Right wall (P1->P2) | Perpendicular | 4 | Centred |

**Total: 36 perpendicular bays.**

**Spawns:**
- S1 (primary): left wall mid-height, local (x=3.0, y=25.0), facing +X
- S2: diagonal top wall at x~57, facing inward perpendicular to the wall slope
- S3: bottom wall right section, local (x=45.0, y=5.0), facing +Y

**Patrol:** 4-waypoint CCW loop around the open lot interior.

**World origin (FlatPlane):** ORIGIN_X=-3.0, ORIGIN_Y=25.0, ORIGIN_Z=0.3, HEADING_DEG=0.0

---

## Shared modules

### `builder.py`

Declarative DSL for assembling floor plans, plus the `BAY_DIMS` table that fixes every bay's
footprint. See [BUILDER.md](BUILDER.md) for the full method reference.

| Symbol | Type | Description |
|--------|------|-------------|
| `BAY_DIMS` | dict | Per-type bay footprint and aisle width: perpendicular (3.1 x 5.7 m, aisle 6.0 m), angled (3.1 x 5.85 m, aisle 3.6 m). `WALL_GAP` (0.5 m) is the minimum clearance from any bay corner to the perimeter |
| `LotBuilder` | class | Top-level builder: define polygon, place bays, add spawns/zones/patrol/obstacles, call `build()` |
| `BayGroup` | class | Returned by every row method; exposes `.bbox`, `.nose_y`, `.back_y`, `.nose_x`, `.back_x` for zone/patrol alignment |
| `PedestrianZone` | class | Axis-aligned zone; construct via `along_row()`, `between_rows()`, `beside_wall()`, or explicit bounds |
| `PatrolPath` | class | Ordered waypoints with `aisle_y()`, `aisle_x()`, `add_diag_from_prev()` helpers |

### `common.py`

Low-level engine: bay-dimension constants, world-frame transform, YAML writer, PNG plotter, and validators. Floor plan modules do not import from `common.py` directly - use `builder.py`.

| Symbol | Type | Description |
|--------|------|-------------|
| `to_world_frame()` | function | Applies rotation and translation from local frame to CARLA world frame (negates Y and yaw for left-handed CARLA convention) |
| `write_layout_yaml()` | function | Serialises the layout dict to a YAML file consumed by `CARLAParkingEnv` |
| `plot_layout()` | function | Renders a bird's-eye PNG of the layout using Matplotlib |

### `generate_layouts.py`

Orchestrator script. Imports each floor plan module's `generate()` function, calls `to_world_frame()`, writes YAML to `configs/layouts/`, and writes PNG to `outputs/raw_derived/layouts/`. Invoked via `make generate-layouts`.

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
- [scripts/analysis/README.md](../evaluation/README.md) - eval analysis tooling (covariance / gate / uncertainty claims)
- [configs/layouts/README.md](../../configs/layouts/README.md) - pre-computed YAML schema reference
- [uncertainty_rl/envs/README.md](../../uncertainty_rl/envs/README.md) - `CARLAParkingEnv` that consumes the YAML files
