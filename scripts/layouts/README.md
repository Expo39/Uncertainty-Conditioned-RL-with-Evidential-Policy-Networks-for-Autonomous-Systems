# scripts/layouts/

Floor plan modules live in `floor_plans/`. Each module defines one lot geometry in **local frame** (origin at lot corner (0,0)). The orchestrator `generate_layouts.py` applies a world-frame transform and writes `configs/layouts/<name>.yaml` + `outputs/layouts/<name>.png`.

Never write world-frame coordinates in a layout module - always work in local frame.

## Quick reference

```bash
make generate-layouts                      # All three layouts
make generate-layouts LAYOUT=rectangle     # Single layout
make docker-inspect INSPECT_LAYOUT=rectangle  # Verify geometry in CARLA
```

## Floor plan summary

| Layout | File | Dims (local frame) | Bays | OOD | World origin (CARLA) |
|--------|------|--------------------|------|-----|----------------------|
| `rectangle` | `floor_plans/rectangle.py` | ~60x42.5 m | 54 | No | x=2.0, y=17.5, z=0.3 |
| `trapezoid` | `floor_plans/trapezoid.py` | front=60, rear=40, depth=50 m | 49 | No | x=0.0, y=30.0, z=0.3 |
| `irregular_a` | `floor_plans/irregular_a.py` | ~80x50 m | 58 | Yes | x=-3.0, y=25.0, z=0.3 |

---

## `floor_plans/rectangle.py` - Standard Rectangle (~60x42.5 m, Training)

Axis-aligned rectangle with centred heterogeneous cluster (perpendicular + angled bays back-to-back), perimeter bays on bottom and right walls, left-wall column of perpendicular bays, and motorcycle bays in the top-right. Three spawn points enable varied approach angles during training.

**Bay groups:**

| Group | Type | Count | Yaw | Notes |
|-------|------|-------|-----|-------|
| Left wall | Perpendicular | 7 | 0 deg (nose +X) | Vertical column starting near top wall |
| Centre row A (lower) | Perpendicular | 12 | 90 deg (nose +Y) | Centred horizontally |
| Centre row B (upper, back-to-back with A) | Angled | 8 | 225 deg | Heterogeneous cluster configuration |
| Bottom wall (west section) | Perpendicular | 12 | 90 deg (nose +Y) | Perimeter row |
| Bottom wall (east section) | Angled 45 deg | 7 | 45 deg | Perimeter diagonal row |
| Right wall | Angled 45 deg | 6 | 45 deg (nose -X) | Perimeter diagonal row |
| Motorcycle corner (always_empty=True) | Motorcycle | 2 | 0 deg | Top-right, always unoccupied |

**Total: 54 bays.**

**Spawns:**
- S1 (primary): left entrance gate, local (x=-2.0, y=17.5), facing +X (east)
- S2: bottom centre aisle, local (x=30.0, y=3.0), facing +Y (north)
- S3: top-right perimeter, local (x=52.0, y=40.5), facing -Y (south)

**Patrol:** 4-waypoint CCW loop tracing lower aisle -> right aisle -> upper aisle -> left aisle. Updated to accommodate new left-wall bay column.

**World origin (FlatPlane):** ORIGIN_X=2.0, ORIGIN_Y=17.5, ORIGIN_Z=0.3, HEADING_DEG=0.0

---

## `floor_plans/trapezoid.py` - Trapezoid (front=60, rear=40, depth=50 m, Training)

Wider at the entrance end (front width=60 m) and narrower at the rear (rear width=40 m), giving non-parallel top and bottom walls. Produces a different LiDAR wall signature compared to the rectangle, encouraging generalisation to non-rectangular geometry.

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
| Right centre cluster row A (low) | Perpendicular | 5 | Back-to-back pair, 8 m aisle |
| Right centre cluster row B (high) | Perpendicular | 5 | Back-to-back with A |
| Mid centre cluster row A (low) | Perpendicular | 7 | Back-to-back pair, 6 m aisle |
| Mid centre cluster row B (high) | Perpendicular | 7 | Back-to-back with A |
| Bottom wall | Angled 45 deg | 7 | Packed from left |
| Top wall | Angled 225 deg | 7 | Packed from left |
| Left wall - angled (top corner) | Angled -45 deg | 6 | Near top-left corner |
| Left wall - perpendicular (below spawn) | Perpendicular | 5 | Below primary spawn |

**Total: 49 bays.**

**Spawns:**
- S1 (primary): entrance gate, local (x=-2.0, y=30.0), facing +X
- S2: diagonal bottom wall at x~38, facing inward perpendicular to wall slope

**Patrol:** 4-waypoint loop entering between left angled cluster and mid perp cluster, exiting through the gap between mid and right perp clusters.

**World origin (FlatPlane):** ORIGIN_X=0.0, ORIGIN_Y=30.0, ORIGIN_Z=0.3, HEADING_DEG=0.0

---

## `floor_plans/irregular_a.py` - Nine-sided Irregular Polygon (~80x50 m, OOD)

Inspired by a shed-style building footprint. Never sampled during training - held out for out-of-distribution evaluation only. Combines three OOD features the agent has not seen: a diagonal top wall, a non-convex bottom notch, and bay groups on all four faces of an interior obstacle rectangle.

**Perimeter vertices (local frame, CCW):**

| Vertex | x | y | Description |
|--------|---|---|-------------|
| P0 | 0.0 | 0.0 | Bottom-left |
| P1 | 53.0 | 0.0 | Notch bottom-left |
| P2 | 53.0 | 8.0 | Notch top-left |
| P3 | 65.0 | 8.0 | Notch top-right |
| P4 | 65.0 | 0.0 | Notch bottom-right |
| P5 | 80.0 | 0.0 | Bottom-right |
| P6 | 80.0 | 37.0 | Diagonal wall start (top-right) |
| P7 | 20.0 | 50.0 | Diagonal/flat wall junction |
| P8 | 0.0 | 50.0 | Top-left |

**Central obstacle (CARLA cone wall):** x_min=23.25, x_max=39.25, y_min=17.0, y_max=21.0

**Bay groups:**

| Group | Type | Count | Notes |
|-------|------|-------|-------|
| Obstacle south face | Perpendicular | 6 | Backs against south face (y_min=17) |
| Obstacle north face | Perpendicular | 6 | Backs against north face (y_max=21) |
| Obstacle west face | Perpendicular | 2 | Backs against west face (x_min=23.25) |
| Obstacle east face | Perpendicular | 2 | Backs against east face (x_max=39.25) |
| Left wall (P8->P0) | Angled 45 deg | 4 | Packing from bottom upward |
| Diagonal top wall (P6->P7) | Angled 45 deg | 11 | Hugging P7 end |
| Top-flat wall (P7->P8), back row | Perpendicular | 7 | Centred along wall |
| Top-flat wall, facing row | Perpendicular | 7 | Back-to-back across 6 m aisle |
| Notch top wall (P2->P3) | Perpendicular | 4 | Centred, nose facing +Y into notch |
| Bottom-left wall (P0->P1) | Parallel | 5 | Centred |
| Right wall (P5->P6) | Parallel | 4 | Centred |

**Total: 58 bays.**

**Spawns:**
- S1 (primary): left wall mid-height, local (x=3.0, y=25.0), facing +X
- S2: diagonal top wall at x~70, facing inward perpendicular to wall slope
- S3: bottom wall right section, local (x=70.0, y=3.0), facing +Y

**Patrol:** 5-waypoint CCW orbit around the central obstacle.

**World origin (FlatPlane):** ORIGIN_X=-3.0, ORIGIN_Y=25.0, ORIGIN_Z=0.3, HEADING_DEG=0.0

---

## Shared modules

### `builder.py`

Declarative DSL for assembling floor plans. See [BUILDER.md](BUILDER.md) for the full method reference.

| Symbol | Type | Description |
|--------|------|-------------|
| `LotBuilder` | class | Top-level builder: define polygon, place bays, add spawns/zones/patrol/obstacles, call `build()` |
| `BayGroup` | class | Returned by every row method; exposes `.bbox`, `.nose_y`, `.back_y`, `.nose_x`, `.back_x` for zone/patrol alignment |
| `PedestrianZone` | class | Axis-aligned zone; construct via `along_row()`, `between_rows()`, `beside_wall()`, or explicit bounds |
| `PatrolPath` | class | Ordered waypoints with `aisle_y()`, `aisle_x()`, `add_diag_from_prev()` helpers |

### `common.py`

Low-level engine: bay-dimension constants, world-frame transform, YAML writer, PNG plotter, and validators. Floor plan modules do not import from `common.py` directly - use `builder.py`.

| Symbol | Type | Description |
|--------|------|-------------|
| `BAY_DIMS` | dict | Standard EAR 05 bay dimensions: perpendicular (2.5x5.0 m, aisle 6.0 m), angled (2.5x5.4 m, aisle 3.6 m), parallel (2.5x8.0 m, aisle 4.0 m) |
| `to_world_frame()` | function | Applies rotation and translation from local frame to CARLA world frame (negates Y and yaw for left-handed CARLA convention) |
| `write_layout_yaml()` | function | Serialises the layout dict to a YAML file consumed by `CARLAParkingEnv` |
| `plot_layout()` | function | Renders a bird's-eye PNG of the layout using Matplotlib |

### `generate_layouts.py`

Orchestrator script. Imports each floor plan module's `generate()` function, calls `to_world_frame()`, writes YAML to `configs/layouts/`, and writes PNG to `outputs/layouts/`. Invoked via `make generate-layouts`.

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

After any change to a floor plan module, re-run `make generate-layouts LAYOUT=<name>` and inspect the PNG in `outputs/layouts/<name>.png` before committing.

## See also

- [BUILDER.md](BUILDER.md) - full `LotBuilder`, `BayGroup`, `PedestrianZone`, `PatrolPath` method reference
- [scripts/inspect/README.md](../inspect/README.md) - CARLA geometry verification
- [configs/layouts/README.md](../../configs/layouts/README.md) - pre-computed YAML schema reference
- [uncertainty_rl/envs/README.md](../../uncertainty_rl/envs/README.md) - `CARLAParkingEnv` that consumes the YAML files
