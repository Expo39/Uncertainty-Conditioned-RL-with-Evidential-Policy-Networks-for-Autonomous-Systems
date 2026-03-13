# scripts/layouts/

Parking lot floor plan modules. Each module defines one lot geometry in **local frame**
(origin at lot corner (0,0)). The orchestrator `scripts/generate_layouts.py` applies a
world-frame transform and writes `configs/layouts/<name>.yaml` + `outputs/layouts/<name>.png`.

Never write world-frame coordinates in a layout module - always work in local frame.

---

## Floor Plans

### `rectangle.py` - Standard Rectangle (60x45 m, Training)

Axis-aligned rectangle. Mixed bay layout designed to maximise training diversity across
all three bay types in a single floor plan.

**Bay groups:**

| Group | Type | Count | Position | Yaw |
|-------|------|-------|----------|-----|
| Centre row A | Perpendicular | 12 | Lower centre, left-shifted | 90 (nose +Y) |
| Centre row B | Angled 45 deg | 8 | Upper centre, back-to-back with row A | 225 (nose lower-left) |
| Bottom wall left | Perpendicular | 10 | y=0, x=1.25 rightward | 90 (nose +Y) |
| Bottom wall right | Angled 45 deg | 7 | y=0, packed from x=60 leftward | 135 (nose upper-left) |
| Top wall | Parallel | 4 | y=45, left side | 0 (nose +X) |
| Right wall column | Parallel | 3 | x=60, shifted down from top | 270 (nose -Y) |
| Inner column | Parallel | 3 | 9 m aisle inward from right column | 270 (nose -Y) |
| Motorcycle | Motorcycle | 2 | Top-right corner, `always_empty=True` | 0 |

**Spawns:**
- S1 (primary): left wall centre (x=0, y=22.5), facing +X
- S2: bottom wall at x=30 (centre of gap between bay groups), facing +Y
- S3: top wall at x=45, facing -Y

**Patrol:** 7-waypoint CCW loop tracing lower aisle -> parallel column aisle -> top aisle -> diagonal -> upper aisle.

**World origin (Town05_Opt):** x=-200, y=0, z=0.3, heading=0 deg

---

### `trapezoid.py` - Trapezoid (front=48, rear=30, depth=44 m, Training)

Wider at the entrance end (y=0..48) and narrower at the rear (y=9..39), giving
non-parallel top and bottom walls. Produces a subtly different LiDAR wall signature
compared to the rectangle, encouraging generalisation to non-rectangular geometry.

**Bay groups:**

| Group | Type | Count | Position | Yaw |
|-------|------|-------|----------|-----|
| Centre cluster (row A) | Perpendicular | 8 | Lower centre, left-shifted | 90 (nose +Y) |
| Centre cluster (row B) | Perpendicular | 8 | Upper centre, back-to-back with row A | 270 (nose -Y) |
| Diagonal bottom wall (P0->P1) | Angled | 7 | Along bottom diagonal wall, 45 deg to slope | Computed from wall angle |
| Top wall (P3->P2) | Parallel | 3 | Along sloped top wall, 9 m from P3 | Aligned with top wall slope |
| Right wall | Parallel | 3 | x=44, 3 m clear of bottom-right corner | 90 (nose +Y) |

**Spawns:**
- S1 (primary): left wall centre (x=0, y=24), facing +X
- S2: diagonal bottom wall near x=38, facing inward perpendicular to wall slope

**Patrol:** 4-waypoint loop through lower and upper aisles of the centre cluster.

**World origin (Town05_Opt):** x=0, y=-90, z=0.3, heading=0 deg

---

### `irregular_a.py` - Nine-sided Irregular Polygon (~80x50 m, OOD)

Inspired by a shed-style building footprint. Never sampled during training - held out
for out-of-distribution evaluation only. Combines three OOD features the agent has not
seen: a sloped top wall, a non-convex bottom notch, and bay groups on all four faces
of an interior obstacle rectangle.

**Perimeter vertices (local frame, CCW):**

| Vertex | x | y | Description |
|--------|---|---|-------------|
| P0 | 0 | 0 | Bottom-left |
| P1 | 53 | 0 | Notch left bottom |
| P2 | 53 | 8 | Notch left top |
| P3 | 65 | 8 | Notch right top |
| P4 | 65 | 0 | Notch right bottom |
| P5 | 80 | 0 | Bottom-right |
| P6 | 80 | 37 | Top-right (diagonal wall start) |
| P7 | 20 | 50 | Diagonal/flat wall junction |
| P8 | 0 | 50 | Left-wall top |

**Bay groups:**

| Group | Type | Count | Position |
|-------|------|-------|----------|
| Obstacle top face (row C) | Perpendicular | 6 | Back at obstacle y_min=17, yaw=90 |
| Obstacle bottom face (row D) | Perpendicular | 6 | Back at obstacle y_max=21, yaw=270 |
| Obstacle left face (row E) | Perpendicular | 2 | Back at obstacle x_min=23.25, yaw=0 |
| Obstacle right face (row F) | Perpendicular | 2 | Back at obstacle x_max=39.25, yaw=180 |
| Top-left cluster (row G) | Perpendicular | 7 | Back against flat top wall (y=50), yaw=270 |
| Top-left cluster (row H) | Perpendicular | 7 | Back-to-back with row G, yaw=90 |
| Diagonal top wall (P6->P7) | Angled | 11 | 45 deg to wall slope, hugging P7 end |
| Left wall | Angled | 4 | Against x=0, starting at y=3, stepping up |
| Notch top wall (y=8) | Perpendicular | 4 | Nose facing +Y into notch recess |
| Bottom wall (x=0..53) | Parallel | 5 | Depth along X, yaw=0 |
| Right wall (x=80) | Parallel | 3 | Depth along Y, yaw=90 |

**Spawns:**
- S1 (primary): left wall mid-height (x=0, y=25), facing +X
- S2: diagonal top wall at x=70, facing inward perpendicular to slope
- S3: bottom wall right section (x=70, y=0), facing +Y

**Patrol:** 5-waypoint loop threading between the obstacle cluster and the top-left cluster.

**World origin (Town05_Opt):** x=100, y=0, z=0.3, heading=0 deg

---

## Shared Modules

### `common.py`

Geometry helpers and constants shared by all layout modules.

| Symbol | Type | Description |
|--------|------|-------------|
| `BAY_DIMS` | dict | Standard EAR 05 bay dimensions: perpendicular (2.5x5.0 m, aisle 6.0 m), angled (2.5x5.4 m, aisle 3.6 m), parallel (2.5x8.0 m, aisle 4.0 m) |
| `BAYS_PER_TYPE` | int | Base bay count per type (5). Layouts may exceed this with a named constant and comment. |
| `angled_bays_along_wall()` | function | Places N angled bays along an arbitrary wall defined by direction vector, offset, and start position. Used for diagonal walls (trapezoid, irregular_a). |
| `ang_offset_from_wall()` | function | Perpendicular offset from wall to bay centre for an angled bay. |
| `ang_x_margin()` | function | Minimum along-wall margin so the first bay corner clears the wall endpoint. |
| `validate_bays_in_polygon()` | function | Raises `ValueError` if any bay centre falls outside the perimeter polygon. |
| `warn_narrow_corridors()` | function | Warns if any bay pair is closer than the minimum aisle width. |
| `to_world_frame()` | function | Applies rotation + translation from local frame to CARLA world frame. |
| `write_layout_yaml()` | function | Serialises the layout dict to a YAML file consumed by `CARLAParkingEnv`. |
| `plot_layout()` | function | Renders a bird's-eye PNG of the layout using Matplotlib. |

### `colours.py`

Single source of truth for all visualisation colours (hex strings). Used by
`common.py` (PNG plots), `scripts/inspect_layout.py` (CARLA debug overlay), and
`scripts/visualise_training.py` (live training view).

| Constant | Colour | Used for |
|----------|--------|----------|
| `HEX_PERP_BAY` | Blue | Perpendicular bay outlines |
| `HEX_ANGLED_BAY` | Yellow | Angled bay outlines |
| `HEX_PARALLEL_BAY` | Violet | Parallel bay outlines |
| `HEX_TARGET_BAY` | Bright green | Currently selected target bay |
| `HEX_PEDESTRIAN_ZONE` | Dark turquoise | Pedestrian zone fill |
| `HEX_PEDESTRIAN_ZONE_EDGE` | Dark cyan | Pedestrian zone border |
| `HEX_PATROL_PATH` | Red | Patrol waypoint path |
| `HEX_LOT` | Light grey | Lot boundary fill |

---

## Coordinate Convention

- All coordinates in layout modules are **local frame**: origin at lot corner (0,0), x pointing into the lot, y pointing along the front wall.
- `to_world_frame()` applies heading rotation then translation to CARLA world coordinates.
- Yaw convention: 0 deg = facing +X, 90 deg = facing +Y, CCW positive (matches CARLA).

## Regenerating Layouts

```bash
make generate-layouts                      # All three
make generate-layouts LAYOUT=rectangle     # One only
make docker-inspect INSPECT_LAYOUT=irregular_a  # Verify in CARLA
```
