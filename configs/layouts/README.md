# configs/layouts/

Pre-computed parking lot geometry. Each `<name>.yaml` is the world-frame (CARLA)
description of one lot - perimeter, bays, spawns, patrol path, pedestrian zones - consumed
by `CARLAParkingEnv` at construction. `flat_plane.xodr` is the OpenDRIVE world all layouts
spawn into.

> **These files are generated. Never edit a `.yaml` by hand.** They are written by
> `make generate-layouts` from the floor plan modules in
> [`scripts/layouts/floor_plans/`](../../scripts/layouts/floor_plans/). A hand-edit is lost
> the next time anyone regenerates, and the only durable source of truth is the Python
> module. This holds for the `origin` block too: when a lot is re-anchored against a
> CARLA measurement, the new value is written to the module's `ORIGIN_X` / `ORIGIN_Y`
> constants and the layout regenerated. See [Regenerating](#regenerating).

## Files

| File | Layout | OOD | Description |
|------|--------|-----|-------------|
| `rectangle.yaml` | `rectangle` | No | 47 perpendicular bays + 2 always-empty motorcycle bays, axis-aligned lot |
| `trapezoid.yaml` | `trapezoid` | Yes | 39 perpendicular bays, tapered (front-wide) lot |
| `irregular_a.yaml` | `irregular_a` | Yes | 30 perpendicular bays, five-sided lot with a diagonal top wall (held out from training) |
| `flat_plane.xodr` | - | - | OpenDRIVE world geometry loaded into CARLA via `generate_opendrive_world()` |

## Schema

All coordinates are in **CARLA's world frame** (left-handed: Y increases rightward, yaw
clockwise-positive), in metres and degrees. The generator emits a key only when the layout
uses it - a lot with no obstacles writes `obstacles: []`, and only the bay types actually
present appear.

| Key | Type | Meaning |
|-----|------|---------|
| `floor_plan` | str | Layout name, matching the file stem |
| `ood` | bool | True for `trapezoid` and `irregular_a` (excluded from training, evaluation-only). Only `rectangle` is trained on |
| `origin` | `{x, y, z, heading_deg}` | Lot origin in the CARLA world frame - the only block re-measured by hand |
| `spawn_transform` | `{x, y, z, yaw_deg}` | Primary spawn pose (exactly one) |
| `extra_spawn_transforms` | list of `{x, y, z, yaw_deg}` | Additional spawns for varied approach angles (at least one) |
| `corners` | list of `{x, y}` | Perimeter polygon vertices, CCW order |
| `bays` | list of bay dicts | One entry per bay (see below) |
| `patrol_waypoints` | list of `{x, y}` | Ordered NPC patrol loop (kept for tooling, though dynamic actors are out of scope) |
| `pedestrian_zones` | list of `{centre_x, centre_y, half_width, half_height}` | Axis-aligned walkway rectangles |
| `obstacles` | list of `{centre_x, centre_y, half_width, half_height}` | Static interior obstacle rectangles (empty for all current layouts) |

### Bay entry

| Key | Type | Meaning |
|-----|------|---------|
| `id` | str | Stable identifier, e.g. `perpendicular_12`. Used by curriculum `allowed_bay_ids` whitelists |
| `bay_type` | str | `perpendicular` (the only drivable target) or `motorcycle` |
| `x, y, z` | float | Bay centre in the CARLA world frame |
| `yaw_deg` | float | Bay nose direction (the heading a parked car faces) |
| `width, depth` | float | Bay footprint - perpendicular 3.1 x 5.7 m |
| `always_empty` | bool | Motorcycle bays only, marking the bay as never a parking target |
| `occupant` | str | Motorcycle bays only, a cosmetic label for the spawned prop |

The active target set is the `perpendicular` bays, optionally narrowed per curriculum stage
by `allowed_bay_ids`. Motorcycle bays are decorative markers - the agent is never asked to
park in one. Success is tested geometrically against the bay polygon via
`car_fully_inside_bay()`, so `x`, `y`, `yaw_deg`, `width`, and `depth` must match the
physical bay exactly.

## Regenerating

```bash
make generate-layouts                          # all three layouts
make generate-layouts LAYOUT=rectangle         # a single layout
make docker-inspect INSPECT_LAYOUT=rectangle   # verify geometry in CARLA afterwards
```

Regenerate only when a floor plan module changes. To re-anchor a lot in CARLA, edit
`origin.x` / `origin.y` (measured with `make docker-inspect`) in the source module's
`ORIGIN_X` / `ORIGIN_Y` constants and regenerate - never the YAML directly.

## See also

- [scripts/layouts/README.md](../../scripts/layouts/README.md) - floor plan modules, bay groups, coordinate frame
- [scripts/layouts/BUILDER.md](../../scripts/layouts/BUILDER.md) - the `LotBuilder` DSL reference
- [configs/README.md](../README.md) - full configs directory map
- [uncertainty_rl/envs/README.md](../../uncertainty_rl/envs/README.md) - `CARLAParkingEnv` that consumes these files
