# configs/layouts/

Pre-computed parking lot geometry. Each `<name>.yaml` is the world-frame (CARLA)
description of one lot - perimeter, bays, spawns, patrol path, pedestrian zones. The env
selects and loads one per episode in `reset()` via `_load_floor_plan()`, through a shared
cache so a file is parsed once per process. `flat_plane.xodr` is the OpenDRIVE world all
layouts spawn into.

> **These files are generated. Never edit a `.yaml` by hand.** They are written by
> `make generate-layouts` from the floor plan modules in
> [`scripts/layouts/floor_plans/`](../../scripts/layouts/floor_plans/).
>
> - A hand-edit is lost the next time anyone regenerates; the Python module is the only
>   durable source of truth, and each YAML carries a regenerate-me header comment saying so.
> - This holds for the `origin` block too. A lot re-anchored against a CARLA measurement
>   has the new value written into the module's `ORIGIN_X` / `ORIGIN_Y` / `ORIGIN_Z` /
>   `HEADING_DEG` constants, then the layout regenerated. See
>   [Regenerating](#regenerating).

## Files

| File | Layout | OOD | Corners | Bays | Description |
|------|--------|-----|---------|------|-------------|
| `rectangle.yaml` | `rectangle` | No | 4 | 47 perpendicular + 2 motorcycle | Axis-aligned lot. The only layout trained on |
| `trapezoid.yaml` | `trapezoid` | Yes | 4 | 39 perpendicular | Tapered lot; two bay rows sit on the slanted walls (yaw 100.3 / 259.7 deg) |
| `irregular_a.yaml` | `irregular_a` | Yes | 5 | 30 perpendicular | Five-sided lot with a diagonal wall (bay row at yaw 125.54 deg) |
| `flat_plane.xodr` | - | - | - | - | OpenDRIVE world geometry loaded into CARLA via `generate_opendrive_world()` |

Every bay in all three layouts is `3.1 x 5.7 m` except the two `1.5 x 3.0 m` motorcycle
bays in `rectangle`. During training (`eval_mode=False`) only non-OOD plans are eligible,
so a policy meets `trapezoid` and `irregular_a` for the first time at evaluation, with
weights frozen. All bays are forward perpendicular - there are no parallel or angled bays,
and the agent has no reverse gear.

## Schema

All coordinates are in **CARLA's world frame** (left-handed: Y increases rightward, yaw
clockwise-positive), in metres and degrees. Floor plan modules are authored in a
right-handed local frame; `to_world_frame()` negates Y and yaw on the way out.

`write_layout_yaml()` emits all ten top-level keys on every layout, in the order below,
so the schema is identical across files. A key with nothing to say is written empty
(`obstacles: []` for all three current layouts) rather than omitted. Only the optional
per-bay keys vary.

| Key | Type | Meaning |
|-----|------|---------|
| `floor_plan` | str | Layout name, matching the file stem |
| `ood` | bool | Provenance only: true for `trapezoid` and `irregular_a`. The env's own eligibility filter reads the `ood` flag under `parking_scenarios.floor_plans` in `env_config.yaml`, not this key - keep the two in step |
| `origin` | `{x, y, z, heading_deg}` | Lot origin in the CARLA world frame - the only block re-measured by hand |
| `spawn_transform` | `{x, y, z, yaw_deg}` | Primary spawn pose (exactly one). Index 0 of the env's spawn pool |
| `extra_spawn_transforms` | list of `{x, y, z, yaw_deg}` | Additional spawns for varied approach angles (2 for `rectangle` and `irregular_a`, 1 for `trapezoid`). Used only when the stage sets `use_extra_spawns: true` |
| `corners` | list of `{x, y}` | Perimeter polygon vertices, in the CARLA frame. Authored CCW in the local frame, so the emitted order reads clockwise once Y is negated |
| `bays` | list of bay dicts | One entry per bay (see below) |
| `patrol_waypoints` | list of `{x, y}` | Ordered NPC patrol loop, 4 waypoints per layout. Read by the NPC controller, but inert while `num_patrol_vehicles_max: 0` |
| `pedestrian_zones` | list of `{centre_x, centre_y, half_width, half_height}` | Axis-aligned walkway rectangles (4 for `rectangle`, 2 each for `trapezoid` and `irregular_a`). Inert while `pedestrian_spawn_probability: 0.0` |
| `obstacles` | list of `{centre_x, centre_y, half_width, half_height}` | Static interior obstacle rectangles. Empty (`[]`) for all three current layouts |

### Bay entry

| Key | Type | Meaning |
|-----|------|---------|
| `id` | str | Stable identifier, `<bay_type>_<index>`, e.g. `perpendicular_12`. The index runs across all bays in the layout, so `rectangle` numbers its motorcycle bays `motorcycle_47` and `motorcycle_48`. Used by curriculum `allowed_bay_ids` whitelists |
| `bay_type` | str | `perpendicular` (the only drivable target) or `motorcycle` |
| `x, y, z` | float | Bay centre in the CARLA world frame |
| `yaw_deg` | float | Bay nose direction (the heading a parked car faces) |
| `width, depth` | float | Bay footprint - perpendicular 3.1 x 5.7 m, motorcycle 1.5 x 3.0 m |
| `always_empty` | bool | Optional. Present only on the two `rectangle` motorcycle bays |
| `occupant` | str | Optional. Cosmetic label for the spawned prop, e.g. `Kawasaki Ninja` |

The env builds its target pool in two steps:

1. Drop every bay with `always_empty: true`, so the motorcycle bays stay decorative
   markers and the agent is never asked to park in one.
2. Optionally narrow to the stage's `allowed_bay_ids`. A whitelisted id that is absent or
   always-empty raises, rather than silently shrinking the pool.

Success is tested geometrically against the bay polygon via `car_fully_inside_bay()`, so
`x`, `y`, `yaw_deg`, `width` and `depth` must match the physical bay exactly.

## Regenerating

```bash
make generate-layouts                          # all three layouts
make generate-layouts LAYOUT=rectangle         # a single layout
make docker-inspect INSPECT_LAYOUT=rectangle   # verify geometry in CARLA afterwards
```

`make generate-layouts` runs on the host in `.venv/` - no Docker, no CARLA. Each run
rewrites `configs/layouts/<name>.yaml` and a bird's-eye PNG under
`outputs/raw_derived/layouts/`. `make docker-inspect` does need the CARLA stack and a
display.

Regenerate only when a floor plan module changes. To re-anchor a lot in CARLA:

1. Measure the world-frame origin with `make docker-inspect`.
2. Write it into the module's `ORIGIN_X` / `ORIGIN_Y` / `ORIGIN_Z` / `HEADING_DEG`
   constants - never edit the YAML directly.
3. Regenerate and re-inspect.

The `--origin` and `--heading` flags on `generate_layouts.py` override those constants for
a one-off check. They apply only in single-layout mode and are ignored with a warning when
generating all three, so a durable re-anchor belongs in the module.

## See also

- [scripts/layouts/README.md](../../scripts/layouts/README.md) - floor plan modules, bay groups, coordinate frame
- [scripts/layouts/BUILDER.md](../../scripts/layouts/BUILDER.md) - the `LotBuilder` DSL reference
- [configs/README.md](../README.md) - full configs directory map
- [uncertainty_rl/envs/README.md](../../uncertainty_rl/envs/README.md) - `CARLAParkingEnv` that consumes these files
