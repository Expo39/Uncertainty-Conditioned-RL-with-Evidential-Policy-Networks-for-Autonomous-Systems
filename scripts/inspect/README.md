# scripts/inspect/

CARLA debug overlay inspector for visually verifying parking lot geometry and sensor placement before training. All modes are driven by a single entry point: `lot_inspector.py`.

Requires a windowed CARLA session (X11 display) and the `carla-server-demo` container, both started automatically by the Make targets below.

## Quick reference

| Task | Command |
|------|---------|
| Lot geometry overlay | `make docker-inspect INSPECT_LAYOUT=rectangle` |
| Sensor placement overlay | `make docker-inspect-sensors` |
| Sensor side / front profile | `make docker-inspect-sensors SENSORS_VIEW=side` |
| Full FOV arc (wide camera) | `make docker-inspect-sensors INSPECT_ZOOM=wide` |
| Live LiDAR scan dots | `make docker-inspect-live` |
| Dryrun - keyboard control | `make docker-inspect-dryrun MANUAL=true` |
| Dryrun - constant action | `make docker-inspect-dryrun` |

## Files

| File | Purpose |
|------|---------|
| `lot_inspector.py` | CLI entry point: argument parsing, env construction, mode dispatch |
| `inspectors/base.py` | `_Inspector` base class: CARLA connection, tick loop, spectator helpers |
| `inspectors/layout.py` | `LayoutInspector`: lot geometry overlays (bays, patrol path, zones) |
| `inspectors/sensor.py` | `SensorInspector`: sensor mount dots and FOV arcs on layout |
| `inspectors/live.py` | `LiveInspector`: real spawned LiDAR with live scan dots |
| `inspectors/dryrun.py` | `DryRunInspector` + `KeyboardController`: full training pipeline, no model |
| `inspectors/__init__.py` | Subpackage exports |
| `_drawing.py` | Free functions for CARLA debug geometry (dots, arcs, labels) |
| `dryrun.sh` | Shell driver: streams logs and tears down inspect containers on exit |
| `markov_analyser.py` | Offline diagnostic for the GNSS tier Markov chain. CPU-only (no CARLA). Run via `make analyse-markov [N_EPISODES=10000] [N_STEPS=1750]`; reports stationary distribution, mean dwell, and time to first contiguous good window for episodes starting in a bad tier. |

## Modes

### Layout (`--mode layout`)

Spawns the full parking lot and draws geometry overlays:

- Bay outlines coloured by type (perpendicular = blue, angled = yellow, parallel = violet)
- Target bay highlighted in bright green
- Spawn and extra-spawn points (yellow / orange)
- Pedestrian zone outlines (turquoise)
- Patrol waypoint path (red)
- Spectator positioned at `max(span * 1.1, 80 m)` height for full-lot framing

```bash
make docker-inspect                               # Default: rectangle layout
make docker-inspect INSPECT_LAYOUT=trapezoid
make docker-inspect INSPECT_LAYOUT=irregular_a
```

<!-- gif:placeholder name="inspect_layout" caption="Layout inspector showing bay outlines, patrol path, and pedestrian zones" -->
![Layout inspector placeholder](../../docs/media/inspect_layout.gif)

### Sensors (`--mode sensors`)

Spawns the ego vehicle and draws static sensor mount dots and FOV arcs:

| Sensor | Colour | FOV overlay |
|--------|--------|-------------|
| IMU | Yellow dot | None |
| GNSS | Magenta dot | None |
| 2D LiDAR | Cyan dot | 270 deg arc at 30 m radius |

Mount positions and ranges are read from `configs/deployment/sim/env_config.yaml`.

```bash
make docker-inspect-sensors                            # Birds-eye (default)
make docker-inspect-sensors SENSORS_VIEW=side          # Side profile (mount heights)
make docker-inspect-sensors SENSORS_VIEW=front         # Front profile
make docker-inspect-sensors INSPECT_ZOOM=wide          # Raise camera to show full arc
```

<!-- gif:placeholder name="inspect_sensors" caption="Sensor inspector showing GNSS, IMU, and LiDAR FOV arc from birds-eye" -->
![Sensor inspector placeholder](../../docs/media/inspect_sensors.gif)

### Live (`--mode live`)

Spawns a real 2D LiDAR on the ego vehicle and displays live scan returns as red debug dots in the CARLA world, with spectator in birds-eye view.

```bash
make docker-inspect-live                               # Default: rectangle layout
make docker-inspect-live INSPECT_LAYOUT=trapezoid
```

### Dryrun (`--mode dryrun`)

Runs the full training pipeline (env reset + step loop) with a constant forward action or keyboard control and no model. Spectator follows the ego vehicle. EKF covariance, GT pose, and EKF-vs-GT errors are printed to the console every 50 steps.

```bash
make docker-inspect-dryrun MANUAL=true                 # Keyboard control (recommended)
make docker-inspect-dryrun                             # Constant forward action
```

Keyboard controls (dryrun, `MANUAL=true`): Up = throttle, Down = brake, Left/Right = steer.

<!-- gif:placeholder name="inspect_dryrun" caption="Dryrun inspector: manual keyboard drive with EKF covariance output" -->
![Dryrun inspector placeholder](../../docs/media/inspect_dryrun.gif)

## Arguments

| Argument | Choices / type | Default | Notes |
|----------|---------------|---------|-------|
| `--mode` | `layout`, `sensors`, `live`, `dryrun` | `sensors` | Inspector mode |
| `--layout` | `rectangle`, `trapezoid`, `irregular_a` | `rectangle` | Floor plan to spawn |
| `--view` | `birds_eye`, `side`, `front` | `birds_eye` | Camera view (sensors mode only) |
| `--zoom` | `close`, `wide` | `close` | Camera height (birds-eye only) |
| `--inspect-view` | `third_person`, `side`, `back`, `front`, `free`, `birds_eye` | `third_person` | Spectator view (dryrun only) |
| `--termination-pause` | float (seconds) | `3.0` | Hold scene after episode end (dryrun only) |
| `--episodes` | int | unlimited | Max episodes (dryrun only) |
| `--duration` | int (seconds) | `86400` | Max run time (24 h) |
| `--manual` | flag | off | Keyboard control (dryrun only) |
| `--host` | string | `carla-server-demo` | CARLA server hostname |
| `--port` | int | `2100` | CARLA server port |

Make variables map directly to CLI arguments: `INSPECT_LAYOUT` -> `--layout`, `SENSORS_VIEW` -> `--view` (sensors mode only), `INSPECT_ZOOM` -> `--zoom`, `MANUAL=true` -> `--manual`.

## Requirements

All modes run inside dedicated Docker containers. The Make targets start these automatically:

- `carla-server-demo` container with a windowed CARLA session
- An X11 display (`$DISPLAY` exported to the container by the Makefile)
- For dryrun: the full three-tier stack (CARLA + ROS 2 + training containers)

## See also

- [scripts/README.md](../README.md) - scripts overview and all Make targets
- [scripts/layouts/README.md](../layouts/README.md) - floor plan reference (layout origins, bay counts)
- [uncertainty_rl/envs/README.md](../../uncertainty_rl/envs/README.md) - `CARLAParkingEnv` that dryrun mode exercises
