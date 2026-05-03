# scripts/inspect/

CARLA debug overlay inspector for visually verifying parking lot geometry and sensor
placement before training. Requires a windowed CARLA session (X11 display) and the
`carla-server-demo` container (started automatically by all Make targets).

## Entry Point

All inspector modes are driven by a single script: **`lot_inspector.py`**.

## File Structure

| File | Purpose |
|------|---------|
| `lot_inspector.py` | CLI entry point: argument parsing, env construction, mode dispatch |
| `inspectors/base.py` | `_Inspector` base class: CARLA connection, tick loop, spectator helpers |
| `inspectors/layout.py` | `LayoutInspector`: lot geometry overlays |
| `inspectors/sensor.py` | `SensorInspector`: sensor mount dots + FOV arcs on layout |
| `inspectors/live.py` | `LiveInspector`: real spawned LiDAR with live scan dots |
| `inspectors/dryrun.py` | `DryRunInspector` + `KeyboardController`: full training pipeline |
| `inspectors/__init__.py` | Subpackage exports |
| `_drawing.py` | Free functions for CARLA debug geometry (dots, arcs, labels) |
| `dryrun.sh` | Shell driver for `make docker-inspect-dryrun` |

## Modes

### Layout mode (`--mode layout`)

Connects to CARLA, spawns the full parking lot, and draws geometry overlays:

- Bay outlines coloured by type (perpendicular=blue, angled=yellow, parallel=violet)
- Target bay highlighted in bright green
- Spawn and extra-spawn points (yellow / orange)
- Pedestrian zone outlines (turquoise)
- Patrol waypoint path (red)
- Spectator positioned at `max(span * 1.1, 80 m)` height for full-lot framing

```bash
make docker-inspect                              # Default: trapezoid layout
make docker-inspect INSPECT_LAYOUT=rectangle
make docker-inspect INSPECT_LAYOUT=irregular_a
```

### Sensors mode (`--mode sensors`)

Spawns the ego vehicle and draws static sensor mount dots and FOV arcs:

| Sensor | Colour | FOV overlay |
|--------|--------|-------------|
| IMU | Yellow dot | None |
| GNSS | Magenta dot | None |
| 2D LiDAR | Cyan dot | 270 deg arc at 30 m radius |

Mount positions and ranges are read from `configs/deployment/sim/env_config.yaml`.

```bash
make docker-inspect-sensors                            # Default: birds-eye
make docker-inspect-sensors INSPECT_VIEW=side          # Side profile (mount heights)
make docker-inspect-sensors INSPECT_VIEW=front         # Front profile
make docker-inspect-sensors INSPECT_ZOOM=wide          # Raise camera to show full arc
```

### Live mode (`--mode live`)

Spawns a real 2D LiDAR on the ego vehicle and shows live scan output as red debug
dots in the CARLA world, with spectator in birds-eye view.

```bash
make docker-inspect-live                               # Default: LiDAR dots
make docker-inspect-live INSPECT_LAYOUT=trapezoid      # Different floor plan
```

### Dryrun mode (`--mode dryrun`)

Runs the full training pipeline (env reset + step loop) with a constant forward
action or keyboard control and no model. Spectator follows the ego vehicle.
EKF covariance, GT pose, and EKF vs GT errors are printed to the console every
50 steps.

```bash
make docker-inspect-dryrun MANUAL=true                 # Keyboard control
make docker-inspect-dryrun                             # Constant forward action
```

## Arguments

| Argument | Choices | Default | Modes |
|----------|---------|---------|-------|
| `--mode` | `layout`, `sensors`, `live`, `dryrun` | `sensors` | all |
| `--layout` | `rectangle`, `trapezoid`, `irregular_a` | `rectangle` | all |
| `--view` | `birds_eye`, `side`, `front` | `birds_eye` | sensors only |
| `--zoom` | `close`, `wide` | `close` | sensors, birds_eye only |
| `--duration` | int (seconds) | `86400` (24 h) | all |
| `--episodes` | int | unlimited | dryrun only |
| `--inspect-view` | `third_person`, `side`, `back`, `front`, `free` | `third_person` | dryrun only |
| `--termination-pause` | float (seconds) | `3.0` | dryrun only |
| `--manual` | flag | off | dryrun only |

## Requirements

All inspector modes run inside dedicated Docker containers. They need:
- A running `carla-server-demo` container (started automatically by Make targets)
- An X11 display (`$DISPLAY` detected automatically by Make targets)
