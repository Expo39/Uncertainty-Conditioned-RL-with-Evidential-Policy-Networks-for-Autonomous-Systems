# scripts/inspect/

CARLA debug overlay inspector for visually verifying parking lot geometry and sensor
placement before training. Requires windowed CARLA (X11 display) and the
`carla-server-demo` container (started automatically by all Make targets).

## Entry Point

All inspector modes are driven by a single script: **`lot_inspector.py`**.

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

## Arguments

| Argument | Choices | Default | Modes |
|----------|---------|---------|-------|
| `--mode` | `layout`, `sensors`, `live`, `dryrun` | `sensors` | all |
| `--layout` | `rectangle`, `trapezoid`, `irregular_a` | `rectangle` | all |
| `--view` | `birds_eye`, `side`, `front` | `birds_eye` | sensors only |
| `--zoom` | `close`, `wide` | `close` | sensors, birds_eye only |
| `--duration` | int | `300` | all |

## Requirements

All inspector modes run inside dedicated Docker containers. They need:
- A running `carla-server-demo` container (started automatically by Make targets)
- An X11 display (`$DISPLAY` detected automatically by Make targets)
- The project root mounted at `/workspace` (handled by docker-compose)
