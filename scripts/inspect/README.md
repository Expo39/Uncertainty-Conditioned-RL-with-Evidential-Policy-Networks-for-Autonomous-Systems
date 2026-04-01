# scripts/inspect/

CARLA debug overlay inspector for visually verifying parking lot geometry and sensor
placement before training. Requires windowed CARLA (X11 display) and the
`carla-server-demo` container (started automatically by all Make targets).

## Entry Point

All inspector modes are driven by a single script: **`lot_inspector.py`**.
The old `inspect_layout.py` and `inspect_sensors.py` have been removed.

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

| Suite | Sensors shown | FOV overlay |
|-------|--------------|-------------|
| `suite_a` | IMU (yellow) + 2D LiDAR (cyan) | 270 deg arc at 30 m radius |
| `suite_b` | IMU (yellow) + 3D LiDAR (green) | 360 deg ring at 100 m radius |
| `suite_c` | IMU (yellow) + 3D LiDAR (green) + RGB camera (orange) | 360 deg ring + 90 deg wedge |

Mount positions and ranges are read from `configs/carla/env_config.yaml`.

```bash
make docker-inspect-sensors                            # Default: suite_a, birds-eye
make docker-inspect-sensors INSPECT_SUITE=suite_b
make docker-inspect-sensors INSPECT_SUITE=suite_c
make docker-inspect-sensors INSPECT_VIEW=side          # Side profile (mount heights)
make docker-inspect-sensors INSPECT_VIEW=front         # Front profile
make docker-inspect-sensors INSPECT_ZOOM=wide          # Raise camera to show full arc
```

### Live mode (`--mode live`)

Spawns real CARLA sensor actors on the ego vehicle and shows live output:

| Suite | Default view | What you see |
|-------|-------------|-------------|
| `suite_a` | Birds-eye | Red LiDAR point cloud debug dots |
| `suite_b` | Birds-eye | Red LiDAR point cloud debug dots (denser, 3D) |
| `suite_c` | Camera | CARLA spectator locked to camera mount (forward view) |

```bash
make docker-inspect-live                               # Default: suite_a, LiDAR dots
make docker-inspect-live INSPECT_SUITE=suite_b
make docker-inspect-live INSPECT_SUITE=suite_c         # Camera view (default for c)
make docker-inspect-live INSPECT_SUITE=suite_c INSPECT_SENSOR=lidar  # Override to LiDAR
make docker-inspect-live INSPECT_LAYOUT=trapezoid      # Different floor plan
```

## Arguments

| Argument | Choices | Default | Modes |
|----------|---------|---------|-------|
| `--mode` | `layout`, `sensors`, `live` | `sensors` | all |
| `--layout` | `rectangle`, `trapezoid`, `irregular_a` | `rectangle` | all |
| `--suite` | `suite_a`, `suite_b`, `suite_c` | `suite_a` | sensors, live |
| `--view` | `birds_eye`, `side`, `front` | `birds_eye` | sensors only |
| `--zoom` | `close`, `wide` | `close` | sensors, birds_eye only |
| `--sensor` | `lidar`, `camera` | `lidar` | live only (suite_c) |
| `--duration` | int | `300` | all |

## Requirements

All inspector modes run inside dedicated Docker containers. They need:
- A running `carla-server-demo` container (started automatically by Make targets)
- An X11 display (`$DISPLAY` detected automatically by Make targets)
- The project root mounted at `/workspace` (handled by docker-compose)
