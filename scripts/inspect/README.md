# scripts/inspect/

CARLA debug overlay scripts for visually verifying lot geometry and sensor placement
before training. Both scripts require windowed CARLA (X11 display) and the demo
CARLA server container (`carla-server-demo`).

## Scripts

### `inspect_layout.py`

Interactive layout inspector. Connects to CARLA, loads a parking lot layout YAML,
spawns the ego vehicle, and draws debug overlays:

- Bay outlines coloured by type (perpendicular=blue, angled=yellow, parallel=violet)
- Target bay highlighted in bright green with label
- Spawn and extra-spawn points (yellow / orange)
- Pedestrian zone outlines (turquoise)
- Patrol waypoint path (red)
- Spectator positioned at `max(span * 1.1, 80 m)` height for full-lot framing

```bash
make docker-inspect                            # Default: trapezoid layout
make docker-inspect INSPECT_LAYOUT=rectangle
make docker-inspect INSPECT_LAYOUT=irregular_a
```

### `inspect_sensors.py`

Sensor-placement inspector. Spawns the ego vehicle in CARLA with a selected sensor
suite and draws labelled dots at each sensor mount position plus FOV coverage arcs:

| Suite | Sensors shown | FOV overlay |
|-------|--------------|-------------|
| `suite_a` | IMU (yellow) + 2D LiDAR (cyan) | 270 deg arc at 5 m radius |
| `suite_b` | IMU (yellow) + 3D LiDAR (green) | 360 deg ring at 5 m radius |
| `suite_c` | IMU (yellow) + 3D LiDAR (green) + RGB camera (orange) | 360 deg ring + 90 deg cone |

Mount positions are read directly from `configs/train_config.yaml` (carla_sensors section).

```bash
make docker-inspect-sensors                        # Default: suite_a
make docker-inspect-sensors INSPECT_SUITE=suite_b
make docker-inspect-sensors INSPECT_SUITE=suite_c
```

## Requirements

Both scripts run inside the `training-inspect` / `training-inspect-sensors` Docker
containers. They need:
- A running `carla-server-demo` container (started automatically by the Make targets)
- An X11 display (the Make targets detect `$DISPLAY` automatically)
- The project root mounted at `/workspace` (handled by docker-compose)
