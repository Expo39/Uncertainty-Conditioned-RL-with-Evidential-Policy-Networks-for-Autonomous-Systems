# scripts/inspect/

CARLA debug overlay inspector for visually verifying parking lot geometry, sensor placement and the env wiring before a training or evaluation run. All five modes are driven by a single entry point: `lot_inspector.py`.

Requires a windowed CARLA session (X11 display) and the `carla-server-demo` container; the dryrun modes additionally need the `ros2-bridge-inspect` EKF. All are started automatically by the Make targets below.

## Quick reference

| Task | Command |
|------|---------|
| Lot geometry overlay | `make docker-inspect INSPECT_LAYOUT=rectangle` |
| Sensor placement overlay | `make docker-inspect-sensors INSPECT_LAYOUT=rectangle` |
| Sensor side / front profile | `make docker-inspect-sensors INSPECT_LAYOUT=rectangle SENSORS_VIEW=side` |
| Full FOV arc (wide camera) | `make docker-inspect-sensors INSPECT_LAYOUT=rectangle INSPECT_ZOOM=wide` |
| Live LiDAR scan dots | `make docker-inspect-live INSPECT_LAYOUT=rectangle` |
| Dryrun - keyboard control | `make docker-inspect-dryrun STAGE=1 BASELINE=full_method MANUAL=true` |
| Dryrun - constant action | `make docker-inspect-dryrun STAGE=1 BASELINE=full_method` |
| Eval dryrun - drive a condition | `make docker-inspect-eval-dryrun SCENARIO=anchor_deployment BASELINE=full_method MANUAL=true` |

`INSPECT_LAYOUT` has no Makefile default: omit it and the compose service falls back
to its own default (`trapezoid` for `docker-inspect`, `rectangle` for
`docker-inspect-sensors` / `docker-inspect-live`). Pass it explicitly.

## Files

| File | Purpose |
|------|---------|
| `__init__.py` | Package marker naming `lot_inspector.py` as the entry point |
| `lot_inspector.py` | CLI entry point: argument parsing, env construction, mode dispatch |
| `inspectors/base.py` | `_Inspector` base class: CARLA connection, tick loop, spectator helpers |
| `inspectors/layout.py` | `LayoutInspector`: lot geometry overlays (bays, patrol path, zones) |
| `inspectors/sensor.py` | `SensorInspector`: sensor mount dots and FOV arcs on layout |
| `inspectors/live.py` | `LiveInspector`: real spawned LiDAR with live scan dots |
| `inspectors/dryrun.py` | `DryRunInspector` + `KeyboardController`: full training pipeline, no model |
| `inspectors/__init__.py` | Subpackage exports |
| `_drawing.py` | Free functions for CARLA debug geometry (dots, arcs, labels) |
| `dryrun.sh` | Shell driver: streams logs and tears down inspect containers on exit |

## Modes

### Layout (`--mode layout`)

Spawns the full parking lot and draws geometry overlays:

- Bay outlines coloured by type. The shipped layouts contain only perpendicular (blue)
  and motorcycle (grey) bays; the angled (yellow) and parallel (violet) colours exist in
  `scripts/colours/` for completeness but no layout uses them
- Target bay highlighted in bright green, with a `TARGET` label; every other bay is
  labelled with its index
- Lot perimeter outline (light grey) and the soft out-of-bounds boundary, the perimeter
  inflated by the env's OOB margin (hot pink)
- Pedestrian zone outlines (turquoise), drawn only when pedestrians are enabled
- Patrol waypoint path (red), drawn only when patrol NPCs are enabled
- Spectator positioned at `max(span * 1.1, 80 m)` height for full-lot framing

Spawn-point markers exist in the drawing helper but are off by default
(`show_spawns=False`), because `use_extra_spawns` is off in every stage and the markers
would advertise start positions the episode never takes.

```bash
make docker-inspect INSPECT_LAYOUT=rectangle
make docker-inspect INSPECT_LAYOUT=trapezoid
make docker-inspect INSPECT_LAYOUT=irregular_a
```

### Sensors (`--mode sensors`)

Spawns the ego vehicle and draws static sensor mount dots and FOV arcs:

| Sensor | Colour | FOV overlay |
|--------|--------|-------------|
| IMU | Yellow dot | None |
| GNSS | Magenta dot | None |
| 2D LiDAR | Cyan dot | Red 270 deg arc, plus a dark-grey arc over the 90 deg rear blind sector |

The FOV arcs are drawn in birds-eye view only. In the side and front profiles the arcs
are replaced by white vertical drop lines from each mount to the ground, so mount heights
can be read off directly, and the lot geometry is suppressed so only the vehicle and its
mounts are visible.

Mount positions and the LiDAR range come from `configs/deployment/sensor_config.yaml`
(range 25 m, FOV 270 deg), which `load_env_config()` injects into the `carla_sensors`
block of `configs/deployment/sim/env_config.yaml`; the env config itself holds only the
CARLA-only spawn keys (`points_per_second`, `sensor_tick`, `upper_fov` / `lower_fov`,
`noise`).

```bash
make docker-inspect-sensors INSPECT_LAYOUT=rectangle SENSORS_VIEW=birds_eye INSPECT_ZOOM=close
make docker-inspect-sensors INSPECT_LAYOUT=rectangle SENSORS_VIEW=side    # Side profile (mount heights)
make docker-inspect-sensors INSPECT_LAYOUT=rectangle SENSORS_VIEW=front   # Front profile
make docker-inspect-sensors INSPECT_LAYOUT=rectangle INSPECT_ZOOM=wide    # Raise camera to show full arc
```

Birds-eye view of the sensor mounts and LiDAR FOV arc on the ego vehicle:

<p align="center">
  <img src="../../docs/media/inspect_sensors.jpeg" alt="Sensor inspector showing GNSS, IMU, and LiDAR FOV arc from birds-eye" width="620">
</p>

### Live (`--mode live`)

Spawns a real 2D LiDAR on the ego vehicle and displays live scan returns as red debug dots in the CARLA world. The lot geometry overlay is redrawn on the same cadence as the other modes so the scan can be read against the bays, and the birds-eye spectator (80 m) follows the vehicle each tick. The spawned sensor is destroyed on exit regardless of how the loop ends.

```bash
make docker-inspect-live INSPECT_LAYOUT=rectangle
make docker-inspect-live INSPECT_LAYOUT=trapezoid
```

`docker-inspect-live` also accepts `INSPECT_SENSOR` (default `lidar`), which is passed to
the compose service as `SENSOR`. The inspector has no matching CLI flag - the 2D LiDAR is
the only sensor `LiveInspector` spawns - so the variable currently has no effect.

### Dryrun (`--mode dryrun`)

Runs the full training pipeline (env reset + step loop) with a constant action or keyboard
control and no model. The env is resolved through the same merge order as
`train_ppo.main()` - stage-merged env config, then the baseline overlay - so the dryrun
matches what `make docker-train STAGE=.. BASELINE=..` would build, including the stage's
`bay_margin`. Two inspection-only overrides are applied: `no_rendering_mode=False`
(windowed CARLA) and `action_repeat=1`, so the camera, keyboard and console readout run
per tick rather than at the policy's decision rate.

The spectator follows the ego vehicle. Every 50 env steps the console prints the
normalised model inputs with their physical equivalents in brackets (white), CARLA ground
truth (yellow), the EKF pose in both odom and world frames plus the EKF-vs-GT speed and
yaw-rate deltas (red), and the live GNSS tier name (cyan). The covariance and obstacle
rows are printed only when the active baseline includes those obs blocks, and the
per-episode summary reports the end reason together with the EKF position and yaw RMSE
against ground truth.

The constant action is read from `inspect.dryrun_action` in
`configs/deployment/sim/env_config.yaml` (`[0.0, 0.5, 0.0]`, i.e. straight ahead at half
throttle); with no such key the inspector falls back to `action_space.sample()`.

```bash
make docker-inspect-dryrun STAGE=1 BASELINE=full_method MANUAL=true INSPECT_EPISODES=5 INSPECT_VIEW=third_person INSPECT_PAUSE=3.0 INSPECT_OOD=false
make docker-inspect-dryrun STAGE=1 BASELINE=full_method INSPECT_VIEW=birds_eye   # Constant action
```

`INSPECT_OOD=true` is an inspection-only override that swaps the stage's floor plans for
the OOD-flagged set in `env_config.yaml`, so held-out layouts can be eyeballed. It fails
loudly if no OOD plans are defined, and is not available in eval dryrun mode.

Keyboard controls (`MANUAL=true`) are latched, not held: each press steps the value.
Up = more throttle / release brake, Down = more brake / release throttle (0.1 per press
on a single bipolar pedal - there is no reverse gear), Left/Right = steer by 0.1,
Space = full stop (zero steer and throttle, full brake), Ctrl+C = quit. The controller
reads raw escape sequences from stdin, so it needs an interactive TTY; without one it
prints a warning and disables itself.

### Eval dryrun (`--mode eval_dryrun`)

Same step loop and console readout as dryrun, but the env is built from a named
`eval_config.yaml` condition through `build_eval_env_factory`, the SAME path the eval
sweep uses. No checkpoint is loaded - you drive with the keyboard (or a constant action)
so you can verify the scenario wiring (scaled sensor noise, held GNSS tier, pinned
occupancy and floor plan) against the printed inputs / GT / EKF before running the
headless sweep. Unlike dryrun, the env config is merged **unstaged** (the sweep is
stage-less), so `STAGE` has no effect here. The bare env is used, without the
`SafetyWrapper`: manual driving has no policy epistemic signal to gate on.

```bash
make docker-inspect-eval-dryrun SCENARIO=anchor_deployment BASELINE=full_method MANUAL=true       # Deployment anchor
make docker-inspect-eval-dryrun SCENARIO=gnss_degraded BASELINE=full_method MANUAL=true           # Held worst GNSS tier
make docker-inspect-eval-dryrun SCENARIO=ood_irregular_rtk_fixed BASELINE=full_method MANUAL=true # Held-out floor plan
```

The `SCENARIO` value is any condition `name` in `configs/eval_config.yaml` - currently
`anchor_deployment`, `anchor_empty`, `gnss_fixed`, `gnss_degraded`, `lidar_degraded`,
`ood_irregular_rtk_fixed` and `gnss_degrade_one_way`. An unknown name exits with the list
of valid ones. The Makefile defaults `SCENARIO` to `anchor_deployment`; omitting it at the
CLI level entirely would make the inspector fall back to the first condition in the file.
Keyboard controls are the same as dryrun.

## Arguments

| Argument | Choices / type | Default | Notes |
|----------|---------------|---------|-------|
| `--mode` | `layout`, `sensors`, `live`, `dryrun`, `eval_dryrun` | `sensors` | Inspector mode |
| `--layout` | `rectangle`, `trapezoid`, `irregular_a` | `rectangle` | Floor plan to spawn |
| `--view` | `birds_eye`, `side`, `front` | `birds_eye` | Camera view (sensors mode only) |
| `--zoom` | `close`, `wide` | `close` | Camera height (birds-eye only) |
| `--inspect-view` | `third_person`, `side`, `back`, `front`, `free`, `birds_eye` | `third_person` | Spectator view (dryrun / eval_dryrun). `free` places the spectator once and leaves it, so CARLA's own controls can fly it |
| `--termination-pause` | float (seconds) | `3.0` | Hold scene after episode end; `0` disables (dryrun / eval_dryrun) |
| `--episodes` | int | `None` (unlimited) | Max episodes (dryrun / eval_dryrun) |
| `--duration` | int (seconds) | `86400` | Max run time (24 h). The compose services for layout / sensors / live override this to `300` |
| `--manual` | flag | off | Keyboard control (dryrun / eval_dryrun) |
| `--stage` | int | `None` -> `DEFAULT_STAGE` (1) | Curriculum stage. Used to load the env config in every mode, but only dryrun builds its env from it; eval_dryrun re-merges unstaged |
| `--baseline` | path | `None` -> `configs/baselines/full_method.yaml` | Baseline obs flags / policy_type (dryrun / eval_dryrun) |
| `--scenario` | string | `None` -> first condition in the file | `eval_config.yaml` condition name (eval_dryrun only) |
| `--eval-config` | path | `configs/eval_config.yaml` | Condition sweep file (eval_dryrun only) |
| `--host` | string | `carla-server-demo` | CARLA server hostname |
| `--port` | int | `2100` | CARLA server port |

Make variables map onto CLI arguments as follows: `INSPECT_LAYOUT` to `--layout` (layout /
sensors / live; the dryrun modes randomise or pin the plan themselves), `SENSORS_VIEW` to
`--view` (sensors mode only), `INSPECT_ZOOM` to `--zoom`, `INSPECT_VIEW` to
`--inspect-view`, `INSPECT_PAUSE` to `--termination-pause`, `INSPECT_EPISODES` to
`--episodes`, `MANUAL=true` to `--manual`, `STAGE` to `--stage` (dryrun only), `BASELINE`
to `--baseline`, and `SCENARIO` to `--scenario` (eval_dryrun only). `BASELINE` is a bare
name: the Makefile expands it to `configs/baselines/<name>.yaml` before passing it on.
`INSPECT_OOD` is read from the environment by the inspector rather than mapped to a flag.

## Requirements

All modes run inside dedicated Docker containers defined in `docker-compose.inspect.yml`,
one Compose profile per mode. The Make targets start these automatically, after tearing
down the core stack so the two do not contend for ports or the GPU:

- `carla-server-demo`: windowed CARLA on port 2100 (the headless training server uses 2000)
- An X11 display (`$DISPLAY` resolved by the Makefile and exported to the container,
  with `xhost +local:docker` around the run)
- For dryrun / eval_dryrun: additionally `ros2-bridge-inspect`, which runs the
  `robot_localization` EKF against the demo server on its own DDS domain
  (`INSPECT_ROS_DOMAIN_ID`, default 43) so it cannot be confused with worker bridges
  left on domain 42

Because `uncertainty_rl/ros2/` is COPYed into the bridge image, edits there need
`make docker-build-no-cache-inspect` (or the full `make docker-build-no-cache`) before the
dryrun modes pick them up. Everything under `scripts/` and `configs/` is bind-mounted and
hot-reloads.

Related targets: `make docker-inspect-down` stops every inspect profile,
`make docker-inspect-dryrun-logs` and `make docker-logs-ros2-inspect` follow the dryrun
and bridge logs from a second terminal.

## See also

- [scripts/README.md](../README.md) - scripts overview and all Make targets
- [scripts/layouts/README.md](../layouts/README.md) - floor plan reference (layout origins, bay counts)
- [uncertainty_rl/envs/README.md](../../uncertainty_rl/envs/README.md) - `CARLAParkingEnv` that dryrun mode exercises
