# configs/deployment/

Configuration shared between CARLA simulation and real-vehicle deployment, plus
subdirectories for environment-specific settings.

## Files

| File | Purpose |
|------|---------|
| `sensor_config.yaml` | Physical sensor specifications and mount positions (shared) |
| `agent_config.yaml` | Master seed, safety parameters, actuator model, ROS 2 pipeline settings, the deployed baseline pointer (shared) |
| `sim/` | CARLA-specific settings (connection, rendering, GNSS noise profiles) |
| `real/` | Real-vehicle deployment settings (datum, actuation calibration, mission) |

The observation flags `include_covariance` and `include_obstacle_obs`, and `policy_type`,
are **not** here. They are owned solely by `configs/baselines/`, which `agent_config.yaml`
points at through its `baseline` key.

## `sensor_config.yaml`

Single source of truth for the physical sensor suite. Used identically in simulation and
on the real vehicle.

| Sensor | Keys | Value | Notes |
|--------|------|-------|-------|
| IMU | `sensors.imu.mount` | `(0.0, 0.0, 0.3)` | Mount position in vehicle body frame (x, y, z), metres. Target part: VectorNav VN-100. |
| RTK-GNSS | `sensors.gnss.mount` | `(0.0, 0.0, 1.6)` | Antenna mount position. Target part: u-blox ZED-F9P. Localisation source for `/odometry/gps`. |
| 2D LiDAR | `sensors.lidar.mount` | `(2.4, 0.0, 0.5)` | Front-bumper mount. Target part: SICK TiM571. |
| 2D LiDAR | `sensors.lidar.range` | `25.0` | Maximum range in metres. |
| 2D LiDAR | `sensors.lidar.channels` | `1` | Single horizontal scan plane. |
| 2D LiDAR | `sensors.lidar.fov_deg` | `270.0` | Physical FOV of the part (rear 90 deg blocked by the car body). |
| 2D LiDAR | `sensors.lidar.frequency_hz` | `15.0` | Native scan rate in Hz. |

The LiDAR feeds obstacle clearance (obs indices 8-12) only, never the EKF.

### What is actually read

`load_env_config()` in `train_ppo.py` injects only the `mount`, `range` and `channels`
keys into the matching `carla_sensors` entry of the merged env config. The mounts are also
read by both launch files to build the static sensor TFs.

| Key | Consumed by |
|-----|-------------|
| `mount` (all three sensors) | `carla_sensors` injection; `build_sensor_tf_nodes()` in `uncertainty_rl/ros2/launch/_common.py`, via `carla_bridge.launch.py` (sim) and `real_vehicle.launch.py` (real) |
| `lidar.range`, `lidar.channels` | `carla_sensors` injection -> `SensorManager` LiDAR blueprint |
| `lidar.fov_deg`, `lidar.frequency_hz` | Nothing. They record the physical part's specification. The CARLA equivalents are `upper_fov` / `lower_fov` / `rotation_frequency` / `sensor_tick` in `sim/env_config.yaml` |

> **Note**: Mount values are the nominal design positions. Confirm them against the
> survey for the specific vehicle before a deployment run.

> **Note**: `fov_deg` is the part's physical FOV. The usable sector is narrower:
> `extract_obstacle_features()` discards the rear hemisphere (`x <= 0`) and any return
> closer than 1.0 m, so only the forward 180 deg reaches the observation.

## `agent_config.yaml`

Behaviour settings that must match the trained policy. Check these before every deployment.

| Key | Default | Notes |
|-----|---------|-------|
| `seed` | `7` | The master RNG seed, and the single source of truth for the whole experiment. It seeds training and the separate-process GNSS and IMU noise relays alike, and nothing else in the tree carries a seed. Change this one line to run a new seed. |
| `baseline` | `configs/baselines/full_method.yaml` | Names the baseline the deployed checkpoint was trained as. Its `include_covariance`, `include_obstacle_obs` and `policy_type`, being the obs shape and policy class, are read from there and must match the checkpoint. |
| `max_ego_speed_ms` | `6.0` | Hard speed cap in m/s. Throttle is cut at or above it, brake is unaffected. |
| `actuator_model.*` | see below | Per-axis slew limits and the brake-overrides-throttle threshold. Applied before the action reaches CARLA or the vehicle. |
| `safety_handoff_threshold` | `1.2` | Total uncertainty, being epistemic plus aleatoric, at or above which the vehicle full-stops and hands off. Calibrate per checkpoint against the calm-condition upper tail. |
| `ros2.*` | see below | Covariance topic and timeouts, EKF convergence timeout, and whether `/set_pose` is published at each reset. |
| `real_world_datum` | `configs/deployment/real/real_world_datum.yaml` | Surveyed datum path. Not tracked - copy the `.example` and fill it in per site. |
| `actuation_calibration` | `configs/deployment/real/actuation_calibration.yaml` | Per-actuator calibration path. |

### `actuator_model`

Applied in `CARLAParkingEnv.step()` before the command reaches CARLA, and mirrored on
the real vehicle. Units are normalised action units per policy decision (0.2 s at
`action_repeat=4`, `carla_timestep=0.05`). The rate limiter measures the delivered
command, not the commanded one, so the clamp cannot be outrun.

| Key | Value | Notes |
|-----|-------|-------|
| `steer_max_delta_per_decision` | `0.20` | Steer slew limit (about 70 deg/s at the road wheel). |
| `throttle_max_delta_per_decision` | `0.5` | Throttle slew limit (0 -> full in 0.4 s). |
| `brake_max_delta_per_decision` | `0.5` | Brake slew limit (0 -> full in 0.4 s). |
| `brake_override_throttle_threshold` | `0.3` | Throttle is zeroed when the post-clamp brake command exceeds this. |

> **Note**: the env's own fallbacks differ from these YAML values
> (`steer_max_delta_per_decision` defaults to `0.15` and
> `brake_override_throttle_threshold` to `0.1` when no `actuator_model` block is passed).
> The YAML is the operative source; the fallbacks only apply if the block is absent.

### `ros2`

| Key | Value | Notes |
|-----|-------|-------|
| `covariance_topic` | `/odometry/filtered` | Accepted for call-site compatibility only. `_CovarianceSubscriber` reads EKF state from the shared JSON file written by `CovarianceExtractorNode`, not over DDS. |
| `covariance_timeout` | `120.0` | Seconds to wait for EKF state before failing the reset. |
| `ekf_convergence_timeout` | `15.0` | Seconds to wait for EKF convergence during frame calibration. |
| `publish_initial_pose` | `true` | Publishes `/set_pose` at each episode reset. |

The real-vehicle inference loop reads the same block and additionally honours
`ros2.actuation_topic` (default `/cmd_vel`), which is not set in the shipped YAML.

`RealWorldInferenceLoop.from_config()` also looks up `model_path` (default
`checkpoints/final_model`) and `max_steps` (default `500`) in this file. Neither key is
present, so both fall back to their defaults; set `model_path` before a real run.

## Config loading chain

`load_env_config()` deep-merges the selected curriculum stage onto `sim/env_config.yaml`
first, then merges the deployment files under it, lower priority losing on conflict:

```
sensor_config.yaml  ->  agent_config.yaml  ->  sim/env_config.yaml (+ curriculum stage)
```

The merged result is passed to `CARLAParkingEnv`. This sits inside the full precedence
chain, which continues with `train_config.yaml` and the baseline overlay and is
documented in [configs/README.md](../README.md#how-configs-are-loaded).

`real/` takes no part in this chain. Those files are read directly by the real-vehicle
deployment path via the two paths named in `agent_config.yaml`.

## See also

- [configs/deployment/sim/README.md](sim/README.md) - CARLA simulation settings
- [configs/deployment/real/README.md](real/README.md) - real-vehicle deployment
- [configs/README.md](../README.md) - full configs directory map
- [uncertainty_rl/envs/real/README.md](../../uncertainty_rl/envs/real/README.md) - the deployment path that consumes these files
- [docs/detailed_notes/envs/actuator_model.md](../../docs/detailed_notes/envs/actuator_model.md) - actuator model derivation
- `docs/AntonioGaldes_Dissertation.pdf`, Appendix B - canonical account of the deployment path
