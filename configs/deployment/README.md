# configs/deployment/

Configuration shared between CARLA simulation and real-vehicle deployment, plus
subdirectories for environment-specific settings.

## Files

| File | Purpose |
|------|---------|
| `sensor_config.yaml` | Physical sensor specifications and mount positions (shared) |
| `agent_config.yaml` | Observation flags, safety parameters, ROS 2 pipeline settings (shared) |
| `sim/` | CARLA-specific settings (connection, rendering, GNSS noise profiles) |
| `real/` | Real-vehicle deployment settings (datum, actuation calibration, mission) |

## `sensor_config.yaml`

Single source of truth for the physical sensor suite. Used identically in simulation and
on the real vehicle.

| Sensor | Key prefix | Notes |
|--------|-----------|-------|
| IMU | `sensors.imu.mount` | Mount position in vehicle body frame (x, y, z). |
| RTK-GNSS | `sensors.gnss.mount` | Antenna mount position. |
| 2D LiDAR | `sensors.lidar` | Range (25 m), FOV (270 deg), frequency (15 Hz), a single channel, and mount position. |

> **Warning**: All mount values are unmeasured placeholders until filled in at the test site.

## `agent_config.yaml`

Behaviour settings that must match the trained policy. Check these before every deployment.

| Key | Default | Notes |
|-----|---------|-------|
| `seed` | `7` | The master RNG seed, and the single source of truth for the whole experiment. It seeds training and the separate-process GNSS and IMU noise relays alike, and nothing else in the tree carries a seed. Change this one line to run a new seed. |
| `baseline` | `configs/baselines/full_method.yaml` | Names the baseline the deployed checkpoint was trained as. Its `include_covariance`, `include_obstacle_obs` and `policy_type`, being the obs shape and policy class, are read from there and must match the checkpoint. |
| `max_ego_speed_ms` | `6.0` | Hard speed cap in m/s. Throttle is cut at or above it, brake is unaffected. |
| `actuator_model.*` | see YAML | Per-axis slew limits in normalised action units per decision, plus the brake-overrides-throttle threshold. Applied before the action reaches CARLA or the vehicle. |
| `safety_handoff_threshold` | `1.2` | Total uncertainty, being epistemic plus aleatoric, at or above which the vehicle full-stops and hands off. Calibrate per checkpoint against the calm-condition upper tail. |
| `ros2.*` | see YAML | Covariance topic and timeouts, EKF convergence timeout, and whether `/set_pose` is published at each reset. |
| `real_world_datum` / `actuation_calibration` | paths | Real-vehicle datum and calibration files loaded by the deployment path. |

## Config loading chain

`load_env_config()` merges the three deployment files and then the selected curriculum
stage, lower priority first:

```
sensor_config.yaml  ->  agent_config.yaml  ->  sim/env_config.yaml  ->  curriculum stage
```

The merged result is passed to `CARLAParkingEnv`. This sits inside the full precedence
chain, which continues with `train_config.yaml` and the baseline overlay and is
documented in [configs/README.md](../README.md#how-configs-are-loaded).

## See also

- [configs/deployment/sim/README.md](sim/README.md) - CARLA simulation settings
- [configs/deployment/real/README.md](real/README.md) - real-vehicle deployment
- [configs/README.md](../README.md) - full configs directory map
