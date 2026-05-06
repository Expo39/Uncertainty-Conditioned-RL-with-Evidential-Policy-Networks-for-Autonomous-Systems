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
| 2D LiDAR | `sensors.lidar` | Range (25 m), FOV (270 deg), frequency (15 Hz), mount position. |

> **Warning**: All mount values are unmeasured placeholders until filled in at the test site.

## `agent_config.yaml`

Behaviour settings that must match the trained policy. Check these before every deployment.

| Key | Default | Notes |
|-----|---------|-------|
| `include_covariance` | `true` | Must match the flag used at training time. |
| `include_obstacle_obs` | `true` | Must match the flag used at training time. |
| `max_ego_speed_ms` | `5.0` | Hard speed cap (m/s). |
| `safety_handoff_threshold` | `5.0` | Epistemic uncertainty above this triggers full stop. |
| `real_world_deployment` | `false` | Set `true` to activate datum-based EKF calibration on the real vehicle. |

## Config loading chain

`load_env_config()` in `train_ppo.py` merges all three files automatically (lower priority listed first):

```
sensor_config.yaml  ->  agent_config.yaml  ->  sim/env_config.yaml
                                                (env_config wins on conflict)
```

The merged result is passed directly to `CARLAParkingEnv`.

## See also

- [configs/deployment/sim/README.md](sim/README.md) - CARLA simulation settings
- [configs/deployment/real/README.md](real/README.md) - real-vehicle deployment
- [configs/README.md](../README.md) - full configs directory map
