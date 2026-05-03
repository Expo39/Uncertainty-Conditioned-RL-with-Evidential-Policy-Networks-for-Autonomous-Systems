# envs/

Gymnasium-compatible CARLA parking environment with real EKF covariance from `robot_localisation` embedded in the observation space.

## Module: `sim/carla_parking.py`

### Class: `CARLAParkingEnv`

A `gymnasium.Env` subclass that places an ego vehicle in a configurable open-space parking lot (FlatPlane generated OpenDRIVE world) and requires it to navigate to a target parking bay while accounting for localisation uncertainty from the robot\_localisation EKF.

**Requires the full Docker stack** (carla-server + ros2-bridge + training containers) for training. No standalone fallback.

### Constructor Parameters (key subset)

All parameters are optional with defaults. Full signature in `sim/carla_parking.py`.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `carla_host` | `"localhost"` | CARLA server hostname |
| `carla_port` | `2000` | CARLA server port |
| `max_steps` | `500` | Episode step limit |
| `ros2_config` | `None` | EKF file paths and timeout |
| `carla_sensors_config` | `None` | Sensor noise parameters |
| `parking_scenarios_config` | `None` | Floor plans, bay occupancy, NPC counts |
| `include_covariance` | `True` | Include EKF std devs in obs (indices 1-3) |
| `include_obstacle_obs` | `True` | Include LiDAR clearance in obs (indices 7-11) |
| `eval_mode` | `False` | When True, OOD floor plans are eligible |
| `action_repeat` | `1` | Repeat each action N simulation steps |
| `gnss_noise_profiles_path` | `None` | Path to `gnss_noise_profiles.yaml` |
| `uncertainty_std_max` | `2.0` | Denominator for uncertainty reward scaling |

### State Space (12-dimensional default)

**All 12 dimensions are available at the real-world deployment site without retraining.** CARLA ground truth is used only for reward computation - never in the observation.

Absolute EKF position (x, y, yaw) is excluded: it accumulates across episodes in the EKF odom frame and carries no consistent signal for the policy. Navigation intent is fully encoded by dx/dy/dyaw (indices 4-6).

| Index | Feature | Source | Description |
|-------|---------|--------|-------------|
| 0 | vyaw | EKF filtered pose | Yaw rate estimate (rad/s) |
| 1-3 | std_x, std_y, std_yaw | EKF covariance | Standard deviations (COVARIANCE_FEATURES_DIM=3) |
| 4-6 | dx, dy, dyaw | Target bay (relative) | Target bay pose in ego body frame |
| 7-8 | left\_dist, left\_bearing | LiDAR scan | Nearest obstacle in left hemisphere (bearing > +15 deg) |
| 9-10 | right\_dist, right\_bearing | LiDAR scan | Nearest obstacle in right hemisphere (bearing < -15 deg) |
| 11 | forward\_dist | LiDAR scan | Nearest obstacle in forward cone (bearing within +/-15 deg) |

**Ablation flags:**
- `include_covariance=False` drops indices 1-3 (3 dims less).
- `include_obstacle_obs=False` drops indices 7-11 (5 dims less).
- Obs dims: 12 (default), 9 (no covariance), 7 (no obstacle obs), 4 (neither).
- Use `compute_obs_dim()` from `envs/_parking_core.py` to compute the active dimension.

### Target Pose Computation

`(dx, dy, dyaw)` is recomputed every step from the current EKF pose and the fixed world-frame target bay coordinates selected at episode start:

```
dx   =  cos(yaw_ego) * (x_target - x_ego) + sin(yaw_ego) * (y_target - y_ego)
dy   = -sin(yaw_ego) * (x_target - x_ego) + cos(yaw_ego) * (y_target - y_ego)
dyaw = angle_wrap(yaw_target - yaw_ego)
```

### Action Space (2-dimensional)

| Action | Range | Description |
|--------|-------|-------------|
| Steering | [-1, 1] | Left/right |
| Longitudinal | [-1, 1] | Forward/brake |

### Reward Function

Potential-based shaping (Ng et al. 1999) with localisation-quality scaling:

```
progress          = (prev_distance - curr_distance) / OUT_OF_BOUNDS_THRESHOLD
uncertainty_scale = clip(max(std_x, std_y) / uncertainty_std_max, 0, 1)
reward            = progress * (1 - uncertainty_scale) - 0.01
```

Progress reward is attenuated when localisation uncertainty is high (large std_x/std_y).
When `include_covariance=False`, `uncertainty_scale` is always 0 (no attenuation).

Terminal rewards: collision = -10.0, success = +10.0.

Success: position error < 0.5 m, orientation error < 10 deg, velocity < 0.1 m/s.

### Termination Conditions

1. **Collision** (physical contact detected by collision sensor): `terminated=True`, -10.0 penalty (ego-fault only; non-ego-fault collision gives 0.0).
2. **Successful park**: `terminated=True`, +10.0 bonus.
3. **Time limit** (`max_steps` steps, default 500): `truncated=True`, no terminal reward.

---

## Parking Lot Setup

### Floor Plans

Three floor plans are defined, all placed on the FlatPlane generated OpenDRIVE world:

| Floor plan | Shape | Training / OOD |
|------------|-------|----------------|
| `rectangle` | Standard rectangular perimeter | Training (sampled uniformly) |
| `trapezoid` | Widened at one end | Training (sampled uniformly) |
| `irregular_a` | Nine-sided irregular polygon (~80x50 m) | OOD only - never sampled during training |

Floor plan geometry (corners, bay positions, spawn transform, patrol waypoints, pedestrian zones) is pre-computed offline and stored in `configs/layouts/*.yaml`. Regenerate with `make generate-layouts`.

### Bay Types (German EAR 05 / FGSV 2005)

Each floor plan has 15 bays - 5 of each type:

| Type | Width | Depth | Aisle |
|------|-------|-------|-------|
| Perpendicular (90 deg) | 2.5 m | 5.0 m | 6.0 m |
| Angled (45 deg) | 2.5 m | 5.4 m | 3.6 m |
| Parallel (forward pull-in) | 2.5 m | 8.0 m | 4.0 m |

### Dynamic Actors

- **NPC patrol vehicles** (0-3 per episode): scripted proportional controller cycling through layout waypoints.
- **Pedestrians** (0-4 per episode): random-walk via `WalkerControl`.
- **Perimeter cones**: `static.prop.constructioncone` at 2 m spacing along floor plan edges.
- **Static parked vehicles**: spawned at occupied bay positions.

### Uncertainty Sources

| Source | EKF effect |
|--------|-----------|
| RTK fix-state tier | Primary source: per-episode GNSS noise from `gnss_noise_profiles.yaml` |
| IMU noise multiplier | Higher noise -> noisier EKF prediction step |
| NPC patrol vehicles | Dynamic obstacles for LiDAR clearance obs |
| Pedestrians | Moving obstacles for LiDAR clearance obs |
| No weather | FlatPlane does not render weather effects |

---

## Key Classes

| Class | File | Purpose |
|-------|------|---------|
| `CARLAParkingEnv` | `sim/carla_parking.py` | Main Gymnasium env |
| `_CovarianceSubscriber` | `covariance_subscriber.py` | File-based EKF state reader (no DDS) |
| `SafetyWrapper` | `safety_wrapper.py` | Action modulation at eval time based on evidential uncertainty |
| `LotSpawner` | `sim/helpers/_lot_spawner.py` | Per-episode static actor spawning (cones, parked vehicles) |
| `NPCController` | `sim/helpers/_npc_controller.py` | Patrol vehicle and pedestrian lifecycle and per-step updates |
| `SensorManager` | `sim/helpers/_sensor_manager.py` | Sensor spawning, callbacks, and cleanup |

### Module-level helpers (`_parking_core.py`)

| Function | Purpose |
|----------|---------|
| `compute_obs_dim` | Active obs dimension from include_covariance / include_obstacle_obs flags |
| `build_observation` | Fill pre-allocated obs buffer from EKF state, uncertainty, and LiDAR |
| `extract_obstacle_features` | Hemispheric LiDAR clearance features (5-element buffer) |
| `load_floor_plan` | Select and cache a floor plan YAML for one episode |
| `wait_for_ekf` | Block until LiDAR and EKF data are both available |
| `calibrate_ekf_frame_offset` | Compute odom-to-world 2D rigid body transform via EKF convergence loop; returns (tx, ty, cos_r, sin_r, r) |

---

## Module: `real/`

Real-world deployment stubs. Not used during simulation training.

| File | Purpose |
|------|---------|
| `deployment_utils.py` | `RealWorldDeployment`: surveyed datum loading + actuator calibration |
| `inference_loop.py` | `RealWorldInferenceLoop`: policy inference loop stub for physical vehicle |

`RealWorldInferenceLoop.run()` mirrors the sim eval loop but calls physical sensor/actuation APIs. `SafetyWrapper.apply()` is active on every step. `RealWorldDeployment.calibrate_action()` maps normalised policy outputs to physical actuator commands via `ActuationCalibration`.
