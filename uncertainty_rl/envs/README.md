# envs/

Gymnasium-compatible CARLA parking environment with real EKF covariance from `robot_localisation` embedded in the observation space.

## Module: `carla_parking.py`

### Class: `CARLAParkingEnv`

A `gymnasium.Env` subclass that places an ego vehicle in a configurable open-space parking lot (Town05\_Opt) and requires it to navigate to a target parking bay while accounting for localisation uncertainty from the robot\_localisation EKF.

**Requires the full Docker stack** (carla-server + ros2-bridge + training containers) for training. No standalone fallback -Docker + ROS 2 are always required.

### Constructor Parameters

```python
CARLAParkingEnv(
    carla_host="localhost",
    carla_port=2000,
    town="Town05_Opt",
    max_steps=500,
    render_mode=None,
    ros2_config={"covariance_topic": "/ekf_uncertainty/covariance", "covariance_timeout": 10.0},
    carla_sensors_config={"imu": {...}},  # Suite A: IMU only (no GNSS)
    carla_conditions_config={"weather_presets": [...], "fog_density_range": [...], ...},
    parking_config={...},   # parking_scenarios section from train_config.yaml
    include_covariance=True,
)
```

### State Space (18-dimensional)

**All 18 dimensions are available at the Lemonworx deployment site without retraining.** CARLA ground truth is used only for reward computation -never in the observation. This ensures identical inputs in simulation and on the real vehicle.

| Index | Feature | Source | Description |
|-------|---------|--------|-------------|
| 0-2 | x, y, yaw | EKF filtered pose | Position and heading estimate (noisy) |
| 3-5 | vx, vy, vyaw | EKF filtered pose | Velocity estimate (noisy) |
| 6-8 | std\_x, std\_y, std\_yaw | EKF covariance | Standard deviations from covariance diagonal |
| 9-11 | cov\_xx, cov\_yy, cov\_yawyaw | EKF covariance | Diagonal covariance elements |
| 12-14 | cov\_xy, cov\_xyaw, cov\_yyaw | EKF covariance | Off-diagonal covariance elements |
| 15-17 | dx, dy, dyaw | Target bay (relative) | Target bay pose in ego body frame |

**Ablation flag:** `include_covariance=False` drops indices 6-14, giving a 9-dim observation (6-dim EKF pose + 3-dim target). Used by vanilla\_ppo and output\_uncertainty baselines.

### Target Pose Computation

`(dx, dy, dyaw)` is recomputed every step from the current EKF pose and the fixed world-frame target bay coordinates selected at episode start:

```
dx   =  cos(yaw_ego) * (x_target - x_ego) + sin(yaw_ego) * (y_target - y_ego)
dy   = -sin(yaw_ego) * (x_target - x_ego) + cos(yaw_ego) * (y_target - y_ego)
dyaw = angle_wrap(yaw_target - yaw_ego)
```

### Action Space (3-dimensional)

| Action | Range | Description |
|--------|-------|-------------|
| Steering | [-1, 1] | Left/right |
| Throttle | [0, 1] | Acceleration (forward only -no reverse) |
| Brake | [0, 1] | Deceleration |

### Reward Function

```
R = -distance - 0.5 * orientation_error - 0.1 * velocity + 100 * success
```

Success: position error < 0.5 m, orientation error < 10 deg, velocity < 0.1 m/s.

See `@todo(AG)` comments in `train_config.yaml` for the planned switch to potential-based shaping (Task 9).

### Termination Conditions

1. **Clearance violation** (<0.8 m to any obstacle): terminated, collision penalty applied.
2. **Successful park**: terminated, +100 bonus applied.
3. **Out of bounds** (>20 m from target): terminated.
4. **Time limit** (500 steps): truncated.

---

## Parking Lot Setup

### Floor Plans

Three floor plans are defined, all placed on Town05\_Opt at episode startup (no per-episode map reload):

| Floor plan | Shape | Training / OOD |
|------------|-------|----------------|
| `rectangle` | Standard rectangular perimeter | Training (sampled uniformly) |
| `trapezoid` | Widened at one end | Training (sampled uniformly) |
| `irregular_a` | Nine-sided irregular polygon (~80x50 m) | OOD only -never sampled during training |

Floor plan geometry (corners, bay positions, spawn transform, patrol waypoints, pedestrian zones) is pre-computed offline and stored in `configs/layouts/*.yaml`. Regenerate with `make generate-layouts`.

### Bay Types (German EAR 05 / FGSV 2005)

Each floor plan has 15 bays -5 of each type:

| Type | Width | Depth | Aisle |
|------|-------|-------|-------|
| Perpendicular (90 deg) | 2.5 m | 5.0 m | 6.0 m |
| Angled (45 deg) | 2.5 m | 5.4 m | 3.6 m |
| Parallel (forward pull-in) | 2.5 m | 8.0 m | 4.0 m |

### Stratified Bay Sampling

Each episode: sample bay type uniformly (1/3 each), then sample one bay of that type. Target bay is always kept empty. Parallel-bay neighbours are always empty (`always_empty: true` in the layout YAML). All other eligible bays are independently filled with ~70% probability.

### Dynamic Actors

- **NPC patrol vehicles** (0-3 per episode): scripted proportional controller cycling through layout waypoints. No Traffic Manager -Town05\_Opt has no OpenDRIVE road network in the parking area.
- **Pedestrians** (0-4 per episode): random-walk via `WalkerControl`. No NavMesh required.
- **Perimeter cones**: `static.prop.trafficcone01` at 2 m spacing along floor plan edges. LiDAR-visible, physics-blocking, `set_simulate_physics(False)`.
- **Static parked vehicles**: spawned at occupied bay positions, `set_simulate_physics(False)`.

### Uncertainty Sources

| Source | EKF effect |
|--------|-----------|
| Bay occupancy | Fewer parked cars = sparser LiDAR features = higher covariance |
| NPC patrol vehicles | Dynamic occlusions raise covariance during passes |
| Pedestrians | Short covariance spikes |
| Weather (rain, cloud) | Sensor degradation raises covariance |
| Fog | Attenuates LiDAR returns at distance |
| Floor plan geometry | Perimeter shape affects scan-match quality |
| Open traversal | Sparse returns far from perimeter = genuinely high covariance |

---

## Key Classes

| Class | Purpose |
|-------|---------|
| `CARLAParkingEnv` | Main Gymnasium env |
| `_CovarianceSubscriber` | rclpy daemon thread -subscribes to EKF covariance, caches pose + uncertainty |
| `VisStateWriter` | Atomic JSON writer for detachable 2D visualiser (`scripts/visualise_training.py`) |

### Module-level helpers

| Function | Purpose |
|----------|---------|
| `_interpolate_cone_positions` | Adaptive cone spacing along polygon edges |
| `_compute_relative_target_pose` | Body-frame transform (dx, dy, dyaw) |
