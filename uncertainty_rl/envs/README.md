# envs/

Gymnasium-compatible CARLA parking environment with real EKF covariance from `robot_localisation` embedded in the observation space.

## Module: `carla_parking.py`

### Class: `CARLAParkingEnv`

A `gymnasium.Env` subclass that places an ego vehicle in a configurable open-space parking lot (FlatPlane generated OpenDRIVE world) and requires it to navigate to a target parking bay while accounting for localisation uncertainty from the robot\_localisation EKF.

**Requires the full Docker stack** (carla-server + ros2-bridge + training containers) for training. No standalone fallback - Docker + ROS 2 are always required.

### Constructor Parameters

```python
CARLAParkingEnv(
    carla_host="localhost",
    carla_port=2000,
    town="FlatPlane",
    max_steps=500,
    render_mode=None,
    ros2_config={"covariance_topic": "/odometry/filtered", "covariance_timeout": 10.0},
    carla_sensors_config={"imu": {...}, "lidar": {...}},  # Suite A: 2D LiDAR + IMU
    parking_scenarios_config={...},  # parking_scenarios section from carla/env_config.yaml
    include_covariance=True,
    include_obstacle_obs=True,
    sensor_suite="suite_a",
    map_load_sleep=5.0,   # seconds to wait after FlatPlane OpenDRIVE load (from config)
)
```

### State Space (20-dimensional default)

**All 20 dimensions are available at the real-world deployment site without retraining.** CARLA ground truth is used only for reward computation - never in the observation. This ensures identical inputs in simulation and on the real vehicle.

| Index | Feature | Source | Description |
|-------|---------|--------|-------------|
| 0-2 | x, y, yaw | EKF filtered pose | Position and heading estimate (noisy) |
| 3-5 | vx, vy, vyaw | EKF filtered pose | Velocity estimate (noisy) |
| 6-8 | std\_x, std\_y, std\_yaw | EKF covariance | Standard deviations (log1p-transformed) |
| 9-11 | cov\_xx, cov\_yy, cov\_yawyaw | EKF covariance | Diagonal elements (log1p-transformed) |
| 12-14 | cov\_xy, cov\_xyaw, cov\_yyaw | EKF covariance | Off-diagonal elements (log1p-transformed) |
| 15-17 | dx, dy, dyaw | Target bay (relative) | Target bay pose in ego body frame |
| 18 | nearest\_dist | LiDAR scan | Distance to nearest obstacle (m) |
| 19 | nearest\_bearing | LiDAR scan | Bearing to nearest obstacle in ego frame (rad) |

**Ablation flags:**
- `include_covariance=False` drops indices 6-14 (9 dims less).
- `include_obstacle_obs=False` drops indices 18-19 (2 dims less).
- Obs dims: 20 (default), 18 (no obstacle), 11 (no covariance), 9 (neither).

Covariance features are log1p-transformed to compress heavy tails from high-uncertainty conditions (rain, sensor noise) that would otherwise distort VecNormalize running statistics.

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
| Throttle | [0, 1] | Acceleration (forward only - no reverse) |
| Brake | [0, 1] | Deceleration |

### Reward Function

Potential-based shaping (Ng et al. 1999):

```
progress = (prev_distance - curr_distance) / OUT_OF_BOUNDS_THRESHOLD
reward   = progress - 0.01  # 0.01/step time penalty
```

Terminal rewards: collision = -10.0, success = +10.0, out-of-bounds = -5.0.

Success: position error < 0.5 m, orientation error < 10 deg, velocity < 0.1 m/s.

### Termination Conditions

1. **Collision** (physical contact detected by collision sensor): terminated, -10.0 penalty.
2. **Successful park**: terminated, +10.0 bonus.
3. **Out of bounds** (>20 m from target): terminated, -5.0 penalty.
4. **Time limit** (500 steps): truncated.

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

### Stratified Bay Sampling

Each episode: sample bay type uniformly (1/3 each), then sample one bay of that type. Target bay is always kept empty. Adjacent bays (same type, index +/-1) are always kept empty for manoeuvring clearance. All other eligible bays are independently filled with ~70% probability.

### Dynamic Actors

- **NPC patrol vehicles** (0-3 per episode): scripted proportional controller cycling through layout waypoints. No Traffic Manager - FlatPlane has no OpenDRIVE road network.
- **Pedestrians** (0-4 per episode): random-walk via `WalkerControl`. No NavMesh required.
- **Perimeter cones**: `static.prop.constructioncone` at 2 m spacing along floor plan edges. LiDAR-visible, physics-blocking, `set_simulate_physics(False)`.
- **Static parked vehicles**: spawned at occupied bay positions, `set_simulate_physics(False)`.

### Uncertainty Sources

| Source | EKF effect |
|--------|-----------|
| Bay occupancy | Fewer parked cars = sparser LiDAR features = higher covariance |
| NPC patrol vehicles | Dynamic occlusions raise covariance during passes |
| Pedestrians | Short covariance spikes |
| Weather (HardRainNoon) | Rain attenuates LiDAR returns = higher covariance |
| Floor plan geometry | Perimeter shape affects scan-match quality |
| Open traversal | Sparse returns far from perimeter = genuinely high covariance |

---

## Key Classes

| Class | Purpose |
|-------|---------|
| `CARLAParkingEnv` | Main Gymnasium env |
| `_CovarianceSubscriber` | rclpy daemon thread - subscribes to EKF covariance, caches pose + uncertainty |

### Module-level helpers

| Function | Purpose |
|----------|---------|
| `_interpolate_cone_positions` | Adaptive cone spacing along polygon edges |
| `_compute_relative_target_pose` | Body-frame transform (dx, dy, dyaw) |
