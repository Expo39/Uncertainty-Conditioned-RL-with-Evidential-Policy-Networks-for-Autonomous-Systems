# envs/

Gymnasium-compatible CARLA parking environment with real EKF covariance from `robot_localisation` embedded in the observation space.

## Module: `carla_parking.py`

### Class: `CARLAParkingEnv`

A `gymnasium.Env` subclass that interfaces with the CARLA simulator for autonomous parking in a parking lot scenario. Uncertainty is produced naturally by the robot_localisation EKF processing noisy CARLA sensors under varying weather and traffic conditions.

**Requires Docker containers** (CARLA + ros2-bridge + training). No standalone fallback for training.

### Constructor Parameters

```python
CARLAParkingEnv(
    carla_host="localhost",
    carla_port=2000,
    town="Town01",
    max_steps=500,
    target_parking_spot=None,
    render_mode=None,
    ros2_config={"covariance_topic": "/ekf_uncertainty/covariance", "covariance_timeout": 10.0},
    carla_sensors_config={"imu": {...}, "gnss": {...}},
    carla_conditions_config={"weather_presets": [...], "num_vehicles": 20, ...},
)
```

### State Space (15D)

**All 15 dimensions come from the EKF** to match real-world deployment. CARLA ground truth is used only for reward computation (not observable to the agent).

| Index | Components | Source | Description |
|-------|-----------|--------|-------------|
| 0-2 | x, y, yaw | EKF filtered pose | Position estimate (noisy) |
| 3-5 | vx, vy, vyaw | EKF filtered pose | Velocity estimate (noisy) |
| 6-8 | sigma_x, sigma_y, sigma_yaw | EKF covariance | Standard deviations |
| 9-14 | cov_xx, cov_yy, cov_yawyaw, cov_xy, cov_xyaw, cov_yyaw | EKF covariance | Covariance matrix elements |

### Action Space (3D)

| Action | Range | Description |
|--------|-------|-------------|
| Steering | [-1, 1] | Left/right |
| Throttle | [0, 1] | Acceleration |
| Brake | [0, 1] | Deceleration |

### Reward Function

```
R = -distance - 0.5 * orientation_error - 0.1 * velocity + 100 * success
```

Success: position error < 0.5m, orientation error < 10 degrees, velocity < 0.1 m/s.

### Parking Lot Scenarios

Each episode randomises physical conditions that affect EKF uncertainty:

- **Weather**: ClearNoon through HardRainNoon and sunset presets
- **Fog**: Density 0-50%, distance 20-100m (degrades GNSS)
- **Traffic**: NPC vehicles (0-60) and pedestrians (0-40) in the lot
- **Sensor noise**: Configurable IMU accelerometer/gyroscope and GNSS lat/lon stddevs

### Class: `_CovarianceSubscriber`

Internal rclpy Node running in a daemon thread. Subscribes to the EKF output, caches the filtered pose (for observation indices 0-5) and the 9-element uncertainty feature vector (for indices 6-14) with thread-safe access.
