# envs/

Gymnasium-compatible CARLA parking environment with SLAM uncertainty embedded in the observation space.

## Module: `carla_parking.py`

### Class: `CARLAParkingEnv`

A `gymnasium.Env` subclass that interfaces with the CARLA simulator for autonomous parking.

### State Space (15D)

| Index | Components | Description |
|-------|-----------|-------------|
| 0-2 | x, y, yaw | Position |
| 3-5 | vx, vy, vyaw | Velocity |
| 6-8 | sigma_x, sigma_y, sigma_yaw | Standard deviations from covariance diagonal |
| 9-14 | cov_xx, cov_yy, cov_yawyaw, cov_xy, cov_xyaw, cov_yyaw | Covariance matrix elements |

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

### Fallback Behaviour

Falls back to simulation mode with synthetic covariance growth if CARLA is unavailable.
