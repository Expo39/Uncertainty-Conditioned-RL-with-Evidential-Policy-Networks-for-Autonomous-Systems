# utils/

Shared utilities for logging, metrics tracking, visualisation, and covariance processing.

## Modules

### `constants.py`

Structural constants fixed by system architecture. **Not tuneable** -changing these requires coordinated updates across all consumers (env, networks, training, evaluation, tests). Tuneable values (timesteps, noise levels, seeds) live in `configs/` YAML files.

| Constant | Value | Description |
|----------|-------|-------------|
| `VEHICLE_STATE_DIM` | 3 | Velocity state: [vx, vy, vyaw] |
| `COVARIANCE_FEATURES_DIM` | 9 | EKF uncertainty features (std\_x, std\_y, std\_yaw + 6 covariance elements) |
| `TARGET_POSE_DIM` | 3 | Relative target bay pose: [dx, dy, dyaw] in ego body frame |
| `OBSTACLE_FEATURES_DIM` | 5 | Hemispheric clearance: [left\_dist, left\_bearing, right\_dist, right\_bearing, forward\_dist] |
| `TOTAL_OBS_DIM` | 20 | Full observation: velocity + covariance + target + clearance (3 + 9 + 3 + 5) |
| `ACTION_DIM` | 2 | [steering, longitudinal] — longitudinal in [-1,1]: positive=throttle, negative=brake |
| `SUCCESS_THRESHOLD_POSITION` | 0.5 m | Parking success position threshold |
| `SUCCESS_THRESHOLD_ORIENTATION` | ~0.175 rad | Parking success orientation threshold (10 deg) |
| `SUCCESS_THRESHOLD_VELOCITY` | 0.1 m/s | Parking success velocity threshold |
| `CLEARANCE_THRESHOLD` | 0.8 m | Reserved — CARLA collision sensor used in practice |
| `OUT_OF_BOUNDS_THRESHOLD` | 20.0 m | Distance from target above which episode terminates |
| `MAX_PARKING_SPEED` | 15.0 m/s | Maximum speed cap for parking manoeuvres |

When `include_covariance=False` and `include_obstacle_obs=False` (both flags off), the observation is 6-dim (`VEHICLE_STATE_DIM + TARGET_POSE_DIM`).

### `covariance_utils.py`

| Function | Purpose |
|----------|---------|
| `extract_2d_covariance_features` | Extract 9-element feature vector from a 3x3 or 6x6 covariance matrix: [std\_x, std\_y, std\_yaw, cov\_xx, cov\_yy, cov\_yawyaw, cov\_xy, cov\_xyaw, cov\_yyaw] |
| `validate_covariance_matrix` | Check symmetry and positive semi-definiteness |
| `get_covariance_dimension` | Returns `COVARIANCE_FEATURES_DIM` (9) |
| `make_diagonal_covariance` | Build a flat 36-element ROS covariance array from a 6-element diagonal (used by `tf_to_odom.py`) |

### `geometry.py`

| Function | Purpose |
|----------|---------|
| `zone_bbox` | Convert a pedestrian zone dict to a `(x_min, x_max, y_min, y_max)` tuple; handles explicit-extents and centre+half-extents YAML formats |
| `point_in_polygon` | Ray-casting OOB test against the actual lot boundary polygon (with `1e-12` division guard for horizontal edges) |
| `yaw_from_quaternion` | Extract yaw from a quaternion using ZYX Euler decomposition, wrapped to `[-pi, pi]` (used by `tf_to_odom.py` and EKF bridge nodes) |
| `wrap_angle_symmetric` | Wrap heading error to `(-pi, pi]` with 180-deg parking symmetry — picks the smaller of `angle` or `angle + pi` |
| `_compute_relative_target_pose` | Target bay pose in ego body frame: returns `(dx, dy, dyaw)` where dx/dy are body-frame displacements and dyaw uses `wrap_angle_symmetric` |
| `_interpolate_cone_positions` | Evenly-spaced cone positions along a closed polygon perimeter with configurable entrance gaps |

### `logging.py`

`DebugLogger` only. Zero-overhead per-step diagnostics for `CARLAParkingEnv` — all methods are no-ops when `debug=False`. When enabled, emits a structured `DEBUG` line each step and caches a compact dict for the visualiser HUD. Training metrics are handled by SB3's built-in logger.

| Method | Purpose |
|--------|---------|
| `log_step(...)` | Emit reward, pose error, yaw error, speed, covariance RMS, obstacle distance, EKF drift, LiDAR point count, and action |
| `log_reset(...)` | Emit floor plan name, target bay ID, and spawn coordinates at episode start |
| `log_actors(...)` | Emit actor counts (static vehicles, patrol NPCs, pedestrians, cones) after spawn |
| `step_debug_dict()` | Return the cached dict from the last `log_step` call (empty when `debug=False`) |

### `visualisation.py`

Matplotlib/seaborn plotting functions plus the real-time state writer used by the detachable visualiser.

| Function / Class | Purpose |
|-----------------|---------|
| `plot_uncertainty_evolution` | Epistemic and aleatoric uncertainty over training steps |
| `plot_trajectory` | Vehicle path with uncertainty ellipses overlaid |
| `plot_training_curves` | Reward, success rate, evidential loss over time |
| `VisStateWriter` | Atomically writes episode state to `outputs/vis_state.json` (tmp + `os.replace`) each step. Read by `scripts/visualise_training.py`. Zero overhead when no visualiser is attached. |

**Plot standards:**
- Style: seaborn `whitegrid`
- Resolution: 300 DPI, `bbox_inches='tight'`
- Font sizes: title 14, axis labels 12, legend 10
- Colours: blue = epistemic, red = aleatoric
- Uncertainty ellipses: 95% confidence (chi-squared = 5.991 for 2 DOF)
