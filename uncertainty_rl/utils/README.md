# utils/

Shared utilities for logging, metrics tracking, visualisation, and covariance processing.

## Modules

### `constants.py`

Structural constants fixed by system architecture. **Not tuneable** - changing these requires
coordinated updates across all consumers (env, networks, training, evaluation, tests).
Tuneable values (timesteps, noise levels, seeds) live in `configs/` YAML files.

| Constant | Value | Description |
|----------|-------|-------------|
| `VEHICLE_STATE_DIM` | 1 | Velocity state: [vyaw] only. vx/vy excluded (no correction source). |
| `COVARIANCE_FEATURES_DIM` | 3 | EKF uncertainty features: [std_x, std_y, std_yaw]. Off-diagonal terms dropped as redundant. |
| `TARGET_POSE_DIM` | 3 | Relative target bay pose: [dx, dy, dyaw] in ego body frame |
| `OBSTACLE_FEATURES_DIM` | 5 | Hemispheric clearance: [left_dist, left_bearing, right_dist, right_bearing, forward_dist] |
| `TOTAL_OBS_DIM` | 12 | Full observation: vyaw + covariance + target + clearance (1+3+3+5), both flags true |
| `ACTION_DIM` | 2 | [steering, longitudinal]. Longitudinal in [-1,1]: positive=throttle, negative=brake |
| `SUCCESS_THRESHOLD_POSITION` | 0.5 m | Parking success position threshold |
| `SUCCESS_THRESHOLD_ORIENTATION` | ~0.175 rad | Parking success orientation threshold (10 deg) |
| `SUCCESS_THRESHOLD_VELOCITY` | 0.1 m/s | Parking success velocity threshold |
| `CLEARANCE_THRESHOLD` | 0.8 m | Reserved - CARLA collision sensor used in practice |
| `OUT_OF_BOUNDS_THRESHOLD` | 20.0 m | Distance from target above which episode terminates |
| `MAX_PARKING_SPEED` | 15.0 m/s | Maximum speed cap for parking manoeuvres |

When `include_covariance=False` and `include_obstacle_obs=False`, the observation is 4-dim
(`VEHICLE_STATE_DIM + TARGET_POSE_DIM`).

### `covariance_utils.py`

| Function | Purpose |
|----------|---------|
| `extract_2d_covariance_features` | Extract 3-element feature vector [std_x, std_y, std_yaw] from a 3x3 or 6x6 covariance matrix |
| `validate_covariance_matrix` | Check symmetry and positive semi-definiteness |
| `get_covariance_dimension` | Returns `COVARIANCE_FEATURES_DIM` (3) |
| `make_diagonal_covariance` | Build a flat 36-element ROS covariance array from a 6-element diagonal |

### `geometry.py`

| Function | Purpose |
|----------|---------|
| `zone_bbox` | Convert a pedestrian zone dict to a `(x_min, x_max, y_min, y_max)` tuple; handles explicit-extents and centre+half-extents YAML formats |
| `point_in_polygon` | Ray-casting OOB test against the actual lot boundary polygon (with `1e-12` division guard for horizontal edges) |
| `yaw_from_quaternion` | Extract yaw from a quaternion using ZYX Euler decomposition, wrapped to `[-pi, pi]` |
| `wrap_angle_symmetric` | Wrap heading error to `(-pi, pi]` with 180-deg parking symmetry |
| `_compute_relative_target_pose` | Target bay pose in ego body frame: returns `(dx, dy, dyaw)` |
| `_interpolate_cone_positions` | Evenly-spaced cone positions along a closed polygon perimeter with configurable entrance gaps |

### `logging.py`

`DebugLogger` only. Zero-overhead per-step diagnostics for `CARLAParkingEnv` - all methods
are no-ops when `debug=False`. When enabled, emits a structured `DEBUG` line each step and
caches a compact dict for the visualiser HUD.

| Method | Purpose |
|--------|---------|
| `log_step(...)` | Emit reward, pose error, yaw error, speed, covariance RMS, obstacle distance, EKF drift, LiDAR point count, and action |
| `log_reset(...)` | Emit floor plan name, target bay ID, and spawn coordinates at episode start |
| `log_actors(...)` | Emit actor counts (static vehicles, patrol NPCs, pedestrians, cones) after spawn |
| `step_debug_dict()` | Return the cached dict from the last `log_step` call (empty when `debug=False`) |

### `visualisation.py`

Matplotlib plotting functions plus the real-time state writer used by the detachable visualiser.

| Function / Class | Purpose |
|-----------------|---------|
| `plot_uncertainty_evolution` | Epistemic and aleatoric uncertainty over training steps |
| `plot_trajectory` | Vehicle path with uncertainty ellipses overlaid |
| `plot_training_curves` | Reward, success rate, evidential loss over time |
| `VisStateWriter` | Atomically writes episode state to `outputs/vis_state.json` each step via tmp + `os.replace` |

**Plot standards:** seaborn `whitegrid`, 300 DPI, `bbox_inches='tight'`, title 14pt, axis
labels 12pt, legend 10pt. Colours: blue = epistemic, red = aleatoric. Uncertainty ellipses:
95% confidence (chi-squared = 5.991 for 2 DOF).

### `actuation_calibration.py`

| Class | Purpose |
|-------|---------|
| `ActuatorMap` | Single-actuator mapping: `output = gain * clip(input - deadband_offset, deadband, 1.0) + bias` |
| `ActuationCalibration` | Wraps steering and longitudinal `ActuatorMap`; identity in simulation, calibrated for real deployment |

Load via `ActuationCalibration.from_config(path)`. Returns identity if the file is absent.

## See also

- `uncertainty_rl/utils/CLAUDE.md` - authoring rules for this module.
- Root `CLAUDE.md` - `constants.py` constants are listed in the coding standards section.
