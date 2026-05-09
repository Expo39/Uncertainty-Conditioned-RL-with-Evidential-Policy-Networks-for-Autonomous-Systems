# utils/

Shared structural constants, covariance processing, geometry helpers, debug logging, and the visualisation state writer used across the full package.

## At a glance

- `constants.py` is the single source of truth for all architectural dimensions and success thresholds - never hardcode these elsewhere
- `ACTION_DIM = 3`: steering $\in [-1, 1]$, drive $\in [-1, 1]$, brake $\in [0, 1]$
- `TOTAL_OBS_DIM = 12` (default, both ablation flags true); use `compute_obs_dim()` in `_parking_core.py` at runtime
- `VisStateWriter` streams environment state for the detachable 2D bird's-eye visualiser via atomic JSON writes
- No tuneable hyperparameters here - those live in `configs/*.yaml`

## Modules

| Module | Purpose |
|--------|---------|
| `constants.py` | Structural constants: dimensions, thresholds. Not tuneable. |
| `covariance_utils.py` | EKF covariance extraction and validation helpers |
| `geometry.py` | Coordinate transforms, polygon tests, angle wrapping |
| `logging.py` | `DebugLogger` - zero-overhead per-step diagnostics |
| `visualisation.py` | `VisStateWriter` - atomic JSON writer for the detachable visualiser |
| `actuation_calibration.py` | `ActuationCalibration` - sim/real actuator mapping |

## constants.py

Structural constants fixed by system architecture. Changing any of these requires coordinated updates across `envs/`, `networks/`, `training/`, `evaluation/`, and `tests/`. Tuneable values (timesteps, noise levels, seeds) live in `configs/*.yaml`.

| Constant | Value | Description |
|----------|-------|-------------|
| `VEHICLE_STATE_DIM` | $1$ | Yaw rate only: $[\dot\psi]$. $v_x$/$v_y$ excluded - no EKF correction source. |
| `COVARIANCE_FEATURES_DIM` | $3$ | EKF uncertainty features: $[\sigma_x, \sigma_y, \sigma_\psi]$. Off-diagonal terms dropped as redundant. |
| `TARGET_POSE_DIM` | $3$ | Relative target bay pose: $[dx, dy, d\psi]$ in ego body frame. |
| `OBSTACLE_FEATURES_DIM` | $5$ | Hemispheric clearance: $[d_\text{left}, \theta_\text{left}, d_\text{right}, \theta_\text{right}, d_\text{fwd}]$. |
| `TOTAL_OBS_DIM` | $12$ | Full observation: $1 + 3 + 3 + 5$ (both ablation flags true). |
| `ACTION_DIM` | $3$ | Steering $\in [-1, 1]$, drive $\in [-1, 1]$, brake $\in [0, 1]$. |
| `SUCCESS_THRESHOLD_POSITION` | $0.5$ m | Parking success position threshold. |
| `SUCCESS_THRESHOLD_ORIENTATION` | $\approx 0.175$ rad | Parking success orientation threshold ($10$ deg). |
| `SUCCESS_THRESHOLD_VELOCITY` | $0.1$ m/s | Parking success velocity threshold. |
| `CLEARANCE_THRESHOLD` | $0.8$ m | Reserved; CARLA collision sensor used in practice. |
| `OUT_OF_BOUNDS_THRESHOLD` | $20.0$ m | Distance from target above which episode terminates. |
| `MAX_PARKING_SPEED` | $15.0$ m/s | Speed cap for parking manoeuvres. |

Observation dimension as a function of ablation flags:

| `include_covariance` | `include_obstacle_obs` | Obs dim |
|---------------------|----------------------|---------|
| True | True | $12$ (default) |
| False | True | $9$ |
| True | False | $7$ |
| False | False | $4$ |

Use `compute_obs_dim()` from `uncertainty_rl/envs/_parking_core.py` at runtime rather than branching on these constants directly.

## covariance_utils.py

| Function | Purpose |
|----------|---------|
| `extract_2d_covariance_features` | Extract $[\sigma_x, \sigma_y, \sigma_\psi]$ from a $3\times 3$ or $6\times 6$ covariance matrix |
| `validate_covariance_matrix` | Check symmetry and positive semi-definiteness |
| `get_covariance_dimension` | Returns `COVARIANCE_FEATURES_DIM` ($3$) |
| `make_diagonal_covariance` | Build a flat 36-element ROS covariance array from a 6-element diagonal (internal helper, not re-exported) |

## geometry.py

| Function | Purpose |
|----------|---------|
| `zone_bbox` | Convert a pedestrian zone dict to $(x_\min, x_\max, y_\min, y_\max)$; handles explicit-extents and centre + half-extents YAML formats |
| `point_in_polygon` | Ray-casting out-of-bounds test against the lot boundary polygon (with $10^{-12}$ division guard for horizontal edges) |
| `yaw_from_quaternion` | Extract yaw from a quaternion using ZYX Euler decomposition, wrapped to $[-\pi, \pi]$ |
| `wrap_angle_symmetric` | Wrap heading error to $(-\pi, \pi]$ with 180-deg parking symmetry |
| `_compute_relative_target_pose` | Target bay pose in ego body frame: returns $(dx, dy, d\psi)$ |
| `_interpolate_cone_positions` | Evenly-spaced cone positions along a closed polygon perimeter with configurable entrance gaps |

## logging.py

`DebugLogger` only. Zero-overhead per-step diagnostics for `CARLAParkingEnv` - all methods are no-ops when `debug=False`. When enabled, emits a structured `DEBUG` line each step and caches a compact dict for the visualiser HUD.

| Method | Purpose |
|--------|---------|
| `log_step(...)` | Emit reward, pose error, yaw error, speed, covariance RMS, obstacle distance, EKF drift, LiDAR point count, and action |
| `log_reset(...)` | Emit floor plan name, target bay ID, and spawn coordinates at episode start |
| `log_actors(...)` | Emit actor counts (static vehicles, patrol NPCs, pedestrians, cones) after spawn |
| `step_debug_dict()` | Return the cached dict from the last `log_step` call (empty when `debug=False`) |

## visualisation.py

`VisStateWriter` only. Atomically writes the full environment state to a JSON file each step via a tmp-file + `os.replace` pattern, so the visualiser never reads a partial write.

| Method | Signature | Purpose |
|--------|-----------|---------|
| `write(...)` | `ego_transform, actor_transforms, target_bay, episode_info, trajectory, bays, corners, pedestrians` | Serialise and atomically write the visualisation state |

The output path is set at construction time. The visualiser ([scripts/visualise/](../../scripts/visualise/)) reads this file every frame.

**Plot standards** (for `evaluation/plot_evaluation_results()` and any future Matplotlib code added here): seaborn `whitegrid`, 300 DPI, `bbox_inches='tight'`, title 14pt, axis labels 12pt, legend 10pt. Colours: blue = epistemic, red = aleatoric. Uncertainty ellipses: 95% confidence ($\chi^2 = 5.991$ for 2 DOF).

## actuation_calibration.py

| Class | Purpose |
|-------|---------|
| `ActuatorMap` | Single-actuator mapping with deadband, gain, bias, and output clamp (internal; not re-exported) |
| `ActuationCalibration` | Wraps steering and drive `ActuatorMap` instances; identity in simulation, calibrated for real deployment |

Load via `ActuationCalibration.from_config(path)`. Returns an identity mapping if the calibration file is absent, so simulation and real deployment share the same code path.

## Key interfaces

```python
from uncertainty_rl.utils import (
    # Constants
    ACTION_DIM, TOTAL_OBS_DIM, VEHICLE_STATE_DIM,
    COVARIANCE_FEATURES_DIM, TARGET_POSE_DIM, OBSTACLE_FEATURES_DIM,
    SUCCESS_THRESHOLD_POSITION, SUCCESS_THRESHOLD_ORIENTATION,
    SUCCESS_THRESHOLD_VELOCITY, OUT_OF_BOUNDS_THRESHOLD,
    # Covariance
    extract_2d_covariance_features, validate_covariance_matrix,
    # Geometry
    zone_bbox, point_in_polygon, wrap_angle_symmetric,
    # Logging
    DebugLogger,
    # Visualisation
    VisStateWriter,
    # Actuation
    ActuationCalibration,
)
```

## Configuration keys consumed

`utils/` does not load YAML configs directly. Constants in `constants.py` are architectural (fixed by the system's state/action space design). Tuneable values belong in `configs/`.

## See also

- [uncertainty_rl/README.md](../README.md) - package overview
- [envs/README.md](../envs/README.md) - `CARLAParkingEnv` which consumes most of these utilities
- [networks/README.md](../networks/README.md) - evidential policy that uses `VEHICLE_STATE_DIM` and `COVARIANCE_FEATURES_DIM` for dual-encoder slicing
- [scripts/visualise/README.md](../../scripts/visualise/README.md) - the visualiser that reads `VisStateWriter` output
