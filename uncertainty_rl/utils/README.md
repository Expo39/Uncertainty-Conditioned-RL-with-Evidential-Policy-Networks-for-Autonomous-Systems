# utils/

Shared structural constants, covariance processing, geometry helpers, debug logging, and the visualisation state writer used across the full package.

## At a glance

- `constants.py` is the single source of truth for all architectural dimensions and success thresholds - never hardcode these elsewhere
- `ACTION_DIM = 3`: steering $\in [-1, 1]$, throttle $\in [0, 1]$, brake $\in [0, 1]$ (throttle and brake are separate non-negative axes; no reverse gear)
- `TOTAL_OBS_DIM = 13` (default, both ablation flags true); use `compute_obs_dim()` in `_parking_core.py` at runtime
- `VisStateWriter` streams environment state for the detachable 2D bird's-eye visualiser via atomic JSON writes
- No tuneable hyperparameters here - those live in `configs/*.yaml`

## Modules

| Module | Purpose |
|--------|---------|
| `constants.py` | Structural constants: dimensions, thresholds. Not tuneable. |
| `bay_success.py` | Per-bay episode accounting for success-rate reporting |
| `config_merge.py` | Config merge hierarchy: `deep_merge`, `apply_baseline`, `BASELINE_KEYS` |
| `covariance_utils.py` | EKF covariance extraction and validation helpers |
| `geometry.py` | Coordinate transforms, polygon tests, angle wrapping |
| `logging.py` | `DebugLogger` - zero-overhead per-step diagnostics |
| `visualisation.py` | `VisStateWriter` - atomic JSON writer for the detachable visualiser |
| `actuation_calibration.py` | `ActuationCalibration` - sim/real actuator mapping |

## constants.py

Structural constants fixed by system architecture. Changing any of these requires coordinated updates across `envs/`, `networks/`, `training/`, `evaluation/`, and `tests/`. Tuneable values (timesteps, noise levels, seeds) live in `configs/*.yaml`.

| Constant | Value | Description |
|----------|-------|-------------|
| `VEHICLE_STATE_DIM` | $2$ | Signed body-frame longitudinal speed and yaw rate: $[v_x, \dot\psi]$. |
| `COVARIANCE_FEATURES_DIM` | $3$ | EKF uncertainty features: $[\sigma_x, \sigma_y, \sigma_\psi]$. Off-diagonal terms dropped as redundant. |
| `TARGET_POSE_DIM` | $3$ | Relative target bay pose: $[dx, dy, d\psi]$ in ego body frame. |
| `OBSTACLE_FEATURES_DIM` | $5$ | Hemispheric clearance: $[d_\text{left}, \theta_\text{left}, d_\text{right}, \theta_\text{right}, d_\text{fwd}]$. |
| `TOTAL_OBS_DIM` | $13$ | Full observation: $2 + 3 + 3 + 5$ (both ablation flags true). |
| `ACTION_DIM` | $3$ | Steering $\in [-1, 1]$, throttle $\in [0, 1]$, brake $\in [0, 1]$. Throttle and brake are separate non-negative axes. No reverse gear. |
| `SUCCESS_THRESHOLD_VELOCITY` | $0.1$ m/s | Parking success velocity threshold. Combined with the geometric in-bay check to define a parked vehicle. |
| `STRICT_BAY_MARGIN` | $-0.25$ m | Strict inward bay-polygon margin (negative inflates the bay by $0.25$ m per side). The published criterion used by evaluation, the demo driver, and the lot inspector. Training/tuning instead read `bay_margin` from `env_config.yaml`, relaxed per curriculum stage. |
| `CORRIDOR_HALF_WIDTH` | $2.0$ m | Cross-track reference: the `on_line` reward factor is $1$ on the bay centreline and $0$ at this offset. |
| `ALONG_TRACK_SCALE` | $6.0$ m | Along-track reference: the `near_depth` reward factor ramps from $1$ at the parked depth to $0$ over this distance. |
| `APPROACH_INNER_ALIGNMENT_CUTOFF` | $\pi/4$ rad | Alignment-factor saturation cutoff in the corridor `aligned` term ($45$ deg). |
| `CORRIDOR_W_ALONG` / `CORRIDOR_W_CROSS` / `CORRIDOR_W_HEAD` | $1.0$ / $2.0$ / $3.0$ | Relative weights of the along-track, cross-track, and heading factors in the corridor potential. |
| `ENDGAME_MOVE_COEF` / `ENDGAME_HOLD_COEF` | $0.008$ / $0.006$ | Finisher bonus coefficients while approaching the parked pose vs holding the stop. |
| `PROGRESS_TARGET` / `PHI_NORM_FLOOR` | $39.0$ / $5.0$ | Per-episode dense-reward normaliser (`phi(start)` scale) and its soft-zero floor. |
| `TIMEOUT_POS_COEF` / `TIMEOUT_YAW_COEF` | $1.5$ / $2.5$ | Graded timeout-penalty weights on final position vs orientation error. |
| `TIMEOUT_PENALTY_FLOOR_NORM` | $-24.0$ | Minimum (most negative) normalised timeout penalty. |
| `STALL_TRUNCATION_DECISIONS` / `STALL_GATE_EKF_STD_M` | $50$ / $0.4$ m | Stall-truncation decision count and the EKF-std gate below which a stalled episode is truncated (above it, wait for localisation to recover). |
| `OBSTACLE_CLEARANCE_SAFE` / `OBSTACLE_CLEARANCE_DANGER` | $0.8$ / $0.3$ m | Clearance-penalty ramp band; SAFE sits below the $\sim 0.98$ m gap a correctly parked car leaves beside an occupied neighbour, so a correct park pays $\sim 0$. |
| `OUT_OF_BOUNDS_THRESHOLD` | $20.0$ m | Radial distance from the target above which the real-world inference loop aborts. The sim path uses the soft polygon boundary below instead. |
| `OOB_INFLATION_MARGIN` | $5.0$ m | Metres the lot polygon is offset outward (uniformly, on every edge) to form the soft out-of-bounds boundary (a run-off skirt beyond the lot edge). |
| `OOB_STEP_PENALTY` | $-0.5$ | Reward applied each policy decision the ego centre is outside the inflated polygon. |
| `OOB_TERMINATION_PENALTY_LIMIT` | $10.0$ | Accumulated out-of-bounds cost at which the episode terminates (no extra crash-magnitude penalty). |
| `OBS_*_SCALE`, `OBS_NORM_CLIP` | various | Fixed physical-range observation scales (`OBS_SPEED_SCALE`, `OBS_YAW_RATE_SCALE`, `OBS_STD_POS_SCALE`, `OBS_STD_YAW_SCALE`, `OBS_TARGET_POS_SCALE`, `OBS_TARGET_YAW_SCALE`, `OBS_OBSTACLE_DIST_SCALE`, `OBS_OBSTACLE_BEARING_SCALE`) applied in `build_observation`, with clip `OBS_NORM_CLIP`. Stage- and layout-invariant: each obs dim is divided by its physical range, so weights transfer on resume and OOD eval is unconfounded (replaces VecNormalize obs-norm). |

Success position and orientation are no longer scalar constants. The success
gate is the geometric polygon-fit check (every corner of the ego bounding
box must lie inside the target bay rectangle, via `car_fully_inside_bay()`)
combined with the velocity threshold above. Any orientation that physically
fits is accepted; the bay's rectangular geometry combined with the car's
dimensions restricts feasible orientations to small yaw errors in practice.
The margin used by the check is supplied at env construction time: training and
tuning pass the `bay_margin` resolved from `env_config.yaml` (relaxed per
curriculum stage), while the demo / inspector / evaluation pass
`STRICT_BAY_MARGIN` - the env itself has no concept of "training vs eval" mode.

Observation dimension as a function of ablation flags:

| `include_covariance` | `include_obstacle_obs` | Obs dim |
|---------------------|----------------------|---------|
| True | True | $13$ (default) |
| False | True | $10$ |
| True | False | $8$ |
| False | False | $5$ |

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
| `inflate_polygon` | Inflate a polygon outward by `margin` metres per bounding-box side (bounding-box-centre scaling). Used to build the soft out-of-bounds boundary from the lot corners |
| `car_fully_inside_bay` | Rectangle-in-rectangle containment: every corner of the ego bounding box must lie inside the bay rectangle (optional inward `margin`). Used by the geometric success gate |
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
    SUCCESS_THRESHOLD_VELOCITY,
    STRICT_BAY_MARGIN,
    CORRIDOR_HALF_WIDTH, ALONG_TRACK_SCALE, APPROACH_INNER_ALIGNMENT_CUTOFF,
    OBSTACLE_CLEARANCE_SAFE, OBSTACLE_CLEARANCE_DANGER,
    OUT_OF_BOUNDS_THRESHOLD,
    OOB_INFLATION_MARGIN, OOB_STEP_PENALTY, OOB_TERMINATION_PENALTY_LIMIT,
    # Covariance
    extract_2d_covariance_features, validate_covariance_matrix,
    # Geometry
    zone_bbox, point_in_polygon, inflate_polygon, car_fully_inside_bay,
    wrap_angle_symmetric,
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
