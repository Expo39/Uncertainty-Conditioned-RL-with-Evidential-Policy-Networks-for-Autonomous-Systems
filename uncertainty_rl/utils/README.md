# utils/

Shared utilities for logging, metrics tracking, visualisation, and covariance processing.

## Modules

### `constants.py`

Structural constants fixed by system architecture. **Not tuneable** -changing these requires coordinated updates across all consumers (env, networks, training, evaluation, tests). Tuneable values (timesteps, noise levels, seeds) live in `configs/` YAML files.

| Constant | Value | Description |
|----------|-------|-------------|
| `VEHICLE_STATE_DIM` | 6 | Core EKF state: [x, y, yaw, vx, vy, vyaw] |
| `COVARIANCE_FEATURES_DIM` | 9 | EKF uncertainty features (std\_x, std\_y, std\_yaw + 6 covariance elements) |
| `TARGET_POSE_DIM` | 3 | Relative target bay pose: [dx, dy, dyaw] in ego body frame |
| `TOTAL_OBS_DIM` | 20 | Full observation: pose + covariance + target + obstacle (6 + 9 + 3 + 2) |
| `ACTION_DIM` | 3 | [steering, throttle, brake] |
| `SUCCESS_THRESHOLD_POSITION` | 0.5 m | Parking success position threshold |
| `SUCCESS_THRESHOLD_ORIENTATION` | ~0.175 rad | Parking success orientation threshold (10 deg) |
| `SUCCESS_THRESHOLD_VELOCITY` | 0.1 m/s | Parking success velocity threshold |
| `CLEARANCE_THRESHOLD` | 0.8 m | Minimum clearance to any obstacle before collision termination |
| `OUT_OF_BOUNDS_THRESHOLD` | 20.0 m | Distance from target above which episode terminates |

When `include_covariance=False` and `include_obstacle_obs=False` (both flags off), the observation is 9-dim (`VEHICLE_STATE_DIM + TARGET_POSE_DIM`).

### `covariance_utils.py`

| Function | Purpose |
|----------|---------|
| `extract_2d_covariance_features` | Extract 9-element feature vector from a 3x3 or 6x6 covariance matrix: [std\_x, std\_y, std\_yaw, cov\_xx, cov\_yy, cov\_yawyaw, cov\_xy, cov\_xyaw, cov\_yyaw] |
| `validate_covariance_matrix` | Check symmetry and positive semi-definiteness |
| `get_covariance_dimension` | Returns 9 (length of the feature vector) |

### `logging.py`

| Class | Purpose |
|-------|---------|
| `MetricsLogger` | Logs training and evaluation metrics to CSV and JSON. Handles per-episode and per-step logging. |
| `UncertaintyTracker` | Sliding-window tracker for monitoring epistemic and aleatoric uncertainty trends during training. |

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
