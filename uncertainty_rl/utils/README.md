# utils/

Shared utilities for logging, metrics tracking, and visualisation.

## Modules

### `logging.py`

| Class | Purpose |
|-------|---------|
| `MetricsLogger` | Logs training and evaluation metrics to CSV and JSON |
| `UncertaintyTracker` | Sliding-window tracker for monitoring uncertainty trends over time |

### `visualisation.py`

Matplotlib/seaborn plotting functions for:

- **Uncertainty evolution** - epistemic and aleatoric uncertainty over training steps
- **Trajectory plots** - vehicle path with uncertainty ellipses
- **Training curves** - reward, success rate, loss over time

### `constants.py`

Structural constants fixed by system architecture. Not tuneable - changing these requires coordinated updates across modules.

| Constant | Value | Description |
|----------|-------|-------------|
| `SUCCESS_THRESHOLD_POSITION` | 0.5 m | Parking success position threshold |
| `SUCCESS_THRESHOLD_ORIENTATION` | ~0.175 rad | Parking success orientation threshold (10 deg) |
| `SUCCESS_THRESHOLD_VELOCITY` | 0.1 m/s | Parking success velocity threshold |
| `VEHICLE_STATE_DIM` | 6 | Core state: [x, y, yaw, vx, vy, vyaw] |
| `COVARIANCE_FEATURES_DIM` | 9 | EKF localisation uncertainty features |
| `TOTAL_OBS_DIM` | 15 | Full observation dimension |
| `ACTION_DIM` | 3 | [steering, throttle, brake] |

Tuneable values (timesteps, seeds, noise levels, etc.) live in `configs/` YAML files.

### `covariance_utils.py`

| Function | Purpose |
|----------|---------|
| `extract_2d_covariance_features` | Extract 9-element feature vector from 3x3 or 6x6 covariance matrix |
| `validate_covariance_matrix` | Check symmetry and positive semi-definiteness |
| `get_covariance_dimension` | Returns 9 (feature vector length) |

### Plot Standards

- Style: seaborn `whitegrid`
- Resolution: 300 DPI
- Font sizes: title 14, axis labels 12, legend 10
- Colours: blue = epistemic, red = aleatoric
- Uncertainty ellipses: 95% confidence (chi-squared = 5.991 for 2 DOF)
- Save with `bbox_inches='tight'`
