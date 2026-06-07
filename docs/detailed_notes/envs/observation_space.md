# observation_space

Extracted from `uncertainty_rl/envs/_parking_core.py`.

## 13-dimensional default observation vector

Layout (`include_covariance=True`, `include_obstacle_obs=True`):

| Index | Feature | Source | Description |
|-------|---------|--------|-------------|
| 0 | speed | EKF filtered twist | Linear speed magnitude (m/s) |
| 1 | vyaw | EKF filtered twist | Yaw rate (rad/s) |
| 2-4 | std_x, std_y, std_yaw | EKF covariance | Standard deviations from 3x3 [x, y, yaw] submatrix diagonal |
| 5-7 | dx, dy, dyaw | Target bay (relative) | Target pose in ego body frame (forward, left, heading error) |
| 8-9 | left_dist, left_bearing | LiDAR | Nearest return with bearing > +15 deg |
| 10-11 | right_dist, right_bearing | LiDAR | Nearest return with bearing < -15 deg |
| 12 | forward_dist | LiDAR | Nearest return with |bearing| <= 15 deg |

`VEHICLE_STATE_DIM=2` covers speed and vyaw (indices 0-1). Ablation dims: 13 (default),
10 (no covariance), 8 (no obstacle obs), 5 (neither). Use `compute_obs_dim()` from
`_parking_core.py` - never hardcode.

## Absolute position excluded by design

Absolute EKF position (x, y, yaw) is excluded from the observation. It accumulates
across episodes in the EKF odom frame and has no consistent signal for the policy
(episode origin shifts each reset). Navigation intent is fully encoded by (dx, dy, dyaw).
This also ensures identical inputs in simulation and on the real vehicle.

## LiDAR hemisphere sectors

Implemented in `extract_obstacle_features()` in `_parking_core.py`:

```
_SECTOR_BOUNDARY = radians(15.0)

left:    bearing  > +15 deg   (approximately left of straight-ahead)
forward: |bearing| <= 15 deg  (narrow forward cone)
right:   bearing  < -15 deg   (approximately right of straight-ahead)
```

The +-15 deg boundary is a design choice: wide enough to catch obstacles in the
manoeuvring path, narrow enough to distinguish left/right approach to a bay slot.
Self-returns (distance < 1.0 m) and the rear hemisphere (x <= 0) are discarded
to match the ~270 deg FOV of a front-bumper-mounted 2D LiDAR.

Per-sector output: (nearest_distance, bearing_to_nearest). Forward sector: distance only
(bearing is implicitly ~0 deg for a forward-pointing cone). Zero when no returns in sector.

## Covariance features

`extract_2d_covariance_features()` (in `utils/covariance_utils.py`) extracts the
COVARIANCE_FEATURES_DIM=3 element feature vector from the 3x3 [x, y, yaw] covariance
submatrix: [std_x, std_y, std_yaw] (standard deviations from diagonal variances).

Off-diagonal covariance terms were removed in the RTK-GNSS migration because they
are typically small at RTK-fixed accuracy and the 3-element vector (matching
COVARIANCE_FEATURES_DIM) is sufficient for the policy to distinguish uncertainty regimes.

## See also

- `uncertainty_rl/envs/_parking_core.py` - `build_observation`, `extract_obstacle_features`
- `uncertainty_rl/utils/covariance_utils.py` - `extract_2d_covariance_features`
- `uncertainty_rl/utils/constants.py` - `COVARIANCE_FEATURES_DIM`, `VEHICLE_STATE_DIM`, `TOTAL_OBS_DIM`
