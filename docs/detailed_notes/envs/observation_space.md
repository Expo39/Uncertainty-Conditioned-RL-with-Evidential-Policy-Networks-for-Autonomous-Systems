# observation_space

Extracted from `uncertainty_rl/envs/_parking_core.py`.

Section 3.3 of the dissertation (`docs/AntonioGaldes_Dissertation.pdf`) is canonical for
the observation design: what each block contributes, why absolute position is excluded,
why only the covariance diagonal is observed, the +/-15 deg LiDAR segmentation, the 1 m
self-return and rear-hemisphere exclusions, and the fixed physical-range normalisation.

This note records the index layout and the constants it maps to, which the code needs
and the dissertation does not enumerate.

## Index layout

Default vector, `include_covariance=True` and `include_obstacle_obs=True`:

| Index | Feature | Source | Constant |
|-------|---------|--------|----------|
| 0 | speed | EKF filtered twist | `VEHICLE_STATE_DIM=2` |
| 1 | vyaw | EKF filtered twist | (with index 0) |
| 2-4 | std_x, std_y, std_yaw | EKF covariance diagonal | `COVARIANCE_FEATURES_DIM=3` |
| 5-7 | dx, dy, dyaw | Target bay in ego body frame | `TARGET_POSE_DIM=3` |
| 8-9 | left_dist, left_bearing | LiDAR left sector | `OBSTACLE_FEATURES_DIM=5` |
| 10-11 | right_dist, right_bearing | LiDAR right sector | (with indices 8-9, 12) |
| 12 | forward_dist | LiDAR forward cone | (with indices 8-11) |

Ablation dims: 13 (default), 10 (no covariance), 8 (no obstacle obs), 5 (neither).
Use `compute_obs_dim()` from `_parking_core.py`; never hardcode the width.

Per-sector output is `(nearest_distance, bearing_to_nearest)`, except the forward cone,
which carries distance only because its bearing is implicitly near zero. A sector with
no returns reports zero.

## Normalisation

`build_observation` returns a copy normalised by fixed physical ranges (`constants.py`
`OBS_*_SCALE`, clipped to `+/-OBS_NORM_CLIP`) via `normalise_observation`; the raw
buffer is retained for internal diagnostics. The scaler is stage- and layout-invariant,
so weights transfer on resume and OOD evaluation stays unconfounded. `VecNormalize`
therefore does reward normalisation only (`norm_obs=False`).

## See also

- `uncertainty_rl/envs/_parking_core.py` - `build_observation`, `extract_obstacle_features`, `compute_obs_dim`
- `uncertainty_rl/utils/covariance_utils.py` - `extract_2d_covariance_features`
- `uncertainty_rl/utils/constants.py` - dimension and scale constants
