# Copilot Instructions

This is the short Copilot brief. The root `CLAUDE.md` is the full source of truth for
standards (British English, Doxygen docstrings, type hints, ASCII-only, no hardcoded
hyperparameters, Docker workflow). This file adds only the per-module facts and
contracts that matter when generating code. When the two disagree, `CLAUDE.md` wins.

## What This App Is

An autonomous parking system that knows when it does not know where it is and drives
more carefully in response. EKF localisation uncertainty (covariance) is fed into an RL
policy, and the policy uses evidential deep learning to quantify its own action
uncertainty. Two layers: "how sure am I about where I am?" (EKF covariance) and "how
sure am I about what to do?" (evidential policy output). MSc dissertation codebase -
trains in CARLA, evaluates across uncertainty levels, designed to transfer to a real
instrumented parking lot.

Three containers (see `docker-compose.yml`): **carla-server** (CARLA 0.9.16 headless),
**ros2-bridge** (ROS 2 Jazzy, `robot_localization` EKF + CARLA bridge), **training**
(NVIDIA NGC PyTorch, ROS 2 Humble, subscribes to EKF covariance via DDS). Code is
bind-mounted for hot-reload. Training requires the full stack - there is no standalone
mode.

## Core standards (full detail in root CLAUDE.md)

- British English everywhere (`localisation`, `behaviour`, `normalise_observations`).
- Doxygen docstrings only (`@file`/`@brief`/`@param`/`@return`); never Google/NumPy/reST.
- Full type hints; `Optional[X]` not `X | None`; `-> None` for void.
- PEP 8, 88-char lines (Black), f-strings, `pathlib.Path`, specific exceptions.
- **ASCII only** - no Greek letters, em-dashes, smart quotes, arrows. Use `gamma`,
  `alpha`, `->`, `deg`, `3x3` (letter x).
- **Never hardcode dimensions or hyperparameters.** Import structural values from
  `uncertainty_rl.utils.constants`; read tuneables from YAML with `.get()` defaults.

## Technical Context

- **Evidential deep learning on the actor only** (never the critic). Output is an NIG
  distribution `(gamma, nu, alpha, beta)`. RL algorithm is PPO via Stable-Baselines3.
- **State space: 13-dim** (the max, with both ablation flags on) =
  `[speed, vyaw, std_x, std_y, std_yaw, dx, dy, dyaw, left_dist, left_bearing,
  right_dist, right_bearing, forward_dist]`.
  - 0-1: EKF vehicle state (signed body-frame speed m/s, yaw rate). `VEHICLE_STATE_DIM=2`.
  - 2-4: EKF covariance features (diagonal std devs only - no off-diagonals).
    `COVARIANCE_FEATURES_DIM=3`. Present when `include_covariance=True`.
  - 5-7: relative target bay pose in ego body frame (dx, dy, dyaw). `TARGET_POSE_DIM=3`.
  - 8-12: hemispheric LiDAR clearance. `OBSTACLE_FEATURES_DIM=5`. Present when
    `include_obstacle_obs=True`.
  - Ablation flags reduce the dim: 13 (both on) / 10 / 8 / 5. Use
    `CARLAParkingEnv._compute_obs_dim()` at runtime; never assume 13.
  - 2D only, no z-axis.
- **Action: 3-dim** `[steering, throttle, brake]`. steering [-1,1], throttle [0,1],
  brake [0,1]. No reverse gear (forward perpendicular bay parking only).
- **Localisation**: EKF via `robot_localization` fusing RTK-GNSS + IMU. The 3 covariance
  features are pulled from the 6x6 EKF matrix at indices `[0,1,5]` for `[x,y,yaw]` (and
  only the diagonal std devs enter the obs). LiDAR is obstacle detection only.
- **Uncertainty source**: per-episode RTK fix-state tier sampling
  (`configs/deployment/sim/gnss_noise_profiles.yaml`) varies GNSS noise, which drives
  EKF covariance variation. NPC traffic and bay occupancy are secondary axes. **No
  weather** - FlatPlane does not render it.
- **Uncertainty formulae**: epistemic = `beta/(alpha-1)`, aleatoric =
  `beta/(nu*(alpha-1))`.
- **Simulator**: CARLA 0.9.16, custom FlatPlane OpenDRIVE world, ROS 2 bridge.

## Per-Directory Context

### `uncertainty_rl/networks/`
Core novel component. `EvidentialLayer` outputs NIG params per action dim.
`UncertaintyConditionedActor` has dual encoders (state + uncertainty); `EvidentialPolicyNetwork`
is a test harness. SB3 integration in `sb3_integration.py`: `EvidentialDistribution`,
`EvidentialActorCriticPolicy`, `EvidentialPPO`, `LayerNormActorCriticPolicy`. LayerNorm,
not BatchNorm. `get_action()` must always return `(action, uncertainty_dict)` with keys
`epistemic`, `aleatoric`, `total`, `gamma`, `nu`, `alpha`, `beta`.

### `uncertainty_rl/envs/`
Gymnasium CARLA parking env (`sim/carla_parking.py`). 13-dim obs (indices above); use
`_compute_obs_dim()` for the active dim. 3-dim action, forward perpendicular bays only.
EKF covariance comes from real `robot_localization` via ROS 2 (Docker required) - never
from CARLA ground truth. Reward is outcome-only: bay-frame corridor progress (dominant),
endgame hold bonus, obstacle clearance, soft out-of-bounds accumulation; collision -25
ego-fault / -10 non-fault, success +50, graded timeout penalty. Success: every corner of
the ego bounding box inside the bay polygon (`car_fully_inside_bay()`, margin set at env
construction) and speed < `SUCCESS_THRESHOLD_VELOCITY` held for `SUCCESS_DWELL_STEPS`.

### `uncertainty_rl/training/`
`train_ppo.py`: SB3 PPO, config-driven, `VecNormalize` wraps envs (eval env
`training=False, norm_reward=False`). Uses `EvidentialActorCriticPolicy`; when
`use_uncertainty_conditioning=True` the actor is `UncertaintyConditionedActor`
(dual-encoder), else flat MLP + `EvidentialLayer`. `tune_hyperparams.py` is the Optuna
study. Always read `documentation/CURRICULUM_PLAN.md` before changing training-side code.

### `uncertainty_rl/evaluation/`
`evaluate.py` sweeps `eval_conditions` (GNSS noise tier x traffic density x held-out /
OOD layouts) to measure degradation. Collects success rate, reward, position/orientation
error, uncertainty estimates -> CSV + seaborn plots. The condition sweep is the
dissertation's core experiment.

### `uncertainty_rl/ros2/`
`CovarianceExtractorNode` subscribes to `/odometry/filtered`, extracts the 3x3 [x,y,yaw]
submatrix from the 6x6 covariance (indices `[0,1,5]`), publishes `CovarianceEstimate`.
`CovarianceMonitorNode` logs/monitors it. `GnssNoiseRelayNode` / `ImuNoiseRelayNode`
(in `sensor_relay/`) inject per-episode sensor noise. QoS RELIABLE; all params via
`declare_parameter()`. ros2-bridge = Jazzy; training container = Humble (DDS is
distro-agnostic).

### `uncertainty_rl/utils/`
`constants.py` (structural dims/thresholds only), `config_merge.py` (the single merge
precedence: `sensor < agent < env (+stage) < train (+baseline)`), `covariance_utils.py`
(`extract_2d_covariance_features`), `geometry.py` (`car_fully_inside_bay`, relative pose),
`logging.py` (`DebugLogger`), `visualisation.py`. Plots: seaborn whitegrid, 300 DPI,
blue=epistemic, red=aleatoric, 95% confidence ellipses (chi-squared=5.991).

### `configs/`
All hyperparameters live in YAML; never hardcode. Only `train_config.yaml`,
`ros2_config.yaml`, `eval_config.yaml` at the root; everything else in subfolders
(`deployment/sim/`, `deployment/real/`, `layouts/`, `baselines/`, `training/`). Env
config is `configs/deployment/sim/env_config.yaml` (+ `curriculum/stage1..7.yaml`).
Always add `.get()` defaults and comment units.

## Data Flow - How Uncertainty Gets Into RL

1. `GnssNoiseRelayNode` samples an RTK fix-state tier per episode and injects that noise
   onto CARLA's GNSS output.
2. CARLA bridge publishes noisy GNSS + IMU to ROS 2 (`/carla/gnss`, `/carla/imu`).
3. `robot_localization` EKF fuses them, outputs `/odometry/filtered` with a 6x6 covariance.
4. `CovarianceExtractorNode` extracts the 3x3 [x,y,yaw] submatrix and writes
   `ekf_state.json`.
5. `CARLAParkingEnv` polls `ekf_state.json`, caches the EKF pose and the 3 diagonal
   covariance features via `extract_2d_covariance_features()`.
6. `step()` concatenates vehicle state (2D) + covariance (3D) + relative target (3D) +
   clearance (5D) -> up-to-13D observation (subject to ablation flags).
7. The evidential policy outputs NIG params and decomposes epistemic/aleatoric.

All containers share `ROS_DOMAIN_ID=42` (set in `docker-compose.yml`).

## Evidential Network Contract

`get_action(state, deterministic=False) -> (action, uncertainty_dict)` where the dict has
`epistemic` (`beta/(alpha-1)`), `aleatoric` (`beta/(nu*(alpha-1))`), `total`, and raw
`gamma`, `nu`, `alpha`, `beta` (lowercase, exact spelling). NIG constraints enforced in
`EvidentialLayer.forward()`: `gamma` unconstrained; `nu > 0` via `softplus + 1e-6`;
`alpha > 1` via `softplus + offset` (finite variance needs alpha > 1); `beta > 0` via
`softplus + 1e-6`.

## Evaluation Condition Sweep

`eval_config.yaml` orders conditions by `gnss_noise_multiplier` (scales the GNSS noise),
plus traffic density and held-out / OOD layouts - e.g. `nominal_empty`/`nominal_busy`
(RTK fixed, 1.0x), `rtk_float` (15x, ~30 cm), `rtk_standalone` (100x, ~2 m),
`rtk_lost`/`worst_case` (250x, ~5 m, safety-handoff demos), `ood_layout` (irregular_a),
`heldout_trapezoid`. Each runs `n_episodes`. No weather conditions exist.

## What NOT to Generate

- Non-ASCII characters or American spellings.
- Google/NumPy/reST docstrings.
- Evidential deep learning on the critic.
- Z-axis state components, reverse gear, weather/fog handling.
- MC dropout or ensemble uncertainty.
- Hardcoded dims/hyperparameters, bare `except:`, `os.path` in new code.
- CARLA ground truth in the observation pipeline (fix localisation instead).

## Common Pitfalls

1. **Hardcoding dims** - import from `constants.py` and use `_compute_obs_dim()`; the obs
   is not always 13 (ablation flags give 13/10/8/5).
2. **alpha <= 1** - infinite NIG variance; `alpha` must be offset above 1.
3. **Uncertainty dict keys** - exact lowercase `epistemic`, `aleatoric`, `total`,
   `gamma`, `nu`, `alpha`, `beta`.
4. **Covariance is diagonal-only [x,y,yaw]** at indices `[0,1,5]`; never use z (index 2)
   or off-diagonal terms.
5. **VecNormalize eval mode** - eval env must use `training=False, norm_reward=False`.
6. **No standalone training** - Docker + ROS 2 always required; the env blocks until
   covariance is received.
7. **`ROS_DOMAIN_ID`** - all nodes must share 42 (already in docker-compose).
