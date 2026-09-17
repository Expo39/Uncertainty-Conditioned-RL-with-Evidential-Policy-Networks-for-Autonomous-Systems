# envs/

Gymnasium-compatible CARLA parking environment with real EKF covariance from `robot_localization` embedded in the observation space.

## At a glance

- Observation comprises EKF kinematics, EKF covariance features, the relative target-bay pose in the ego body frame, and hemispheric LiDAR clearance. The active dimension is derived from the structural constants in [`uncertainty_rl/utils/constants.py`](../utils/constants.py) and the `include_covariance` / `include_obstacle_obs` ablation flags. Use `compute_obs_dim()` rather than hardcoding.
- Continuous action space `[steering, throttle, brake]` with no reverse gear, covering forward perpendicular bay parking only. See [Action space](#action-space) below.
- Reward is outcome-only in the sense that no contribution reads the localisation covariance or the policy's uncertainty estimates. The signal is dense, combining corridor-potential progress with terminal payments. Uncertainty enters as observation features and through the evidential head alone.
- Three pre-computed floor plans: `rectangle` (training), and `trapezoid` + `irregular_a` (OOD evaluation only).
- Sim-to-real capable: all observation features come from EKF and LiDAR, never CARLA ground truth.
- Requires the full Docker stack for training (carla-server + ros2-bridge + training).

## Modules

| Module | Class / purpose |
|--------|----------------|
| `sim/carla_parking.py` | `CARLAParkingEnv` - main Gymnasium env |
| `factory.py` | `make_env` - constructs the env with the bay margin appropriate to the caller |
| `covariance_subscriber.py` | `_CovarianceSubscriber` - file-based EKF state reader (no DDS) |
| `safety_wrapper.py` | `SafetyWrapper` - action modulation at eval time based on evidential uncertainty |
| `_parking_core.py` | Pure helper functions shared by sim and real-world deployment |
| `sim/helpers/_lot_spawner.py` | `LotSpawner` - per-episode static actor spawning (cones, parked vehicles) |
| `sim/helpers/_npc_controller.py` | `NPCController` - patrol vehicle and pedestrian lifecycle |
| `sim/helpers/_sensor_manager.py` | `SensorManager` - sensor spawning, callbacks, and cleanup |
| `real/deployment_utils.py` | `RealWorldDeployment` - surveyed datum loading and actuator calibration |
| `real/inference_loop.py` | `RealWorldInferenceLoop` - policy inference loop for physical vehicle |

## Internal data flow

```mermaid
flowchart TB
    subgraph ros2["ros2-bridge"]
        EKF["robot_localization EKF\n/odometry/filtered"]
        EXT["CovarianceExtractorNode"]
        EKF --> EXT
    end

    subgraph carla["carla-server"]
        LID["2D LiDAR\nCARLA API"]
        GT["ground truth pose\nCARLA API  (reward only)"]
    end

    subgraph env["CARLAParkingEnv"]
        COV["_CovarianceSubscriber\nekf_state.json"]
        OBS["build_observation()"]
        REW["_compute_reward()"]
    end

    EXT -->|ekf_state.json| COV
    COV -->|EKF kinematics + std_x, std_y, std_yaw| OBS
    LID -->|hemispheric clearance| OBS
    COV -->|dx, dy, dyaw| OBS
    GT -->|distance to target| REW
    OBS --> REW
```

## State space

CARLA ground truth is used **only** in `_compute_reward()` - never in the observation.
Absolute EKF position is excluded: it drifts across episodes in the odom frame and carries
no consistent signal. Navigation intent is encoded by the relative target-bay pose.

Feature blocks (in order of indices):

| Block | Source | Description |
|-------|--------|-------------|
| Vehicle state | EKF filtered twist | Body-frame longitudinal speed and yaw rate |
| EKF covariance | EKF | Diagonal standard deviations $\sigma_x, \sigma_y, \sigma_\psi$ (when `include_covariance=True`) |
| Relative target pose | Bay layout + EKF pose | $dx, dy, d\psi$ of the target bay in the ego body frame |
| Hemispheric LiDAR clearance | LiDAR scan | Nearest left, right, and forward obstacle features (when `include_obstacle_obs=True`) |

Structural constants and their concrete values live in
[`uncertainty_rl/utils/constants.py`](../utils/constants.py)
(`VEHICLE_STATE_DIM`, `COVARIANCE_FEATURES_DIM`, `TARGET_POSE_DIM`,
`OBSTACLE_FEATURES_DIM`, `TOTAL_OBS_DIM`). At runtime call
`compute_obs_dim(include_covariance, include_obstacle_obs)` for the
flag-aware dimension.

## Action space

```math
a = [\text{steering},\ \text{throttle},\ \text{brake}]
\qquad
\text{steering} \in [-1,1],\quad \text{throttle},\ \text{brake} \in [0,1]
```

Throttle and brake are independent non-negative axes so a held stop
(throttle = 0, brake > 0) is a stable region of the action space. No
reverse gear: forward perpendicular bay parking only.

## Target pose computation

$(dx, dy, d\psi)$ is recomputed every step from the current EKF pose and the fixed world-frame
target bay coordinates selected at episode start. See
[`uncertainty_rl/utils/geometry.py`](../utils/geometry.py) for the exact
expression and the CARLA-vs-REP-103 chirality handling.

## Reward function

Five contributions, three of them dense:

- **Corridor progress**, a telescoping bay-frame potential difference weighting
  cross-track and heading above along-track, so the vehicle is drawn onto the centreline
  and squared up before advancing. The dominant term.
- **Endgame finisher**, gated on the conjunction of the corridor factors, so stopping
  short and crooked pays close to nothing.
- **Obstacle clearance**, a smooth penalty for drifting toward an occupied neighbour.
- **Out-of-bounds**, charged per decision outside the inflated lot polygon and
  terminating once the accumulated cost reaches its limit.
- **Terminal payments**: success +50, collision -25 at ego fault and -10 otherwise, and
  a graded timeout penalty on final position and orientation error.

**"Outcome-only" refers to what the reward is blind to, not to sparsity.** No
contribution reads the localisation covariance or either uncertainty estimate, so
uncertainty is input-only and any uncertainty-dependent behaviour is emergent rather
than incentivised.

`_compute_reward()` in [`sim/carla_parking.py`](sim/carla_parking.py) is the source of
truth for every coefficient. The timeout penalty and stall truncation are applied in
`step()`.

## Success criterion

Success is geometric rather than a scalar pose tolerance, and the inward bay margin is
supplied at construction time, so the env has no training-versus-evaluation mode. The
criterion, its constants and the margin split are defined in
[utils/README.md](../utils/README.md#constantspy). The factory that selects the right
margin per caller is [`factory.py`](factory.py).

## Floor plans

| Floor plan | Shape | Bays | Role |
|-----------|-------|------|------|
| `rectangle` | Four-sided rectangular perimeter | 47 perpendicular, 2 motorcycle | Training, and the in-distribution evaluation anchor |
| `trapezoid` | Four-sided, widened at one end | 39 perpendicular | OOD only, never seen during training |
| `irregular_a` | Five-sided irregular polygon | 30 perpendicular | OOD only, never seen during training |

Geometry (corners, bay positions, spawn transform, patrol waypoints, pedestrian zones) is
pre-computed offline. Regenerate with `make generate-layouts`. The counts above are read
from the generated YAMLs, which remain authoritative along with the exact bay identifiers.
See [`configs/layouts/`](../../configs/layouts/). None of the three defines any obstacle,
so `obstacles` is empty in every layout.

## Episode randomisation

| Condition | Range | Effect |
|-----------|-------|--------|
| RTK fix-state tier | Start tier sampled per episode from the weights in `gnss_noise_profiles.yaml`, then wandering mid-episode along the Markov chain | Primary EKF uncertainty source |
| NPC patrol vehicles | Configurable in `parking_scenarios` | Dynamic LiDAR obstacles |
| Pedestrians | Configurable in `parking_scenarios` | Moving LiDAR obstacles |
| Bay occupancy rate | Configurable | Static parked vehicle density |
| No weather | N/A | Rendering is disabled (`no_rendering_mode`), and the generated FlatPlane world carries no weather model |

The tier is not held fixed for the episode. A start tier is drawn from the per-tier
weights and the fix state then walks the neighbour-only chain, so degradation and
recovery both occur mid-manoeuvre. The chain is stage-invariant.

Setting the `fixed_*` keys in `parking_scenarios` bypasses the sampler, which is how
curriculum overrides are expressed. `fixed_gnss_tier` is omitted from every stage, so the
start tier is always sampled. See
[configs/deployment/sim/curriculum/README.md](../../configs/deployment/sim/curriculum/README.md).

## Key interfaces

```python
from uncertainty_rl.envs import CARLAParkingEnv

env = CARLAParkingEnv(
    ros2_config=ros2_cfg,
    carla_sensors_config=sensor_cfg,
    parking_scenarios_config=scenario_cfg,
    include_covariance=True,
    include_obstacle_obs=True,
)

obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step(action)
```

## `_parking_core.py` helpers

| Function | Purpose |
|----------|---------|
| `compute_obs_dim` | Active obs dimension from ablation flags |
| `build_observation` | Fill the raw obs buffer from EKF state, uncertainty, and LiDAR, then return a `normalise_observation` copy (the policy obs) |
| `normalise_observation` | Scale each obs dim by its fixed physical range (`constants.py` `OBS_*_SCALE`) and clip to `+/-OBS_NORM_CLIP` - stage- and layout-invariant, so `VecNormalize` does reward-norm only (`norm_obs=False`) and OOD eval is unconfounded |
| `extract_obstacle_features` | Hemispheric LiDAR clearance (5-element buffer) |
| `load_floor_plan` | Select and cache a floor plan YAML for one episode |
| `wait_for_ekf` | Block until LiDAR and EKF data are both available |
| `calibrate_ekf_frame_offset` | Odom-to-world 2D rigid body transform |

## Configuration keys consumed

| Config file | Keys |
|-------------|------|
| [`configs/deployment/sim/env_config.yaml`](../../configs/deployment/sim/env_config.yaml) | `carla_host`, `carla_port`, `max_steps`, `action_repeat`, `include_covariance`, `include_obstacle_obs`, `carla_sensors.*`, `parking_scenarios.*` |
| [`configs/deployment/sim/gnss_noise_profiles.yaml`](../../configs/deployment/sim/gnss_noise_profiles.yaml) | RTK fix-state tiers and per-episode sampling weights |
| [`configs/layouts/*.yaml`](../../configs/layouts/) | Floor plan geometry (corners, bays, spawn, patrol, zones) |
| [`uncertainty_rl/utils/constants.py`](../utils/constants.py) | Structural dimensions and success thresholds |

## See also

- [scripts/visualise/README.md](../../scripts/visualise/README.md) - the 2D viewer, with a
  clip of an episode driven under the GNSS tier drift this env applies
- [uncertainty_rl/README.md](../README.md) - package overview
- [networks/README.md](../networks/README.md) - evidential actor that consumes this observation
- [ros2/README.md](../ros2/README.md) - EKF covariance extraction pipeline
- [docs/detailed_notes/envs/observation_space.md](../../docs/detailed_notes/envs/observation_space.md) - observation design rationale
- [docs/detailed_notes/envs/layout.md](../../docs/detailed_notes/envs/layout.md) - floor plan geometry and DSL
