# Copilot Instructions

## What This App Is

An autonomous parking system that knows when it doesn't know where it is and drives more carefully in response. It feeds EKF localisation uncertainty directly into an RL policy, and the policy uses evidential deep learning to quantify its own action uncertainty. Two layers of uncertainty awareness: "how sure am I about where I am?" (EKF covariance) and "how sure am I about what to do?" (evidential policy output).

MSc dissertation codebase - trains in CARLA simulation, evaluates across uncertainty levels, designed to transfer to a real instrumented parking lot.

### Three-Container Architecture

1. **carla-server** - CARLA 0.9.16 headless simulation with GPU passthrough. Generates realistic sensor noise and NPC traffic.
2. **ros2-bridge** - ROS 2 Jazzy running `robot_localization` EKF + CARLA bridge. Publishes real covariance from noisy sensors.
3. **training** - NVIDIA NGC PyTorch container. Training/evaluation happens here. Subscribes to EKF covariance via ROS 2 DDS.

Containers communicate over internal Docker network. Code is bind-mounted for hot-reload - edit on host, run in container.

## Language - British English ALWAYS

Use British English in all generated code: `localisation`, `initialisation`, `normalisation`, `optimisation`, `behaviour`, `colour`, `licence`, `minimisation`, `serialisation`, `visualisation`, `manoeuvre`, `defence`, `modelling`, `favour`, `honour`, `recognise`, `analyse`, `categorise`, `summarise`, `centre`, `metre`.

This includes: comments, docstrings, log messages, print statements, variable names where descriptive (e.g., `normalise_observations` not `normalize_observations`).

## Commenting Style - Doxygen ONLY

Use **Doxygen-style** docstrings consistently. Never Google-style, NumPy-style, or reST-style.

### Module docstring (every .py file)
```python
"""
@file filename.py
@brief One-line description.

Longer description if needed.
"""
```

### Class docstring
```python
class MyClass:
    """
    @class MyClass
    @brief One-line description.

    Longer description.
    """
```

### Method/function docstring
```python
def my_function(self, param: int, flag: bool = True) -> str:
    """
    @brief One-line description.
    @param param: Description.
    @param flag: Description.
    @return Description of return value.
    """
```

### Additional tags
- `@note` for important notes
- `@warning` for warnings
- `@todo(AB)` for planned work
- `@see` for cross-references

### Inline comments
Use `#` sparingly. Explain *why*, not *what*:
```python
# Offset alpha by 1.0 to ensure finite variance in the NIG distribution
alpha = F.softplus(out[..., 2]) + 1.0
```

## Type Hints - Always

Full type annotations on all function signatures:
```python
def process(self, data: np.ndarray, threshold: float = 0.5) -> Tuple[bool, Dict[str, float]]:
```
Use `-> None` for void functions. Import from `typing`. Use `Optional[X]` not `X | None`.

## Code Style

- PEP 8, 88-char line length (Black compatible).
- Imports: stdlib -> third-party -> local, blank line separated.
- No wildcard imports.
- `pathlib.Path` over `os.path` for new code.
- f-strings for formatting.
- `snake_case` functions/variables, `PascalCase` classes, `UPPER_SNAKE_CASE` constants.
- **ASCII only - no non-ASCII characters anywhere**. Every character must be printable ASCII (U+0020 to U+007E). No Greek letters, em-dashes, multiplication signs, degree symbols, smart quotes, or arrows. Use `gamma` not the Greek letter, `-` not em-dash, `3x3` with letter x, `deg` not degree symbol, `->` not arrow.
- Never hardcode hyperparameters - use YAML config with `.get()` defaults.
- Use `constants.py` for structural values: import dimensions and thresholds from `uncertainty_rl.utils.constants` rather than hardcoding `17`, `12`, `11`, `6`, `5`, `3`, `2`, `0.5`, `0.1`, etc.
- Specific exceptions, not bare `except:`.

## Technical Context

- **Evidential deep learning** on the **actor only** (not critic). Outputs NIG distribution: (gamma, nu, alpha, beta).
- **State space**: 17-dim default - `[vx, vy, vyaw, std_x, std_y, std_yaw, cov_xy, cov_xyaw, cov_yyaw, dx, dy, dyaw, left_dist, left_bearing, right_dist, right_bearing, forward_dist]`. Indices 0-2 EKF velocity; 3-8 EKF covariance features (std devs + off-diagonal cross-cov; diagonal variances dropped); 9-11 relative target pose (ego body frame); 12-16 hemispheric LiDAR clearance. 2D only, no z-axis. Ablation flags `include_covariance` and `include_obstacle_obs` reduce dims to 12/11/6.
- **Localisation**: EKF via `robot_localization`. Covariance extracted from 6x6 at indices [0,1,5] for [x, y, yaw].
- **RL**: PPO via Stable-Baselines3.
- **Simulator**: CARLA 0.9.16 with ROS 2 Jazzy bridge.
- **Uncertainty formulae**: epistemic = `beta/(alpha-1)`, aleatoric = `beta/(nu*(alpha-1))`.
- **Evidential loss**: `L = NLL(gamma,nu,alpha,beta,y) + lambda*|y-gamma|*(2*nu+alpha)`.
- **Uncertainty source**: Per-episode RTK fix-state tier sampling (from `configs/gnss_noise_profiles.yaml`) varies GNSS sensor noise, which drives EKF covariance variation. NPC vehicles/pedestrians and varying bay occupancy provide secondary variation. No weather effects - FlatPlane does not render them. Docker + ROS 2 always required for training.

## Documentation Links

Always consult official docs when generating code for these packages. Do not guess at API signatures - check docs.

### Python / ROS 2 Libraries

- **PyTorch**: https://pytorch.org/docs/stable/ - `nn.Module`, `nn.Linear`, `F.softplus`, `Normal`
- **Gymnasium**: https://gymnasium.farama.org/ - `gym.Env`, `spaces.Box`, `reset()`/`step()` API
- **Stable-Baselines3**: https://stable-baselines3.readthedocs.io/en/master/ - `PPO`, `VecNormalize`, custom policies
- **CARLA Python API**: https://carla.readthedocs.io/en/0.9.16/python_api/ - `Client`, `World`, `Vehicle`, `VehicleControl`
- **ROS 2 Jazzy rclpy**: https://docs.ros.org/en/jazzy/p/rclpy/ - `Node`, subscriptions, publishers, QoS
- **robot_localization**: https://docs.ros.org/en/jazzy/p/robot_localization/ - EKF, `Odometry` covariance
- **NumPy**: https://numpy.org/doc/stable/
- **Matplotlib**: https://matplotlib.org/stable/api/
- **Seaborn**: https://seaborn.pydata.org/
- **pytest**: https://docs.pytest.org/en/stable/

### Docker Infrastructure (Dockerfiles + docker-compose.yml)

Consult these when modifying Dockerfiles, docker-compose.yml, or debugging container build/runtime issues:

- **CARLA Docker image**: https://carla.readthedocs.io/en/0.9.15/build_docker/ - `carlasim/carla:0.9.15`, headless flags (`-RenderOffScreen`), port config (2000-2002)
- **CARLA ROS bridge**: https://github.com/carla-simulator/ros-bridge - ROS 2 bridge for CARLA sensors. `master` branch (no `ros2` branch). Cloned in `ros2/Dockerfile`
- **NVIDIA NGC PyTorch**: https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch - `nvcr.io/nvidia/pytorch:24.10-py3` base for training container (last Ubuntu 22.04 tag). Ubuntu 22.04 -> ROS 2 Humble
- **ROS 2 Docker images**: https://hub.docker.com/_/ros - `ros:jazzy-ros-base-noble` base for ros2-bridge container
- **Docker Compose**: https://docs.docker.com/compose/compose-file/ - Service orchestration, healthchecks, GPU reservations, volumes, networks
- **NVIDIA Container Toolkit**: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/ - `runtime: nvidia`, GPU passthrough, `NVIDIA_VISIBLE_DEVICES`
- **colcon**: https://colcon.readthedocs.io/en/released/ - ROS 2 workspace build tool, `colcon build --symlink-install`
- **rosdep**: https://docs.ros.org/en/jazzy/Tutorials/Intermediate/Rosdep.html - Dependency resolution for ROS 2 packages

## Per-Directory Context

### `uncertainty_rl/networks/`
Core novel component. `EvidentialLayer` outputs 4 NIG params per action dim. `EvidentialPolicyNetwork` is the full actor. `UncertaintyConditionedActor` has dual encoders (state + uncertainty). `LayerNorm` used, not `BatchNorm`. Softplus + offset constraints on nu, alpha, beta are mandatory. `get_action()` must always return `(action, uncertainty_dict)` with keys: `epistemic`, `aleatoric`, `total`, `gamma`, `nu`, `alpha`, `beta`.

### `uncertainty_rl/envs/`
Gymnasium-compatible CARLA parking env. 13-dim state (0=signed body-frame speed, 1=vyaw, 2-4 EKF covariance std_x/std_y/std_yaw, 5-7 relative target pose dx/dy/dyaw, 8-12 hemispheric LiDAR clearance). Use `compute_obs_dim()` for the actual dim under `include_covariance` / `include_obstacle_obs` ablation flags (13/10/8/5). 3-dim action `[steering, throttle, brake]`: steering in [-1, 1] left to right, throttle in [0, 1], brake in [0, 1] (no reverse gear). Forward perpendicular bay parking only. EKF uncertainty comes from real `robot_localisation` covariance via ROS 2 (Docker required). No standalone fallback for training. Reward: corridor progress (bay-frame potential, dominant term), endgame hold bonus, obstacle clearance term, soft out-of-bounds accumulation; collision -25 ego-fault / -10 non-fault, success +50, graded timeout penalty. Success: every corner of the ego bounding box must lie inside the bay polygon (geometric `car_fully_inside_bay()` check, with the inward margin set at env construction time: the active curriculum stage's `bay_margin` for training, `STRICT_BAY_MARGIN` for evaluation/demo/inspector) and speed < `SUCCESS_THRESHOLD_VELOCITY` (0.1 m/s), held for `SUCCESS_DWELL_STEPS` consecutive steps. Any orientation that physically fits is accepted.

### `uncertainty_rl/training/`
`train_ppo.py` implements SB3 PPO training with config-driven hyperparameters. `VecNormalize` wraps envs. Eval env uses `training=False`. Uses `EvidentialActorCriticPolicy` (custom SB3 policy subclass) with optional dual-encoder (`use_uncertainty_conditioning` config flag).

### `uncertainty_rl/evaluation/`
Sweeps across `eval_conditions` (weather, fog, sensor noise multipliers, traffic) to test degradation. `EvaluationMetrics` collects success rate, reward, position/orientation errors, uncertainty estimates. Generates CSV + seaborn plots. The condition sweep is the centrepiece of the dissertation's experimental chapter.

### `uncertainty_rl/ros2/`
ROS 2 Jazzy nodes. `CovarianceExtractorNode` subscribes to `/odometry/filtered`, extracts 3x3 [x,y,yaw] submatrix from 6x6 covariance (indices [0,1,5]), publishes as `CovarianceEstimate` custom message. QoS: RELIABLE. All params via `declare_parameter()`.

### `uncertainty_rl/utils/`
`MetricsLogger` (CSV/JSON), `UncertaintyTracker` (sliding window). Visualisation uses seaborn whitegrid, 300 DPI, blue=epistemic, red=aleatoric, 95% confidence ellipses (chi-squared=5.991).

### `configs/`
All hyperparameters live in YAML. Never hardcode. Always add `.get()` defaults in consuming code. Comment units.

## What NOT to Generate

- Greek letters or non-ASCII characters (use ASCII equivalents: `gamma`, `alpha`, `std_x`, etc.)
- American English spellings
- Google/NumPy/reST-style docstrings
- Evidential deep learning on the critic
- Z-axis state components
- MC dropout or ensemble uncertainty methods
- Hardcoded hyperparameters
- Bare `except:` blocks
- `os.path` in new code (use `pathlib.Path`)

## Developer Workflows - Critical Commands

### Docker is Required for Training
**Always** use Docker for training/evaluation. There is no standalone mode - the env requires ROS 2 covariance from `robot_localization` running in ros2-bridge container.

```bash
# Initial setup (once)
make docker-build        # ~15-20 min first time
make docker-up           # Start all 3 containers
make docker-ps           # Verify carla-server and ros2-bridge are healthy

# Typical dev workflow
make docker-shell        # Interactive bash in training container
make docker-train-short  # 10k step smoke test
make docker-train        # Full 1M step training
make docker-eval         # Sweep across eval_conditions

# Debugging
make docker-logs-ros2    # Check EKF is publishing covariance
make docker-logs-carla   # Check CARLA server logs
make docker-dev          # Start stack + drop into training shell
```

### Testing Without Docker
Tests that don't require CARLA/ROS 2 can run natively:
```bash
pytest tests/test_evidential_policy.py  # Pure PyTorch tests
pytest tests/ -m "not integration"      # Skip CARLA/ROS 2 tests
```

Integration tests requiring CARLA:
```bash
make docker-test  # Run full suite inside training container
```

### Config-Driven Development
Never hardcode hyperparameters. All tuneable values live in `configs/*.yaml`:
- `train_config.yaml` - PPO params, sensor noise, weather, episode length
- `eval_config.yaml` - Condition sweep definitions (weather, fog, traffic)
- `ros2_config.yaml` - Topic names, QoS settings

When consuming configs, always use `.get()` with defaults:
```python
learning_rate = config.get("learning_rate", 3e-4)
max_steps = config.get("max_steps", 500)
```

### Makefile Commands
`make help` shows all commands. Key ones:
- **docker-build** / **docker-up** / **docker-down** - Lifecycle
- **docker-train** / **docker-train-short** - Training
- **docker-eval** - Evaluation across conditions
- **docker-test** - Pytest inside container
- **docker-shell** - Interactive bash
- **docker-logs** - Follow all logs
- **docker-clean** - Stop and remove volumes

## Data Flow - How Uncertainty Gets Into RL

1. **GNSS noise relay** (`GnssNoiseRelayNode`) samples an RTK fix-state tier per episode from `gnss_noise_profiles.yaml` and injects that noise level onto CARLA's GNSS sensor output
2. **CARLA bridge** publishes noisy GNSS + IMU to ROS 2 (`/carla/gnss`, `/carla/imu`)
3. **robot_localization EKF** fuses GNSS + IMU, outputs `/odometry/filtered` with 6x6 covariance
4. **CovarianceExtractorNode** (`uncertainty_rl/ros2/`) subscribes to `/odometry/filtered`, extracts 3x3 [x,y,yaw] submatrix, writes `ekf_state.json` (DDS bypass for training container)
5. **CARLAParkingEnv** (`uncertainty_rl/envs/carla_parking.py`) polls `ekf_state.json` via `_CovarianceSubscriber`, caches latest pose + 6-element covariance features via `extract_2d_covariance_features()`
6. **Gymnasium `step()`** concatenates velocity (3D) + covariance features (6D) + relative target (3D) + hemispheric clearance (5D) -> 17D observation
7. **Evidential policy** receives 17D observation, outputs NIG params (gamma, nu, alpha, beta), decomposes into epistemic/aleatoric uncertainty

**Critical**: Training container must subscribe to ROS 2 topics via DDS. Set `ROS_DOMAIN_ID=42` in all containers (already in `docker-compose.yml`).

## Evidential Network Contract

All evidential networks must satisfy:
```python
def get_action(state: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    @return (action, uncertainty_dict) where uncertainty_dict contains:
        - 'epistemic': beta/(alpha-1)
        - 'aleatoric': beta/(nu*(alpha-1))
        - 'total': epistemic + aleatoric
        - 'gamma', 'nu', 'alpha', 'beta': raw NIG params
    """
```

NIG parameter constraints (enforced in `EvidentialLayer.forward()`):
- `gamma` - mean, no constraint
- `nu` - precision, `> 0` via `F.softplus(x) + 1e-6`
- `alpha` - shape, `> 1` via `F.softplus(x) + 1.0` (finite variance requires alpha > 1)
- `beta` - rate, `> 0` via `F.softplus(x) + 1e-6`

Evidential loss (see `uncertainty_rl/networks/evidential_policy.py`):
```python
L = NLL(gamma, nu, alpha, beta, y) + lambda * |y - gamma| * (2*nu + alpha)
```

## Evaluation Condition Sweep

The dissertation's core experiment: test agent across escalating localisation uncertainty. `eval_config.yaml` defines ordered conditions keyed on `gnss_noise_multiplier` (scales the GNSS noise tier):
1. **clear_low_noise** - 0.5x GNSS noise multiplier, no NPC traffic
2. **clear_nominal** - 1.0x noise, light NPC traffic
3. **noisy_moderate** - 1.5x noise, moderate NPC traffic + pedestrians
4. **noisy_degraded** - 2.0x noise, dense NPC traffic
5. **high_noise** - 3.0x noise, dense traffic + pedestrians
6. **extreme_noise** - 5.0x GNSS noise (simulates multipath / RTK loss of fix)

Each condition runs 100 episodes. Metrics collected:
- Success rate (ego bounding box fully inside the bay polygon via car_fully_inside_bay(), velocity <0.1 m/s, held for success_dwell_steps)
- Mean reward, position/orientation error
- Epistemic/aleatoric uncertainty evolution
- Collision rate, timeout rate

Outputs: CSV + seaborn plots (300 DPI, whitegrid, blue=epistemic, red=aleatoric, 95% confidence ellipses via chi-squared=5.991).

## Testing Patterns

Use fixtures from `tests/conftest.py`:
- `state_batch`, `single_state`, `action_batch` - random tensors
- `low_uncertainty_state`, `high_uncertainty_state` - controlled uncertainty levels
- `train_config` - minimal config dict

Test structure (see `tests/test_evidential_policy.py`):
```python
class TestEvidentialLayer:
    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        self.layer = EvidentialLayer(input_dim=64, output_dim=3)

    def test_nu_is_positive(self) -> None:
        x = torch.randn(8, 64)
        _, nu, _, _ = self.layer(x)
        assert (nu > 0).all()
```

Mark integration tests: `@pytest.mark.integration` (requires CARLA/ROS 2, skip with `-m "not integration"`).

## Stable-Baselines3 Integration

Uses `EvidentialActorCriticPolicy` - a custom SB3 `ActorCriticPolicy` subclass with `EvidentialPPO`.

Training flow (`uncertainty_rl/training/train_ppo.py`):
1. Load config from YAML
2. Create vectorised envs via `make_env()` callable (allows per-env CARLA port offset)
3. Wrap in `VecNormalize(training=True)` for obs/reward normalisation
4. Create eval env with `VecNormalize(training=False, norm_reward=False)`
5. Instantiate `EvidentialPPO` with `EvidentialActorCriticPolicy`, network arch from config
6. Register `CheckpointCallback` (every 50k steps), `EvalCallback` (every 10k steps)
7. Train with `model.learn(total_timesteps, callbacks)`

When `use_uncertainty_conditioning=True` in config, the actor uses `UncertaintyConditionedActor` (dual-encoder splitting velocity vs covariance features). When `False`, a flat MLP + `EvidentialLayer` is used.

## ROS 2 Custom Messages

`uncertainty_rl_msgs/msg/CovarianceEstimate.msg`:
```
float64 x
float64 y
float64 yaw
float64[9] covariance  # Flattened 3x3 [x,y,yaw] covariance
```

Build process (handled in ros2-bridge Dockerfile):
```bash
colcon build --packages-select uncertainty_rl_msgs
source install/setup.bash
```

QoS settings: `RELIABLE` with depth 10 (see `CovarianceExtractorNode` and `_CovarianceSubscriber`).

## Common Pitfalls

1. **Forgetting `ROS_DOMAIN_ID`** - All ROS 2 nodes must use same domain ID (42). Already set in docker-compose.
2. **Hardcoding dims** - Import from `uncertainty_rl.utils.constants`: `TOTAL_OBS_DIM`, `ACTION_DIM`, `COVARIANCE_FEATURES_DIM`, `SUCCESS_THRESHOLD_*`.
3. **Alpha <= 1** - Causes infinite variance in NIG. Must offset by 1.0: `F.softplus(x) + 1.0`.
4. **Uncertainty dict keys** - Must be `epistemic`, `aleatoric`, `total`, `gamma`, `nu`, `alpha`, `beta` (lowercase, exact spelling).
5. **2D vs 3D covariance** - `extract_2d_covariance_features()` accepts 3x3 or 6x6, extracts [x,y,yaw] at indices [0,1,5]. Never use z-axis (index 2).
6. **VecNormalize eval mode** - Eval env must use `training=False, norm_reward=False` or normalisation stats will drift.
7. **CARLA port conflicts** - Multi-env training requires port offset: `carla_port + rank`. Already handled in `make_env()`.
8. **No standalone training** - Docker + ROS 2 always required. The env blocks until covariance is received from ros2-bridge.
