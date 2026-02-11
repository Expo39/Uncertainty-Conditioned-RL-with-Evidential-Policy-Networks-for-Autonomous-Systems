# Copilot Instructions

## What This App Is

An autonomous parking system that knows when it doesn't know where it is and drives more carefully in response. It feeds SLAM localisation uncertainty directly into an RL policy, and the policy uses evidential deep learning to quantify its own action uncertainty. Two layers of uncertainty awareness: "how sure am I about where I am?" (SLAM covariance) and "how sure am I about what to do?" (evidential policy output).

MSc dissertation codebase — trains in CARLA simulation, evaluates across uncertainty levels, designed to transfer to a real instrumented parking lot at Lemonworx LTD.

## Language — British English ALWAYS

Use British English in all generated code: `localisation`, `initialisation`, `normalisation`, `optimisation`, `behaviour`, `colour`, `licence`, `minimisation`, `serialisation`, `visualisation`, `manoeuvre`, `defence`, `modelling`, `favour`, `honour`, `recognise`, `analyse`, `categorise`, `summarise`, `centre`, `metre`.

This includes: comments, docstrings, log messages, print statements, variable names where descriptive (e.g., `normalise_observations` not `normalize_observations`).

## Commenting Style — Doxygen ONLY

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

## Type Hints — Always

Full type annotations on all function signatures:
```python
def process(self, data: np.ndarray, threshold: float = 0.5) -> Tuple[bool, Dict[str, float]]:
```
Use `-> None` for void functions. Import from `typing`. Use `Optional[X]` not `X | None`.

## Code Style

- PEP 8, 88-char line length (Black compatible).
- Imports: stdlib → third-party → local, blank line separated.
- No wildcard imports.
- `pathlib.Path` over `os.path` for new code.
- f-strings for formatting.
- `snake_case` functions/variables, `PascalCase` classes, `UPPER_SNAKE_CASE` constants.
- Never hardcode hyperparameters — use YAML config with `.get()` defaults.
- Specific exceptions, not bare `except:`.

## Technical Context

- **Evidential deep learning** on the **actor only** (not critic). Outputs NIG distribution: (γ, ν, α, β).
- **State space**: 15-dim — `[x, y, yaw, vx, vy, vyaw, σ_x, σ_y, σ_yaw, cov_xx, cov_yy, cov_ψψ, cov_xy, cov_xψ, cov_yψ]`. 2D only, no z-axis.
- **SLAM**: EKF via `robot_localization`. Covariance extracted from 6×6 at indices [0,1,5] for [x, y, yaw].
- **RL**: PPO primary, SAC secondary. Stable-Baselines3.
- **Simulator**: CARLA 0.9.13+ with ROS 2 Jazzy bridge.
- **Uncertainty formulae**: epistemic = `β/(α-1)`, aleatoric = `β/(ν(α-1))`.
- **Evidential loss**: `L = NLL(γ,ν,α,β,y) + λ|y-γ|(2ν+α)`.

## Documentation Links

Always consult official docs when generating code for these packages:

- **PyTorch**: https://pytorch.org/docs/stable/ — `nn.Module`, `nn.Linear`, `F.softplus`, `Normal`
- **Gymnasium**: https://gymnasium.farama.org/ — `gym.Env`, `spaces.Box`, `reset()`/`step()` API
- **Stable-Baselines3**: https://stable-baselines3.readthedocs.io/en/master/ — `PPO`, `SAC`, `VecNormalize`, custom policies
- **CARLA Python API**: https://carla.readthedocs.io/en/0.9.13/python_api/ — `Client`, `World`, `Vehicle`, `VehicleControl`
- **ROS 2 Jazzy rclpy**: https://docs.ros.org/en/jazzy/p/rclpy/ — `Node`, subscriptions, publishers, QoS
- **robot_localization**: https://docs.ros.org/en/jazzy/p/robot_localization/ — EKF, `Odometry` covariance
- **NumPy**: https://numpy.org/doc/stable/
- **Matplotlib**: https://matplotlib.org/stable/api/
- **Seaborn**: https://seaborn.pydata.org/
- **pytest**: https://docs.pytest.org/en/stable/

Do not guess at API signatures — check docs.

## Per-Directory Context

### `uncertainty_rl/networks/`
Core novel component. `EvidentialLayer` outputs 4 NIG params per action dim. `EvidentialPolicyNetwork` is the full actor. `UncertaintyConditionedActor` has dual encoders (state + uncertainty). `LayerNorm` used, not `BatchNorm`. Softplus + offset constraints on ν, α, β are mandatory. `get_action()` must always return `(action, uncertainty_dict)` with keys: `epistemic`, `aleatoric`, `total`, `gamma`, `nu`, `alpha`, `beta`.

### `uncertainty_rl/envs/`
Gymnasium-compatible CARLA parking env. 15-dim state, 3-dim action `[steering, throttle, brake]`. SLAM uncertainty currently simulated (covariance grows with velocity). Falls back to zero-state if CARLA unavailable. Reward: `-distance - 0.5*orientation_error - 0.1*velocity + 100*success`. Success: <0.5m, <10°, <0.1 m/s.

### `uncertainty_rl/training/`
SB3 training with config-driven hyperparameters. `VecNormalize` wraps envs. Eval env uses `training=False`. The evidential policy is NOT yet integrated into SB3 — needs custom policy class wrapping `EvidentialPolicyNetwork` as actor. PPO is the primary algorithm (a `train_ppo.py` is needed).

### `uncertainty_rl/evaluation/`
Sweeps across `uncertainty_noise_std` values to test degradation. `EvaluationMetrics` collects success rate, reward, position/orientation errors, uncertainty estimates. Generates CSV + seaborn plots. The noise sweep is the centrepiece of the dissertation's experimental chapter.

### `uncertainty_rl/ros2/`
ROS 2 Jazzy nodes. `CovarianceExtractorNode` subscribes to `/odometry/filtered`, extracts 3×3 [x,y,yaw] submatrix from 6×6 covariance (indices [0,1,5]), publishes as `Float64MultiArray`. QoS: RELIABLE. All params via `declare_parameter()`.

### `uncertainty_rl/utils/`
`MetricsLogger` (CSV/JSON), `UncertaintyTracker` (sliding window). Visualisation uses seaborn whitegrid, 300 DPI, blue=epistemic, red=aleatoric, 95% confidence ellipses (χ²=5.991).

### `configs/`
All hyperparameters live in YAML. Never hardcode. Always add `.get()` defaults in consuming code. Comment units.

## What NOT to Generate

- American English spellings
- Google/NumPy/reST-style docstrings
- Evidential deep learning on the critic
- Z-axis state components
- MC dropout or ensemble uncertainty methods
- Hardcoded hyperparameters
- Bare `except:` blocks
- `os.path` in new code (use `pathlib.Path`)
