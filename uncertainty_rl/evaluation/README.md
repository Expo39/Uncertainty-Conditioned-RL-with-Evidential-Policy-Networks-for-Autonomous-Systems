# evaluation/

Performance evaluation across varying physical conditions. Tests whether the uncertainty-conditioned policy degrades more gracefully than baselines as localisation uncertainty increases.

## At a glance

- 14 de-confounded conditions: every condition varies exactly ONE factor against the in-distribution anchor (rectangle lot, occupancy 0.5, no dynamic actors)
- Base noise from `env_config.yaml`; per-condition multipliers applied by `_scale_sensor_noise()` (LiDAR noise always enabled at eval, matching the training realism floor)
- Success: every corner of the ego bounding box inside the bay polygon (`car_fully_inside_bay()` at `STRICT_BAY_MARGIN`) with speed $< 0.1$ m/s, held for `SUCCESS_DWELL_STEPS`
- `n_episodes = 100` per condition, deterministic (mean) actions
- OOD conditions use `irregular_a` floor plan (never seen during training)
- No weather variation - FlatPlane does not render weather effects

## Modules

| Module | Class / purpose |
|--------|----------------|
| `evaluate.py` | Orchestration: `evaluate_agent()` - single-condition episode loop; `evaluate_across_conditions()` - full condition sweep (14 conditions, returns `(DataFrame, run_output_dir)`); `main()` - CLI. Re-exports the moved symbols below so `...evaluation.evaluate.*` import paths stay stable |
| `metrics.py` | `EvaluationMetrics` - per-condition results dataclass (success rate, outcome taxonomy rates, uncertainty stats, per-episode records); `_classify_outcome()` - failure-mode taxonomy. No torch / stable-baselines3 dependency |
| `env_builder.py` | The condition -> env contract: `_scale_sensor_noise()` - `base * multiplier`; `build_eval_env_factory()` - single source of truth, returns the BARE env factory + SafetyWrapper params; `make_eval_env()` - wraps it in SafetyWrapper + DummyVecEnv for the sweep |
| `plots.py` | `plot_evaluation_results()` - seaborn bar charts + stacked failure-mode figure (the only module pulling in matplotlib/seaborn) |
| `__init__.py` | Lazily re-exports `EvaluationMetrics`, `evaluate_agent`, `evaluate_across_conditions`, `plot_evaluation_results` |

## Internal data flow

```mermaid
flowchart TB
    subgraph cfg["configs/"]
        EC["eval_config.yaml\nconditions, n_episodes"]
        NC["deployment/sim/env_config.yaml\nbase sensor noise"]
    end

    subgraph ckpt["checkpoint/"]
        MDL["model.zip"]
        VN["vec_normalize.pkl"]
    end

    subgraph ev["evaluate.py + env_builder.py"]
        SCALE["env_builder._scale_sensor_noise()\nbase * multiplier"]
        LOOP["evaluate_across_conditions()\n14 conditions"]
        AGENT["evaluate_agent()\nn_episodes"]
        ENV["env_builder.make_eval_env()\nCARLAParkingEnv"]
    end

    subgraph out["evaluation_results/baseline/leaf/"]
        CSV["evaluation_results.csv\nepisode_records.csv"]
        PNG["evaluation_plots.png\nfailure_modes.png"]
    end

    EC --> LOOP
    NC --> SCALE
    MDL --> AGENT
    VN --> ENV
    SCALE --> ENV
    LOOP --> AGENT
    AGENT --> ENV
    ENV -->|obs + reward| AGENT
    AGENT -->|EvaluationMetrics| LOOP
    LOOP --> CSV
    LOOP --> PNG
```

## Evaluation conditions

De-confounded sweep: every condition varies exactly ONE factor against the
in-distribution anchor (rectangle lot, bay occupancy 0.5, no dynamic actors).
GNSS multipliers derive from `configs/deployment/sim/gnss_noise_profiles.yaml`
as `metric_stddev_m / 0.020` (the RTK-fixed base); a condition with a multiplier
resolves to the matching fix-state tier and HOLDS it for the whole episode (the
relay's Markov drift is suppressed via the per-episode `hold_tier` flag), so the
level is a clean independent variable. The held conditions still go through the
exact per-episode publish training uses (tier signal + GNSS datum re-latch),
which is why no condition needs a separate code path. The anchor leaves the flag
off and runs the training noise process itself (per-episode tier sampling +
mid-episode Markov drift). The exact list lives in `configs/eval_config.yaml`.

### Anchor

| Condition | GNSS | Bay occ. | Notes |
|-----------|------|----------|-------|
| `anchor_deployment` | training Markov process | 0.5 | The deployment condition |

### GNSS axis (occupancy 0.5, rectangle)

| Condition | GNSS mult | Approx. noise | Notes |
|-----------|-----------|---------------|-------|
| `gnss_rtk_fixed` | 1.0x | ~0.02 m | Nominal RTK fixed |
| `gnss_rtk_float` | 18.0x | ~0.36 m | Marginal for 2.5 m bays |
| `gnss_standalone` | 90.0x | ~1.8 m | Cautious or abort expected |
| `gnss_degraded` | 250.0x | ~5.0 m | Worst tier; safety handoff expected |

### Occupancy axis (GNSS RTK fixed, rectangle)

| Condition | Bay occ. | Notes |
|-----------|----------|-------|
| `occupancy_empty` | 0.0 | Below training minimum 0.2 (mild OOD) |
| `occupancy_min` | 0.2 | Training minimum |
| `occupancy_max` | 0.8 | Training maximum |

### LiDAR axis (GNSS RTK fixed, occupancy 0.5, rectangle)

Degrades only the obstacle channel (obs 8-12). The EKF fuses GNSS + IMU and
never consumes LiDAR, so the localisation stds stay at the RTK-fixed floor:
an EKF-std safety gate is structurally blind to this condition, while the
evidential head sees the corrupted obstacle features.

| Condition | LiDAR mult | Approx. noise | Notes |
|-----------|------------|---------------|-------|
| `lidar_degraded` | 25.0x | 0.5 m 1-sigma | EKF-blind sensor degradation |

### Held-out layout generalisation (occupancy 0.5)

Training uses the `rectangle` floor plan only; `trapezoid` is held out and
`irregular_a` (five-sided lot with a diagonal top wall) is the designated OOD
layout. Success here measures generalisation across lot geometry.

| Condition | Floor plan | GNSS mult |
|-----------|-----------|-----------|
| `heldout_trapezoid_rtk_fixed` | `trapezoid` | 1.0x |
| `heldout_trapezoid_rtk_float` | `trapezoid` | 18.0x |
| `ood_irregular_rtk_fixed` | `irregular_a` | 1.0x |
| `ood_irregular_rtk_float` | `irregular_a` | 18.0x |

### Stress beyond training ranges

| Condition | GNSS mult | IMU mult | Bay occ. | Notes |
|-----------|-----------|----------|----------|-------|
| `gnss_stress_imu` | 250.0x | 3.0x | 0.5 | Worst GNSS tier + IMU stress; handoff demo |

## Metrics collected

| Metric | Source |
|--------|--------|
| Success rate | `info["success"]` from `CARLAParkingEnv.step()` |
| Mean episode reward | Accumulated per episode |
| Mean steps to termination | Episode length |
| Epistemic uncertainty | `get_action_with_uncertainty()` mean over episode (evidential only) |
| Aleatoric uncertainty | `get_action_with_uncertainty()` mean over episode (evidential only) |

**Success criteria**: judged geometrically in the env, not by scalar thresholds.
Every corner of the ego bounding box must lie inside the target bay polygon
(`car_fully_inside_bay()` with `STRICT_BAY_MARGIN` at evaluation) and speed must be
below `SUCCESS_THRESHOLD_VELOCITY`, held for `SUCCESS_DWELL_STEPS` consecutive steps.
See `uncertainty_rl/utils/constants.py`.

## Key interfaces

```python
from uncertainty_rl.evaluation import (
    EvaluationMetrics,
    evaluate_agent,
    evaluate_across_conditions,
    plot_evaluation_results,
)

df, run_output_dir = evaluate_across_conditions(
    model_path="checkpoints/full_method/seed42_12062026-0536/final_model",
    eval_config_path="configs/eval_config.yaml",
    env_config_path="configs/deployment/sim/env_config.yaml",
    train_config_path="configs/train_config.yaml",
)
# df: one row per condition; run_output_dir: evaluation_results/<baseline>/<leaf>
plot_evaluation_results(df, output_dir=run_output_dir)
```

Run via Make:

```bash
make docker-eval          # Full 13-condition sweep (100 episodes per condition)
make eval-visualise-2d    # Detachable 2D bird's-eye replay after evaluation
```

## Configuration keys consumed

| Config file | Keys |
|-------------|------|
| `configs/eval_config.yaml` | `model_path`, `n_episodes`, `deterministic`, `debug`, `near_miss_threshold_m`, `eval_conditions` (connection/timing come from env_config) |
| `configs/deployment/sim/env_config.yaml` | `carla_sensors.gnss.*`, `carla_sensors.imu.*` (base noise, scaled by condition multipliers) |
| `configs/baselines/<arm>.yaml` | `policy_type`, `include_covariance`, `include_obstacle_obs` (model class + obs shape of the evaluated checkpoint) |
| `uncertainty_rl/utils/constants.py` | `SUCCESS_THRESHOLD_VELOCITY`, `STRICT_BAY_MARGIN` (the strict margin applied during evaluation/demo/inspector; training reads `bay_margin` from config, relaxed per curriculum stage). Success is tested geometrically via `car_fully_inside_bay()` in `utils/geometry.py`. |

<!-- img:placeholder name="eval_degradation" caption="Success rate and epistemic uncertainty across the 9 evaluation conditions" -->
![Evaluation degradation placeholder](../../docs/media/eval_degradation.png)

<!-- gif:placeholder name="baseline_comparison" caption="Vanilla PPO vs full method side by side under degraded GNSS" -->
![Baseline comparison placeholder](../../docs/media/baseline_comparison.gif)

<!-- gif:placeholder name="safety_handoff" caption="SafetyWrapper under rtk_lost conditions - aleatoric throttle cap slows the approach, then epistemic crosses the handoff threshold and the vehicle brakes to a stop" -->
![Safety handoff placeholder](../../docs/media/safety_handoff.gif)

## See also

- [uncertainty_rl/README.md](../README.md) - package overview
- [training/README.md](../training/README.md) - training the model evaluated here
- [networks/README.md](../networks/README.md) - evidential policy providing uncertainty estimates
- [docs/detailed_notes/envs/observation_space.md](../../docs/detailed_notes/envs/observation_space.md) - observation design that underpins the evaluation metrics
