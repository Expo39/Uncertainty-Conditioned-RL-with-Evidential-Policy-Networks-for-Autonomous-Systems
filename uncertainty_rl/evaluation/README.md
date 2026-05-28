# evaluation/

Performance evaluation across varying physical conditions. Tests whether the uncertainty-conditioned policy degrades more gracefully than baselines as localisation uncertainty increases.

## At a glance

- 9 evaluation conditions sweep GNSS noise from $1\times$ (RTK fixed, ~2 cm) to $250\times$ (~5 m)
- Base noise from `env_config.yaml`; per-condition multipliers applied by `_scale_sensor_noise()`
- Success: position error $< 0.5$ m, orientation $< 10$ deg, speed $< 0.1$ m/s
- `n_episodes = 100` per condition, deterministic (mean) actions
- OOD conditions use `irregular_a` floor plan (never seen during training)
- No weather variation - FlatPlane does not render weather effects

## Modules

| Module | Class / purpose |
|--------|----------------|
| `evaluate.py` | `EvaluationMetrics` - per-condition results dataclass; `evaluate_agent()` - single-condition episode loop; `evaluate_across_conditions()` - full 9-condition sweep; `plot_evaluation_results()` - seaborn bar charts and CSV export |
| `__init__.py` | Re-exports `EvaluationMetrics`, `evaluate_agent`, `evaluate_across_conditions`, `plot_evaluation_results` |

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

    subgraph ev["evaluate.py"]
        SCALE["_scale_sensor_noise()\nbase * multiplier"]
        LOOP["evaluate_across_conditions()\n9 conditions"]
        AGENT["evaluate_agent()\nn_episodes"]
        ENV["CARLAParkingEnv"]
    end

    subgraph out["evaluation_results/"]
        CSV["metrics.csv"]
        PNG["plots/*.png"]
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

Primary uncertainty axis: GNSS noise tier (RTK fix state). `gnss_noise_multiplier` scales the base RTK-fixed stddev (0.02 m). No weather variation.

### Condition ladder

```
GNSS noise escalation (base stddev = 0.02 m at 1x):

  mult      approx. noise   condition
  -------   -------------   ---------------------------------------------------
     1.0x       ~0.02 m    nominal_empty  (RTK fixed, empty lot)
     1.0x       ~0.02 m    nominal_busy   (RTK fixed, full traffic)
    15.0x       ~0.30 m    rtk_float      (marginal localisation)
    15.0x       ~0.30 m    rtk_float_busy (RTK float + elevated IMU)
   100.0x       ~2.00 m    rtk_standalone (policy should decline to park)
   250.0x       ~5.00 m    rtk_lost       (safety handoff expected)
   250.0x       ~5.00 m    worst_case     (RTK lost + IMU 6x, handoff demo)
                  OOD       ood_layout           (irregular_a, RTK fixed)
                  OOD       ood_layout_degraded  (irregular_a, RTK float)
```

### Performance sweep (in-distribution floor plans)

| Condition | GNSS mult | IMU mult | Patrol | Ped. prob | Bay occ. | Notes |
|-----------|-----------|----------|--------|-----------|----------|-------|
| `nominal_empty` | $1.0\times$ | $1.0\times$ | 0 | 0.0 | 0.6 | RTK fixed, best-case |
| `nominal_busy` | $1.0\times$ | $1.0\times$ | 1 | 1.0 | 0.8 | RTK fixed, full traffic |
| `rtk_float` | $15.0\times$ | $1.0\times$ | 1 | 0.8 | 0.6 | Marginal localisation |
| `rtk_float_busy` | $15.0\times$ | $1.5\times$ | 1 | 1.0 | 0.8 | Compound degradation |
| `rtk_standalone` | $100.0\times$ | $1.0\times$ | 1 | 0.8 | 0.6 | Policy should decline |
| `rtk_lost` | $250.0\times$ | $2.0\times$ | 1 | 1.0 | 0.8 | Safety handoff expected |

### OOD calibration (epistemic uncertainty check)

Tests whether epistemic uncertainty rises on the `irregular_a` floor plan (56 bays, nine-sided irregular polygon), which is never seen during training.

| Condition | Floor plan | GNSS mult | IMU mult | Patrol | Ped. prob | Bay occ. |
|-----------|-----------|-----------|----------|--------|-----------|----------|
| `ood_layout` | `irregular_a` | $1.0\times$ | $1.0\times$ | 1 | 1.0 | 0.6 |
| `ood_layout_degraded` | `irregular_a` | $15.0\times$ | $1.5\times$ | 1 | 1.0 | 0.6 |

### Safety handoff demonstration

Beyond the training distribution on all axes simultaneously.

| Condition | Floor plan | GNSS mult | IMU mult | Patrol | Ped. prob | Bay occ. |
|-----------|-----------|-----------|----------|--------|-----------|----------|
| `worst_case` | `rectangle` | $250.0\times$ | $6.0\times$ | 0 | 1.0 | 0.0 |

## Metrics collected

| Metric | Source |
|--------|--------|
| Success rate | `info["success"]` from `CARLAParkingEnv.step()` |
| Mean episode reward | Accumulated per episode |
| Mean steps to termination | Episode length |
| Epistemic uncertainty | `get_action_with_uncertainty()` mean over episode (evidential only) |
| Aleatoric uncertainty | `get_action_with_uncertainty()` mean over episode (evidential only) |

**Success criteria** (from `configs/eval_config.yaml` `success_criteria`):

```math
\text{position error} < 0.5\,\text{m}
\qquad
\text{orientation error} < 10\,\text{deg}
\qquad
\text{speed} < 0.1\,\text{m/s}
```

## Key interfaces

```python
from uncertainty_rl.evaluation import (
    EvaluationMetrics,
    evaluate_agent,
    evaluate_across_conditions,
    plot_evaluation_results,
)

metrics = evaluate_across_conditions(
    model_path="checkpoints/final_model",
    eval_config=eval_cfg,
    env_config=env_cfg,
    train_config=train_cfg,
)
# metrics: List[EvaluationMetrics], one per condition
plot_evaluation_results(metrics, output_dir="evaluation_results/")
```

Run via Make:

```bash
make docker-eval          # Full 9-condition sweep (100 episodes per condition)
make eval-visualise-2d    # Detachable 2D bird's-eye replay after evaluation
```

## Configuration keys consumed

| Config file | Keys |
|-------------|------|
| `configs/eval_config.yaml` | `model_path`, `carla_host`, `carla_port`, `n_episodes`, `deterministic`, `eval_conditions`, `output_dir`, `success_criteria.*` |
| `configs/deployment/sim/env_config.yaml` | `carla_sensors.gnss.*`, `carla_sensors.imu.*` (base noise, scaled by condition multipliers) |
| `configs/train_config.yaml` | `policy_type` (selects evidential vs standard path for uncertainty logging) |
| `uncertainty_rl/utils/constants.py` | `SUCCESS_THRESHOLD_VELOCITY`, `SUCCESS_BAY_MARGIN` (success position/orientation are tested geometrically via `car_fully_inside_bay()` in `utils/geometry.py`) |

<!-- gif:placeholder name="eval_degradation" caption="Success rate and epistemic uncertainty across the 9 evaluation conditions" -->
![Evaluation degradation placeholder](docs/media/eval_degradation.gif)

## See also

- [uncertainty_rl/README.md](../README.md) - package overview
- [training/README.md](../training/README.md) - training the model evaluated here
- [networks/README.md](../networks/README.md) - evidential policy providing uncertainty estimates
- [docs/detailed_notes/observation_space.md](../../docs/detailed_notes/observation_space.md) - observation design that underpins the evaluation metrics
