# configs/baselines/

Override configs for the 2x2 ablation study. Each file contains only the keys that differ
from `configs/train_config.yaml`. All PPO hyperparameters and environment settings are
inherited unchanged.

## The 2x2 matrix

|  | Standard policy | Evidential policy |
|--|----------------|------------------|
| **No covariance in obs** (9-dim) | `vanilla_ppo` | `output_uncertainty` |
| **Covariance in obs** (12-dim) | `input_uncertainty` | `full_method` |

| Baseline | `include_covariance` | `policy_type` | Obs dim | What it tests |
|----------|---------------------|---------------|---------|---------------|
| `vanilla_ppo` | false | standard | 9 | No uncertainty awareness at all |
| `input_uncertainty` | true | standard | 12 | Uncertainty in input only |
| `output_uncertainty` | false | evidential | 9 | Uncertainty in output only |
| `full_method` | true | evidential | 12 | Full contribution (both) |

All four baselines use `include_obstacle_obs: true` so the 5 LiDAR dims are never
the experimental variable.

## Running a single baseline

```bash
make docker-train BASELINE=vanilla_ppo
make docker-train BASELINE=full_method
```

## Running the full ablation (4 baselines x 10 seeds)

```bash
make docker-experiment       # ~15-18 hours per seed on RTX 4070 Ti Super
make docker-experiment-dry   # Dry-run: print what would run without training
```

Results are written to `logs/<baseline_name>/seed_<N>/` and checkpoints to
`checkpoints/<baseline_name>/seed_<N>/`.

## See also

- [configs/README.md](../README.md) - full configs directory map
- [configs/train_config.yaml](../train_config.yaml) - base hyperparameters inherited by all baselines
- [uncertainty_rl/training/README.md](../../uncertainty_rl/training/README.md) - training and tuning entry points
