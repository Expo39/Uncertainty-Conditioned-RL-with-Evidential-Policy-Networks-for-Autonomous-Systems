# configs/baselines/

Override configs for the 2x2 ablation study. Each file contains only the keys
that differ from [`configs/train_config.yaml`](../train_config.yaml). All PPO
hyperparameters and environment settings are inherited unchanged.

## The 2x2 matrix

|  | Standard policy | Evidential policy |
|--|----------------|------------------|
| **No covariance in obs** | `vanilla_ppo` | `output_uncertainty` |
| **Covariance in obs**    | `input_uncertainty` | `full_method` |

| Baseline | `include_covariance` | `policy_type` | What it tests |
|----------|---------------------|---------------|---------------|
| `vanilla_ppo` | false | standard | No uncertainty awareness at all |
| `input_uncertainty` | true | standard | Uncertainty in input only |
| `output_uncertainty` | false | evidential | Uncertainty in output only |
| `full_method` | true | evidential | Full contribution (both) |

All four baselines use `include_obstacle_obs: true` so the LiDAR feature block
is never the experimental variable. The active observation dimension is
derived at runtime from these flags via `compute_obs_dim()` -see
[`uncertainty_rl/utils/constants.py`](../../uncertainty_rl/utils/constants.py)
for the structural constants.

## Running a single baseline

```bash
make docker-train BASELINE=vanilla_ppo
make docker-train BASELINE=full_method
```

## Running the full ablation

```bash
make docker-experiment       # Full 4-baseline x N-seed sweep
make docker-experiment-dry   # Dry-run: print what would run without training
```

Results are written to `logs/<baseline_name>_seed<N>_<timestamp>/` and checkpoints
to `checkpoints/<baseline_name>_seed<N>_<timestamp>/` (derived from `baseline_name`
plus the base dirs in `train_config.yaml`).

## See also

- [configs/README.md](../README.md) - full configs directory map
- [configs/train_config.yaml](../train_config.yaml) - base hyperparameters inherited by all baselines
- [uncertainty_rl/training/README.md](../../uncertainty_rl/training/README.md) - training and tuning entry points
