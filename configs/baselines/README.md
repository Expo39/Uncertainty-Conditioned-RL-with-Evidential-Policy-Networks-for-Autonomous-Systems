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

## Running a baseline

A baseline is selected with the bare `BASELINE=<name>` variable; `STAGE` and
`CHECKPOINT` pick the curriculum stage and resume point as usual.

```bash
make docker-train BASELINE=vanilla_ppo STAGE=1
make docker-train BASELINE=full_method STAGE=1
```

The full 2x2 ablation is run by training each cell in turn (per baseline, per seed,
through the curriculum). Output is nested by baseline: checkpoints, logs, and
`bay_successes/` land under `<root>/<baseline>/<leaf>/`, where `<leaf>` is
`seed<N>_<DDMMYYYY-HHMM>`. The directory names are derived in code from `baseline_name`
plus the base dirs in `train_config.yaml`; you only ever pass the bare `BASELINE` and
`CHECKPOINT` names.

## See also

- [configs/README.md](../README.md) - full configs directory map
- [configs/train_config.yaml](../train_config.yaml) - base hyperparameters inherited by all baselines
- [uncertainty_rl/training/README.md](../../uncertainty_rl/training/README.md) - training and tuning entry points
