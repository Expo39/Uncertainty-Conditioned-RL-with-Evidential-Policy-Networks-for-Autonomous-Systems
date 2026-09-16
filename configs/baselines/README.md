# configs/baselines/

Override configs for the 2x2 ablation study. Each file contains only the keys
that differ from [`configs/train_config.yaml`](../train_config.yaml). All PPO
hyperparameters and environment settings are inherited unchanged.

## The 2x2 matrix

|  | Standard policy | Evidential policy |
|--|----------------|------------------|
| **No covariance in obs** | `vanilla_ppo` | `output_uncertainty` |
| **Covariance in obs**    | `input_uncertainty` | `full_method` |

| Baseline | `include_covariance` | `policy_type` | Obs dim | What it tests |
|----------|---------------------|---------------|---------|---------------|
| `vanilla_ppo` | false | standard | 10 | No uncertainty awareness at all |
| `input_uncertainty` | true | standard | 13 | Uncertainty in input only |
| `output_uncertainty` | false | evidential | 10 | Uncertainty in output only |
| `full_method` | true | evidential | 13 | Full contribution (both) |

A baseline may set only the four keys in `BASELINE_KEYS`, namely `baseline_name`,
`include_covariance`, `include_obstacle_obs` and `policy_type`. `apply_baseline()`
raises on anything else, so a baseline cannot silently alter a training hyperparameter.

All four use `include_obstacle_obs: true`, so the LiDAR feature block is never the
experimental variable. The active observation dimension is derived at runtime from these
flags by `compute_obs_dim()` in
[`uncertainty_rl/envs/_parking_core.py`](../../uncertainty_rl/envs/_parking_core.py),
which sums the structural constants in
[`uncertainty_rl/utils/constants.py`](../../uncertainty_rl/utils/constants.py).

## Running a baseline

A baseline is selected with the bare `BASELINE=<name>` variable, while `STAGE` and
`CHECKPOINT` pick the curriculum stage and resume point as usual.

```bash
make docker-train BASELINE=vanilla_ppo STAGE=1
make docker-train BASELINE=full_method STAGE=1
```

The full 2x2 ablation is run by training each cell in turn, per baseline and per seed,
through the curriculum. Output is nested by baseline under each of the separate
`checkpoints/`, `logs/` and `outputs/raw/bay_successes/` roots, each following the same
`<root>/<baseline>/<leaf>/` pattern. Directory names are derived in code from
`baseline_name` and the base directories in `train_config.yaml`. See
[COMMANDS.md](../../COMMANDS.md) for the bare-name convention and
[USAGE.md](../../USAGE.md) for the full output tree.

## See also

- [configs/README.md](../README.md) - full configs directory map
- [configs/train_config.yaml](../train_config.yaml) - base hyperparameters inherited by all baselines
- [uncertainty_rl/training/README.md](../../uncertainty_rl/training/README.md) - training and tuning entry points
