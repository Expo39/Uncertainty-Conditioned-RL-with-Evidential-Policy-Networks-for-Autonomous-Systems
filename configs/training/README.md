# configs/training/

Optuna hyperparameter search configuration.

## Files

| File | Purpose |
|------|---------|
| `tuning_config.yaml` | Study settings, search space bounds, pruner and sampler config |

## `tuning_config.yaml`

Controls the Optuna study run by `make docker-tune`. After tuning completes, the best
trial's values are written automatically back into `configs/train_config.yaml`.

```bash
make docker-tune         # Run the study
make docker-tune-dry     # Print what would run without training
```

### Key settings

| Key | Value | Notes |
|-----|-------|-------|
| `study_name` | `"uncertainty_rl_tuning"` | SQLite storage key. |
| `n_trials` | `35` | Trial budget for the 8-parameter space. |
| `timesteps_per_trial` | `100000` | ~49 PPO updates per trial - enough to rank configs. |
| `eval_metric` | `"env/mean_progress_reward"` | Non-zero even when the agent rarely succeeds early on. |
| `sampler.multivariate` | `true` | Captures cross-parameter interactions (e.g. learning rate vs batch size). |

### Search space

| Parameter | Range | Scale |
|-----------|-------|-------|
| `learning_rate` | 1e-5 to 1e-3 | Log |
| `n_steps` | 1024, 2048, 4096 | Categorical |
| `batch_size` | 64, 128, 256 | Categorical |
| `n_epochs` | 3, 5, 10 | Categorical |
| `gamma` | 0.98 to 0.999 | Log (via 1-(1-gamma)) |
| `ent_coef` | 1e-6 to 0.01 | Log |
| `lambda_reg` | 1e-5 to 0.01 | Log |
| `lambda_reg_warmup_steps` | 10000 to 100000 | Log |

### Resuming a study

Re-running `make docker-tune` resumes from the last completed trial stored in the SQLite
database at `logs/tuning/optuna_study.db`. No configuration change needed.

## See also

- [configs/README.md](../README.md) - full configs directory map
- [configs/train_config.yaml](../train_config.yaml) - base training hyperparameters (updated by tuning)
- [documentation/detailed_notes/hyperparameter_search.md](../../documentation/detailed_notes/hyperparameter_search.md) - search space rationale
- [uncertainty_rl/training/README.md](../../uncertainty_rl/training/README.md) - training and tuning entry points
