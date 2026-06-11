# configs/training/

Optuna hyperparameter search configuration.

## Files

| File | Purpose |
|------|---------|
| `tuning_config.yaml` | Study settings, search space bounds, pruner and sampler config |

## `tuning_config.yaml`

Controls the Optuna study run by `make docker-tune`. After tuning completes,
the best trial's values are written automatically back into
[`configs/train_config.yaml`](../train_config.yaml).

```bash
make docker-tune         # Run the study
make docker-tune-dry     # Print what would run without training
```

The study configuration (study name, trial budget, evaluation metric, sampler
options) and the search-space bounds (per-parameter ranges, scales,
categoricals) live in `tuning_config.yaml`. Read that file directly for the
live settings rather than relying on a duplicate in this README - the values
change as the project iterates and the tuner writes back into
`train_config.yaml` whenever a study completes.

### Resuming a study

Re-running `make docker-tune` resumes from the last completed trial stored in
the SQLite database at `logs/tuning/optuna_study.db`. No configuration change
needed.

## See also

- [configs/README.md](../README.md) - full configs directory map
- [configs/train_config.yaml](../train_config.yaml) - base training hyperparameters (updated by tuning)
- [docs/detailed_notes/training/hyperparameter_search.md](../../docs/detailed_notes/training/hyperparameter_search.md) - search space rationale
- [uncertainty_rl/training/README.md](../../uncertainty_rl/training/README.md) - training and tuning entry points
