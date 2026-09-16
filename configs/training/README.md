# configs/training/

Optuna hyperparameter search configuration.

## Files

| File | Purpose |
|------|---------|
| `tuning_config.yaml` | Study settings, search space bounds, pruner and sampler config |

## `tuning_config.yaml`

Controls the Optuna study run by `make docker-tune`. After tuning completes, the best
trial's values are written back into
[`configs/train_config.yaml`](../train_config.yaml), with the original preserved under
`logs/tuning/backups/`.

```bash
make docker-tune STAGE=1 BASELINE=vanilla_ppo   # the recommended configuration
```

Stage 1 is the recommended tuning stage. Trials run from scratch on a budget of roughly
100k steps, and stage 1 is the only stage where that budget yields a non-zero
success-rate objective, leaving later stages with nothing but the progress tiebreaker.

The study tunes the stage-invariant structural PPO parameters alone. `learning_rate` and
`ent_coef` are deliberately excluded, since the per-stage override blocks would clobber
any value the tuner chose, making the search a no-op for those two.

The study configuration (study name, trial budget, evaluation metric, sampler
options) and the search-space bounds (per-parameter ranges, scales,
categoricals) live in `tuning_config.yaml`. Read that file directly for the
live settings rather than relying on a duplicate in this README - the values
change as the project iterates and the tuner writes back into
`train_config.yaml` whenever a study completes.

### Resuming a study

Re-running `make docker-tune` resumes from the last completed trial stored in the SQLite
database at `logs/tuning/optuna_study.db`, with no configuration change needed. Runs
that name a baseline suffix the filename, so the four arms keep separate studies.

## See also

- [configs/README.md](../README.md) - full configs directory map
- [configs/train_config.yaml](../train_config.yaml) - base training hyperparameters (updated by tuning)
- [docs/detailed_notes/training/hyperparameter_search.md](../../docs/detailed_notes/training/hyperparameter_search.md) - search space rationale
- [uncertainty_rl/training/README.md](../../uncertainty_rl/training/README.md) - training and tuning entry points
