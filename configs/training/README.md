# configs/training/

Optuna hyperparameter search configuration.

> **This search was never run.** Every reported result uses the committed defaults in
> [`configs/train_config.yaml`](../train_config.yaml). Tuning per arm would make the
> configuration an extra variable and confound the 3x2 ablation, so hyperparameters are
> fixed across every arm and all seeds. The pipeline below is retained for future
> work and produced no reported result. Section 3.8.1 of
> [the dissertation](../../docs/AntonioGaldes_Dissertation.pdf) is canonical.

## Files

| File | Purpose |
|------|---------|
| `tuning_config.yaml` | Study settings, search space bounds, pruner and sampler config |

## `tuning_config.yaml`

Would control the Optuna study run by `make docker-tune`.

```bash
make docker-tune STAGE=1 BASELINE=vanilla_ppo
```

`make docker-tune` consumes `STAGE` and `BASELINE` (`LAYOUT` is echoed only - the floor
plan comes from the stage file). It brings the stack down, back up, then runs
`scripts/training/tune.sh`, which passes `--stage N` and `--baseline
configs/baselines/<name>.yaml` through to `uncertainty_rl.training.tune_hyperparams`.
Omitting `STAGE` defaults to stage 1; omitting `BASELINE` tunes at the full-method
observation space.

Stage 1 is the recommended tuning stage. Trials run from scratch on the
`timesteps_per_trial` budget (100k), and stage 1 is the only stage where that budget
yields a non-zero success-rate objective, leaving later stages with nothing but the
progress tiebreaker.

### Where the best trial is written

This depends on whether `--baseline` was given explicitly:

| Invocation | Write-back target | `train_config.yaml` |
|------------|-------------------|---------------------|
| No `BASELINE` | `train_config.yaml` via `apply_best_params()`, original backed up to `logs/tuning/backups/` | Overwritten |
| `BASELINE=<name>` | `logs/tuning/results/best_params_<name>.yaml` only | Left untouched |

The per-baseline case deliberately leaves the shared file alone, since several arms writing
into one file would overwrite each other. Every run also saves a standalone copy of the
best params under `logs/tuning/results/`.

### What is searched

`sample_hyperparams()` samples exactly eight structural PPO parameters, with the bounds
read from `search_space:` in `tuning_config.yaml`:

| Parameter | Default bounds | Sampling |
|-----------|----------------|----------|
| `n_steps` | {1024, 2048, 4096} | categorical |
| `batch_size` | {64, 128, 256} | categorical, prunes the trial if > `n_steps` |
| `n_epochs` | {3, 5, 10} | categorical |
| `gamma` | [0.98, 0.999] | log scale on `one_minus_gamma` |
| `gae_lambda` | [0.90, 0.98] | uniform |
| `clip_range` | [0.1, 0.3] | uniform |
| `vf_coef` | [0.25, 1.0] | uniform |
| `max_grad_norm` | [0.3, 2.0] | uniform |

Excluded on purpose:
- `learning_rate` and `ent_coef` are stage-owned schedules set through
  `standard_overrides` / `evidential_overrides`, so tuning them here is a no-op.
- `net_arch`, `activation` and `policy_type` are architectural and locked across stages -
  changing them stops saved weights loading on resume.
- `evidential.*` is ablation-specific and cannot be tuned on the vanilla config without
  breaking fairness between arms.

### Study settings

| Key | Value | Meaning |
|-----|-------|---------|
| `study_name` | `uncertainty_rl_tuning` | Suffixed with the baseline name for a per-baseline run |
| `storage_path` | `logs/tuning/optuna_study.db` | SQLite; filename suffixed per baseline |
| `n_trials` | 40 | Trial budget |
| `timesteps_per_trial` | 100000 | Per-trial training budget |
| `tuning_seed` | 777 | TPE sampler seed; required, no fallback |
| `sampler.multivariate` | true | TPESampler captures parameter interactions |
| `sampler.n_startup_trials` | 12 | Random exploration before TPE models the density |
| `pruner.n_startup_trials` | 8 | MedianPruner: first 8 trials run to completion |
| `pruner.n_warmup_steps` | 5 | Rollouts before pruning may trigger |
| `pruner.n_min_trials` | 5 | Completed trials required before pruning any |

The objective maximises `env/success_rate`, falling back to `1e-3 *
env/mean_progress_reward` as a tiebreaker while no success has been observed. Trial
output nests as `logs/tuning/<baseline>/trial_<N>/` and
`checkpoints/tuning/<baseline>/trial_<N>/`.

`tuning_config.yaml` is the live source for all of the above; read it directly if the
tables here and the file disagree.

### Resuming a study

Re-running `make docker-tune` resumes from the last completed trial stored in the SQLite
database at `logs/tuning/optuna_study.db`, with no configuration change needed. Runs
that name a baseline suffix the filename, so every arm keeps a separate study.

## See also

- [configs/README.md](../README.md) - full configs directory map
- [configs/train_config.yaml](../train_config.yaml) - the committed hyperparameters every reported result uses
- [docs/detailed_notes/training/hyperparameter_search.md](../../docs/detailed_notes/training/hyperparameter_search.md) - search space rationale and references
- [docs/detailed_notes/training/ablation_hpo_methodology.md](../../docs/detailed_notes/training/ablation_hpo_methodology.md) - why no HPO was run for the ablation
- [uncertainty_rl/training/README.md](../../uncertainty_rl/training/README.md) - training and tuning entry points
