# configs/baselines/

Override configs for the 3x2 ablation study (actor head x covariance input). Each
file sets only the four ablation keys; every PPO hyperparameter comes from
[`configs/train_config.yaml`](../train_config.yaml) and every environment setting from
the merged env config plus the curriculum stage file.

One fixed hyperparameter configuration is applied identically to all six arms and all
three seeds (42, 123, 7). No hyperparameter optimisation was run: tuning per arm would
make the configuration an extra variable and confound the ablation.

The one deliberate exception is the entropy schedule:

- Each curriculum stage file carries two override blocks, `standard_overrides` and
  `evidential_overrides`, selected by `policy_type` in
  `_apply_stage_training_overrides()`.
- The blocks are identical apart from `ent_coef` / `ent_coef_final`, held slightly higher
  for the evidential and heteroscedastic arms, whose action std is an actor output.
  `standard` selects `standard_overrides`; `evidential` and `heteroscedastic` select
  `evidential_overrides`; any other `policy_type` raises `ValueError`.
- The split lives in the stage files, not in these baseline files, which cannot set a
  hyperparameter at all.

## The 3x2 matrix

|  | Standard policy | Heteroscedastic policy | Evidential policy |
|--|----------------|------------------------|------------------|
| **No covariance in obs** | `vanilla_ppo` | `heteroscedastic` | `output_uncertainty` |
| **Covariance in obs**    | `input_uncertainty` | `heteroscedastic_input` | `full_method` |

The three actor heads differ only in how the action variance is produced:

- **Standard** (`LayerNormActorCriticPolicy`): one state-independent `log_std` parameter.
- **Heteroscedastic** (`HeteroscedasticActorCriticPolicy`): a state-dependent `log_std`
  output, with the same backbone, initial variance, clamp and squashing as the
  evidential head, trained with the plain PPO loss (no NIG prior anchor).
- **Evidential** (`EvidentialActorCriticPolicy`): NIG `(gamma, nu, alpha, beta)`, action
  variance `beta / (alpha - 1)`, trained with the PPO loss plus the prior anchor.

| File | `baseline_name` | `include_covariance` | `include_obstacle_obs` | `policy_type` | Obs dim | What it tests |
|------|-----------------|---------------------|------------------------|---------------|---------|---------------|
| `vanilla_ppo.yaml` | `vanilla_ppo` | false | true | `standard` | 10 | No uncertainty awareness at all |
| `input_uncertainty.yaml` | `input_uncertainty` | true | true | `standard` | 13 | Uncertainty in input only |
| `heteroscedastic.yaml` | `heteroscedastic` | false | true | `heteroscedastic` | 10 | State-dependent action variance only |
| `heteroscedastic_input.yaml` | `heteroscedastic_input` | true | true | `heteroscedastic` | 13 | Input uncertainty with state-dependent action variance |
| `output_uncertainty.yaml` | `output_uncertainty` | false | true | `evidential` | 10 | Uncertainty in output only |
| `full_method.yaml` | `full_method` | true | true | `evidential` | 13 | Full contribution (both) |

A baseline may set only the four keys in the `BASELINE_KEYS` frozenset in
[`uncertainty_rl/utils/config_merge.py`](../../uncertainty_rl/utils/config_merge.py):
`baseline_name`, `include_covariance`, `include_obstacle_obs`, `policy_type`.

`apply_baseline()` raises `ValueError` on any other key, so a baseline cannot silently
alter a training hyperparameter or an env setting. Omitting `--baseline` defaults to
`full_method.yaml`.

All six use `include_obstacle_obs: true`, so the LiDAR feature block is never the
experimental variable. The active observation dimension is derived at runtime by
`compute_obs_dim()` in
[`uncertainty_rl/envs/_parking_core.py`](../../uncertainty_rl/envs/_parking_core.py),
which sums the structural constants in
[`uncertainty_rl/utils/constants.py`](../../uncertainty_rl/utils/constants.py):

| Block | Constant | Size | Gate |
|-------|----------|------|------|
| Vehicle state | `VEHICLE_STATE_DIM` | 2 | always |
| Relative target pose | `TARGET_POSE_DIM` | 3 | always |
| EKF covariance | `COVARIANCE_FEATURES_DIM` | 3 | `include_covariance` |
| LiDAR clearance | `OBSTACLE_FEATURES_DIM` | 5 | `include_obstacle_obs` |

Base is 2 + 3 = 5, giving 13 / 10 / 8 / 5 across the four flag combinations; only 13 and
10 occur here, since all six arms keep the obstacle block on.

`policy_type` and the two obs flags are architectural, so they must stay constant across
curriculum stages or saved weights cannot load. The stage override allowlist in
`train_ppo.py` rejects them for exactly that reason.

## Running a baseline

A baseline is selected with the bare `BASELINE=<name>` variable (never a path - the
recipe expands it to `configs/baselines/<name>.yaml`), while `STAGE` and `CHECKPOINT`
pick the curriculum stage and the resume leaf.

```bash
# Stage 1, fresh run
make docker-train BASELINE=vanilla_ppo STAGE=1 LAYOUT=rectangle
make docker-train BASELINE=full_method STAGE=1 LAYOUT=rectangle

# Stage 2, resuming that arm's stage 1 leaf
make docker-train BASELINE=full_method STAGE=2 LAYOUT=rectangle CHECKPOINT=1_42_11062026-0628
```

The seed is not a make variable for training: it is read from `seed:` in
[`configs/deployment/agent_config.yaml`](../deployment/agent_config.yaml), so a new seed
means editing that one line before launching the arm.

The full 3x2 ablation is run by training each cell in turn, per baseline and per seed,
through the six curriculum stages. Directory names are derived in code from
`baseline_name` and the base `log_dir` / `checkpoint_dir` in `train_config.yaml`, never
set per baseline:

| Root | Pattern |
|------|---------|
| `checkpoints/` | `checkpoints/<baseline>/<leaf>/` |
| `logs/` | `logs/<baseline>/<leaf>/` |
| `outputs/raw/bay_successes/training/` | `.../training/seed_<N>/<baseline>/<leaf>/` |

`<leaf>` is `<stage>_<seed>_<DDMMYYYY-HHMM>` for a training run, or `trial_<N>` under
Optuna. See [COMMANDS.md](../../COMMANDS.md) for the bare-name convention and
[USAGE.md](../../USAGE.md) for the full output tree.

## See also

- [configs/README.md](../README.md) - full configs directory map
- [configs/train_config.yaml](../train_config.yaml) - base hyperparameters inherited by all baselines
- [uncertainty_rl/training/README.md](../../uncertainty_rl/training/README.md) - training and tuning entry points
