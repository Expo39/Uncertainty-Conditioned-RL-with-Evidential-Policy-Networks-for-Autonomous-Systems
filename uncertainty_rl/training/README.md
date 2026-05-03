# training/

RL training scripts for uncertainty-conditioned parking using Stable-Baselines3.

## Dockerfile

`Dockerfile` builds the **training** container: NVIDIA NGC PyTorch base (Ubuntu 22.04) with Stable-Baselines3, evidential networks, rclpy (ROS 2 Humble), and all Python dependencies from `pyproject.toml`. Orchestrated via `docker-compose.yml` at the project root. ROS 2 Humble matches the NGC Ubuntu 22.04 base; DDS wire protocol is distro-agnostic so the training container communicates with the Jazzy ros2-bridge container seamlessly.

## Modules

### `train_ppo.py`

Config-driven PPO training loop with:

- **Policy switching**: reads `policy_type` from config -`"evidential"` uses `EvidentialPPO` + `EvidentialActorCriticPolicy`; `"standard"` uses SB3 `PPO` + `MlpPolicy`
- **VecNormalize** wraps the environment for observation normalisation only (`norm_reward=False`)
- **Checkpointing** saves model and VecNormalize statistics together
- **TensorBoard** logging for training metrics (including evidential reg loss and uncertainty estimates)
- All hyperparameters loaded from `configs/train_config.yaml`
- **Returns `TrainResult`** dataclass with final metrics, model path, and log directory (enables Optuna integration)
- **Accepts `extra_callbacks`** for external callback integration (e.g., trial evaluation during Optuna studies)

### `tune_hyperparams.py`

Optuna hyperparameter tuning orchestrator with:

- **`sample_hyperparams()`**: Samples from YAML-driven search space (8 PPO + evidential parameters, all bounds configured in `configs/training/tuning_config.yaml`)
- **`TrialEvalCallback`**: SB3 callback that reads training metrics from `EnvDiagnosticsCallback` and reports to Optuna for pruning decisions
- **`objective()`**: Optuna objective function that runs a short training trial (150k steps default), evaluates `env/mean_progress_reward`, and handles CARLA crashes gracefully
- **`apply_best_params()`**: Writes best trial hyperparameters back to `configs/train_config.yaml`. Creates timestamped backup in `configs/backups/`
- **`run_study()`**: Creates and executes the Optuna study using `TPESampler` + `MedianPruner`, persists in SQLite for resumable tuning. Saves best params to `logs/tuning/results/best_params.yaml`
- **`main()`**: CLI entry point (run via `make docker-tune` or `uncertainty-rl-tune` console script)

### Usage

```bash
# Inside the training container (make docker-shell):
python uncertainty_rl/training/train_ppo.py \
    --train-config configs/train_config.yaml \
    --env-config configs/deployment/sim/env_config.yaml \
    --total-timesteps 1000000
```

### Policy Type

`train_ppo.py` reads `policy_type` from the config to select the agent:

| `policy_type` | Agent | Policy | Observation |
|---------------|-------|--------|-------------|
| `"evidential"` | `EvidentialPPO` | `EvidentialActorCriticPolicy` | 12-dim (full method) or 7-dim (cov off) |
| `"standard"` | `PPO` | `MlpPolicy` | 12-dim (obs on) or 7-dim (cov off) |

The `include_covariance` and `include_obstacle_obs` flags (from baseline YAML) control observation dimensionality. Default (both true): 12-dim. See `uncertainty_rl.utils.constants` for exact breakdowns.

### Key Config Parameters (from `configs/train_config.yaml`)

| Parameter | Value | Notes |
|-----------|-------|-------|
| `learning_rate` | 0.0003 | Linear decay to 0 applied in train\_ppo.py |
| `batch_size` | 256 | 40 off-policy steps per rollout (within CaRL-validated range) |
| `n_steps` | 2048 | ~4 episodes per PPO update (single CARLA env) |
| `n_epochs` | 5 | Fewer epochs to avoid policy drift |
| `net_arch` | [256, 256] | Both actor and critic |
| `target_kl` | 0.02 | Early epoch stopping to limit policy drift per update |
| `total_timesteps` | 1,000,000 | |
| `evidential.lambda_reg` | 0.01 | NIG regularisation coefficient |
| `parking_scenarios.*` | see YAML | Floor plan files, bay occupancy, cone spacing, NPC counts |

See `configs/train_config.yaml` for the full parameter list with per-parameter justifications.

## Hyperparameter Tuning (Optuna)

Optuna-based systematic search of 8 hyperparameters (PPO + evidential settings). Tuning bounds and study settings live in `configs/training/tuning_config.yaml`; best params are written back to `configs/train_config.yaml` after the study completes.

```bash
# 1. Edit tuning settings (optional): n_trials, timesteps_per_trial, seed
vim configs/training/tuning_config.yaml

# 2. Run tuning
make docker-tune

# 3. Best params are now in configs/train_config.yaml
# 4. Normal training uses tuned hyperparameters
make docker-train
```

### Search Space

All bounds and options are defined in `configs/training/tuning_config.yaml` (YAML-driven, no hardcoded ranges):

| Parameter | Range | Type |
|-----------|-------|------|
| `learning_rate` | [1e-5, 2e-3] | log scale |
| `n_steps` | {1024, 2048, 4096} | categorical |
| `batch_size` | {64, 128, 256} | categorical (constrained <= n_steps) |
| `n_epochs` | {3, 5, 10} | categorical |
| `gamma` | [0.97, 0.999] | log scale (via 1 - (1-gamma)) |
| `ent_coef` | [1e-6, 0.01] | log scale |
| `evidential.lambda_reg` | [1e-5, 0.01] | log scale |
| `evidential.lambda_reg_warmup_steps` | [10000, 100000] | log scale |

See `documentation/detailed_notes/hyperparameter_search.md` for search space design rationale.

### Study Configuration

Edit `configs/training/tuning_config.yaml`:

```yaml
study_name: "uncertainty_rl_tuning"
storage_path: "logs/tuning/optuna_study.db"
n_trials: 50                          # Number of trials to run
timesteps_per_trial: 150000          # Training steps per trial
seed: 42                              # Reproducibility
```

### Resume Interrupted Tuning

The SQLite storage (`logs/tuning/optuna_study.db`) persists across runs. To resume a paused study:

```bash
# Just re-run docker-tune — it will continue from the last completed trial
make docker-tune
```

## Ablation Study

The 2x2 ablation (4 baselines x N seeds) runs `train_ppo.py` once per baseline config. Each baseline YAML in `configs/baselines/` overrides only the keys that differ from `train_config.yaml` - all baselines share identical PPO hyperparameters.

```bash
make docker-train    # Single training run
```
