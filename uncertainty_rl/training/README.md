# training/

PPO training loop and Optuna hyperparameter tuning for the uncertainty-conditioned parking agent.

## At a glance

- `train_ppo.py` is config-driven: all hyperparameters come from `configs/train_config.yaml`
- `policy_type: "evidential"` selects `EvidentialPPO` + `EvidentialActorCriticPolicy`; `"standard"` uses SB3 `PPO` + `MlpPolicy`
- `VecNormalize` wraps the environment with observation and reward normalisation (`norm_obs=True`, `norm_reward=True`)
- $\lambda_{\text{reg}}$ is linearly annealed from $0$ to $0.001$ over the first $50\,000$ steps
- Learning rate decays linearly: $\alpha(t) = \alpha_0 \cdot (1 - t / T)$
- Optuna TPE + MedianPruner study over 10 parameters; 40 trials x $100\,000$ steps each, optimising `env/success_rate` with `env/mean_progress_reward` as a tiebreaker
- No evaluation environment during training - two CARLA clients on a synchronous server deadlock

## Modules

| Module | Class / purpose |
|--------|----------------|
| `train_ppo.py` | `train()` - main training loop, config loading, VecNormalize, checkpointing, TensorBoard |
| `tune_hyperparams.py` | `run_study()` - Optuna study; `sample_hyperparams()`, `TrialEvalCallback`, `apply_best_params()` |
| `__init__.py` | Re-exports `train`, `load_config`, `load_env_config`, `merge_configs`, `TrainResult` |

## Internal data flow

```mermaid
flowchart TB
    subgraph cfg["configs/"]
        TC["train_config.yaml"]
        EC["deployment/sim/env_config.yaml"]
    end

    subgraph train["train_ppo.py"]
        ENV["CARLAParkingEnv\nVecNormalize"]
        PPO["EvidentialPPO\n(or PPO)"]
        CB["EvidentialLossCallback\nCheckpointCallback"]
    end

    subgraph out["outputs"]
        CKPT["checkpoints/\nmodel.zip + vec_normalize.pkl"]
        TB["logs/\nTensorBoard"]
    end

    TC -->|hyperparams| PPO
    EC -->|env settings| ENV
    ENV -->|obs + reward| PPO
    PPO -->|policy update| CB
    CB --> CKPT
    CB --> TB
```

## Key config parameters

From `configs/train_config.yaml`:

| Parameter | Value | Notes |
|-----------|-------|-------|
| `learning_rate` | $3 \times 10^{-4}$ | Linear decay to $0$ over $T$ steps |
| `n_steps` | $2048$ | Steps per PPO update (~4 episodes, single CARLA env) |
| `batch_size` | $256$ | Mini-batch size for gradient updates |
| `n_epochs` | $5$ | Gradient steps per rollout |
| `gamma` | $0.99$ | High $\gamma$: success bonus at ~step 300 must propagate back |
| `gae_lambda` | $0.95$ | GAE trace decay |
| `clip_range` | $0.2$ | PPO surrogate clip threshold $\epsilon$ |
| `ent_coef` | $0.02$ | Entropy bonus to break cold-start exploration (Tuner may revise) |
| `vf_coef` | $0.5$ | Value function loss weight |
| `max_grad_norm` | $0.5$ | Gradient clipping |
| `target_kl` | $0.02$ | Early-stop epochs when KL exceeds threshold |
| `total_timesteps` | $1\,000\,000$ | |
| `checkpoint_freq` | $50\,000$ | Checkpoint interval (steps) |
| `seed` | $42$ | Base seed; ablation uses `random_seeds` 0-9 |
| `evidential.lambda_reg` | $0.001$ | NIG reg coefficient (after warmup) |
| `evidential.lambda_reg_warmup_steps` | $50\,000$ | Linear annealing window |
| `evidential.use_uncertainty_conditioning` | `true` | Dual-encoder actor enabled |

## PPO objective

The clipped surrogate loss with value function and entropy terms:

```math
\mathcal{L}(\theta) = \mathbb{E}\!\left[
    \min\!\left(r_t(\theta)\hat{A}_t,\;
    \mathrm{clip}\!\left(r_t(\theta),\,1-\epsilon,\,1+\epsilon\right)\hat{A}_t\right)
    - c_1 \mathcal{L}_{\text{VF}} + c_2 \mathcal{H}[\pi_\theta]
\right]
```

where $r_t(\theta) = \pi_\theta(a_t | s_t) / \pi_{\theta_{\text{old}}}(a_t | s_t)$, $\epsilon = 0.2$, $c_1 = 0.5$, $c_2 = 0.005$.

Training stops early in each epoch if $\mathrm{KL}(\pi_{\theta_{\text{old}}} \,\|\, \pi_\theta) > 0.02$.

## Evidential regularisation annealing

$\lambda_{\text{reg}}$ ramps linearly from $0$ to avoid destabilising early training:

```math
\lambda_{\text{reg}}(t) = \lambda_{\text{reg}} \cdot \min\!\left(1,\; \frac{t}{t_{\text{warmup}}}\right)
\qquad
\lambda_{\text{reg}} = 0.001,\quad t_{\text{warmup}} = 50\,000
```

Total loss at each update:

```math
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{PPO}} + \lambda_{\text{reg}}(t) \cdot \mathcal{L}_{\text{reg}}
```

## Policy type switching

| `policy_type` | Agent | Policy | Obs consumed by actor |
|---------------|-------|--------|----------------------|
| `"evidential"` | `EvidentialPPO` | `EvidentialActorCriticPolicy` | $obs[0\!:\!2]$ (speed, vyaw) + $obs[2\!:\!5]$ (std) in dual-encoder mode |
| `"standard"` | `PPO` | `MlpPolicy` | Full obs |

`include_covariance` and `include_obstacle_obs` flags (set per baseline) control obs dimensionality:
$13$ (both on), $10$ (covariance off), $8$ (obstacle off), $5$ (both off).

## Key interfaces

```python
from uncertainty_rl.training import train, load_config, load_env_config, merge_configs, TrainResult

cfg = merge_configs(
    load_config("configs/train_config.yaml"),
    load_env_config("configs/deployment/sim/env_config.yaml"),
)
result: TrainResult = train(cfg)
# result.model_path, result.log_dir, result.final_metrics
```

Training via Make:

```bash
make docker-train          # Full 1 M-step run (evidential, full method)
make docker-train-short    # Short smoke-test run
make docker-tune           # Optuna hyperparameter search
```

## Hyperparameter tuning (Optuna)

TPE sampler + MedianPruner study. Settings from `configs/training/tuning_config.yaml`:

| Setting | Value |
|---------|-------|
| `study_name` | `"uncertainty_rl_tuning"` |
| `n_trials` | $40$ |
| `timesteps_per_trial` | $100\,000$ |
| `eval_metric` | `"env/success_rate"` (with `env/mean_progress_reward` tiebreaker) |
| `seed` | $42$ |

### Search space

All bounds are defined in `configs/training/tuning_config.yaml`:

| Parameter | Range | Scale |
|-----------|-------|-------|
| `learning_rate` | $[10^{-5},\, 10^{-3}]$ | log |
| `n_steps` | $\{1024, 2048, 4096\}$ | categorical |
| `batch_size` | $\{64, 128, 256\}$ (constrained $\le$ `n_steps`) | categorical |
| `n_epochs` | $\{3, 5, 10\}$ | categorical |
| `gamma` | $[0.98,\, 0.999]$ | log (via $1-(1-\gamma)$) |
| `gae_lambda` | $[0.90,\, 0.98]$ | linear |
| `clip_range` | $[0.1,\, 0.3]$ | linear |
| `ent_coef` | $[10^{-6},\, 5 \times 10^{-2}]$ | log |
| `evidential.lambda_reg` | $[10^{-5},\, 10^{-2}]$ | log |
| `evidential.lambda_reg_warmup_steps` | $[10\,000,\, 100\,000]$ | log |

The primary objective is `env/success_rate` from `EnvDiagnosticsCallback`. Until any episode succeeds, trials are ranked by a scaled `env/mean_progress_reward` tiebreaker so early non-successful trials remain comparable. After the study completes, `apply_best_params()` writes the winning values back to `configs/train_config.yaml` and creates a timestamped backup in `logs/tuning/backups/`.

```bash
# Resume a paused study - just re-run; SQLite persists trial history
make docker-tune
```

## Ablation study

The $2 \times 2$ ablation runs `train_ppo.py` once per baseline config. Each file in `configs/baselines/` overrides only the keys that differ from `train_config.yaml`:

| Baseline | `policy_type` | `include_covariance` | Obs dim |
|----------|--------------|----------------------|---------|
| `vanilla_ppo` | `standard` | `false` | $9$ |
| `input_uncertainty` | `standard` | `true` | $12$ |
| `output_uncertainty` | `evidential` | `false` | $9$ |
| `full_method` | `evidential` | `true` | $12$ |

All four baselines share identical PPO hyperparameters from `train_config.yaml`.

## Configuration keys consumed

| Config file | Keys |
|-------------|------|
| `configs/train_config.yaml` | All PPO hyperparameters, `net_arch`, `activation`, `evidential.*`, `policy_type`, `total_timesteps`, `seed`, `checkpoint_freq` |
| `configs/deployment/sim/env_config.yaml` | `carla_host`, `carla_port`, `max_steps`, `include_covariance`, `include_obstacle_obs`, `carla_sensors.*`, `parking_scenarios.*` |
| `configs/training/tuning_config.yaml` | `study_name`, `n_trials`, `timesteps_per_trial`, `seed`, search space bounds |
| `uncertainty_rl/utils/constants.py` | `TOTAL_OBS_DIM` (12), `ACTION_DIM` (2), `VEHICLE_STATE_DIM` (1), `COVARIANCE_FEATURES_DIM` (3) |

<!-- gif:placeholder name="training_curves" caption="PPO reward and evidential uncertainty metrics over 1 M steps" -->
![Training curves placeholder](docs/media/training_curves.gif)

## See also

- [uncertainty_rl/README.md](../README.md) - package overview
- [networks/README.md](../networks/README.md) - EvidentialPPO and EvidentialActorCriticPolicy
- [evaluation/README.md](../evaluation/README.md) - evaluation after training completes
- [docs/detailed_notes/hyperparameter_search.md](../../docs/detailed_notes/hyperparameter_search.md) - search space design rationale
- [docs/detailed_notes/evidential_nig_initialisation.md](../../docs/detailed_notes/evidential_nig_initialisation.md) - NIG init and regularisation derivation
