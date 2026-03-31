# training/

RL training scripts for uncertainty-conditioned parking using Stable-Baselines3.

## Dockerfile

`Dockerfile` builds the **training** container: NVIDIA NGC PyTorch base (Ubuntu 22.04) with Stable-Baselines3, evidential networks, rclpy (ROS 2 Humble), and all Python dependencies from `pyproject.toml`. Orchestrated via `docker-compose.yml` at the project root. ROS 2 Humble matches the NGC Ubuntu 22.04 base; DDS wire protocol is distro-agnostic so the training container communicates with the Jazzy ros2-bridge container seamlessly.

## Module: `train_ppo.py`

Config-driven PPO training loop with:

- **Policy switching**: reads `policy_type` from config -`"evidential"` uses `EvidentialPPO` + `EvidentialActorCriticPolicy`; `"standard"` uses SB3 `PPO` + `MlpPolicy`
- **VecNormalize** wraps the environment for observation and reward normalisation
- **Checkpointing** saves model and VecNormalize statistics together
- **TensorBoard** logging for training metrics (including evidential reg loss and uncertainty estimates)
- All hyperparameters loaded from `configs/train_config.yaml`

### Usage

```bash
# Inside the training container (make docker-shell):
python uncertainty_rl/training/train_ppo.py \
    --config configs/train_config.yaml \
    --total-timesteps 1000000
```

### Policy Type

`train_ppo.py` reads `policy_type` from the config to select the agent:

| `policy_type` | Agent | Policy | Observation |
|---------------|-------|--------|-------------|
| `"evidential"` | `EvidentialPPO` | `EvidentialActorCriticPolicy` | 20-dim (full method) or 11-dim (output\_uncertainty) |
| `"standard"` | `PPO` | `MlpPolicy` | 20-dim (input\_uncertainty) or 11-dim (vanilla\_ppo) |

The `include_covariance` and `include_obstacle_obs` flags (from baseline YAML) control observation dimensionality. `include_obstacle_obs: true` in all 4 baselines so obstacle dims are not the experimental variable.

### Key Config Parameters (from `configs/train_config.yaml`)

| Parameter | Value | Notes |
|-----------|-------|-------|
| `learning_rate` | 0.0003 | Linear decay to 0 applied in train\_ppo.py |
| `batch_size` | 256 | 40 off-policy steps per rollout (within CaRL-validated range) |
| `n_steps` | 2048 | ~4 episodes per PPO update (single CARLA env) |
| `n_epochs` | 5 | Fewer epochs to avoid policy drift |
| `net_arch` | [256, 256] | Both actor and critic |
| `target_kl` | 0.02 | Early epoch stopping; 0.02 for noisy weather-randomised landscape |
| `total_timesteps` | 1,000,000 | |
| `evidential.lambda_reg` | 0.01 | NIG regularisation coefficient |
| `sensor_suite` | suite\_a | 2D LiDAR + IMU |
| `parking_scenarios.*` | see YAML | Floor plan files, bay occupancy, cone spacing, NPC counts |

See `configs/train_config.yaml` for the full parameter list with per-parameter justifications.

## Ablation Study Orchestration: `scripts/run_experiment.py`

Runs the full 2x2 ablation study (4 baselines x 10 seeds). Each baseline config in `configs/baselines/` overrides only the keys that differ from `train_config.yaml` -all baselines share identical PPO hyperparameters.

```bash
make experiment-dry       # Plan runs without training (no Docker needed)
make docker-experiment    # Full ablation inside container
```
