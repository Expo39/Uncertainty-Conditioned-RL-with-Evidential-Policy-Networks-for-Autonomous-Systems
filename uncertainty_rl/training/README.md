# training/

RL training scripts using Stable-Baselines3.

## Module: `train_ppo.py`

Config-driven PPO training loop with:

- **VecNormalize** for observation and reward normalisation
- **Checkpointing** saves model and VecNormalize statistics
- **TensorBoard** logging for training metrics
- All hyperparameters loaded from `configs/train_config.yaml`

### Usage

```bash
python uncertainty_rl/training/train_ppo.py \
    --config configs/train_config.yaml \
    --total-timesteps 1000000
```

### Current Status

Uses SB3's standard `MlpPolicy`. The evidential policy network is **not yet integrated** as a custom SB3 policy class - this is a planned integration step.

### Key Config Parameters (from `configs/train_config.yaml`)

- `learning_rate`: 0.0003
- `batch_size`: 64
- `n_steps`: 2048
- `n_epochs`: 10
- `net_arch`: [256, 256]
- `uncertainty_noise_std`: 0.1 (metres)
- `total_timesteps`: 1,000,000

See `configs/train_config.yaml` for the full parameter list.
