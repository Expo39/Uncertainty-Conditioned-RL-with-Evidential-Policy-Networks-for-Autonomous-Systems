# training/

RL training scripts using Stable-Baselines3.

## Module: `train_sac.py`

Config-driven SAC training loop with:

- **VecNormalize** for observation and reward normalisation
- **Checkpointing** saves model, replay buffer, and VecNormalize statistics
- **TensorBoard** logging for training metrics
- All hyperparameters loaded from `configs/train_config.yaml`

### Usage

```bash
python uncertainty_rl/training/train_sac.py \
    --config configs/train_config.yaml \
    --total-timesteps 1000000
```

### Current Status

Uses SB3's standard `MlpPolicy`. The evidential policy network is **not yet integrated** as a custom SB3 policy class — this is a planned integration step.

### Key Config Parameters

- `learning_rate`: 0.0003
- `batch_size`: 256
- `buffer_size`: 1,000,000
- `net_arch`: [256, 256]
- `uncertainty_noise_std`: 0.1 (metres)
