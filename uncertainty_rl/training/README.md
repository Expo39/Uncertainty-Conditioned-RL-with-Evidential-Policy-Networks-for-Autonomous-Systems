# training/

RL training scripts using Stable-Baselines3.

## Dockerfile

`Dockerfile` builds the **training** container: NVIDIA NGC PyTorch base with Stable-Baselines3, evidential networks, rclpy (ROS 2 Humble), and all Python dependencies from `pyproject.toml`. Orchestrated via `docker-compose.yml` at the project root. ROS 2 Humble is used because the NGC base is Ubuntu 22.04; DDS communication with the Jazzy ros2-bridge container is seamless.

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
- `total_timesteps`: 1,000,000
- `ros2.covariance_topic`: `/ekf_uncertainty/covariance`
- `carla_sensors.imu.noise_accel_stddev_*`: 0.1 (m/s^2)
- `carla_conditions.num_vehicles`: 20

See `configs/train_config.yaml` for the full parameter list.
