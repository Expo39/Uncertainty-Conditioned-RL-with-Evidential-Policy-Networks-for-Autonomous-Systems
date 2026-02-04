# Quick Reference Guide

## Command Cheatsheet

### Installation
```bash
# Install package
pip install -e .

# Test installation
python test_installation.py
```

### Training
```bash
# Basic training
python src/uncertainty_rl/training/train_sac.py --config configs/train_config.yaml

# Training with custom parameters
python src/uncertainty_rl/training/train_sac.py \
    --config configs/train_config.yaml \
    --total-timesteps 2000000 \
    --checkpoint-dir ./my_checkpoints
```

### Evaluation
```bash
# Evaluate trained model
python src/uncertainty_rl/evaluation/evaluate.py \
    --model-path checkpoints/final_model \
    --config configs/eval_config.yaml

# Custom noise levels
python src/uncertainty_rl/evaluation/evaluate.py \
    --model-path checkpoints/final_model \
    --noise-levels 0.1 0.2 0.5 1.0 \
    --n-episodes 50
```

### ROS 2
```bash
# Start covariance extractor
python src/uncertainty_rl/ros2/covariance_extractor.py

# With custom topics
ros2 run uncertainty_rl covariance_extractor \
    --ros-args -p odom_topic:=/my_odom
```

### CARLA
```bash
# Start CARLA
cd /path/to/carla
./CarlaUE4.sh

# Headless mode
./CarlaUE4.sh -RenderOffScreen

# Custom port
./CarlaUE4.sh -carla-port=2001
```

## Key Classes and Functions

### Evidential Policy Network
```python
from uncertainty_rl.networks.evidential_policy import EvidentialPolicyNetwork

network = EvidentialPolicyNetwork(
    state_dim=15,
    action_dim=3,
    hidden_dims=[256, 256]
)

action, uncertainty = network.get_action(state)
```

### CARLA Environment
```python
from uncertainty_rl.envs.carla_parking import CARLAParkingEnv

env = CARLAParkingEnv(
    uncertainty_noise_std=0.1,
    max_steps=500
)

state, info = env.reset()
next_state, reward, terminated, truncated, info = env.step(action)
```

### Training
```python
from uncertainty_rl.training.train_sac import train

train(
    config_path="configs/train_config.yaml",
    total_timesteps=1000000
)
```

## Configuration Parameters

### Key Training Parameters
- `learning_rate`: 0.0003 (default)
- `buffer_size`: 1000000
- `batch_size`: 256
- `uncertainty_noise_std`: 0.1 (metres)

### Key Environment Parameters
- `max_steps`: 500 (episode length)
- `town`: "Town01" (CARLA map)
- `carla_port`: 2000

### Success Criteria
- Position error < 0.5m
- Orientation error < 10°
- Velocity < 0.1 m/s

## State Vector (15D)

1. **Position**: x, y, yaw
2. **Velocity**: vx, vy, vyaw
3. **Uncertainty (std)**: σx, σy, σyaw
4. **Covariance**: cov_xx, cov_yy, cov_yawyaw, cov_xy, cov_xyaw, cov_yyaw

## Action Vector (3D)

1. **Steering**: [-1.0, 1.0]
2. **Throttle**: [0.0, 1.0]
3. **Brake**: [0.0, 1.0]

## Directory Structure

```
checkpoints/      - Saved models
logs/            - TensorBoard logs
evaluation_results/ - Evaluation outputs
configs/         - Configuration files
examples/        - Example scripts
src/             - Source code
```

## Troubleshooting

### CARLA won't connect
- Check CARLA is running: `ps aux | grep Carla`
- Try different port: `--carla-port 2001`

### Out of memory
- Reduce `batch_size` in config
- Reduce `buffer_size` in config

### ROS 2 imports fail
- Source ROS 2: `source /opt/ros/humble/setup.bash`

## Useful Links

- [CARLA Documentation](https://carla.readthedocs.io/)
- [Stable-Baselines3 Docs](https://stable-baselines3.readthedocs.io/)
- [Gymnasium Docs](https://gymnasium.farama.org/)
- [ROS 2 Docs](https://docs.ros.org/)
