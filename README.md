# Uncertainty-Conditioned RL with Evidential Policy Networks for Autonomous Vehicles

Uncertainty-Conditioned Reinforcement Learning for Autonomous Parking: Propagating SLAM localisation uncertainty through evidential deep learning policies for safer control in unmapped environments.

## Overview

This project implements an uncertainty-aware reinforcement learning system for autonomous parking that:

1. **Gymnasium Environment**: Custom CARLA-based parking environment with SLAM uncertainty in state representation
2. **Evidential Policy Networks**: PyTorch implementation of evidential deep learning for epistemic and aleatoric uncertainty quantification
3. **SAC Training**: Integration with Stable-Baselines3 for training with Soft Actor-Critic algorithm
4. **Comprehensive Evaluation**: Tools for evaluating performance across varying uncertainty levels
5. **ROS 2 Integration**: Node for extracting covariance from robot_localization

## Features

- **Modular Architecture**: Clean separation of concerns with well-organised package structure
- **Type Hints**: Full type annotations throughout the codebase
- **British English**: Documentation and comments in British English (with standard code exceptions)
- **Uncertainty Quantification**: Separate tracking of epistemic (model) and aleatoric (data) uncertainty
- **Configuration-Driven**: YAML configurations for easy experimentation
- **Comprehensive Logging**: TensorBoard integration and custom metrics tracking

## Installation

### Prerequisites

- Python 3.8 or higher
- CARLA Simulator 0.9.13 or higher
- ROS 2 (Humble or Foxy) for robot_localization integration
- CUDA-capable GPU (recommended for training)

### Step 1: Clone the Repository

```bash
git clone https://github.com/Expo39/Uncertainty-Conditioned-RL-with-Evidential-Policy-Networks-for-AVs.git
cd Uncertainty-Conditioned-RL-with-Evidential-Policy-Networks-for-AVs
```

### Step 2: Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -e .
```

Or install from requirements.txt:

```bash
pip install -r requirements.txt
```

### Step 4: Install CARLA

Download and extract CARLA 0.9.13 or higher from [CARLA Releases](https://github.com/carla-simulator/carla/releases).

Add CARLA Python API to your PYTHONPATH:

```bash
export PYTHONPATH=$PYTHONPATH:/path/to/carla/PythonAPI/carla/dist/carla-0.9.13-py3.7-linux-x86_64.egg
```

### Step 5: Install ROS 2 (Optional, for robot_localization integration)

Follow the [ROS 2 installation guide](https://docs.ros.org/en/humble/Installation.html) for your platform.

## Quick Start

### 1. Start CARLA Simulator

```bash
cd /path/to/carla
./CarlaUE4.sh
```

Or run in headless mode:

```bash
./CarlaUE4.sh -RenderOffScreen
```

### 2. Train the Agent

```bash
python src/uncertainty_rl/training/train_sac.py \
    --config configs/train_config.yaml \
    --total-timesteps 1000000 \
    --log-dir ./logs \
    --checkpoint-dir ./checkpoints
```

### 3. Evaluate the Trained Agent

```bash
python src/uncertainty_rl/evaluation/evaluate.py \
    --model-path checkpoints/final_model \
    --config configs/eval_config.yaml \
    --noise-levels 0.05 0.1 0.2 0.5 1.0 \
    --n-episodes 100 \
    --output-dir ./evaluation_results
```

### 4. Run ROS 2 Covariance Extractor (Optional)

```bash
# Source ROS 2
source /opt/ros/humble/setup.bash

# Run the node
python src/uncertainty_rl/ros2/covariance_extractor.py
```

Or with custom configuration:

```bash
ros2 run uncertainty_rl covariance_extractor --ros-args -p odom_topic:=/my_odom
```

## Project Structure

```
.
├── src/
│   └── uncertainty_rl/
│       ├── __init__.py
│       ├── envs/
│       │   ├── __init__.py
│       │   └── carla_parking.py        # CARLA parking environment
│       ├── networks/
│       │   ├── __init__.py
│       │   └── evidential_policy.py    # Evidential policy networks
│       ├── training/
│       │   ├── __init__.py
│       │   └── train_sac.py            # SAC training script
│       ├── evaluation/
│       │   ├── __init__.py
│       │   └── evaluate.py             # Evaluation script
│       ├── ros2/
│       │   ├── __init__.py
│       │   └── covariance_extractor.py # ROS 2 node
│       └── utils/
│           ├── __init__.py
│           └── logging.py              # Logging utilities
├── configs/
│   ├── train_config.yaml               # Training configuration
│   ├── eval_config.yaml                # Evaluation configuration
│   └── ros2_config.yaml                # ROS 2 configuration
├── requirements.txt                     # Python dependencies
├── setup.py                            # Package setup
├── .gitignore
└── README.md
```

## Configuration

All configuration files are in the `configs/` directory:

### Training Configuration (`train_config.yaml`)

Key parameters:
- `uncertainty_noise_std`: SLAM uncertainty level during training
- `learning_rate`: SAC learning rate
- `buffer_size`: Replay buffer size
- `net_arch`: Neural network architecture
- `evidential.lambda_reg`: Regularisation coefficient for evidential loss

### Evaluation Configuration (`eval_config.yaml`)

Key parameters:
- `noise_levels`: List of uncertainty levels to test
- `n_episodes`: Number of evaluation episodes per level
- `success_criteria`: Thresholds for successful parking

### ROS 2 Configuration (`ros2_config.yaml`)

Key parameters:
- `odom_topic`: Input odometry topic from robot_localization
- `covariance_topic`: Output covariance topic
- `publish_rate`: Publishing frequency

## Usage Examples

### Training with Custom Hyperparameters

Edit `configs/train_config.yaml` or override via command line:

```bash
python src/uncertainty_rl/training/train_sac.py \
    --config configs/train_config.yaml \
    --total-timesteps 2000000 \
    --seed 123
```

### Evaluating Across Uncertainty Levels

```bash
python src/uncertainty_rl/evaluation/evaluate.py \
    --model-path checkpoints/best_model \
    --config configs/eval_config.yaml \
    --noise-levels 0.1 0.2 0.3 0.4 0.5 \
    --n-episodes 50
```

This generates:
- CSV file with detailed metrics
- Plots showing performance vs. uncertainty
- Statistics on success rates and errors

### Using the Evidential Policy Network

```python
import torch
from uncertainty_rl.networks.evidential_policy import EvidentialPolicyNetwork

# Create network
state_dim = 15  # Including uncertainty features
action_dim = 3  # Steering, throttle, brake
network = EvidentialPolicyNetwork(state_dim, action_dim)

# Get action with uncertainty
state = torch.randn(1, state_dim)
action, uncertainty_dict = network.get_action(state, deterministic=False)

print(f"Action: {action}")
print(f"Epistemic uncertainty: {uncertainty_dict['epistemic']}")
print(f"Aleatoric uncertainty: {uncertainty_dict['aleatoric']}")
```

### Using the CARLA Environment

```python
from uncertainty_rl.envs.carla_parking import CARLAParkingEnv

# Create environment
env = CARLAParkingEnv(
    carla_host="localhost",
    carla_port=2000,
    town="Town01",
    uncertainty_noise_std=0.1,
)

# Reset environment
state, info = env.reset()

# Take actions
for _ in range(100):
    action = env.action_space.sample()  # Random action
    state, reward, terminated, truncated, info = env.step(action)
    
    if terminated or truncated:
        break

env.close()
```

## Methodology

### Evidential Deep Learning

The system uses Normal-Inverse-Gamma (NIG) distributions to model uncertainty:

- **Epistemic Uncertainty**: Reflects model uncertainty, reduces with more data
- **Aleatoric Uncertainty**: Reflects inherent data noise, irreducible

The evidential loss function:
```
L = NLL(γ, ν, α, β, y) + λ * |y - γ| * (2ν + α)
```

Where:
- γ (gamma): Mean of the Gaussian
- ν (nu): Precision parameter
- α (alpha): Shape parameter of Inverse-Gamma
- β (beta): Rate parameter of Inverse-Gamma

### State Representation

State vector (15-dimensional):
1. Position: x, y, yaw
2. Velocity: vx, vy, vyaw
3. Uncertainty (std): σx, σy, σyaw
4. Covariance: cov_xx, cov_yy, cov_yawyaw, cov_xy, cov_xyaw, cov_yyaw

### Reward Function

```
R = -distance + -orientation_error * 0.5 - velocity * 0.1 + success_bonus
```

Success criteria:
- Position error < 0.5m
- Orientation error < 10°
- Velocity < 0.1 m/s

## Visualisation

The evaluation script generates plots showing:
- Success rate vs. uncertainty level
- Average reward vs. uncertainty level
- Position error vs. uncertainty level
- Epistemic/aleatoric uncertainty estimates

## Troubleshooting

### CARLA Connection Issues

If you cannot connect to CARLA:

```bash
# Check if CARLA is running
ps aux | grep Carla

# Try different port
./CarlaUE4.sh -carla-port=2001
```

### GPU Memory Issues

Reduce batch size in `configs/train_config.yaml`:

```yaml
batch_size: 128  # Reduced from 256
```

### ROS 2 Import Errors

Ensure ROS 2 is sourced:

```bash
source /opt/ros/humble/setup.bash
```

## Citation

If you use this code in your research, please cite:

```bibtex
@software{uncertainty_conditioned_rl,
  title={Uncertainty-Conditioned RL with Evidential Policy Networks for AVs},
  author={Uncertainty-Conditioned RL Team},
  year={2024},
  url={https://github.com/Expo39/Uncertainty-Conditioned-RL-with-Evidential-Policy-Networks-for-AVs}
}
```

## Licence

This project is released under the MIT Licence. See LICENSE file for details.

## Acknowledgements

- [CARLA Simulator](https://carla.org/)
- [Stable-Baselines3](https://stable-baselines3.readthedocs.io/)
- [Gymnasium](https://gymnasium.farama.org/)
- [ROS 2](https://docs.ros.org/)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Contact

For questions and support, please open an issue on GitHub.
