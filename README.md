# Uncertainty-Conditioned RL with Evidential Policy Networks for Autonomous Systems

Propagating SLAM localisation uncertainty through evidential deep learning policies for safer autonomous parking in unmapped environments.

## Overview

Most self-driving systems rely on expensive HD maps or vision-only approaches that cannot quantify uncertainty. This system uses SLAM to build its own map in real time and feeds the SLAM uncertainty directly into an RL policy that controls an anutonomous sytem. The policy uses **evidential deep learning** (Normal-Inverse-Gamma distributions) to quantify its own uncertainty about what action to take - enabling safety handoffs or conservative driving when confidence is low.

**Two layers of uncertainty awareness:**
1. *"How sure am I about where I am?"* - from SLAM (EKF via `robot_localization`)
2. *"How sure am I about what to do?"* - from the evidential policy network

## Installation

### Docker (Recommended)

**Prerequisites:** Docker Engine 20.10+, Docker Compose v2, `nvidia-docker2`, NVIDIA GPU with CUDA 12.1+, 20GB+ disk space.

```bash
git clone git@github.com:Expo39/Uncertainty-Conditioned-RL-with-Evidential-Policy-Networks-for-Autonomous-Systems.git
cd Uncertainty-Conditioned-RL-with-Evidential-Policy-Networks-for-AVs

make docker-build              # Build all images (~15-20 min first time)
make docker-up                 # Start CARLA + ROS 2 + training stack
make docker-ps                 # Verify services are healthy
make docker-shell              # Interactive shell in training container
```

Three containers are built:
- **carla-server** - CARLA 0.9.15 headless with GPU passthrough
- **ros2-bridge** - ROS 2 Jazzy + CARLA bridge + `robot_localisation` EKF
- **training** - NVIDIA NGC PyTorch + SB3 + evidential networks

Code directories are bind-mounted for hot-reloading - edit on host, changes reflect immediately.

### Native (Alternative)

**Prerequisites:** Python 3.10+, CARLA 0.9.13+, CUDA GPU.

```bash
python -m venv venv && source venv/bin/activate
pip install -e .
```

CARLA Python API must be on `PYTHONPATH`. ROS 2 Jazzy required only for `robot_localization` integration.

## Quick Start

```bash
# 1. Start CARLA (or use Docker)
./CarlaUE4.sh -RenderOffScreen

# 2. Train
python uncertainty_rl/training/train_ppo.py \
    --config configs/train_config.yaml \
    --total-timesteps 1000000

# 3. Evaluate across noise levels
python uncertainty_rl/evaluation/evaluate.py \
    --model-path checkpoints/final_model \
    --config configs/eval_config.yaml \
    --noise-levels 0.05 0.1 0.2 0.5 1.0

# 4. ROS 2 covariance extractor (optional)
python uncertainty_rl/ros2/covariance_extractor.py
```

### Docker Commands

| Command | Purpose |
|---------|---------|
| `make docker-up` / `docker-down` | Start / stop all containers |
| `make docker-shell` | Interactive bash in training container |
| `make docker-train` | Run full training |
| `make docker-train-short` | 10k steps smoke test |
| `make docker-eval` | Run evaluation |
| `make docker-test` | Run pytest |
| `make docker-logs` | Follow all container logs |
| `make docker-dev` | Start stack + drop into training shell |
| `make docker-clean` | Stop and remove volumes |

Run `make help` for the full list.

## Architecture

```
uncertainty_rl/                      # Main Python package
|-- networks/evidential_policy.py    # Evidential layers, NIG distributions
|-- envs/carla_parking.py            # CARLA Gymnasium environment (15D state, 3D action)
|-- training/
|   |-- train_ppo.py                 # PPO training with SB3
|   +-- Dockerfile                   # Training container (NGC PyTorch + SB3)
|-- evaluation/evaluate.py           # Noise sweep, metrics, plots
|-- ros2/
|   |-- covariance_extractor.py      # Bridge to robot_localization EKF
|   +-- Dockerfile                   # ROS 2 bridge container (Jazzy + robot_localisation)
+-- utils/
    |-- logging.py                   # MetricsLogger, UncertaintyTracker
    +-- visualisation.py             # Trajectory plots, uncertainty evolution, training curves
configs/                             # YAML hyperparameters (train, eval, ROS 2)
tests/                               # pytest suite mirroring uncertainty_rl/ structure
```

## Configuration

All hyperparameters live in `configs/` YAML files - never hardcoded in source.

| File | Key Parameters |
|------|---------------|
| `train_config.yaml` | `learning_rate` (0.0003), `batch_size` (256), `buffer_size` (1M), `uncertainty_noise_std` (0.1m), `net_arch` ([256, 256]), `evidential.lambda_reg` (0.01) |
| `eval_config.yaml` | `noise_levels`, `n_episodes` (100), `success_criteria` thresholds |
| `ros2_config.yaml` | `odom_topic`, `covariance_topic`, `publish_rate` (10 Hz) |

## Troubleshooting

| Problem | Fix |
|---------|-----|
| CARLA won't connect | Check `ps aux \| grep Carla`, try `--carla-port 2001` |
| Out of GPU memory | Reduce `batch_size` / `buffer_size` in config |
| ROS 2 imports fail | `source /opt/ros/humble/setup.bash` |
| Docker CARLA not starting | `make docker-logs-carla`, verify GPU: `docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi` |
| Docker permission errors | `sudo chown -R $USER:$USER logs/ checkpoints/ results/` |

## Citation

TODO

## LICENCE

TODO

## Acknowledgements

[CARLA](https://carla.org/) | [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) | [Gymnasium](https://gymnasium.farama.org/) | [ROS 2](https://docs.ros.org/)
