# Uncertainty-Conditioned RL with Evidential Policy Networks for Autonomous Systems

Propagating EKF localisation uncertainty through evidential deep learning policies for safer autonomous parking in unmapped environments.

## Overview

Most self-driving systems rely on expensive HD maps or vision-only approaches that cannot quantify uncertainty. This system uses an EKF (Extended Kalman Filter) for real-time localisation and feeds the EKF uncertainty directly into an RL policy that controls an autonomous system. The policy uses **evidential deep learning** (Normal-Inverse-Gamma distributions) to quantify its own uncertainty about what action to take, enabling safety handoffs or conservative driving when confidence is low.

**Two layers of uncertainty awareness:**
1. *"How sure am I about where I am?"* - from the EKF (via `robot_localization`)
2. *"How sure am I about what to do?"* - from the evidential policy network

## Installation

Everything runs inside Docker containers. You do not need to install Python, CARLA, ROS 2, or any ML dependencies on your host. You only need Git, Docker, and Make.

---

### Ubuntu

**1. Install prerequisites:**
```bash
sudo apt update
sudo apt install -y make git curl
```

**2. Install Docker Engine:**
```bash
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER
newgrp docker
```

**3. Install the NVIDIA Container Toolkit:**
```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt update
sudo apt install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

Verify GPU passthrough is working:
```bash
docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi
```

You should see your GPU listed in the output.

**4. Clone and build:**
```bash
git clone git@github.com:Expo39/Uncertainty-Conditioned-RL-with-Evidential-Policy-Networks-for-Autonomous-Systems.git
cd Uncertainty-Conditioned-RL-with-Evidential-Policy-Networks-for-Autonomous-Systems
make docker-build    # build all images (~15-20 min first time)
make docker-up       # start CARLA + ROS 2 + training stack
make docker-ps       # verify all services are healthy
make docker-shell    # open a shell inside the training container
```

---

### What gets built

Three containers orchestrated via `docker-compose.yml`:

| Container | Contents |
|-----------|----------|
| `carla-server` | CARLA 0.9.16 headless simulation with GPU passthrough |
| `ros2-bridge` | ROS 2 Jazzy + CARLA bridge + `robot_localisation` EKF |
| `training` | NVIDIA NGC PyTorch + Stable-Baselines3 + evidential networks |

Code directories are bind-mounted. Edit files on the host and changes reflect immediately inside containers.

---

### CPU-Only Development (No GPU, No CARLA)

Unit tests, evidential network development, linting, and type checking all work without Docker or a GPU. Requires Python 3.10+ installed on the host.

```bash
pip install -e ".[dev]"
make verify
```

## Quick Start

Once the stack is up (`make docker-up`), use these commands from the host:

```bash
make docker-train    # run full training inside the container
make docker-eval     # run evaluation inside the container
make docker-test     # run the full test suite inside the container
make docker-shell    # open an interactive shell for manual commands
make docker-down     # stop all containers when done
```

### All Docker Commands

| Command | Purpose |
|---------|---------|
| `make docker-up` | Start all containers |
| `make docker-down` | Stop all containers |
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
| CARLA container not starting | Run `make docker-logs-carla` to see errors |
| GPU not visible in containers | Verify NVIDIA Container Toolkit: `docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi` |
| Out of GPU memory | Reduce `batch_size` / `buffer_size` in `configs/train_config.yaml` |
| Docker permission errors | `sudo usermod -aG docker $USER` then log out and back in |

## Citation

TODO

## LICENCE

TODO

## Acknowledgements

[CARLA](https://carla.org/) | [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) | [Gymnasium](https://gymnasium.farama.org/) | [ROS 2](https://docs.ros.org/)