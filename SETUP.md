# Setup

Installation and first run. Day-to-day operation is covered in
[USAGE.md](USAGE.md), and the complete Make target reference in
[COMMANDS.md](COMMANDS.md).

Everything runs inside Docker. Git, Docker Engine, Make and the NVIDIA Container
Toolkit are required on the host. No installation of Python, CARLA or ROS 2 on the
host is needed.

---

## 1 - Install Docker Engine (Ubuntu)

```bash
sudo apt update && sudo apt install -y make git curl
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER && newgrp docker
```

## 2 - Install NVIDIA Container Toolkit

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt update && sudo apt install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker
```

GPU passthrough is then verified with:

```bash
docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi
```

## 3 - Clone and Build

```bash
git clone <repo-url>
cd Uncertainty-Conditioned-RL-with-Evidential-Policy-Networks-for-Autonomous-Systems
make docker-build            # first build - bakes ROS 2 nodes into images (~15-20 min)
make docker-up               # start CARLA + ROS 2 + training stack
make docker-ps               # verify all three services are healthy
```

> **Build cache.** The ROS 2 node sources are COPYed into the `ros2-bridge` image
> rather than bind-mounted. After editing anything under `uncertainty_rl/ros2/` or any
> `Dockerfile`, rebuild with `make docker-build-no-cache`. The plain `make docker-build`
> uses the layer cache and can serve a stale image without reporting an error. It is
> safe only for `pyproject.toml` changes and for bind-mounted files.

---

## Container Architecture

Three containers are orchestrated through `docker-compose.yml`:

| Container | Image Base | Contents |
|-----------|-----------|----------|
| `carla-server` | CARLA 0.9.16 | Headless simulation with GPU passthrough |
| `ros2-bridge` | ROS 2 Jazzy | CARLA bridge, `robot_localization` EKF, noise relay nodes |
| `training` | NGC PyTorch 24.10 | Stable-Baselines3, evidential networks, rclpy (Humble) |

The two ROS 2 distributions differ because the NGC PyTorch base image is built on
Ubuntu 22.04, which carries Humble, whereas the bridge runs on Ubuntu 24.04 and
carries Jazzy. DDS is distribution-agnostic, so the two communicate without a shim.

Code directories are bind-mounted. Edits made on the host are reflected immediately
inside the containers.

---

## Host Python Setup

Two categories of command run on the host rather than inside Docker:

| Command | Why it runs on the host |
|---------|------------------------|
| `make visualise`, `make eval-visualise-2d` | Opens a Pygame window, and the containers are headless |
| `make generate-layouts` | Writes layout PNGs through Matplotlib, needing neither CARLA nor a GPU |
| `make analyse-*`, `make figures`, `make analysis-bundle` | Read the evaluation CSVs with pandas, needing neither CARLA nor a GPU |
| `make training-curves`, `make tb-scalars` | Parse TensorBoard event files through the reader library alone, not the dashboard container |

All of these use the project's `.venv/`, which the Makefile manages automatically:

```bash
make install   # one-time setup: creates .venv/ and installs the package + dev dependencies
```

Note that `torch`, `stable_baselines3` and `gymnasium` are deliberately absent from
`.venv/`. Unit tests are therefore run with `make docker-test-unit` rather than
`make test-unit`, which fails on the host. The host checks are `make lint`,
`make typecheck`, `make format` and `make verify`, the last of which mirrors CI exactly.

---

## First Run

```bash
# 1. Start the full stack
make docker-up

# 2. Train a curriculum stage (STAGE defaults to 1, the curriculum head)
make docker-train STAGE=1 BASELINE=full_method

# 3. Resume the next stage from the previous stage's run leaf (bare name, not a path)
make docker-train STAGE=2 BASELINE=full_method CHECKPOINT=6_42_11062026-0628

# 4. Attach the 2D visualiser at any time (host terminal, non-blocking)
make visualise

# 5. Run the evaluation sweep
make docker-eval BASELINE=full_method CHECKPOINT=6_42_11062026-0628

# 6. Stop when done
make docker-down
```

`BASELINE` and `CHECKPOINT` are bare names rather than paths, with the Make recipes
reconstructing the full paths. The convention is set out in
[COMMANDS.md](COMMANDS.md).

Training follows a six-stage single-phase curriculum. Every observation
channel is live in every stage, and the range of one axis is ramped per stage, moving
from bays and margin through to obstacle occupancy. The GNSS degradation process is a
fixed, stage-invariant Markov chain with always-on mid-episode drift, so localisation
uncertainty is present from stage 1 rather than being treated as a ramped axis. Each
stage resumes from the checkpoint of its predecessor. Further detail is given in the
[curriculum stage README](configs/deployment/sim/curriculum/README.md).

Architectural settings, namely `net_arch`, `activation`, `policy_type`,
`include_covariance`, `include_obstacle_obs` and the observation and action
dimensions, must not change between stages, since saved weights would otherwise fail
to load.

---

## Verifying an Installation

```bash
make docker-ps            # all three services healthy
make docker-test-unit     # unit tests, inside the container
make verify               # lint + typecheck + import, exactly what CI runs
make docker-train-short   # 10k-step smoke test across the full stack
```

---

## Troubleshooting

| Symptom | Likely cause |
|---------|--------------|
| `ModuleNotFoundError: No module named 'torch'` | A test was run on the host. Use `make docker-test-unit`. |
| ROS 2 node changes have no effect | The layer cache served a stale image. Rebuild with `make docker-build-no-cache`. |
| `nvidia-smi` fails inside a container | The NVIDIA Container Toolkit is not configured. Repeat step 2. |
| The environment blocks awaiting the EKF | `ros2-bridge` is unhealthy. Inspect `make docker-logs-ros2`. |
| Weights fail to load on a stage resume | An architectural setting was changed between stages. See the note above. |
