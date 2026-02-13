# docker/

Dockerfiles for the containerised research stack. Orchestrated by `docker-compose.yml` at root.

## Files

| File | Base Image | Purpose |
|------|-----------|---------|
| `Dockerfile.ros2` | `ros:jazzy-ros-base-noble` | ROS 2 Jazzy + CARLA bridge + `robot_localisation` EKF for covariance extraction |
| `Dockerfile.training` | `nvcr.io/nvidia/pytorch` | NVIDIA NGC PyTorch + Stable-Baselines3 + evidential networks. All Python deps from `pyproject.toml` |

## Architecture

```
+----------------+    +----------------+    +----------------+
| carla-server   |<-->|  ros2-bridge   |<-->|   training     |
|  (pre-built)   |    | Dockerfile.    |    | Dockerfile.    |
|                |    |    ros2        |    |   training     |
+----------------+    +----------------+    +----------------+
```

All containers share a bridge network (`uncertainty-rl-network`). Code directories are bind-mounted for hot-reloading.

## Building

```bash
make docker-build           # Build all images
make docker-build-no-cache  # Clean rebuild
```
