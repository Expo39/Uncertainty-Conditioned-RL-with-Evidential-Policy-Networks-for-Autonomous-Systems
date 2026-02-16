# ros2/

ROS 2 ament_python package for SLAM covariance extraction from the `robot_localisation` EKF.

## Dockerfile

`Dockerfile` builds the **ros2-bridge** container: ROS 2 Jazzy base, CARLA ROS bridge, `robot_localisation` EKF, and custom covariance extraction nodes. Orchestrated via `docker-compose.yml` at the project root.

## Package Structure

This directory is both an ament_python package (built by colcon in the ros2-bridge container) and a subpackage of `uncertainty_rl`. Node code lives in one place only: `uncertainty_rl_ros2/covariance_extractor.py`.

| File | Purpose |
|------|---------|
| `__init__.py` | Conditional re-export from `uncertainty_rl_ros2` (graceful fallback without rclpy) |
| `package.xml` | ament_python manifest with dependencies |
| `setup.py` / `setup.cfg` | Python package setup for colcon |
| `resource/uncertainty_rl_ros2` | Empty ament_index marker (required by ament) |
| `uncertainty_rl_msgs/` | Custom message package: `CovarianceEstimate.msg` (ament_cmake) |
| `uncertainty_rl_ros2/` | Single source of truth for node code |
| `launch/carla_bridge.launch.py` | Launches CARLA bridge + EKF + covariance extractor (params from YAML) |

## Launch File

`launch/carla_bridge.launch.py` orchestrates the full pipeline:

1. **CARLA ROS bridge** — publishes sensor topics from CARLA simulator
2. **robot_localisation EKF** — fuses odometry + IMU (optionally GNSS), outputs `/odometry/filtered`
3. **CovarianceExtractorNode** — extracts 3x3 covariance, publishes to `/slam_uncertainty/covariance`

## Nodes

| Node | Purpose |
|------|---------|
| `CovarianceExtractorNode` | Subscribes to `/odometry/filtered`, extracts 3x3 [x, y, yaw] submatrix, publishes `CovarianceEstimate` (semantic fields: header, x, y, yaw, covariance[9]) |
| `CovarianceMonitorNode` | Debug/visualisation node for monitoring covariance values (subscribes to `CovarianceEstimate`) |

## EKF Sensor Fusion

EKF sensor inputs are an open decision (see TODO.md Task 4). Preferred: **odometry + IMU only** (parking manoeuvres are short, drift stays small). Alternatives: add GNSS or LiDAR-SLAM. Configured with `two_d_mode: true`. See `configs/ros2_config.yaml` for full EKF parameters including topic remappings and fusion matrix configs.

## Configuration

Parameters set via `configs/ros2_config.yaml`:
- `carla_topics.*`: CARLA ROS bridge topic names (odometry, imu; optionally gnss)
- `ekf.*`: EKF frequency, 2D mode, fusion configs
- `odom_topic`: Input odometry topic (default: `/odometry/filtered`)
- `covariance_topic`: Output covariance topic
- `publish_rate`: Update rate in Hz (default: 10)

## QoS

Uses RELIABLE QoS profile to ensure no dropped covariance messages.
