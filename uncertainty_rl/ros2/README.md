# ros2/

ROS 2 ament_python package for EKF covariance extraction from the `robot_localisation` EKF.

## Dockerfile

`Dockerfile` builds the **ros2-bridge** container: ROS 2 Jazzy base, CARLA ROS bridge, `robot_localisation` EKF + `navsat_transform_node`, and custom covariance extraction nodes. Orchestrated via `docker-compose.yml` at the project root.

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

1. **CARLA ROS bridge** -- publishes sensor topics from CARLA simulator
2. **Static TF publishers** -- connect sensor frames to the `ego_vehicle` body frame
3. **GnssNoiseRelayNode** -- injects per-episode noise into GNSS, publishes `/gnss/noisy`
4. **navsat_transform_node** -- converts NavSatFix to Odometry via UTM projection, publishes `/odometry/gps`
5. **robot_localisation EKF** -- fuses GNSS odometry + IMU, outputs `/odometry/filtered`
6. **CovarianceExtractorNode** -- extracts 3x3 covariance, writes `ekf_state.json`

## Nodes

| Node | Purpose |
|------|---------|
| `GnssNoiseRelayNode` | Adds per-episode Gaussian noise to CARLA GNSS, stamps `position_covariance`; publishes `/gnss/noisy` |
| `CovarianceExtractorNode` | Subscribes to `/odometry/filtered`, extracts 3x3 [x, y, yaw] submatrix, publishes `CovarianceEstimate` (semantic fields: header, x, y, yaw, vx, vy, vyaw, covariance[9]) |
| `CovarianceMonitorNode` | Debug/visualisation node for monitoring covariance values (subscribes to `CovarianceEstimate`) |

## EKF Sensor Fusion

EKF sensor fusion: **RTK-GNSS + IMU**. Configured with `two_d_mode: true`. See `configs/ros2_config.yaml` for full EKF parameters including topic remappings and fusion matrix configs.

## Configuration

Parameters set via `configs/ros2_config.yaml`:
- `carla_topics.*`: CARLA ROS bridge topic names (odometry, imu, gnss)
- `ekf.*`: EKF frequency, 2D mode, fusion configs
- `navsat_transform.*`: navsat_transform_node settings
- `gnss_noise_relay.*`: GnssNoiseRelayNode settings
- `odom_topic`: Input odometry topic (default: `/odometry/filtered`)
- `covariance_topic`: Output covariance topic
- `publish_rate`: Update rate in Hz (default: 10)

## QoS

Uses RELIABLE QoS profile to ensure no dropped covariance messages.
