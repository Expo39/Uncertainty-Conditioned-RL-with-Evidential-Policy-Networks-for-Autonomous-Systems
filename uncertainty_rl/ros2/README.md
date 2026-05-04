# ros2/

ROS 2 ament_python package for EKF covariance extraction from the `robot_localisation` EKF.

## Dockerfile

`Dockerfile` builds the **ros2-bridge** container: ROS 2 Jazzy base, CARLA ROS bridge,
`robot_localisation` EKF, and custom sensor relay and covariance extraction nodes.
Orchestrated via `docker-compose.yml` at the project root.

## Package Structure

This directory is both an ament_python package (built by colcon in the ros2-bridge
container) and a subpackage of `uncertainty_rl`. Node code lives in one place only:
`uncertainty_rl_ros2/`.

| File | Purpose |
|------|---------|
| `__init__.py` | Conditional re-export from `uncertainty_rl_ros2` (graceful fallback without rclpy) |
| `package.xml` | ament_python manifest with dependencies |
| `setup.py` / `setup.cfg` | Python package setup for colcon |
| `resource/uncertainty_rl_ros2` | Empty ament_index marker (required by ament) |
| `uncertainty_rl_msgs/` | Custom message package: `CovarianceEstimate.msg` (ament_cmake) |
| `uncertainty_rl_ros2/` | Single source of truth for node code |
| `launch/carla_bridge.launch.py` | Launches CARLA bridge + static TFs + GnssNoiseRelay + IMU relay + EKF + CovarianceExtractorNode |
| `launch/real_vehicle.launch.py` | Real-vehicle pipeline: navsat_transform + EKF + CovarianceExtractorNode (no sim nodes) |

## Launch File (Simulation)

`launch/carla_bridge.launch.py` orchestrates the full simulation pipeline:

1. **CARLA ROS bridge** - publishes sensor topics from CARLA (passive mode, no tick)
2. **Static TF publishers** - connect sensor frames to the `ego_vehicle` body frame
3. **GnssNoiseRelayNode** - injects per-episode GNSS noise, projects lat/lon to local XY
   via flat-earth, publishes `/odometry/gps` (Odometry) and `/gnss/heading` (COG)
4. **ImuNoiseRelayNode** - stamps realistic VN-100 covariance on CARLA IMU messages
5. **robot_localisation EKF** - fuses GNSS Odometry + IMU, outputs `/odometry/filtered`
6. **CovarianceExtractorNode** - extracts 3x3 covariance, writes `ekf_state.json`

## Nodes

| Node | Purpose |
|------|---------|
| `GnssNoiseRelayNode` | Adds per-episode Gaussian noise to CARLA GNSS; projects to Odometry via flat-earth; derives COG heading |
| `ImuNoiseRelayNode` | Stamps VN-100 datasheet covariance on CARLA IMU; applies gyro/accel ZUPT |
| `CovarianceExtractorNode` | Subscribes to `/odometry/filtered`, extracts 3x3 [x, y, yaw] submatrix, writes `ekf_state.json`; publishes `CovarianceEstimate` (header, x, y, yaw, vyaw, covariance[9]) |
| `CovarianceMonitorNode` | Debug node: logs pose and uncertainty statistics from `CovarianceEstimate` |

## EKF Sensor Fusion

EKF sensor fusion: **RTK-GNSS + IMU**. Configured with `two_d_mode: true`.
See `configs/ros2_config.yaml` for full EKF parameters including topic remappings
and fusion matrix configs.

## Configuration

Parameters set via `configs/ros2_config.yaml`:
- `ekf.*`: EKF frequency, 2D mode, fusion configs
- `gnss_noise_relay.*`: GnssNoiseRelayNode settings (input topic, noise, COG)
- `imu_noise_relay.*`: ImuNoiseRelayNode settings (variances, ZUPT thresholds)
- `odom_topic`: Input odometry topic (default: `/odometry/filtered`)
- `covariance_topic`: Output covariance topic
- `publish_rate`: Covariance publish rate in Hz (default: 10)

## QoS

RELIABLE profile for input subscriptions; BEST_EFFORT for outputs to
`robot_localisation` (which subscribes BEST_EFFORT - a RELIABLE publisher
paired with a BEST_EFFORT subscriber delivers no messages in ROS 2).
