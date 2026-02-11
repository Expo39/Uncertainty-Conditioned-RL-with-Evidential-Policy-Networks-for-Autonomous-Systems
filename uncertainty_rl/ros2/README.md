# ros2/

ROS 2 Jazzy nodes for extracting SLAM covariance from the `robot_localisation` EKF.

## Module: `covariance_extractor.py`

### Nodes

| Node | Purpose |
|------|---------|
| `CovarianceExtractorNode` | Subscribes to `/odometry/filtered`, extracts 3x3 [x, y, yaw] submatrix from 6x6 pose covariance, publishes as `Float64MultiArray` |
| `CovarianceMonitorNode` | Debug/visualisation node for monitoring covariance values |

### Covariance Extraction

The 6x6 `nav_msgs/Odometry` pose covariance is reduced to a 2D [x, y, yaw] representation by extracting rows/columns at indices [0, 1, 5].

### Configuration

Parameters set via `configs/ros2_config.yaml`:
- `odom_topic`: Input odometry topic (default: `/odometry/filtered`)
- `covariance_topic`: Output covariance topic
- `publish_rate`: Update rate in Hz (default: 10)

### QoS

Uses RELIABLE QoS profile to ensure no dropped covariance messages.
