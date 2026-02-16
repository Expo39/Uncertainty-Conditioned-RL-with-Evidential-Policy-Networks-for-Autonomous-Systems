"""
@file carla_bridge.launch.py
@brief Launch file for CARLA bridge, robot_localisation EKF, and covariance extraction.

Orchestrates the full sensor-to-covariance pipeline:
1. CARLA ROS bridge (publishes noisy sensor data from CARLA to ROS 2 topics)
2. robot_localisation EKF node (fuses sensor data, outputs /odometry/filtered)
3. CovarianceExtractorNode (extracts 3x3 [x, y, yaw] covariance, publishes
   CovarianceEstimate to /slam_uncertainty/covariance for the training
   container to consume)

EKF and covariance extractor parameters are loaded from configs/ros2_config.yaml
(path configurable via ROS2_CONFIG_PATH environment variable) per Henki ROS 2
best practices (no hardcoded parameters in launch files).

@note CARLA ROS bridge topic names depend on the bridge version and vehicle
      role name. The defaults below assume the standard carla_ros_bridge output.
      Verify with `ros2 topic list` after launching.
"""

import os
from pathlib import Path

import yaml
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _load_ros2_config() -> dict:
    """
    @brief Load ROS 2 configuration from YAML file.
    @return Configuration dictionary. Returns empty dict if file not found.
    """
    config_path = os.environ.get(
        "ROS2_CONFIG_PATH", "/workspace/configs/ros2_config.yaml"
    )
    path = Path(config_path)

    if not path.exists():
        # Fallback: check relative to this file (for local development)
        repo_root = Path(__file__).resolve().parents[3]
        path = repo_root / "configs" / "ros2_config.yaml"

    if path.exists():
        with open(path) as f:
            return yaml.safe_load(f) or {}

    return {}


def generate_launch_description() -> LaunchDescription:
    """
    @brief Generate launch description for the full CARLA + EKF + covariance stack.
    @return LaunchDescription with all nodes and launch arguments.
    """
    config = _load_ros2_config()

    # -- Launch arguments --------------------------------------------------

    carla_host_arg = DeclareLaunchArgument(
        "carla_host",
        default_value=os.environ.get("CARLA_HOST", "carla-server"),
        description="CARLA server hostname.",
    )

    carla_port_arg = DeclareLaunchArgument(
        "carla_port",
        default_value=os.environ.get("CARLA_PORT", "2000"),
        description="CARLA server port.",
    )

    town_arg = DeclareLaunchArgument(
        "town",
        default_value="Town01",
        description="CARLA town/map to load.",
    )

    # -- CARLA ROS bridge --------------------------------------------------
    # The bridge connects to CARLA and publishes sensor data as ROS 2 topics.
    # Noisy sensors (IMU, GNSS) attached to the ego vehicle in CARLA produce
    # inherently noisy data that the EKF must filter.

    try:
        from ament_index_python.packages import get_package_share_directory

        carla_bridge_share = get_package_share_directory("carla_ros_bridge")
        carla_bridge_launch = os.path.join(
            carla_bridge_share, "carla_ros_bridge.launch.py"
        )

        carla_bridge = IncludeLaunchDescription(
            PythonLaunchDescriptionSource(carla_bridge_launch),
            launch_arguments={
                "host": LaunchConfiguration("carla_host"),
                "port": LaunchConfiguration("carla_port"),
                "town": LaunchConfiguration("town"),
                "synchronous_mode": "true",
                "fixed_delta_seconds": "0.05",
            }.items(),
        )
    except Exception:
        # If carla_ros_bridge is not installed, skip it
        # (useful for testing the launch file structure)
        carla_bridge = None

    # -- robot_localisation EKF node ---------------------------------------
    # Fuses noisy odometry and IMU data from the CARLA ROS bridge to produce
    # a filtered pose estimate with covariance at /odometry/filtered.
    #
    # Parameters are loaded from configs/ros2_config.yaml (ekf section).
    #
    # @note The odom0 and imu0 topic names must match the CARLA ROS bridge
    #       output. Verify with `ros2 topic list` after bridge startup.

    ekf_config = config.get("ekf", {})

    ekf_node = Node(
        package="robot_localization",
        executable="ekf_node",
        name="ekf_filter_node",
        parameters=[ekf_config],
        remappings=[
            ("odometry/filtered", "/odometry/filtered"),
        ],
    )

    # -- Covariance extractor node -----------------------------------------
    # Subscribes to /odometry/filtered, extracts the 3x3 [x, y, yaw]
    # covariance submatrix, and publishes a CovarianceEstimate message to
    # /slam_uncertainty/covariance for the training container to consume.

    covariance_extractor = Node(
        package="uncertainty_rl_ros2",
        executable="covariance_extractor",
        name="covariance_extractor",
        parameters=[
            {
                "odom_topic": config.get("odom_topic", "/odometry/filtered"),
                "covariance_topic": config.get(
                    "covariance_topic", "/slam_uncertainty/covariance"
                ),
                "publish_rate": config.get("publish_rate", 10.0),
            }
        ],
    )

    # -- Assemble launch description ---------------------------------------

    launch_actions = [
        carla_host_arg,
        carla_port_arg,
        town_arg,
        ekf_node,
        covariance_extractor,
    ]

    # Only include CARLA bridge if the package was found
    if carla_bridge is not None:
        launch_actions.insert(3, carla_bridge)

    return LaunchDescription(launch_actions)
