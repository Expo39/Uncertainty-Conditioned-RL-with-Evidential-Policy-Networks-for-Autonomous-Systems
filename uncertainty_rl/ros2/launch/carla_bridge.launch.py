"""
@file carla_bridge.launch.py
@brief Launch file for CARLA bridge, robot_localisation EKF, and covariance extraction.

Orchestrates the full sensor-to-covariance pipeline:
1. CARLA ROS bridge (publishes noisy sensor data from CARLA to ROS 2 topics)
2. robot_localisation EKF node (fuses sensor data, outputs /odometry/filtered)
3. CovarianceExtractorNode (extracts 3x3 [x, y, yaw] covariance, publishes to
   /slam_uncertainty/covariance for the training container to consume)

@note CARLA ROS bridge topic names depend on the bridge version and vehicle
      role name. The defaults below assume the standard carla_ros_bridge output.
      Verify with `ros2 topic list` after launching.
"""

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    """
    @brief Generate launch description for the full CARLA + EKF + covariance stack.
    @return LaunchDescription with all nodes and launch arguments.
    """
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
    # @note The odom0 and imu0 topic names must match the CARLA ROS bridge
    #       output. Verify with `ros2 topic list` after bridge startup.
    #       Common CARLA bridge topics:
    #         /carla/ego_vehicle/odometry
    #         /carla/ego_vehicle/imu

    ekf_node = Node(
        package="robot_localization",
        executable="ekf_node",
        name="ekf_filter_node",
        parameters=[
            {
                "frequency": 20.0,
                "two_d_mode": True,
                # Odometry input from CARLA bridge
                "odom0": "/carla/ego_vehicle/odometry",
                "odom0_config": [
                    True, True, False,    # x, y, z
                    False, False, True,   # roll, pitch, yaw
                    True, True, False,    # vx, vy, vz
                    False, False, True,   # vroll, vpitch, vyaw
                    False, False, False,  # ax, ay, az
                ],
                # IMU input from CARLA bridge
                "imu0": "/carla/ego_vehicle/imu",
                "imu0_config": [
                    False, False, False,  # x, y, z
                    False, False, True,   # roll, pitch, yaw
                    False, False, False,  # vx, vy, vz
                    False, False, True,   # vroll, vpitch, vyaw
                    True, True, False,    # ax, ay, az
                ],
                "publish_tf": True,
                "world_frame": "odom",
                "odom_frame": "odom",
                "base_link_frame": "base_link",
            }
        ],
        remappings=[
            ("odometry/filtered", "/odometry/filtered"),
        ],
    )

    # -- Covariance extractor node -----------------------------------------
    # Subscribes to /odometry/filtered, extracts the 3x3 [x, y, yaw]
    # covariance submatrix, and publishes to /slam_uncertainty/covariance
    # for the training container to consume via rclpy.

    covariance_extractor = Node(
        package="uncertainty_rl_ros2",
        executable="covariance_extractor",
        name="covariance_extractor",
        parameters=[
            {
                "odom_topic": "/odometry/filtered",
                "covariance_topic": "/slam_uncertainty/covariance",
                "publish_rate": 10.0,
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
