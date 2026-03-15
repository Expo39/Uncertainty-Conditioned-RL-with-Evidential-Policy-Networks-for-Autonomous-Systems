"""
@file carla_bridge.launch.py
@brief Launch file for CARLA bridge, Cartographer, robot_localisation EKF,
       and covariance extraction.

Orchestrates the full sensor-to-covariance pipeline:
1. CARLA ROS bridge (publishes noisy sensor data from CARLA to ROS 2 topics)
2. pointcloud_to_laserscan (converts 360 deg PointCloud2 to LaserScan)
3. laser_filter_node (clips rear 90 deg to simulate 270 deg SICK TiM 5xx FOV)
4. Cartographer (2D SLAM scan-matching; publishes /scan_matched_odometry)
5. robot_localisation EKF node (fuses IMU + scan-matched odometry,
   outputs /odometry/filtered with covariance driven by scan quality)
6. CovarianceExtractorNode (extracts 3x3 [x, y, yaw] covariance + velocity,
   publishes CovarianceEstimate to /ekf_uncertainty/covariance for the training
   container to consume)

LiDAR pipeline (Suite A -- default):
  /carla/ego_vehicle/lidar (PointCloud2, 360 deg, 1 channel)
      -> pc2_to_scan node     -> /lidar/scan_raw (LaserScan, 360 deg)
      -> lidar_filter node    -> /lidar/scan     (LaserScan, 270 deg)
      -> Cartographer node    -> /scan_matched_odometry (Odometry + covariance)
      -> EKF odom0 correction -> /odometry/filtered (filtered pose + covariance)

LiDAR pipeline (Suite B/C -- 3D LiDAR, selected via SENSOR_SUITE env var):
  /carla/ego_vehicle/lidar_3d (PointCloud2, 360 deg, 16 channels)
      -> Cartographer node    -> /scan_matched_odometry (Odometry + covariance)
      -> EKF odom0 correction -> /odometry/filtered (filtered pose + covariance)
  (pc2_to_scan and lidar_filter are Suite A only; Suite B/C feed PointCloud2
   directly to Cartographer using num_point_clouds=1 in cartographer_config_3d.lua)

Suite C RGB camera:
  /carla/ego_vehicle/rgb_front (Image) -- available but not consumed here.
  The camera listener is a no-op; this launch file does not subscribe to it.
  @todo(AG) Pending supervisor decision on Suite C utility.

Cartographer config: configs/cartographer/cartographer_config.lua (Suite A) or
configs/cartographer/cartographer_config_3d.lua (Suite B/C). Selected automatically
from the SENSOR_SUITE environment variable (default: suite_a).

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
                # Increase timeout to 30s: CARLA can take >2s to load a new town
                "timeout": "30",
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

    # -- 270 deg LiDAR FOV filter pipeline (Suite A) -----------------------
    # CARLA sensor.lidar.ray_cast always scans 360 deg. The real SICK TiM 5xx
    # scans 270 deg (rear 90 deg obstructed by vehicle body at front bumper).
    # Two nodes clip the scan before it reaches Cartographer (pending):
    #   pc2_to_scan: PointCloud2 (3D) -> LaserScan (2D, horizontal slice)
    #   lidar_filter: LaserScan (360 deg) -> LaserScan (270 deg)
    #
    # Filter angles from ros2_config.yaml lidar_filter section.
    # @note Cartographer subscribes to /lidar/scan (filtered output).
    #       These nodes are wired here; Cartographer is added separately.

    lidar_filter_config = config.get("lidar_filter", {})

    pc2_to_scan = Node(
        package="pointcloud_to_laserscan",
        executable="pointcloud_to_laserscan_node",
        name="pc2_to_scan",
        parameters=[
            {
                "target_frame": "ego_vehicle",
                "min_height": -0.1,
                "max_height": 0.1,
                "range_min": 0.1,
                "range_max": 30.0,
            }
        ],
        remappings=[
            ("cloud_in", "/carla/ego_vehicle/lidar"),
            ("scan", "/lidar/scan_raw"),
        ],
    )

    lidar_filter = Node(
        package="laser_filters",
        executable="scan_to_scan_filter_chain",
        name="lidar_angular_filter",
        parameters=[
            {
                "filter_chain_params_name": "laser_filters",
                "laser_filters": [
                    {
                        "name": "angular_bounds_filter",
                        "type": "laser_filters/LaserScanAngularBoundsFilter",
                        "params": {
                            "lower_angle": lidar_filter_config.get("angle_min", -2.356),
                            "upper_angle": lidar_filter_config.get("angle_max", 2.356),
                        },
                    }
                ],
            }
        ],
        remappings=[
            ("scan", "/lidar/scan_raw"),
            ("scan_filtered", "/lidar/scan"),
        ],
    )

    # -- Cartographer 2D SLAM node (all suites) ------------------------------
    # Performs scan-matching and publishes /scan_matched_odometry
    # (nav_msgs/Odometry with covariance). Covariance reflects scan-matching
    # quality: sparse scans (few parked cars, bad weather) produce higher
    # covariance, driving the thesis uncertainty signal.
    #
    # odom0 in ros2_config.yaml points to /scan_matched_odometry so the EKF
    # uses it as the correction step in addition to the IMU prediction step.
    #
    # Suite A (2D LiDAR, LaserScan):
    #   config: cartographer_config.lua, input topic: /lidar/scan
    # Suite B/C (3D LiDAR, PointCloud2):
    #   config: cartographer_config_3d.lua, input topic: /carla/ego_vehicle/lidar_3d
    #
    # Config directory is /workspace/configs/ (default). Override via
    # CARTOGRAPHER_CONFIG_PATH env var (full path to the .lua file).

    sensor_suite = os.environ.get("SENSOR_SUITE", "suite_a")
    configs_dir = "/workspace/configs/cartographer"

    if sensor_suite in ("suite_b", "suite_c"):
        cartographer_basename = "cartographer_config_3d.lua"
        # Suite B/C: 3D LiDAR publishes PointCloud2 directly (no LaserScan filter)
        cartographer_scan_topic = "/carla/ego_vehicle/lidar_3d"
        cartographer_scan_remap = ("points2", cartographer_scan_topic)
    else:
        # Suite A default: filtered 270 deg LaserScan
        cartographer_basename = "cartographer_config.lua"
        cartographer_scan_topic = "/lidar/scan"
        cartographer_scan_remap = ("scan", cartographer_scan_topic)

    cartographer_node = Node(
        package="cartographer_ros",
        executable="cartographer_node",
        name="cartographer_node",
        arguments=[
            "-configuration_directory",
            configs_dir,
            "-configuration_basename",
            cartographer_basename,
        ],
        remappings=[
            cartographer_scan_remap,
            # Publish the scan-matched pose as odometry for the EKF odom0 input
            ("odom", "/scan_matched_odometry"),
        ],
    )

    cartographer_occupancy_grid = Node(
        package="cartographer_ros",
        executable="cartographer_occupancy_grid_node",
        name="cartographer_occupancy_grid_node",
        parameters=[{"resolution": 0.05}],
    )

    # -- Covariance extractor node -----------------------------------------
    # Subscribes to /odometry/filtered, extracts the 3x3 [x, y, yaw]
    # covariance submatrix and EKF-filtered velocity, and publishes a
    # CovarianceEstimate message to /ekf_uncertainty/covariance for the
    # training container to consume.

    covariance_extractor = Node(
        package="uncertainty_rl_ros2",
        executable="covariance_extractor",
        name="covariance_extractor",
        parameters=[
            {
                "odom_topic": config.get("odom_topic", "/odometry/filtered"),
                "covariance_topic": config.get(
                    "covariance_topic", "/ekf_uncertainty/covariance"
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
        pc2_to_scan,
        lidar_filter,
        cartographer_node,
        cartographer_occupancy_grid,
        covariance_extractor,
    ]

    # Only include CARLA bridge if the package was found
    if carla_bridge is not None:
        launch_actions.insert(3, carla_bridge)

    return LaunchDescription(launch_actions)
