"""
@file real_vehicle.launch.py
@brief Launch file for the real-vehicle sensor-to-covariance pipeline.
"""

import importlib.util
import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch_ros.actions import Node

# Load shared launch helpers by file path so the import is not shadowed by
# the installed ROS 2 'launch' package of the same name.
_common_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "_common.py")
_spec = importlib.util.spec_from_file_location("launch_common", _common_path)
assert _spec is not None, "Failed to load _common.py spec"
_common_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_common_mod)  # type: ignore[union-attr]
build_sensor_tf_nodes = _common_mod.build_sensor_tf_nodes
load_yaml = _common_mod.load_yaml
static_tf = _common_mod.static_tf


def generate_launch_description() -> LaunchDescription:
    """
    @brief Generate the real-vehicle sensor pipeline launch description.
    @return LaunchDescription with the sensor relay, static TFs, EKF, and
            CovarianceExtractorNode.
    """
    ros2_cfg = load_yaml("/workspace/configs/ros2_config.yaml", "ROS2_CONFIG_PATH")
    sensor_cfg = load_yaml(
        "/workspace/configs/deployment/sensor_config.yaml", "SENSOR_CONFIG_PATH"
    ).get("sensors", {})
    datum_doc = load_yaml(
        "/workspace/configs/deployment/real/real_world_datum.yaml",
        "REAL_WORLD_DATUM_PATH",
    ).get("datum", {})

    real_cfg = ros2_cfg.get("real_vehicle", {})
    gnss_fix_topic: str = str(real_cfg.get("gnss_fix_topic", "/gnss/fix"))
    imu_input_topic: str = str(real_cfg.get("imu_topic", "/imu/data"))

    # Held identical to sim so the EKF and CovarianceExtractor configs need no
    # per-deployment variant.
    gnss_odom_topic: str = "/odometry/gps"
    heading_topic: str = "/gnss/heading"
    imu_stamped_topic: str = "/imu/data/stamped"
    odom_filtered_topic: str = str(ros2_cfg.get("odom_topic", "/odometry/filtered"))
    covariance_topic: str = str(
        ros2_cfg.get("covariance_topic", "/ekf_uncertainty/covariance")
    )

    datum_lat: float = float(datum_doc.get("datum_lat", 0.0))
    datum_lon: float = float(datum_doc.get("datum_lon", 0.0))

    if datum_lat == 0.0 and datum_lon == 0.0:
        raise RuntimeError(
            "real_world_datum.yaml contains unmeasured placeholder coordinates "
            "(datum_lat=0.0, datum_lon=0.0). Survey the parking lot origin and "
            "fill in the datum before launching the real-vehicle pipeline."
        )

    launch_args = [
        DeclareLaunchArgument(
            "gnss_fix_topic",
            default_value=gnss_fix_topic,
            description="Physical GNSS driver topic (sensor_msgs/NavSatFix).",
        ),
        DeclareLaunchArgument(
            "imu_topic",
            default_value=imu_input_topic,
            description="Physical IMU driver topic (sensor_msgs/Imu).",
        ),
    ]

    # ROS-conventional frame names here, rather than CARLA's ego_vehicle/* ones.
    sensor_tf_nodes = build_sensor_tf_nodes(
        sensor_cfg,
        body_frame="base_link",
        imu_frame="imu_link",
        lidar_frame="lidar_link",
        gnss_frame="gnss_link",
        use_sim_time=False,
    )
    sensor_tf_nodes.append(
        # Identity, for nav_msgs consumers that need the full chain.
        static_tf("map_to_odom_tf", "map", "odom", 0.0, 0.0, 0.0)
    )

    # The sim executable with noise injection off, so real receiver and IMU
    # values pass through unmodified except for ZUPT. What remains is the
    # projection, COG derivation and covariance stamping the EKF needs.
    gnss_relay_cfg = ros2_cfg.get("gnss_noise_relay", {})
    imu_relay_cfg = ros2_cfg.get("imu_noise_relay", {})

    sensor_relay_node = Node(
        package="uncertainty_rl_ros2",
        executable="sensor_relay",
        name="sensor_relay",
        parameters=[
            {
                "use_sim_time": False,
                "input_topic": gnss_fix_topic,
                "output_topic": gnss_relay_cfg.get("output_topic", "/gnss/noisy"),
                "odom_output_topic": gnss_odom_topic,
                "heading_output_topic": heading_topic,
                "enable_gnss_noise": False,
                "enable_markov_transitions": False,
                "datum_lat": datum_lat,
                "datum_lon": datum_lon,
                "cog_min_displacement_m": gnss_relay_cfg.get(
                    "cog_min_displacement_m", 0.05
                ),
                "enable_cog_heading": gnss_relay_cfg.get("enable_cog_heading", True),
                # The real receiver has its own geometry-driven anisotropy and
                # its own dropouts, so simulating either would double-count.
                "enable_gnss_anisotropy": False,
                "gnss_dropout_probability": 0.0,
                "imu_topic": imu_stamped_topic,
                "imu_input_topic": imu_input_topic,
                "imu_output_topic": imu_stamped_topic,
                "enable_imu_noise": False,
                "imu_gyro_variance": imu_relay_cfg.get("imu_gyro_variance", 7.4631e-8),
                "imu_accel_variance": imu_relay_cfg.get(
                    "imu_accel_variance", 3.77245e-5
                ),
                "zupt_threshold_rad_s": imu_relay_cfg.get(
                    "zupt_threshold_rad_s", 0.015
                ),
                "accel_zupt_threshold_ms2": imu_relay_cfg.get(
                    "accel_zupt_threshold_ms2", 0.2
                ),
                # The physical IMU already has its own scale-factor error.
                "imu_gyro_scale_factor_limit": 0.0,
                "imu_accel_scale_factor_limit": 0.0,
            }
        ],
    )

    # Same params as sim down to pose0, so the COG-derived yaw correction is
    # available here too; only frames, topics and use_sim_time differ.
    ekf_params = {
        **ros2_cfg.get("ekf", {}),
        "use_sim_time": False,
        "base_link_frame": "base_link",
        "world_frame": "odom",
        "odom0": gnss_odom_topic,
        "imu0": imu_stamped_topic,
        "pose0": heading_topic,
    }
    ekf_node = Node(
        package="robot_localization",
        executable="ekf_node",
        name="ekf_filter_node",
        parameters=[ekf_params],
        remappings=[("odometry/filtered", odom_filtered_topic)],
    )

    covariance_extractor = Node(
        package="uncertainty_rl_ros2",
        executable="covariance_extractor",
        name="covariance_extractor",
        parameters=[
            {
                "use_sim_time": False,
                "odom_topic": odom_filtered_topic,
                "covariance_topic": covariance_topic,
                "publish_rate": float(ros2_cfg.get("publish_rate", 10.0)),
                "twist_in_odom_frame": ros2_cfg.get("twist_in_odom_frame", False),
            }
        ],
    )

    actions = [
        *launch_args,
        *sensor_tf_nodes,
        sensor_relay_node,
        ekf_node,
        covariance_extractor,
    ]

    return LaunchDescription(actions)
