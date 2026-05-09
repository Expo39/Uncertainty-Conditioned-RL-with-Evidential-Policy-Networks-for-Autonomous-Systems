"""
@file real_vehicle.launch.py
@brief Launch file for the real-vehicle sensor-to-covariance pipeline.
"""

import importlib.util
import math
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
    @return LaunchDescription with navsat_transform, static TFs, EKF, and
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
    imu_topic: str = str(real_cfg.get("imu_topic", "/imu/data"))

    # Internal topic names match sim so CovarianceExtractorNode config is identical.
    gnss_odom_topic: str = "/odometry/gps"
    odom_filtered_topic: str = str(ros2_cfg.get("odom_topic", "/odometry/filtered"))
    covariance_topic: str = str(
        ros2_cfg.get("covariance_topic", "/ekf_uncertainty/covariance")
    )

    datum_lat: float = float(datum_doc.get("datum_lat", 0.0))
    datum_lon: float = float(datum_doc.get("datum_lon", 0.0))
    datum_yaw: float = math.radians(float(datum_doc.get("heading_deg", 0.0)))

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
            default_value=imu_topic,
            description="Physical IMU driver topic (sensor_msgs/Imu).",
        ),
    ]

    # -- Static TF nodes ---------------------------------------------------
    # Real vehicle uses ROS-conventional frame names (base_link, imu_link, etc.)
    # rather than CARLA's ego_vehicle/* names.
    sensor_tf_nodes = build_sensor_tf_nodes(
        sensor_cfg,
        body_frame="base_link",
        imu_frame="imu_link",
        lidar_frame="lidar_link",
        gnss_frame="gnss_link",
        use_sim_time=False,
    )
    sensor_tf_nodes.append(
        # map->odom identity for nav_msgs consumers that need the full chain.
        static_tf("map_to_odom_tf", "map", "odom", 0.0, 0.0, 0.0)
    )

    # -- navsat_transform_node ---------------------------------------------
    # Converts NavSatFix -> metric Odometry in the local ENU frame.
    # Replaces GnssNoiseRelayNode's flat-earth projection from the sim pipeline.
    # Launch aborts above if datum_lat/datum_lon are still 0.0 placeholders.
    navsat_cfg = ros2_cfg.get("navsat_transform", {})
    navsat_node = Node(
        package="robot_localization",
        executable="navsat_transform_node",
        name="navsat_transform_node",
        parameters=[
            {
                "use_sim_time": False,
                "datum": [datum_lat, datum_lon, datum_yaw],
                "magnetic_declination_radians": float(
                    navsat_cfg.get("magnetic_declination_radians", 0.0)
                ),
                "yaw_offset": float(navsat_cfg.get("yaw_offset", 0.0)),
                "zero_altitude": True,
                "broadcast_utm_transform": navsat_cfg.get(
                    "broadcast_utm_transform", False
                ),
                "publish_filtered_gps": navsat_cfg.get("publish_filtered_gps", False),
                "use_odometry_yaw": navsat_cfg.get("use_odometry_yaw", False),
                "wait_for_datum": navsat_cfg.get("wait_for_datum", False),
                "frequency": float(navsat_cfg.get("frequency", 20.0)),
                "delay": float(navsat_cfg.get("delay", 3.0)),
            }
        ],
        remappings=[
            ("imu/data", imu_topic),
            ("gps/fix", gnss_fix_topic),
            ("odometry/gps", gnss_odom_topic),
        ],
    )

    # -- EKF node ----------------------------------------------------------
    # Same EKF params as sim except use_sim_time=False, base_link_frame=base_link,
    # and no pose0 (COG heading): navsat_transform handles heading internally.
    ekf_params = {
        **ros2_cfg.get("ekf", {}),
        "use_sim_time": False,
        "base_link_frame": "base_link",
        "world_frame": "odom",
        "odom0": gnss_odom_topic,
        "imu0": imu_topic,
        "pose0": "",
    }
    ekf_node = Node(
        package="robot_localization",
        executable="ekf_node",
        name="ekf_filter_node",
        parameters=[ekf_params],
        remappings=[("odometry/filtered", odom_filtered_topic)],
    )

    # -- CovarianceExtractorNode -------------------------------------------
    # Identical config to sim: subscribes to /odometry/filtered, writes ekf_state.json.
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
        navsat_node,
        ekf_node,
        covariance_extractor,
    ]

    return LaunchDescription(actions)
