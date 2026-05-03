"""
@file real_vehicle.launch.py
@brief Launch file for the real-vehicle sensor-to-covariance pipeline.

Mirrors carla_bridge.launch.py but replaces all CARLA-specific nodes with
their physical-sensor equivalents. No CARLA bridge, no GnssNoiseRelayNode,
no ImuNoiseRelayNode. use_sim_time=False throughout (wall clock).

Full pipeline launched here:

  Physical GNSS driver  -> /gnss/fix  (sensor_msgs/NavSatFix, raw lat/lon)
                                |
                     navsat_transform_node  <- datum lat/lon from real_world_datum.yaml
                                |
                       /odometry/gps  (nav_msgs/Odometry, metric XY in odom frame)
                                |
  Physical IMU driver -> /imu/data ----> robot_localisation EKF
                                |
                       /odometry/filtered  (pose + 6x6 covariance)
                                |
                     CovarianceExtractorNode
                                |
                         ekf_state.json  -> inference_loop.py

navsat_transform_node handles the NavSatFix -> metric Odometry conversion that
GnssNoiseRelayNode performs in simulation. It reads the lot datum (lat/lon of
the EKF origin) from configs/deployment/real/real_world_datum.yaml and uses it
to anchor the flat-earth projection. The datum must be surveyed on-site and
filled in before deployment.

Topic names for physical sensor drivers are configured under real_vehicle in
configs/ros2_config.yaml:
  real_vehicle.gnss_fix_topic  -> topic published by the physical GNSS driver
  real_vehicle.imu_topic       -> topic published by the physical IMU driver

@note navsat_transform_node requires the IMU to be publishing before it
      will output /odometry/gps. If /odometry/gps is silent, check that the
      IMU driver is running and the datum lat/lon are non-zero.

@note The physical IMU driver must publish sensor_msgs/Imu with a populated
      angular_velocity_covariance. Zero covariance causes the EKF to weight
      IMU updates incorrectly. Verify with:
        ros2 topic echo /imu/data | grep -A3 angular_velocity_covariance

@author Antonio Galdes
"""

import math
import os
from pathlib import Path
from typing import List

import yaml
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch_ros.actions import Node


def _load_yaml(path_str: str, env_override: str = "") -> dict:
    """
    @brief Load a YAML config, optionally overridden by an env var.
    @param path_str: Default file path inside the container.
    @param env_override: Env var name whose value overrides path_str.
    @return Parsed dictionary, or empty dict if not found.
    """
    resolved = os.environ.get(env_override, path_str) if env_override else path_str
    path = Path(resolved)

    if not path.exists():
        try:
            rel = path.relative_to("configs")
        except ValueError:
            rel = Path(path.name)
        path = Path(__file__).resolve().parents[3] / "configs" / rel

    if path.exists():
        with open(path) as f:
            return yaml.safe_load(f) or {}
    return {}


def _static_tf(
    name: str,
    parent: str,
    child: str,
    x: float,
    y: float,
    z: float,
) -> Node:
    """
    @brief Create a zero-rotation static_transform_publisher node.

    Always uses use_sim_time=False (wall clock). Timestamp=0 from static
    publishers is valid at all times in TF2.

    @param name: Node name.
    @param parent: Parent TF frame ID.
    @param child: Child TF frame ID.
    @param x: Translation X (metres, vehicle body frame).
    @param y: Translation Y (metres, vehicle body frame).
    @param z: Translation Z (metres, vehicle body frame).
    @return Configured static_transform_publisher Node.
    """
    return Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name=name,
        parameters=[{"use_sim_time": False}],
        arguments=[
            "--x", str(x),
            "--y", str(y),
            "--z", str(z),
            "--roll", "0",
            "--pitch", "0",
            "--yaw", "0",
            "--frame-id", parent,
            "--child-frame-id", child,
        ],
    )


def _build_sensor_tf_nodes(sensors_cfg: dict) -> List[Node]:
    """
    @brief Build static TF nodes for physical sensor mount positions.

    Mount positions come from configs/deployment/sensor_config.yaml under
    sensors.imu.mount, sensors.gnss.mount, and sensors.lidar.mount.
    All values are UNMEASURED placeholders -- update with physical measurements
    before deployment.

    @warning IMU lever arm from the GNSS antenna must be accurate to ~2 cm.
             A 5 cm error introduces ~5 cm * sin(heading_change) positional bias.

    @param sensors_cfg: sensors dict from sensor_config.yaml.
    @return List of static TF publisher nodes.
    """
    imu_mount = sensors_cfg.get("imu", {}).get("mount", {})
    gnss_mount = sensors_cfg.get("gnss", {}).get("mount", {})
    lidar_mount = sensors_cfg.get("lidar", {}).get("mount", {})

    return [
        _static_tf(
            "body_to_imu_tf",
            "base_link",
            "imu_link",
            float(imu_mount.get("x", 0.0)),
            float(imu_mount.get("y", 0.0)),
            float(imu_mount.get("z", 0.3)),
        ),
        _static_tf(
            "body_to_gnss_tf",
            "base_link",
            "gnss_link",
            float(gnss_mount.get("x", 0.0)),
            float(gnss_mount.get("y", 0.0)),
            float(gnss_mount.get("z", 1.6)),
        ),
        # LiDAR frame for rviz / diagnostics. EKF does not use LiDAR.
        _static_tf(
            "body_to_lidar_tf",
            "base_link",
            "lidar_link",
            float(lidar_mount.get("x", 2.4)),
            float(lidar_mount.get("y", 0.0)),
            float(lidar_mount.get("z", 0.5)),
        ),
        # map -> odom identity: lets any consumer that needs the full
        # map->odom->base_link chain find it. The EKF publishes odom->base_link.
        _static_tf("map_to_odom_tf", "map", "odom", 0.0, 0.0, 0.0),
    ]


def generate_launch_description() -> LaunchDescription:
    """
    @brief Generate the real-vehicle sensor pipeline launch description.
    @return LaunchDescription with navsat_transform, static TFs, EKF, and
            CovarianceExtractorNode.
    """
    ros2_cfg = _load_yaml(
        "/workspace/configs/ros2_config.yaml", "ROS2_CONFIG_PATH"
    )
    sensor_cfg = _load_yaml(
        "/workspace/configs/deployment/sensor_config.yaml", "SENSOR_CONFIG_PATH"
    ).get("sensors", {})
    datum_doc = _load_yaml(
        "/workspace/configs/deployment/real/real_world_datum.yaml",
        "REAL_WORLD_DATUM_PATH",
    ).get("datum", {})

    real_cfg = ros2_cfg.get("real_vehicle", {})

    # Physical sensor driver topic names.
    gnss_fix_topic: str = str(real_cfg.get("gnss_fix_topic", "/gnss/fix"))
    imu_topic: str = str(real_cfg.get("imu_topic", "/imu/data"))

    # Internal pipeline topics (same names as sim so CovarianceExtractorNode
    # config is identical between sim and real).
    gnss_odom_topic: str = "/odometry/gps"
    odom_filtered_topic: str = str(ros2_cfg.get("odom_topic", "/odometry/filtered"))
    covariance_topic: str = str(
        ros2_cfg.get("covariance_topic", "/ekf_uncertainty/covariance")
    )

    # Datum lat/lon: anchor for navsat_transform_node flat-earth projection.
    # Sourced from real_world_datum.yaml -- must be surveyed on-site.
    datum_lat: float = float(datum_doc.get("datum_lat", 0.0))
    datum_lon: float = float(datum_doc.get("datum_lon", 0.0))
    datum_yaw: float = math.radians(float(datum_doc.get("heading_deg", 0.0)))

    launch_args = [
        DeclareLaunchArgument(
            "gnss_fix_topic",
            default_value=gnss_fix_topic,
            description=(
                "Physical GNSS driver topic (sensor_msgs/NavSatFix). "
                "navsat_transform_node converts this to metric Odometry."
            ),
        ),
        DeclareLaunchArgument(
            "imu_topic",
            default_value=imu_topic,
            description="Physical IMU driver topic (sensor_msgs/Imu).",
        ),
    ]

    # -- Static TF nodes -------------------------------------------------------
    sensor_tf_nodes = _build_sensor_tf_nodes(sensor_cfg)

    # -- navsat_transform_node -------------------------------------------------
    # Converts sensor_msgs/NavSatFix -> nav_msgs/Odometry in the local metric
    # (flat-earth ENU) frame. Publishes on /odometry/gps for the EKF odom0 input.
    #
    # Requires:
    #   - /gnss/fix  : NavSatFix from the physical GNSS driver
    #   - /imu/data  : Imu from the physical IMU driver (for initial yaw)
    #   - datum_lat/lon: surveyed anchor point (from real_world_datum.yaml)
    #
    # datum_lat/lon pin the origin of the local metric frame.  When set to the
    # surveyed lot reference point, the navsat_transform output is in the same
    # local coordinate frame used by calibrate_ekf_frame_offset() in
    # _parking_core.py, which maps EKF odom -> lot layout frame at mission start.
    #
    # @warning If datum_lat/datum_lon are both 0.0 (unmeasured placeholders),
    #          navsat_transform will auto-latch the datum to the first GNSS fix.
    #          This means the EKF origin drifts between missions.  Fill in the
    #          surveyed values in real_world_datum.yaml before deployment.
    navsat_cfg = ros2_cfg.get("navsat_transform", {})
    navsat_node = Node(
        package="robot_localization",
        executable="navsat_transform_node",
        name="navsat_transform_node",
        parameters=[
            {
                "use_sim_time": False,
                # Datum anchor: the physical reference point surveyed on-site.
                # navsat_transform pins the local metric frame origin here.
                "datum": [datum_lat, datum_lon, datum_yaw],
                # magnetic_declination_radians: set for your site from
                # https://www.ngdc.noaa.gov/geomag/calculators/magcalc.shtml
                # Affects heading derived from GNSS COG. 0.0 = no correction.
                "magnetic_declination_radians": float(
                    navsat_cfg.get("magnetic_declination_radians", 0.0)
                ),
                # yaw_offset: additional heading rotation applied to the IMU
                # yaw before passing to navsat_transform. Use when the IMU x-axis
                # does not point forward along the vehicle longitudinal axis.
                "yaw_offset": float(navsat_cfg.get("yaw_offset", 0.0)),
                # zero_altitude: collapse z to zero (2D parking lot, no terrain).
                "zero_altitude": True,
                # broadcast_utm_transform: publish the UTM -> odom TF if needed
                # for diagnostics in rviz. Not required for the EKF pipeline.
                "broadcast_utm_transform": navsat_cfg.get(
                    "broadcast_utm_transform", False
                ),
                # publish_filtered_gps: republish EKF-corrected position as
                # NavSatFix for diagnostics. Disabled to reduce topic clutter.
                "publish_filtered_gps": navsat_cfg.get("publish_filtered_gps", False),
                # use_odometry_yaw: use EKF yaw rather than GNSS COG for the
                # local frame heading. False: navsat_transform derives yaw from
                # IMU on startup, then updates from GNSS COG when moving.
                "use_odometry_yaw": navsat_cfg.get("use_odometry_yaw", False),
                # wait_for_datum: block until datum is received rather than
                # auto-latching on first fix. False: use the datum parameter above.
                "wait_for_datum": navsat_cfg.get("wait_for_datum", False),
                "frequency": float(navsat_cfg.get("frequency", 20.0)),
                "delay": float(navsat_cfg.get("delay", 3.0)),
            }
        ],
        remappings=[
            # navsat_transform subscribes to these topics by default name;
            # remap to the actual physical driver topic names.
            ("imu/data", imu_topic),
            ("gps/fix", gnss_fix_topic),
            # Output: metric Odometry consumed by the EKF as odom0.
            ("odometry/gps", gnss_odom_topic),
        ],
    )

    # -- EKF node --------------------------------------------------------------
    # Same parameters as sim except:
    #   - use_sim_time=False
    #   - base_link_frame=base_link (ROS convention; sim uses ego_vehicle)
    #   - imu0 / odom0 remapped to real hardware topics
    #   - pose0 (COG heading) removed: navsat_transform handles heading
    #     internally via the IMU, so no separate heading correction is needed
    ekf_params = {
        **ros2_cfg.get("ekf", {}),
        "use_sim_time": False,
        "base_link_frame": "base_link",
        "world_frame": "odom",
        "odom0": gnss_odom_topic,
        "imu0": imu_topic,
        # Remove sim-only pose0 (COG heading from GnssNoiseRelayNode).
        # navsat_transform already accounts for IMU yaw in its projection.
        "pose0": "",
    }

    ekf_node = Node(
        package="robot_localization",
        executable="ekf_node",
        name="ekf_filter_node",
        parameters=[ekf_params],
        remappings=[
            ("odometry/filtered", odom_filtered_topic),
        ],
    )

    # -- CovarianceExtractorNode -----------------------------------------------
    # Identical to sim: subscribes to /odometry/filtered, writes ekf_state.json.
    # inference_loop.py reads ekf_state.json via _CovarianceSubscriber --
    # no change needed on the Python side.
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
