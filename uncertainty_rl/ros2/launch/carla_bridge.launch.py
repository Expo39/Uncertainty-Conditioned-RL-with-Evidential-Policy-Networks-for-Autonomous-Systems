"""
@file carla_bridge.launch.py
@brief Launch file for CARLA bridge, RTK-GNSS+IMU EKF, and covariance extraction.

The bridge runs in **passive mode** (passive=True): it does NOT call world.tick().
Only the training container (CARLAParkingEnv.step -> world.tick) advances the
simulation. The bridge auto-discovers sensors spawned by the training container
(register_all_sensors=True) and publishes their data as ROS 2 topics.

Orchestrates the full sensor-to-covariance pipeline:
1. CARLA ROS bridge (passive; publishes sensor data from CARLA to ROS 2 topics)
2. Static TF publishers (sensor frames to vehicle body)
3. GnssNoiseRelayNode (adds per-episode noise to GNSS, stamps covariance)
4. navsat_transform_node (UTM projection: NavSatFix -> Odometry)
5. robot_localisation EKF (fuses IMU + GNSS odometry -> /odometry/filtered)
6. CovarianceExtractorNode (extracts 3x3 [x, y, yaw] covariance + velocity ->
   ekf_state.json for the training container)

TF tree (ego_vehicle is root):
  ego_vehicle -> ego_vehicle/imu
  ego_vehicle -> ego_vehicle/lidar
  ego_vehicle -> ego_vehicle/gnss

@note The CARLA ROS bridge in passive mode publishes each sensor as a separate
      TF tree under "map". Static TF publishers create the unified body frame.

@author Antonio Galdes
"""

import os
from pathlib import Path
from typing import Dict, List

import yaml
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _load_yaml(path_str: str, env_override: str = "") -> dict:
    """
    @brief Load a YAML config, optionally overridden by an env var.
    @param path_str: Default file path inside the container.
    @param env_override: Env var name that overrides the path (optional).
    @return Parsed dictionary, or empty dict if not found.
    """
    resolved = os.environ.get(env_override, path_str) if env_override else path_str
    path = Path(resolved)

    if not path.exists():
        # Fallback: resolve relative to the project root for local development.
        # Preserve any subdirectory structure from the original path by taking
        # the portion after "configs/" rather than just path.name.
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
    use_sim_time: bool = True,
) -> Node:
    """
    @brief Create a static_transform_publisher node with zero rotation.
    @param name: Node name.
    @param parent: Parent TF frame ID.
    @param child: Child TF frame ID.
    @param x: Translation X (metres).
    @param y: Translation Y (metres).
    @param z: Translation Z (metres).
    @param use_sim_time: Whether to use sim time (from ros2_config.yaml).
    @return Node for the static transform publisher.
    """
    return Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name=name,
        # Static TFs must use wall clock (use_sim_time=False) so they publish
        # with timestamp=0 (valid at all times in TF2). With use_sim_time=True
        # they would publish at the current sim time which the EKF
        # (also on sim time) would accept -- but only after /clock arrives.
        # Using False ensures the static TF is available immediately at startup
        # before /clock is published, and TF2 treats timestamp=0 as eternal.
        parameters=[{"use_sim_time": False}],
        arguments=[
            "--x",
            str(x),
            "--y",
            str(y),
            "--z",
            str(z),
            "--roll",
            "0",
            "--pitch",
            "0",
            "--yaw",
            "0",
            "--frame-id",
            parent,
            "--child-frame-id",
            child,
        ],
    )


def _build_sensor_tf_nodes(
    sensors_config: Dict, use_sim_time: bool = True
) -> List[Node]:
    """
    @brief Build static TF nodes connecting sensor frames to the ego_vehicle body frame.

    The CARLA bridge publishes each sensor as a separate TF tree under "map".
    These static TFs create a unified tree:
      ego_vehicle -> ego_vehicle/imu
      ego_vehicle -> ego_vehicle/lidar
      ego_vehicle -> ego_vehicle/gnss

    Mount positions are read from env_config.yaml (carla_sensors section).
    On the real car, update mount values to match physical sensor positions.

    @param sensors_config: carla_sensors dict from env_config.yaml.
    @param use_sim_time: Whether to use sim time (from ros2_config.yaml).
    @return List of static TF publisher nodes.
    """
    imu_mount = sensors_config.get("imu", {}).get("mount", {})
    lidar_mount = sensors_config.get("lidar", {}).get("mount", {})
    gnss_mount = sensors_config.get("gnss", {}).get("mount", {})

    return [
        # ego_vehicle -> ego_vehicle/imu
        _static_tf(
            "body_to_imu_tf",
            "ego_vehicle",
            "ego_vehicle/imu",
            float(imu_mount.get("x", 0.0)),
            float(imu_mount.get("y", 0.0)),
            float(imu_mount.get("z", 0.3)),
            use_sim_time=use_sim_time,
        ),
        # ego_vehicle -> ego_vehicle/lidar (obstacle detection only)
        _static_tf(
            "body_to_lidar_tf",
            "ego_vehicle",
            "ego_vehicle/lidar",
            float(lidar_mount.get("x", 2.4)),
            float(lidar_mount.get("y", 0.0)),
            float(lidar_mount.get("z", 0.5)),
            use_sim_time=use_sim_time,
        ),
        # ego_vehicle -> ego_vehicle/gnss
        _static_tf(
            "body_to_gnss_tf",
            "ego_vehicle",
            "ego_vehicle/gnss",
            float(gnss_mount.get("x", 0.0)),
            float(gnss_mount.get("y", 0.0)),
            float(gnss_mount.get("z", 1.8)),
            use_sim_time=use_sim_time,
        ),
    ]


def generate_launch_description() -> LaunchDescription:
    """
    @brief Generate launch description for the CARLA + GNSS + EKF + covariance stack.
    @return LaunchDescription with all nodes and launch arguments.
    """
    ros2_config = _load_yaml("/workspace/configs/ros2_config.yaml", "ROS2_CONFIG_PATH")
    env_config = _load_yaml("/workspace/configs/carla/env_config.yaml")

    # Time source for all ROS 2 nodes. Read from ros2_config.yaml so it can be
    # changed without modifying Python source. Default true: CARLA bridge runs
    # with use_sim_time=true and publishes /clock, so all nodes that do TF
    # lookups against bridge-published transforms must use the same time source.
    use_sim_time: bool = ros2_config.get("use_sim_time", True)

    # synchronous_mode has NO effect when passive=true: the bridge skips
    # apply_settings entirely and never registers as a synchronous CARLA client.
    # The training container's world.tick() is the sole tick driver.
    bridge_sync_mode = "false"

    # -- Launch arguments --------------------------------------------------

    launch_args = [
        DeclareLaunchArgument(
            "carla_host",
            default_value=os.environ.get("CARLA_HOST", "carla-server"),
            description="CARLA server hostname.",
        ),
        DeclareLaunchArgument(
            "carla_port",
            default_value=os.environ.get("CARLA_PORT", "2000"),
            description="CARLA server port.",
        ),
        DeclareLaunchArgument(
            "town",
            default_value=env_config.get("town", "FlatPlane"),
            description="CARLA town/map to load (read from env_config.yaml).",
        ),
    ]

    # -- CARLA ROS bridge --------------------------------------------------

    try:
        from ament_index_python.packages import get_package_share_directory

        bridge_share = get_package_share_directory("carla_ros_bridge")
        carla_bridge = IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(bridge_share, "carla_ros_bridge.launch.py")
            ),
            launch_arguments={
                "host": LaunchConfiguration("carla_host"),
                "port": LaunchConfiguration("carla_port"),
                "town": LaunchConfiguration("town"),
                # No effect when passive=true (bridge never calls world.tick()).
                "synchronous_mode": bridge_sync_mode,
                "fixed_delta_seconds": "0.05",
                "passive": "true",
                "register_all_sensors": "true",
                "timeout": "30",
                # Prevent the bridge from sending vehicle_control_cmd to the ego
                # vehicle and overriding the training container's apply_control().
                # The bridge matches control targets by role_name; "hero" does not
                # match our "ego_vehicle" actor, so no controls are injected.
                "ego_vehicle_role_name": "hero",
                # Publish CARLA sim time on /clock so all ROS nodes using
                # use_sim_time=true have a consistent time source. Without this
                # the EKF waits forever for /clock and never starts.
                "publish_clock": "true",
            }.items(),
        )
    except Exception:
        carla_bridge = None

    # -- EKF node ----------------------------------------------------------

    # Spread into a new dict to avoid mutating the live ros2_config object.
    ekf_params = {**ros2_config.get("ekf", {}), "use_sim_time": use_sim_time}

    ekf_node = Node(
        package="robot_localization",
        executable="ekf_node",
        name="ekf_filter_node",
        parameters=[ekf_params],
    )

    # -- Static TF: sensor mount tree --------------------------------------

    sensors_config = env_config.get("carla_sensors", {})
    static_tf_nodes = _build_sensor_tf_nodes(sensors_config, use_sim_time)

    # -- GNSS noise relay node ---------------------------------------------
    # Adds per-episode noise to CARLA GNSS and stamps position_covariance
    # for navsat_transform_node. Noise tier signalled via gnss_noise_config.json.

    gnss_relay_cfg = ros2_config.get("gnss_noise_relay", {})
    gnss_noise_relay_node = Node(
        package="uncertainty_rl_ros2",
        executable="gnss_noise_relay",
        name="gnss_noise_relay",
        parameters=[
            {
                "use_sim_time": use_sim_time,
                "input_topic": gnss_relay_cfg.get(
                    "input_topic", "/carla/ego_vehicle/gnss"
                ),
                "output_topic": gnss_relay_cfg.get("output_topic", "/gnss/noisy"),
                "base_metric_stddev_m": gnss_relay_cfg.get(
                    "base_metric_stddev_m", 0.02
                ),
            }
        ],
    )

    # -- navsat_transform_node (robot_localisation package) -----------------
    # Converts NavSatFix (/gnss/noisy) -> Odometry (/odometry/gps) via UTM.

    navsat_cfg = ros2_config.get("navsat_transform", {})
    navsat_transform_node = Node(
        package="robot_localization",
        executable="navsat_transform_node",
        name="navsat_transform_node",
        parameters=[
            {
                "use_sim_time": use_sim_time,
                "frequency": navsat_cfg.get("frequency", 20.0),
                "zero_altitude": navsat_cfg.get("zero_altitude", True),
                "publish_filtered_gps": navsat_cfg.get(
                    "publish_filtered_gps", False
                ),
                "use_odometry_yaw": navsat_cfg.get("use_odometry_yaw", False),
                "yaw_offset": navsat_cfg.get("yaw_offset", 0.0),
                "wait_for_datum": navsat_cfg.get("wait_for_datum", False),
                "broadcast_utm_transform": navsat_cfg.get(
                    "broadcast_utm_transform", True
                ),
            }
        ],
        remappings=[
            ("gps/fix", "/gnss/noisy"),
            ("imu", "/carla/ego_vehicle/imu"),
            ("odometry/filtered", "/odometry/filtered"),
        ],
    )

    # -- Covariance extractor node -----------------------------------------

    covariance_extractor = Node(
        package="uncertainty_rl_ros2",
        executable="covariance_extractor",
        name="covariance_extractor",
        parameters=[
            {
                "use_sim_time": use_sim_time,
                "odom_topic": ros2_config.get("odom_topic", "/odometry/filtered"),
                "covariance_topic": ros2_config.get(
                    "covariance_topic", "/ekf_uncertainty/covariance"
                ),
                "publish_rate": ros2_config.get("publish_rate", 10.0),
                # When world_frame=odom, robot_localisation publishes twist in
                # the odom (world-aligned) frame. The extractor rotates it into
                # the vehicle body frame before writing to ekf_state.json.
                "twist_in_odom_frame": ros2_config.get("twist_in_odom_frame", True),
            }
        ],
    )

    # -- Assemble launch description ---------------------------------------
    # Startup order matters: bridge must be up before navsat_transform tries to
    # subscribe to sensor topics; static TFs must exist before the EKF starts.

    actions = [*launch_args]

    # CARLA ROS bridge first so sensor topics exist.
    if carla_bridge is not None:
        actions.append(carla_bridge)

    # Static TFs before the EKF so the frame tree is complete.
    actions.extend(static_tf_nodes)

    actions.extend(
        [
            gnss_noise_relay_node,
            navsat_transform_node,
            ekf_node,
            covariance_extractor,
        ]
    )

    return LaunchDescription(actions)
