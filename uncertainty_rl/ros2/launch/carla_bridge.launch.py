"""
@file carla_bridge.launch.py
@brief Launch file for CARLA bridge, Cartographer, robot_localisation EKF,
       and covariance extraction.

The bridge runs in **passive mode** (passive=True): it does NOT call world.tick().
Only the training container (CARLAParkingEnv.step -> world.tick) advances the
simulation. The bridge auto-discovers sensors spawned by the training container
(register_all_sensors=True) and publishes their data as ROS 2 topics.

Orchestrates the full sensor-to-covariance pipeline:
1. CARLA ROS bridge (passive; publishes sensor data from CARLA to ROS 2 topics)
2. Static TF publishers (connect ego_vehicle body frame to all sensor frames)
3. Cartographer (scan-matching on raw PointCloud2; publishes TF odom -> tracking_frame)
4. TfToOdomNode (converts Cartographer TF to nav_msgs/Odometry on /scan_matched_odometry)
5. robot_localisation EKF (fuses IMU + scan-matched odometry -> /odometry/filtered)
6. CovarianceExtractorNode (extracts 3x3 [x, y, yaw] covariance + velocity ->
   /ekf_uncertainty/covariance for the training container)

@note The CARLA ROS bridge in passive mode publishes sensor TF frames directly
      under "map" (e.g. map -> ego_vehicle/lidar) without an intermediate
      "ego_vehicle" body frame. Static TF publishers create the body frame:
      ego_vehicle/lidar -> ego_vehicle -> ego_vehicle/imu (+ lidar_3d for Suite B/C).

Cartographer config selected by SENSOR_SUITE and CARTOGRAPHER_MODE (4 configs):
  slam + suite_a   -> cartographer_config.lua
  slam + suite_b/c -> cartographer_config_3d.lua
  loc  + suite_a   -> cartographer_config_loc.lua
  loc  + suite_b/c -> cartographer_config_3d_loc.lua

Environment variables:
  CARTOGRAPHER_MODE  slam (default) or loc (pure localisation)
  CARTOGRAPHER_MAP   path to .pbstream (required when CARTOGRAPHER_MODE=loc)
  SENSOR_SUITE       suite_a (default), suite_b, or suite_c

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
) -> Node:
    """
    @brief Create a static_transform_publisher node with zero rotation.
    @param name: Node name.
    @param parent: Parent TF frame ID.
    @param child: Child TF frame ID.
    @param x: Translation X (metres).
    @param y: Translation Y (metres).
    @param z: Translation Z (metres).
    @return Node for the static transform publisher.
    """
    return Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name=name,
        parameters=[{"use_sim_time": True}],
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


def _build_sensor_tf_nodes(sensors_config: Dict, is_3d: bool) -> List[Node]:
    """
    @brief Build static TF nodes connecting sensor frames to the ego_vehicle body frame.

    The CARLA bridge publishes each sensor as a separate TF tree under "map".
    These static TFs create a unified tree:
      ego_vehicle/lidar -> ego_vehicle -> ego_vehicle/imu
                                       -> ego_vehicle/lidar_3d (Suite B/C)

    Mount positions are read from env_config.yaml (carla_sensors section).
    On the real car, update mount values to match physical sensor positions.

    @param sensors_config: carla_sensors dict from env_config.yaml.
    @param is_3d: True if using Suite B/C (3D LiDAR).
    @return List of static TF publisher nodes.
    """
    lidar_mount = sensors_config.get("lidar", {}).get("mount", {})
    imu_mount = sensors_config.get("imu", {}).get("mount", {})

    nodes = [
        # ego_vehicle/lidar -> ego_vehicle (inverse of LiDAR mount offset)
        _static_tf(
            "lidar_to_body_tf",
            "ego_vehicle/lidar",
            "ego_vehicle",
            -float(lidar_mount.get("x", 2.4)),
            -float(lidar_mount.get("y", 0.0)),
            -float(lidar_mount.get("z", 0.5)),
        ),
        # ego_vehicle -> ego_vehicle/imu
        _static_tf(
            "body_to_imu_tf",
            "ego_vehicle",
            "ego_vehicle/imu",
            float(imu_mount.get("x", 0.0)),
            float(imu_mount.get("y", 0.0)),
            float(imu_mount.get("z", 0.3)),
        ),
    ]

    if is_3d:
        lidar3d_mount = sensors_config.get("lidar_3d", {}).get("mount", {})
        nodes.append(
            _static_tf(
                "body_to_lidar3d_tf",
                "ego_vehicle",
                "ego_vehicle/lidar_3d",
                float(lidar3d_mount.get("x", -0.5)),
                float(lidar3d_mount.get("y", 0.0)),
                float(lidar3d_mount.get("z", 1.9)),
            )
        )

    return nodes


def _select_cartographer_config(mode: str, is_3d: bool) -> str:
    """
    @brief Select the Cartographer Lua config basename.
    @param mode: "slam" or "loc".
    @param is_3d: True if using Suite B/C (3D LiDAR).
    @return Lua config file basename.
    """
    configs = {
        ("slam", False): "cartographer_config.lua",
        ("slam", True): "cartographer_config_3d.lua",
        ("loc", False): "cartographer_config_loc.lua",
        ("loc", True): "cartographer_config_3d_loc.lua",
    }
    key = (mode, is_3d)
    if key not in configs:
        raise ValueError(
            f"Unknown CARTOGRAPHER_MODE '{mode}'. "
            "Expected 'slam' or 'loc'."
        )
    return configs[key]


def generate_launch_description() -> LaunchDescription:
    """
    @brief Generate launch description for the full CARLA + EKF + covariance stack.
    @return LaunchDescription with all nodes and launch arguments.
    """
    ros2_config = _load_yaml("/workspace/configs/ros2_config.yaml", "ROS2_CONFIG_PATH")
    env_config = _load_yaml("/workspace/configs/carla/env_config.yaml")

    # -- Environment variables ---------------------------------------------

    sensor_suite = os.environ.get("SENSOR_SUITE", "suite_a")
    cartographer_mode = os.environ.get("CARTOGRAPHER_MODE", "slam")
    cartographer_map = os.environ.get("CARTOGRAPHER_MAP", "")
    is_3d = sensor_suite in ("suite_b", "suite_c")

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
                # Sensor publishing is unaffected: register_all_sensors=True
                # discovers all sensors regardless of actor role_name.
                "ego_vehicle_role_name": "hero",
            }.items(),
        )
    except Exception:
        carla_bridge = None

    # -- EKF node ----------------------------------------------------------

    # Spread into a new dict to avoid mutating the live ros2_config object.
    ekf_params = {**ros2_config.get("ekf", {}), "use_sim_time": True}

    ekf_node = Node(
        package="robot_localization",
        executable="ekf_node",
        name="ekf_filter_node",
        parameters=[ekf_params],
    )

    # -- Static TF: sensor mount tree --------------------------------------

    sensors_config = env_config.get("carla_sensors", {})
    static_tf_nodes = _build_sensor_tf_nodes(sensors_config, is_3d)

    # All suites feed raw PointCloud2 directly to Cartographer (num_point_clouds=1).
    # CARLA's ray_cast does not collide with the parent actor, so 360 deg is used
    # for both mapping and localisation, ensuring consistency. On the real robot,
    # rebuild the pbstream with the physical sensor's native FOV.

    # -- Cartographer node -------------------------------------------------

    cartographer_basename = _select_cartographer_config(cartographer_mode, is_3d)

    carto_args = [
        "-configuration_directory",
        "/workspace/configs/cartographer",
        "-configuration_basename",
        cartographer_basename,
    ]
    if cartographer_mode == "loc":
        if not cartographer_map:
            raise RuntimeError(
                "CARTOGRAPHER_MODE=loc requires CARTOGRAPHER_MAP env var "
                "pointing to a .pbstream file. Run `make docker-map` first."
            )
        if not os.path.isfile(cartographer_map):
            raise RuntimeError(
                f"CARTOGRAPHER_MAP pbstream not found: '{cartographer_map}'. "
                f"Run `make docker-map LAYOUT=<layout>` to generate it first."
            )
        carto_args += [
            "-load_state_filename",
            cartographer_map,
            "-load_frozen_state",
            "true",
        ]

    # All suites feed raw PointCloud2 to Cartographer on the "points2" topic.
    # Suite A: single-channel 2D LiDAR at /carla/ego_vehicle/lidar (360 deg).
    # Suite B/C: 16-channel 3D LiDAR at /carla/ego_vehicle/lidar_3d (360 deg).
    lidar_topic = (
        "/carla/ego_vehicle/lidar_3d" if is_3d else "/carla/ego_vehicle/lidar"
    )
    # No odom input remapping: Cartographer uses LiDAR + IMU only.
    # The "odom" remapping would only be needed if feeding an external odometry
    # source into Cartographer, which we do not do.
    carto_remappings = [
        ("points2", lidar_topic),
        ("imu", "/carla/ego_vehicle/imu"),
    ]

    cartographer_node = Node(
        package="cartographer_ros",
        executable="cartographer_node",
        name="cartographer_node",
        parameters=[{"use_sim_time": True}],
        arguments=carto_args,
        remappings=carto_remappings,
    )

    # -- TF-to-Odometry bridge (Cartographer -> EKF) -------------------------
    #
    # Cartographer publishes its pose estimate via TF (odom -> tracking_frame)
    # but not as an Odometry topic. The EKF needs nav_msgs/Odometry on odom0.
    # This node bridges the gap and publishes dynamic covariance: inflated when
    # Cartographer loses scan-match lock (stale TF) or relocalises (TF jump).
    # This varying covariance propagates through the EKF and becomes the
    # localisation uncertainty signal in the RL policy observation.

    # tracking_frame depends on sensor suite: Suite A uses the 2D LiDAR frame,
    # Suite B/C use the 3D LiDAR frame (both are Cartographer's published_frame).
    tf_tracking_frame = (
        "ego_vehicle/lidar_3d" if is_3d else "ego_vehicle/lidar"
    )

    # Dynamic covariance parameters read from ros2_config.yaml (tf_to_odom section).
    # This allows real-robot tuning without touching Python source.
    tf_cfg = ros2_config.get("tf_to_odom", {})
    tf_to_odom_node = Node(
        package="uncertainty_rl_ros2",
        executable="tf_to_odom",
        name="tf_to_odom",
        parameters=[
            {
                "use_sim_time": True,
                "odom_frame": ros2_config.get("ekf", {}).get("odom_frame", "odom"),
                "tracking_frame": tf_tracking_frame,
                # Must match ekf.base_link_frame so robot_localisation correctly
                # interprets the velocity as expressed in the body frame.
                "body_frame": ros2_config.get("ekf", {}).get(
                    "base_link_frame", "ego_vehicle"
                ),
                "publish_topic": "/scan_matched_odometry",
                "publish_rate": ros2_config.get("ekf", {}).get("frequency", 20.0),
                "base_xy_variance": tf_cfg.get("base_xy_variance", 0.05),
                "base_yaw_variance": tf_cfg.get("base_yaw_variance", 0.05),
                "stale_threshold_sec": tf_cfg.get("stale_threshold_sec", 0.15),
                "staleness_scale": tf_cfg.get("staleness_scale", 100.0),
                "stale_max_sec": tf_cfg.get("stale_max_sec", 2.0),
                "jump_threshold_m": tf_cfg.get("jump_threshold_m", 1.0),
                "jump_scale": tf_cfg.get("jump_scale", 50.0),
                "jump_decay_steps": int(tf_cfg.get("jump_decay_steps", 10)),
            }
        ],
    )

    # -- Covariance extractor node -----------------------------------------

    covariance_extractor = Node(
        package="uncertainty_rl_ros2",
        executable="covariance_extractor",
        name="covariance_extractor",
        parameters=[
            {
                "use_sim_time": True,
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
    # Startup order matters: bridge must be up before Cartographer tries to
    # subscribe to sensor topics; static TFs must exist before the EKF starts.

    actions = [*launch_args]

    # CARLA ROS bridge first so sensor topics exist when Cartographer starts.
    if carla_bridge is not None:
        actions.append(carla_bridge)

    # Static TFs before the EKF and Cartographer so the frame tree is complete.
    actions.extend(static_tf_nodes)

    actions.extend([
        cartographer_node,
        tf_to_odom_node,
        ekf_node,
        covariance_extractor,
    ])

    # Occupancy grid node only needed during SLAM mapping, not training.
    if cartographer_mode == "slam":
        actions.append(
            Node(
                package="cartographer_ros",
                executable="cartographer_occupancy_grid_node",
                name="cartographer_occupancy_grid_node",
                parameters=[{"use_sim_time": True, "resolution": 0.05}],
            )
        )

    return LaunchDescription(actions)
