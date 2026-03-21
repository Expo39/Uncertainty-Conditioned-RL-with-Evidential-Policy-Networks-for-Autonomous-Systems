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
3. Cartographer (scan-matching on PointCloud2; publishes /scan_matched_odometry)
4. robot_localisation EKF (fuses IMU + scan-matched odometry -> /odometry/filtered)
5. CovarianceExtractorNode (extracts 3x3 [x, y, yaw] covariance + velocity ->
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
        # Fallback: check relative to this file (for local development)
        path = Path(__file__).resolve().parents[3] / "configs" / path.name

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

    Mount positions are read from train_config.yaml (carla_sensors section).
    On the real car, update mount values to match physical sensor positions.

    @param sensors_config: carla_sensors dict from train_config.yaml.
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
    return configs[(mode, is_3d)]


def generate_launch_description() -> LaunchDescription:
    """
    @brief Generate launch description for the full CARLA + EKF + covariance stack.
    @return LaunchDescription with all nodes and launch arguments.
    """
    ros2_config = _load_yaml("/workspace/configs/ros2_config.yaml", "ROS2_CONFIG_PATH")
    train_config = _load_yaml("/workspace/configs/train_config.yaml")

    # -- Environment variables ---------------------------------------------

    sensor_suite = os.environ.get("SENSOR_SUITE", "suite_a")
    cartographer_mode = os.environ.get("CARTOGRAPHER_MODE", "slam")
    cartographer_map = os.environ.get("CARTOGRAPHER_MAP", "")
    is_3d = sensor_suite in ("suite_b", "suite_c")

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
            default_value="Town01",
            description="CARLA town/map to load.",
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
                "synchronous_mode": "true",
                "fixed_delta_seconds": "0.05",
                "passive": "true",
                "register_all_sensors": "true",
                "timeout": "30",
            }.items(),
        )
    except Exception:
        carla_bridge = None

    # -- EKF node ----------------------------------------------------------

    ekf_node = Node(
        package="robot_localization",
        executable="ekf_node",
        name="ekf_filter_node",
        parameters=[ros2_config.get("ekf", {})],
        remappings=[("odometry/filtered", "/odometry/filtered")],
    )

    # -- Static TF: sensor mount tree --------------------------------------

    sensors_config = train_config.get("carla_sensors", {})
    static_tf_nodes = _build_sensor_tf_nodes(sensors_config, is_3d)

    # -- Cartographer node -------------------------------------------------

    cartographer_basename = _select_cartographer_config(cartographer_mode, is_3d)
    lidar_topic = "/carla/ego_vehicle/lidar_3d" if is_3d else "/carla/ego_vehicle/lidar"

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
        carto_args += [
            "-load_state_filename",
            cartographer_map,
            "-load_frozen_state",
            "true",
        ]

    cartographer_node = Node(
        package="cartographer_ros",
        executable="cartographer_node",
        name="cartographer_node",
        arguments=carto_args,
        remappings=[
            ("points2", lidar_topic),
            ("odom", "/scan_matched_odometry"),
        ],
    )

    # -- Covariance extractor node -----------------------------------------

    covariance_extractor = Node(
        package="uncertainty_rl_ros2",
        executable="covariance_extractor",
        name="covariance_extractor",
        parameters=[
            {
                "odom_topic": ros2_config.get("odom_topic", "/odometry/filtered"),
                "covariance_topic": ros2_config.get(
                    "covariance_topic", "/ekf_uncertainty/covariance"
                ),
                "publish_rate": ros2_config.get("publish_rate", 10.0),
            }
        ],
    )

    # -- Assemble launch description ---------------------------------------

    actions = [
        *launch_args,
        ekf_node,
        *static_tf_nodes,
        cartographer_node,
        covariance_extractor,
    ]

    if cartographer_mode == "slam":
        actions.append(
            Node(
                package="cartographer_ros",
                executable="cartographer_occupancy_grid_node",
                name="cartographer_occupancy_grid_node",
                parameters=[{"resolution": 0.05}],
            )
        )

    if carla_bridge is not None:
        actions.insert(len(launch_args), carla_bridge)

    return LaunchDescription(actions)
