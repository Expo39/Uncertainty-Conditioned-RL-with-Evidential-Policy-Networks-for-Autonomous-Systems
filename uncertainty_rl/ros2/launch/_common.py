"""
@file _common.py
@brief Shared helpers for carla_bridge.launch.py and real_vehicle.launch.py.
"""

import os
from pathlib import Path
from typing import List

import yaml
from launch_ros.actions import Node


def load_yaml(path_str: str, env_override: str = "") -> dict:
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


def static_tf(
    name: str,
    parent: str,
    child: str,
    x: float,
    y: float,
    z: float,
    use_sim_time: bool = False,
) -> Node:
    """
    @brief Create a zero-rotation static_transform_publisher node.

    Always passes use_sim_time=False by default. Timestamp=0 from static
    publishers is valid at all times in TF2, so wall clock is correct here
    regardless of whether the rest of the stack uses sim time.

    @param name: Node name.
    @param parent: Parent TF frame ID.
    @param child: Child TF frame ID.
    @param x: Translation X (metres).
    @param y: Translation Y (metres).
    @param z: Translation Z (metres).
    @param use_sim_time: Passed through to the node parameter (default False).
    @return Configured static_transform_publisher Node.
    """
    return Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name=name,
        parameters=[{"use_sim_time": use_sim_time}],
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


def build_sensor_tf_nodes(
    sensors_config: dict,
    body_frame: str = "ego_vehicle",
    imu_frame: str = "ego_vehicle/imu",
    lidar_frame: str = "ego_vehicle/lidar",
    gnss_frame: str = "ego_vehicle/gnss",
    use_sim_time: bool = False,
) -> List[Node]:
    """
    @brief Build static TF nodes connecting sensor frames to the vehicle body frame.

    @param sensors_config: Dict with imu/lidar/gnss mount sub-dicts.
    @param body_frame: Vehicle body frame ID.
    @param imu_frame: IMU child frame ID.
    @param lidar_frame: LiDAR child frame ID.
    @param gnss_frame: GNSS child frame ID.
    @param use_sim_time: Forwarded to static_tf() (default False).
    @return List of static TF publisher nodes.
    """
    # Sim: sensors_config is carla_sensors sub-dict from env_config.yaml.
    # Real: sensors_config is sensors sub-dict from sensor_config.yaml.
    # Both use the same mount key structure: {imu: {mount: {x,y,z}}, ...}.
    imu_mount = sensors_config.get("imu", {}).get("mount", {})
    lidar_mount = sensors_config.get("lidar", {}).get("mount", {})
    gnss_mount = sensors_config.get("gnss", {}).get("mount", {})

    return [
        static_tf(
            "body_to_imu_tf",
            body_frame,
            imu_frame,
            float(imu_mount.get("x", 0.0)),
            float(imu_mount.get("y", 0.0)),
            float(imu_mount.get("z", 0.3)),
            use_sim_time=use_sim_time,
        ),
        static_tf(
            "body_to_lidar_tf",
            body_frame,
            lidar_frame,
            float(lidar_mount.get("x", 2.4)),
            float(lidar_mount.get("y", 0.0)),
            float(lidar_mount.get("z", 0.5)),
            use_sim_time=use_sim_time,
        ),
        static_tf(
            "body_to_gnss_tf",
            body_frame,
            gnss_frame,
            float(gnss_mount.get("x", 0.75)),
            float(gnss_mount.get("y", 0.0)),
            float(gnss_mount.get("z", 1.6)),
            use_sim_time=use_sim_time,
        ),
    ]
