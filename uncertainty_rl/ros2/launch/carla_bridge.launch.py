"""
@file carla_bridge.launch.py
@brief Launch file for CARLA bridge, RTK-GNSS+IMU EKF, and covariance extraction.

The bridge runs passive: it never calls world.tick(), so the training container
(CARLAParkingEnv.step) remains the sole tick driver. It auto-discovers the
sensors that container spawns and republishes them as ROS 2 topics.
"""

import importlib.util
import os

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    IncludeLaunchDescription,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

# Load shared launch helpers by file path so the import is not shadowed by
# the installed ROS 2 'launch' package of the same name.
_common_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "_common.py")
_spec = importlib.util.spec_from_file_location("launch_common", _common_path)
assert _spec is not None, "Failed to load _common.py spec"
_common_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_common_mod)  # type: ignore[union-attr]
load_yaml = _common_mod.load_yaml
static_tf = _common_mod.static_tf


def generate_launch_description() -> LaunchDescription:
    """
    @brief Generate launch description for the CARLA + GNSS + EKF + covariance stack.
    @return LaunchDescription with all nodes and launch arguments.
    """
    ros2_config = load_yaml("/workspace/configs/ros2_config.yaml", "ROS2_CONFIG_PATH")
    sensor_config = load_yaml(
        "/workspace/configs/deployment/sensor_config.yaml", "SENSOR_CONFIG_PATH"
    )
    agent_config = load_yaml(
        "/workspace/configs/deployment/agent_config.yaml", "AGENT_CONFIG_PATH"
    )
    # Precedence must match the three-layer merge in load_env_config().
    env_config = {
        **sensor_config,
        **agent_config,
        **load_yaml("/workspace/configs/deployment/sim/env_config.yaml"),
    }

    use_sim_time: bool = ros2_config.get("use_sim_time", True)

    # Inert while passive=true: the bridge skips apply_settings entirely and
    # never registers as a synchronous CARLA client.
    bridge_sync_mode = "false"

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
                "synchronous_mode": bridge_sync_mode,
                "fixed_delta_seconds": "0.05",
                "passive": "true",
                "register_all_sensors": "true",
                "timeout": "30",
                # Deliberately NOT "ego_vehicle": the bridge matches control
                # targets by role_name, so a non-matching name stops it sending
                # vehicle_control_cmd over the training container's
                # apply_control().
                "ego_vehicle_role_name": "hero",
                # The EKF nodes run with use_sim_time, so without /clock they
                # would wait for a time source that never arrives.
                "publish_clock": "true",
            }.items(),
        )
    except Exception:
        carla_bridge = None

    # Spread into a new dict to avoid mutating the live ros2_config object.
    ekf_params = {**ros2_config.get("ekf", {}), "use_sim_time": use_sim_time}

    ekf_node = Node(
        package="robot_localization",
        executable="ekf_node",
        name="ekf_filter_node",
        parameters=[ekf_params],
    )

    static_tf_nodes = [
        # A placeholder anchor at the map origin: the EKF supersedes it once it
        # starts publishing odom -> ego_vehicle.
        static_tf("map_to_ego_vehicle_tf", "map", "ego_vehicle", 0.0, 0.0, 0.0),
        # Identity, for consumers needing the full map->odom->body chain.
        static_tf("map_to_odom_tf", "map", "odom", 0.0, 0.0, 0.0),
    ]

    gnss_relay_cfg = ros2_config.get("gnss_noise_relay", {})
    imu_relay_cfg = ros2_config.get("imu_noise_relay", {})

    datum_lat: float = float(env_config.get("gnss_datum_lat", 0.0))
    datum_lon: float = float(env_config.get("gnss_datum_lon", 0.0))

    sensor_relay_node = Node(
        package="uncertainty_rl_ros2",
        executable="sensor_relay",
        name="sensor_relay",
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
                "enable_gnss_noise": gnss_relay_cfg.get("enable_gnss_noise", True),
                "enable_markov_transitions": gnss_relay_cfg.get(
                    "enable_markov_transitions", True
                ),
                "datum_lat": datum_lat,
                "datum_lon": datum_lon,
                "odom_output_topic": gnss_relay_cfg.get(
                    "odom_output_topic", "/odometry/gps"
                ),
                "heading_output_topic": gnss_relay_cfg.get(
                    "heading_output_topic", "/gnss/heading"
                ),
                "cog_min_displacement_m": gnss_relay_cfg.get(
                    "cog_min_displacement_m", 0.05
                ),
                "enable_cog_heading": gnss_relay_cfg.get("enable_cog_heading", True),
                "enable_gnss_anisotropy": gnss_relay_cfg.get(
                    "enable_gnss_anisotropy", True
                ),
                "aniso_ratio_max": gnss_relay_cfg.get("aniso_ratio_max", 1.5),
                "gnss_dropout_probability": gnss_relay_cfg.get(
                    "gnss_dropout_probability", 0.02
                ),
                "imu_topic": imu_relay_cfg.get(
                    "imu_output_topic", "/carla/ego_vehicle/imu/stamped"
                ),
                "noise_profiles_path": gnss_relay_cfg.get(
                    "noise_profiles_path",
                    "/workspace/configs/deployment/sim/gnss_noise_profiles.yaml",
                ),
                "imu_input_topic": imu_relay_cfg.get(
                    "imu_input_topic", "/carla/ego_vehicle/imu"
                ),
                "imu_output_topic": imu_relay_cfg.get(
                    "imu_output_topic", "/carla/ego_vehicle/imu/stamped"
                ),
                "enable_imu_noise": imu_relay_cfg.get("enable_imu_noise", True),
                "imu_gyro_variance": imu_relay_cfg.get("imu_gyro_variance", 1.0e-7),
                "imu_accel_variance": imu_relay_cfg.get("imu_accel_variance", 3.76e-5),
                "zupt_threshold_rad_s": imu_relay_cfg.get("zupt_threshold_rad_s", 0.03),
                "accel_zupt_threshold_ms2": imu_relay_cfg.get(
                    "accel_zupt_threshold_ms2", 0.2
                ),
                "imu_gyro_scale_factor_limit": imu_relay_cfg.get(
                    "imu_gyro_scale_factor_limit", 0.005
                ),
                "imu_accel_scale_factor_limit": imu_relay_cfg.get(
                    "imu_accel_scale_factor_limit", 0.005
                ),
                # The master seed in agent_config.yaml is the one source shared
                # by training and this separate-process noise container, so
                # network init and noise realisations stay in step.
                "seed": agent_config.get("seed", 42),
            }
        ],
    )

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
                # Ignored by the node: robot_localisation already publishes
                # twist in the child (body) frame per nav_msgs/Odometry.
                "twist_in_odom_frame": ros2_config.get("twist_in_odom_frame", False),
            }
        ],
    )

    # Reports which key topics are publishing, so a stalled pipeline can be
    # diagnosed from the ros2-bridge container logs alone.
    pipeline_diag = ExecuteProcess(
        cmd=[
            "bash",
            "-c",
            "sleep 20 && source /opt/ros/jazzy/setup.bash && "
            "echo '=== EKF pipeline diagnostic (t+20s) ===' && "
            "echo '-- /clock hz:' && "
            "timeout 2 ros2 topic hz /clock --window 10 2>&1 | head -3 || "
            "echo 'SILENT'; "
            "echo '-- /carla/ego_vehicle/imu hz:' && "
            "timeout 2 ros2 topic hz /carla/ego_vehicle/imu --window 10 2>&1 | head -3 "
            "|| echo 'SILENT'; "
            "echo '-- /carla/ego_vehicle/imu/stamped hz:' && "
            "timeout 2 ros2 topic hz /carla/ego_vehicle/imu/stamped --window 10 2>&1 "
            "| head -3 || echo 'SILENT'; "
            "echo '-- /carla/ego_vehicle/imu frame_id:' && "
            "timeout 3 ros2 topic echo /carla/ego_vehicle/imu --once 2>&1 | "
            "grep frame_id | head -1 || echo 'no message'; "
            "echo '-- /gnss/noisy hz:' && "
            "timeout 2 ros2 topic hz /gnss/noisy --window 10 2>&1 | head -3 "
            "|| echo 'SILENT'; "
            "echo '-- /odometry/filtered hz:' && "
            "timeout 2 ros2 topic hz /odometry/filtered --window 10 2>&1 | head -3 "
            "|| echo 'SILENT'; "
            "echo '-- /odometry/gps hz:' && "
            "timeout 2 ros2 topic hz /odometry/gps --window 10 2>&1 | head -3 "
            "|| echo 'SILENT'; "
            "echo '-- /tf publishers (bridge dynamic TF frames):' && "
            "timeout 3 ros2 topic echo /tf --once 2>&1 | grep -E 'frame_id|child_frame' "
            "| head -20 || echo 'no /tf'; "
            "echo '-- /tf_static publishers:' && "
            "timeout 3 ros2 topic echo /tf_static --once 2>&1 | "
            "grep -E 'frame_id|child_frame' | head -20 || echo 'no /tf_static'; "
            "echo '-- EKF node log (last 10 lines):' && "
            "ros2 node info /ekf_filter_node 2>&1 | head -20 || echo 'EKF node not found'; "
            "echo '-- TF ego_vehicle -> ego_vehicle/imu:' && "
            "timeout 3 ros2 run tf2_ros tf2_echo ego_vehicle ego_vehicle/imu 2>&1 "
            "| head -5 || echo 'TF lookup failed'; "
            "echo '=== end diagnostic ==='",
        ],
        output="screen",
    )

    actions = [*launch_args]

    # Bridge first, so the sensor topics exist and /clock publishes.
    if carla_bridge is not None:
        actions.append(carla_bridge)

    # Then the static TFs, so the frame tree is complete before the EKF starts
    # trying to resolve sensor frames against it.
    actions.extend(static_tf_nodes)

    actions.extend(
        [
            sensor_relay_node,
            ekf_node,
            covariance_extractor,
            pipeline_diag,
        ]
    )

    return LaunchDescription(actions)
