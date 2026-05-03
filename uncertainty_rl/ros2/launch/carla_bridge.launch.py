"""
@file carla_bridge.launch.py
@brief Launch file for CARLA bridge, RTK-GNSS+IMU EKF, and covariance extraction.

The bridge runs in **passive mode** (passive=True): it does NOT call world.tick().
Only the training container (CARLAParkingEnv.step -> world.tick) advances the
simulation. The bridge auto-discovers sensors spawned by the training container
(register_all_sensors=True) and publishes their data as ROS 2 topics.
"""

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from launch._common import load_yaml, static_tf


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
    # Merge: sensor_config < agent_config < env_config (env wins on conflict).
    # Matches the three-layer merge in load_env_config() in train_ppo.py.
    env_config = {
        **sensor_config,
        **agent_config,
        **load_yaml("/workspace/configs/deployment/sim/env_config.yaml"),
    }

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

    static_tf_nodes = [
        # Anchors ego_vehicle at the map origin. The EKF will override this by
        # publishing odom -> ego_vehicle as it fuses IMU and GNSS data.
        static_tf("map_to_ego_vehicle_tf", "map", "ego_vehicle", 0.0, 0.0, 0.0),
        # Identity map -> odom for consumers that need the full map->odom->body chain.
        static_tf("map_to_odom_tf", "map", "odom", 0.0, 0.0, 0.0),
    ]

    # -- Sensor relay node -------------------------------------------------

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
                "base_metric_stddev_m": gnss_relay_cfg.get("base_metric_stddev_m", 0.02),
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
                "cog_min_speed_ms": gnss_relay_cfg.get("cog_min_speed_ms", 0.3),
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
                "use_sim_time": use_sim_time,
                "odom_topic": ros2_config.get("odom_topic", "/odometry/filtered"),
                "covariance_topic": ros2_config.get(
                    "covariance_topic", "/ekf_uncertainty/covariance"
                ),
                "publish_rate": ros2_config.get("publish_rate", 10.0),
                # robot_localisation publishes twist in the child frame (body
                # frame) per nav_msgs/Odometry convention. No rotation needed.
                "twist_in_odom_frame": ros2_config.get("twist_in_odom_frame", False),
            }
        ],
    )

    # -- Pipeline diagnostic: log topic status after 20 s ------------------
    # Prints which key topics are publishing so stalls can be diagnosed from
    # the ros2-bridge container logs. Safe to leave in permanently.
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

    # -- Assemble launch description ---------------------------------------
    # Startup order: bridge first (so sensor topics exist and /clock publishes),
    # then static TFs (so the EKF can resolve sensor frames), then the rest.

    actions = [*launch_args]

    # CARLA ROS bridge first so sensor topics exist.
    if carla_bridge is not None:
        actions.append(carla_bridge)

    # Static TFs before the EKF so the frame tree is complete.
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
