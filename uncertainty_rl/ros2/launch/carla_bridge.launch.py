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
3. GnssNoiseRelayNode (adds per-episode noise to GNSS, converts lat/lon to local
   XY via flat-earth projection, publishes Odometry on /odometry/gps; also relays
   IMU with realistic angular_velocity_covariance stamped)
4. robot_localisation EKF (fuses IMU/stamped + GNSS odometry -> /odometry/filtered)
5. CovarianceExtractorNode (extracts 3x3 [x, y, yaw] covariance + velocity ->
   ekf_state.json for the training container)

TF tree (map is root):
  map -> ego_vehicle          (static identity -- bridge does not publish this)
  map -> odom                 (static identity -- for nav_msgs consumers)
  map -> ego_vehicle/imu      (dynamic, published by CARLA bridge ImuSensor)
  map -> ego_vehicle/gnss     (dynamic, published by CARLA bridge GnssSensor)
  map -> ego_vehicle/lidar    (dynamic, published by CARLA bridge LidarSensor)
  odom -> ego_vehicle         (dynamic, published by EKF as its filtered output)

@note The CARLA bridge in passive mode publishes sensor frame TFs under "map"
      but does NOT publish map->ego_vehicle (no TFSensor pseudo-actor is spawned).
      A static identity map->ego_vehicle TF is published so the EKF can resolve
      sensor offsets relative to base_link_frame=ego_vehicle.

@author Antonio Galdes
"""

import os
from pathlib import Path
from typing import Dict, List

import yaml
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription
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
    sensors_config: Dict,
    use_sim_time: bool = True,
) -> List[Node]:
    """
    @brief Build static TF nodes connecting sensor frames to the ego_vehicle body frame.

    The CARLA bridge publishes each sensor as a separate TF tree under "map".
    These static TFs create a unified tree:
      ego_vehicle -> ego_vehicle/imu
      ego_vehicle -> ego_vehicle/lidar
      ego_vehicle -> ego_vehicle/gnss

    Mount positions come from carla_sensors.*.mount in env_config.yaml.
    These are the single source of truth for both simulation and the real
    vehicle -- update them directly when physical sensor positions are measured.

    @warning IMU lever arm from GNSS antenna must be accurate to ~2 cm.
             A 5 cm error causes ~5 cm * sin(heading_change) positional bias.

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
    sensor_config = _load_yaml(
        "/workspace/configs/deployment/sensor_config.yaml", "SENSOR_CONFIG_PATH"
    )
    # Merge: sensor_config provides shared keys, env_config overrides with
    # CARLA-specific keys. Matches load_env_config() in train_ppo.py.
    env_config = {
        **sensor_config,
        **_load_yaml("/workspace/configs/deployment/sim/env_config.yaml"),
    }

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
    # NOTE: The CARLA ROS bridge in passive (non-synchronous) mode publishes
    # ALL sensor TF as "map -> sensor_frame" (see sensor.py get_ros_transform).
    # It also publishes "map -> ego_vehicle" via the TFSensor pseudo-actor.
    # Our static TFs (ego_vehicle -> ego_vehicle/imu etc.) conflict with the
    # bridge's dynamic "map -> ego_vehicle/imu" transforms, causing TF2 to
    # reject one of the two transforms for each sensor frame (two-parent error).
    # The EKF can look up "ego_vehicle -> ego_vehicle/imu" via the bridge's
    # transforms using the path:
    #   ego_vehicle -> [inv(map->ego_vehicle)] -> map -> ego_vehicle/imu
    # This correctly yields the physical sensor offset. We therefore do NOT
    # publish static sensor TFs and let the bridge be the sole TF source.

    # Static TFs for EKF body frame and odom anchor.
    #
    # The CARLA bridge in passive mode only publishes sensor frame TFs, NOT
    # the vehicle body frame. Specifically, the bridge publishes:
    #   map -> ego_vehicle/imu   (ImuSensor, dynamic, via /tf)
    #   map -> ego_vehicle/gnss  (GnssSensor, dynamic, via /tf)
    #   map -> ego_vehicle/lidar (LidarSensor, dynamic, via /tf)
    # It does NOT publish map -> ego_vehicle. The TFSensor pseudo-actor is only
    # created when the bridge discovers a dedicated TF sensor actor attached to
    # the ego vehicle -- which we do not spawn.
    #
    # The EKF uses base_link_frame=ego_vehicle. Without ego_vehicle in the TF
    # tree, the EKF cannot look up the sensor offsets and never produces output.
    #
    # Fix: publish a static identity map -> ego_vehicle TF. This gives ego_vehicle
    # a stable root in the map frame. The EKF can then look up:
    #   ego_vehicle -> ego_vehicle/imu
    # via the path:
    #   ego_vehicle -> [inv(map->ego_vehicle)] -> map -> ego_vehicle/imu
    #
    # The EKF publishes odom -> ego_vehicle as its output. The static map -> odom
    # identity TF lets any nav_msgs consumer that needs map->odom find it.
    static_tf_nodes = [
        # Anchors ego_vehicle at the map origin. The EKF will override this by
        # publishing odom -> ego_vehicle as it fuses IMU and GNSS data.
        _static_tf(
            "map_to_ego_vehicle_tf",
            "map",
            "ego_vehicle",
            0.0,
            0.0,
            0.0,
            use_sim_time=False,
        ),
        # Identity map -> odom for consumers that need the full map->odom->body chain.
        _static_tf(
            "map_to_odom_tf",
            "map",
            "odom",
            0.0,
            0.0,
            0.0,
            use_sim_time=False,
        ),
    ]

    # -- Sensor relay node -------------------------------------------------
    # Co-spins GnssNoiseRelayNode and ImuNoiseRelayNode in a single process.
    # GNSS: adds per-episode noise, flat-earth projection -> /odometry/gps.
    # IMU: stamps realistic angular_velocity_covariance -> /imu/stamped.

    gnss_relay_cfg = ros2_config.get("gnss_noise_relay", {})
    carla_topics = ros2_config.get("carla_topics", {})

    # Datum lat/lon for flat-earth projection (from env_config.yaml).
    datum_lat: float = float(env_config.get("gnss_datum_lat", 0.0))
    datum_lon: float = float(env_config.get("gnss_datum_lon", 0.0))

    sensor_relay_node = Node(
        package="uncertainty_rl_ros2",
        executable="sensor_relay",
        name="sensor_relay",
        parameters=[
            {
                "use_sim_time": use_sim_time,
                # GNSS relay parameters
                "input_topic": gnss_relay_cfg.get(
                    "input_topic", "/carla/ego_vehicle/gnss"
                ),
                "output_topic": gnss_relay_cfg.get("output_topic", "/gnss/noisy"),
                "base_metric_stddev_m": gnss_relay_cfg.get(
                    "base_metric_stddev_m", 0.02
                ),
                "enable_markov_transitions": gnss_relay_cfg.get(
                    "enable_markov_transitions", True
                ),
                "datum_lat": datum_lat,
                "datum_lon": datum_lon,
                "odom_output_topic": gnss_relay_cfg.get(
                    "odom_output_topic", "/odometry/gps"
                ),
                # IMU relay parameters
                "imu_input_topic": carla_topics.get(
                    "imu", "/carla/ego_vehicle/imu"
                ),
                "imu_output_topic": carla_topics.get(
                    "imu_stamped", "/carla/ego_vehicle/imu/stamped"
                ),
                "imu_gyro_variance": gnss_relay_cfg.get("imu_gyro_variance", 1.0e-7),
                "zupt_threshold_rad_s": gnss_relay_cfg.get(
                    "zupt_threshold_rad_s", 0.03
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
