"""
@file carla_parking.py
@brief CARLA parking environment with EKF covariance from robot_localisation.

This module implements a Gymnasium-compatible environment for autonomous parking
in CARLA simulator. Localisation uncertainty comes from the robot_localisation
EKF node (via ROS 2 DDS), driven by noisy CARLA sensors, weather conditions,
and dynamic traffic - not from a simulated noise model.

The training container subscribes to the covariance topic published by
CovarianceExtractorNode in the ros2-bridge container. Docker and ROS 2 are
always required; there is no standalone fallback for training.
"""

import logging
import random
import threading
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, cast

import gymnasium as gym
import numpy as np
from gymnasium import spaces

try:
    import carla
except ImportError:
    carla = None  # Running without CARLA (CI or tests)

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy
    from uncertainty_rl_msgs.msg import CovarianceEstimate

    _ROS2_AVAILABLE = True
except ImportError:
    _ROS2_AVAILABLE = False

from uncertainty_rl.utils.constants import (
    COVARIANCE_FEATURES_DIM,
    SUCCESS_THRESHOLD_ORIENTATION,
    SUCCESS_THRESHOLD_POSITION,
    TOTAL_OBS_DIM,
)
from uncertainty_rl.utils.covariance_utils import extract_2d_covariance_features

logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    # mypy always sees Node as the base class (rclpy is in ignore_missing_imports)
    from rclpy.node import Node as _NodeBase
else:
    # At runtime, fall back to object when rclpy is not installed
    _NodeBase = Node if _ROS2_AVAILABLE else object


class _CovarianceSubscriber(_NodeBase):
    """
    @class _CovarianceSubscriber
    @brief Lightweight rclpy Node that subscribes to EKF covariance.

    Caches the latest 9-element uncertainty feature vector in a thread-safe
    manner. Runs via rclpy.spin() in a daemon thread so it does not block
    Gymnasium step().

    The CovarianceExtractorNode publishes a CovarianceEstimate message with
    semantic fields (x, y, yaw, covariance[9]). We reshape the covariance
    field into a 3x3 matrix and call extract_2d_covariance_features().
    """

    def __init__(
        self,
        covariance_topic: str = "/ekf_uncertainty/covariance",
        node_name: str = "covariance_subscriber",
    ) -> None:
        """
        @brief Initialise the covariance subscriber node.
        @param covariance_topic: ROS 2 topic to subscribe to.
        @param node_name: Unique node name (important when multiple envs exist).
        """
        if not _ROS2_AVAILABLE:
            return

        super().__init__(node_name)

        self._lock = threading.Lock()
        self._latest_uncertainty: Optional[np.ndarray] = None
        self._message_count = 0

        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            depth=10,
        )

        self._subscription = self.create_subscription(
            CovarianceEstimate,
            covariance_topic,
            self._covariance_callback,
            qos,
        )
        self.get_logger().info(f"Subscribed to covariance topic: {covariance_topic}")

    def _covariance_callback(self, msg: "CovarianceEstimate") -> None:
        """
        @brief Callback for incoming covariance messages.
        @param msg: CovarianceEstimate with semantic fields (x, y, yaw, covariance).
        """
        cov_matrix = np.array(msg.covariance).reshape(3, 3)

        # Extract the 9 uncertainty features used in the state vector
        features = extract_2d_covariance_features(cov_matrix)

        with self._lock:
            self._latest_uncertainty = features
            self._message_count += 1

    def get_latest_uncertainty(self) -> Optional[np.ndarray]:
        """
        @brief Get the most recent 9-element uncertainty feature vector.
        @return Array of shape (9,) or None if no message received yet.
        """
        with self._lock:
            if self._latest_uncertainty is not None:
                return cast(np.ndarray, self._latest_uncertainty.copy())
            return None

    @property
    def has_data(self) -> bool:
        """
        @brief Check whether at least one covariance message has been received.
        @return True if data is available.
        """
        with self._lock:
            return self._latest_uncertainty is not None


class CARLAParkingEnv(gym.Env):
    """
    @class CARLAParkingEnv
    @brief CARLA-based parking environment with real EKF uncertainty.

    This environment simulates an autonomous parking scenario where the agent must
    park a vehicle whilst accounting for localisation uncertainty. Uncertainty is
    produced naturally by the robot_localisation EKF processing noisy CARLA sensors
    under varying weather and traffic conditions - not by a simulated noise model.

    @note Requires Docker containers running (CARLA, ros2-bridge, training).
          There is no standalone fallback for training.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        carla_host: str = "localhost",
        carla_port: int = 2000,
        town: str = "Town01",
        max_steps: int = 500,
        target_parking_spot: Optional[Tuple[float, float, float]] = None,
        render_mode: Optional[str] = None,
        ros2_config: Optional[Dict[str, Any]] = None,
        carla_sensors_config: Optional[Dict[str, Any]] = None,
        carla_conditions_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        @brief Constructor for CARLAParkingEnv.
        @param carla_host: CARLA server host address.
        @param carla_port: CARLA server port.
        @param town: CARLA town/map to use.
        @param max_steps: Maximum episode length.
        @param target_parking_spot: Target parking spot coordinates (x, y, yaw).
        @param render_mode: Rendering mode ('human', 'rgb_array', or None).
        @param ros2_config: ROS 2 settings (covariance_topic, covariance_timeout).
        @param carla_sensors_config: Sensor noise parameters (imu, gnss subsections).
        @param carla_conditions_config: Weather and traffic settings.
        """
        super().__init__()

        self.carla_host = carla_host
        self.carla_port = carla_port
        self.town = town
        self.max_steps = max_steps
        self.render_mode = render_mode

        # Parse configs with defaults
        ros2_config = ros2_config or {}
        self._covariance_topic = ros2_config.get(
            "covariance_topic", "/ekf_uncertainty/covariance"
        )
        self._covariance_timeout = ros2_config.get("covariance_timeout", 10.0)

        self._sensors_config = carla_sensors_config or {}
        self._conditions_config = carla_conditions_config or {}

        # CARLA client and world (initialised in reset)
        self.client: Optional[Any] = None
        self.world: Optional[Any] = None
        self.vehicle: Optional[Any] = None
        self.spectator: Optional[Any] = None

        # Spawned actors to clean up each episode
        self._spawned_sensors: List[Any] = []
        self._spawned_npcs: List[Any] = []

        # Parking spot (x, y, yaw in radians)
        if target_parking_spot is None:
            self.target_parking_spot = np.array([0.0, 0.0, 0.0])
        else:
            self.target_parking_spot = np.array(target_parking_spot)

        # Episode state
        self.steps = 0
        self.done = False

        # Observation and action spaces
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(TOTAL_OBS_DIM,),
            dtype=np.float32,
        )

        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.0, 0.0]),
            high=np.array([1.0, 1.0, 1.0]),
            dtype=np.float32,
        )

        # ROS 2 covariance subscriber
        self._cov_subscriber: Optional[_CovarianceSubscriber] = None
        self._spin_thread: Optional[threading.Thread] = None
        self._init_ros2()

    def _init_ros2(self) -> None:
        """
        @brief Initialise rclpy and start the covariance subscriber in a daemon thread.

        Guards against double-initialisation when multiple env instances exist.
        When rclpy is unavailable (CI/tests), logs a warning and continues with
        zero uncertainty in state - the env is non-functional for training in this case.
        """
        if not _ROS2_AVAILABLE:
            logger.warning(
                "rclpy not available. Covariance subscriber disabled. "
                "The environment will return zero uncertainty features. "
                "This is only acceptable for CI/unit tests, not for training."
            )
            return

        if not rclpy.ok():
            rclpy.init()

        # Use unique node name to support multiple env instances
        node_name = f"covariance_subscriber_{id(self)}"
        self._cov_subscriber = _CovarianceSubscriber(
            covariance_topic=self._covariance_topic,
            node_name=node_name,
        )

        self._spin_thread = threading.Thread(
            target=rclpy.spin,
            args=(self._cov_subscriber,),
            daemon=True,
        )
        self._spin_thread.start()
        logger.info("ROS 2 covariance subscriber started in daemon thread.")

    def _connect_to_carla(self) -> None:
        """
        @brief Establish connection to CARLA simulator.
        """
        try:
            self.client = carla.Client(self.carla_host, self.carla_port)
            self.client.set_timeout(10.0)
            self.world = self.client.get_world()

            # Load the specified town if not already loaded
            current_map = self.world.get_map()
            if current_map.name.split("/")[-1] != self.town:
                self.world = self.client.load_world(self.town)

        except Exception as e:
            logger.error(f"Could not connect to CARLA: {e}")
            self.client = None
            self.world = None

    def _spawn_vehicle(self) -> None:
        """
        @brief Spawn the ego vehicle in the world.
        """
        if self.world is None:
            return

        blueprint_library = self.world.get_blueprint_library()
        vehicle_bp = blueprint_library.filter("vehicle.tesla.model3")[0]

        spawn_points = self.world.get_map().get_spawn_points()
        if spawn_points:
            spawn_point = random.choice(spawn_points)
        else:
            spawn_point = carla.Transform(
                carla.Location(x=0, y=0, z=0.5),
                carla.Rotation(pitch=0, yaw=0, roll=0),
            )

        self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
        time.sleep(0.5)

    def _spawn_sensors(self) -> None:
        """
        @brief Spawn noisy IMU and GNSS sensors attached to the ego vehicle.

        Sensor noise parameters come from the carla_sensors config section.
        These noisy readings feed into the CARLA ROS bridge, which publishes
        them as ROS 2 topics consumed by the robot_localisation EKF.
        """
        if self.vehicle is None or self.world is None:
            return

        blueprint_library = self.world.get_blueprint_library()
        imu_config = self._sensors_config.get("imu", {})
        gnss_config = self._sensors_config.get("gnss", {})

        # --- IMU sensor ---
        imu_bp = blueprint_library.find("sensor.other.imu")
        imu_bp.set_attribute(
            "noise_accel_stddev_x",
            str(imu_config.get("noise_accel_stddev_x", 0.1)),
        )
        imu_bp.set_attribute(
            "noise_accel_stddev_y",
            str(imu_config.get("noise_accel_stddev_y", 0.1)),
        )
        imu_bp.set_attribute(
            "noise_accel_stddev_z",
            str(imu_config.get("noise_accel_stddev_z", 0.1)),
        )
        imu_bp.set_attribute(
            "noise_gyro_stddev_x",
            str(imu_config.get("noise_gyro_stddev_x", 0.01)),
        )
        imu_bp.set_attribute(
            "noise_gyro_stddev_y",
            str(imu_config.get("noise_gyro_stddev_y", 0.01)),
        )
        imu_bp.set_attribute(
            "noise_gyro_stddev_z",
            str(imu_config.get("noise_gyro_stddev_z", 0.01)),
        )
        imu_bp.set_attribute(
            "sensor_tick",
            str(imu_config.get("sensor_tick", 0.05)),
        )

        imu_transform = carla.Transform(carla.Location(x=0.0, z=0.0))
        imu_sensor = self.world.spawn_actor(
            imu_bp, imu_transform, attach_to=self.vehicle
        )
        self._spawned_sensors.append(imu_sensor)

        # --- GNSS sensor ---
        gnss_bp = blueprint_library.find("sensor.other.gnss")
        gnss_bp.set_attribute(
            "noise_alt_stddev",
            str(gnss_config.get("noise_alt_stddev", 0.5)),
        )
        gnss_bp.set_attribute(
            "noise_lat_stddev",
            str(gnss_config.get("noise_lat_stddev", 0.00001)),
        )
        gnss_bp.set_attribute(
            "noise_lon_stddev",
            str(gnss_config.get("noise_lon_stddev", 0.00001)),
        )
        gnss_bp.set_attribute(
            "noise_alt_bias",
            str(gnss_config.get("noise_alt_bias", 0.0)),
        )
        gnss_bp.set_attribute(
            "noise_lat_bias",
            str(gnss_config.get("noise_lat_bias", 0.0)),
        )
        gnss_bp.set_attribute(
            "noise_lon_bias",
            str(gnss_config.get("noise_lon_bias", 0.0)),
        )
        gnss_bp.set_attribute(
            "sensor_tick",
            str(gnss_config.get("sensor_tick", 0.1)),
        )

        gnss_transform = carla.Transform(carla.Location(x=0.0, z=0.0))
        gnss_sensor = self.world.spawn_actor(
            gnss_bp, gnss_transform, attach_to=self.vehicle
        )
        self._spawned_sensors.append(gnss_sensor)

        logger.debug(
            f"Spawned IMU and GNSS sensors with noise config: "
            f"imu={imu_config}, gnss={gnss_config}"
        )

    def _configure_weather(self) -> None:
        """
        @brief Set random weather conditions for the current episode.

        Randomly selects a weather preset and adds randomised fog within
        the configured range. This degrades sensor quality and increases
        EKF uncertainty naturally.
        """
        if self.world is None:
            return

        presets = self._conditions_config.get("weather_presets", ["ClearNoon"])
        preset_name = random.choice(presets)

        # Get the weather preset from CARLA
        weather = getattr(carla.WeatherParameters, preset_name, None)
        if weather is None:
            logger.warning(f"Unknown weather preset '{preset_name}', using ClearNoon.")
            weather = carla.WeatherParameters.ClearNoon

        # Apply randomised fog on top of the preset
        fog_range = self._conditions_config.get("fog_density_range", [0.0, 0.0])
        fog_density = random.uniform(fog_range[0], fog_range[1])

        fog_dist_range = self._conditions_config.get(
            "fog_distance_range", [20.0, 100.0]
        )
        fog_distance = random.uniform(fog_dist_range[0], fog_dist_range[1])

        weather.fog_density = fog_density
        weather.fog_distance = fog_distance

        self.world.set_weather(weather)
        logger.info(
            f"Weather: {preset_name}, fog_density={fog_density:.1f}, "
            f"fog_distance={fog_distance:.1f}m"
        )

    def _spawn_traffic(self) -> None:
        """
        @brief Spawn NPC vehicles and pedestrians via the CARLA traffic manager.

        Moving objects in the scene cause dynamic occlusions and
        data association challenges, increasing EKF uncertainty.
        """
        if self.world is None or self.client is None:
            return

        num_vehicles = self._conditions_config.get("num_vehicles", 0)
        num_pedestrians = self._conditions_config.get("num_pedestrians", 0)

        blueprint_library = self.world.get_blueprint_library()
        spawn_points = self.world.get_map().get_spawn_points()

        # --- NPC vehicles ---
        vehicle_bps = blueprint_library.filter("vehicle.*")
        available_spawns = list(spawn_points)
        random.shuffle(available_spawns)

        traffic_manager = self.client.get_trafficmanager()
        traffic_manager.set_global_distance_to_leading_vehicle(2.5)

        for i in range(min(num_vehicles, len(available_spawns))):
            bp = random.choice(vehicle_bps)
            if bp.has_attribute("color"):
                color = random.choice(bp.get_attribute("color").recommended_values)
                bp.set_attribute("color", color)

            npc = self.world.try_spawn_actor(bp, available_spawns[i])
            if npc is not None:
                npc.set_autopilot(True, traffic_manager.get_port())
                self._spawned_npcs.append(npc)

        # --- Pedestrians ---
        walker_bps = blueprint_library.filter("walker.pedestrian.*")
        walker_controller_bp = blueprint_library.find("controller.ai.walker")

        for _ in range(num_pedestrians):
            bp = random.choice(walker_bps)
            if bp.has_attribute("is_invincible"):
                bp.set_attribute("is_invincible", "false")

            spawn_loc = self.world.get_random_location_from_navigation()
            if spawn_loc is None:
                continue

            spawn_transform = carla.Transform(location=spawn_loc)
            walker = self.world.try_spawn_actor(bp, spawn_transform)
            if walker is None:
                continue

            controller = self.world.spawn_actor(
                walker_controller_bp,
                carla.Transform(),
                attach_to=walker,
            )
            controller.start()
            controller.go_to_location(self.world.get_random_location_from_navigation())
            controller.set_max_speed(1.0 + random.random())

            self._spawned_npcs.append(walker)
            self._spawned_npcs.append(controller)

        logger.info(
            f"Spawned traffic: {num_vehicles} vehicles requested, "
            f"{num_pedestrians} pedestrians requested."
        )

    def _wait_for_covariance(self) -> None:
        """
        @brief Block until the first EKF covariance message arrives.

        Ticks the CARLA simulation while waiting so sensors produce data
        for the EKF to process. Raises RuntimeError on timeout.
        """
        if self._cov_subscriber is None:
            return

        start = time.monotonic()
        tick_interval = 0.05  # Match carla_timestep

        while not self._cov_subscriber.has_data:
            elapsed = time.monotonic() - start
            if elapsed > self._covariance_timeout:
                raise RuntimeError(
                    f"No covariance message received within "
                    f"{self._covariance_timeout}s timeout. "
                    f"Check that the ros2-bridge container is running and "
                    f"the EKF + CovarianceExtractorNode are publishing."
                )

            # Tick CARLA so sensors produce data
            if self.world is not None:
                self.world.tick()
            time.sleep(tick_interval)

        logger.debug(
            f"First covariance message received after "
            f"{time.monotonic() - start:.2f}s."
        )

    def _cleanup_actors(self) -> None:
        """
        @brief Destroy all spawned actors (sensors, NPCs, ego vehicle).

        Called during reset() and close() to ensure clean state.
        """
        # Destroy sensors
        for sensor in self._spawned_sensors:
            if sensor is not None and sensor.is_alive:
                sensor.stop()
                sensor.destroy()
        self._spawned_sensors.clear()

        # Destroy NPCs (controllers first, then walkers/vehicles)
        for npc in reversed(self._spawned_npcs):
            if npc is not None and npc.is_alive:
                if "controller" in npc.type_id:
                    npc.stop()
                npc.destroy()
        self._spawned_npcs.clear()

        # Destroy ego vehicle
        if self.vehicle is not None:
            if self.vehicle.is_alive:
                self.vehicle.destroy()
            self.vehicle = None

    def _get_state(self) -> np.ndarray:
        """
        @brief Get current state with real EKF uncertainty.
        @return State vector of shape (TOTAL_OBS_DIM,) = (15,).

        Vehicle state (indices 0-5) comes from CARLA Python API (ground truth).
        Uncertainty features (indices 6-14) come from the EKF covariance
        subscriber. If no covariance message is available yet (should only
        happen briefly at episode start before _wait_for_covariance() completes),
        zeros are used.
        """
        if self.vehicle is None or self.world is None:
            return np.zeros(TOTAL_OBS_DIM, dtype=np.float32)

        # Get vehicle transform and velocity from CARLA (ground truth)
        transform = self.vehicle.get_transform()
        velocity = self.vehicle.get_velocity()
        angular_vel = self.vehicle.get_angular_velocity()

        # Position and orientation
        x = transform.location.x
        y = transform.location.y
        yaw = np.deg2rad(transform.rotation.yaw)

        # Velocity
        vx = velocity.x
        vy = velocity.y
        vyaw = np.deg2rad(angular_vel.z)

        # Vehicle state vector (6 elements)
        vehicle_state = np.array([x, y, yaw, vx, vy, vyaw], dtype=np.float32)

        # Uncertainty features from EKF covariance (9 elements)
        if self._cov_subscriber is not None:
            uncertainty = self._cov_subscriber.get_latest_uncertainty()
        else:
            uncertainty = None

        if uncertainty is None:
            uncertainty = np.zeros(COVARIANCE_FEATURES_DIM, dtype=np.float32)
        else:
            uncertainty = uncertainty.astype(np.float32)

        # Concatenate: [vehicle_state(6), uncertainty_features(9)] = 15
        state: np.ndarray = np.concatenate([vehicle_state, uncertainty])
        return state

    def _compute_reward(self, state: np.ndarray) -> Tuple[float, bool]:
        """
        @brief Compute reward based on parking objective.
        @param state: Current state vector.
        @return Tuple of (reward, done) where done indicates episode termination.
        """
        x, y, yaw = state[0], state[1], state[2]
        vx, vy = state[3], state[4]

        # Distance to target parking spot
        target_x, target_y, target_yaw = self.target_parking_spot
        position_error = np.sqrt((x - target_x) ** 2 + (y - target_y) ** 2)
        orientation_error = np.abs(
            np.arctan2(np.sin(yaw - target_yaw), np.cos(yaw - target_yaw))
        )

        # Velocity magnitude
        velocity_magnitude = np.sqrt(vx**2 + vy**2)

        # Reward components
        distance_reward = -position_error
        orientation_reward = -orientation_error * 0.5
        velocity_penalty = -velocity_magnitude * 0.1

        # Success bonus (using shared constants)
        success = (
            position_error < SUCCESS_THRESHOLD_POSITION
            and orientation_error < SUCCESS_THRESHOLD_ORIENTATION
            and velocity_magnitude < 0.1
        )

        success_bonus = 100.0 if success else 0.0

        reward = distance_reward + orientation_reward + velocity_penalty + success_bonus

        # Episode termination conditions
        done = success or position_error > 20.0 or self.steps >= self.max_steps

        return float(reward), bool(done)

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        @brief Reset the environment for a new episode.
        @param seed: Random seed for reproducibility.
        @param options: Additional options for reset.
        @return Tuple of (initial_state, info_dict).

        Each reset randomises weather and traffic to expose the agent to
        varied uncertainty conditions during training.
        """
        super().reset(seed=seed)

        # Reset episode state
        self.steps = 0
        self.done = False

        # Clean up previous episode actors
        self._cleanup_actors()

        # Connect to CARLA if not already connected
        if self.client is None:
            self._connect_to_carla()

        # Set up new episode
        self._configure_weather()
        self._spawn_vehicle()
        self._spawn_sensors()
        self._spawn_traffic()

        # Wait for EKF to start producing covariance
        self._wait_for_covariance()

        # Get initial state
        state = self._get_state()

        info: Dict[str, Any] = {
            "episode": {"r": 0, "l": 0},
        }

        return state, info

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        @brief Execute one environment step.
        @param action: Action vector [steering, throttle, brake].
        @return Tuple of (next_state, reward, terminated, truncated, info).
        """
        self.steps += 1

        # Apply action to vehicle
        if self.vehicle is not None:
            control = carla.VehicleControl()
            control.steer = float(np.clip(action[0], -1.0, 1.0))
            control.throttle = float(np.clip(action[1], 0.0, 1.0))
            control.brake = float(np.clip(action[2], 0.0, 1.0))
            self.vehicle.apply_control(control)

            # Step the simulation
            if self.world is not None:
                self.world.tick()

        # Get new state (includes real EKF uncertainty)
        state = self._get_state()

        # Compute reward
        reward, terminated = self._compute_reward(state)

        # Truncation (time limit)
        truncated = self.steps >= self.max_steps

        info: Dict[str, Any] = {
            "steps": self.steps,
        }

        return state, reward, terminated, truncated, info

    def render(self):
        """
        @brief Render the environment.
        @return RGB array if render_mode is 'rgb_array', None otherwise.
        """
        if self.render_mode == "human" and self.world is not None:
            if self.vehicle is not None:
                transform = self.vehicle.get_transform()
                spectator = self.world.get_spectator()
                spectator.set_transform(
                    carla.Transform(
                        transform.location + carla.Location(z=50),
                        carla.Rotation(pitch=-90),
                    )
                )
        elif self.render_mode == "rgb_array":
            return np.zeros((600, 800, 3), dtype=np.uint8)

        return None

    def close(self) -> None:
        """
        @brief Clean up all resources including CARLA actors and ROS 2 nodes.
        """
        self._cleanup_actors()

        # Shut down ROS 2 subscriber
        if self._cov_subscriber is not None:
            self._cov_subscriber.destroy_node()
            self._cov_subscriber = None

        if self.client is not None:
            self.client = None

        self.world = None

        super().close()
