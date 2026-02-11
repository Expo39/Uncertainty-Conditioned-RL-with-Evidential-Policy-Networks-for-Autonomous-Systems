"""
@file carla_parking.py
@brief CARLA Gymnasium parking environment with SLAM uncertainty integration.

This module implements a Gymnasium-compatible environment for autonomous parking
in CARLA simulator with SLAM localisation uncertainty in the state representation.
"""

import random
import time
from typing import Any, Dict, Optional, Tuple

import carla
import gymnasium as gym
import numpy as np
from gymnasium import spaces


class CARLAParkingEnv(gym.Env):
    """
    @class CARLAParkingEnv
    @brief CARLA-based parking environment with SLAM uncertainty.

    This environment simulates an autonomous parking scenario where the agent must
    park a vehicle whilst accounting for localisation uncertainty from SLAM.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        carla_host: str = "localhost",
        carla_port: int = 2000,
        town: str = "Town01",
        uncertainty_noise_std: float = 0.1,
        max_steps: int = 500,
        target_parking_spot: Optional[Tuple[float, float, float]] = None,
        render_mode: Optional[str] = None,
    ) -> None:
        """
        @brief Constructor for CARLAParkingEnv.
        @param carla_host: CARLA server host address.
        @param carla_port: CARLA server port.
        @param town: CARLA town/map to use.
        @param uncertainty_noise_std: Standard deviation of position uncertainty (metres).
        @param max_steps: Maximum episode length.
        @param target_parking_spot: Target parking spot coordinates (x, y, yaw).
        @param render_mode: Rendering mode ('human', 'rgb_array', or None).
        """
        super().__init__()

        self.carla_host = carla_host
        self.carla_port = carla_port
        self.town = town
        self.uncertainty_noise_std = uncertainty_noise_std
        self.max_steps = max_steps
        self.render_mode = render_mode

        # CARLA client and world (initialised in reset)
        self.client: Optional[carla.Client] = None
        self.world: Optional[carla.World] = None
        self.vehicle: Optional[carla.Vehicle] = None
        self.spectator: Optional[carla.Actor] = None

        # Parking spot (x, y, yaw in radians)
        if target_parking_spot is None:
            self.target_parking_spot = np.array([0.0, 0.0, 0.0])
        else:
            self.target_parking_spot = np.array(target_parking_spot)

        # Episode state
        self.steps = 0
        self.done = False

        # State: [x, y, yaw, vx, vy, vyaw,
        #         uncertainty_x, uncertainty_y, uncertainty_yaw,
        #         cov_xx, cov_yy, cov_yawyaw, cov_xy, cov_xyaw, cov_yyaw]
        # 6 base state + 3 uncertainty (std) + 6 covariance elements = 15 dimensional
        state_dim = 15
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float32
        )

        # Action: [steering, throttle, brake]
        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.0, 0.0]),
            high=np.array([1.0, 1.0, 1.0]),
            dtype=np.float32,
        )

        # Covariance matrix for SLAM uncertainty simulation
        self.covariance_matrix = np.eye(3) * (uncertainty_noise_std**2)

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
            print(f"Warning: Could not connect to CARLA: {e}")
            print("Running in simulation mode without CARLA connection.")
            self.client = None
            self.world = None

    def _spawn_vehicle(self) -> None:
        """
        @brief Spawn the ego vehicle in the world.
        """
        if self.world is None:
            return

        # Get vehicle blueprint
        blueprint_library = self.world.get_blueprint_library()
        vehicle_bp = blueprint_library.filter("vehicle.tesla.model3")[0]

        # Get spawn points
        spawn_points = self.world.get_map().get_spawn_points()
        if spawn_points:
            spawn_point = random.choice(spawn_points)
        else:
            # Default spawn point
            spawn_point = carla.Transform(
                carla.Location(x=0, y=0, z=0.5), carla.Rotation(pitch=0, yaw=0, roll=0)
            )

        # Spawn vehicle
        self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)

        # Wait for vehicle to spawn
        time.sleep(0.5)

    def _get_state(self) -> np.ndarray:
        """
        @brief Get current state with SLAM uncertainty.
        @return State vector including position, velocity, and uncertainty estimates.
        """
        if self.vehicle is None or self.world is None:
            # Simulation mode - return dummy state
            return np.zeros(15, dtype=np.float32)

        # Get vehicle transform and velocity
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

        # Simulate SLAM uncertainty (position uncertainty grows with velocity)
        velocity_magnitude = np.sqrt(vx**2 + vy**2)
        dynamic_noise_std = self.uncertainty_noise_std * (
            1.0 + 0.1 * velocity_magnitude
        )

        # Update covariance matrix (simple model)
        self.covariance_matrix[0, 0] = dynamic_noise_std**2  # x variance
        self.covariance_matrix[1, 1] = dynamic_noise_std**2  # y variance
        self.covariance_matrix[2, 2] = (dynamic_noise_std * 0.1) ** 2  # yaw variance

        # Extract uncertainty (standard deviations)
        uncertainty_x = np.sqrt(self.covariance_matrix[0, 0])
        uncertainty_y = np.sqrt(self.covariance_matrix[1, 1])
        uncertainty_yaw = np.sqrt(self.covariance_matrix[2, 2])

        # Covariance elements (upper triangle)
        cov_xx = self.covariance_matrix[0, 0]
        cov_yy = self.covariance_matrix[1, 1]
        cov_yawyaw = self.covariance_matrix[2, 2]
        cov_xy = self.covariance_matrix[0, 1]
        cov_xyaw = self.covariance_matrix[0, 2]
        cov_yyaw = self.covariance_matrix[1, 2]

        # Construct state vector
        state = np.array(
            [
                x,
                y,
                yaw,
                vx,
                vy,
                vyaw,
                uncertainty_x,
                uncertainty_y,
                uncertainty_yaw,
                cov_xx,
                cov_yy,
                cov_yawyaw,
                cov_xy,
                cov_xyaw,
                cov_yyaw,
            ],
            dtype=np.float32,
        )

        return state

    def _compute_reward(self, state: np.ndarray) -> Tuple[float, bool]:
        """
        @brief Compute reward based on parking objective.
        @param state: Current state vector.
        @return Tuple of (reward, done) where done indicates episode termination.
        """
        # Extract position and orientation
        x, y, yaw = state[0], state[1], state[2]
        vx, vy, vyaw = state[3], state[4], state[5]

        # Distance to target parking spot
        target_x, target_y, target_yaw = self.target_parking_spot
        position_error = np.sqrt((x - target_x) ** 2 + (y - target_y) ** 2)
        orientation_error = np.abs(
            np.arctan2(np.sin(yaw - target_yaw), np.cos(yaw - target_yaw))
        )

        # Velocity magnitude
        velocity_magnitude = np.sqrt(vx**2 + vy**2)

        # Reward components
        # 1. Distance reward (negative of distance)
        distance_reward = -position_error

        # 2. Orientation reward
        orientation_reward = -orientation_error * 0.5

        # 3. Velocity penalty (should slow down near target)
        velocity_penalty = -velocity_magnitude * 0.1

        # 4. Success bonus
        success_threshold_pos = 0.5  # metres
        success_threshold_ori = np.deg2rad(10)  # 10 degrees
        success = (
            position_error < success_threshold_pos
            and orientation_error < success_threshold_ori
            and velocity_magnitude < 0.1
        )

        success_bonus = 100.0 if success else 0.0

        # Total reward
        reward = distance_reward + orientation_reward + velocity_penalty + success_bonus

        # Episode termination conditions
        done = success or position_error > 20.0 or self.steps >= self.max_steps

        return float(reward), done

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        @brief Reset the environment.
        @param seed: Random seed for reproducibility.
        @param options: Additional options for reset.
        @return Tuple of (initial_state, info_dict).
        """
        super().reset(seed=seed)

        # Reset episode state
        self.steps = 0
        self.done = False

        # Connect to CARLA if not already connected
        if self.client is None:
            self._connect_to_carla()

        # Clean up existing vehicle
        if self.vehicle is not None:
            self.vehicle.destroy()
            self.vehicle = None

        # Spawn new vehicle
        self._spawn_vehicle()

        # Get initial state
        state = self._get_state()

        info = {
            "episode": {"r": 0, "l": 0},
            "uncertainty_noise_std": self.uncertainty_noise_std,
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

        # Get new state
        state = self._get_state()

        # Compute reward
        reward, terminated = self._compute_reward(state)

        # Truncation (time limit)
        truncated = self.steps >= self.max_steps

        info = {
            "steps": self.steps,
            "uncertainty_noise_std": self.uncertainty_noise_std,
        }

        return state, reward, terminated, truncated, info

    def render(self) -> Optional[np.ndarray]:
        """
        @brief Render the environment.
        @return RGB array if render_mode is 'rgb_array', None otherwise.
        """
        if self.render_mode == "human" and self.world is not None:
            # Update spectator view to follow vehicle
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
            # Would need to attach a camera sensor and return image
            # Placeholder for now
            return np.zeros((600, 800, 3), dtype=np.uint8)

        return None

    def close(self) -> None:
        """
        @brief Clean up resources.
        """
        if self.vehicle is not None:
            self.vehicle.destroy()
            self.vehicle = None

        if self.client is not None:
            self.client = None

        self.world = None
