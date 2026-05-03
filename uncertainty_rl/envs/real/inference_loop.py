"""
@file inference_loop.py
@brief Real-world inference loop stub for the autonomous parking system.

Runs the trained evidential policy on a physical vehicle. SafetyWrapper is
always active: aleatoric uncertainty caps longitudinal speed; epistemic
uncertainty above the handoff threshold commands a full stop.
"""

import logging
import math
import threading
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

if TYPE_CHECKING:
    from uncertainty_rl.envs.real.deployment_utils import RealWorldDeployment

import numpy as np

from uncertainty_rl.envs._parking_core import (
    build_observation,
    calibrate_ekf_frame_offset,
    compute_obs_dim,
    extract_obstacle_features,
    wait_for_ekf,
)
from uncertainty_rl.envs.covariance_subscriber import _CovarianceSubscriber
from uncertainty_rl.envs.safety_wrapper import SafetyWrapper
from uncertainty_rl.utils.constants import (
    OBSTACLE_FEATURES_DIM,
    OUT_OF_BOUNDS_THRESHOLD,
    SUCCESS_THRESHOLD_ORIENTATION,
    SUCCESS_THRESHOLD_POSITION,
    SUCCESS_THRESHOLD_VELOCITY,
)

logger = logging.getLogger("uncertainty_rl.envs.real.inference_loop")


class RealWorldInferenceLoop:
    """
    @class RealWorldInferenceLoop
    @brief Runs the trained policy on a physical vehicle with SafetyWrapper active.
    """

    def __init__(
        self,
        policy: Any,
        deployment: "RealWorldDeployment",
        target_bay: Dict[str, Any],
        safety_wrapper_cfg: Dict[str, float],
        ros2_cfg: Dict[str, Any],
        include_covariance: bool = True,
        include_obstacle_obs: bool = True,
        max_steps: int = 500,
    ) -> None:
        """
        @brief Construct the inference loop.
        @param policy: Loaded EvidentialActorCriticPolicy (from EvidentialPPO).
        @param deployment: RealWorldDeployment instance (surveyed datum + actuation calibration).
        @param target_bay: Dict with keys x, y, yaw in lot layout frame.
        @param safety_wrapper_cfg: Dict with keys aleatoric_scaling, handoff_threshold.
        @param ros2_cfg: ROS 2 config dict (covariance_timeout, ekf_convergence_timeout,
                         ekf_state_file).
        @param include_covariance: Must match the trained policy's obs flags.
        @param include_obstacle_obs: Must match the trained policy's obs flags.
        @param max_steps: Maximum control steps before mission timeout.
        """
        self._policy = policy
        self._deployment = deployment
        self._target_bay = target_bay
        self._max_steps = max_steps
        self._include_covariance = include_covariance
        self._include_obstacle_obs = include_obstacle_obs
        self._aleatoric_scaling: float = safety_wrapper_cfg.get(
            "aleatoric_scaling", 0.5
        )
        self._handoff_threshold: float = safety_wrapper_cfg.get(
            "handoff_threshold", 5.0
        )
        self._ros2_cfg = ros2_cfg
        self._ekf_convergence_timeout: float = float(
            ros2_cfg.get("ekf_convergence_timeout", 15.0)
        )
        self._covariance_timeout: float = float(
            ros2_cfg.get("covariance_timeout", 120.0)
        )

        # Pre-allocate obs buffers (same pattern as CARLAParkingEnv).
        obs_dim = compute_obs_dim(include_covariance, include_obstacle_obs)
        self._obs_buffer = np.zeros(obs_dim, dtype=np.float32)
        self._obstacle_features_buffer = np.zeros(OBSTACLE_FEATURES_DIM, dtype=np.float32)

        # EKF odom-to-world offset; computed once in prepare().
        self._ekf_odom_offset: Tuple[float, float, float, float, float] = (
            0.0, 0.0, 1.0, 0.0, 0.0
        )

        # File-based EKF covariance subscriber (same as sim).
        self._cov_subscriber = _CovarianceSubscriber(ros2_config=ros2_cfg)

        # ROS 2 publisher for vehicle commands; initialised lazily in run().
        self._twist_publisher: Optional[Any] = None
        self._ros2_node: Optional[Any] = None
        self._actuation_topic: str = str(
            ros2_cfg.get("actuation_topic", "/cmd_vel")
        )

        # Thread-safe operator stop flag -- set by external signal or geofence.
        self._operator_stop: threading.Event = threading.Event()

    @classmethod
    def from_config(
        cls,
        mission_path: str,
        agent_config_path: str = "configs/deployment/agent_config.yaml",
    ) -> "RealWorldInferenceLoop":
        """
        @brief Load model and deployment configs from YAML and construct the loop.
        @param mission_path: Path to configs/deployment/real/mission.yaml.
        @param agent_config_path: Path to configs/deployment/agent_config.yaml.
        @return Constructed RealWorldInferenceLoop.
        """
        try:
            import yaml as _yaml

            from uncertainty_rl.envs.real.deployment_utils import (
                RealWorldDeployment,
            )
            from uncertainty_rl.networks.sb3_integration import EvidentialPPO
        except ImportError as exc:
            raise ImportError(
                "uncertainty_rl package not found. "
                "Ensure the training container environment is active."
            ) from exc

        with open(agent_config_path) as f:
            agent_cfg: Dict[str, Any] = _yaml.safe_load(f) or {}

        model_path: str = agent_cfg.get("model_path", "checkpoints/final_model")
        logger.info("Loading model from %s", model_path)
        model = EvidentialPPO.load(model_path)

        deployment, target_bay = RealWorldDeployment.from_mission(
            mission_path=mission_path,
            datum_path=agent_cfg.get("real_world_datum"),
            calibration_path=agent_cfg.get("actuation_calibration"),
        )

        return cls(
            policy=model.policy,
            deployment=deployment,
            target_bay=target_bay,
            safety_wrapper_cfg={
                "aleatoric_scaling": agent_cfg.get("safety_aleatoric_scaling", 0.5),
                "handoff_threshold": agent_cfg.get("safety_handoff_threshold", 5.0),
            },
            ros2_cfg=agent_cfg.get("ros2", {}),
            include_covariance=bool(agent_cfg.get("include_covariance", True)),
            include_obstacle_obs=bool(agent_cfg.get("include_obstacle_obs", True)),
            max_steps=int(agent_cfg.get("max_steps", 500)),
        )

    # ------------------------------------------------------------------
    # Mission startup
    # ------------------------------------------------------------------

    def prepare(self) -> None:
        """
        @brief Block until sensors are ready, then calibrate the EKF frame offset.

        Call this once before run(). Waits for the CovarianceExtractorNode to
        publish its first valid EKF state, then runs calibrate_ekf_frame_offset()
        using the surveyed datum as the world reference. The resulting odom-to-
        world transform is stored in self._ekf_odom_offset for use in every
        subsequent _get_observation() call.

        @raises RuntimeError if sensors do not become ready within the configured
                covariance_timeout.
        @warning datum_loaded() is checked but not enforced -- if the datum was
                 not loaded, the identity transform is used and a warning is logged.
        """
        if not self._deployment.datum_loaded():
            logger.warning(
                "Real-world datum not loaded. "
                "EKF frame calibration will use identity transform. "
                "Check configs/deployment/real/real_world_datum.yaml."
            )

        wait_for_ekf(
            has_lidar=lambda: self._get_lidar_scan() is not None,
            has_ekf=lambda: self._cov_subscriber.has_data,
            timeout=self._covariance_timeout,
        )

        world_x, world_y, world_yaw = self._deployment.reference_pose()
        self._ekf_odom_offset = calibrate_ekf_frame_offset(
            world_x=world_x,
            world_y=world_y,
            world_yaw=world_yaw,
            get_pose=self._cov_subscriber.get_latest_pose,
            timeout=self._ekf_convergence_timeout,
        )

    # ------------------------------------------------------------------
    # Sensor reads (partially implemented)
    # ------------------------------------------------------------------

    def _get_lidar_scan(self) -> Optional[np.ndarray]:
        """
        @brief Return the latest 2D LiDAR point cloud in vehicle body frame.
        @return (N, 2) float32 array of (x_fwd, y_left) returns, or None.

        @todo Implement by reading from the physical LiDAR interface, e.g.:
            - Subscribe to a ROS 2 sensor_msgs/LaserScan or PointCloud2 topic.
            - Convert to (N, 2) float32 with x-forward, y-left convention.
            - The scan may be polled here or cached via a background thread.
        """
        return None

    def _get_observation(self) -> np.ndarray:
        """
        @brief Assemble the observation vector from live EKF state and LiDAR.

        Applies the stored odom-to-world offset to convert the EKF odom-frame
        pose to lot layout frame, then delegates to build_observation() in
        _parking_core - identical to CARLAParkingEnv._get_state().

        @return Observation vector of shape (obs_dim,) matching the trained policy.
        """
        raw_ekf_pose, uncertainty = self._cov_subscriber.get_latest_state()

        world_pose: Optional[np.ndarray] = None
        if raw_ekf_pose is not None:
            # ROS REP-103 y is negated relative to lot layout y (same as sim).
            ekf_odom_x = float(raw_ekf_pose[0])
            ekf_odom_y = -float(raw_ekf_pose[1])
            ekf_odom_yaw = float(raw_ekf_pose[2])
            tx, ty, cos_r, sin_r, r = self._ekf_odom_offset
            wx = cos_r * ekf_odom_x - sin_r * ekf_odom_y + tx
            wy = sin_r * ekf_odom_x + cos_r * ekf_odom_y + ty
            wyaw = ekf_odom_yaw + r
            world_pose = np.array(
                [wx, wy, wyaw, float(raw_ekf_pose[3])],
                dtype=np.float32,
            )
        else:
            logger.debug("EKF pose unavailable - obs will use zero pose.")

        obstacle_features = extract_obstacle_features(
            self._get_lidar_scan() if self._include_obstacle_obs else None,
            self._obstacle_features_buffer,
        )

        return build_observation(
            world_pose,
            uncertainty,
            self._target_bay,
            obstacle_features,
            self._include_covariance,
            self._include_obstacle_obs,
            self._obs_buffer,
        )

    # ------------------------------------------------------------------
    # Actuation and termination (stubs)
    # ------------------------------------------------------------------

    def _apply_action(self, steering: float, longitudinal: float) -> None:
        """
        @brief Publish a calibrated action to the vehicle via geometry_msgs/Twist.
        @param steering: Calibrated steering command in [-1, 1]; mapped to angular.z.
        @param longitudinal: Calibrated longitudinal command in [-1, 1]; mapped to linear.x.

        Publishes on the topic configured by ros2.actuation_topic (default: /cmd_vel).
        The node and publisher are initialised lazily on the first call inside run().
        """
        if self._twist_publisher is None:
            raise RuntimeError(
                "_apply_action() called before ROS 2 publisher was initialised. "
                "Call run() rather than invoking _apply_action() directly."
            )

        try:
            from geometry_msgs.msg import Twist  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "geometry_msgs not available. "
                "Ensure the ROS 2 environment is sourced inside the ros2-bridge container."
            ) from exc

        msg = Twist()
        msg.linear.x = float(np.clip(longitudinal, -1.0, 1.0))
        msg.angular.z = float(np.clip(steering, -1.0, 1.0))
        self._twist_publisher.publish(msg)

    def _is_done(self) -> Tuple[bool, bool]:
        """
        @brief Check episode termination conditions from live EKF state.
        @return Tuple (terminated, truncated).

        """
        # Operator override always truncates immediately.
        if self._operator_stop.is_set():
            logger.warning("Operator stop requested- truncating mission.")
            return False, True

        raw_ekf_pose, _ = self._cov_subscriber.get_latest_state()
        if raw_ekf_pose is None:
            logger.debug("_is_done(): EKF pose unavailable- continuing.")
            return False, False

        # Apply the same odom-to-world transform used in _get_observation().
        ekf_odom_x = float(raw_ekf_pose[0])
        ekf_odom_y = -float(raw_ekf_pose[1])
        ekf_odom_yaw = float(raw_ekf_pose[2])
        tx, ty, cos_r, sin_r, r = self._ekf_odom_offset
        world_x = cos_r * ekf_odom_x - sin_r * ekf_odom_y + tx
        world_y = sin_r * ekf_odom_x + cos_r * ekf_odom_y + ty
        world_yaw = ekf_odom_yaw + r

        # vyaw (index 3) is used as a proxy for speed magnitude.
        yaw_rate = float(raw_ekf_pose[3])

        target_x = float(self._target_bay["x"])
        target_y = float(self._target_bay["y"])
        target_yaw = float(self._target_bay["yaw"])

        pos_error = math.hypot(world_x - target_x, world_y - target_y)
        yaw_error = abs(
            math.atan2(
                math.sin(world_yaw - target_yaw),
                math.cos(world_yaw - target_yaw),
            )
        )

        # Out-of-bounds: mission cannot recover if the vehicle wanders too far.
        if pos_error > OUT_OF_BOUNDS_THRESHOLD:
            logger.warning(
                "Out-of-bounds: pos_error=%.2f m > %.1f m - terminating.",
                pos_error,
                OUT_OF_BOUNDS_THRESHOLD,
            )
            return True, False

        # Success: all three criteria met simultaneously.
        success = (
            pos_error < SUCCESS_THRESHOLD_POSITION
            and yaw_error < SUCCESS_THRESHOLD_ORIENTATION
            and abs(yaw_rate) < SUCCESS_THRESHOLD_VELOCITY
        )
        if success:
            logger.info(
                "Parking success: pos_error=%.3f m  yaw_error=%.2f deg  yaw_rate=%.3f rad/s",
                pos_error,
                math.degrees(yaw_error),
                yaw_rate,
            )
            return True, False

        return False, False

    def request_stop(self) -> None:
        """
        @brief Signal the mission loop to truncate at the next step.

        Thread-safe. Call from a signal handler, operator UI, or geofence
        monitor to request a graceful mission abort. The loop will publish
        a zero-velocity Twist before returning.
        """
        logger.info("request_stop() called -- setting operator stop flag.")
        self._operator_stop.set()

    def _init_ros2(self) -> None:
        """
        @brief Initialise rclpy node and Twist publisher for actuation.

        @raises ImportError if rclpy or geometry_msgs are not installed.
        @raises RuntimeError if rclpy.init() has already been called externally
                and fails to initialise a second context.
        """
        try:
            import rclpy  # type: ignore[import]
            from geometry_msgs.msg import Twist  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "rclpy or geometry_msgs not available. "
                "Run inside the ros2-bridge container with the ROS 2 environment sourced."
            ) from exc

        if not rclpy.ok():
            rclpy.init()

        self._ros2_node = rclpy.create_node("real_world_inference_loop")
        self._twist_publisher = self._ros2_node.create_publisher(
            Twist, self._actuation_topic, 10
        )
        logger.info(
            "ROS 2 Twist publisher ready on topic '%s'.", self._actuation_topic
        )

    def _shutdown_ros2(self) -> None:
        """
        @brief Publish a zero-velocity stop command and destroy the ROS 2 node.

        Called at the end of run() to ensure the vehicle is commanded to stop
        regardless of how the mission loop exits. The node is then destroyed
        to release resources.
        """
        if self._twist_publisher is not None:
            try:
                from geometry_msgs.msg import Twist  # type: ignore[import]

                stop_msg = Twist()
                self._twist_publisher.publish(stop_msg)
                logger.info("Zero-velocity stop command published.")
            except Exception:
                logger.warning("Failed to publish stop command during shutdown.", exc_info=True)

        if self._ros2_node is not None:
            try:
                self._ros2_node.destroy_node()
            except Exception:
                logger.warning("Failed to destroy ROS 2 node.", exc_info=True)

        self._twist_publisher = None
        self._ros2_node = None

    # ------------------------------------------------------------------
    # Safety
    # ------------------------------------------------------------------

    def _apply_safety_wrapper(
        self,
        action: np.ndarray,
        epistemic: float,
        aleatoric: float,
    ) -> Tuple[np.ndarray, bool]:
        """
        @brief Apply SafetyWrapper interception logic to a raw policy action.
        @param action: Raw policy action [steering, longitudinal].
        @param epistemic: Epistemic uncertainty from evidential actor.
        @param aleatoric: Aleatoric uncertainty from evidential actor.
        @return Tuple (modulated_action, handoff_triggered).
        """
        modulated, handoff = SafetyWrapper.apply(
            action,
            epistemic=epistemic,
            aleatoric=aleatoric,
            aleatoric_scaling=self._aleatoric_scaling,
            handoff_threshold=self._handoff_threshold,
        )
        if handoff:
            logger.warning(
                "Safety handoff triggered (epistemic=%.3f >= threshold=%.3f). "
                "Commanding full stop.",
                epistemic,
                self._handoff_threshold,
            )
        return modulated, handoff

    # ------------------------------------------------------------------
    # Mission loop
    # ------------------------------------------------------------------

    def run(self) -> Dict[str, Any]:
        """
        @brief Prepare sensors and run one parking mission on the physical vehicle.

        Initialises the ROS 2 Twist publisher, blocks until sensors are ready
        via prepare(), then steps the policy in a loop until success, timeout,
        out-of-bounds, or operator stop. Always publishes a zero-velocity stop
        command before returning.

        @return Dict with mission outcome stats: steps, handoff_count,
                terminated (success or OOB), truncated (operator stop or timeout).
        """
        import torch as th

        self._operator_stop.clear()
        self._init_ros2()

        try:
            self.prepare()

            logger.info(
                "Starting real-world parking mission (max_steps=%d).", self._max_steps
            )
            steps = 0
            handoff_count = 0
            terminated = False
            truncated = False

            while steps < self._max_steps:
                obs = self._get_observation()
                obs_tensor = th.as_tensor(obs[np.newaxis], dtype=th.float32)

                action_tensor, uncertainty_dict = (
                    self._policy.get_action_with_uncertainty(
                        obs_tensor, deterministic=True
                    )
                )
                action = action_tensor.cpu().numpy()[0]
                epistemic = float(uncertainty_dict["epistemic"].mean().item())
                aleatoric = float(uncertainty_dict["aleatoric"].mean().item())

                modulated_action, handoff = self._apply_safety_wrapper(
                    action, epistemic, aleatoric
                )
                if handoff:
                    handoff_count += 1

                steering, longitudinal = self._deployment.calibrate_action(
                    float(modulated_action[0]),
                    float(modulated_action[1]),
                )
                self._apply_action(steering, longitudinal)

                steps += 1
                terminated, truncated = self._is_done()

                if terminated or truncated:
                    break

            if steps >= self._max_steps and not (terminated or truncated):
                logger.warning(
                    "Mission timed out after %d steps.", self._max_steps
                )
                truncated = True

        finally:
            self._shutdown_ros2()

        logger.info(
            "Mission complete: steps=%d handoffs=%d terminated=%s truncated=%s",
            steps,
            handoff_count,
            terminated,
            truncated,
        )

        return {
            "steps": steps,
            "handoff_count": handoff_count,
            "terminated": terminated,
            "truncated": truncated,
        }
