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
    SUCCESS_THRESHOLD_VELOCITY,
)

# Placeholder for the sim's polygon-fit check (car_fully_inside_bay).
# @todo(AG) replace once deployment vehicle extents are surveyed and
# added to the deployment config.
_REAL_WORLD_POS_THRESHOLD_M = 0.5
_REAL_WORLD_YAW_THRESHOLD_RAD = math.radians(15.0)

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
        @param safety_wrapper_cfg: Dict with key handoff_threshold (total predictive
                         uncertainty at/above which control is handed off).
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
        self._handoff_threshold: float = float(
            safety_wrapper_cfg.get("handoff_threshold", 1.2)
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
        self._obstacle_features_buffer = np.zeros(
            OBSTACLE_FEATURES_DIM, dtype=np.float32
        )

        # Pre-allocated world pose buffer [x, y, yaw, vyaw, vx_body].
        self._world_pose_buffer = np.zeros(5, dtype=np.float32)
        # True once the buffer contains valid data (set after first successful EKF read).
        self._world_pose_valid: bool = False

        # EKF odom-to-world offset; computed once in prepare().
        self._ekf_odom_offset: Tuple[float, float, float, float, float] = (
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
        )

        # Cached static target bay values
        self._target_x: float = 0.0
        self._target_y: float = 0.0
        self._target_yaw: float = 0.0

        # File-based EKF covariance subscriber (same as sim).
        self._cov_subscriber = _CovarianceSubscriber(ros2_config=ros2_cfg)

        # ROS 2 publisher for vehicle commands; initialised lazily in run().
        self._twist_publisher: Optional[Any] = None
        self._ros2_node: Optional[Any] = None
        self._twist_msg: Optional[Any] = None
        self._twist_stop_msg: Optional[Any] = None
        self._actuation_topic: str = str(ros2_cfg.get("actuation_topic", "/cmd_vel"))

        # Thread-safe operator stop flag - set by external signal or geofence.
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

            from uncertainty_rl.envs.real.deployment_utils import RealWorldDeployment
            from uncertainty_rl.networks.sb3_integration import EvidentialPPO
        except ImportError as exc:
            raise ImportError(
                "uncertainty_rl package not found. "
                "Ensure the training container environment is active."
            ) from exc

        with open(agent_config_path) as f:
            agent_cfg: Dict[str, Any] = _yaml.safe_load(f) or {}

        # Observation flags live ONLY in the baseline files (single source of
        # truth, shared with sim training). Deployment names the baseline the
        # checkpoint was trained as via agent_config `baseline`; its
        # include_covariance / include_obstacle_obs must match the weights.
        baseline_path: str = agent_cfg.get(
            "baseline", "configs/baselines/full_method.yaml"
        )
        with open(baseline_path) as f:
            baseline_cfg: Dict[str, Any] = _yaml.safe_load(f) or {}

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
                "handoff_threshold": agent_cfg.get("safety_handoff_threshold", 1.2),
            },
            ros2_cfg=agent_cfg.get("ros2", {}),
            include_covariance=bool(baseline_cfg.get("include_covariance", True)),
            include_obstacle_obs=bool(baseline_cfg.get("include_obstacle_obs", True)),
            max_steps=int(agent_cfg.get("max_steps", 500)),
        )

    def prepare(self) -> None:
        """
        @brief Block until sensors are ready, then calibrate the EKF frame offset.

        Call once before run(). The resulting odom-to-world transform is stored
        in self._ekf_odom_offset for every subsequent _get_observation() call.

        @raises RuntimeError if sensors do not become ready within the configured
                covariance_timeout.
        @warning datum_loaded() is checked but not enforced - if the datum was
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

        # Cache static target values
        self._target_x = float(self._target_bay["x"])
        self._target_y = float(self._target_bay["y"])
        self._target_yaw = float(self._target_bay["yaw"])

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

    def _odom_to_world(
        self, raw_ekf_pose: np.ndarray
    ) -> Tuple[float, float, float, float, float]:
        """
        @brief Apply the stored odom-to-world transform to a raw EKF pose.
        @param raw_ekf_pose: Array [odom_x, odom_y, odom_yaw, vyaw, vx_body].
        @return Tuple (world_x, world_y, world_yaw, vyaw, vx_body).
        """
        # ROS REP-103 y is negated relative to lot layout y (same as sim).
        ox = float(raw_ekf_pose[0])
        oy = -float(raw_ekf_pose[1])
        oyaw = float(raw_ekf_pose[2])
        tx, ty, cos_r, sin_r, r = self._ekf_odom_offset
        wx = cos_r * ox - sin_r * oy + tx
        wy = sin_r * ox + cos_r * oy + ty
        wyaw = oyaw + r
        vyaw = float(raw_ekf_pose[3])
        # vx is body-frame, so invariant under the rigid odom-to-world transform.
        vx_body = float(raw_ekf_pose[4]) if len(raw_ekf_pose) > 4 else 0.0
        self._world_pose_buffer[0] = wx
        self._world_pose_buffer[1] = wy
        self._world_pose_buffer[2] = wyaw
        self._world_pose_buffer[3] = vyaw
        self._world_pose_buffer[4] = vx_body
        self._world_pose_valid = True
        return wx, wy, wyaw, vyaw, vx_body

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
            wx, wy, wyaw, vyaw, vx_body = self._odom_to_world(raw_ekf_pose)
            world_pose = np.array([wx, wy, wyaw, vyaw, vx_body], dtype=np.float32)
        else:
            self._world_pose_valid = False
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

    def _apply_action(self, steering: float, throttle: float, brake: float) -> None:
        """
        @brief Publish a calibrated action to the vehicle via geometry_msgs/Twist.
        @param steering: Calibrated steering command in [-1, 1]; mapped to angular.z.
        @param throttle: Calibrated throttle command in [0, 1].
        @param brake: Calibrated brake command in [0, 1].

        throttle and brake combine into a single signed longitudinal command
        (linear.x = throttle - brake) for the downstream /cmd_vel actuation node;
        forward-only, so the result is clamped to [-1, 1]. Publishes on the topic
        configured by ros2.actuation_topic (default: /cmd_vel). The node and
        publisher are initialised lazily on the first call inside run().
        """
        if self._twist_publisher is None:
            raise RuntimeError(
                "_apply_action() called before ROS 2 publisher was initialised. "
                "Call run() rather than invoking _apply_action() directly."
            )

        # throttle drives forward, brake decelerates: combine into one signed axis.
        drive = throttle - brake
        # Scalar clamping is faster than np.clip for individual floats.
        assert self._twist_msg is not None
        self._twist_msg.linear.x = (
            drive if -1.0 <= drive <= 1.0 else (-1.0 if drive < -1.0 else 1.0)
        )
        self._twist_msg.linear.y = 0.0
        self._twist_msg.angular.z = (
            steering if -1.0 <= steering <= 1.0 else (-1.0 if steering < -1.0 else 1.0)
        )
        self._twist_publisher.publish(self._twist_msg)

    def _is_done(self) -> Tuple[bool, bool]:
        """
        @brief Check episode termination conditions from the world pose cached by
               _get_observation() this step.
        @return Tuple (terminated, truncated).
        """
        # Operator override always truncates immediately.
        if self._operator_stop.is_set():
            logger.warning("Operator stop requested - truncating mission.")
            return False, True

        # Reuse the world pose already computed by _get_observation() this step.
        if not self._world_pose_valid:
            logger.debug("_is_done(): EKF pose unavailable - continuing.")
            return False, False

        world_x = float(self._world_pose_buffer[0])
        world_y = float(self._world_pose_buffer[1])
        world_yaw = float(self._world_pose_buffer[2])
        # Body-frame longitudinal velocity (m/s); compared to the same
        # SUCCESS_THRESHOLD_VELOCITY the sim env checks.
        vx_body = float(self._world_pose_buffer[4])

        pos_error = math.hypot(world_x - self._target_x, world_y - self._target_y)
        yaw_error = abs(
            math.atan2(
                math.sin(world_yaw - self._target_yaw),
                math.cos(world_yaw - self._target_yaw),
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

        # Success: all three criteria met simultaneously. Scalar placeholder
        # for the polygon-fit check used in sim - replace once deployment
        # vehicle extents are surveyed and the bay polygon is loaded.
        success = (
            pos_error < _REAL_WORLD_POS_THRESHOLD_M
            and yaw_error < _REAL_WORLD_YAW_THRESHOLD_RAD
            and abs(vx_body) < SUCCESS_THRESHOLD_VELOCITY
        )
        if success:
            logger.info(
                "Parking success: pos_error=%.3f m  yaw_error=%.2f deg  speed=%.3f m/s",
                pos_error,
                math.degrees(yaw_error),
                vx_body,
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
        logger.info("request_stop() called - setting operator stop flag.")
        self._operator_stop.set()

    def _init_ros2(self) -> None:
        """
        @brief Initialise rclpy node, Twist publisher, and pre-allocate Twist messages.

        Pre-allocates self._twist_msg (reused every _apply_action call) and
        self._twist_stop_msg (zero-velocity, published on shutdown).

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
        # Pre-allocate messages to avoid per-step allocation.
        self._twist_msg = Twist()
        self._twist_stop_msg = Twist()
        logger.info("ROS 2 Twist publisher ready on topic '%s'.", self._actuation_topic)

    def _shutdown_ros2(self) -> None:
        """
        @brief Publish a zero-velocity stop command and destroy the ROS 2 node.

        Called at the end of run() to ensure the vehicle is commanded to stop
        regardless of how the mission loop exits. The node is then destroyed
        to release resources.
        """
        if self._twist_publisher is not None:
            try:
                self._twist_publisher.publish(self._twist_stop_msg)
                logger.info("Zero-velocity stop command published.")
            except Exception:
                logger.warning(
                    "Failed to publish stop command during shutdown.", exc_info=True
                )

        if self._ros2_node is not None:
            try:
                self._ros2_node.destroy_node()
            except Exception:
                logger.warning("Failed to destroy ROS 2 node.", exc_info=True)

        self._twist_publisher = None
        self._ros2_node = None
        self._twist_msg = None
        self._twist_stop_msg = None

    def _apply_safety_wrapper(
        self,
        action: np.ndarray,
        epistemic: float,
        aleatoric: float,
    ) -> Tuple[np.ndarray, bool]:
        """
        @brief Apply SafetyWrapper interception logic to a raw policy action.
        @param action: Raw policy action [steering, throttle, brake].
        @param epistemic: Epistemic uncertainty from evidential actor.
        @param aleatoric: Aleatoric uncertainty from evidential actor.
        @return Tuple (modulated_action, handoff_triggered).
        """
        total = epistemic + aleatoric
        modulated, handoff = SafetyWrapper.apply(
            action,
            total_uncertainty=total,
            handoff_threshold=self._handoff_threshold,
        )
        if handoff:
            logger.warning(
                "Safety handoff triggered (total=%.3f >= threshold=%.3f). "
                "Commanding full stop.",
                total,
                self._handoff_threshold,
            )
        return modulated, handoff

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

            with th.no_grad():
                while steps < self._max_steps:
                    obs = self._get_observation()
                    # Zero-copy view: obs_buffer is already float32.
                    obs_tensor = th.from_numpy(obs).unsqueeze_(0)

                    action_tensor, uncertainty_dict = (
                        self._policy.get_action_with_uncertainty(
                            obs_tensor, deterministic=True
                        )
                    )
                    action = action_tensor.detach().numpy()[0]
                    epistemic = uncertainty_dict["epistemic"].mean().item()
                    aleatoric = uncertainty_dict["aleatoric"].mean().item()

                    modulated_action, handoff = self._apply_safety_wrapper(
                        action, epistemic, aleatoric
                    )
                    if handoff:
                        handoff_count += 1

                    steering, throttle, brake = self._deployment.calibrate_action(
                        float(modulated_action[0]),
                        float(modulated_action[1]),
                        float(modulated_action[2]),
                    )
                    self._apply_action(steering, throttle, brake)

                    steps += 1
                    terminated, truncated = self._is_done()

                    if terminated or truncated:
                        break

            if steps >= self._max_steps and not (terminated or truncated):
                logger.warning("Mission timed out after %d steps.", self._max_steps)
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
