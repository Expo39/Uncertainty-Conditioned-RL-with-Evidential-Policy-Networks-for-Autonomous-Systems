"""
@file inference_loop.py
@brief Real-world inference loop stub for the autonomous parking system.

This module defines the inference loop that runs the trained evidential policy
on a physical vehicle. It mirrors the evaluation loop in evaluate.py but
replaces CARLAParkingEnv with the physical vehicle API.

The SafetyWrapper is always active in real-world deployment:
  - Aleatoric uncertainty -> longitudinal speed cap (unpredictable outcomes).
  - Epistemic uncertainty -> full stop + handoff (out-of-distribution state).

Architecture:

    obs (from EKF + LiDAR) -> EvidentialActorCriticPolicy
                                    |
                          gamma, nu, alpha, beta
                                    |
                          epistemic, aleatoric
                                    |
                              SafetyWrapper
                                    |
                          action[0] = steering
                          action[1] = longitudinal (capped)
                                    |
                          RealWorldDeployment.calibrate_action()
                                    |
                          physical vehicle actuators

@note This file is a stub. The physical vehicle API calls are marked with
      TODO(AG) and must be implemented at the deployment site once the
      vehicle interface is confirmed.

@todo(AG) Implement physical vehicle observation pipeline (EKF state read,
          LiDAR scan parsing, target bay relative pose computation).
@todo(AG) Implement physical vehicle actuation API (CAN bus / ROS 2 topic).
@todo(AG) Implement episode termination conditions for real-world operation
          (manual override, geofence breach, success detection).

@see uncertainty_rl.envs.safety_wrapper.SafetyWrapper
@see uncertainty_rl.envs.real.real_world_deployment.RealWorldDeployment
@see uncertainty_rl.evaluation.evaluate for the simulation equivalent.

@author Antonio Galdes
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger("uncertainty_rl.envs.real.inference_loop")


class RealWorldInferenceLoop:
    """
    @class RealWorldInferenceLoop
    @brief Runs the trained policy on a physical vehicle with SafetyWrapper active.

    Intended usage at the deployment site:

        loop = RealWorldInferenceLoop.from_config(
            model_path="outputs/checkpoints/best_model.zip",
            mission_path="configs/deployment/real/mission.yaml",
            datum_path="configs/deployment/real/real_world_datum.yaml",
            calibration_path="configs/deployment/real/actuation_calibration.yaml",
            safety_aleatoric_scaling=0.5,
            safety_handoff_threshold=5.0,
        )
        loop.run()

    @note All TODO(AG) blocks mark where physical vehicle API calls must be
          inserted. Everything else (policy inference, uncertainty extraction,
          SafetyWrapper interception, actuation calibration) is fully implemented.
    """

    def __init__(
        self,
        policy: Any,
        deployment: Any,
        safety_wrapper_cfg: Dict[str, float],
        max_steps: int = 500,
    ) -> None:
        """
        @brief Construct the inference loop.
        @param policy: Loaded EvidentialActorCriticPolicy (from EvidentialPPO).
        @param deployment: RealWorldDeployment instance (datum + calibration).
        @param safety_wrapper_cfg: Dict with keys aleatoric_scaling, handoff_threshold.
        @param max_steps: Maximum control steps before mission timeout.
        """
        self._policy = policy
        self._deployment = deployment
        self._max_steps = max_steps
        self._aleatoric_scaling: float = safety_wrapper_cfg.get(
            "aleatoric_scaling", 0.5
        )
        self._handoff_threshold: float = safety_wrapper_cfg.get(
            "handoff_threshold", 5.0
        )

    @classmethod
    def from_config(
        cls,
        mission_path: str,
        agent_config_path: str = "configs/deployment/agent_config.yaml",
    ) -> "RealWorldInferenceLoop":
        """
        @brief Load model and all deployment configs, construct the loop.

        All parameters are read from YAML -- no duplication in code:
          - model_path, max_steps         -> agent_config.yaml
          - safety params                 -> agent_config.yaml
          - datum, calibration paths      -> agent_config.yaml
          - target_bay_id, layout_file    -> mission.yaml

        @param mission_path: Path to configs/deployment/real/mission.yaml.
        @param agent_config_path: Path to configs/deployment/agent_config.yaml.
        @return Constructed RealWorldInferenceLoop.
        """
        try:
            import yaml as _yaml

            from uncertainty_rl.envs.real.real_world_deployment import (
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

        deployment, _target_bay = RealWorldDeployment.from_mission(
            mission_path=mission_path,
            datum_path=agent_cfg.get("real_world_datum"),
            calibration_path=agent_cfg.get("actuation_calibration"),
        )

        return cls(
            policy=model.policy,
            deployment=deployment,
            safety_wrapper_cfg={
                "aleatoric_scaling": agent_cfg.get("safety_aleatoric_scaling", 0.5),
                "handoff_threshold": agent_cfg.get("safety_handoff_threshold", 5.0),
            },
            max_steps=int(agent_cfg.get("max_steps", 500)),
        )

    def _get_observation(self) -> np.ndarray:
        """
        @brief Read the current observation from physical sensors.
        @return Observation vector matching the trained policy's obs space.

        @todo(AG) Implement:
            1. Read EKF filtered pose + covariance from /odometry/filtered
               (via rclpy subscription or shared JSON written by CovarianceExtractorNode).
            2. Parse 2D LiDAR scan for hemispheric clearance features.
            3. Compute dx/dy/dyaw relative to target bay in ego body frame
               (use RealWorldDeployment.reference_pose() for EKF frame offset).
            4. Assemble and return the obs vector in the same order as
               CARLAParkingEnv._get_state() (see envs/sim/carla_parking.py).
        """
        raise NotImplementedError(
            "Physical observation pipeline not yet implemented. "
            "See TODO(AG) in _get_observation()."
        )

    def _apply_action(self, steering: float, longitudinal: float) -> None:
        """
        @brief Send calibrated action to the physical vehicle actuators.
        @param steering: Calibrated steering command in [-1, 1].
        @param longitudinal: Calibrated longitudinal command in [-1, 1].

        @todo(AG) Implement:
            Publish to the vehicle's actuation interface, e.g.:
            - ROS 2 topic: geometry_msgs/Twist or custom VehicleControl msg.
            - CAN bus command via socketcan or proprietary SDK.
        """
        raise NotImplementedError(
            "Physical actuation interface not yet implemented. "
            "See TODO(AG) in _apply_action()."
        )

    def _is_done(self) -> Tuple[bool, bool]:
        """
        @brief Check episode termination conditions.
        @return Tuple (terminated, truncated).

        @todo(AG) Implement:
            - terminated: success (position/orientation/velocity within thresholds)
              or collision detected via bumper sensor / sudden deceleration.
            - truncated: manual operator override, geofence breach, or timeout.
        """
        raise NotImplementedError(
            "Real-world termination conditions not yet implemented. "
            "See TODO(AG) in _is_done()."
        )

    def _apply_safety_wrapper(
        self,
        action: np.ndarray,
        epistemic: float,
        aleatoric: float,
    ) -> Tuple[np.ndarray, bool]:
        """
        @brief Apply SafetyWrapper logic directly (mirrors SafetyWrapper.step()).

        Replicates the wrapper inline rather than wrapping a Gymnasium env
        because the real-world loop does not use a Gymnasium env instance.

        @param action: Raw policy action [steering, longitudinal].
        @param epistemic: Epistemic uncertainty from evidential actor.
        @param aleatoric: Aleatoric uncertainty from evidential actor.
        @return Tuple (modulated_action, handoff_triggered).
        """
        modulated = action.copy()

        # Aleatoric: cap longitudinal only -- steering is unrestricted.
        aleatoric_scale = 1.0 / (1.0 + self._aleatoric_scaling * aleatoric)
        modulated[1] = float(np.clip(modulated[1], -1.0, aleatoric_scale))

        # Epistemic: full stop if above threshold.
        handoff = epistemic >= self._handoff_threshold
        if handoff:
            modulated = np.zeros_like(action)
            logger.warning(
                "Safety handoff triggered (epistemic=%.3f >= threshold=%.3f). "
                "Commanding full stop.",
                epistemic,
                self._handoff_threshold,
            )

        return modulated, handoff

    def run(self) -> Dict[str, Any]:
        """
        @brief Run one parking mission on the physical vehicle.
        @return Dict with mission outcome stats.

        @note max_steps is set from agent_config.yaml via from_config().
        """
        import torch as th

        logger.info("Starting real-world parking mission (max_steps=%d).", self._max_steps)
        steps = 0
        handoff_count = 0

        while steps < self._max_steps:
            obs = self._get_observation()
            obs_tensor = th.as_tensor(obs[np.newaxis], dtype=th.float32)

            # Policy inference -- single forward pass.
            action_tensor, uncertainty_dict = (
                self._policy.get_action_with_uncertainty(
                    obs_tensor, deterministic=True
                )
            )
            action = action_tensor.cpu().numpy()[0]
            epistemic = float(uncertainty_dict["epistemic"].mean().item())
            aleatoric = float(uncertainty_dict["aleatoric"].mean().item())

            # Safety wrapper interception.
            modulated_action, handoff = self._apply_safety_wrapper(
                action, epistemic, aleatoric
            )

            if handoff:
                handoff_count += 1

            # Actuation calibration then send to vehicle.
            steering, longitudinal = self._deployment.calibrate_action(
                float(modulated_action[0]),
                float(modulated_action[1]),
            )
            self._apply_action(steering, longitudinal)

            steps += 1
            terminated, truncated = self._is_done()

            if terminated or truncated:
                break

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
