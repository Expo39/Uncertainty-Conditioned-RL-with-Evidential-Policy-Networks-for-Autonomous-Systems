"""
@file actuation_calibration.py
@brief Actuation calibration layer for sim-to-real transfer.

Maps the policy's normalised action output [steering, longitudinal] to
physical actuator commands for the real vehicle. In simulation this is a
pass-through (identity mapping). On the real vehicle, each actuator has a
different gain, deadband, and saturation characteristic that must be measured
via a calibration run before deployment.

The calibration is defined in a YAML file (configs/actuation_calibration.yaml)
and loaded at env construction time. When the file is absent or
real_world_deployment=False, the identity mapping is used.

Mapping model (per actuator):
    output = gain * clip(input - deadband_offset, deadband, 1.0) + bias

where:
    gain            -- scale factor (1.0 = no change)
    deadband        -- minimum input magnitude to produce non-zero output
    deadband_offset -- shift applied before deadband test (for asymmetric response)
    bias            -- constant offset added to non-zero outputs
    min_output      -- clamp lower bound on final output
    max_output      -- clamp upper bound on final output

All parameters default to identity (gain=1.0, deadband=0.0, bias=0.0,
min/max = [-1, 1]) so the layer is a safe no-op until calibrated.

@note To populate: perform a calibration run on the real vehicle, measure
      the steering angle and throttle/brake response to known policy outputs,
      and fit the gain/deadband/bias values. Record the procedure in
      documentation/design/actuation_calibration.md.

@see uncertainty_rl.envs.carla_parking.CARLAParkingEnv.step()

@author Antonio Galdes
"""

from typing import Any, Dict, Optional, Tuple

import numpy as np


class ActuatorMap:
    """
    @class ActuatorMap
    @brief Single-actuator mapping from policy output to physical command.
    """

    def __init__(self, params: Dict[str, Any]) -> None:
        """
        @brief Construct from a parameter dict (one actuator block in the YAML).
        @param params: Dict with keys: gain, deadband, deadband_offset, bias,
               min_output, max_output.
        """
        self.gain: float = float(params.get("gain", 1.0))
        self.deadband: float = float(params.get("deadband", 0.0))
        self.deadband_offset: float = float(params.get("deadband_offset", 0.0))
        self.bias: float = float(params.get("bias", 0.0))
        self.min_output: float = float(params.get("min_output", -1.0))
        self.max_output: float = float(params.get("max_output", 1.0))

    def apply(self, value: float) -> float:
        """
        @brief Apply the actuation mapping to a single normalised value.
        @param value: Policy output in [-1, 1].
        @return Physical actuator command, clamped to [min_output, max_output].
        """
        shifted = value - self.deadband_offset
        if abs(shifted) < self.deadband:
            # Inside deadband -- output is zero (or bias if non-zero).
            mapped = self.bias
        else:
            mapped = self.gain * shifted + self.bias
        return float(np.clip(mapped, self.min_output, self.max_output))


class ActuationCalibration:
    """
    @class ActuationCalibration
    @brief Maps policy [steering, longitudinal] outputs to physical actuator commands.

    In simulation (identity mode): pass-through with no transformation.
    In real-world deployment: applies per-actuator gain/deadband/bias from
    configs/actuation_calibration.yaml.

    Usage:
        calibration = ActuationCalibration.from_config(config_path)
        steering_cmd, longitudinal_cmd = calibration.apply(steering, longitudinal)

    @note All parameters default to identity. The calibration YAML is populated
          from a measured calibration run on the real vehicle before deployment.
    @see documentation/design/actuation_calibration.md (to be written at deployment)
    """

    def __init__(
        self,
        steering: Optional[ActuatorMap] = None,
        longitudinal: Optional[ActuatorMap] = None,
    ) -> None:
        """
        @brief Construct with per-actuator maps. Defaults to identity if None.
        @param steering: Steering actuation map. None = identity.
        @param longitudinal: Longitudinal (throttle/brake) actuation map. None = identity.
        """
        self._steering = steering or ActuatorMap({})
        self._longitudinal = longitudinal or ActuatorMap({})

    @classmethod
    def identity(cls) -> "ActuationCalibration":
        """
        @brief Return an identity calibration (no transformation).
        @return ActuationCalibration with default (pass-through) maps.
        """
        return cls()

    @classmethod
    def from_config(cls, config_path: str) -> "ActuationCalibration":
        """
        @brief Load calibration from a YAML file.

        Returns identity calibration if the file is absent or unpopulated.

        @param config_path: Path to actuation_calibration.yaml.
        @return ActuationCalibration instance.
        """
        try:
            import yaml
            with open(config_path, "r") as f:
                doc = yaml.safe_load(f) or {}
            cal = doc.get("calibration", {})
            steering_map = ActuatorMap(cal.get("steering", {}))
            longitudinal_map = ActuatorMap(cal.get("longitudinal", {}))
            return cls(steering=steering_map, longitudinal=longitudinal_map)
        except (OSError, KeyError):
            return cls.identity()

    def apply(self, steering: float, longitudinal: float) -> Tuple[float, float]:
        """
        @brief Apply calibration to policy action outputs.
        @param steering: Policy steering output in [-1, 1].
        @param longitudinal: Policy longitudinal output in [-1, 1].
        @return Tuple (steering_cmd, longitudinal_cmd) mapped to physical range.
        """
        return self._steering.apply(steering), self._longitudinal.apply(longitudinal)
