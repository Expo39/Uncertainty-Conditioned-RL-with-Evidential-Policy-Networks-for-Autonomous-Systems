"""
@file actuation_calibration.py
@brief Actuation calibration layer for sim-to-real transfer.

Maps normalised policy actions [steering, drive, brake] to physical actuator
commands.
"""

from typing import Any, Dict, Optional, Tuple


class ActuatorMap:
    """
    @class ActuatorMap
    @brief Single-actuator mapping from policy output to physical command.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

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
        mapped = (
            self.bias
            if abs(shifted) < self.deadband
            else self.gain * shifted + self.bias
        )
        if mapped < self.min_output:
            return self.min_output
        if mapped > self.max_output:
            return self.max_output
        return mapped


class ActuationCalibration:
    """
    @class ActuationCalibration
    @brief Maps policy [steering, drive, brake] outputs to physical actuator commands.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        steering: Optional[ActuatorMap] = None,
        drive: Optional[ActuatorMap] = None,
        brake: Optional[ActuatorMap] = None,
    ) -> None:
        """
        @brief Construct with per-actuator maps. Defaults to identity if None.
        @param steering: Steering actuation map. None = identity.
        @param drive: Drive (signed throttle / reverse) actuation map. None = identity.
        @param brake: Friction brake actuation map. None = identity.
        """
        self._steering = steering or ActuatorMap({})
        self._drive = drive or ActuatorMap({})
        self._brake = brake or ActuatorMap({})

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
            drive_map = ActuatorMap(cal.get("drive", {}))
            brake_map = ActuatorMap(cal.get("brake", {}))
            return cls(steering=steering_map, drive=drive_map, brake=brake_map)
        except (OSError, KeyError):
            return cls.identity()

    def apply(
        self, steering: float, drive: float, brake: float
    ) -> Tuple[float, float, float]:
        """
        @brief Apply calibration to policy action outputs.
        @param steering: Policy steering output in [-1, 1].
        @param drive: Policy drive output in [-1, 1]; negative = reverse.
        @param brake: Policy brake output in [0, 1].
        @return Tuple (steering_cmd, drive_cmd, brake_cmd) mapped to physical range.
        """
        return (
            self._steering.apply(steering),
            self._drive.apply(drive),
            self._brake.apply(brake),
        )
