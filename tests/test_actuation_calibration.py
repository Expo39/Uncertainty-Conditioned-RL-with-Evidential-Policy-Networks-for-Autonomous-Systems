"""
@file test_actuation_calibration.py
@brief Unit tests for ActuatorMap and ActuationCalibration.

CPU-only. No CARLA, ROS 2, or GPU required.
"""

import tempfile
from pathlib import Path

import pytest
import yaml

from uncertainty_rl.utils.actuation_calibration import (
    ActuationCalibration,
    ActuatorMap,
)


# ---------------------------------------------------------------------------
# TestActuatorMap
# ---------------------------------------------------------------------------


class TestActuatorMap:
    """
    @class TestActuatorMap
    @brief Tests for the single-actuator mapping.
    """

    def test_identity_params_pass_through(self) -> None:
        """
        @brief Default params (gain=1, deadband=0, bias=0) must be identity.
        """
        m = ActuatorMap({})
        assert m.apply(0.5) == pytest.approx(0.5)
        assert m.apply(-0.7) == pytest.approx(-0.7)
        assert m.apply(0.0) == pytest.approx(0.0)

    def test_gain_scales_output(self) -> None:
        """
        @brief Output must be gain * input when no deadband or bias.
        """
        m = ActuatorMap({"gain": 2.0})
        assert m.apply(0.4) == pytest.approx(0.8)

    def test_deadband_zeroes_small_inputs(self) -> None:
        """
        @brief Inputs whose |shifted value| < deadband must return bias (default 0).
        """
        m = ActuatorMap({"deadband": 0.1})
        assert m.apply(0.05) == pytest.approx(0.0)
        assert m.apply(-0.05) == pytest.approx(0.0)

    def test_input_outside_deadband_is_scaled(self) -> None:
        """
        @brief Input outside deadband must be mapped via gain.
        """
        m = ActuatorMap({"gain": 1.0, "deadband": 0.1})
        # |0.5| > 0.1, so mapped = 1.0 * 0.5 = 0.5
        assert m.apply(0.5) == pytest.approx(0.5)

    def test_bias_added_inside_deadband(self) -> None:
        """
        @brief When inside deadband, output is bias (not zero) when bias != 0.
        """
        m = ActuatorMap({"deadband": 0.2, "bias": 0.05})
        assert m.apply(0.1) == pytest.approx(0.05)

    def test_output_clamped_to_min_max(self) -> None:
        """
        @brief Output must be clamped to [min_output, max_output].
        """
        m = ActuatorMap({"gain": 10.0, "min_output": -0.5, "max_output": 0.5})
        assert m.apply(1.0) == pytest.approx(0.5)
        assert m.apply(-1.0) == pytest.approx(-0.5)

    def test_deadband_offset_shifts_zero_point(self) -> None:
        """
        @brief deadband_offset shifts the centre of the deadband.
        """
        m = ActuatorMap({"deadband": 0.1, "deadband_offset": 0.5})
        # shifted = 0.55 - 0.5 = 0.05, inside deadband -> output = 0
        assert m.apply(0.55) == pytest.approx(0.0)
        # shifted = 0.7 - 0.5 = 0.2, outside deadband -> output = 0.2
        assert m.apply(0.7) == pytest.approx(0.2)


# ---------------------------------------------------------------------------
# TestActuationCalibrationIdentity
# ---------------------------------------------------------------------------


class TestActuationCalibrationIdentity:
    """
    @class TestActuationCalibrationIdentity
    @brief Tests for the identity (simulation) calibration.
    """

    def test_identity_classmethod_returns_instance(self) -> None:
        """
        @brief ActuationCalibration.identity() must return an instance.
        """
        cal = ActuationCalibration.identity()
        assert isinstance(cal, ActuationCalibration)

    def test_identity_apply_is_passthrough(self) -> None:
        """
        @brief identity().apply() must return inputs unchanged.
        """
        cal = ActuationCalibration.identity()
        s, lon = cal.apply(0.3, -0.5)
        assert s == pytest.approx(0.3)
        assert lon == pytest.approx(-0.5)

    def test_default_constructor_is_identity(self) -> None:
        """
        @brief ActuationCalibration() with no args must behave as identity.
        """
        cal = ActuationCalibration()
        s, lon = cal.apply(0.8, 0.2)
        assert s == pytest.approx(0.8)
        assert lon == pytest.approx(0.2)


# ---------------------------------------------------------------------------
# TestActuationCalibrationFromConfig
# ---------------------------------------------------------------------------


class TestActuationCalibrationFromConfig:
    """
    @class TestActuationCalibrationFromConfig
    @brief Tests for loading calibration from a YAML file.
    """

    def _write_config(self, path: Path, data: dict) -> None:
        with open(path, "w") as f:
            yaml.dump(data, f)

    def test_loads_steering_gain(self) -> None:
        """
        @brief Steering gain from YAML must be applied.
        """
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "cal.yaml"
            self._write_config(p, {
                "calibration": {
                    "steering": {"gain": 0.5},
                    "longitudinal": {},
                }
            })
            cal = ActuationCalibration.from_config(str(p))
            s, _ = cal.apply(1.0, 0.0)
            assert s == pytest.approx(0.5)

    def test_loads_longitudinal_deadband(self) -> None:
        """
        @brief Longitudinal deadband from YAML must suppress small inputs.
        """
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "cal.yaml"
            self._write_config(p, {
                "calibration": {
                    "steering": {},
                    "longitudinal": {"deadband": 0.15},
                }
            })
            cal = ActuationCalibration.from_config(str(p))
            _, lon = cal.apply(0.0, 0.05)
            assert lon == pytest.approx(0.0)

    def test_missing_file_returns_identity(self) -> None:
        """
        @brief from_config() must return identity when the file does not exist.
        """
        cal = ActuationCalibration.from_config("/nonexistent/path/cal.yaml")
        s, lon = cal.apply(0.6, -0.3)
        assert s == pytest.approx(0.6)
        assert lon == pytest.approx(-0.3)

    def test_empty_yaml_returns_identity(self) -> None:
        """
        @brief from_config() must return identity when the YAML is empty.
        """
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "empty.yaml"
            p.write_text("")
            cal = ActuationCalibration.from_config(str(p))
            s, lon = cal.apply(0.4, 0.9)
            assert s == pytest.approx(0.4)
            assert lon == pytest.approx(0.9)

    def test_apply_returns_tuple_of_two_floats(self) -> None:
        """
        @brief apply() must return a 2-tuple of floats.
        """
        cal = ActuationCalibration.identity()
        result = cal.apply(0.1, 0.2)
        assert len(result) == 2
        assert isinstance(result[0], float)
        assert isinstance(result[1], float)
