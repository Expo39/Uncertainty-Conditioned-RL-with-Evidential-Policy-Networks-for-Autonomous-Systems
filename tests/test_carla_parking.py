"""
@file test_carla_parking.py
@brief Unit tests for the CARLA parking environment and geometry helpers.

All tests are CPU-only (no CARLA server, no ROS 2, no GPU). The environment
falls back gracefully when CARLA and rclpy are unavailable.
"""

import json
import math
import tempfile
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock

import numpy as np
import pytest

from uncertainty_rl.envs._parking_core import (
    build_observation,
    compute_obs_dim,
    extract_obstacle_features,
    load_floor_plan,
    wait_for_ekf,
)
from uncertainty_rl.utils.constants import (
    ACTION_DIM,
    COVARIANCE_FEATURES_DIM,
    OBSTACLE_FEATURES_DIM,
    TARGET_POSE_DIM,
    TOTAL_OBS_DIM,
    VEHICLE_STATE_DIM,
)
from uncertainty_rl.utils.geometry import (
    _compute_relative_target_pose,
    _interpolate_cone_positions,
    point_in_polygon,
    wrap_angle_symmetric,
    yaw_from_quaternion,
    zone_bbox,
)
from uncertainty_rl.utils.visualisation import VisStateWriter

# ---------------------------------------------------------------------------
# Pure geometry: _interpolate_cone_positions
# ---------------------------------------------------------------------------


class TestInterpolateConePositions:
    """
    @class TestInterpolateConePositions
    @brief Tests for perimeter cone interpolation.
    """

    def test_rectangle_four_edges(self) -> None:
        """
        @brief A 10m x 8m rectangle with 2m spacing should produce at least 4 cones.
        """
        corners = [(0.0, 0.0), (10.0, 0.0), (10.0, 8.0), (0.0, 8.0)]
        positions = _interpolate_cone_positions(corners, spacing=2.0)
        assert len(positions) >= 4

    def test_spacing_respected(self) -> None:
        """
        @brief No two adjacent cones on the same edge should be further than
               spacing + small tolerance apart.
        """
        corners = [(0.0, 0.0), (20.0, 0.0), (20.0, 10.0), (0.0, 10.0)]
        spacing = 3.0
        positions = _interpolate_cone_positions(corners, spacing=spacing)

        # Check distances between consecutive positions on first edge (y ~ 0)
        edge_pts = [(x, y) for x, y, _ in positions if abs(y) < 0.01]
        for i in range(len(edge_pts) - 1):
            dx = edge_pts[i + 1][0] - edge_pts[i][0]
            dy = edge_pts[i + 1][1] - edge_pts[i][1]
            dist = math.sqrt(dx * dx + dy * dy)
            assert (
                dist <= spacing + 0.5
            ), f"Spacing {dist:.2f} exceeds {spacing + 0.5:.2f}"

    def test_count_scales_with_perimeter(self) -> None:
        """
        @brief Larger perimeter at same spacing produces more cones.
        """
        small = [(0.0, 0.0), (5.0, 0.0), (5.0, 5.0), (0.0, 5.0)]
        large = [(0.0, 0.0), (20.0, 0.0), (20.0, 20.0), (0.0, 20.0)]
        spacing = 2.0
        assert len(_interpolate_cone_positions(large, spacing)) > len(
            _interpolate_cone_positions(small, spacing)
        )

    def test_empty_corners_returns_empty(self) -> None:
        """
        @brief Empty corner list should return empty positions.
        """
        positions = _interpolate_cone_positions([], spacing=2.0)
        assert positions == []


# ---------------------------------------------------------------------------
# Pure geometry: _compute_relative_target_pose
# ---------------------------------------------------------------------------


class TestComputeRelativeTargetPose:
    """
    @class TestComputeRelativeTargetPose
    @brief Tests for ego-frame target pose computation.
    """

    def test_identity_ego_at_target(self) -> None:
        """
        @brief When ego is at target, relative pose should be (0, 0, 0).
        """
        dx, dy, dyaw = _compute_relative_target_pose(
            x_ego=5.0,
            y_ego=3.0,
            yaw_ego=0.5,
            x_target=5.0,
            y_target=3.0,
            yaw_target=0.5,
        )
        assert abs(dx) < 1e-6
        assert abs(dy) < 1e-6
        assert abs(dyaw) < 1e-6

    def test_target_directly_ahead(self) -> None:
        """
        @brief Target directly ahead (same yaw, positive x offset) -> dx > 0, dy ~ 0.
        """
        dx, dy, dyaw = _compute_relative_target_pose(
            x_ego=0.0,
            y_ego=0.0,
            yaw_ego=0.0,
            x_target=10.0,
            y_target=0.0,
            yaw_target=0.0,
        )
        assert dx > 0.0
        assert abs(dy) < 1e-6
        assert abs(dyaw) < 1e-6

    def test_target_to_left(self) -> None:
        """
        @brief Target 5m to the left (ego facing +x) -> dy > 0, dx ~ 0.
        """
        dx, dy, dyaw = _compute_relative_target_pose(
            x_ego=0.0,
            y_ego=0.0,
            yaw_ego=0.0,
            x_target=0.0,
            y_target=5.0,
            yaw_target=0.0,
        )
        assert abs(dx) < 1e-6
        assert dy > 0.0

    def test_yaw_wrap_in_range(self) -> None:
        """
        @brief dyaw always in (-pi, pi] regardless of raw angle difference.
        """
        _, _, dyaw = _compute_relative_target_pose(
            x_ego=0.0,
            y_ego=0.0,
            yaw_ego=0.1,
            x_target=0.0,
            y_target=0.0,
            yaw_target=3.0,
        )
        assert -math.pi < dyaw <= math.pi

    def test_rotated_ego_frame(self) -> None:
        """
        @brief Ego facing +y: target ahead in world +y -> dx > 0 in body frame.
        """
        yaw_ego = math.pi / 2.0  # Facing +y
        dx, dy, dyaw = _compute_relative_target_pose(
            x_ego=0.0,
            y_ego=0.0,
            yaw_ego=yaw_ego,
            x_target=0.0,
            y_target=8.0,
            yaw_target=yaw_ego,
        )
        assert dx > 0.0
        assert abs(dy) < 1e-5


# ---------------------------------------------------------------------------
# Observation space shape tests
# ---------------------------------------------------------------------------


class TestObservationSpaceShape:
    """
    @class TestObservationSpaceShape
    @brief Verify obs space dim based on include_covariance flag.
    """

    def test_12_dim_with_covariance(self) -> None:
        """
        @brief include_covariance=True -> 12-dim observation space (default).
        """
        from uncertainty_rl.envs.sim.carla_parking import CARLAParkingEnv

        env = CARLAParkingEnv(max_steps=5, include_covariance=True)
        assert env.observation_space.shape == (TOTAL_OBS_DIM,)  # 12
        env.close()

    def test_9_dim_without_covariance(self) -> None:
        """
        @brief include_covariance=False, include_obstacle_obs=True (default) -> 9-dim.
        """
        from uncertainty_rl.envs.sim.carla_parking import CARLAParkingEnv

        env = CARLAParkingEnv(max_steps=5, include_covariance=False)
        expected = VEHICLE_STATE_DIM + TARGET_POSE_DIM + OBSTACLE_FEATURES_DIM  # 9
        assert env.observation_space.shape == (expected,)
        env.close()

    def test_4_dim_without_covariance_or_obstacles(self) -> None:
        """
        @brief include_covariance=False, include_obstacle_obs=False -> 4-dim.
        """
        from uncertainty_rl.envs.sim.carla_parking import CARLAParkingEnv

        env = CARLAParkingEnv(
            max_steps=5, include_covariance=False, include_obstacle_obs=False
        )
        expected = VEHICLE_STATE_DIM + TARGET_POSE_DIM  # 1 + 3 = 4
        assert env.observation_space.shape == (expected,)
        env.close()

    def test_action_space_shape(self) -> None:
        """
        @brief Action space must be 2-dim: [steering, drive].
        """
        from uncertainty_rl.envs.sim.carla_parking import CARLAParkingEnv

        env = CARLAParkingEnv(max_steps=5)
        assert env.action_space.shape == (ACTION_DIM,)
        env.close()

    def test_action_space_bounds(self) -> None:
        """
        @brief steering in [-1,1], drive in [-1,1] (positive=throttle,
               negative=brake).
        """
        from uncertainty_rl.envs.sim.carla_parking import CARLAParkingEnv

        env = CARLAParkingEnv(max_steps=5)
        np.testing.assert_array_equal(env.action_space.low, [-1.0, -1.0])
        np.testing.assert_array_equal(env.action_space.high, [1.0, 1.0])
        env.close()


# ---------------------------------------------------------------------------
# Gymnasium API contract (requires live CARLA server - integration only)
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestGymnasiumAPIContract:
    """
    @class TestGymnasiumAPIContract
    @brief Verify reset() and step() return correct types with a live CARLA server.
    """

    def test_reset_returns_tuple_of_obs_and_info(self) -> None:
        """
        @brief reset() -> (np.ndarray, dict).
        """
        from uncertainty_rl.envs.sim.carla_parking import CARLAParkingEnv

        env = CARLAParkingEnv(max_steps=5)
        result = env.reset()
        assert isinstance(result, tuple) and len(result) == 2
        obs, info = result
        assert isinstance(obs, np.ndarray)
        assert isinstance(info, dict)
        env.close()

    def test_reset_obs_shape_21(self) -> None:
        """
        @brief reset() obs shape must be (21,) when include_covariance=True (default).
        """
        from uncertainty_rl.envs.sim.carla_parking import CARLAParkingEnv

        env = CARLAParkingEnv(max_steps=5, include_covariance=True)
        obs, _ = env.reset()
        assert obs.shape == (TOTAL_OBS_DIM,)
        env.close()

    def test_reset_obs_shape_9_no_cov(self) -> None:
        """
        @brief reset() observation shape must be (9,) when include_covariance=False.
        """
        from uncertainty_rl.envs.sim.carla_parking import CARLAParkingEnv

        env = CARLAParkingEnv(max_steps=5, include_covariance=False)
        obs, _ = env.reset()
        expected = VEHICLE_STATE_DIM + TARGET_POSE_DIM
        assert obs.shape == (expected,)
        env.close()

    def test_reset_obs_dtype_float32(self) -> None:
        """
        @brief Observation must be float32.
        """
        from uncertainty_rl.envs.sim.carla_parking import CARLAParkingEnv

        env = CARLAParkingEnv(max_steps=5)
        obs, _ = env.reset()
        assert obs.dtype == np.float32
        env.close()

    def test_step_returns_five_tuple(self) -> None:
        """
        @brief step() -> (obs, reward, terminated, truncated, info).
        """
        from uncertainty_rl.envs.sim.carla_parking import CARLAParkingEnv

        env = CARLAParkingEnv(max_steps=5)
        env.reset()
        action = env.action_space.sample()
        result = env.step(action)
        assert isinstance(result, tuple) and len(result) == 5
        obs, reward, terminated, truncated, info = result
        assert obs.shape[0] == TOTAL_OBS_DIM
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)
        env.close()

    def test_truncated_at_max_steps(self) -> None:
        """
        @brief truncated=True after max_steps, steps counter does not exceed max.
        """
        from uncertainty_rl.envs.sim.carla_parking import CARLAParkingEnv

        env = CARLAParkingEnv(max_steps=3)
        env.reset()
        for _ in range(3):
            _, _, terminated, truncated, _ = env.step(env.action_space.sample())
            if terminated or truncated:
                break
        assert env.steps <= 3
        env.close()


# ---------------------------------------------------------------------------
# Bay sampling helpers (via mock layout)
# ---------------------------------------------------------------------------


class TestBaySampling:
    """
    @class TestBaySampling
    @brief Tests for _sample_target_bay() via mocked floor plan layouts.
    """

    def _make_env_with_layout(self, layout: Dict[str, Any]) -> Any:
        from uncertainty_rl.envs.sim.carla_parking import CARLAParkingEnv

        env = CARLAParkingEnv(max_steps=5)
        env._current_layout = layout
        # _sample_target_bay() reads _bays_by_type / _bay_type_keys, which are
        # normally populated by _load_floor_plan(). Group the supplied bays here
        # so the helper does not require a CARLA connection.
        eligible = [
            b for b in layout.get("bays", []) if not b.get("always_empty", False)
        ]
        bays_by_type: Dict[str, List[Dict[str, Any]]] = {}
        for bay in eligible:
            bay_type_key = bay.get("bay_type", "perpendicular")
            bays_by_type.setdefault(bay_type_key, []).append(bay)
        env._bays_by_type = bays_by_type
        env._bay_type_keys = list(bays_by_type.keys())
        return env

    def test_all_bay_types_reachable(self) -> None:
        """
        @brief With 5 bays per type, all three types should be sampled in 300 trials.
        """
        bays: List[Dict[str, Any]] = []
        for i in range(5):
            bays.append(
                {
                    "bay_id": f"perp_{i}",
                    "bay_type": "perpendicular",
                    "x": float(i * 3),
                    "y": 0.0,
                    "yaw": 0.0,
                    "width": 2.5,
                    "depth": 5.0,
                }
            )
            bays.append(
                {
                    "bay_id": f"angl_{i}",
                    "bay_type": "angled",
                    "x": float(i * 3),
                    "y": 10.0,
                    "yaw": 0.785,
                    "width": 2.5,
                    "depth": 5.4,
                }
            )
            bays.append(
                {
                    "bay_id": f"para_{i}",
                    "bay_type": "parallel",
                    "x": float(i * 9),
                    "y": 20.0,
                    "yaw": 0.0,
                    "width": 2.5,
                    "depth": 8.0,
                }
            )

        env = self._make_env_with_layout({"bays": bays})

        seen_types = set()
        for _ in range(300):
            env._sample_target_bay()
            seen_types.add(env._target_bay.get("bay_type"))

        assert seen_types == {"perpendicular", "angled", "parallel"}
        env.close()


# ---------------------------------------------------------------------------
# VisStateWriter
# ---------------------------------------------------------------------------


class TestVisStateWriter:
    """
    @class TestVisStateWriter
    @brief Tests for the atomic vis_state.json writer.
    """

    def test_write_produces_valid_json(self) -> None:
        """
        @brief write() must produce valid JSON at the output path.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            out = Path(tmp_dir) / "vis_state.json"
            writer = VisStateWriter(out)

            writer.write(
                ego_transform={"x": 1.0, "y": 2.0, "yaw": 0.3},
                actor_transforms=[{"x": 5.0, "y": 5.0, "yaw": 0.0, "type": "npc"}],
                target_bay={
                    "x": 10.0,
                    "y": 0.0,
                    "yaw": 1.5,
                    "bay_type": "perpendicular",
                },
                episode_info={"step": 42, "floor_plan": "rectangle"},
                trajectory=[(0.0, 0.0), (0.5, 0.1)],
            )

            assert out.exists()
            data = json.loads(out.read_text())
            assert data["ego"]["x"] == pytest.approx(1.0)
            assert data["episode_info"]["step"] == 42

    def test_tmp_file_cleaned_up(self) -> None:
        """
        @brief The .tmp file must not remain after write() completes.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            out = Path(tmp_dir) / "vis_state.json"
            writer = VisStateWriter(out)

            writer.write(
                ego_transform={"x": 0.0, "y": 0.0, "yaw": 0.0},
                actor_transforms=[],
                target_bay={"x": 0.0, "y": 0.0, "yaw": 0.0},
                episode_info={},
            )

            tmp = out.with_suffix(".tmp")
            assert not tmp.exists(), ".tmp file should be removed by os.replace"

    def test_write_round_trip_keys(self) -> None:
        """
        @brief All top-level keys must be present in the written JSON.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            out = Path(tmp_dir) / "vis_state.json"
            writer = VisStateWriter(out)

            writer.write(
                ego_transform={"x": 3.0, "y": 4.0, "yaw": 0.0},
                actor_transforms=[],
                target_bay={"bay_id": "t1"},
                episode_info={"floor_plan": "trapezoid"},
                bays=[{"bay_id": "b1"}],
                corners=[{"x": 0.0, "y": 0.0}],
                pedestrians=[{"x": 1.0, "y": 1.0}],
            )

            data = json.loads(out.read_text())
            for key in [
                "ego",
                "actors",
                "target_bay",
                "episode_info",
                "trajectory",
                "bays",
                "corners",
                "pedestrians",
            ]:
                assert key in data, f"Missing key: {key}"

    def test_write_creates_parent_dirs(self) -> None:
        """
        @brief write() must create parent directories if they do not exist.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            out = Path(tmp_dir) / "subdir" / "deep" / "vis_state.json"
            writer = VisStateWriter(out)

            writer.write(
                ego_transform={"x": 0.0, "y": 0.0, "yaw": 0.0},
                actor_transforms=[],
                target_bay={},
                episode_info={},
            )

            assert out.exists()


# ---------------------------------------------------------------------------
# Covariance features in the observation
# ---------------------------------------------------------------------------


class TestCovarianceObservation:
    """
    @class TestCovarianceObservation
    @brief Verify EKF covariance features are written to obs[1:1+COVARIANCE_FEATURES_DIM].

    The EKF produces COVARIANCE_FEATURES_DIM (3) uncertainty features
    (std_x, std_y, std_yaw). _get_state() writes them verbatim into the
    observation at indices 1-3, immediately after the vyaw scalar.
    """

    def test_covariance_features_written_to_obs(self) -> None:
        """
        @brief Covariance features appear unchanged in obs[1:1+COVARIANCE_FEATURES_DIM].
        """
        from unittest.mock import MagicMock

        from uncertainty_rl.envs.sim.carla_parking import CARLAParkingEnv
        from uncertainty_rl.utils.constants import COVARIANCE_FEATURES_DIM

        env = CARLAParkingEnv(max_steps=5, include_covariance=True)

        # Inject a mock subscriber that returns known covariance values.
        raw = np.array([1.0, 4.0, 9.0], dtype=np.float32)
        assert len(raw) == COVARIANCE_FEATURES_DIM

        # The _read_ekf_state callable is bound at construction time, when the
        # subscriber is still None. Override it directly so _get_state() reads
        # the test's covariance values.
        env._read_ekf_state = lambda: (None, raw.copy())

        # Simulate a minimal vehicle mock so _get_state() doesn't early-return.
        mock_vehicle = MagicMock()
        mock_vehicle.get_transform.return_value = MagicMock(
            location=MagicMock(x=0.0, y=0.0),
            rotation=MagicMock(yaw=0.0),
        )
        mock_vehicle.get_velocity.return_value = MagicMock(x=0.0, y=0.0)
        mock_vehicle.get_angular_velocity.return_value = MagicMock(z=0.0)
        env.vehicle = mock_vehicle
        env.world = MagicMock()
        env._target_bay = {"x": 0.0, "y": 0.0, "yaw": 0.0, "width": 2.5, "depth": 5.0}

        obs = env._get_state()

        np.testing.assert_allclose(
            obs[1 : 1 + COVARIANCE_FEATURES_DIM],
            raw,
            rtol=1e-5,
            err_msg="Covariance features must be written verbatim to obs[1:4]",
        )

    def test_zero_uncertainty_stays_zero(self) -> None:
        """
        @brief Zero uncertainty should remain zero in the observation.
        """
        from unittest.mock import MagicMock

        from uncertainty_rl.envs.sim.carla_parking import CARLAParkingEnv
        from uncertainty_rl.utils.constants import COVARIANCE_FEATURES_DIM

        env = CARLAParkingEnv(max_steps=5, include_covariance=True)

        zero_cov = np.zeros(COVARIANCE_FEATURES_DIM, dtype=np.float32)
        # _read_ekf_state is bound at construction; override it directly.
        env._read_ekf_state = lambda: (None, zero_cov.copy())

        mock_vehicle = MagicMock()
        mock_vehicle.get_transform.return_value = MagicMock(
            location=MagicMock(x=0.0, y=0.0),
            rotation=MagicMock(yaw=0.0),
        )
        mock_vehicle.get_velocity.return_value = MagicMock(x=0.0, y=0.0)
        mock_vehicle.get_angular_velocity.return_value = MagicMock(z=0.0)
        env.vehicle = mock_vehicle
        env.world = MagicMock()
        env._target_bay = {"x": 0.0, "y": 0.0, "yaw": 0.0, "width": 2.5, "depth": 5.0}

        obs = env._get_state()
        np.testing.assert_array_equal(
            obs[1 : 1 + COVARIANCE_FEATURES_DIM],
            np.zeros(COVARIANCE_FEATURES_DIM),
        )


# ---------------------------------------------------------------------------
# zone_bbox
# ---------------------------------------------------------------------------


class TestZoneBbox:
    """
    @class TestZoneBbox
    @brief Tests for the zone_bbox() bounding-box converter.
    """

    def test_format_a_explicit_extents(self) -> None:
        """
        @brief Format A (x_min/x_max/y_min/y_max) is returned unchanged.
        """
        zone = {"x_min": -5.0, "x_max": 10.0, "y_min": -3.0, "y_max": 7.0}
        result = zone_bbox(zone)
        assert result == (-5.0, 10.0, -3.0, 7.0)

    def test_format_b_centre_half_extents(self) -> None:
        """
        @brief Format B (centre + half-extents) expands to correct bounds.
        """
        zone = {"centre_x": 4.0, "centre_y": 2.0, "half_width": 3.0, "half_height": 1.0}
        x_min, x_max, y_min, y_max = zone_bbox(zone)
        assert x_min == pytest.approx(1.0)
        assert x_max == pytest.approx(7.0)
        assert y_min == pytest.approx(1.0)
        assert y_max == pytest.approx(3.0)

    def test_returns_four_tuple(self) -> None:
        """
        @brief Return value is always a 4-tuple.
        """
        zone = {"x_min": 0.0, "x_max": 1.0, "y_min": 0.0, "y_max": 1.0}
        result = zone_bbox(zone)
        assert isinstance(result, tuple)
        assert len(result) == 4

    def test_all_values_are_floats(self) -> None:
        """
        @brief All four returned values are Python floats.
        """
        zone = {"x_min": 1, "x_max": 3, "y_min": 2, "y_max": 5}
        result = zone_bbox(zone)
        for val in result:
            assert isinstance(val, float)

    def test_format_a_takes_priority_when_x_min_present(self) -> None:
        """
        @brief If x_min is in the dict, Format A is used regardless of
               centre_x/half_width keys being present.
        """
        zone = {
            "x_min": 0.0,
            "x_max": 8.0,
            "y_min": 0.0,
            "y_max": 4.0,
            "centre_x": 999.0,
            "centre_y": 999.0,
            "half_width": 999.0,
            "half_height": 999.0,
        }
        result = zone_bbox(zone)
        assert result == (0.0, 8.0, 0.0, 4.0)

    def test_zero_extent_zone(self) -> None:
        """
        @brief Zero half-extents (Format B) produce a degenerate box where
               x_min == x_max and y_min == y_max.
        """
        zone = {"centre_x": 5.0, "centre_y": 3.0, "half_width": 0.0, "half_height": 0.0}
        x_min, x_max, y_min, y_max = zone_bbox(zone)
        assert x_min == pytest.approx(5.0)
        assert x_max == pytest.approx(5.0)
        assert y_min == pytest.approx(3.0)
        assert y_max == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# _compute_reward
# ---------------------------------------------------------------------------


def _make_env_for_reward() -> Any:
    """
    @brief Build a minimal CARLAParkingEnv with mocked CARLA objects suitable
           for exercising _compute_reward() without a real CARLA server.
    @return Configured env instance with mock vehicle and sensor manager.
    """
    from uncertainty_rl.envs.sim.carla_parking import CARLAParkingEnv

    env = CARLAParkingEnv(max_steps=100)

    # Mock sensor manager - no collision by default
    mock_sm = MagicMock()
    mock_sm.consume_collision.return_value = (False, False)
    env._sensor_manager = mock_sm

    # Default target bay at origin
    env._target_bay = {"x": 0.0, "y": 0.0, "yaw": 0.0}
    env._prev_distance = 5.0

    # Layout with generous corners so OOB check doesn't fire unless intended
    env._current_layout = {
        "corners": [
            {"x": -50.0, "y": -50.0},
            {"x": 50.0, "y": -50.0},
            {"x": 50.0, "y": 50.0},
            {"x": -50.0, "y": 50.0},
        ]
    }
    return env


def _set_vehicle(
    env: Any, x: float, y: float, yaw_deg: float, vx: float = 0.0, vy: float = 0.0
) -> None:
    """
    @brief Attach a mock CARLA vehicle to env with the given position and velocity.
    @param env: CARLAParkingEnv instance.
    @param x: Vehicle x position (metres).
    @param y: Vehicle y position (metres).
    @param yaw_deg: Vehicle heading (degrees, CARLA convention).
    @param vx: Velocity x component (m/s).
    @param vy: Velocity y component (m/s).
    """
    mock_vehicle = MagicMock()
    mock_vehicle.get_transform.return_value = MagicMock(
        location=MagicMock(x=x, y=y),
        rotation=MagicMock(yaw=yaw_deg),
    )
    mock_vehicle.get_velocity.return_value = MagicMock(x=vx, y=vy)
    env.vehicle = mock_vehicle


class TestComputeReward:
    """
    @class TestComputeReward
    @brief Tests for CARLAParkingEnv._compute_reward().
    """

    def test_vehicle_none_returns_zero_no_termination(self) -> None:
        """
        @brief When vehicle is None (CARLA not connected), reward is 0 and not
               terminated.
        """
        from uncertainty_rl.envs.sim.carla_parking import CARLAParkingEnv

        env = CARLAParkingEnv(max_steps=5)
        env.vehicle = None
        reward, terminated, success, diag = env._compute_reward()
        assert reward == 0.0
        assert terminated is False
        assert success is False

    def test_collision_ego_fault_returns_minus_fifteen_and_terminates(self) -> None:
        """
        @brief Ego-fault collision -> reward = -15, terminated = True, success = False.
        """
        env = _make_env_for_reward()
        _set_vehicle(env, x=5.0, y=5.0, yaw_deg=0.0)
        env._sensor_manager.consume_collision.return_value = (True, True)

        reward, terminated, success, diag = env._compute_reward()

        assert reward == pytest.approx(-15.0)
        assert terminated is True
        assert success is False

    def test_collision_non_ego_fault_returns_minus_five(self) -> None:
        """
        @brief Non-ego-fault collision -> reward = -5.0, terminated = True, success = False.
        """
        env = _make_env_for_reward()
        _set_vehicle(env, x=5.0, y=5.0, yaw_deg=0.0)
        env._sensor_manager.consume_collision.return_value = (True, False)

        reward, terminated, success, diag = env._compute_reward()

        assert reward == pytest.approx(-5.0)
        assert terminated is True
        assert success is False

    def test_success_returns_plus_fifteen_after_dwell(self) -> None:
        """
        @brief Success requires all thresholds to hold for success_dwell_steps
               consecutive steps. Before the dwell is complete, the episode
               does not terminate.
        """
        from uncertainty_rl.utils.constants import (
            SUCCESS_THRESHOLD_POSITION,
            SUCCESS_THRESHOLD_VELOCITY,
        )

        dwell = 3
        env = _make_env_for_reward()
        env._success_dwell_steps = dwell
        _set_vehicle(
            env,
            x=SUCCESS_THRESHOLD_POSITION * 0.5,
            y=0.0,
            yaw_deg=0.0,
            vx=SUCCESS_THRESHOLD_VELOCITY * 0.5,
        )
        env._target_bay = {"x": 0.0, "y": 0.0, "yaw": 0.0}

        # Steps 1 and 2: in-bay but dwell not yet satisfied
        for step in range(1, dwell):
            reward, terminated, success, diag = env._compute_reward()
            assert terminated is False, f"should not terminate on step {step}"
            assert success is False

        # Final dwell step: success fires
        reward, terminated, success, diag = env._compute_reward()
        assert reward == pytest.approx(15.0)
        assert terminated is True
        assert success is True

    def test_drive_through_does_not_count_as_success(self) -> None:
        """
        @brief A single step inside the bay thresholds (drive-through) must not
               trigger success. The dwell counter resets when the vehicle leaves.
        """
        from uncertainty_rl.utils.constants import (
            SUCCESS_THRESHOLD_POSITION,
            SUCCESS_THRESHOLD_VELOCITY,
        )

        env = _make_env_for_reward()
        env._success_dwell_steps = 5
        env._target_bay = {"x": 0.0, "y": 0.0, "yaw": 0.0}

        # One step inside thresholds (simulates fast drive-through)
        _set_vehicle(
            env,
            x=SUCCESS_THRESHOLD_POSITION * 0.5,
            y=0.0,
            yaw_deg=0.0,
            vx=SUCCESS_THRESHOLD_VELOCITY * 0.5,
        )
        reward, terminated, success, _ = env._compute_reward()
        assert terminated is False
        assert success is False
        assert env._success_counter == 1

        # Vehicle leaves the bay - counter resets
        _set_vehicle(env, x=5.0, y=0.0, yaw_deg=0.0, vx=2.0)
        reward, terminated, success, _ = env._compute_reward()
        assert terminated is False
        assert success is False
        assert env._success_counter == 0

    def test_progress_reward_positive_when_closing_in(self) -> None:
        """
        @brief Moving toward target gives positive progress minus the time penalty.
        """
        env = _make_env_for_reward()
        env._prev_distance = 10.0
        env._target_bay = {"x": 0.0, "y": 0.0, "yaw": 0.0}
        # Current distance is ~5m (vehicle at (5,0))
        _set_vehicle(env, x=5.0, y=0.0, yaw_deg=0.0)

        reward, terminated, success, diag = env._compute_reward()

        # progress = (10 - 5) / 20 = 0.25; reward = 0.25 - 0.01 = 0.24
        assert reward > 0.0
        assert terminated is False
        assert success is False

    def test_progress_reward_negative_when_moving_away(self) -> None:
        """
        @brief Moving away from target gives a negative reward.
        """
        env = _make_env_for_reward()
        env._prev_distance = 2.0
        env._target_bay = {"x": 0.0, "y": 0.0, "yaw": 0.0}
        # Current distance ~10m (vehicle moved away)
        _set_vehicle(env, x=10.0, y=0.0, yaw_deg=0.0)

        reward, terminated, success, diag = env._compute_reward()

        assert reward < 0.0
        assert terminated is False

    def test_time_penalty_always_applied(self) -> None:
        """
        @brief Even when making zero progress, reward includes the -0.01 time penalty.
        """
        env = _make_env_for_reward()
        dist = 5.0
        env._prev_distance = dist
        env._target_bay = {"x": 0.0, "y": 0.0, "yaw": 0.0}
        # Vehicle stays at exactly the same distance (no progress)
        _set_vehicle(env, x=dist, y=0.0, yaw_deg=0.0)

        reward, terminated, success, diag = env._compute_reward()

        # progress = 0, so reward = 0 - 0.01 = -0.01
        assert reward == pytest.approx(-0.01, abs=1e-4)

    def test_prev_distance_updated_after_step(self) -> None:
        """
        @brief _prev_distance is updated to the current position error after each call.
        """
        env = _make_env_for_reward()
        env._prev_distance = 10.0
        env._target_bay = {"x": 0.0, "y": 0.0, "yaw": 0.0}
        _set_vehicle(env, x=3.0, y=4.0, yaw_deg=0.0)  # distance = 5.0

        env._compute_reward()

        assert env._prev_distance == pytest.approx(5.0)

    def test_yaw_180_offset_not_valid_no_reverse(self) -> None:
        """
        @brief With no reverse gear, a 180-deg yaw offset is a full orientation
        error, not a valid nose-out. It must not trigger success even when the
        position and speed thresholds are met.
        """
        from uncertainty_rl.utils.constants import (
            SUCCESS_THRESHOLD_POSITION,
            SUCCESS_THRESHOLD_VELOCITY,
        )

        env = _make_env_for_reward()
        env._target_bay = {"x": 0.0, "y": 0.0, "yaw": 0.0}

        # Vehicle yaw = 180 deg: opposite to the bay's target yaw.
        _set_vehicle(
            env,
            x=SUCCESS_THRESHOLD_POSITION * 0.5,
            y=0.0,
            yaw_deg=180.0,
            vx=SUCCESS_THRESHOLD_VELOCITY * 0.5,
        )
        reward, terminated, success, diag = env._compute_reward()

        assert success is False, "180-deg yaw offset must not count as success"

    def test_diag_keys_present(self) -> None:
        """
        @brief diag dict must contain all five expected keys on every code path.
        """
        env = _make_env_for_reward()
        env._target_bay = {"x": 0.0, "y": 0.0, "yaw": 0.0}
        _set_vehicle(env, x=5.0, y=0.0, yaw_deg=0.0)

        _, _, _, diag = env._compute_reward()

        for key in (
            "pos_error",
            "orientation_error",
            "speed",
            "collision",
            "progress_reward",
        ):
            assert key in diag, f"Missing diag key: {key}"

    def test_diag_pos_error_matches_distance(self) -> None:
        """
        @brief diag['pos_error'] equals the Euclidean distance to the target.
        """
        env = _make_env_for_reward()
        env._target_bay = {"x": 0.0, "y": 0.0, "yaw": 0.0}
        _set_vehicle(env, x=3.0, y=4.0, yaw_deg=0.0)  # distance = 5.0

        _, _, _, diag = env._compute_reward()

        assert diag["pos_error"] == pytest.approx(5.0, abs=1e-6)

    def test_diag_collision_flag_set_on_collision(self) -> None:
        """
        @brief diag['collision'] == 1.0 when a collision is detected.
        """
        env = _make_env_for_reward()
        _set_vehicle(env, x=5.0, y=5.0, yaw_deg=0.0)
        env._sensor_manager.consume_collision.return_value = (True, True)

        _, _, _, diag = env._compute_reward()

        assert diag["collision"] == pytest.approx(1.0)

    def test_diag_collision_zero_on_normal_step(self) -> None:
        """
        @brief diag['collision'] == 0.0 when no collision occurred.
        """
        env = _make_env_for_reward()
        _set_vehicle(env, x=5.0, y=0.0, yaw_deg=0.0)

        _, _, _, diag = env._compute_reward()

        assert diag["collision"] == pytest.approx(0.0)

    def test_diag_progress_reward_positive_when_closing(self) -> None:
        """
        @brief diag['progress_reward'] is positive when the vehicle closes on target.
        """
        env = _make_env_for_reward()
        env._prev_distance = 10.0
        env._target_bay = {"x": 0.0, "y": 0.0, "yaw": 0.0}
        _set_vehicle(env, x=5.0, y=0.0, yaw_deg=0.0)

        _, _, _, diag = env._compute_reward()

        assert diag["progress_reward"] > 0.0

    def test_diag_vehicle_none_returns_zeros(self) -> None:
        """
        @brief When vehicle is None, diag is all zeros (no crash on missing vehicle).
        """
        from uncertainty_rl.envs.sim.carla_parking import CARLAParkingEnv

        env = CARLAParkingEnv(max_steps=5)
        env.vehicle = None
        _, _, _, diag = env._compute_reward()

        assert all(v == 0.0 for v in diag.values())


# ---------------------------------------------------------------------------
# Info dict keys (step())
# ---------------------------------------------------------------------------


class TestStepInfoDict:
    """
    @class TestStepInfoDict
    @brief Tests that step() info dict contains all expected keys with correct types.

    Uses vehicle=None (no CARLA connection) - step() still builds the full info
    dict and returns sensible zero-values when the vehicle is absent.
    """

    def test_info_contains_required_keys(self) -> None:
        """
        @brief info dict must have steps, success, collision, timeout, floor_plan,
               pos_error, orientation_error, speed, and progress_reward.
        """
        from uncertainty_rl.envs.sim.carla_parking import CARLAParkingEnv

        env = CARLAParkingEnv(max_steps=5)
        # vehicle=None: step() skips CARLA calls but still builds the full info dict.
        env.vehicle = None
        _, _, _, _, info = env.step(env.action_space.sample())

        for key in (
            "steps",
            "success",
            "collision",
            "timeout",
            "floor_plan",
            "pos_error",
            "orientation_error",
            "speed",
            "progress_reward",
        ):
            assert key in info, f"Missing info key: {key}"

    def test_info_timeout_true_at_max_steps(self) -> None:
        """
        @brief info['timeout'] is True when the episode is truncated by max_steps.
        """
        from uncertainty_rl.envs.sim.carla_parking import CARLAParkingEnv

        env = CARLAParkingEnv(max_steps=1)
        env.vehicle = None
        _, _, _, truncated, info = env.step(env.action_space.sample())

        assert truncated is True
        assert info["timeout"] is True

    def test_info_pos_error_is_nonnegative(self) -> None:
        """
        @brief pos_error is always >= 0 (it is a Euclidean distance).
        """
        env = _make_env_for_reward()
        _set_vehicle(env, x=3.0, y=4.0, yaw_deg=0.0)
        _, _, _, _, info = env.step(env.action_space.sample())

        assert info["pos_error"] >= 0.0

    def test_info_speed_is_nonnegative(self) -> None:
        """
        @brief speed is always >= 0 (it is a scalar magnitude).
        """
        env = _make_env_for_reward()
        _set_vehicle(env, x=0.0, y=0.0, yaw_deg=0.0, vx=-2.0, vy=-1.0)
        _, _, _, _, info = env.step(env.action_space.sample())

        assert info["speed"] >= 0.0


# ---------------------------------------------------------------------------
# Pure geometry: point_in_polygon
# ---------------------------------------------------------------------------


class TestPointInPolygon:
    """
    @class TestPointInPolygon
    @brief Tests for the ray-casting point-in-polygon check.
    """

    def test_centre_of_unit_square_is_inside(self) -> None:
        """
        @brief Centre of a unit square must be inside.
        """
        corners = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
        assert point_in_polygon(0.5, 0.5, corners) is True

    def test_point_outside_square_is_not_inside(self) -> None:
        """
        @brief Point clearly outside the square must return False.
        """
        corners = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
        assert point_in_polygon(2.0, 2.0, corners) is False

    def test_point_inside_triangle(self) -> None:
        """
        @brief Point inside a triangle must return True.
        """
        corners = [(0.0, 0.0), (4.0, 0.0), (2.0, 4.0)]
        assert point_in_polygon(2.0, 1.0, corners) is True

    def test_point_outside_triangle(self) -> None:
        """
        @brief Point outside a triangle must return False.
        """
        corners = [(0.0, 0.0), (4.0, 0.0), (2.0, 4.0)]
        assert point_in_polygon(3.5, 3.5, corners) is False

    def test_negative_coordinates(self) -> None:
        """
        @brief Polygon with negative coordinates must work correctly.
        """
        corners = [(-2.0, -2.0), (2.0, -2.0), (2.0, 2.0), (-2.0, 2.0)]
        assert point_in_polygon(0.0, 0.0, corners) is True
        assert point_in_polygon(3.0, 0.0, corners) is False


# ---------------------------------------------------------------------------
# Pure geometry: yaw_from_quaternion
# ---------------------------------------------------------------------------


class TestYawFromQuaternion:
    """
    @class TestYawFromQuaternion
    @brief Tests for quaternion -> yaw extraction.
    """

    def test_identity_quaternion_gives_zero_yaw(self) -> None:
        """
        @brief Identity quaternion (0, 0, 0, 1) must give yaw = 0.
        """
        assert yaw_from_quaternion(0.0, 0.0, 0.0, 1.0) == pytest.approx(0.0)

    def test_90_deg_yaw(self) -> None:
        """
        @brief Quaternion for 90-deg rotation about z gives yaw = pi/2.
        """
        # q = (0, 0, sin(pi/4), cos(pi/4))
        s = math.sin(math.pi / 4)
        c = math.cos(math.pi / 4)
        assert yaw_from_quaternion(0.0, 0.0, s, c) == pytest.approx(
            math.pi / 2, abs=1e-6
        )

    def test_minus_90_deg_yaw(self) -> None:
        """
        @brief Quaternion for -90-deg rotation gives yaw = -pi/2.
        """
        s = math.sin(-math.pi / 4)
        c = math.cos(-math.pi / 4)
        assert yaw_from_quaternion(0.0, 0.0, s, c) == pytest.approx(
            -math.pi / 2, abs=1e-6
        )

    def test_output_in_minus_pi_to_pi(self) -> None:
        """
        @brief yaw_from_quaternion must always return a value in [-pi, pi].
        """
        for angle in np.linspace(-math.pi, math.pi, 20):
            s = math.sin(angle / 2)
            c = math.cos(angle / 2)
            yaw = yaw_from_quaternion(0.0, 0.0, s, c)
            assert -math.pi <= yaw <= math.pi


# ---------------------------------------------------------------------------
# Pure geometry: wrap_angle_symmetric
# ---------------------------------------------------------------------------


class TestWrapAngleSymmetric:
    """
    @class TestWrapAngleSymmetric
    @brief Tests for the 180-deg symmetric angle wrap.
    """

    def test_zero_returns_zero(self) -> None:
        """
        @brief wrap_angle_symmetric(0) must be 0.
        """
        assert wrap_angle_symmetric(0.0) == pytest.approx(0.0)

    def test_pi_maps_to_zero(self) -> None:
        """
        @brief pi and -pi are equivalent to 0 under parking symmetry.
        """
        # Both pi and -pi should map to 0 (symmetric heading error)
        assert abs(wrap_angle_symmetric(math.pi)) < math.pi / 2

    def test_small_positive_returns_itself(self) -> None:
        """
        @brief Small positive angle < pi/2 must return itself (no flip needed).
        """
        angle = 0.3
        result = wrap_angle_symmetric(angle)
        assert result == pytest.approx(angle, abs=1e-6)

    def test_result_magnitude_le_pi_over_2(self) -> None:
        """
        @brief Result magnitude must always be <= pi/2 (smaller of two orientations).
        """
        for angle in np.linspace(-math.pi, math.pi, 50):
            result = wrap_angle_symmetric(angle)
            assert abs(result) <= math.pi / 2 + 1e-9

    def test_large_angle_flipped(self) -> None:
        """
        @brief An angle of 2pi/3 should flip to -pi/3 (smaller absolute value).
        """
        angle = 2 * math.pi / 3
        result = wrap_angle_symmetric(angle)
        assert abs(result) == pytest.approx(math.pi / 3, abs=1e-6)


# ---------------------------------------------------------------------------
# _parking_core: extract_obstacle_features
# ---------------------------------------------------------------------------


class TestExtractObstacleFeatures:
    """
    @class TestExtractObstacleFeatures
    @brief Tests for hemispheric LiDAR clearance feature extraction.
    """

    def _empty_out(self) -> np.ndarray:
        return np.zeros(OBSTACLE_FEATURES_DIM, dtype=np.float32)

    def test_none_scan_returns_zeros(self) -> None:
        """
        @brief None scan must leave the output buffer all-zero.
        """
        out = self._empty_out()
        result = extract_obstacle_features(None, out)
        np.testing.assert_array_equal(result, np.zeros(OBSTACLE_FEATURES_DIM))

    def test_empty_scan_returns_zeros(self) -> None:
        """
        @brief Empty scan array must leave the output buffer all-zero.
        """
        out = self._empty_out()
        result = extract_obstacle_features(np.empty((0, 2), dtype=np.float32), out)
        np.testing.assert_array_equal(result, np.zeros(OBSTACLE_FEATURES_DIM))

    def test_forward_point_populates_forward_dist(self) -> None:
        """
        @brief A point directly ahead must populate forward_dist (index 4).
        """
        scan = np.array([[3.0, 0.0]], dtype=np.float32)  # x=3, y=0 -> bearing=0
        out = self._empty_out()
        result = extract_obstacle_features(scan, out)
        assert result[4] == pytest.approx(3.0, abs=1e-3)

    def test_left_point_populates_left_dist(self) -> None:
        """
        @brief A point to the left (bearing > 15 deg) must populate left_dist (index 0).
        """
        scan = np.array([[2.0, 2.0]], dtype=np.float32)  # bearing ~45 deg
        out = self._empty_out()
        result = extract_obstacle_features(scan, out)
        expected_dist = math.sqrt(2.0**2 + 2.0**2)
        assert result[0] == pytest.approx(expected_dist, rel=1e-3)

    def test_right_point_populates_right_dist(self) -> None:
        """
        @brief A point to the right (bearing < -15 deg) must populate right_dist (index 2).
        """
        scan = np.array([[2.0, -2.0]], dtype=np.float32)  # bearing ~-45 deg
        out = self._empty_out()
        result = extract_obstacle_features(scan, out)
        expected_dist = math.sqrt(2.0**2 + 2.0**2)
        assert result[2] == pytest.approx(expected_dist, rel=1e-3)

    def test_self_return_filtered_out(self) -> None:
        """
        @brief Points closer than 1 m must be discarded as self-returns.
        """
        scan = np.array([[0.3, 0.0]], dtype=np.float32)
        out = self._empty_out()
        result = extract_obstacle_features(scan, out)
        np.testing.assert_array_equal(result, np.zeros(OBSTACLE_FEATURES_DIM))

    def test_rear_hemisphere_filtered_out(self) -> None:
        """
        @brief Points with x <= 0 (rear hemisphere) must be discarded.
        """
        scan = np.array([[-3.0, 0.0]], dtype=np.float32)
        out = self._empty_out()
        result = extract_obstacle_features(scan, out)
        np.testing.assert_array_equal(result, np.zeros(OBSTACLE_FEATURES_DIM))

    def test_nearest_point_selected_per_sector(self) -> None:
        """
        @brief When multiple points are in the same sector, the nearest is chosen.
        """
        scan = np.array(
            [
                [5.0, 0.0],  # forward, dist=5
                [2.0, 0.0],  # forward, dist=2 (nearer)
            ],
            dtype=np.float32,
        )
        out = self._empty_out()
        result = extract_obstacle_features(scan, out)
        assert result[4] == pytest.approx(2.0, abs=1e-3)


# ---------------------------------------------------------------------------
# _parking_core: build_observation
# ---------------------------------------------------------------------------


class TestBuildObservation:
    """
    @class TestBuildObservation
    @brief Tests for observation vector construction.
    """

    def _make_target(self, x: float = 5.0, y: float = 0.0, yaw: float = 0.0) -> dict:
        return {"x": x, "y": y, "yaw": yaw}

    def _ekf_pose(
        self, x: float = 0.0, y: float = 0.0, yaw: float = 0.0, vyaw: float = 0.1
    ) -> np.ndarray:
        return np.array([x, y, yaw, vyaw], dtype=np.float32)

    def test_full_obs_has_correct_dim(self) -> None:
        """
        @brief Full obs (cov + obstacles) must equal TOTAL_OBS_DIM.
        """
        dim = compute_obs_dim(include_covariance=True, include_obstacle_obs=True)
        buf = np.zeros(dim, dtype=np.float32)
        obs = build_observation(
            ekf_pose=self._ekf_pose(),
            uncertainty=np.zeros(COVARIANCE_FEATURES_DIM, dtype=np.float32),
            target_bay=self._make_target(),
            obstacle_features=np.zeros(OBSTACLE_FEATURES_DIM, dtype=np.float32),
            include_covariance=True,
            include_obstacle_obs=True,
            obs_buffer=buf,
        )
        assert obs.shape == (TOTAL_OBS_DIM,)

    def test_no_covariance_obs_has_correct_dim(self) -> None:
        """
        @brief Without covariance, obs dim = VEHICLE_STATE_DIM + TARGET_POSE_DIM + OBSTACLE_FEATURES_DIM.
        """
        dim = compute_obs_dim(include_covariance=False, include_obstacle_obs=True)
        buf = np.zeros(dim, dtype=np.float32)
        obs = build_observation(
            ekf_pose=self._ekf_pose(),
            uncertainty=None,
            target_bay=self._make_target(),
            obstacle_features=np.zeros(OBSTACLE_FEATURES_DIM, dtype=np.float32),
            include_covariance=False,
            include_obstacle_obs=True,
            obs_buffer=buf,
        )
        expected = VEHICLE_STATE_DIM + TARGET_POSE_DIM + OBSTACLE_FEATURES_DIM
        assert obs.shape == (expected,)

    def test_none_ekf_pose_gives_all_zeros(self) -> None:
        """
        @brief None ekf_pose must produce an observation of all zeros.
        """
        dim = compute_obs_dim(include_covariance=True, include_obstacle_obs=True)
        buf = np.zeros(dim, dtype=np.float32)
        obs = build_observation(
            ekf_pose=None,
            uncertainty=None,
            target_bay=self._make_target(),
            obstacle_features=np.zeros(OBSTACLE_FEATURES_DIM, dtype=np.float32),
            include_covariance=True,
            include_obstacle_obs=True,
            obs_buffer=buf,
        )
        np.testing.assert_array_equal(obs, np.zeros(dim))

    def test_uncertainty_written_to_covariance_indices(self) -> None:
        """
        @brief Uncertainty values must appear at indices 1-3 when include_covariance=True.
        """
        dim = compute_obs_dim(include_covariance=True, include_obstacle_obs=False)
        buf = np.zeros(dim, dtype=np.float32)
        unc = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        obs = build_observation(
            ekf_pose=self._ekf_pose(),
            uncertainty=unc,
            target_bay=self._make_target(),
            obstacle_features=np.zeros(OBSTACLE_FEATURES_DIM, dtype=np.float32),
            include_covariance=True,
            include_obstacle_obs=False,
            obs_buffer=buf,
        )
        np.testing.assert_allclose(obs[1 : 1 + COVARIANCE_FEATURES_DIM], unc)

    def test_returns_copy_not_buffer(self) -> None:
        """
        @brief build_observation must return a copy; mutating result must not
               affect subsequent calls.
        """
        dim = compute_obs_dim(include_covariance=True, include_obstacle_obs=True)
        buf = np.zeros(dim, dtype=np.float32)
        obs = build_observation(
            ekf_pose=self._ekf_pose(),
            uncertainty=np.zeros(COVARIANCE_FEATURES_DIM, dtype=np.float32),
            target_bay=self._make_target(),
            obstacle_features=np.zeros(OBSTACLE_FEATURES_DIM, dtype=np.float32),
            include_covariance=True,
            include_obstacle_obs=True,
            obs_buffer=buf,
        )
        obs[:] = 999.0
        assert buf[0] != 999.0


# ---------------------------------------------------------------------------
# _parking_core: load_floor_plan
# ---------------------------------------------------------------------------


class TestLoadFloorPlan:
    """
    @class TestLoadFloorPlan
    @brief Tests for floor plan selection and caching logic.
    """

    def _write_layout(self, path: Path) -> None:
        import yaml

        layout = {"bays": [{"id": "bay_01", "x": 0.0, "y": 0.0, "yaw": 0.0}]}
        with open(path, "w") as f:
            yaml.dump(layout, f)

    def test_loads_eligible_plan(self) -> None:
        """
        @brief load_floor_plan must return a valid (name, layout) tuple.
        """
        with tempfile.TemporaryDirectory() as d:
            layout_file = Path(d) / "rectangle.yaml"
            self._write_layout(layout_file)
            config = {"rectangle": {"layout_file": str(layout_file), "ood": False}}
            cache: dict = {}
            name, layout = load_floor_plan(config, eval_mode=False, layout_cache=cache)
            assert name == "rectangle"
            assert "bays" in layout

    def test_raises_when_no_eligible_plans(self) -> None:
        """
        @brief RuntimeError when all plans are OOD and eval_mode=False.
        """
        config = {"ood_only": {"layout_file": "x.yaml", "ood": True}}
        with pytest.raises(RuntimeError):
            load_floor_plan(config, eval_mode=False, layout_cache={})

    def test_ood_plan_eligible_in_eval_mode(self) -> None:
        """
        @brief OOD plans must be eligible when eval_mode=True.
        """
        with tempfile.TemporaryDirectory() as d:
            layout_file = Path(d) / "irregular.yaml"
            self._write_layout(layout_file)
            config = {"irregular_a": {"layout_file": str(layout_file), "ood": True}}
            cache: dict = {}
            name, layout = load_floor_plan(config, eval_mode=True, layout_cache=cache)
            assert name == "irregular_a"

    def test_layout_cached_after_first_load(self) -> None:
        """
        @brief Second call with the same path must use the cache (no re-read).
        """
        with tempfile.TemporaryDirectory() as d:
            layout_file = Path(d) / "rect.yaml"
            self._write_layout(layout_file)
            config = {"rectangle": {"layout_file": str(layout_file), "ood": False}}
            cache: dict = {}
            load_floor_plan(config, eval_mode=False, layout_cache=cache)
            assert len(cache) == 1
            load_floor_plan(config, eval_mode=False, layout_cache=cache)
            assert len(cache) == 1  # still only one entry

    def test_raises_file_not_found(self) -> None:
        """
        @brief FileNotFoundError when the layout YAML does not exist.
        """
        config = {"missing": {"layout_file": "/no/such/file.yaml", "ood": False}}
        with pytest.raises(FileNotFoundError):
            load_floor_plan(config, eval_mode=False, layout_cache={})


# ---------------------------------------------------------------------------
# _parking_core: wait_for_ekf
# ---------------------------------------------------------------------------


class TestWaitForEkf:
    """
    @class TestWaitForEkf
    @brief Tests for the EKF readiness polling helper.
    """

    def test_returns_immediately_when_both_ready(self) -> None:
        """
        @brief wait_for_ekf must return without error when both inputs are ready.
        """
        wait_for_ekf(
            has_lidar=lambda: True,
            has_ekf=lambda: True,
            timeout=1.0,
            tick_interval=0.001,
        )

    def test_raises_on_timeout_when_lidar_missing(self) -> None:
        """
        @brief RuntimeError when LiDAR never becomes ready within timeout.
        """
        with pytest.raises(RuntimeError, match="LiDAR"):
            wait_for_ekf(
                has_lidar=lambda: False,
                has_ekf=lambda: True,
                timeout=0.05,
                tick_interval=0.01,
            )

    def test_raises_on_timeout_when_ekf_missing(self) -> None:
        """
        @brief RuntimeError when EKF never becomes ready within timeout.
        """
        with pytest.raises(RuntimeError, match="EKF"):
            wait_for_ekf(
                has_lidar=lambda: True,
                has_ekf=lambda: False,
                timeout=0.05,
                tick_interval=0.01,
            )

    def test_tick_fn_called_while_waiting(self) -> None:
        """
        @brief tick_fn must be invoked on each poll iteration while waiting.
        """
        tick_calls = []

        def tick() -> None:
            tick_calls.append(1)

        # Becomes ready after 2 ticks
        counter = [0]

        def has_ekf() -> bool:
            counter[0] += 1
            return counter[0] >= 3

        wait_for_ekf(
            has_lidar=lambda: True,
            has_ekf=has_ekf,
            timeout=1.0,
            tick_fn=tick,
            tick_interval=0.001,
        )
        assert len(tick_calls) >= 2
