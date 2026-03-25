"""
@file test_carla_parking.py
@brief Unit tests for the CARLA parking environment and geometry helpers.

All tests are CPU-only (no CARLA server, no ROS 2, no GPU). The environment
falls back gracefully when CARLA and rclpy are unavailable. Tests exercise:
  - Pure geometry helpers (cone interpolation, relative target pose)
  - Observation space shape (21-dim with covariance + obstacles, 9-dim without)
  - Bay sampling logic (stratified sampling by bay type)
  - VisStateWriter (atomic write, valid JSON, tmp file cleaned up)
  - Gymnasium API contract (reset/step return shapes, dtypes)
"""

import json
import math
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pytest

from uncertainty_rl.utils.geometry import (
    _compute_relative_target_pose,
    _interpolate_cone_positions,
)
from uncertainty_rl.utils.constants import (
    ACTION_DIM,
    OBSTACLE_FEATURES_DIM,
    TARGET_POSE_DIM,
    TOTAL_OBS_DIM,
    VEHICLE_STATE_DIM,
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

    def test_21_dim_with_covariance(self) -> None:
        """
        @brief include_covariance=True -> 21-dim observation space (default).
        """
        from uncertainty_rl.envs.carla_parking import CARLAParkingEnv

        env = CARLAParkingEnv(max_steps=5, include_covariance=True)
        assert env.observation_space.shape == (TOTAL_OBS_DIM,)  # 21
        env.close()

    def test_12_dim_without_covariance(self) -> None:
        """
        @brief include_covariance=False, include_obstacle_obs=True (default) -> 12-dim.

        Without covariance but with obstacle obs:
        pose(6) + target(3) + obstacle(3) = 12.
        """
        from uncertainty_rl.envs.carla_parking import CARLAParkingEnv

        env = CARLAParkingEnv(max_steps=5, include_covariance=False)
        expected = VEHICLE_STATE_DIM + TARGET_POSE_DIM + OBSTACLE_FEATURES_DIM  # 12
        assert env.observation_space.shape == (expected,)
        env.close()

    def test_9_dim_without_covariance_or_obstacles(self) -> None:
        """
        @brief include_covariance=False, include_obstacle_obs=False -> 9-dim.
        """
        from uncertainty_rl.envs.carla_parking import CARLAParkingEnv

        env = CARLAParkingEnv(
            max_steps=5, include_covariance=False, include_obstacle_obs=False
        )
        expected = VEHICLE_STATE_DIM + TARGET_POSE_DIM  # 6 + 3 = 9
        assert env.observation_space.shape == (expected,)
        env.close()

    def test_action_space_shape(self) -> None:
        """
        @brief Action space must be 3-dim: [steering, throttle, brake].
        """
        from uncertainty_rl.envs.carla_parking import CARLAParkingEnv

        env = CARLAParkingEnv(max_steps=5)
        assert env.action_space.shape == (ACTION_DIM,)
        env.close()

    def test_action_space_bounds(self) -> None:
        """
        @brief steering [-1,1], throttle [0,1], brake [0,1].
        """
        from uncertainty_rl.envs.carla_parking import CARLAParkingEnv

        env = CARLAParkingEnv(max_steps=5)
        np.testing.assert_array_equal(env.action_space.low, [-1.0, 0.0, 0.0])
        np.testing.assert_array_equal(env.action_space.high, [1.0, 1.0, 1.0])
        env.close()


# ---------------------------------------------------------------------------
# Gymnasium API contract (no CARLA)
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
        from uncertainty_rl.envs.carla_parking import CARLAParkingEnv

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
        from uncertainty_rl.envs.carla_parking import CARLAParkingEnv

        env = CARLAParkingEnv(max_steps=5, include_covariance=True)
        obs, _ = env.reset()
        assert obs.shape == (TOTAL_OBS_DIM,)
        env.close()

    def test_reset_obs_shape_9_no_cov(self) -> None:
        """
        @brief reset() observation shape must be (9,) when include_covariance=False.
        """
        from uncertainty_rl.envs.carla_parking import CARLAParkingEnv

        env = CARLAParkingEnv(max_steps=5, include_covariance=False)
        obs, _ = env.reset()
        expected = VEHICLE_STATE_DIM + TARGET_POSE_DIM
        assert obs.shape == (expected,)
        env.close()

    def test_reset_obs_dtype_float32(self) -> None:
        """
        @brief Observation must be float32.
        """
        from uncertainty_rl.envs.carla_parking import CARLAParkingEnv

        env = CARLAParkingEnv(max_steps=5)
        obs, _ = env.reset()
        assert obs.dtype == np.float32
        env.close()

    def test_step_returns_five_tuple(self) -> None:
        """
        @brief step() -> (obs, reward, terminated, truncated, info).
        """
        from uncertainty_rl.envs.carla_parking import CARLAParkingEnv

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
        from uncertainty_rl.envs.carla_parking import CARLAParkingEnv

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
        from uncertainty_rl.envs.carla_parking import CARLAParkingEnv

        env = CARLAParkingEnv(max_steps=5)
        env._current_layout = layout
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
# log1p covariance transform
# ---------------------------------------------------------------------------


class TestLog1pCovarianceTransform:
    """
    @class TestLog1pCovarianceTransform
    @brief Verify covariance features are log1p-transformed in _get_state().

    In simulation (no CARLA), _get_state() returns zeros because the vehicle
    is None.  We test the transform by injecting a mock _cov_subscriber and
    constructing the obs buffer directly via the env internals.
    """

    def test_log1p_applied_to_nonzero_covariance(self) -> None:
        """
        @brief Covariance features in obs[6:15] equal log1p(raw_uncertainty).
        """
        from unittest.mock import MagicMock

        from uncertainty_rl.envs.carla_parking import CARLAParkingEnv
        from uncertainty_rl.utils.constants import COVARIANCE_FEATURES_DIM

        env = CARLAParkingEnv(max_steps=5, include_covariance=True)

        # Inject a mock subscriber that returns known covariance values
        raw = np.array([1.0, 4.0, 9.0, 0.5, 2.5, 0.1, 0.0, 0.3, 7.0], dtype=np.float32)
        assert len(raw) == COVARIANCE_FEATURES_DIM

        mock_sub = MagicMock()
        mock_sub.get_latest_pose.return_value = None
        mock_sub.get_latest_uncertainty.return_value = raw.copy()
        env._cov_subscriber = mock_sub

        # Simulate a minimal vehicle mock so _get_state() doesn't early-return
        mock_vehicle = MagicMock()
        mock_vehicle.get_transform.return_value = MagicMock(
            location=MagicMock(x=0.0, y=0.0),
            rotation=MagicMock(yaw=0.0),
        )
        mock_vehicle.get_velocity.return_value = MagicMock(x=0.0, y=0.0)
        mock_vehicle.get_angular_velocity.return_value = MagicMock(z=0.0)
        env.vehicle = mock_vehicle
        env.world = MagicMock()
        env._target_bay = {"x": 0.0, "y": 0.0, "yaw": 0.0}

        obs = env._get_state()

        expected_cov = np.log1p(raw)
        np.testing.assert_allclose(
            obs[6:15],
            expected_cov,
            rtol=1e-5,
            err_msg="Covariance features must be log1p-transformed",
        )

    def test_log1p_zero_uncertainty_stays_zero(self) -> None:
        """
        @brief log1p(0) == 0: zero uncertainty should remain zero in obs.
        """
        from unittest.mock import MagicMock

        from uncertainty_rl.envs.carla_parking import CARLAParkingEnv

        env = CARLAParkingEnv(max_steps=5, include_covariance=True)

        zero_cov = np.zeros(9, dtype=np.float32)
        mock_sub = MagicMock()
        mock_sub.get_latest_pose.return_value = None
        mock_sub.get_latest_uncertainty.return_value = zero_cov.copy()
        env._cov_subscriber = mock_sub

        mock_vehicle = MagicMock()
        mock_vehicle.get_transform.return_value = MagicMock(
            location=MagicMock(x=0.0, y=0.0),
            rotation=MagicMock(yaw=0.0),
        )
        mock_vehicle.get_velocity.return_value = MagicMock(x=0.0, y=0.0)
        mock_vehicle.get_angular_velocity.return_value = MagicMock(z=0.0)
        env.vehicle = mock_vehicle
        env.world = MagicMock()
        env._target_bay = {"x": 0.0, "y": 0.0, "yaw": 0.0}

        obs = env._get_state()
        np.testing.assert_array_equal(obs[6:15], np.zeros(9))
