"""
@file test_covariance_subscriber.py
@brief Unit tests for the file-based covariance subscriber.

Tests cover JSON file reading, seq-based staleness guard, cache invalidation,
get_latest_uncertainty(), get_latest_pose(), get_latest_state(), and has_data
-- all without requiring ROS 2 or rclpy.  The _CovarianceSubscriber is
constructed with _ROS2_AVAILABLE forced to False so no Node superclass
initialisation occurs.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_ekf_json(
    path: Path,
    x: float,
    y: float,
    yaw: float,
    vyaw: float,
    cov_flat: list,
    seq: int = 1,
) -> None:
    """
    @brief Write a valid ekf_state.json to the given path.
    @param path: Target file path.
    @param x: EKF x position.
    @param y: EKF y position.
    @param yaw: EKF yaw in radians.
    @param vyaw: Yaw rate.
    @param cov_flat: Flat 9-element 3x3 covariance list.
    @param seq: Monotonic write sequence counter (default 1).
    """
    data = {
        "seq": seq,
        "x": x,
        "y": y,
        "yaw": yaw,
        "vyaw": vyaw,
        "covariance": cov_flat,
    }
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(data))
    tmp.replace(path)


def _make_subscriber(ekf_path: Path):
    """
    @brief Construct a _CovarianceSubscriber with ROS 2 disabled, pointing at
           a custom ekf_state.json path.
    @param ekf_path: Path to use instead of the default ekf_state.json.
    @return Tuple of (subscriber, module).
    """
    import threading

    import uncertainty_rl.envs.covariance_subscriber as mod

    with (
        patch.object(mod, "_ROS2_AVAILABLE", False),
        patch.object(mod, "_EKF_STATE_PATH", ekf_path),
    ):
        sub = mod._CovarianceSubscriber.__new__(mod._CovarianceSubscriber)
        sub._lock = threading.Lock()
        sub._latest_uncertainty = None
        sub._latest_pose = None
        sub._valid_after_seq = 0
        sub._last_read_seq = 0
        # _read_file() now uses self._ekf_state_path (set in __init__).
        # Bypass __init__ sets it directly so tests remain self-contained.
        sub._ekf_state_path = ekf_path
    return sub, mod


def _patch_read_file_path(sub: object, mod: object, ekf_path: Path) -> None:
    """
    @brief Update the instance EKF path used by _read_file.
    """
    sub._ekf_state_path = ekf_path  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# _read_file: valid JSON
# ---------------------------------------------------------------------------


class TestReadFileValid:
    """
    @class TestReadFileValid
    @brief Tests for _read_file() with well-formed JSON files.
    """

    def test_returns_true_on_valid_file(self) -> None:
        """
        @brief _read_file() returns True when the file exists with valid content.
        """
        import uncertainty_rl.envs.covariance_subscriber as mod

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "ekf_state.json"
            cov = [0.01, 0.0, 0.0, 0.0, 0.01, 0.0, 0.0, 0.0, 0.005]
            _write_ekf_json(path, 1.0, 2.0, 0.5, 0.05, cov, seq=1)

            with (
                patch.object(mod, "_ROS2_AVAILABLE", False),
                patch.object(mod, "_EKF_STATE_PATH", path),
            ):
                sub, _ = _make_subscriber(path)
                _patch_read_file_path(sub, mod, path)
                result = sub._read_file()

        assert result is True

    def test_populates_uncertainty_array(self) -> None:
        """
        @brief _read_file() populates _latest_uncertainty with shape (9,).
        """
        import uncertainty_rl.envs.covariance_subscriber as mod

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "ekf_state.json"
            cov = [0.04, 0.0, 0.0, 0.0, 0.04, 0.0, 0.0, 0.0, 0.01]
            _write_ekf_json(path, 0.0, 0.0, 0.0, 0.0, cov, seq=1)

            with (
                patch.object(mod, "_EKF_STATE_PATH", path),
                patch.object(mod, "_ROS2_AVAILABLE", False),
            ):
                sub, _ = _make_subscriber(path)
                _patch_read_file_path(sub, mod, path)
                sub._read_file()
                assert sub._latest_uncertainty is not None
                assert sub._latest_uncertainty.shape == (9,)

    def test_populates_pose_array(self) -> None:
        """
        @brief _read_file() populates _latest_pose with shape (4,).
        """
        import uncertainty_rl.envs.covariance_subscriber as mod

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "ekf_state.json"
            cov = [0.01] * 9
            _write_ekf_json(path, 3.0, -1.5, 1.2, 0.3, cov, seq=1)

            with (
                patch.object(mod, "_EKF_STATE_PATH", path),
                patch.object(mod, "_ROS2_AVAILABLE", False),
            ):
                sub, _ = _make_subscriber(path)
                _patch_read_file_path(sub, mod, path)
                sub._read_file()
                assert sub._latest_pose is not None
                assert sub._latest_pose.shape == (4,)

    def test_pose_values_match_file(self) -> None:
        """
        @brief Pose values read from file match what was written.
        """
        import uncertainty_rl.envs.covariance_subscriber as mod

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "ekf_state.json"
            cov = [0.01] * 9
            _write_ekf_json(path, 5.0, -3.0, 0.78, 0.4, cov, seq=1)

            with (
                patch.object(mod, "_EKF_STATE_PATH", path),
                patch.object(mod, "_ROS2_AVAILABLE", False),
            ):
                sub, _ = _make_subscriber(path)
                _patch_read_file_path(sub, mod, path)
                sub._read_file()
                pose = sub._latest_pose
                assert pose is not None
                np.testing.assert_allclose(
                    pose, [5.0, -3.0, 0.78, 0.4], rtol=1e-5
                )

    def test_updates_last_read_seq(self) -> None:
        """
        @brief _read_file() updates _last_read_seq to match the file's seq.
        """
        import uncertainty_rl.envs.covariance_subscriber as mod

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "ekf_state.json"
            _write_ekf_json(path, 0.0, 0.0, 0.0, 0.0, [0.01] * 9, seq=42)

            with (
                patch.object(mod, "_EKF_STATE_PATH", path),
                patch.object(mod, "_ROS2_AVAILABLE", False),
            ):
                sub, _ = _make_subscriber(path)
                _patch_read_file_path(sub, mod, path)
                sub._read_file()

        assert sub._last_read_seq == 42


# ---------------------------------------------------------------------------
# _read_file: missing / malformed file
# ---------------------------------------------------------------------------


class TestReadFileMissing:
    """
    @class TestReadFileMissing
    @brief Tests for _read_file() when the file is absent or malformed.
    """

    def test_returns_false_when_file_missing(self) -> None:
        """
        @brief _read_file() returns False when ekf_state.json does not exist.
        """
        import uncertainty_rl.envs.covariance_subscriber as mod

        missing = Path("/tmp/does_not_exist_ekf_state_123456.json")
        with (
            patch.object(mod, "_EKF_STATE_PATH", missing),
            patch.object(mod, "_ROS2_AVAILABLE", False),
        ):
            sub, _ = _make_subscriber(missing)
            _patch_read_file_path(sub, mod, missing)
            result = sub._read_file()

        assert result is False

    def test_returns_false_on_invalid_json(self) -> None:
        """
        @brief _read_file() returns False on malformed JSON.
        """
        import uncertainty_rl.envs.covariance_subscriber as mod

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "ekf_state.json"
            path.write_text("{not valid json")

            with (
                patch.object(mod, "_EKF_STATE_PATH", path),
                patch.object(mod, "_ROS2_AVAILABLE", False),
            ):
                sub, _ = _make_subscriber(path)
                _patch_read_file_path(sub, mod, path)
                result = sub._read_file()

        assert result is False

    def test_returns_false_on_missing_key(self) -> None:
        """
        @brief _read_file() returns False when a required key is absent.
        """
        import uncertainty_rl.envs.covariance_subscriber as mod

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "ekf_state.json"
            # Missing 'covariance' key
            path.write_text(
                json.dumps(
                    {
                        "seq": 1,
                        "x": 1.0,
                        "y": 2.0,
                        "yaw": 0.0,
                        "vyaw": 0.0,
                    }
                )
            )

            with (
                patch.object(mod, "_EKF_STATE_PATH", path),
                patch.object(mod, "_ROS2_AVAILABLE", False),
            ):
                sub, _ = _make_subscriber(path)
                _patch_read_file_path(sub, mod, path)
                result = sub._read_file()

        assert result is False

    def test_leaves_cache_none_on_failure(self) -> None:
        """
        @brief Failed _read_file() does not alter the cached uncertainty.
        """
        import uncertainty_rl.envs.covariance_subscriber as mod

        missing = Path("/tmp/ekf_never_written.json")
        with (
            patch.object(mod, "_EKF_STATE_PATH", missing),
            patch.object(mod, "_ROS2_AVAILABLE", False),
        ):
            sub, _ = _make_subscriber(missing)
            _patch_read_file_path(sub, mod, missing)
            sub._read_file()

        assert sub._latest_uncertainty is None
        assert sub._latest_pose is None


# ---------------------------------------------------------------------------
# Seq-based staleness guard
# ---------------------------------------------------------------------------


class TestStalenessGuard:
    """
    @class TestStalenessGuard
    @brief Tests for seq-based staleness guard in _read_file().

    The guard rejects a file whose `seq` field is <= _valid_after_seq,
    preventing stale pre-reset data from leaking across episode boundaries.
    This is clock-skew-proof -- no file mtime comparison is performed.
    """

    def test_rejects_file_with_seq_equal_to_valid_after_seq(self) -> None:
        """
        @brief _read_file() rejects a file whose seq == _valid_after_seq.
        """
        import uncertainty_rl.envs.covariance_subscriber as mod

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "ekf_state.json"
            _write_ekf_json(path, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, [0.01] * 9, seq=5)

            with (
                patch.object(mod, "_EKF_STATE_PATH", path),
                patch.object(mod, "_ROS2_AVAILABLE", False),
            ):
                sub, _ = _make_subscriber(path)
                _patch_read_file_path(sub, mod, path)
                sub._valid_after_seq = 5  # same seq as the file
                result = sub._read_file()

        assert result is False
        assert sub._latest_uncertainty is None

    def test_rejects_file_with_seq_below_valid_after_seq(self) -> None:
        """
        @brief _read_file() rejects a file whose seq < _valid_after_seq.
        """
        import uncertainty_rl.envs.covariance_subscriber as mod

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "ekf_state.json"
            _write_ekf_json(path, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, [0.01] * 9, seq=3)

            with (
                patch.object(mod, "_EKF_STATE_PATH", path),
                patch.object(mod, "_ROS2_AVAILABLE", False),
            ):
                sub, _ = _make_subscriber(path)
                _patch_read_file_path(sub, mod, path)
                sub._valid_after_seq = 10  # file seq 3 < barrier 10
                result = sub._read_file()

        assert result is False

    def test_accepts_file_with_seq_above_valid_after_seq(self) -> None:
        """
        @brief _read_file() accepts a file whose seq > _valid_after_seq.
        """
        import uncertainty_rl.envs.covariance_subscriber as mod

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "ekf_state.json"
            _write_ekf_json(path, 1.0, 2.0, 0.3, 0.0, 0.0, 0.0, [0.01] * 9, seq=6)

            with (
                patch.object(mod, "_EKF_STATE_PATH", path),
                patch.object(mod, "_ROS2_AVAILABLE", False),
            ):
                sub, _ = _make_subscriber(path)
                _patch_read_file_path(sub, mod, path)
                sub._valid_after_seq = 5  # file seq 6 > barrier 5
                result = sub._read_file()

        assert result is True

    def test_legacy_file_without_seq_uses_zero_seq(self) -> None:
        """
        @brief A file without a seq field is treated as seq=0 and accepted
               when _valid_after_seq is also 0 (backwards compatibility).
        """
        import uncertainty_rl.envs.covariance_subscriber as mod

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "ekf_state.json"
            # No seq field -- old extractor format
            data = {
                "x": 1.0,
                "y": 2.0,
                "yaw": 0.3,
                "vyaw": 0.0,
                "covariance": [0.01] * 9,
            }
            path.write_text(json.dumps(data))

            with (
                patch.object(mod, "_EKF_STATE_PATH", path),
                patch.object(mod, "_ROS2_AVAILABLE", False),
            ):
                sub, _ = _make_subscriber(path)
                _patch_read_file_path(sub, mod, path)
                sub._valid_after_seq = 0  # default -- seq=0 file not accepted
                result = sub._read_file()

        # seq=0 (default) is NOT > barrier 0, so it is rejected
        assert result is False


# ---------------------------------------------------------------------------
# invalidate()
# ---------------------------------------------------------------------------


class TestInvalidate:
    """
    @class TestInvalidate
    @brief Tests for the invalidate() method.
    """

    def test_invalidate_clears_cached_uncertainty(self) -> None:
        """
        @brief invalidate() sets _latest_uncertainty to None.
        """
        import uncertainty_rl.envs.covariance_subscriber as mod

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "ekf_state.json"
            with patch.object(mod, "_ROS2_AVAILABLE", False):
                sub, _ = _make_subscriber(path)
                sub._latest_uncertainty = np.zeros(9)
                sub._latest_pose = np.zeros(6)
                sub.invalidate()

        assert sub._latest_uncertainty is None
        assert sub._latest_pose is None

    def test_invalidate_records_last_read_seq_as_barrier(self) -> None:
        """
        @brief invalidate() sets _valid_after_seq to the current _last_read_seq
               so any file with the same or older seq is rejected afterwards.
        """
        import uncertainty_rl.envs.covariance_subscriber as mod

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "ekf_state.json"
            with patch.object(mod, "_ROS2_AVAILABLE", False):
                sub, _ = _make_subscriber(path)
                sub._last_read_seq = 17
                sub.invalidate()

        assert sub._valid_after_seq == 17

    def test_invalidate_then_new_seq_accepted(self) -> None:
        """
        @brief After invalidate(), a file with a higher seq is accepted.
        """
        import uncertainty_rl.envs.covariance_subscriber as mod

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "ekf_state.json"

            with (
                patch.object(mod, "_EKF_STATE_PATH", path),
                patch.object(mod, "_ROS2_AVAILABLE", False),
            ):
                sub, _ = _make_subscriber(path)
                _patch_read_file_path(sub, mod, path)

                # Simulate: last episode ended at seq=10
                sub._last_read_seq = 10
                sub.invalidate()  # barrier = 10

                # New extractor write arrives with seq=11
                _write_ekf_json(path, 1.0, 2.0, 0.0, 0.0, 0.0, 0.0, [0.02] * 9, seq=11)
                result = sub._read_file()

        assert result is True
        assert sub._last_read_seq == 11

    def test_invalidate_then_same_seq_rejected(self) -> None:
        """
        @brief After invalidate(), a file with the same seq as the barrier is
               rejected (ensures strict greater-than comparison).
        """
        import uncertainty_rl.envs.covariance_subscriber as mod

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "ekf_state.json"

            with (
                patch.object(mod, "_EKF_STATE_PATH", path),
                patch.object(mod, "_ROS2_AVAILABLE", False),
            ):
                sub, _ = _make_subscriber(path)
                _patch_read_file_path(sub, mod, path)

                sub._last_read_seq = 10
                sub.invalidate()  # barrier = 10

                # Stale file still has seq=10
                _write_ekf_json(path, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, [0.01] * 9, seq=10)
                result = sub._read_file()

        assert result is False


# ---------------------------------------------------------------------------
# get_latest_uncertainty() and get_latest_pose()
# ---------------------------------------------------------------------------


class TestGetLatest:
    """
    @class TestGetLatest
    @brief Tests for the public get_latest_* accessors.
    """

    def test_get_latest_uncertainty_returns_copy(self) -> None:
        """
        @brief get_latest_uncertainty() returns a copy -- mutating it does not
               affect the cached value.
        """
        import uncertainty_rl.envs.covariance_subscriber as mod

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "ekf_state.json"
            cov = [0.04, 0.0, 0.0, 0.0, 0.04, 0.0, 0.0, 0.0, 0.01]
            _write_ekf_json(path, 0.0, 0.0, 0.0, 0.0, cov, seq=1)

            with (
                patch.object(mod, "_EKF_STATE_PATH", path),
                patch.object(mod, "_ROS2_AVAILABLE", False),
            ):
                sub, _ = _make_subscriber(path)
                _patch_read_file_path(sub, mod, path)
                unc = sub.get_latest_uncertainty()
                assert unc is not None
                original = unc.copy()
                unc[:] = 99.0  # mutate the returned array
                unc2 = sub.get_latest_uncertainty()
                assert unc2 is not None
                np.testing.assert_allclose(unc2, original, rtol=1e-5)

    def test_get_latest_uncertainty_none_when_no_file(self) -> None:
        """
        @brief get_latest_uncertainty() returns None when no file exists.
        """
        import uncertainty_rl.envs.covariance_subscriber as mod

        missing = Path("/tmp/ekf_no_file_999.json")
        with (
            patch.object(mod, "_EKF_STATE_PATH", missing),
            patch.object(mod, "_ROS2_AVAILABLE", False),
        ):
            sub, _ = _make_subscriber(missing)
            _patch_read_file_path(sub, mod, missing)
            result = sub.get_latest_uncertainty()

        assert result is None

    def test_get_latest_pose_shape(self) -> None:
        """
        @brief get_latest_pose() returns an array of shape (6,).
        """
        import uncertainty_rl.envs.covariance_subscriber as mod

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "ekf_state.json"
            cov = [0.01] * 9
            _write_ekf_json(path, 1.0, 2.0, 0.5, 1.0, 0.0, 0.1, cov, seq=1)

            with (
                patch.object(mod, "_EKF_STATE_PATH", path),
                patch.object(mod, "_ROS2_AVAILABLE", False),
            ):
                sub, _ = _make_subscriber(path)
                _patch_read_file_path(sub, mod, path)
                pose = sub.get_latest_pose()

        assert pose is not None
        assert pose.shape == (6,)

    def test_get_latest_pose_none_when_no_file(self) -> None:
        """
        @brief get_latest_pose() returns None when no file exists.
        """
        import uncertainty_rl.envs.covariance_subscriber as mod

        missing = Path("/tmp/ekf_no_file_pose_999.json")
        with (
            patch.object(mod, "_EKF_STATE_PATH", missing),
            patch.object(mod, "_ROS2_AVAILABLE", False),
        ):
            sub, _ = _make_subscriber(missing)
            _patch_read_file_path(sub, mod, missing)
            result = sub.get_latest_pose()

        assert result is None


# ---------------------------------------------------------------------------
# has_data property
# ---------------------------------------------------------------------------


class TestHasData:
    """
    @class TestHasData
    @brief Tests for the has_data property.
    """

    def test_has_data_true_when_file_readable(self) -> None:
        """
        @brief has_data is True after a valid file has been read.
        """
        import uncertainty_rl.envs.covariance_subscriber as mod

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "ekf_state.json"
            cov = [0.02] * 9
            _write_ekf_json(path, 0.0, 0.0, 0.0, 0.0, cov, seq=1)

            with (
                patch.object(mod, "_EKF_STATE_PATH", path),
                patch.object(mod, "_ROS2_AVAILABLE", False),
            ):
                sub, _ = _make_subscriber(path)
                _patch_read_file_path(sub, mod, path)
                result = sub.has_data

        assert result is True

    def test_has_data_false_when_file_missing(self) -> None:
        """
        @brief has_data is False when no file has been written yet.
        """
        import uncertainty_rl.envs.covariance_subscriber as mod

        missing = Path("/tmp/ekf_has_data_missing.json")
        with (
            patch.object(mod, "_EKF_STATE_PATH", missing),
            patch.object(mod, "_ROS2_AVAILABLE", False),
        ):
            sub, _ = _make_subscriber(missing)
            _patch_read_file_path(sub, mod, missing)
            result = sub.has_data

        assert result is False

    def test_has_data_false_after_invalidate(self) -> None:
        """
        @brief has_data is False immediately after invalidate() even if a file
               exists, because the cached data was cleared.
        """
        import uncertainty_rl.envs.covariance_subscriber as mod

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "ekf_state.json"
            cov = [0.02] * 9
            _write_ekf_json(path, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, cov, seq=5)

            with (
                patch.object(mod, "_EKF_STATE_PATH", path),
                patch.object(mod, "_ROS2_AVAILABLE", False),
            ):
                sub, _ = _make_subscriber(path)
                _patch_read_file_path(sub, mod, path)
                # First read succeeds and caches data
                assert sub.has_data is True
                # Invalidate sets barrier = last_read_seq = 5
                sub.invalidate()
                # File still has seq=5 which is not > barrier 5, so rejected
                result = sub.has_data

        assert result is False
