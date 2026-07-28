"""
@file covariance_subscriber.py
@brief File-based covariance reader for EKF uncertainty features.

Reads the latest EKF state from a shared JSON file written by the
CovarianceExtractorNode in the ros2-bridge container. This avoids DDS
cross-distro serialisation issues between ROS 2 Humble (training container)
and Jazzy (ros2-bridge container).
"""

import json
import logging
import math
import os
import threading
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, cast

import numpy as np

from uncertainty_rl.utils.covariance_utils import extract_2d_covariance_features

# Cached mtime sentinel meaning "never read"
_MTIME_UNSET: int = -1

# Default shared file path
_EKF_STATE_PATH = Path("/workspace/outputs/ekf_state.json")

# File-based /set_pose signal
_INITIAL_POSE_PATH = Path("/workspace/outputs/initial_pose.json")

# File-based GNSS noise config signal
_EPISODE_CONFIG_PATH = Path("/workspace/outputs/episode_config.json")

logger = logging.getLogger(__name__)


class _CovarianceSubscriber:
    """
    @class _CovarianceSubscriber
    @brief Reads EKF state from a shared JSON file + signals /set_pose.

    The CovarianceExtractorNode (ros2-bridge, Jazzy) writes the latest EKF
    pose, velocity, and 3x3 covariance to a shared file. This class reads
    that file on demand - no DDS subscription needed.
    """

    # -----------------------------------------------------------------------
    # Construction
    # -----------------------------------------------------------------------

    def __init__(
        self,
        covariance_topic: str = "/odometry/filtered",
        node_name: str = "covariance_subscriber",
        ros2_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        @brief Initialise the covariance reader.
        @param covariance_topic: Unused - EKF state is read from the shared
               JSON file, not via DDS. Accepted for call-site compatibility.
        @param node_name: Unused - no rclpy Node. Accepted for compatibility.
        @param ros2_config: Optional ROS 2 config dict from train_config.yaml.
        """
        self._lock = threading.Lock()
        self._latest_uncertainty: Optional[np.ndarray] = None
        self._latest_pose: Optional[np.ndarray] = None
        # Sequence number of the last-seen write from the extractor node.
        # invalidate() records the current seq; _read_file() only accepts a
        # file whose `seq` field is strictly greater than _valid_after_seq.
        self._valid_after_seq: int = 0
        self._last_read_seq: int = 0
        # Monotonically increasing counter for initial_pose.json writes.
        self._initial_pose_seq: int = 0
        # Last observed mtime (nanoseconds) of the EKF state file.
        # Unchanged mtime means the file content has not changed - skip parse.
        self._last_mtime_ns: int = _MTIME_UNSET

        # Per-worker EKF state file path
        config = ros2_config or {}
        ekf_state_file: str = config.get(
            "ekf_state_file",
            os.environ.get("EKF_STATE_FILE", str(_EKF_STATE_PATH)),
        )
        self._ekf_state_path: Path = Path(ekf_state_file)

        # Initial pose file path
        self._initial_pose_path: Path = Path(
            config.get(
                "initial_pose_file",
                os.environ.get("INITIAL_POSE_FILE", str(_INITIAL_POSE_PATH)),
            )
        )
        self._initial_pose_tmp: Path = self._initial_pose_path.with_suffix(".json.tmp")

        self._episode_config_path: Path = Path(
            config.get(
                "episode_config_file",
                os.environ.get("EPISODE_CONFIG_FILE", str(_EPISODE_CONFIG_PATH)),
            )
        )
        self._episode_config_tmp: Path = self._episode_config_path.with_suffix(
            ".json.tmp"
        )
        # Monotonically increasing counter for episode_config.json writes.
        self._episode_config_seq: int = 0

        # Ensure output directory exists once at construction time.
        output_dir = self._ekf_state_path.parent
        output_dir.mkdir(parents=True, exist_ok=True)

        # Remove stale JSON files from previous sessions so the ros2-bridge
        # nodes do not pick up old state on startup.
        for stale in [
            self._ekf_state_path,
            self._initial_pose_path,
            self._initial_pose_tmp,
            self._episode_config_path,
            self._episode_config_tmp,
        ]:
            try:
                stale.unlink(missing_ok=True)
            except OSError:
                pass

        logger.info(
            "Covariance reader: ekf_file=%s, initial_pose_file=%s",
            self._ekf_state_path,
            self._initial_pose_path,
        )

    # -----------------------------------------------------------------------
    # EKF state read interface
    # -----------------------------------------------------------------------

    def invalidate(self) -> None:
        """
        @brief Mark cached data as stale at the current write sequence.

        Records the seq number from the last successfully read JSON write.
        After this call, _read_file() only accepts a file whose `seq` field
        is strictly greater than the recorded seq, guaranteeing the next
        reading is a genuinely post-reset write from the extractor node.

        Call this at episode reset before publish_initial_pose and before
        _wait_for_covariance.
        """
        with self._lock:
            # Barrier at the last-seen write seq.  Any file with seq <= this
            # value was written before the reset and will be rejected.
            self._valid_after_seq = self._last_read_seq
            self._latest_uncertainty = None
            self._latest_pose = None
            # Reset mtime cache so next _read_file() always re-parses the file
            # rather than seeing a match against pre-reset content.
            self._last_mtime_ns = _MTIME_UNSET

    def _read_file(self) -> bool:
        """
        @brief Read the latest EKF state from the shared JSON file.

        Only accepts the file if its `seq` field is strictly greater than
        the seq recorded at the last invalidate() call, preventing stale
        pre-reset data from being returned during the wait period at episode
        start.  This is clock-skew-proof: the seq is written by the extractor
        node and compared numerically, so Docker container clock differences
        cannot cause a stale read to pass the guard.

        Updates both _latest_pose and _latest_uncertainty atomically under
        the lock so callers never see a partially-updated state.

        @return True if fresh (post-invalidation) data was read successfully.
        """
        try:
            stat = self._ekf_state_path.stat()
        except FileNotFoundError:
            return False

        mtime_ns: int = stat.st_mtime_ns

        with self._lock:
            valid_after_seq = self._valid_after_seq
            last_mtime = self._last_mtime_ns
            cached_pose = self._latest_pose
            cached_seq = self._last_read_seq

        # If file has not changed and post-invalidation data is available, skip parse.
        if (
            mtime_ns == last_mtime
            and cached_pose is not None
            and cached_seq > valid_after_seq
        ):
            return True

        try:
            data = json.loads(self._ekf_state_path.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            return False

        try:
            seq: int = int(data.get("seq", 0))
            if seq <= valid_after_seq:
                return False
            cov_3x3 = np.asarray(data["covariance"], dtype=np.float64).reshape(3, 3)
            features = extract_2d_covariance_features(cov_3x3)
            # vx defaults to 0.0 for compatibility with stale ekf_state.json
            # files written by older extractor versions that omit the field.
            pose = np.array(
                [
                    data["x"],
                    data["y"],
                    data["yaw"],
                    data["vyaw"],
                    data.get("vx", 0.0),
                ],
                dtype=np.float64,
            )
        except (KeyError, ValueError):
            return False

        # Reject NaN/Inf writes. Treating these as "no data" causes the
        # env to fall back to the CARLA ground-truth pose rather than feeding
        # NaN observations directly into the policy.
        if not np.all(np.isfinite(pose)) or not np.all(np.isfinite(features)):
            logger.warning(
                "Rejecting EKF state write (seq=%d): contains NaN/Inf "
                "(EKF may still be initialising).",
                seq,
            )
            return False

        with self._lock:
            self._latest_pose = pose
            self._latest_uncertainty = features
            self._last_read_seq = seq
            self._last_mtime_ns = mtime_ns
        return True

    def get_latest_state(
        self,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        @brief Read the shared file once and return both pose and uncertainty.

        Reads the file exactly once per call, avoiding a
        redundant stat + JSON parse when both values are needed (e.g. _get_state).

        @return Tuple of (pose, uncertainty) where:
                pose: shape (5,) = [x, y, yaw, vyaw, vx], or None.
                uncertainty: shape (COVARIANCE_FEATURES_DIM,) feature vector, or None.
        """
        self._read_file()
        with self._lock:
            pose = (
                cast(np.ndarray, self._latest_pose.copy())
                if self._latest_pose is not None
                else None
            )
            uncertainty = (
                cast(np.ndarray, self._latest_uncertainty.copy())
                if self._latest_uncertainty is not None
                else None
            )
        return pose, uncertainty

    def get_latest_uncertainty(self) -> Optional[np.ndarray]:
        """
        @brief Get the most recent uncertainty feature vector.

        @note Use get_latest_state() when pose is also needed to avoid a
              second file read.
        @return Array of shape (COVARIANCE_FEATURES_DIM,) or None if no data available.
        """
        self._read_file()
        with self._lock:
            if self._latest_uncertainty is not None:
                return cast(np.ndarray, self._latest_uncertainty.copy())
            return None

    def get_latest_pose(self) -> Optional[np.ndarray]:
        """
        @brief Get the most recent EKF pose and velocity estimate.

        @note Use get_latest_state() when uncertainty is also needed to avoid
              a second file read.
        @return Array of shape (5,) = [x, y, yaw, vyaw, vx] or None.
        """
        self._read_file()
        with self._lock:
            if self._latest_pose is not None:
                return cast(np.ndarray, self._latest_pose.copy())
            return None

    def get_active_tier(self) -> Optional[str]:
        """
        @brief Read the LIVE GNSS fix-state tier from episode_config.json.

        The gnss_noise_relay walks the fix-state Markov chain at 20 Hz and
        writes the current tier back into episode_config.json on every
        transition (GnssNoiseRelayNode._write_active_tier). Reading it here is
        how the env recovers the true per-tick tier: the env's own
        _current_gnss_tier only holds the START tier sampled at reset, not the
        relay's live drift. Read-only; never writes.

        @return The active tier name (e.g. 'rtk_fixed'), or None if the file is
                absent/unreadable or carries no tier (e.g. CI/tests with no relay).
        """
        try:
            data = json.loads(self._episode_config_path.read_text())
        except (OSError, json.JSONDecodeError):
            return None
        tier = data.get("tier_name")
        return str(tier) if tier else None

    # -----------------------------------------------------------------------
    # Episode signal writers
    # -----------------------------------------------------------------------

    def publish_initial_pose(self, x: float, y: float, yaw: float) -> None:
        """
        @brief Signal the spawn pose to the ros2-bridge via a shared file.

        Writes initial_pose.json with the spawn position in CARLA world frame.
        The CovarianceExtractorNode in the ros2-bridge container watches this
        file and publishes /set_pose locally (same DDS domain as the EKF).

        The CARLA-to-ROS frame conversion (negate y and yaw) is applied by the
        extractor node at publish time, keeping this file in CARLA convention.

        @param x: Spawn X in CARLA world frame (metres).
        @param y: Spawn Y in CARLA world frame (metres).
        @param yaw: Spawn heading in radians (CARLA convention).
        """
        self._initial_pose_seq += 1
        data = {
            "seq": self._initial_pose_seq,
            "x": float(x),
            "y": float(y),
            "yaw": float(yaw),
        }
        try:
            with open(self._initial_pose_tmp, "w") as f:
                json.dump(data, f)
            os.replace(self._initial_pose_tmp, self._initial_pose_path)
            logger.info(
                "Initial pose written: x=%.2f y=%.2f yaw=%.1fdeg (seq=%d)",
                x,
                y,
                math.degrees(yaw),
                self._initial_pose_seq,
            )
        except OSError as exc:
            logger.warning("Failed to write initial_pose.json: %s", exc)

    def publish_episode_config(
        self,
        tier_name: str,
        datum_lat: Optional[float] = None,
        datum_lon: Optional[float] = None,
        hold_tier: bool = False,
        degrade_one_way: bool = False,
        degrade_rate_scale: float = 1.0,
    ) -> None:
        """
        @brief Signal the GNSS noise tier and spawn datum to the ros2-bridge.
        @param tier_name: RTK fix-state tier name (e.g. 'rtk_fixed') - the tier
                          the episode STARTS in.
        @param datum_lat: Latitude (degrees) of vehicle spawn (CARLA geolocation).
        @param datum_lon: Longitude (degrees) of vehicle spawn.
        @param hold_tier: If True, the relay holds this tier for the whole
                          episode (Markov drift suppressed) - the controlled
                          fixed-level evaluation conditions. Training leaves it
                          False so the chain wanders.
        @param degrade_one_way: If True, the relay lets the Markov chain only
                          degrade (never recover) - the monotone-degradation eval
                          condition. Mutually exclusive with hold_tier (a held tier
                          has no drift). Training leaves it False.
        @param degrade_rate_scale: Multiplier on the one-way chain's downward
                          transition mass. 1.0 (the default everywhere else) is
                          the datasheet-anchored schedule; the drift condition
                          raises it so the walk to the worst tier completes
                          inside the episode horizon. Ignored unless
                          degrade_one_way is set.
        """
        self._episode_config_seq += 1
        data: Dict[str, Any] = {
            "seq": self._episode_config_seq,
            "tier_name": tier_name,
            "hold_tier": bool(hold_tier),
            "degrade_one_way": bool(degrade_one_way),
            "degrade_rate_scale": float(degrade_rate_scale),
        }
        if datum_lat is not None:
            data["datum_lat"] = datum_lat
        if datum_lon is not None:
            data["datum_lon"] = datum_lon
        try:
            with open(self._episode_config_tmp, "w") as f:
                json.dump(data, f)
            os.replace(self._episode_config_tmp, self._episode_config_path)
            logger.info(
                "GNSS noise config written: tier=%s hold=%s degrade_one_way=%s "
                "degrade_rate_scale=%.2f (seq=%d)",
                tier_name,
                hold_tier,
                degrade_one_way,
                degrade_rate_scale,
                self._episode_config_seq,
            )
        except OSError as exc:
            logger.warning("Failed to write episode_config.json: %s", exc)

    @property
    def has_data(self) -> bool:
        """
        @brief Check whether fresh EKF state data is available.

        Attempts a file read if no data is cached yet. Returns True if a
        valid (post-invalidation) reading is held in memory.

        @return True if data file exists and was read successfully.
        """
        with self._lock:
            if self._latest_uncertainty is not None:
                return True
        # Nothing cached - attempt a read.
        self._read_file()
        with self._lock:
            return self._latest_uncertainty is not None
