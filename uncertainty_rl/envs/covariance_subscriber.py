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

# Default shared file path
_EKF_STATE_PATH = Path("/workspace/outputs/ekf_state.json")

# File-based /set_pose signal
_INITIAL_POSE_PATH = Path("/workspace/outputs/initial_pose.json")

# File-based GNSS noise config signal
_GNSS_NOISE_CONFIG_PATH = Path("/workspace/outputs/gnss_noise_config.json")

logger = logging.getLogger(__name__)


class _CovarianceSubscriber:
    """
    @class _CovarianceSubscriber
    @brief Reads EKF state from a shared JSON file + signals /set_pose.

    The CovarianceExtractorNode (ros2-bridge, Jazzy) writes the latest EKF
    pose, velocity, and 3x3 covariance to a shared file. This class reads
    that file on demand - no DDS subscription needed.
    """

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
        self._initial_pose_tmp: Path = self._initial_pose_path.with_suffix(
            ".json.tmp"
        )

        # GNSS noise config file path
        self._gnss_noise_config_path: Path = Path(
            config.get(
                "gnss_noise_config_file",
                os.environ.get(
                    "GNSS_NOISE_CONFIG_FILE", str(_GNSS_NOISE_CONFIG_PATH)
                ),
            )
        )
        self._gnss_noise_config_tmp: Path = (
            self._gnss_noise_config_path.with_suffix(".json.tmp")
        )
        # Monotonically increasing counter for gnss_noise_config.json writes.
        self._gnss_noise_config_seq: int = 0

        # Remove stale JSON files from previous sessions so the ros2-bridge
        # nodes do not pick up old state on startup.
        for stale in [
            self._ekf_state_path,
            self._initial_pose_path,
            self._initial_pose_tmp,
            self._gnss_noise_config_path,
            self._gnss_noise_config_tmp,
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
            if not self._ekf_state_path.exists():
                return False
            data = json.loads(self._ekf_state_path.read_text())
            seq: int = int(data.get("seq", 0))
            with self._lock:
                valid_after_seq = self._valid_after_seq
            if seq <= valid_after_seq:
                return False
            cov_3x3 = np.array(data["covariance"]).reshape(3, 3)
            features = extract_2d_covariance_features(cov_3x3)
            pose = np.array(
                [
                    data["x"],
                    data["y"],
                    data["yaw"],
                    data["vyaw"],
                ],
                dtype=np.float64,
            )
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
            return True
        except (json.JSONDecodeError, KeyError, ValueError):
            return False

    def get_latest_state(
        self,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        @brief Read the shared file once and return both pose and uncertainty.

        Reads the file exactly once per call, avoiding a
        redundant stat + JSON parse when both values are needed (e.g. _get_state).

        @return Tuple of (pose, uncertainty) where:
                pose: shape (4,) = [x, y, yaw, vyaw], or None.
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
        @return Array of shape (4,) = [x, y, yaw, vyaw] or None.
        """
        self._read_file()
        with self._lock:
            if self._latest_pose is not None:
                return cast(np.ndarray, self._latest_pose.copy())
            return None

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
            os.makedirs(self._initial_pose_path.parent, exist_ok=True)
            with open(self._initial_pose_tmp, "w") as f:
                json.dump(data, f)
            os.replace(str(self._initial_pose_tmp), str(self._initial_pose_path))
            logger.info(
                "Initial pose written: x=%.2f y=%.2f yaw=%.1fdeg (seq=%d)",
                x,
                y,
                math.degrees(yaw),
                self._initial_pose_seq,
            )
        except OSError as exc:
            logger.warning("Failed to write initial_pose.json: %s", exc)

    def publish_gnss_noise_config(
        self,
        tier_name: str,
        datum_lat: Optional[float] = None,
        datum_lon: Optional[float] = None,
        spawn_yaw: Optional[float] = None,
    ) -> None:
        """
        @brief Signal the GNSS noise tier, spawn datum, and initial yaw to the ros2-bridge.

        Writes gnss_noise_config.json with the episode's RTK fix-state tier
        name, the geolocation of the vehicle spawn point, and the spawn yaw.
        
        Re-latching the datum each episode ensures that GNSS Odometry (0, 0)
        and /set_pose (0, 0) agree at episode reset, eliminating the systematic
        EKF drift that occurs when /set_pose and GNSS use different origins.

        @param tier_name: RTK fix-state tier name (e.g. 'rtk_fixed').
        @param datum_lat: Latitude (degrees) of vehicle spawn (CARLA geolocation).
               When provided, the relay re-latches the GNSS flat-earth datum.
               When None, the relay falls back to auto-latching on the next
               GNSS callback (real-vehicle mode without CARLA API).
        @param datum_lon: Longitude (degrees) of vehicle spawn.
        @param spawn_yaw: Vehicle heading at spawn in radians, CARLA convention
               (same as used by publish_initial_pose and the EKF /set_pose).
               Seeds the COG heading so the EKF receives a correct initial yaw
               before the first valid COG reading. When None, the relay waits
               for the first valid COG reading before publishing any heading.
        """
        self._gnss_noise_config_seq += 1
        data: Dict[str, Any] = {
            "seq": self._gnss_noise_config_seq,
            "tier_name": tier_name,
        }
        if datum_lat is not None:
            data["datum_lat"] = datum_lat
        if datum_lon is not None:
            data["datum_lon"] = datum_lon
        if spawn_yaw is not None:
            data["spawn_yaw"] = spawn_yaw
        try:
            os.makedirs(self._gnss_noise_config_path.parent, exist_ok=True)
            with open(self._gnss_noise_config_tmp, "w") as f:
                json.dump(data, f)
            os.replace(
                str(self._gnss_noise_config_tmp),
                str(self._gnss_noise_config_path),
            )
            logger.info(
                "GNSS noise config written: tier=%s (seq=%d)",
                tier_name,
                self._gnss_noise_config_seq,
            )
        except OSError as exc:
            logger.warning("Failed to write gnss_noise_config.json: %s", exc)

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
