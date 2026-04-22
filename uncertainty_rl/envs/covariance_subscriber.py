"""
@file covariance_subscriber.py
@brief File-based covariance reader for EKF uncertainty features.

Reads the latest EKF state from a shared JSON file written by the
CovarianceExtractorNode in the ros2-bridge container. This avoids DDS
cross-distro serialisation issues between ROS 2 Humble (training container)
and Jazzy (ros2-bridge container).

The file is written atomically (via rename) by the extractor node at the
EKF publish rate (~20 Hz) to /workspace/outputs/ekf_state.json, which is
on a Docker shared volume visible to both containers.

The /set_pose signal for EKF state reset is also file-based: the
training container writes initial_pose.json and the CovarianceExtractorNode
in ros2-bridge reads it and publishes on /set_pose. Similarly, GNSS noise tier
config is signalled via gnss_noise_config.json for the GnssNoiseRelayNode.
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

# Default shared file path (Docker volume mount: outputs/ is rw in both containers).
# Individual _CovarianceSubscriber instances override this via ros2_config["ekf_state_file"]
# so that parallel CARLA workers each read from their own ros2-bridge's output file.
_EKF_STATE_PATH = Path("/workspace/outputs/ekf_state.json")

# File-based /set_pose signal.  The training container writes this file at
# episode reset; the CovarianceExtractorNode (ros2-bridge, Jazzy) watches it
# and publishes /set_pose locally.  Same DDS-bypass pattern as ekf_state.json.
_INITIAL_POSE_PATH = Path("/workspace/outputs/initial_pose.json")

# File-based GNSS noise config signal.  The training container writes this at
# episode reset with the current noise tier; the GnssNoiseRelayNode (ros2-bridge)
# reads it and applies dynamic noise to CARLA GNSS output.
_GNSS_NOISE_CONFIG_PATH = Path("/workspace/outputs/gnss_noise_config.json")

logger = logging.getLogger(__name__)


class _CovarianceSubscriber:
    """
    @class _CovarianceSubscriber
    @brief Reads EKF state from a shared JSON file + signals /set_pose.

    The CovarianceExtractorNode (ros2-bridge, Jazzy) writes the latest EKF
    pose, velocity, and 3x3 covariance to a shared file. This class reads
    that file on demand -- no DDS subscription needed.

    The /set_pose signal is also file-based: this class writes
    initial_pose.json and the extractor node reads it and publishes
    /set_pose within the ros2-bridge container (same DDS domain).
    """

    def __init__(
        self,
        covariance_topic: str = "/odometry/filtered",
        node_name: str = "covariance_subscriber",
        ros2_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        @brief Initialise the covariance reader.
        @param covariance_topic: Unused -- EKF state is read from the shared
               JSON file, not via DDS. Accepted for call-site compatibility.
        @param node_name: Unused -- no rclpy Node. Accepted for compatibility.
        @param ros2_config: Optional ROS 2 config dict from train_config.yaml.
        """
        self._lock = threading.Lock()
        self._latest_uncertainty: Optional[np.ndarray] = None
        self._latest_pose: Optional[np.ndarray] = None
        # Sequence number of the last-seen write from the extractor node.
        # invalidate() records the current seq; _read_file() only accepts a
        # file whose `seq` field is strictly greater than _valid_after_seq.
        # This is clock-skew-proof -- mtime comparisons across Docker container
        # clocks are unreliable on some host configurations.
        self._valid_after_seq: int = 0
        self._last_read_seq: int = 0
        # Monotonically increasing counter for initial_pose.json writes.
        self._initial_pose_seq: int = 0

        # Per-worker EKF state file path. Priority order:
        #   1. ros2_config["ekf_state_file"] -- set by make_env() for rank > 0
        #   2. EKF_STATE_FILE env var -- set by docker-compose.parallel.yml
        #   3. _EKF_STATE_PATH module constant -- single-instance default
        config = ros2_config or {}
        ekf_state_file: str = config.get(
            "ekf_state_file",
            os.environ.get("EKF_STATE_FILE", str(_EKF_STATE_PATH)),
        )
        self._ekf_state_path: Path = Path(ekf_state_file)

        # Initial pose file path (shared volume, same dir as EKF state).
        self._initial_pose_path: Path = Path(
            config.get(
                "initial_pose_file",
                os.environ.get("INITIAL_POSE_FILE", str(_INITIAL_POSE_PATH)),
            )
        )
        self._initial_pose_tmp: Path = self._initial_pose_path.with_suffix(
            ".json.tmp"
        )

        # GNSS noise config file path (shared volume, same dir as EKF state).
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
            # seq field added in extractor v2; fall back to mtime guard for
            # old extractor images that predate the seq field.
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
                    data["vx"],
                    data["vy"],
                    data["vyaw"],
                ],
                dtype=np.float64,
            )
            # Reject NaN/Inf writes: the EKF can diverge (e.g. Cholesky failure
            # on the first prediction step before the first GNSS correction) and
            # write NaN to every field. Treating these as "no data" causes the
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
                pose: shape (6,) = [x, y, yaw, vx, vy, vyaw], or None.
                uncertainty: shape (9,) uncertainty feature vector, or None.
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
        @brief Get the most recent 9-element uncertainty feature vector.

        @note Use get_latest_state() when pose is also needed to avoid a
              second file read.
        @return Array of shape (9,) or None if no data available.
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
        @return Array of shape (6,) = [x, y, yaw, vx, vy, vyaw] or None.
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

    def publish_gnss_noise_config(self, tier_name: str) -> None:
        """
        @brief Signal the GNSS noise tier to the ros2-bridge via a shared file.

        Writes gnss_noise_config.json with the episode's RTK fix-state tier
        name. GnssNoiseRelayNode reads this file and calls _apply_tier() to
        look up the corresponding noise parameters from its own _tier_params
        table, which is the single source of truth for stddev values.

        The seq field is a monotonically increasing counter so the relay can
        detect a new episode even when the tier name is unchanged.

        @param tier_name: RTK fix-state tier name (e.g. 'rtk_fixed').
        """
        self._gnss_noise_config_seq += 1
        data: Dict[str, Any] = {
            "seq": self._gnss_noise_config_seq,
            "tier_name": tier_name,
        }
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
        # Nothing cached -- attempt a read.
        self._read_file()
        with self._lock:
            return self._latest_uncertainty is not None
