"""
@file real_world_deployment.py
@brief Real-world deployment utilities for the autonomous parking system.

This module owns everything that is specific to deploying the trained policy
on a real vehicle rather than in CARLA simulation:

  - Loading the surveyed lot datum (UTM easting/northing -> lot layout frame).
  - Computing the EKF odom-frame-to-lot-layout-frame rigid transform from the
    datum, replacing the CARLA ground-truth path in carla_parking.py.
  - Loading and applying per-actuator gain/deadband/bias calibration so the
    policy's normalised [-1, 1] outputs drive the physical vehicle correctly.

None of this code has any CARLA dependency. It is imported by CARLAParkingEnv
only when real_world_deployment=True.

Typical usage inside CARLAParkingEnv:

    self._deployment = RealWorldDeployment.from_config(
        datum_path="configs/real_world_datum.yaml",
        calibration_path="configs/actuation_calibration.yaml",
    )

    # At episode reset, instead of calling vehicle.get_transform():
    world_x, world_y, world_yaw = self._deployment.reference_pose()

    # In step(), before sending to CARLA/vehicle:
    steer_cmd, long_cmd = self._deployment.calibrate_action(steer, longitudinal)

@see configs/real_world_datum.yaml
@see configs/actuation_calibration.yaml
@see documentation/design/sim_to_real_transfer.md

@author Antonio Galdes
"""

import logging
import math
from typing import Any, Dict, Optional, Tuple

import yaml

from uncertainty_rl.utils.actuation_calibration import ActuationCalibration

logger = logging.getLogger(__name__)


class RealWorldDeployment:
    """
    @class RealWorldDeployment
    @brief Encapsulates all real-vehicle deployment logic.

    In simulation, CARLAParkingEnv uses CARLA ground-truth transforms for EKF
    frame calibration and identity actuation. On the real vehicle both of those
    assumptions break:

      1. There is no CARLA -- the EKF frame offset must be derived from a
         surveyed datum point measured at the test site.
      2. The physical actuators (steering rack, throttle, brake) do not respond
         linearly to normalised [-1, 1] commands -- they have deadbands, gain
         differences, and biases that must be calibrated.

    This class owns both concerns and exposes a clean interface so
    carla_parking.py does not need to contain any real-world-specific logic.

    @note All values in datum and calibration files are UNMEASURED placeholders
          until filled in at the deployment site. See the YAML files for the
          measurement procedure.
    """

    def __init__(
        self,
        datum: Dict[str, Any],
        actuation: ActuationCalibration,
    ) -> None:
        """
        @brief Construct from pre-loaded datum dict and actuation calibration.
        @param datum: Contents of the 'datum' key from real_world_datum.yaml.
        @param actuation: ActuationCalibration instance (identity if uncalibrated).
        """
        self._datum = datum
        self._actuation = actuation

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_config(
        cls,
        datum_path: Optional[str],
        calibration_path: Optional[str],
    ) -> "RealWorldDeployment":
        """
        @brief Load datum and actuation calibration from YAML files.

        Returns a RealWorldDeployment with identity actuation if either file
        is absent or unpopulated. Logs a warning for missing datum (required
        for correct EKF calibration) but does not raise -- the caller will see
        identity transform values and can decide how to handle them.

        @param datum_path: Path to configs/real_world_datum.yaml.
        @param calibration_path: Path to configs/actuation_calibration.yaml.
        @return RealWorldDeployment instance.
        """
        datum: Dict[str, Any] = {}
        if datum_path:
            try:
                with open(datum_path, "r") as f:
                    doc = yaml.safe_load(f) or {}
                datum = doc.get("datum", {})
                logger.info(
                    "Real-world datum loaded from %s: "
                    "utm=(%.2f, %.2f) heading=%.1f deg lot=(%.2f, %.2f)",
                    datum_path,
                    datum.get("utm_easting", 0.0),
                    datum.get("utm_northing", 0.0),
                    datum.get("heading_deg", 0.0),
                    datum.get("lot_x", 0.0),
                    datum.get("lot_y", 0.0),
                )
            except (OSError, KeyError) as exc:
                logger.warning(
                    "Failed to load real-world datum from %s: %s. "
                    "EKF calibration will use identity transform.",
                    datum_path,
                    exc,
                )

        actuation = (
            ActuationCalibration.from_config(calibration_path)
            if calibration_path
            else ActuationCalibration.identity()
        )

        return cls(datum=datum, actuation=actuation)

    # ------------------------------------------------------------------
    # EKF frame calibration
    # ------------------------------------------------------------------

    def reference_pose(self) -> Tuple[float, float, float]:
        """
        @brief Return the vehicle's reference pose in the lot layout frame.

        This is the position of the datum marker (a surveyed physical point in
        the parking lot) expressed in the lot layout coordinate frame -- the
        same frame used in configs/layouts/*.yaml and the bird's-eye PNGs.

        Used by CARLAParkingEnv._calibrate_ekf_frame_offset() as the
        ground-truth pose when CARLA is not available.

        @return Tuple (x_m, y_m, yaw_rad) in lot layout frame.
        @warning Returns (0, 0, 0) if datum was not loaded. Check logs.
        """
        x = float(self._datum.get("lot_x", 0.0))
        y = float(self._datum.get("lot_y", 0.0))
        yaw = math.radians(float(self._datum.get("heading_deg", 0.0)))
        return x, y, yaw

    def datum_loaded(self) -> bool:
        """
        @brief Return True if a non-empty datum was successfully loaded.
        @return True if datum dict is populated.
        """
        return bool(self._datum)

    # ------------------------------------------------------------------
    # Actuation calibration
    # ------------------------------------------------------------------

    def calibrate_action(
        self, steering: float, longitudinal: float
    ) -> Tuple[float, float]:
        """
        @brief Map policy action outputs to physical actuator commands.

        Applies the per-actuator gain/deadband/bias loaded from
        configs/actuation_calibration.yaml. Returns the inputs unchanged
        if the calibration file was absent or uncalibrated (identity mapping).

        @param steering: Policy steering output in [-1, 1].
        @param longitudinal: Policy longitudinal output in [-1, 1].
        @return Tuple (steering_cmd, longitudinal_cmd) in physical range.
        """
        return self._actuation.apply(steering, longitudinal)
