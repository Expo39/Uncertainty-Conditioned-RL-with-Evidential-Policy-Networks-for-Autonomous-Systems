"""
@file deployment_utils.py
@brief Real-world deployment utilities: surveyed datum loading and actuator calibration.

Owns the static mission configuration for a real-world deployment: the surveyed
lot datum (EKF frame reference), the target bay resolved from the layout YAML,
and the per-actuator calibration map. No ROS 2 or runtime sensor state.

@warning Never run on hardware. The datum and calibration values ship as
         unmeasured placeholders, so both the frame transform and the actuator
         map reduce to identities until a site survey and a calibration run
         supply real figures.
"""

import logging
import math
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import yaml

from uncertainty_rl.envs._parking_core import _layout_cache, load_floor_plan
from uncertainty_rl.utils.actuation_calibration import ActuationCalibration

logger = logging.getLogger(__name__)


class RealWorldDeployment:
    """
    @class RealWorldDeployment
    @brief Surveyed lot datum + per-actuator calibration for real-vehicle deployment.

    Owns the EKF frame reference (lot_x/y/heading from real_world_datum.yaml) and
    maps normalised policy actions to physical actuator commands via
    ActuationCalibration.
    """

    def __init__(
        self,
        datum: Dict[str, Any],
        actuation: ActuationCalibration,
    ) -> None:
        """
        @brief Construct from pre-loaded datum dict and actuation calibration.
        @param datum: Contents of the 'datum' key from real_world_datum.yaml.
        @param actuation: ActuationCalibration instance.
        """
        self._datum = datum
        self._actuation = actuation
        self._lot_x: float = float(datum.get("lot_x", 0.0))
        self._lot_y: float = float(datum.get("lot_y", 0.0))
        self._lot_yaw_rad: float = math.radians(float(datum.get("heading_deg", 0.0)))

    @classmethod
    def from_config(
        cls,
        datum_path: Optional[str],
        calibration_path: Optional[str],
    ) -> "RealWorldDeployment":
        """
        @brief Load datum and actuation calibration from YAML files.

        Returns a RealWorldDeployment with identity actuation if either file
        is absent or unpopulated.

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

    @classmethod
    def from_mission(
        cls,
        mission_path: str,
        datum_path: Optional[str] = None,
        calibration_path: Optional[str] = None,
    ) -> Tuple["RealWorldDeployment", Dict[str, Any]]:
        """
        @brief Load a complete real-world mission from configs/deployment/real/.

        Reads mission.yaml to get target_bay_id and layout_file, loads the
        layout, resolves the target bay, then constructs the deployment object.

        @param mission_path: Path to mission.yaml.
        @param datum_path: Path to real_world_datum.yaml (optional).
        @param calibration_path: Path to actuation_calibration.yaml (optional).
        @return Tuple of (RealWorldDeployment instance, target bay dict).
        @raises FileNotFoundError if mission.yaml or layout_file do not exist.
        @raises ValueError if target_bay_id is not found in the layout.
        """
        mission_file = Path(mission_path)
        if not mission_file.exists():
            raise FileNotFoundError(
                f"Mission config not found: {mission_path}. "
                "Create configs/deployment/real/mission.yaml and set "
                "target_bay_id and layout_file."
            )

        with open(mission_file, "r") as f:
            mission = yaml.safe_load(f) or {}

        bay_id = str(mission.get("target_bay_id", ""))
        layout_file = str(mission.get("layout_file", ""))

        if not bay_id:
            raise ValueError(
                f"target_bay_id not set in {mission_path}. "
                "Edit the file and set the bay ID before deploying."
            )
        if not layout_file:
            raise ValueError(
                f"layout_file not set in {mission_path}. "
                "Set it to the layout YAML for the deployment site."
            )

        logger.info(
            "Mission loaded: target_bay_id='%s'  layout='%s'",
            bay_id,
            layout_file,
        )

        # Load layout using the shared cache (same call as sim side).
        _, layout = load_floor_plan(
            floor_plans_config={"deployment": {"layout_file": layout_file}},
            eval_mode=True,
            layout_cache=_layout_cache,
        )

        deployment = cls.from_config(
            datum_path=datum_path,
            calibration_path=calibration_path,
        )
        target_bay = deployment.set_target_bay(bay_id, layout)
        return deployment, target_bay

    def reference_pose(self) -> Tuple[float, float, float]:
        """
        @brief Return the surveyed datum pose in the lot layout frame.
        @return Tuple (x_m, y_m, yaw_rad). Returns (0, 0, 0) if datum not loaded.
        @warning Check logs for load failure before trusting this value.
        """
        return self._lot_x, self._lot_y, self._lot_yaw_rad

    def datum_loaded(self) -> bool:
        """
        @brief Return True if a non-empty datum was successfully loaded.
        @return True if datum dict is populated.
        """
        return bool(self._datum)

    def set_target_bay(
        self,
        bay_id: str,
        layout: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        @brief Look up a bay by ID in the loaded layout and return its dict.

        @param bay_id: Bay identifier string matching the 'id' field in the
                       layout YAML (e.g. 'B01', 'A03').
        @param layout: Parsed layout dict (from _parking_core.load_floor_plan).
        @return Bay dict with keys: x, y, yaw, width, depth, bay_type, bay_id.
        @raises ValueError if bay_id is not found or the bay is always_empty.
        """
        bays = layout.get("bays", [])
        for bay in bays:
            if bay.get("id", bay.get("bay_id", "")) == bay_id:
                if bay.get("always_empty", False):
                    raise ValueError(
                        f"Bay '{bay_id}' is marked always_empty and cannot "
                        "be used as a target."
                    )
                return {
                    "x": float(bay["x"]),
                    "y": float(bay["y"]),
                    "yaw": (
                        float(bay["yaw"])
                        if "yaw" in bay
                        else math.radians(float(bay.get("yaw_deg", 0.0)))
                    ),
                    "width": float(bay.get("width", 2.5)),
                    "depth": float(bay.get("depth", 5.0)),
                    "bay_type": bay.get("bay_type", "perpendicular"),
                    "bay_id": bay_id,
                }
        raise ValueError(
            f"Bay ID '{bay_id}' not found in layout. "
            f"Available IDs: {[b.get('id', b.get('bay_id', '')) for b in bays]}"
        )

    def calibrate_action(
        self, steering: float, throttle: float, brake: float
    ) -> Tuple[float, float, float]:
        """
        @brief Map policy action outputs to physical actuator commands.

        @param steering: Policy steering output in [-1, 1].
        @param throttle: Policy throttle output in [0, 1].
        @param brake: Policy brake output in [0, 1].
        @return Tuple (steering_cmd, throttle_cmd, brake_cmd) in physical range.
        """
        return self._actuation.apply(steering, throttle, brake)
