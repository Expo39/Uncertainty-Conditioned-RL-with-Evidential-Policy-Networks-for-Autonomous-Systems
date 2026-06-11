"""
@file lot_inspector.py
@brief Entry point for the unified CARLA parking lot and sensor inspector.
"""

import argparse
import os
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import carla  # noqa: F401 - imported for the sys.exit guard below
except ImportError:
    print("ERROR: carla Python package not found.  Run inside the training container.")
    sys.exit(1)

from scripts.inspect.inspectors import (
    DryRunInspector,
    LayoutInspector,
    LiveInspector,
    SensorInspector,
    _Inspector,
)
from uncertainty_rl.envs.sim.carla_parking import CARLAParkingEnv

# Suppress Gymnasium's float64->float32 precision warning for unbounded obs.
warnings.filterwarnings(
    "ignore",
    message=".*Box.*precision lowered.*",
    category=UserWarning,
)

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _build_env(
    host: str,
    port: int,
    layout: str,
    train_cfg: Dict[str, Any],
    sensors_cfg: Optional[Dict[str, Any]] = None,
    full_lot: bool = True,
) -> CARLAParkingEnv:
    """
    @brief Build and reset a CARLAParkingEnv configured for the inspector.

    @param host: CARLA server hostname.
    @param port: CARLA server port.
    @param layout: Floor plan name ('rectangle', 'trapezoid', 'irregular_a').
    @param train_cfg: Full train_config.yaml dict.
    @param sensors_cfg: Optional sensors config dict (for sensor mode).
    @param full_lot: If True spawn full lot (bays, NPCs, cones).  If False,
                     suppress NPCs/cones to keep scene clean for sensor mode.
    @return Pre-reset CARLAParkingEnv instance.
    """
    scenarios = dict(train_cfg.get("parking_scenarios", {}))
    scenarios["floor_plans"] = {
        layout: {
            "weight": 1.0,
            "always_empty": [],
            "layout_file": f"configs/layouts/{layout}.yaml",
        }
    }
    if not full_lot:
        scenarios["bay_occupancy_rate"] = 0.0
        scenarios["num_patrol_vehicles_max"] = 0
        scenarios["num_pedestrians_max"] = 0
        scenarios["perimeter_cone_spacing"] = 9999.0

    env = CARLAParkingEnv(
        carla_host=host,
        carla_port=port,
        town=train_cfg.get("town", "FlatPlane"),
        parking_scenarios_config=scenarios,
        carla_sensors_config=sensors_cfg,
        include_covariance=False,
        include_obstacle_obs=False,
    )
    return env


def main() -> None:
    """@brief Parse arguments and run the selected inspector mode."""
    parser = argparse.ArgumentParser(
        description=(
            "Unified CARLA parking lot + sensor inspector.  "
            "Shows sensors in the context of the actual parking lot layout."
        )
    )
    parser.add_argument(
        "--mode",
        default="sensors",
        choices=["layout", "sensors", "live", "dryrun"],
        help=(
            "Inspector mode: 'layout' = lot geometry only; "
            "'sensors' = sensors on lot; "
            "'live' = real spawned sensors with live output "
            "(LiDAR debug dots); "
            "'dryrun' = full training pipeline with random actions (no model), "
            "spectator follows ego, EKF covariance printed to console. "
            "Default: sensors."
        ),
    )
    parser.add_argument(
        "--layout",
        default="rectangle",
        choices=["rectangle", "trapezoid", "irregular_a"],
        help="Floor plan to spawn (default: rectangle).",
    )
    parser.add_argument(
        "--view",
        default="birds_eye",
        choices=["birds_eye", "side", "front"],
        help=(
            "Spectator view for sensors mode only: "
            "'birds_eye' = top-down showing FOV arcs against the lot (default); "
            "'side' = left-profile showing sensor mount heights; "
            "'front' = front-profile showing sensor lateral positions.  "
            "Not used in live mode."
        ),
    )
    parser.add_argument(
        "--zoom",
        default="close",
        choices=["close", "wide"],
        help=(
            "Camera height (birds_eye only): "
            "'close' = 80 m above vehicle (lot detail, default); "
            "'wide' = high enough to see the full FOV arc boundary."
        ),
    )
    parser.add_argument(
        "--host", default="carla-server-demo", help="CARLA server hostname."
    )
    parser.add_argument("--port", type=int, default=2100, help="CARLA server port.")
    parser.add_argument(
        "--duration",
        type=int,
        default=86400,
        help="Seconds to run (default: 86400 = 24 h). Episodes end via max_steps or Ctrl+C.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help=(
            "Maximum number of episodes for dryrun mode (default: unlimited).  "
            "Ignored in all other modes."
        ),
    )
    parser.add_argument(
        "--inspect-view",
        default="third_person",
        choices=["third_person", "side", "back", "front", "free", "birds_eye"],
        dest="inspect_view",
        help=(
            "Spectator view for dryrun mode (default: third_person).  "
            "'free' places the spectator overhead once and then does not move "
            "it, so you can fly around with CARLA's own controls.  "
            "'birds_eye' is a top-down camera that follows the ego vehicle "
            "with the vehicle's heading aligned to screen up.  "
            "Ignored in all other modes."
        ),
    )
    parser.add_argument(
        "--termination-pause",
        type=float,
        default=3.0,
        dest="termination_pause",
        help=(
            "Seconds to hold the scene after an episode ends before resetting "
            "(default: 3.0).  Set to 0 to disable.  Dryrun mode only."
        ),
    )
    parser.add_argument(
        "--manual",
        action="store_true",
        default=False,
        help=(
            "Enable keyboard control in dryrun mode. "
            "Arrow keys: Up=throttle, Down=brake, Left/Right=steer. "
            "Requires pynput (installed in inspect container). "
            "Ignored in all other modes."
        ),
    )
    parser.add_argument(
        "--stage",
        type=int,
        default=None,
        help=(
            "Curriculum stage (1..N). Dryrun builds the env at this stage's "
            "difficulty, identical to training. Omit to default to stage 1. "
            "Dryrun mode only."
        ),
    )
    parser.add_argument(
        "--baseline",
        type=str,
        default=None,
        help=(
            "Baseline override config (configs/baselines/*.yaml). Sets the "
            "observation flags / policy_type the dryrun env is built with, "
            "identical to training. Omit to default to the full method. "
            "Dryrun mode only."
        ),
    )
    args = parser.parse_args()

    from uncertainty_rl.training.train_ppo import DEFAULT_STAGE, load_env_config

    # Difficulty knobs live only in the stage files; resolve the stage the same way
    # training does (--stage, defaulting to the curriculum head) so the inspector
    # reflects a real training condition, not the env constructor defaults. The
    # layout/sensor/live modes override the layout and occupancy themselves; the
    # dryrun mode additionally applies the baseline + train_config below to build an
    # env identical to a training rollout.
    stage = args.stage if args.stage is not None else DEFAULT_STAGE
    train_cfg = load_env_config("configs/deployment/sim/env_config.yaml", stage=stage)

    print(f"Connecting to CARLA at {args.host}:{args.port} ...")
    if args.mode != "dryrun":
        print(f"Mode: {args.mode}  |  Layout: {args.layout}", end="")
        if args.mode == "sensors":
            print(f"  |  View: {args.view}", end="")
        print()

    if args.mode == "layout":
        env = _build_env(
            args.host,
            args.port,
            args.layout,
            train_cfg,
            full_lot=True,
        )
        env.reset()
        if env.world is None:
            print("ERROR: Could not connect to CARLA.")
            env.close()
            sys.exit(1)

        inspector: _Inspector = LayoutInspector(env, args.duration)
        inspector.place_spectator()  # type: ignore[attr-defined]
        print("Debug overlays:")
        print("  blue=perpendicular bays | grey=motorcycle bays")
        print(
            "  bright green=TARGET | yellow=SPAWN"
            " | turquoise=pedestrian zones | red=patrol path"
        )

    elif args.mode == "sensors":
        sensors_cfg = dict(train_cfg.get("carla_sensors", {}))

        env = _build_env(
            args.host,
            args.port,
            args.layout,
            train_cfg,
            sensors_cfg=sensors_cfg,
            # Side/front views: ego only. Birds-eye: full lot for context.
            full_lot=(args.view == "birds_eye"),
        )
        env.reset()
        if env.world is None or env.vehicle is None:
            print("ERROR: Could not connect to CARLA or spawn vehicle.")
            env.close()
            sys.exit(1)

        inspector = SensorInspector(
            env,
            args.duration,
            train_cfg,
            args.view,
            args.zoom,
        )
        inspector.place_spectator()  # type: ignore[attr-defined]
        print("Layout overlays:")
        print("  blue=perpendicular | yellow=angled | violet=parallel | green=TARGET")
        print("Sensor overlays:")
        print("  yellow=IMU | cyan=2D LiDAR | magenta=GNSS")
        print("FOV arcs:")
        print("  light-blue arc = 270 deg 2D LiDAR")

    elif args.mode == "dryrun":
        # Build the env through the SAME resolution as train_ppo.main() so the
        # dryrun is, by construction, identical to a training rollout: merge
        # train_config onto the stage-merged env config, then overlay the baseline
        # (defaulting to the full method). Stage difficulty and obs flags therefore
        # match what `make docker-train STAGE=.. BASELINE=..` would use.
        from uncertainty_rl.envs import make_env
        from uncertainty_rl.training.train_ppo import (
            DEFAULT_BASELINE,
            load_config,
            merge_configs,
        )
        from uncertainty_rl.utils.config_merge import apply_baseline
        from uncertainty_rl.utils.constants import STRICT_BAY_MARGIN

        baseline_path = args.baseline if args.baseline is not None else DEFAULT_BASELINE
        dryrun_cfg = merge_configs(load_config("configs/train_config.yaml"), train_cfg)
        apply_baseline(dryrun_cfg, load_config(baseline_path))
        # Windowed CARLA for visual inspection (training runs headless).
        dryrun_cfg["no_rendering_mode"] = False
        # Tick-level stepping (inspection-only override): the spectator camera,
        # keyboard input, and console readout all run per env.step(), so the
        # training action_repeat would drop them to the policy's decision rate
        # and make manual driving feel like a slideshow. The observation build
        # path is identical either way; training keeps action_repeat from
        # env_config.
        dryrun_cfg["action_repeat"] = 1
        print(
            f"Dryrun: stage {stage}, baseline "
            f"'{dryrun_cfg.get('baseline_name', Path(baseline_path).stem)}' "
            f"(bay_margin={dryrun_cfg.get('bay_margin')}, "
            f"include_covariance={dryrun_cfg.get('include_covariance')}, "
            f"include_obstacle_obs={dryrun_cfg.get('include_obstacle_obs')})"
        )

        # INSPECT_OOD is an inspection-only override: when true, swap the stage's
        # floor plans for the OOD-only set so you can eyeball held-out layouts.
        # Default false keeps the dryrun on the stage's training layouts.
        inspect_ood = os.environ.get("INSPECT_OOD", "false").lower() == "true"
        if inspect_ood:
            scenarios = dict(dryrun_cfg.get("parking_scenarios", {}))
            ood_plans = {
                name: {k: v for k, v in cfg.items() if k != "ood"}
                for name, cfg in scenarios.get("floor_plans", {}).items()
                if cfg.get("ood", False)
            }
            if not ood_plans:
                raise RuntimeError(
                    "INSPECT_OOD=true but no OOD floor plans are defined in "
                    "env_config.yaml parking_scenarios.floor_plans."
                )
            scenarios["floor_plans"] = ood_plans
            dryrun_cfg["parking_scenarios"] = scenarios

        # bay_margin from the resolved config (the stage's margin), exactly as the
        # training factory reads it - not a hardcoded constant.
        dryrun_bay_margin = float(dryrun_cfg.get("bay_margin", STRICT_BAY_MARGIN))
        env = make_env(
            dryrun_cfg,
            bay_margin=dryrun_bay_margin,
            rank=0,
            host_override=args.host,
            port_override=args.port,
        )()
        env.reset()
        if env.world is None or env.vehicle is None:
            print("ERROR: Could not connect to CARLA or spawn vehicle.")
            env.close()
            sys.exit(1)

        dryrun_action = train_cfg.get("inspect", {}).get("dryrun_action", None)
        inspector = DryRunInspector(
            env,
            args.duration,
            args.episodes,
            dryrun_action,
            initial_view=args.inspect_view,
            termination_pause=args.termination_pause,
            manual=args.manual,
        )
        inspector.place_spectator()  # type: ignore[attr-defined]
        if args.manual:
            action_desc = "keyboard (Up=throttle, Down=brake, Left/Right=steer)"
        else:
            action_desc = f"constant {dryrun_action}"
        layout_mode = "OOD" if inspect_ood else "training"
        _plans = list(dryrun_cfg.get("parking_scenarios", {}).get("floor_plans", {}))
        print(f"Dry-run mode: full training pipeline, action={action_desc}, no model.")
        print(f"  Layouts ({layout_mode}): {_plans} (randomised per-episode)")
        print(
            f"  View: {args.inspect_view}  |  " f"pause: {args.termination_pause:.1f}s"
        )
        print("  Press Ctrl+C to stop.")

    else:  # live
        sensors_cfg = dict(train_cfg.get("carla_sensors", {}))

        env = _build_env(
            args.host,
            args.port,
            args.layout,
            train_cfg,
            sensors_cfg=sensors_cfg,
            full_lot=True,  # Spawn lot so LiDAR has scene context
        )
        env.reset()
        if env.world is None or env.vehicle is None:
            print("ERROR: Could not connect to CARLA or spawn vehicle.")
            env.close()
            sys.exit(1)

        inspector = LiveInspector(env, args.duration, train_cfg)
        inspector.place_spectator()  # type: ignore[attr-defined]
        print("Live sensor mode:")
        print("  LiDAR hit points -> red debug dots in CARLA world.")

    try:
        inspector.run()
    finally:
        env.close()
        print("Done.")


if __name__ == "__main__":
    main()
