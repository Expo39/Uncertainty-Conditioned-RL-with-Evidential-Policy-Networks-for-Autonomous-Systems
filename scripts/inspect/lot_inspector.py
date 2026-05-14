"""
@file lot_inspector.py
@brief Entry point for the unified CARLA parking lot and sensor inspector.
"""

import argparse
import os
import random
import sys
import warnings
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
        choices=["third_person", "side", "back", "front", "free"],
        dest="inspect_view",
        help=(
            "Spectator view for dryrun mode (default: third_person).  "
            "'free' places the spectator overhead once and then does not move "
            "it, so you can fly around with CARLA's own controls.  "
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
    args = parser.parse_args()

    from uncertainty_rl.training.train_ppo import load_env_config

    train_cfg = load_env_config("configs/deployment/sim/env_config.yaml")

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
        print("  blue=perpendicular bays | yellow=angled bays | violet=parallel bays")
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
        # build the env via train_ppo.make_env() so
        # dryrun is, by construction, identical to a training rollout (same
        # constructor kwargs, same defaults).
        from uncertainty_rl.training.train_ppo import make_env

        dryrun_cfg = dict(train_cfg)
        dryrun_cfg["no_rendering_mode"] = False

        # Randomise layout based on INSPECT_OOD flag.
        # INSPECT_OOD=true: sample from OOD layouts only.
        # INSPECT_OOD=false (default): sample from training layouts only.
        inspect_ood = os.environ.get("INSPECT_OOD", "false").lower() == "true"

        scenarios = dict(dryrun_cfg.get("parking_scenarios", {}))
        original_floor_plans = scenarios.get("floor_plans", {})

        # Filter floor plans by OOD flag, matching load_floor_plan() logic.
        eligible_plans = {}
        for name, cfg in original_floor_plans.items():
            is_ood = cfg.get("ood", False)
            if inspect_ood:
                # OOD mode: include OOD plans only
                if is_ood:
                    # Strip ood flag so the env's eval_mode filter accepts it.
                    eligible_plans[name] = {k: v for k, v in cfg.items() if k != "ood"}
            else:
                # Training mode: exclude OOD plans
                if not is_ood:
                    eligible_plans[name] = cfg

        if not eligible_plans:
            raise RuntimeError(
                f"No eligible floor plans found. "
                f"INSPECT_OOD={inspect_ood}, available plans: {list(original_floor_plans.keys())}"
            )

        # Pass ALL eligible plans to the environment (do not pin to one).
        # The environment's load_floor_plan() will randomise per-episode.
        scenarios["floor_plans"] = eligible_plans
        dryrun_cfg["parking_scenarios"] = scenarios

        env = make_env(
            dryrun_cfg,
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
        print(f"Dry-run mode: full training pipeline, action={action_desc}, no model.")
        print(f"  Layouts ({layout_mode}): {list(eligible_plans.keys())} (randomised per-episode)")
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
