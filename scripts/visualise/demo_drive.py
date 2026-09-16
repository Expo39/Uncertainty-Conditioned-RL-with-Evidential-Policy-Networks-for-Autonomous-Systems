"""
@file demo_drive.py
@brief Load a trained checkpoint and drive in CARLA for visual inspection.
"""

import argparse
import csv
import os
import re
import signal
import time
from datetime import datetime
from pathlib import Path
from types import FrameType
from typing import Any, Dict, List, Optional, TextIO, cast
from zoneinfo import ZoneInfo

import numpy as np
import torch as th
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from uncertainty_rl.envs import make_env
from uncertainty_rl.networks.sb3_integration import EvidentialPPO
from uncertainty_rl.utils.bay_success import BaySuccessTracker
from uncertainty_rl.utils.constants import OOB_INFLATION_MARGIN, STRICT_BAY_MARGIN


def _parse_args() -> argparse.Namespace:
    """
    @brief Parse CLI arguments for the demo driver.
    @return Parsed namespace.
    """
    parser = argparse.ArgumentParser(
        description="Load a checkpoint and drive in CARLA for visual inspection."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/final_model",
        help="Path to trained model checkpoint.",
    )
    parser.add_argument(
        "--env-config",
        type=str,
        default="configs/deployment/sim/env_config.yaml",
        help="Path to environment config (CARLA, sensors, parking scenarios).",
    )
    parser.add_argument(
        "--train-config",
        type=str,
        default="configs/train_config.yaml",
        help="Path to training config (for policy_type used in model loading).",
    )
    parser.add_argument(
        "--baseline",
        type=str,
        default=None,
        help=(
            "Path to a baseline override config (configs/baselines/*.yaml). When "
            "set, its keys (policy_type, include_covariance, include_obstacle_obs) "
            "overlay the train/env configs so the demo loads the policy class and "
            "builds the observation space the checkpoint was TRAINED with, rather "
            "than the train_config/agent_config defaults (the full method). Without "
            "this, evaluating a vanilla checkpoint against the evidential defaults "
            "mismatches both the loader class and the obs dimensionality. Mirrors "
            "the --baseline overlay in train_ppo.py."
        ),
    )
    parser.add_argument(
        "--stage",
        type=int,
        default=None,
        help="Curriculum stage (1..N). When set, deep-merges the stage env_config "
        "override so the demo evaluates the policy on the SAME difficulty it was "
        "trained at (e.g. Stage 1's single fixed bay + loose margin). Omit to use "
        "the base env_config difficulty.",
    )
    parser.add_argument(
        "--gnss-tier",
        type=str,
        default=None,
        choices=["fixed", "float", "standalone", "degraded"],
        help="Hold one RTK fix-state tier for the whole drive (bypasses the "
        "per-episode tier sampling and the Markov drift), so the localisation "
        "uncertainty is a fixed, controlled level. Maps to a tier in "
        "gnss_noise_profiles.yaml: fixed->rtk_fixed (~2 cm), float->rtk_float "
        "(~36 cm), standalone (~1.8 m), degraded (~5 m). Omit to run the normal "
        "training noise process (sampled start tier + Markov drift).",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=0,
        help="Number of episodes to run (0 = run indefinitely).",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Enable CARLA 3D spectator rendering (requires display).",
    )
    parser.add_argument(
        "--playback-speed",
        type=float,
        default=1.0,
        help="Wall-clock playback rate relative to simulated time (default 1.0 "
        "= real time). Below 1.0 slows the drive down, which is easier to "
        "follow in a recording; above 1.0 speeds it up. Ignored with "
        "--no-realtime.",
    )
    parser.add_argument(
        "--no-realtime",
        dest="realtime",
        action="store_false",
        help="Step as fast as CARLA allows instead of pacing to wall-clock "
        "time. By default the demo runs at real-time speed so the drive is "
        "watchable; pass this to run flat out.",
    )
    parser.add_argument(
        "--no-trace",
        dest="trace",
        action="store_false",
        help="Disable per-step CSV trace logging. By default each policy "
        "decision is traced to "
        "outputs/raw/demo_traces/<baseline>/<checkpoint_leaf>/<demo_stamp>/episode_<N>.csv "
        "(step, speed, pos / orientation error, delivered commands, reward, "
        "success, and the evidential policy uncertainty) for offline "
        "behaviour and calibration analysis.",
    )
    parser.set_defaults(realtime=True, trace=True)
    return parser.parse_args()


# Trace CSV column order. One row per POLICY DECISION: env.step() spans the
# full action_repeat tick window and returns one real transition per decision,
# so every logged row carries real state.
_TRACE_COLUMNS = [
    "step",
    # Episode routing, constant per episode but logged per row so each trace is
    # self-describing ("from spawn_id to bay_id") without a run_info lookup.
    "bay_id",
    "spawn_id",
    "speed_ms",
    "pos_error_m",
    "orientation_error_rad",
    "steer_cmd",
    "throttle_cmd",
    "brake_cmd",
    "reward",
    "success",
    "epistemic",
    "aleatoric",
    # Ground-truth world pose (CARLA) and EKF world pose (what the policy
    # sees). Logged side by side so a trace can be checked for EKF accuracy:
    # ekf_* should track gt_* within the EKF error budget. ekf_* are NaN on
    # steps where no genuine EKF estimate was available (CI/test fallback).
    "gt_x",
    "gt_y",
    "gt_yaw",
    "gt_vx",
    "gt_vyaw",
    "ekf_x",
    "ekf_y",
    "ekf_yaw",
    "ekf_vx",
    "ekf_vyaw",
    # EKF 1-sigma localisation stds (m, m, rad). Populated for every baseline
    # (the EKF always runs); NaN when no genuine EKF estimate was available.
    "ekf_std_x",
    "ekf_std_y",
    "ekf_std_yaw",
    # Live GNSS fix-state tier (info["gnss_tier"]) and its noise multiplier vs
    # rtk_fixed - the faithful tier label, since the reported ekf_std saturates
    # and cannot be used to recover the tier post hoc. Empty on steps with no
    # active tier (e.g. CI/test fallback).
    "gnss_tier",
    "gnss_multiplier",
]


# CARLA debug primitives persist until they expire, so the lifetime must NOT
# outlive the redraw interval by much - otherwise the previous episode's
# target bay stays painted green over the new one.
# Mirrors scripts/inspect/inspectors/base.py (_REDRAW_INTERVAL / _OVERLAY_LIFE).
_OVERLAY_REDRAW_S = 3.0
_OVERLAY_LIFETIME_S = 3.5

# Map the short --gnss-tier choice to the tier key in gnss_noise_profiles.yaml.
_GNSS_TIER_NAMES = {
    "fixed": "rtk_fixed",
    "float": "rtk_float",
    "standalone": "standalone",
    "degraded": "degraded",
}


def _make_env(
    env_config: Dict[str, Any], held_gnss_tier: Optional[str] = None
) -> DummyVecEnv:
    """
    @brief Create the CARLA parking environment from environment config.
    @param env_config: Parsed environment configuration dictionary.
    @param held_gnss_tier: If set, the gnss_noise_profiles.yaml tier name to hold
           for the whole drive (no per-episode sampling, no Markov drift). None
           runs the normal training noise process.
    @return Vectorised environment.

    Reads the success acceptance margin from the (stage-merged) env_config
    `bay_margin`, exactly as training does, so the demo scores the policy
    under the same criterion it was trained at.
    """
    # Env vars override config (e.g. CARLA_HOST=carla-server-demo for 3D view).
    host_override = os.environ.get("CARLA_HOST")
    port_override = (
        int(os.environ["CARLA_PORT"]) if "CARLA_PORT" in os.environ else None
    )
    bay_margin = float(env_config.get("bay_margin", STRICT_BAY_MARGIN))
    return DummyVecEnv(
        [
            make_env(
                env_config,
                bay_margin=bay_margin,
                rank=0,
                held_gnss_tier_override=held_gnss_tier,
                host_override=host_override,
                port_override=port_override,
            )
        ]
    )


def _inner_env(env: Any) -> Any:
    """
    @brief Unwrap the vectorised/normalised env down to CARLAParkingEnv.
    @param env: The env as held by the demo loop.
    @return The underlying CARLAParkingEnv instance.
    """
    inner = env.unwrapped.envs[0]
    while hasattr(inner, "env"):
        inner = inner.env
    return inner


def _set_tick_pacing(env: Any, seconds_per_tick: float) -> None:
    """
    @brief Ask the env to pace each simulation tick to wall-clock time.

    Only meaningful with a window open: it makes CARLA's synchronous-mode frames
    arrive evenly instead of in one burst per policy decision.

    @param env: The env as held by the demo loop.
    @param seconds_per_tick: Wall-clock budget for one tick.
    """
    try:
        _inner_env(env)._tick_wall_seconds = max(0.0, seconds_per_tick)
    except Exception as exc:  # noqa: BLE001 - pacing is cosmetic
        print(f"Tick pacing unavailable: {exc}")


def _draw_demo_overlays(env: Any) -> None:
    """
    @brief Draw the lot-geometry overlays into the CARLA window for one episode.

    Reuses the inspector's drawing helpers so the 3D demo shows the same bay
    outlines, target-bay highlight, spawn, patrol path and pedestrian zones the
    layout inspector does - without them the windowed view is bare tarmac and
    the viewer cannot tell which bay the car is aiming for.

    Failures are swallowed: the overlay is decoration, and a drawing error must
    never stop the drive.

    @param env: The vectorised env wrapping CARLAParkingEnv.
    """
    try:
        from scripts.inspect._drawing import _draw_layout_overlays

        inner = _inner_env(env)
        # Wipe the previous draw first. Primitives persist until they expire, so
        # without this the last episode's target bay stays highlighted green
        # until its lifetime runs out - over the top of the new target. Its own
        # try/except keeps a clearing failure from also skipping the draw below.
        if inner.world is not None:
            try:
                inner.world.debug.clear_debug_shape()
                inner.world.debug.clear_debug_string()
            except Exception as exc:  # noqa: BLE001 - stale overlays are cosmetic
                print(f"Overlay clear skipped: {exc}")
        layout = getattr(inner, "_current_layout", None)
        if not layout or inner.world is None:
            return
        _draw_layout_overlays(
            inner.world,
            layout,
            target_bay_id=inner._target_bay.get("bay_id", ""),
            life_time=_OVERLAY_LIFETIME_S,
            show_patrol=False,
            show_pedestrians=False,
            show_spawns=False,
            oob_inflation_margin=OOB_INFLATION_MARGIN,
        )
    except Exception as exc:  # noqa: BLE001 - decoration must never break the run
        print(f"Overlay drawing skipped: {exc}")


def _sigterm_to_keyboard_interrupt(signum: int, frame: Optional[FrameType]) -> None:
    """
    @brief Convert SIGTERM into KeyboardInterrupt for a graceful shutdown.
    @param signum: Received signal number (unused).
    @param frame: Interrupted stack frame (unused).

    The demo runs as the container's init process, where SIGTERM (what
    `docker stop` / `docker compose down` send) has no default disposition and
    the daemon escalates to SIGKILL after the grace period - skipping the
    finally block that flushes bay_successes.csv and closes the CARLA env.
    Raising KeyboardInterrupt routes a stop through the same graceful path as
    Ctrl+C.
    """
    raise KeyboardInterrupt


def main() -> None:
    """
    @brief Load checkpoint and run deterministic episodes in a loop.
    """
    args = _parse_args()

    # load_env_config merges sensor/agent/env + the stage difficulty; difficulty
    # lives only in the stage files and obs/policy flags only in the baseline
    # files, so the demo resolves the same defaults as training when the flags
    # are omitted. The demo must match the checkpoint's obs shape exactly.
    from uncertainty_rl.training.train_ppo import (
        DEFAULT_BASELINE,
        DEFAULT_STAGE,
        load_env_config,
    )
    from uncertainty_rl.utils.config_merge import apply_baseline

    stage = args.stage if args.stage is not None else DEFAULT_STAGE
    baseline_path = args.baseline if args.baseline is not None else DEFAULT_BASELINE

    env_config: Dict[str, Any] = load_env_config(args.env_config, stage=stage)
    with open(args.train_config, "r") as f:
        train_config: Dict[str, Any] = yaml.safe_load(f)

    # apply_baseline is the single overlay shared with train_ppo.py /
    # tune_hyperparams.py; policy_type drives loader-class selection below,
    # include_covariance / include_obstacle_obs drive what make_env builds, so the
    # baseline is applied to both dicts.
    with open(baseline_path, "r") as f:
        baseline_override: Dict[str, Any] = yaml.safe_load(f)
    apply_baseline(train_config, baseline_override)
    apply_baseline(env_config, baseline_override)
    print(
        f"Stage {stage}, baseline "
        f"'{baseline_override.get('baseline_name', Path(baseline_path).stem)}': "
        f"policy_type={train_config.get('policy_type')}, "
        f"include_covariance={env_config.get('include_covariance')}, "
        f"include_obstacle_obs={env_config.get('include_obstacle_obs')}"
    )

    # Load model: match the class used during training so the policy type is correct.
    # Show only the trailing <baseline>/<leaf>/file tail rather than the full path.
    _checkpoint_label = "/".join(Path(args.checkpoint).parts[-3:])
    print(f"Loading model from {_checkpoint_label}...")
    policy_type = train_config.get("policy_type", "evidential")
    if policy_type == "evidential":
        model: PPO = EvidentialPPO.load(args.checkpoint)
    else:
        model = PPO.load(args.checkpoint)

    # Create environment. With --gnss-tier, hold that fix-state tier for the
    # whole drive (no sampling / drift); otherwise run the normal training noise
    # process - the default behaviour.
    held_gnss_tier = (
        _GNSS_TIER_NAMES[args.gnss_tier] if args.gnss_tier is not None else None
    )
    if held_gnss_tier is not None:
        print(f"Holding GNSS tier '{held_gnss_tier}' for the whole drive.")
    # --render turns on the chase spectator inside CARLA's own window, which is
    # what the 3D demo captures. The env only moves the spectator when its
    # render_mode is "human", so set it here rather than leaving render() inert.
    if args.render:
        env_config["render_mode"] = "human"
    base_env = _make_env(env_config, held_gnss_tier=held_gnss_tier)
    env = base_env

    # Apply normalisation statistics if available
    vec_normalize_path = Path(args.checkpoint).parent / "vec_normalize.pkl"
    if vec_normalize_path.exists():
        env = VecNormalize.load(str(vec_normalize_path), base_env)
        env.training = False
        env.norm_reward = False
        print(f"Loaded normalisation stats from {vec_normalize_path!s}")

    is_evidential = isinstance(model, EvidentialPPO) and hasattr(
        model.policy, "get_action_with_uncertainty"
    )

    # Hoist evidential policy handles outside the step loop.
    _get_action = model.policy.get_action_with_uncertainty if is_evidential else None

    episode = 0

    # Output dirs mirror the checkpoint tree plus a per-demo-run level:
    # <baseline>/<checkpoint_leaf>/<demo_stamp>/, so repeated demos of the same
    # checkpoint group under one folder and never collide with each other.
    # Local wall-clock time (container runs UTC); Europe/Malta is DST-aware.
    run_stamp = datetime.now(ZoneInfo("Europe/Malta")).strftime("%d-%m-%Y-%H%M%S")
    _ckpt_leaf = Path(args.checkpoint).parent.name
    _ckpt_baseline = Path(args.checkpoint).parent.parent.name
    # A real training leaf ends in the <DDMMYYYY-HHMM> stamp - both the current
    # <stage>_<seed>_<stamp> and the older seed<N>_<stamp> forms match; trial_<N>
    # and bare paths do not, so they fall back to a flat <demo_stamp>/.
    if _ckpt_baseline and re.search(r"\d{8}-\d{4}$", _ckpt_leaf):
        _run_subtree = Path(_ckpt_baseline) / _ckpt_leaf / run_stamp
    else:
        _run_subtree = Path(run_stamp)

    # Per-step trace logging (one CSV per episode), enabled by default.
    trace_dir: Optional[Path] = None
    if args.trace:
        # Under outputs/raw/: traces are a raw artefact, and every consumer
        # (make trace-tier-breakdown, the ekf_sawtooth figure) reads them there.
        trace_dir = Path("outputs") / "raw" / "demo_traces" / _run_subtree
        trace_dir.mkdir(parents=True, exist_ok=True)
        print(f"Trace logging enabled: {trace_dir}/episode_<N>.csv")

    # Per-bay success accounting, mirroring the evaluate.py eval tree so any
    # visualiser-driven run (2D or 3D) leaves a bay_successes.csv. One tracker
    # for the whole run (the demo has no per-condition split), dumped in the
    # finally block so a Ctrl+C stop still flushes it.
    bay_tracker = BaySuccessTracker()
    bay_dir = Path("outputs") / "raw" / "bay_successes" / "eval" / _run_subtree

    # Installed only once the trackers exist, so a stop during setup (where
    # there is nothing to flush) keeps the daemon's plain kill behaviour.
    signal.signal(signal.SIGTERM, _sigterm_to_keyboard_interrupt)

    print("Driving. Close the visualiser or Ctrl+C to stop.")

    # Wall-clock seconds one env.step() should occupy at real-time playback.
    # A step spans action_repeat ticks, so the budget is the whole window, not
    # a single tick; PLAYBACK_SPEED > 1 runs proportionally faster.
    step_wall_seconds = (
        float(env_config.get("carla_timestep", 0.05))
        * max(1, int(env_config.get("action_repeat", 1)))
        / max(0.05, args.playback_speed)
    )
    # With the window open, let the ENV pace each tick so CARLA's frames arrive
    # evenly at the sim rate. Sleeping only once per decision here would bunch
    # action_repeat frames together and play as a slideshow.
    paced_in_env = args.realtime and args.render
    if paced_in_env:
        repeat = max(1, int(env_config.get("action_repeat", 1)))
        _set_tick_pacing(env, step_wall_seconds / repeat)
    if args.realtime:
        print(
            f"Real-time playback: {step_wall_seconds:.3f}s per decision "
            f"(speed x{args.playback_speed:g})."
        )

    # Current episode's trace file handle. Hoisted out of the loop so the
    # finally block can close it if the run is interrupted mid-episode.
    trace_file: Optional[TextIO] = None

    try:
        while args.episodes == 0 or episode < args.episodes:
            obs = cast(np.ndarray, env.reset())
            done_arr = np.array([False])
            steps = 0
            episode += 1

            # Draw the lot overlays for the new episode's layout and target bay.
            # Refreshed periodically inside the step loop below, so the short
            # primitive lifetime cannot leave the previous episode's target bay
            # highlighted once this episode picks a different one.
            next_overlay_redraw = 0.0
            if args.render:
                _draw_demo_overlays(env)
                next_overlay_redraw = time.monotonic() + _OVERLAY_REDRAW_S

            # Open a fresh per-episode trace CSV.
            trace_file = None
            trace_writer: Optional[Any] = None
            if trace_dir is not None:
                trace_file = open(trace_dir / f"episode_{episode}.csv", "w", newline="")
                trace_writer = csv.writer(trace_file)
                trace_writer.writerow(_TRACE_COLUMNS)

            while not done_arr[0]:
                # Wall-clock timestamp at the start of this step, used to pace
                # the loop to real-time when --realtime is set (the default).
                step_start = time.monotonic()

                # Per-decision evidential uncertainty (mean over action axes).
                # NaN for a non-evidential policy, which exposes no uncertainty.
                epistemic = float("nan")
                aleatoric = float("nan")
                if is_evidential and _get_action is not None:
                    # Move the observation onto the model's device: the model
                    # may load onto CUDA while th.as_tensor(obs) defaults to
                    # CPU, which crashes the dual-encoder actor's first matmul.
                    obs_tensor = th.as_tensor(obs).to(model.device)
                    action_tensor, unc = _get_action(obs_tensor, deterministic=True)
                    action = action_tensor.cpu().numpy()
                    # unc["epistemic"] / ["aleatoric"] are (1, ACTION_DIM); the
                    # mean over axes is the single per-step calibration scalar.
                    epistemic = float(unc["epistemic"].mean().item())
                    aleatoric = float(unc["aleatoric"].mean().item())
                else:
                    action, _ = model.predict(obs, deterministic=True)

                step_result = env.step(action)
                obs = cast(np.ndarray, step_result[0])
                rewards = cast(np.ndarray, step_result[1])
                done_arr = cast(np.ndarray, step_result[2])
                # DummyVecEnv.step() returns (obs, rewards, dones, infos) - 4 elements.
                infos = cast(List[Dict[str, Any]], step_result[3])
                steps += 1

                # The VecEnv wrapper can inject its own keys
                # (terminal_observation, TimeLimit.truncated) into the dict, so
                # key on "pos_error", which only the env's own info carries,
                # to log exactly one row per policy decision.
                info0 = infos[0]
                # Target bay id and spawn id are logged per row here; the
                # run_info.txt this run also writes carries only run-level
                # fields (checkpoint, demo_run, episodes_completed).
                if trace_writer is not None and "pos_error" in info0:
                    # steer_cmd / throttle_cmd / brake_cmd are the post-clamp
                    # commands actually delivered to CARLA (rate-limited,
                    # brake-overrides-throttle applied) - not the policy's raw
                    # pre-clamp output, so these describe what the car did.
                    trace_writer.writerow(
                        [
                            steps,
                            info0.get("target_bay_id", ""),
                            info0.get("spawn_id", ""),
                            f"{info0.get('speed', 0.0):.4f}",
                            f"{info0.get('pos_error', 0.0):.4f}",
                            f"{info0.get('orientation_error', 0.0):.4f}",
                            f"{info0.get('steer_cmd', 0.0):.4f}",
                            f"{info0.get('throttle_cmd', 0.0):.4f}",
                            f"{info0.get('brake_cmd', 0.0):.4f}",
                            f"{float(rewards[0]):.4f}",
                            int(bool(info0.get("success", False))),
                            f"{epistemic:.6f}",
                            f"{aleatoric:.6f}",
                            # GT then EKF world pose (NaN-safe formatting).
                            f"{info0.get('gt_x', float('nan')):.4f}",
                            f"{info0.get('gt_y', float('nan')):.4f}",
                            f"{info0.get('gt_yaw', float('nan')):.4f}",
                            f"{info0.get('gt_vx', float('nan')):.4f}",
                            f"{info0.get('gt_vyaw', float('nan')):.4f}",
                            f"{info0.get('ekf_x', float('nan')):.4f}",
                            f"{info0.get('ekf_y', float('nan')):.4f}",
                            f"{info0.get('ekf_yaw', float('nan')):.4f}",
                            f"{info0.get('ekf_vx', float('nan')):.4f}",
                            f"{info0.get('ekf_vyaw', float('nan')):.4f}",
                            f"{info0.get('ekf_std_x', float('nan')):.4f}",
                            f"{info0.get('ekf_std_y', float('nan')):.4f}",
                            f"{info0.get('ekf_std_yaw', float('nan')):.4f}",
                            # Live fix-state tier and its noise multiplier.
                            str(info0.get("gnss_tier", "")),
                            f"{info0.get('gnss_multiplier', float('nan')):.4f}",
                        ]
                    )

                if args.render:
                    env.render()
                    # Re-issue the overlays before they expire, so they persist
                    # for the whole episode without any one draw outliving it.
                    if time.monotonic() >= next_overlay_redraw:
                        _draw_demo_overlays(env)
                        next_overlay_redraw = time.monotonic() + _OVERLAY_REDRAW_S

                # Pace the loop to wall-clock time so the drive is watchable:
                # CARLA's synchronous mode advances physics instantly, so
                # without this sleep the episode plays back many times faster.
                # The budget is the whole action_repeat window, not one tick.
                if args.realtime and not paced_in_env:
                    elapsed = time.monotonic() - step_start
                    remaining = step_wall_seconds - elapsed
                    if remaining > 0.0:
                        time.sleep(remaining)

            if trace_file is not None:
                trace_file.close()

            # Record this terminated episode against its target bay. Read the bay
            # id from the nested target_bay dict, falling back to the flat
            # target_bay_id key (the same robust read as evaluate.py); an empty id
            # is ignored by record().
            terminal_info = infos[0].get("terminal_info", infos[0])
            success = bool(terminal_info.get("success", False))
            target_bay = terminal_info.get("target_bay", {})
            bay_id = target_bay.get("bay_id", "") or terminal_info.get(
                "target_bay_id", ""
            )
            bay_tracker.record(
                bay_id=str(bay_id),
                success=success,
                bay_type=str(target_bay.get("bay_type", "")),
            )

            result = "SUCCESS" if success else "FAIL"
            print(f"  Episode {episode}: {result} ({steps} steps)")

    except KeyboardInterrupt:
        print(f"\nStopped after {episode} episodes.")
    finally:
        # Close the current episode's trace file if the run was interrupted
        # mid-episode (the normal per-episode close happens in the loop above).
        if trace_file is not None and not trace_file.closed:
            trace_file.close()
        # Flush per-bay success counts. In the finally block so Ctrl+C (the normal
        # way to stop the demo) still writes whatever episodes completed. Skipped
        # when no episode terminated, so an immediately-killed run writes nothing.
        if bay_tracker.total_attempts > 0:
            bay_tracker.dump(
                bay_dir,
                run_info={
                    "checkpoint": args.checkpoint,
                    "demo_run": run_stamp,
                    "episodes_completed": bay_tracker.total_attempts,
                },
            )
            print(f"Bay successes written: {bay_dir}/bay_successes.csv")
        else:
            print(
                "No bay successes recorded (no episode completed, or the env "
                "info carried no target bay id)."
            )
        env.close()


if __name__ == "__main__":
    main()
