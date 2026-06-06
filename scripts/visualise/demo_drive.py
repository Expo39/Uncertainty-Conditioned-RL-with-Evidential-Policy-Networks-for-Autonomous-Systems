"""
@file demo_drive.py
@brief Load a trained checkpoint and drive in CARLA for visual inspection.
"""

import argparse
import csv
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, TextIO, cast

import numpy as np
import torch as th
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from uncertainty_rl.envs import make_env
from uncertainty_rl.networks.sb3_integration import EvidentialPPO
from uncertainty_rl.utils.constants import STRICT_BAY_MARGIN


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
        "decision is traced to outputs/demo_traces/<timestamp>/episode_<N>.csv "
        "(step, speed, pos / orientation error, delivered commands, reward, "
        "success, and the evidential policy uncertainty) for offline "
        "behaviour and calibration analysis.",
    )
    parser.set_defaults(realtime=True, trace=True)
    return parser.parse_args()


# Trace CSV column order. One row per POLICY DECISION (not per sim tick): with
# action_repeat > 1 the env returns an empty info dict on the intermediate
# repeat ticks, and those are skipped (see the step loop) so every logged row
# carries real state. Columns cover parking quality (speed, position and
# orientation error), the post-clamp commands actually delivered to CARLA, the
# total reward, the episode outcome, and the evidential policy uncertainty
# (epistemic and aleatoric, mean over action axes) - the calibration signal
# that is the point of the project.
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
]


def _make_env(env_config: Dict[str, Any]) -> DummyVecEnv:
    """
    @brief Create the CARLA parking environment from environment config.
    @param env_config: Parsed environment configuration dictionary.
    @return Vectorised environment.

    The success acceptance margin is read from the (stage-merged) env_config
    `bay_margin`, exactly as training does - so the demo scores the policy under
    the SAME criterion it was trained at. With --stage N the stage's looser
    margin applies (e.g. Stage 1's -0.75); with no stage the base env_config
    value (the strict -0.25) applies. No hardcoded margin, so the demo success
    cannot silently diverge from the training success metric.
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
                host_override=host_override,
                port_override=port_override,
            )
        ]
    )


def _write_run_info(trace_dir: Path, checkpoint: str, demo_stamp: str) -> None:
    """
    @brief Write provenance metadata for a demo trace run.

    Parses the seed and training-run start time from the checkpoint directory
    name (format <baseline>_seed<N>_<DDMMYYYY-HHMM>) and writes them, alongside
    the checkpoint path and the demo run timestamp, to run_info.txt in the
    trace directory. Fields that cannot be parsed are recorded as "unknown" so
    the file is always written.

    @param trace_dir: Directory where episode CSVs are written.
    @param checkpoint: Path to the loaded model checkpoint.
    @param demo_stamp: DD-MM-YYYY-HHMMSS timestamp of this demo run.
    @return None.
    """
    # The checkpoint path is e.g. checkpoints/<run_name>/final_model; the run
    # name is the parent directory.
    run_name = Path(checkpoint).parent.name

    seed = "unknown"
    run_start = "unknown"
    seed_match = re.search(r"seed(\d+)", run_name)
    if seed_match:
        seed = seed_match.group(1)
    # Training run start stamp is the trailing DDMMYYYY-HHMM block.
    stamp_match = re.search(r"(\d{8})-(\d{4})$", run_name)
    if stamp_match:
        d, t = stamp_match.group(1), stamp_match.group(2)
        # DDMMYYYY-HHMM -> DD-MM-YYYY HH:MM (European, human-readable).
        run_start = f"{d[0:2]}-{d[2:4]}-{d[4:8]} {t[0:2]}:{t[2:4]}"

    lines = [
        f"checkpoint: {checkpoint}",
        f"run_name: {run_name}",
        f"seed: {seed}",
        f"training_run_started: {run_start}",
        f"demo_run: {demo_stamp}",
    ]
    (trace_dir / "run_info.txt").write_text("\n".join(lines) + "\n")
    print(
        f"Run info written: {trace_dir}/run_info.txt (seed={seed}, "
        f"trained {run_start})"
    )


def main() -> None:
    """
    @brief Load checkpoint and run deterministic episodes in a loop.
    """
    args = _parse_args()

    # Load configs. load_env_config merges sensor/agent/env + the stage difficulty.
    # Difficulty lives only in the stage files and obs/policy flags only in the
    # baseline files, so the demo resolves the same defaults as training: stage 1
    # and full_method when the flags are omitted. The demo must match the geometry
    # AND obs shape the checkpoint was trained at - a mismatch is a hard shape error.
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
        f"'{baseline_override.get('baseline_name', baseline_path)}': "
        f"policy_type={train_config.get('policy_type')}, "
        f"include_covariance={env_config.get('include_covariance')}, "
        f"include_obstacle_obs={env_config.get('include_obstacle_obs')}"
    )

    # Load model: match the class used during training so the policy type is correct.
    print(f"Loading model from {args.checkpoint}...")
    policy_type = train_config.get("policy_type", "evidential")
    if policy_type == "evidential":
        model: PPO = EvidentialPPO.load(args.checkpoint)
    else:
        model = PPO.load(args.checkpoint)

    # Create environment
    base_env = _make_env(env_config)
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

    # Per-step trace logging. One timestamped folder per demo run, one CSV
    # per episode. Written under outputs/ (the rw-mounted volume) so the
    # traces survive the --rm container exit.
    trace_dir: Optional[Path] = None
    if args.trace:
        run_stamp = datetime.now().strftime("%d-%m-%Y-%H%M%S")
        trace_dir = Path("outputs") / "demo_traces" / run_stamp
        trace_dir.mkdir(parents=True, exist_ok=True)
        print(f"Trace logging enabled: {trace_dir}/episode_<N>.csv")
        # Provenance so the CSVs are not anonymous (checkpoint, seed, start
        # time parsed from the checkpoint directory name).
        _write_run_info(trace_dir, args.checkpoint, run_stamp)

    print("Driving. Close the visualiser or Ctrl+C to stop.")

    # Current episode's trace file handle. Hoisted out of the loop so the
    # finally block can close it if the run is interrupted mid-episode.
    trace_file: Optional[TextIO] = None

    try:
        while args.episodes == 0 or episode < args.episodes:
            obs = cast(np.ndarray, env.reset())
            done_arr = np.array([False])
            steps = 0
            episode += 1

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

                # With action_repeat > 1 the env returns its empty info dict on
                # the intermediate repeat ticks (no observation is built and no
                # reward is computed there); only the final tick of each repeat
                # carries real state. The VecEnv wrapper can inject its own keys
                # (terminal_observation, TimeLimit.truncated) into that dict, so
                # an emptiness test is unreliable - key on "pos_error", which the
                # env writes only on real steps. This logs one row per policy
                # decision and drops the all-zero filler rows that made earlier
                # traces 75% noise at action_repeat=4.
                info0 = infos[0]
                # The target bay id and spawn id are written per row in the
                # episode CSV (see _TRACE_COLUMNS), so run_info.txt no longer
                # carries per-episode bay info - it holds only run-level header.
                if trace_writer is not None and "pos_error" in info0:
                    # action is shape (1, 3): [steer, throttle, brake]. The env
                    # exposes the post-clamp commands actually delivered to
                    # CARLA (rate-limited and brake-overrides-throttle applied)
                    # as steer_cmd / throttle_cmd / brake_cmd - those, not the
                    # policy's raw pre-clamp output, describe what the car did.
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
                        ]
                    )

                if args.render:
                    env.render()

                # Pace the loop to wall-clock time so the drive is watchable.
                # CARLA runs in synchronous mode, where world.tick() advances
                # physics instantly - without this sleep the episode would
                # play back many times faster than real time. carla_timestep
                # (default 0.05s = 20 Hz) is the sim seconds one step covers.
                if args.realtime:
                    timestep = float(infos[0].get("carla_timestep", 0.05))
                    elapsed = time.monotonic() - step_start
                    remaining = timestep - elapsed
                    if remaining > 0.0:
                        time.sleep(remaining)

            if trace_file is not None:
                trace_file.close()

            success = infos[0].get("success", False)
            result = "SUCCESS" if success else "FAIL"
            print(f"  Episode {episode}: {result} ({steps} steps)")

    except KeyboardInterrupt:
        print(f"\nStopped after {episode} episodes.")
    finally:
        # Close the current episode's trace file if the run was interrupted
        # mid-episode (the normal per-episode close happens in the loop above).
        if trace_file is not None and not trace_file.closed:
            trace_file.close()
        env.close()


if __name__ == "__main__":
    main()
