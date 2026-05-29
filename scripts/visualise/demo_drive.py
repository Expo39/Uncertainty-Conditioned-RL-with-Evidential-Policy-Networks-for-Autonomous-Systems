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
from uncertainty_rl.utils.constants import EVAL_BAY_MARGIN


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
        help="Disable per-step CSV trace logging. By default each episode is "
        "traced to logs/demo_runs/<timestamp>/episode_<N>.csv (step, speed, "
        "applied + raw throttle/brake/steer, pos error, reward) for offline "
        "behaviour analysis.",
    )
    parser.set_defaults(realtime=True, trace=True)
    return parser.parse_args()


# Trace CSV column order. Logged once per environment step.
_TRACE_COLUMNS = [
    "step",
    "speed_ms",
    "pos_error_m",
    "orientation_error_rad",
    "steer_applied",
    "throttle_applied",
    "brake_applied",
    "steer_raw",
    "throttle_raw",
    "brake_raw",
    "reward",
    "progress_reward",
    "approach_reward",
    "parked_bonus",
    "uncertainty_scale",
    "steer_cmd",
    "throttle_cmd",
    "brake_cmd",
]


def _make_env(env_config: Dict[str, Any]) -> DummyVecEnv:
    """
    @brief Create the CARLA parking environment from environment config.
    @param env_config: Parsed environment configuration dictionary.
    @return Vectorised environment.
    """
    # Env vars override config (e.g. CARLA_HOST=carla-server-demo for 3D view).
    host_override = os.environ.get("CARLA_HOST")
    port_override = (
        int(os.environ["CARLA_PORT"]) if "CARLA_PORT" in os.environ else None
    )
    return DummyVecEnv(
        [
            make_env(
                env_config,
                bay_margin=EVAL_BAY_MARGIN,
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

    # Load configs. load_env_config merges sensor_config.yaml (shared keys)
    # with env_config.yaml (CARLA-specific keys) into one unified dict.
    from uncertainty_rl.training.train_ppo import load_env_config

    env_config: Dict[str, Any] = load_env_config(args.env_config)
    with open(args.train_config, "r") as f:
        train_config: Dict[str, Any] = yaml.safe_load(f)

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

                if is_evidential and _get_action is not None:
                    # Move the observation onto the model's device: the model
                    # may load onto CUDA while th.as_tensor(obs) defaults to
                    # CPU, which crashes the dual-encoder actor's first matmul.
                    obs_tensor = th.as_tensor(obs).to(model.device)
                    action_tensor, _ = _get_action(obs_tensor, deterministic=True)
                    action = action_tensor.cpu().numpy()
                else:
                    action, _ = model.predict(obs, deterministic=True)

                step_result = env.step(action)
                obs = cast(np.ndarray, step_result[0])
                rewards = cast(np.ndarray, step_result[1])
                done_arr = cast(np.ndarray, step_result[2])
                # DummyVecEnv.step() returns (obs, rewards, dones, infos) - 4 elements.
                infos = cast(List[Dict[str, Any]], step_result[3])
                steps += 1

                if trace_writer is not None:
                    # action is shape (1, 3) from the vec env: [steer, throttle,
                    # brake]. raw = the policy's pre-clip output; applied = what
                    # the env actually sends to CARLA (steer clipped to [-1, 1],
                    # throttle and brake clipped to [0, 1]).
                    raw = np.asarray(action, dtype=np.float32).reshape(-1)
                    s_raw, t_raw, b_raw = (
                        float(raw[0]),
                        float(raw[1]),
                        float(raw[2]),
                    )
                    info0 = infos[0]
                    trace_writer.writerow(
                        [
                            steps,
                            f"{info0.get('speed', 0.0):.4f}",
                            f"{info0.get('pos_error', 0.0):.4f}",
                            f"{info0.get('orientation_error', 0.0):.4f}",
                            f"{float(np.clip(s_raw, -1.0, 1.0)):.4f}",
                            f"{float(np.clip(t_raw, 0.0, 1.0)):.4f}",
                            f"{float(np.clip(b_raw, 0.0, 1.0)):.4f}",
                            f"{s_raw:.4f}",
                            f"{t_raw:.4f}",
                            f"{b_raw:.4f}",
                            f"{float(rewards[0]):.4f}",
                            f"{info0.get('progress_reward', 0.0):.4f}",
                            f"{info0.get('approach_reward', 0.0):.4f}",
                            f"{info0.get('parked_bonus', 0.0):.4f}",
                            f"{info0.get('uncertainty_scale', 0.0):.4f}",
                            f"{info0.get('steer_cmd', 0.0):.4f}",
                            f"{info0.get('throttle_cmd', 0.0):.4f}",
                            f"{info0.get('brake_cmd', 0.0):.4f}",
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
