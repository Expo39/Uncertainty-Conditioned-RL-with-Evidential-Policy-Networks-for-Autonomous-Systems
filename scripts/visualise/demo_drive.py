"""
@file demo_drive.py
@brief Load a trained checkpoint and drive in CARLA for visual inspection.

No metrics, no condition sweep, no plots. Just loads the model and runs
deterministic episodes in a loop so you can watch the agent park. Works
with both the 2D bird's-eye visualiser (make visualise) and the 3D CARLA
spectator view (--render flag).

Usage:
    python scripts/visualise/demo_drive.py --checkpoint checkpoints/final_model
    python scripts/visualise/demo_drive.py --checkpoint checkpoints/final_model \
        --render
    python scripts/visualise/demo_drive.py --checkpoint checkpoints/final_model \
        --episodes 5
"""

import argparse
import os
from typing import Any, Dict, List, cast

import numpy as np
import torch as th
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from uncertainty_rl.envs import CARLAParkingEnv
from uncertainty_rl.networks.sb3_integration import EvidentialPPO


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
    return parser.parse_args()


def _make_env(env_config: Dict[str, Any]) -> DummyVecEnv:
    """
    @brief Create the CARLA parking environment from environment config.
    @param env_config: Parsed environment configuration dictionary.
    @return Vectorised environment.
    """

    def _init() -> CARLAParkingEnv:
        # Env vars override config (e.g. CARLA_HOST=carla-server-demo for 3D view)
        carla_host = os.environ.get(
            "CARLA_HOST", env_config.get("carla_host", "carla-server")
        )
        carla_port = int(
            os.environ.get("CARLA_PORT", env_config.get("carla_port", 2000))
        )
        return CARLAParkingEnv(
            carla_host=carla_host,
            carla_port=carla_port,
            town=env_config.get("town", "FlatPlane"),
            max_steps=env_config.get("max_steps", 500),
            ros2_config=env_config.get("ros2", {}),
            carla_sensors_config=env_config.get("carla_sensors", {}),
            parking_scenarios_config=env_config.get("parking_scenarios", {}),
            include_covariance=bool(env_config.get("include_covariance", True)),
            include_obstacle_obs=bool(env_config.get("include_obstacle_obs", True)),
            gnss_noise_profiles_path=env_config.get("gnss_noise_profiles", None),
        )

    return DummyVecEnv([_init])


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
    vec_normalize_path = os.path.join(
        os.path.dirname(args.checkpoint), "vec_normalize.pkl"
    )
    if os.path.exists(vec_normalize_path):
        env = VecNormalize.load(vec_normalize_path, base_env)
        env.training = False
        env.norm_reward = False
        print(f"Loaded normalisation stats from {vec_normalize_path}")

    is_evidential = isinstance(model, EvidentialPPO) and hasattr(
        model.policy, "get_action_with_uncertainty"
    )

    episode = 0
    print("Driving. Close the visualiser or Ctrl+C to stop.")

    try:
        while args.episodes == 0 or episode < args.episodes:
            obs = cast(np.ndarray, env.reset())
            done_arr = np.array([False])
            steps = 0
            episode += 1

            while not done_arr[0]:
                if is_evidential:
                    obs_tensor = th.as_tensor(obs)
                    policy = model.policy
                    get_action = policy.get_action_with_uncertainty
                    action_tensor, _ = get_action(obs_tensor, deterministic=True)
                    action = action_tensor.cpu().numpy()
                else:
                    action, _ = model.predict(obs, deterministic=True)

                step_result = env.step(action)
                obs = cast(np.ndarray, step_result[0])
                done_arr = cast(np.ndarray, step_result[2])
                # DummyVecEnv.step() returns (obs, rewards, dones, infos) - 4 elements.
                infos = cast(List[Dict[str, Any]], step_result[3])
                steps += 1

                if args.render:
                    env.render()

                if done_arr[0]:
                    success = infos[0].get("success", False)
                    result = "SUCCESS" if success else "FAIL"
                    print(f"  Episode {episode}: {result} ({steps} steps)")
                    break

    except KeyboardInterrupt:
        print(f"\nStopped after {episode} episodes.")
    finally:
        env.close()


if __name__ == "__main__":
    main()
