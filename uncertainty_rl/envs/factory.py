"""
@file factory.py
@brief Single env construction path for training, evaluation, and inspection.

Centralises CARLAParkingEnv kwargs so the same dynamics (action_repeat,
actuator_model, max_ego_speed_ms, etc.) flow into every context. Callers
pass a merged config dict; this module owns the mapping from config keys
to constructor kwargs.
"""

from pathlib import Path
from typing import Any, Callable, Dict, Optional

import gymnasium as gym

from uncertainty_rl.envs.sim.carla_parking import CARLAParkingEnv


def make_env(
    config: Dict[str, Any],
    bay_margin: float,
    rank: int = 0,
    carla_sensors_override: Optional[Dict[str, Any]] = None,
    host_override: Optional[str] = None,
    port_override: Optional[int] = None,
    held_gnss_tier_override: Optional[str] = None,
    degrade_one_way_override: bool = False,
    degrade_rate_scale: float = 1.0,
) -> Callable[[], gym.Env]:
    """
    @brief Create a callable that returns a new environment instance.
    @param config: Configuration dictionary.
    @param bay_margin: Geometric success margin (metres) to use for this env.
           Required and explicit, no default: training/tuning callers pass the
           per-stage value merged in from a curriculum stage YAML; evaluation,
           demo, and inspector callers pass STRICT_BAY_MARGIN.
    @param rank: Environment rank for seeding.
    @param carla_sensors_override: Override sensor noise config (for evaluation).
    @param host_override: Override the per-worker CARLA host (e.g. the dryrun
           inspector, which connects to carla-server-demo); None uses the
           per-rank training host.
    @param port_override: Override the per-worker CARLA port; None uses the
           per-rank training port.
    @param held_gnss_tier_override: If set, locks GNSS noise to this fix-state
           tier for the whole episode, bypassing per-episode sampling and Markov
           drift, so evaluation can make GNSS level a controlled variable; None
           runs the training noise process.
    @param degrade_one_way_override: If True, the episode starts at rtk_fixed and
           the chain may only degrade, never recover (the monotone-degradation
           eval condition); has no effect when held_gnss_tier_override is set.
    @param degrade_rate_scale: Multiplier on the one-way chain's downward
           transition mass so the walk to the worst tier completes inside the
           episode horizon; 1.0 is the datasheet-anchored default everywhere
           except the drift condition. Ignored unless degrade_one_way_override.
    @return Callable that creates and returns a CARLAParkingEnv instance.
    """

    worker_port = (
        port_override
        if port_override is not None
        else config.get("carla_port", 2000) + rank * 1000
    )
    worker_host = (
        host_override if host_override is not None else f"uncertainty-rl-carla-{rank}"
    )

    ros2_config = config.get("ros2", {}).copy()
    if rank > 0:
        base_ekf = ros2_config.get(
            "ekf_state_file", "/workspace/outputs/ekf_state.json"
        )
        p = Path(base_ekf)
        ros2_config["ekf_state_file"] = str(p.parent / f"{p.stem}_{rank}{p.suffix}")

    vis_path: Optional[str] = None if rank == 0 else f"outputs/vis_history_{rank}.jsonl"

    def _init() -> gym.Env:
        env = CARLAParkingEnv(
            carla_host=worker_host,
            carla_port=worker_port,
            town=config.get("town", "FlatPlane"),
            max_steps=config.get("max_steps", 500),
            ros2_config=ros2_config,
            carla_sensors_config=(
                carla_sensors_override
                if carla_sensors_override is not None
                else config.get("carla_sensors", {})
            ),
            parking_scenarios_config=config.get("parking_scenarios", {}),
            include_covariance=config.get("include_covariance", True),
            include_obstacle_obs=config.get("include_obstacle_obs", True),
            carla_timestep=config.get("carla_timestep", 0.05),
            debug=config.get("debug", False),
            map_load_sleep=config.get("map_load_sleep", 5.0),
            action_repeat=config.get("action_repeat", 1),
            no_rendering_mode=config.get("no_rendering_mode", False),
            max_ego_speed_ms=config.get("max_ego_speed_ms", 8.0),
            use_extra_spawns=config.get("use_extra_spawns", False),
            gnss_noise_profiles_path=config.get("gnss_noise_profiles", None),
            held_gnss_tier_override=held_gnss_tier_override,
            degrade_one_way_override=degrade_one_way_override,
            degrade_rate_scale=degrade_rate_scale,
            vis_output_path=vis_path,
            bay_margin=bay_margin,
            actuator_model=config.get("actuator_model", None),
            # Only set for a windowed demo drive: "human" makes render() move the
            # CARLA spectator to follow the ego. Unset (None) in training and
            # evaluation, where render() is never called.
            render_mode=config.get("render_mode", None),
        )
        return env

    return _init
