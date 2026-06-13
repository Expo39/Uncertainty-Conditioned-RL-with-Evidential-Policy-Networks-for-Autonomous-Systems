"""
@file env_builder.py
@brief Maps an eval condition to a CARLA env (the condition -> env contract).

This is the single source of truth for how an eval_config.yaml condition becomes
a running environment: it scales the sensor noise, pins occupancy / floor plan /
GNSS tier, and builds the env through the SHARED training factory so evaluation
inherits the exact dynamics the policy trained under. build_eval_env_factory()
returns the bare (unwrapped) factory; make_eval_env() adds the SafetyWrapper +
DummyVecEnv the headless sweep needs. Kept separate from evaluate.py so the
eval-dryrun inspector and the sweep share one definition of the scenario.
"""

from typing import Any, Callable, Dict, Optional, Tuple

try:
    from stable_baselines3.common.vec_env import DummyVecEnv
except ImportError:
    DummyVecEnv = None  # type: ignore[assignment,misc]

try:
    from uncertainty_rl.envs import make_env
    from uncertainty_rl.envs.safety_wrapper import SafetyWrapper
except ImportError:
    make_env = None  # type: ignore[assignment,misc]
    SafetyWrapper = None  # type: ignore[assignment,misc]

from uncertainty_rl.utils.constants import STRICT_BAY_MARGIN


def _scale_sensor_noise(
    base_sensors: Dict[str, Any],
    imu_multiplier: float,
    lidar_multiplier: float = 1.0,
) -> Dict[str, Any]:
    """
    @brief Scale base sensor noise parameters by the condition-specific multipliers.
    @param base_sensors: Base sensor config from the merged env config.
    @param imu_multiplier: Multiplier for all IMU noise stddev values.
    @param lidar_multiplier: Multiplier for all LiDAR noise stddev values.
    @return New sensor config dict with scaled noise values.

    Only keys containing "stddev" are scaled (bias and tick-rate keys are not).
    The LiDAR noise `enabled` flag is forced on: every curriculum stage trains
    with LiDAR noise enabled (the constant realism floor), but the flag is owned
    by the stage files, which the stage-less eval config merge never applies -
    left untouched it would default to off and break train/eval parity.
    """
    base_imu: Dict[str, Any] = base_sensors.get("imu", {})
    scaled_imu = {
        k: (v * imu_multiplier if "stddev" in k else v) for k, v in base_imu.items()
    }
    scaled_lidar: Dict[str, Any] = dict(base_sensors.get("lidar", {}))
    base_lidar_noise: Dict[str, Any] = scaled_lidar.get("noise", {})
    scaled_lidar_noise = {
        k: (v * lidar_multiplier if "stddev" in k else v)
        for k, v in base_lidar_noise.items()
    }
    scaled_lidar_noise["enabled"] = True
    scaled_lidar["noise"] = scaled_lidar_noise
    return {**base_sensors, "imu": scaled_imu, "lidar": scaled_lidar}


def build_eval_env_factory(
    condition: Dict[str, Any],
    config: Dict[str, Any],
    base_sensors: Dict[str, Any],
    env_config: Optional[Dict[str, Any]] = None,
    host_override: Optional[str] = None,
    port_override: Optional[int] = None,
) -> Tuple[Callable[[], Any], float, float]:
    """
    @brief Build the raw (unwrapped) eval env factory for a physical condition.
    @param condition: Condition dict with noise multipliers and traffic counts.
    @param config: Evaluation configuration dictionary.
    @param base_sensors: Base sensor noise config from env_config.yaml.
    @param env_config: Environment config for parking_scenarios and obs flags.
    @param host_override: Override the CARLA host (e.g. carla-server-demo for
           the windowed inspect stack). None uses the per-rank default.
    @param port_override: Override the CARLA port. None uses the per-rank
           default.
    @return Tuple of (env_factory, aleatoric_scaling, handoff_threshold). The
            factory yields a bare CARLAParkingEnv built with the condition's
            scaled sensor noise, pinned occupancy/floor-plan/GNSS tier, and the
            exact training dynamics. The two SafetyWrapper parameters are
            returned so callers that DO want the wrapper (the headless eval
            sweep) can apply it, while the manual eval-dryrun inspector can use
            the bare env it needs to reach env.vehicle / env.world directly.

    @note This is the single source of truth for the condition -> env mapping;
          make_eval_env() wraps it in SafetyWrapper + DummyVecEnv for the sweep,
          and the eval-dryrun inspector consumes the bare factory so what you
          drive in is, by construction, the same scenario the sweep evaluates.
    """
    # Scale sensor noise by condition multipliers
    scaled_sensors = _scale_sensor_noise(
        base_sensors,
        imu_multiplier=condition.get("imu_noise_multiplier", 1.0),
        lidar_multiplier=condition.get("lidar_noise_multiplier", 1.0),
    )

    # Build parking_scenarios_config: NPC counts and lot layout from condition
    # overrides + training defaults. eval_config.yaml uses num_patrol_vehicles
    # (not num_vehicles) to match CARLAParkingEnv's parking_scenarios_config keys.
    base_scenarios: Dict[str, Any] = (
        dict(env_config.get("parking_scenarios", {})) if env_config is not None else {}
    )
    floor_plans: Dict[str, Any] = base_scenarios.get("floor_plans", {})

    # Perimeter cone flag: per-condition > eval_config global > env_config default.
    _cones_fallback = base_scenarios.get(
        "spawn_perimeter_cones", config.get("spawn_perimeter_cones", False)
    )
    spawn_cones: bool = condition.get("spawn_perimeter_cones", _cones_fallback)

    # In evaluation, occupancy is fixed per condition (min == max).
    # eval_config.yaml uses bay_occupancy_rate (a single value); training uses
    # bay_occupancy_min/max for the per-episode uniform resample range.
    bay_occupancy: float = condition.get(
        "bay_occupancy_rate", base_scenarios.get("bay_occupancy_max", 0.6)
    )

    # Dynamic actors (patrol vehicles, pedestrians) are out of scope: training
    # uses static parked cars only, so both default to zero and a condition
    # must opt in explicitly to deviate from the training distribution.
    parking_config: Dict[str, Any] = {
        "spawn_perimeter_cones": spawn_cones,
        "num_patrol_vehicles_max": condition.get("num_patrol_vehicles", 0),
        "pedestrian_spawn_probability": condition.get(
            "pedestrian_spawn_probability", 0.0
        ),
        "bay_occupancy_min": bay_occupancy,
        "bay_occupancy_max": bay_occupancy,
        "floor_plans": floor_plans,
        # Pin the lot per condition. Training pins fixed_floor_plan via the
        # stage files (rectangle in every stage), which the stage-less eval
        # merge never applies; without the pin the env SAMPLES among non-OOD
        # plans, leaking the held-out trapezoid into every rectangle condition.
        # Fixed selection also bypasses the ood eligibility filter, so the
        # irregular_a conditions are loadable.
        "fixed_floor_plan": condition.get("floor_plan", "rectangle"),
        # Episode START tier: match training, which since the Markov redesign
        # SAMPLES the start tier from the init weights in gnss_noise_profiles.yaml
        # (no curriculum stage sets fixed_gnss_tier) and lets the chain wander
        # from there. So the default here is None - the anchor runs the exact
        # training process (weighted random start + drift), not a forced clean
        # start, which would make eval easier than training. Honoured only if a
        # config still sets fixed_gnss_tier; irrelevant when held_gnss_tier
        # overrides (a held tier suppresses both the sampler and the drift).
        "fixed_gnss_tier": base_scenarios.get("fixed_gnss_tier", None),
    }

    # Env-specific settings resolved once from env_config.
    _ec = env_config if env_config is not None else {}

    # Held GNSS tier override: locks the named fix-state tier for this eval
    # condition (no Markov drift). None runs the training noise process.
    held_tier: Optional[str] = condition.get("held_gnss_tier", None)

    # SafetyWrapper parameters from agent_config.yaml (merged into env_config).
    aleatoric_scaling: float = float(_ec.get("safety_aleatoric_scaling", 0.5))
    handoff_threshold: float = float(_ec.get("safety_handoff_threshold", 5.0))

    # Build the env through the shared factory so evaluation inherits the EXACT
    # dynamics the policy was trained with - action_repeat (decision = N ticks),
    # max_ego_speed_ms (speed governor), carla_timestep, and the actuator slew /
    # brake model. Constructing CARLAParkingEnv directly here once silently
    # dropped these kwargs, leaving the policy stepped at the wrong rate with no
    # rate limiter and a too-high speed cap - it failed every episode. The
    # condition-scaled sensor noise overrides the base config, and the per-bay
    # / occupancy / tier pins from parking_config replace the env's scenarios.
    eval_config_dict: Dict[str, Any] = {
        **_ec,
        "parking_scenarios": parking_config,
        "debug": config.get("debug", False),
    }
    env_factory = make_env(
        eval_config_dict,
        bay_margin=STRICT_BAY_MARGIN,
        rank=0,
        carla_sensors_override=scaled_sensors,
        held_gnss_tier_override=held_tier,
        host_override=host_override,
        port_override=port_override,
    )
    return env_factory, aleatoric_scaling, handoff_threshold


def make_eval_env(
    condition: Dict[str, Any],
    config: Dict[str, Any],
    base_sensors: Dict[str, Any],
    env_config: Optional[Dict[str, Any]] = None,
) -> DummyVecEnv:
    """
    @brief Create the vectorised evaluation environment for a physical condition.
    @param condition: Condition dict with noise multipliers and traffic counts.
    @param config: Evaluation configuration dictionary.
    @param base_sensors: Base sensor noise config from env_config.yaml.
    @param env_config: Environment config for parking_scenarios and obs flags.
    @return Vectorised evaluation environment (SafetyWrapper + DummyVecEnv).
    """
    env_factory, aleatoric_scaling, handoff_threshold = build_eval_env_factory(
        condition,
        config,
        base_sensors,
        env_config,
    )

    def _init() -> Any:
        return SafetyWrapper(
            env_factory(),
            aleatoric_scaling=aleatoric_scaling,
            handoff_threshold=handoff_threshold,
        )

    env = DummyVecEnv([_init])
    return env
