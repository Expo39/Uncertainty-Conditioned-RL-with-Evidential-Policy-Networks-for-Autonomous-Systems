"""
@file evaluate.py
@brief Evaluation script for trained agents across physical conditions.

This module provides comprehensive evaluation of trained agents under varying
physical conditions (sensor noise, traffic density) that produce different
EKF uncertainty levels.
"""

import argparse
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml

try:
    import torch as th
except ImportError:
    th = None  # type: ignore[assignment,misc]

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
except ImportError:
    PPO = None  # type: ignore[assignment,misc]
    DummyVecEnv = None  # type: ignore[assignment,misc]
    VecNormalize = None  # type: ignore[assignment,misc]

try:
    from uncertainty_rl.envs import CARLAParkingEnv
    from uncertainty_rl.envs.safety_wrapper import SafetyWrapper
    from uncertainty_rl.networks.sb3_integration import EvidentialPPO
except ImportError:
    CARLAParkingEnv = None  # type: ignore[assignment,misc]
    SafetyWrapper = None  # type: ignore[assignment,misc]
    EvidentialPPO = None  # type: ignore[assignment,misc]

from uncertainty_rl.utils.bay_success import BaySuccessTracker
from uncertainty_rl.utils.constants import STRICT_BAY_MARGIN

logger = logging.getLogger("uncertainty_rl.evaluation")


@dataclass
class EvaluationMetrics:
    """
    @class EvaluationMetrics
    @brief Container for evaluation metrics.

    Fields cover per-condition success rate, reward, step counts, final-state
    pose errors (one entry per episode, read from the terminal info), per-step
    evidential uncertainty estimates (evidential policy only), the per-episode
    outcome taxonomy counts, and the full per-episode records that downstream
    calibration analysis (uncertainty-vs-outcome, abort-threshold sweeps)
    consumes via episode_records.csv.
    """

    success_rate: float = 0.0
    average_reward: float = 0.0
    average_steps: float = 0.0
    position_errors: List[float] = field(default_factory=list)
    orientation_errors: List[float] = field(default_factory=list)
    epistemic_uncertainties: List[float] = field(default_factory=list)
    aleatoric_uncertainties: List[float] = field(default_factory=list)
    # Episodes per outcome class: success / collision / out_of_bounds /
    # handoff / near_miss / stuck (see _classify_outcome).
    outcome_counts: Dict[str, int] = field(default_factory=dict)
    # One dict per episode: outcome, final errors, lengths, uncertainty stats.
    episode_records: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """
        @brief Convert metrics to dictionary.
        @return Dictionary of metrics. Rates are percentages of episodes,
                matching the success_rate convention.
        """

        def _mean_std(lst: List[float]) -> Tuple[float, float]:
            if not lst:
                return 0.0, 0.0
            arr = np.asarray(lst)
            return float(arr.mean()), float(arr.std())

        pos_mean, pos_std = _mean_std(self.position_errors)
        ori_mean, ori_std = _mean_std(self.orientation_errors)
        epi_mean, _ = _mean_std(self.epistemic_uncertainties)
        ale_mean, _ = _mean_std(self.aleatoric_uncertainties)

        n_outcomes = sum(self.outcome_counts.values())

        def _rate(outcome: str) -> float:
            if n_outcomes == 0:
                return 0.0
            return self.outcome_counts.get(outcome, 0) / n_outcomes * 100.0

        return {
            "success_rate": self.success_rate,
            "average_reward": self.average_reward,
            "average_steps": self.average_steps,
            "mean_position_error": pos_mean,
            "std_position_error": pos_std,
            "mean_orientation_error": ori_mean,
            "std_orientation_error": ori_std,
            "mean_epistemic_uncertainty": epi_mean,
            "mean_aleatoric_uncertainty": ale_mean,
            "max_epistemic_uncertainty": (
                float(max(self.epistemic_uncertainties))
                if self.epistemic_uncertainties
                else 0.0
            ),
            "max_aleatoric_uncertainty": (
                float(max(self.aleatoric_uncertainties))
                if self.aleatoric_uncertainties
                else 0.0
            ),
            "collision_rate": _rate("collision"),
            "out_of_bounds_rate": _rate("out_of_bounds"),
            "handoff_rate": _rate("handoff"),
            "near_miss_rate": _rate("near_miss"),
            "stuck_rate": _rate("stuck"),
        }


def _classify_outcome(
    terminal_info: Dict[str, Any],
    near_miss_threshold: float,
) -> str:
    """
    @brief Classify a terminated episode into the failure-mode taxonomy.
    @param terminal_info: Terminal step info dict (post-wrapper, pre-reset).
    @param near_miss_threshold: Final position error (m) below which a
           timed-out episode counts as a near miss rather than stuck.
    @return One of: "success", "collision", "out_of_bounds", "handoff",
            "near_miss", "stuck".

    Priority: a collision or run-off is reported as such even if a safety
    handoff fired on the same step - the physical outcome outranks the
    intervention. Handoff (SafetyWrapper truncation on high epistemic
    uncertainty) is its own class: the vehicle stopped deliberately, which
    the safety analysis must not conflate with a blocked or imprecise park.
    Remaining timeouts split on the final position error: close misses are
    precision shortfalls, far ones blocked or abandoned approaches.
    """
    if terminal_info.get("success", False):
        return "success"
    if terminal_info.get("collision", False):
        return "collision"
    if terminal_info.get("oob", False):
        return "out_of_bounds"
    if terminal_info.get("safety_handoff", False):
        return "handoff"
    if float(terminal_info.get("pos_error", float("inf"))) < near_miss_threshold:
        return "near_miss"
    return "stuck"


def _scale_sensor_noise(
    base_sensors: Dict[str, Any],
    imu_multiplier: float,
) -> Dict[str, Any]:
    """
    @brief Scale base sensor noise parameters by the condition-specific IMU multiplier.
    @param base_sensors: Base sensor config from train_config.yaml.
    @param imu_multiplier: Multiplier for all IMU noise stddev values.
    @return New sensor config dict with scaled noise values.
    """
    base_imu: Dict[str, Any] = base_sensors.get("imu", {})
    scaled_imu = {
        k: (v * imu_multiplier if "stddev" in k else v) for k, v in base_imu.items()
    }
    return {**base_sensors, "imu": scaled_imu}


def make_eval_env(
    condition: Dict[str, Any],
    config: Dict[str, Any],
    base_sensors: Dict[str, Any],
    env_config: Optional[Dict[str, Any]] = None,
) -> DummyVecEnv:
    """
    @brief Create evaluation environment for a specific physical condition.
    @param condition: Condition dict with noise multipliers and traffic counts.
    @param config: Evaluation configuration dictionary.
    @param base_sensors: Base sensor noise config from env_config.yaml.
    @param env_config: Environment config for parking_scenarios and obs flags.
    @return Vectorised evaluation environment.
    """
    # Scale sensor noise by condition multipliers
    scaled_sensors = _scale_sensor_noise(
        base_sensors,
        imu_multiplier=condition.get("imu_noise_multiplier", 1.0),
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
    }
    # Allow per-condition floor plan override (e.g. OOD evaluation)
    if "floor_plan" in condition:
        floor_plan_name: str = condition["floor_plan"]
        if floor_plan_name in floor_plans:
            parking_config["floor_plans"] = {
                floor_plan_name: floor_plans[floor_plan_name]
            }

    # Observation flags and env-specific settings resolved once from env_config
    _ec = env_config if env_config is not None else {}
    include_covariance: bool = _ec.get("include_covariance", True)
    include_obstacle_obs: bool = _ec.get("include_obstacle_obs", True)
    use_extra_spawns: bool = _ec.get("use_extra_spawns", False)
    gnss_profiles_path: Optional[str] = _ec.get("gnss_noise_profiles", None)

    # GNSS noise multiplier override: locks tier for this eval condition.
    gnss_override: Optional[float] = condition.get("gnss_noise_multiplier", None)

    # debug: per-step DebugLogger diagnostics - off by default, same as training.
    debug: bool = config.get("debug", False)

    # SafetyWrapper parameters from agent_config.yaml (merged into env_config).
    aleatoric_scaling: float = float(_ec.get("safety_aleatoric_scaling", 0.5))
    handoff_threshold: float = float(_ec.get("safety_handoff_threshold", 5.0))

    # Connection / timing / ROS 2 come from env_config so evaluation runs the same
    # environment the policy was trained in (single source of truth - eval_config
    # holds only the condition sweep, not env settings).
    def _init() -> Any:
        base_env: Any = CARLAParkingEnv(
            carla_host=_ec.get("carla_host", "localhost"),
            carla_port=_ec.get("carla_port", 2000),
            town=_ec.get("town", "FlatPlane"),
            max_steps=_ec.get("max_steps", 500),
            ros2_config=_ec.get("ros2", {}),
            carla_sensors_config=scaled_sensors,
            parking_scenarios_config=parking_config,
            include_covariance=include_covariance,
            include_obstacle_obs=include_obstacle_obs,
            use_extra_spawns=use_extra_spawns,
            gnss_noise_profiles_path=gnss_profiles_path,
            gnss_noise_multiplier_override=gnss_override,
            # Evaluation is judged at the strict published criterion, not the
            # env default (0.0) or any relaxed training/curriculum margin.
            bay_margin=STRICT_BAY_MARGIN,
            debug=debug,
        )
        return SafetyWrapper(
            base_env,
            aleatoric_scaling=aleatoric_scaling,
            handoff_threshold=handoff_threshold,
        )

    env = DummyVecEnv([_init])
    return env


def evaluate_agent(
    model: PPO,
    env: Union[DummyVecEnv, VecNormalize],
    n_episodes: int = 100,
    deterministic: bool = True,
    render: bool = False,
    bay_tracker: Optional[BaySuccessTracker] = None,
    near_miss_threshold: float = 1.5,
) -> EvaluationMetrics:
    """
    @brief Evaluate agent performance.
    @param model: Trained PPO model.
    @param env: Evaluation environment.
    @param n_episodes: Number of evaluation episodes.
    @param deterministic: Use deterministic actions.
    @param render: Render episodes.
    @param bay_tracker: Optional per-bay success tracker. When supplied, each
           terminated episode is recorded against its target bay so the caller
           can dump a per-bay success CSV across the whole condition sweep.
    @param near_miss_threshold: Final position error (m) separating near_miss
           from stuck for timed-out episodes (see _classify_outcome).
    @return EvaluationMetrics object with results.

    @note For EvidentialPPO models, uses get_action_with_uncertainty() to
          collect per-step epistemic and aleatoric uncertainty estimates.
          Success is determined from the environment's info dict (set by
          CARLAParkingEnv.step()) rather than re-computing from final state.
          EKF localisation stds are read from the per-step info (the EKF runs
          for every baseline), so the uncertainty-gating analysis covers the
          no-covariance arms too.
    """
    metrics = EvaluationMetrics()

    # Per-episode uncertainty accumulators, cleared at each reset. step_fn
    # appends to these as well as the flat per-condition lists in metrics so
    # the per-episode record can report mean/max without re-slicing.
    ep_epistemic: List[float] = []
    ep_aleatoric: List[float] = []
    # Action-distribution std per decision: sqrt(aleatoric) for the evidential
    # head (its predicted outcome variance IS the sampling variance).
    ep_action_std: List[float] = []

    episode_rewards = np.empty(n_episodes, dtype=np.float64)
    episode_steps = np.empty(n_episodes, dtype=np.int32)
    success_flags = np.zeros(n_episodes, dtype=bool)

    # Detect evidential policy once; hoist method references out of the loop.
    is_evidential = isinstance(model, EvidentialPPO) and hasattr(
        model.policy, "get_action_with_uncertainty"
    )

    # Bind a single step function to eliminate the per-step branch.
    _StepReturn = Tuple[np.ndarray, float, np.ndarray, List[Dict[str, Any]]]
    step_fn: Callable[[np.ndarray], _StepReturn]
    if is_evidential:
        _get_action_with_uncertainty = (
            model.policy.get_action_with_uncertainty  # type: ignore[union-attr]
        )
        _set_uncertainty = env.env_method
        _policy_device = model.policy.device

        def step_fn(obs: np.ndarray) -> _StepReturn:  # type: ignore[misc]
            obs_tensor = th.as_tensor(obs, device=_policy_device)
            action_tensor, uncertainty_dict = _get_action_with_uncertainty(
                obs_tensor, deterministic=deterministic
            )
            action = action_tensor.cpu().numpy()
            epistemic = float(uncertainty_dict["epistemic"].mean().item())
            aleatoric = float(uncertainty_dict["aleatoric"].mean().item())
            metrics.epistemic_uncertainties.append(epistemic)
            metrics.aleatoric_uncertainties.append(aleatoric)
            ep_epistemic.append(epistemic)
            ep_aleatoric.append(aleatoric)
            ep_action_std.append(float(np.sqrt(aleatoric)))
            _set_uncertainty("set_uncertainty", epistemic, aleatoric)
            next_obs, reward, done, infos = env.step(action)
            return next_obs, float(reward[0]), done, infos

        const_action_std = float("nan")

    else:

        def step_fn(obs: np.ndarray) -> _StepReturn:  # type: ignore[misc]
            action, _states = model.predict(obs, deterministic=deterministic)
            next_obs, reward, done, infos = env.step(action)
            return next_obs, float(reward[0]), done, infos

        # The standard Gaussian policy's only confidence signal: exp(log_std),
        # a learned state-INDEPENDENT parameter in SB3 PPO. Recorded per
        # episode so the calibration analysis can show in-data that it carries
        # no per-state information (a constant column), in contrast to the
        # evidential head's state-conditional uncertainty.
        _log_std = getattr(getattr(model, "policy", None), "log_std", None)
        const_action_std = (
            float(_log_std.detach().exp().mean().item())
            if _log_std is not None
            else float("nan")
        )

    def _mean_max(values: List[float]) -> Tuple[float, float]:
        if not values:
            return float("nan"), float("nan")
        return float(np.mean(values)), float(np.max(values))

    for episode in range(n_episodes):
        obs: np.ndarray = env.reset()
        episode_reward = 0.0
        steps = 0
        done = np.array([False])
        ep_epistemic.clear()
        ep_aleatoric.clear()
        ep_action_std.clear()
        # EKF localisation stds (m / rad) at each decision, read from the
        # step info - populated for every baseline because the EKF always
        # runs; include_covariance only controls whether the policy SEES them.
        ep_std_pos: List[float] = []
        ep_std_yaw: List[float] = []

        while not done[0]:
            obs, step_reward, done, infos = step_fn(obs)
            episode_reward += step_reward
            steps += 1
            _info0 = infos[0]
            _std_x = float(_info0.get("ekf_std_x", float("nan")))
            _std_y = float(_info0.get("ekf_std_y", float("nan")))
            _std_yaw = float(_info0.get("ekf_std_yaw", float("nan")))
            if np.isfinite(_std_x) and np.isfinite(_std_y):
                ep_std_pos.append((_std_x + _std_y) / 2.0)
            if np.isfinite(_std_yaw):
                ep_std_yaw.append(_std_yaw)
            if done[0]:
                # DummyVecEnv.step() returns (obs, rewards, dones, infos) - 4 elements.
                # On the terminal step DummyVecEnv has already auto-reset the
                # wrapped env, so read the terminal info from terminal_info when
                # present (Gymnasium auto-reset stashes the pre-reset info there)
                # and fall back to the live info dict otherwise.
                terminal_info = infos[0].get("terminal_info", infos[0])
                episode_success = bool(terminal_info.get("success", False))
                success_flags[episode] = episode_success
                outcome = _classify_outcome(terminal_info, near_miss_threshold)
                metrics.outcome_counts[outcome] = (
                    metrics.outcome_counts.get(outcome, 0) + 1
                )
                # Final-state pose errors: one entry per episode.
                final_pos_error = float(terminal_info.get("pos_error", float("nan")))
                final_ori_error = float(
                    terminal_info.get("orientation_error", float("nan"))
                )
                metrics.position_errors.append(final_pos_error)
                metrics.orientation_errors.append(final_ori_error)
                # The bay id lives in the nested target_bay dict; fall back to
                # the flat target_bay_id key so recording survives any wrapper
                # that drops the nested dict. An empty id is ignored by record().
                target_bay = terminal_info.get("target_bay", {})
                bay_id = target_bay.get("bay_id", "") or terminal_info.get(
                    "target_bay_id", ""
                )
                if bay_tracker is not None:
                    bay_tracker.record(
                        bay_id=str(bay_id),
                        success=episode_success,
                        bay_type=str(target_bay.get("bay_type", "")),
                    )
                epi_mean, epi_max = _mean_max(ep_epistemic)
                ale_mean, ale_max = _mean_max(ep_aleatoric)
                std_pos_mean, std_pos_max = _mean_max(ep_std_pos)
                std_yaw_mean, _ = _mean_max(ep_std_yaw)
                # Action std: state-conditional sqrt(aleatoric) for the
                # evidential head; the constant exp(log_std) for the standard
                # Gaussian policy (its only confidence measure).
                if is_evidential:
                    act_std_mean, act_std_max = _mean_max(ep_action_std)
                else:
                    act_std_mean = act_std_max = const_action_std
                metrics.episode_records.append(
                    {
                        "episode": episode + 1,
                        "bay_id": str(bay_id),
                        "spawn_id": terminal_info.get("spawn_id", ""),
                        "outcome": outcome,
                        "success": int(episode_success),
                        "steps": steps,
                        "reward": episode_reward,
                        "final_pos_error_m": final_pos_error,
                        "final_orientation_error_rad": final_ori_error,
                        "final_speed_ms": float(
                            terminal_info.get("speed", float("nan"))
                        ),
                        "mean_epistemic": epi_mean,
                        "max_epistemic": epi_max,
                        "mean_aleatoric": ale_mean,
                        "max_aleatoric": ale_max,
                        "mean_action_std": act_std_mean,
                        "max_action_std": act_std_max,
                        "ekf_std_pos_mean_m": std_pos_mean,
                        "ekf_std_pos_max_m": std_pos_max,
                        "ekf_std_yaw_mean_rad": std_yaw_mean,
                    }
                )
                if render:
                    env.render()

        episode_rewards[episode] = episode_reward
        episode_steps[episode] = steps

    metrics.success_rate = float(success_flags.sum()) / n_episodes * 100.0
    metrics.average_reward = float(episode_rewards.mean())
    metrics.average_steps = float(episode_steps.mean())

    return metrics


def evaluate_across_conditions(
    model_path: str,
    eval_config_path: str,
    env_config_path: str,
    train_config_path: str,
    n_episodes: int = 0,
    output_dir: str = "./evaluation_results",
    baseline_path: Optional[str] = None,
) -> Tuple[pd.DataFrame, str]:
    """
    @brief Evaluate agent across different physical conditions.
    @param model_path: Path to trained model.
    @param eval_config_path: Path to evaluation configuration file.
    @param env_config_path: Path to environment config (sensors, parking scenarios).
    @param train_config_path: Path to training config (shared hyperparameters).
    @param n_episodes: Episodes per condition. 0 means read from eval_config
        (n_episodes key), falling back to 100.
    @param output_dir: Root results directory; this run writes into the
        <baseline>/<leaf> subdirectory underneath it.
    @param baseline_path: Baseline config naming the evaluated ablation cell. Its
        include_covariance / include_obstacle_obs / policy_type drive the obs shape
        and model class. None defaults to the full method (DEFAULT_BASELINE).
    @return Tuple of (results DataFrame, resolved run output directory).
    """
    from uncertainty_rl.training.train_ppo import (
        DEFAULT_BASELINE,
        load_config,
        load_env_config,
    )
    from uncertainty_rl.utils.config_merge import apply_baseline

    # Load configurations
    with open(eval_config_path, "r") as f:
        eval_config: Dict[str, Any] = yaml.safe_load(f)

    # load_env_config merges sensor_config.yaml (shared keys) with env_config.yaml
    # (CARLA-specific keys) so env_config is the single unified config for the env.
    env_config: Dict[str, Any] = load_env_config(env_config_path)

    # Obs flags and policy_type live only in the baseline files. Overlay the
    # evaluated baseline (full method by default) so the eval env builds the obs
    # shape and loads the policy class the checkpoint was trained as.
    baseline_cfg = load_config(baseline_path or DEFAULT_BASELINE)
    apply_baseline(env_config, baseline_cfg)

    base_sensors = env_config.get("carla_sensors", {})
    conditions = eval_config.get("eval_conditions", [])
    # n_episodes: caller can override; fall back to eval_config, then hard default.
    n_episodes = n_episodes or int(eval_config.get("n_episodes", 100))
    # Timeout episodes closer than this (m) to the bay are near_miss, else stuck.
    near_miss_threshold = float(eval_config.get("near_miss_threshold_m", 1.5))

    # Load model
    logger.info("Loading model from %s...", "/".join(Path(model_path).parts[-3:]))
    # Load model: use EvidentialPPO when the baseline specifies policy_type=evidential
    # so that isinstance(model, EvidentialPPO) is True and uncertainty is collected.
    policy_type = baseline_cfg.get("policy_type", "evidential")
    if policy_type == "evidential":
        model: PPO = EvidentialPPO.load(model_path)
    else:
        model = PPO.load(model_path)

    # Load normalisation statistics if available
    vec_normalize_path = os.path.join(os.path.dirname(model_path), "vec_normalize.pkl")

    results = []
    episode_rows: List[Dict[str, Any]] = []
    vec_normalize_exists = os.path.exists(vec_normalize_path)
    if not vec_normalize_exists:
        logger.warning(
            "No VecNormalize stats found at %s. "
            "Running without observation normalisation.",
            vec_normalize_path,
        )
    deterministic: bool = eval_config.get("deterministic", True)

    # Per-bay success accounting. Each condition gets its own tracker dumped to
    # outputs/bay_successes/eval/<baseline>/<leaf>/<condition>/, mirroring the
    # training tree, because a bay's success at RTK-fixed and RTK-degraded are
    # distinct questions and must not be conflated. The leaf is the checkpoint's
    # parent directory name (seed<N>_<timestamp>); the baseline comes from the
    # evaluated baseline config, falling back to the checkpoint's grandparent so
    # the path is unambiguous even for ad-hoc checkpoints.
    _eval_leaf = Path(model_path).parent.name or datetime.now().strftime(
        "%d-%m-%Y-%H%M%S"
    )
    _eval_baseline = baseline_cfg.get(
        "baseline_name", Path(model_path).parent.parent.name
    )
    _eval_run_name = f"{_eval_baseline}/{_eval_leaf}"
    _bay_eval_root = Path("./outputs/bay_successes/eval") / _eval_baseline / _eval_leaf
    # Results nest by <baseline>/<leaf>, mirroring checkpoints/logs/bay
    # successes, so successive ablation arms and seeds never overwrite each
    # other's evaluation_results.csv / episode_records.csv.
    run_output_dir = os.path.join(output_dir, _eval_baseline, _eval_leaf)

    for condition in conditions:
        name = condition.get("name", "unknown")
        description = condition.get("description", "")
        logger.info("Evaluating condition: %s - %s", name, description)
        bay_tracker = BaySuccessTracker()

        # Create environment for this condition
        base_env = make_eval_env(condition, eval_config, base_sensors, env_config)
        eval_env: Union[DummyVecEnv, VecNormalize] = base_env

        # Apply normalisation if available
        if vec_normalize_exists:
            eval_env = VecNormalize.load(vec_normalize_path, base_env)
            eval_env.training = False
            eval_env.norm_reward = False

        # Evaluate
        metrics = evaluate_agent(
            model=model,
            env=eval_env,
            n_episodes=n_episodes,
            deterministic=deterministic,
            bay_tracker=bay_tracker,
            near_miss_threshold=near_miss_threshold,
        )

        # Dump this condition's per-bay success counts. The GNSS field reports
        # "markov" when no multiplier override is set, i.e. the condition runs
        # the training noise process (tier sampling + drift).
        bay_tracker.dump(
            _bay_eval_root / name,
            run_info={
                "run_name": _eval_run_name,
                "model_path": model_path,
                "condition": name,
                "description": description,
                "n_episodes": n_episodes,
                "gnss_noise_multiplier": condition.get(
                    "gnss_noise_multiplier", "markov"
                ),
                "evaluated": datetime.now().strftime("%d-%m-%Y %H:%M"),
            },
        )

        # Accumulate per-episode records across the sweep (one tidy table).
        for record in metrics.episode_records:
            episode_rows.append({"condition": name, **record})

        # Store results: merge metrics dict with condition metadata in one pass
        optional_fields = (
            {"floor_plan": condition["floor_plan"]} if "floor_plan" in condition else {}
        )
        result = {
            **metrics.to_dict(),
            "condition": name,
            "description": description,
            # NaN marks the in-distribution Markov condition (no fixed level).
            "gnss_noise_multiplier": condition.get(
                "gnss_noise_multiplier", float("nan")
            ),
            "imu_noise_multiplier": condition.get("imu_noise_multiplier", 1.0),
            "num_patrol_vehicles": condition.get("num_patrol_vehicles", 0),
            "pedestrian_spawn_probability": condition.get(
                "pedestrian_spawn_probability", 0.0
            ),
            "bay_occupancy_rate": condition.get("bay_occupancy_rate", float("nan")),
            **optional_fields,
        }
        results.append(result)

        logger.info(
            "  success_rate=%.1f%%  handoff=%.1f%%  collision=%.1f%%  "
            "near_miss=%.1f%%  stuck=%.1f%%  avg_steps=%.0f",
            metrics.success_rate,
            result["handoff_rate"],
            result["collision_rate"],
            result["near_miss_rate"],
            result["stuck_rate"],
            metrics.average_steps,
        )

        # Clean up
        eval_env.close()

    # Create DataFrame
    df = pd.DataFrame(results)

    # Save results
    os.makedirs(run_output_dir, exist_ok=True)
    csv_path = os.path.join(run_output_dir, "evaluation_results.csv")
    df.to_csv(csv_path, index=False)
    logger.info("Results saved to %s", csv_path)

    # Per-episode records: the raw table behind every aggregate, and the input
    # to the calibration analysis (uncertainty-vs-outcome, abort thresholds).
    episodes_df = pd.DataFrame(episode_rows)
    episodes_csv_path = os.path.join(run_output_dir, "episode_records.csv")
    episodes_df.to_csv(episodes_csv_path, index=False)
    logger.info("Per-episode records saved to %s", episodes_csv_path)

    return df, run_output_dir


def plot_evaluation_results(
    df: pd.DataFrame,
    output_dir: str = "./evaluation_results",
) -> None:
    """
    @brief Create visualisations of evaluation results across conditions.
    @param df: DataFrame with evaluation results.
    @param output_dir: Directory to save plots.
    """
    sns.set_style("whitegrid")

    conditions = df["condition"].tolist()
    x_arr = np.arange(len(conditions))
    x_list = x_arr.tolist()

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # Plots 1-3: single-series bar charts, data-driven
    bar_specs = [
        (
            axes[0, 0],
            "success_rate",
            "steelblue",
            "Success Rate (%)",
            "Success Rate vs Condition",
        ),
        (
            axes[0, 1],
            "average_reward",
            "forestgreen",
            "Average Reward",
            "Average Reward vs Condition",
        ),
        (
            axes[1, 0],
            "average_steps",
            "firebrick",
            "Average Steps",
            "Average Steps to Termination vs Condition",
        ),
    ]
    for ax, col, colour, ylabel, title in bar_specs:
        ax.bar(x_list, df[col], color=colour, alpha=0.8)
        ax.set_xticks(x_list)
        ax.set_xticklabels(conditions, rotation=45, ha="right")
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.grid(True, alpha=0.3, axis="y")

    # Plot 4: Policy uncertainty estimates (grouped bars)
    if "mean_epistemic_uncertainty" in df.columns:
        bar_width = 0.35
        axes[1, 1].bar(
            x_arr - bar_width / 2,
            df["mean_epistemic_uncertainty"],
            bar_width,
            label="Epistemic",
            alpha=0.8,
        )
        axes[1, 1].bar(
            x_arr + bar_width / 2,
            df["mean_aleatoric_uncertainty"],
            bar_width,
            label="Aleatoric",
            alpha=0.8,
        )
        axes[1, 1].set_xticks(x_list)
        axes[1, 1].set_xticklabels(conditions, rotation=45, ha="right")
        axes[1, 1].set_ylabel("Uncertainty", fontsize=12)
        axes[1, 1].set_title("Policy Uncertainty Estimates", fontsize=14)
        axes[1, 1].legend(fontsize=10)
        axes[1, 1].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()

    plot_path = os.path.join(output_dir, "evaluation_plots.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    logger.info("Plots saved to %s", plot_path)

    plt.close()

    # Failure-mode breakdown: stacked outcome shares per condition. Success at
    # the base, then the failure taxonomy - the "degrades gracefully" figure.
    outcome_specs = [
        ("success_rate", "Success", "steelblue"),
        ("near_miss_rate", "Near miss", "gold"),
        ("stuck_rate", "Stuck", "darkorange"),
        ("handoff_rate", "Handoff", "slategrey"),
        ("out_of_bounds_rate", "Out of bounds", "sienna"),
        ("collision_rate", "Collision", "firebrick"),
    ]
    present = [s for s in outcome_specs if s[0] in df.columns]
    if present:
        fig2, ax2 = plt.subplots(figsize=(12, 6))
        bottom = np.zeros(len(conditions))
        for col, label, colour in present:
            values = df[col].to_numpy(dtype=float)
            ax2.bar(x_list, values, bottom=bottom, label=label, color=colour)
            bottom += values
        ax2.set_xticks(x_list)
        ax2.set_xticklabels(conditions, rotation=45, ha="right")
        ax2.set_ylabel("Share of Episodes (%)", fontsize=12)
        ax2.set_title("Episode Outcomes vs Condition", fontsize=14)
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        outcome_path = os.path.join(output_dir, "failure_modes.png")
        plt.savefig(outcome_path, dpi=300, bbox_inches="tight")
        logger.info("Failure-mode plot saved to %s", outcome_path)
        plt.close()


def main() -> None:
    """
    @brief Main entry point for evaluation script.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate trained agent across physical conditions"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to trained model",
    )
    parser.add_argument(
        "--eval-config",
        type=str,
        default="configs/eval_config.yaml",
        help="Path to evaluation configuration file",
    )
    parser.add_argument(
        "--env-config",
        type=str,
        default="configs/deployment/sim/env_config.yaml",
        help="Path to environment config (sensors, parking scenarios)",
    )
    parser.add_argument(
        "--train-config",
        type=str,
        default="configs/train_config.yaml",
        help="Path to training config (shared hyperparameters)",
    )
    parser.add_argument(
        "--baseline",
        type=str,
        default=None,
        help=(
            "Baseline config naming the evaluated ablation cell "
            "(configs/baselines/*.yaml). Its include_covariance / "
            "include_obstacle_obs / policy_type drive the obs shape and model "
            "class. Omit to evaluate the full method."
        ),
    )
    parser.add_argument(
        "--n-episodes",
        type=int,
        default=0,
        help="Episodes per condition (0 = read from eval_config.yaml)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./evaluation_results",
        help="Directory for output files",
    )

    args = parser.parse_args()

    # Configure logging level: DEBUG when debug:true in eval_config, else INFO.
    # This also enables the env's DebugLogger per-step output.
    with open(args.eval_config, "r") as _f:
        _cfg: Dict[str, Any] = yaml.safe_load(_f)
    _log_level = logging.DEBUG if _cfg.get("debug", False) else logging.INFO
    logging.basicConfig(
        level=_log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Run evaluation. Results land in <output-dir>/<baseline>/<leaf>/.
    df, run_output_dir = evaluate_across_conditions(
        model_path=args.model_path,
        eval_config_path=args.eval_config,
        env_config_path=args.env_config,
        train_config_path=args.train_config,
        n_episodes=args.n_episodes,
        output_dir=args.output_dir,
        baseline_path=args.baseline,
    )

    # Create plots alongside the CSVs.
    plot_evaluation_results(df, output_dir=run_output_dir)

    logger.info("Evaluation complete.")


if __name__ == "__main__":
    main()
