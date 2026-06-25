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
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
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
    from uncertainty_rl.networks.sb3_integration import EvidentialPPO
except ImportError:
    EvidentialPPO = None  # type: ignore[assignment,misc]

# The metric schema, condition -> env contract, and plotting now live in
# dedicated modules; re-exported here so existing import paths
# (uncertainty_rl.evaluation.evaluate.*) keep working unchanged.
from uncertainty_rl.evaluation.env_builder import (  # noqa: F401
    _scale_sensor_noise,
    build_eval_env_factory,
    make_eval_env,
)
from uncertainty_rl.evaluation.metrics import (  # noqa: F401
    EvaluationMetrics,
    _classify_outcome,
)
from uncertainty_rl.evaluation.plots import plot_evaluation_results  # noqa: F401
from uncertainty_rl.utils.bay_success import BaySuccessTracker
from uncertainty_rl.utils.constants import SUCCESS_THRESHOLD_VELOCITY

logger = logging.getLogger("uncertainty_rl.evaluation")

# Run-identifier timestamps use local wall-clock time (the container runs UTC);
# Europe/Malta is DST-aware (UTC+2 summer, UTC+1 winter).
_LOCAL_TZ = ZoneInfo("Europe/Malta")

# GNSS fix-state tier whose onset marks "degraded" for handover-latency timing.
# Matches the worst tier in configs/deployment/sim/gnss_noise_profiles.yaml; the
# step info reports the live tier name (env step() "gnss_tier").
DEGRADED_TIER_NAME = "degraded"


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
    # head - the TRUE predicted outcome variance (confidence signal), un-floored,
    # so it can fall below the training-time sampling-std floor.
    ep_action_std: List[float] = []
    # Optional capture of real (normalised) observations for the on-manifold
    # covariance probe (scripts/evaluation/covariance_probe.py --real-obs).
    # Off by default; enabled by the EVAL_DUMP_OBS env var (a positive integer
    # cap on how many observations to keep). Captured across all conditions so
    # the probe sees the full eval-state distribution.
    _obs_cap = int(os.environ.get("EVAL_DUMP_OBS", "0") or "0")
    captured_obs: List[np.ndarray] = []
    # Per-step EKF calibration pairs (predicted std vs actual GT-EKF error),
    # collected for every baseline so the calibration analysis covers the
    # no-covariance arms too. Always on (cheap; one small dict per step).
    calibration_pairs: List[Dict[str, Any]] = []
    # Per-step uncertainty trace for evidential heads. The per-step gating
    # analysis needs epistemic[t]/aleatoric[t] at matched states (e.g. the first
    # few steps, before policies diverge) to test instantaneous response rather
    # than the time-averaged episode aggregate. EVAL_PER_STEP_CAP bounds how many
    # leading steps per episode are kept (0 disables); the early steps are the
    # informative ones for novelty, so capping the head of each episode keeps the
    # file small without losing the matched-state window.
    per_step_records: List[Dict[str, Any]] = []
    _per_step_cap = int(os.environ.get("EVAL_PER_STEP_CAP", "0") or "0")

    episode_rewards = np.empty(n_episodes, dtype=np.float64)
    episode_steps = np.empty(n_episodes, dtype=np.int32)
    success_flags = np.zeros(n_episodes, dtype=bool)

    # Detect evidential policy once; hoist method references out of the loop.
    # EvidentialPPO is None when SB3 is unavailable (host/CI without torch), so
    # guard the isinstance against the None sentinel - isinstance(x, None) raises.
    is_evidential = (
        EvidentialPPO is not None
        and isinstance(model, EvidentialPPO)
        and hasattr(model.policy, "get_action_with_uncertainty")
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
        # The wrapper consumes set_uncertainty() to modulate actions; the bare env
        # (EVAL_DISABLE_SAFETY_WRAPPER) has no such method, so feeding it is a no-op
        # there. Probe once rather than try/except every step.
        _has_set_uncertainty = hasattr(env.envs[0], "set_uncertainty")

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
            if _has_set_uncertainty:
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
        # Handover-timing trace. handoff_step = first decision the SafetyWrapper
        # triggered a full-stop handoff; degraded_onset_step = first decision the
        # GNSS tier reached the degraded multiplier (the drift crossing, per
        # episode). Both are step indices, NaN if the event never occurred; the
        # latency and its onset regime (spawn vs mid-episode switch) are derived
        # downstream from the condition, which this loop does not know.
        handoff_step: float = float("nan")
        degraded_onset_step: float = float("nan")
        # Caution-behaviour trace: per-step speed (while MOVING, so the terminal
        # park-stop does not drag the mean to zero), yaw-rate magnitude, brake
        # command, and action jerk (||cmd_t - cmd_{t-1}||). Aggregated per episode
        # and binned against EKF std downstream to test whether the policy drives
        # more cautiously as localisation uncertainty rises.
        ep_speed_moving: List[float] = []
        ep_abs_vyaw: List[float] = []
        ep_brake: List[float] = []
        ep_action_jerk: List[float] = []
        prev_cmd: Optional[np.ndarray] = None

        while not done[0]:
            if _obs_cap and len(captured_obs) < _obs_cap:
                # obs is the (1, obs_dim) batch the policy is about to act on.
                captured_obs.append(np.asarray(obs, dtype=np.float32)[0].copy())
            obs, step_reward, done, infos = step_fn(obs)
            episode_reward += step_reward
            steps += 1
            _info0 = infos[0]
            _std_x = float(_info0.get("ekf_std_x", float("nan")))
            _std_y = float(_info0.get("ekf_std_y", float("nan")))
            _std_yaw = float(_info0.get("ekf_std_yaw", float("nan")))
            # Calibration pairs: the EKF's PREDICTED uncertainty (std) against its
            # ACTUAL error (ground truth minus EKF estimate). gt_* / ekf_* are in
            # the info for every baseline (GT is reward-only, never observed), so
            # this measures whether the covariance fed to the policy is honest -
            # the precondition for conditioning on it being justified at all.
            _gt_x = float(_info0.get("gt_x", float("nan")))
            _gt_y = float(_info0.get("gt_y", float("nan")))
            _gt_yaw = float(_info0.get("gt_yaw", float("nan")))
            _ekf_x = float(_info0.get("ekf_x", float("nan")))
            _ekf_y = float(_info0.get("ekf_y", float("nan")))
            _ekf_yaw = float(_info0.get("ekf_yaw", float("nan")))
            if np.isfinite(_gt_x) and np.isfinite(_ekf_x) and np.isfinite(_std_x):
                err_x = _gt_x - _ekf_x
                err_y = _gt_y - _ekf_y
                # Wrap heading error to [-pi, pi] before taking magnitude.
                err_yaw = (_gt_yaw - _ekf_yaw + np.pi) % (2.0 * np.pi) - np.pi
                calibration_pairs.append(
                    {
                        "condition": "",  # filled by the caller per condition
                        "std_x": _std_x,
                        "std_y": _std_y,
                        "std_yaw": _std_yaw,
                        "abs_err_x": abs(err_x),
                        "abs_err_y": abs(err_y),
                        "abs_err_pos": float(np.hypot(err_x, err_y)),
                        "abs_err_yaw": abs(float(err_yaw)),
                    }
                )
            if np.isfinite(_std_x) and np.isfinite(_std_y):
                ep_std_pos.append((_std_x + _std_y) / 2.0)
            if np.isfinite(_std_yaw):
                ep_std_yaw.append(_std_yaw)
            # Handover-timing capture: first handoff step and first step the GNSS
            # tier reaches DEGRADED_TIER_NAME. Recorded as step indices so the
            # analysis can compute latency against the right onset reference.
            if np.isnan(handoff_step) and bool(_info0.get("safety_handoff", False)):
                handoff_step = float(steps)
            if (
                np.isnan(degraded_onset_step)
                and str(_info0.get("gnss_tier", "")) == DEGRADED_TIER_NAME
            ):
                degraded_onset_step = float(steps)
            # Caution-behaviour per step. Speed is kept only while moving (above
            # the success velocity floor) so the terminal park-stop does not bias
            # the mean toward zero. Jerk is the change in the post-clamp command
            # actually delivered to CARLA (steer/throttle/brake).
            _speed = float(_info0.get("speed", float("nan")))
            if np.isfinite(_speed) and _speed >= SUCCESS_THRESHOLD_VELOCITY:
                ep_speed_moving.append(_speed)
            _vyaw = float(_info0.get("ekf_vyaw", float("nan")))
            if np.isfinite(_vyaw):
                ep_abs_vyaw.append(abs(_vyaw))
            _cmd = np.array(
                [
                    float(_info0.get("steer_cmd", float("nan"))),
                    float(_info0.get("throttle_cmd", float("nan"))),
                    float(_info0.get("brake_cmd", float("nan"))),
                ],
                dtype=np.float32,
            )
            if np.all(np.isfinite(_cmd)):
                ep_brake.append(float(_cmd[2]))
                if prev_cmd is not None:
                    ep_action_jerk.append(float(np.linalg.norm(_cmd - prev_cmd)))
                prev_cmd = _cmd
            # Per-step uncertainty trace (evidential heads only). step_fn has just
            # appended this decision's epistemic/aleatoric to the episode lists, so
            # read them off the tail. Keep only the leading steps per episode so a
            # full sweep stays a few MB, not hundreds.
            if _per_step_cap and steps <= _per_step_cap:
                # Behaviour + EKF columns per step (every arm) so caution-vs-
                # covariance is readable WITHIN an episode. Standard heads have no
                # state uncertainty: epistemic/aleatoric NaN, action_std the
                # constant exp(log_std). abs_err_pos_m is the GT-EKF position error.
                _err_pos = (
                    float(np.hypot(_gt_x - _ekf_x, _gt_y - _ekf_y))
                    if np.isfinite(_gt_x) and np.isfinite(_ekf_x)
                    else float("nan")
                )
                per_step_records.append(
                    {
                        "condition": "",  # filled by the caller per condition
                        "episode": episode + 1,
                        "step": steps,
                        "epistemic": (
                            ep_epistemic[-1] if is_evidential else float("nan")
                        ),
                        "aleatoric": (
                            ep_aleatoric[-1] if is_evidential else float("nan")
                        ),
                        "action_std": (
                            ep_action_std[-1] if is_evidential else const_action_std
                        ),
                        "ekf_std_pos_m": (_std_x + _std_y) / 2.0,
                        "ekf_std_yaw_rad": _std_yaw,
                        "speed_ms": _speed,
                        "abs_yaw_rate_rads": (
                            abs(_vyaw) if np.isfinite(_vyaw) else float("nan")
                        ),
                        "throttle_cmd": float(_cmd[1]),
                        "brake_cmd": float(_cmd[2]),
                        "abs_err_pos_m": _err_pos,
                    }
                )
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
                        # Handover timing (step indices, NaN if never). The onset
                        # regime and latency are derived per condition downstream.
                        "handoff_step": handoff_step,
                        "degraded_onset_step": degraded_onset_step,
                        # Caution behaviour (per-episode means). Binned against
                        # EKF std downstream to test drive-carefully-when-unsure:
                        # speed/jerk/vyaw should FALL and brake RISE as std grows.
                        "mean_speed_moving_ms": _mean_max(ep_speed_moving)[0],
                        "mean_abs_vyaw_rads": _mean_max(ep_abs_vyaw)[0],
                        "mean_brake_cmd": _mean_max(ep_brake)[0],
                        "mean_action_jerk": _mean_max(ep_action_jerk)[0],
                        "n_moving_steps": len(ep_speed_moving),
                    }
                )
                if render:
                    env.render()

        episode_rewards[episode] = episode_reward
        episode_steps[episode] = steps

    metrics.success_rate = float(success_flags.sum()) / n_episodes * 100.0
    metrics.average_reward = float(episode_rewards.mean())
    metrics.average_steps = float(episode_steps.mean())
    if captured_obs:
        metrics.captured_observations = captured_obs
    if calibration_pairs:
        metrics.calibration_pairs = calibration_pairs
    if per_step_records:
        metrics.per_step_records = per_step_records

    return metrics


def evaluate_across_conditions(
    model_path: str,
    eval_config_path: str,
    env_config_path: str,
    train_config_path: str,
    n_episodes: int = 0,
    output_dir: str = "./evaluation_results",
    baseline_path: Optional[str] = None,
    condition_names: Optional[List[str]] = None,
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
    @param condition_names: If given, restrict the sweep to the conditions whose
        name is in this list (order follows eval_config). None or empty evaluates
        every condition in eval_config.
    @return Tuple of (results DataFrame, resolved run output directory).
    @warning Raises ValueError if any requested condition name is absent from
        eval_config, so a typo fails loudly rather than silently evaluating nothing.
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

    # Optional subset selection: keep only the named conditions, preserving the
    # eval_config order. A requested name with no match is a configuration error
    # (likely a typo), so fail loudly rather than silently sweep nothing.
    if condition_names:
        requested = set(condition_names)
        available = {c.get("name", "unknown") for c in conditions}
        missing = requested - available
        if missing:
            raise ValueError(
                "Unknown eval condition(s) %s; available: %s"
                % (sorted(missing), sorted(available))
            )
        conditions = [c for c in conditions if c.get("name") in requested]
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
    # Real observations captured across the whole sweep when EVAL_DUMP_OBS is set
    # (the on-manifold covariance-probe input). Empty otherwise.
    captured_obs_all: List[Any] = []
    # EKF calibration pairs (predicted std vs actual error) across the sweep,
    # tagged by condition - the honesty-of-the-covariance table.
    calibration_rows: List[Dict[str, Any]] = []
    # Per-step uncertainty trace across the sweep, tagged by condition - the
    # per-step gating table (EVAL_PER_STEP_CAP leading steps per episode).
    per_step_rows: List[Dict[str, Any]] = []
    vec_normalize_exists = os.path.exists(vec_normalize_path)
    if not vec_normalize_exists:
        logger.warning(
            "No VecNormalize stats found at %s. "
            "Running without observation normalisation.",
            vec_normalize_path,
        )
    deterministic: bool = eval_config.get("deterministic", True)

    # Per-bay success accounting. Each condition gets its own tracker dumped to
    # outputs/bay_successes/eval/seed_<N>/<baseline>/<leaf>/<condition>/, mirroring
    # the training tree, because a bay's success at RTK-fixed and RTK-degraded are
    # distinct questions and must not be conflated. The leaf is the checkpoint's
    # parent directory name (<stage>_<seed>_<timestamp>); the baseline comes from
    # the evaluated baseline config, falling back to the checkpoint's grandparent
    # so the path is unambiguous even for ad-hoc checkpoints. The seed_<N> segment
    # is parsed from the leaf (single source of truth) so a second seed's bay
    # successes never overwrite the first's; an unparseable leaf falls back to
    # seed_unknown.
    _eval_leaf = Path(model_path).parent.name or datetime.now(_LOCAL_TZ).strftime(
        "%d-%m-%Y-%H%M%S"
    )
    _eval_baseline = baseline_cfg.get(
        "baseline_name", Path(model_path).parent.parent.name
    )
    _eval_run_name = f"{_eval_baseline}/{_eval_leaf}"
    _leaf_fields = _eval_leaf.split("_")
    _eval_seed = _leaf_fields[1] if len(_leaf_fields) >= 2 else "unknown"
    _bay_eval_root = (
        Path("./outputs/bay_successes/eval")
        / f"seed_{_eval_seed}"
        / _eval_baseline
        / _eval_leaf
    )
    # Results nest by <baseline>/<leaf>/<wrapper_variant>, mirroring checkpoints/
    # logs/bay successes. The with_wrapper/without_wrapper leaf keeps the two
    # SafetyWrapper variants of the SAME checkpoint side by side for the A/B.
    _wrapper_variant = (
        "without_wrapper"
        if bool(int(os.environ.get("EVAL_DISABLE_SAFETY_WRAPPER", "0") or "0"))
        else "with_wrapper"
    )
    run_output_dir = os.path.join(
        output_dir, _eval_baseline, _eval_leaf, _wrapper_variant
    )

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
        # "markov" when no held-tier override is set, i.e. the condition runs
        # the training noise process (tier sampling + drift).
        bay_tracker.dump(
            _bay_eval_root / name,
            run_info={
                "run_name": _eval_run_name,
                "model_path": model_path,
                "condition": name,
                "description": description,
                "n_episodes": n_episodes,
                "held_gnss_tier": condition.get("held_gnss_tier", "markov"),
                "evaluated": datetime.now(_LOCAL_TZ).strftime("%d-%m-%Y %H:%M"),
            },
        )

        # Accumulate per-episode records across the sweep (one tidy table).
        for record in metrics.episode_records:
            episode_rows.append({"condition": name, **record})

        # Accumulate any captured real observations (EVAL_DUMP_OBS) across all
        # conditions so the on-manifold covariance probe sees the full eval-state
        # distribution, not one condition's slice.
        if metrics.captured_observations:
            captured_obs_all.extend(metrics.captured_observations)

        # Accumulate EKF calibration pairs, tagging each with this condition so
        # the analysis can split honesty by GNSS tier.
        for pair in metrics.calibration_pairs:
            calibration_rows.append({**pair, "condition": name})

        # Accumulate the per-step uncertainty trace, tagged by condition, so the
        # per-step gating analysis can compare epistemic[t] across conditions at
        # matched steps.
        for rec in metrics.per_step_records:
            per_step_rows.append({**rec, "condition": name})

        # Store results: merge metrics dict with condition metadata in one pass
        optional_fields = (
            {"floor_plan": condition["floor_plan"]} if "floor_plan" in condition else {}
        )
        result = {
            **metrics.to_dict(),
            "condition": name,
            "description": description,
            # "markov" marks the in-distribution condition (no held tier - the
            # training noise process of tier sampling + Markov drift runs).
            "held_gnss_tier": condition.get("held_gnss_tier", "markov"),
            "imu_noise_multiplier": condition.get("imu_noise_multiplier", 1.0),
            "lidar_noise_multiplier": condition.get("lidar_noise_multiplier", 1.0),
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

    # EKF calibration records (predicted std vs actual error) for the
    # honesty-of-the-covariance analysis (scripts/evaluation/calibration.py).
    if calibration_rows:
        calib_df = pd.DataFrame(calibration_rows)
        calib_csv_path = os.path.join(run_output_dir, "calibration_records.csv")
        calib_df.to_csv(calib_csv_path, index=False)
        logger.info("EKF calibration records saved to %s", calib_csv_path)

    # Per-step uncertainty trace (EVAL_PER_STEP_CAP leading steps per episode),
    # the input to the per-step / matched-state gating analysis. Only written
    # when EVAL_PER_STEP_CAP > 0 and the head is evidential.
    if per_step_rows:
        per_step_df = pd.DataFrame(per_step_rows)
        per_step_csv_path = os.path.join(run_output_dir, "per_step_records.csv")
        per_step_df.to_csv(per_step_csv_path, index=False)
        logger.info("Per-step uncertainty trace saved to %s", per_step_csv_path)

    # Dump captured real observations for the on-manifold covariance probe.
    if captured_obs_all:
        obs_path = os.path.join(run_output_dir, "real_observations.npy")
        np.save(obs_path, np.asarray(captured_obs_all, dtype=np.float32))
        logger.info(
            "Captured %d real observations to %s (on-manifold probe input)",
            len(captured_obs_all),
            obs_path,
        )

    return df, run_output_dir


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
    parser.add_argument(
        "--conditions",
        type=str,
        nargs="+",
        default=None,
        help=(
            "Restrict the sweep to these eval_config condition names (space "
            "separated). Omit to evaluate every condition in eval_config."
        ),
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
        condition_names=args.conditions,
    )

    # Create plots alongside the CSVs.
    plot_evaluation_results(df, output_dir=run_output_dir)

    logger.info("Evaluation complete.")


if __name__ == "__main__":
    main()
