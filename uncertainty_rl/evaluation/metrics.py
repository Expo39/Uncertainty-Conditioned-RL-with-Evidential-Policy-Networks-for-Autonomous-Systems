"""
@file metrics.py
@brief Evaluation metrics container and per-episode outcome taxonomy.

Holds the data model the evaluation loop fills in (EvaluationMetrics) and the
single-episode failure-mode classifier (_classify_outcome). Kept separate from
the orchestration in evaluate.py so the metric schema and taxonomy - the parts
the downstream calibration analysis depends on - can be read and tested in
isolation, without importing torch / stable-baselines3.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import numpy as np


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
    # Optional capture of real normalised observations (one ndarray per decision)
    # for the on-manifold covariance probe. Empty unless EVAL_DUMP_OBS is set.
    captured_observations: List[Any] = field(default_factory=list)
    # Per-step EKF calibration pairs (predicted std vs actual GT-EKF error), one
    # dict per decision, for the honesty-of-the-covariance analysis. Populated
    # for every baseline (ground truth is reward-only, never in the obs).
    calibration_pairs: List[Dict[str, Any]] = field(default_factory=list)
    # Per-step uncertainty trace (one dict per decision: step index within the
    # episode, epistemic, aleatoric, EKF stds). Populated only for evidential
    # heads, capped per episode by EVAL_PER_STEP_CAP. Lets the per-step gating
    # analysis test whether epistemic[t] tracks the instantaneous situation
    # rather than drifting with episode length (a real-time handoff needs the
    # former). @see evaluate_agent.
    per_step_records: List[Dict[str, Any]] = field(default_factory=list)

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
