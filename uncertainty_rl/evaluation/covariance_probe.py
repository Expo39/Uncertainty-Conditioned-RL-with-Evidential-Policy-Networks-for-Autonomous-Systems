"""
@file covariance_probe.py
@brief Measure a trained policy's action sensitivity to the EKF covariance input.

Holds one observation fixed and sweeps ONLY the covariance block (obs indices
VEHICLE_STATE_DIM through VEHICLE_STATE_DIM + COVARIANCE_FEATURES_DIM) from low
(certain) to high (uncertain), reporting how the deterministic action moves. A
growing action delta isolates the covariance as the cause, since nothing else
in the observation changes; a flat delta means the policy ignores the input.

Needs torch + stable-baselines3, so it runs in the training container via
`make docker-covariance-probe BASELINE=<name> CHECKPOINT=<leaf>`.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch as th

from uncertainty_rl.utils.constants import (
    COVARIANCE_FEATURES_DIM,
    OBS_STD_POS_SCALE,
    OBS_STD_YAW_SCALE,
    VEHICLE_STATE_DIM,
)

## Covariance block bounds in the observation vector (obs[2:5] by default).
_COV_START = VEHICLE_STATE_DIM
_COV_END = VEHICLE_STATE_DIM + COVARIANCE_FEATURES_DIM

## Raw EKF position stds (metres, 1-sigma) spanning clean to degraded, with a
## proportional yaw std. The EKF smooths the raw GNSS noise, so the worst tier
## settles near ~1.0 m position std rather than the 5 m raw GNSS figure.
_TIER_POS_STDS_M: List[Tuple[str, float, float]] = [
    ("rtk_fixed", 0.02, 0.016),
    ("rtk_float", 0.36, 0.022),
    ("standalone", 0.47, 0.038),
    ("degraded", 1.00, 0.060),
]


def _normalise_covariance(std_x: float, std_y: float, std_yaw: float) -> np.ndarray:
    """
    @brief Scale raw EKF stds to the policy's normalised covariance features.
    @param std_x: EKF position std in x (metres).
    @param std_y: EKF position std in y (metres).
    @param std_yaw: EKF heading std (radians).
    @return Length-COVARIANCE_FEATURES_DIM normalised covariance block, matching
            build_observation()'s scaling (divide by the fixed physical ceilings).
    """
    return np.array(
        [
            std_x / OBS_STD_POS_SCALE,
            std_y / OBS_STD_POS_SCALE,
            std_yaw / OBS_STD_YAW_SCALE,
        ],
        dtype=np.float32,
    )


def _make_base_observation(obs_dim: int) -> np.ndarray:
    """
    @brief Build a near-bay observation whose covariance block the caller sweeps.

    The non-covariance features are held fixed across the sweep, so any action
    change is attributable solely to the covariance. The ego is placed close to
    the bay and roughly aligned - the regime where acting on a wrong position
    has consequences. The covariance block starts clean and is overwritten per
    sweep step.

    @param obs_dim: The policy's observation dimension (from the loaded model).
    @return A length-obs_dim normalised observation.
    """
    obs = np.zeros(obs_dim, dtype=np.float32)
    # Vehicle state (speed, vyaw): mild forward speed, no rotation.
    obs[0] = 0.4  # normalised speed - approaching the bay
    obs[1] = 0.0
    # Covariance block: start clean (overwritten in the sweep).
    std_x, std_y, std_yaw = _TIER_POS_STDS_M[0][1], _TIER_POS_STDS_M[0][1], 0.016
    obs[_COV_START:_COV_END] = _normalise_covariance(std_x, std_y, std_yaw)
    # Target-pose / LiDAR features stay at neutral placeholders; only their
    # delta across the sweep matters, and that delta is zero (they are fixed).
    if obs_dim > _COV_END:
        obs[_COV_END] = 0.05  # small along-track offset (near the bay)
    return obs


def run_probe(model_path: str) -> None:
    """
    @brief Load a trained evidential policy and sweep its covariance input.
    @param model_path: Path to the saved model (without or with .zip).

    Prints the deterministic action [steer, throttle, brake] and the policy's
    aleatoric/epistemic at each covariance level, plus the action delta from the
    clean baseline.
    """
    from uncertainty_rl.networks.sb3_integration import EvidentialPPO

    model = EvidentialPPO.load(model_path, device="cpu")
    policy = model.policy
    policy.set_training_mode(False)

    obs_dim = int(policy.observation_space.shape[0])
    base = _make_base_observation(obs_dim)

    print(f"Covariance probe: {model_path}")
    print(f"obs_dim={obs_dim}  covariance block = obs[{_COV_START}:{_COV_END}]")
    print("Sweeping only the covariance block; all other features held fixed.\n")
    header = (
        f"{'tier':12s}{'std_pos(m)':>11s}{'steer':>9s}{'throttle':>9s}"
        f"{'brake':>9s}{'aleatoric':>11s}{'epistemic':>11s}{'|d action|':>11s}"
    )
    print(header)
    print("-" * len(header))

    baseline_action: np.ndarray = np.zeros(3, dtype=np.float32)
    for i, (name, std_pos, std_yaw) in enumerate(_TIER_POS_STDS_M):
        obs = base.copy()
        obs[_COV_START:_COV_END] = _normalise_covariance(std_pos, std_pos, std_yaw)
        obs_t = th.as_tensor(obs, dtype=th.float32).unsqueeze(0)

        action_t, unc = policy.get_action_with_uncertainty(obs_t, deterministic=True)
        action = action_t.squeeze(0).cpu().numpy()
        aleatoric = float(unc["aleatoric"].mean().item())
        epistemic = float(unc["epistemic"].mean().item())

        if i == 0:
            baseline_action = action.copy()
        delta = float(np.linalg.norm(action - baseline_action))

        print(
            f"{name:12s}{std_pos:>11.3f}{action[0]:>9.3f}{action[1]:>9.3f}"
            f"{action[2]:>9.3f}{aleatoric:>11.4f}{epistemic:>11.4f}{delta:>11.4f}"
        )

    print(
        "\n|d action| growing as std rises means the policy conditions on the "
        "covariance;\na flat near-zero column means it ignores the input."
    )


def main() -> None:
    """
    @brief CLI entry point for the covariance causal probe.
    """
    parser = argparse.ArgumentParser(
        description="Probe whether a trained policy uses the EKF covariance input."
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the trained model (e.g. checkpoints/full_method/<leaf>/final_model).",
    )
    args = parser.parse_args()
    run_probe(args.model_path)


if __name__ == "__main__":
    main()
