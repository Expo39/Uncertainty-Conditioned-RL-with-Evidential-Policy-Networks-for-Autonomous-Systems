"""
@file markov_analyser.py
@brief Offline diagnostic for the GNSS tier Markov chain.

Loads the per-step transition matrix and per-episode initial-tier sampling
weights. Reports:

 - Static chain properties: init weight, stationary distribution, mean
   dwell per tier.
 - Per-episode occupation across simulated rollouts.
 - Time to first contiguous good window for episodes starting in a bad
   tier, across a few window lengths.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import yaml


_TIER_ORDER: List[str] = ["rtk_fixed", "rtk_float", "standalone", "degraded"]


def _load_profiles(profiles_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    @brief Load init-tier weights and transition matrix from the profiles YAML.

    @return Tuple (init_weights, transition_matrix). Both indexed in
            _TIER_ORDER. Matrix rows are validated to sum to 1.0.
    """
    with profiles_path.open("r") as f:
        data = yaml.safe_load(f)

    tiers = data["tiers"]
    weights = np.array(
        [float(tiers[name].get("weight", 0.0)) for name in _TIER_ORDER],
        dtype=np.float64,
    )
    weights = weights / weights.sum()

    matrix_section = data.get("transition_matrix")
    if matrix_section is None:
        raise ValueError(f"Missing 'transition_matrix' section in {profiles_path}.")

    rows = []
    for name in _TIER_ORDER:
        row = matrix_section[name]
        if len(row) != len(_TIER_ORDER):
            raise ValueError(
                f"transition_matrix row '{name}' has length {len(row)}, "
                f"expected {len(_TIER_ORDER)}."
            )
        rows.append([float(x) for x in row])
    P = np.array(rows, dtype=np.float64)

    if not np.allclose(P.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError(
            f"transition_matrix rows must sum to 1.0; got {P.sum(axis=1).tolist()}."
        )
    return weights, P


def _stationary_distribution(P: np.ndarray) -> np.ndarray:
    """@brief Left eigenvector of P with eigenvalue 1, normalised to sum 1."""
    eigvals, eigvecs = np.linalg.eig(P.T)
    idx = int(np.argmin(np.abs(eigvals - 1.0)))
    pi = np.real(eigvecs[:, idx])
    return pi / pi.sum()


def _simulate_episodes(
    P: np.ndarray,
    init_weights: np.ndarray,
    n_episodes: int,
    n_steps: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """@brief Roll out N episodes of the tier chain. Returns (N, T) tier indices."""
    cdf = np.cumsum(P, axis=1)
    init_cdf = np.cumsum(init_weights)
    out = np.empty((n_episodes, n_steps), dtype=np.int8)
    out[:, 0] = np.searchsorted(init_cdf, rng.random(n_episodes))
    for t in range(1, n_steps):
        u = rng.random(n_episodes)
        out[:, t] = (cdf[out[:, t - 1]] < u[:, None]).sum(axis=1)
    return out


def _first_good_window(
    rollouts: np.ndarray,
    good_mask: np.ndarray,
    window_len_steps: int,
) -> np.ndarray:
    """@brief First step where a contiguous good window of length W ends; -1 if none."""
    is_good = good_mask[rollouts]
    csum = np.concatenate(
        [np.zeros((is_good.shape[0], 1), dtype=int), is_good.cumsum(axis=1)],
        axis=1,
    )
    window_sum = csum[:, window_len_steps:] - csum[:, :-window_len_steps]
    has = window_sum == window_len_steps
    any_match = has.any(axis=1)
    return np.where(any_match, has.argmax(axis=1) + window_len_steps - 1, -1)


def _resolve_good_mask(good_tiers: Sequence[str]) -> np.ndarray:
    """@brief Boolean mask over _TIER_ORDER selecting the named good tiers."""
    unknown = [t for t in good_tiers if t not in _TIER_ORDER]
    if unknown:
        raise ValueError(f"Unknown tier(s) in --good-tiers: {unknown}")
    return np.array([t in good_tiers for t in _TIER_ORDER])


def _format_seconds(steps: float, dt: float) -> str:
    """@brief Human-readable seconds string from a step count."""
    return "  never" if steps < 0 else f"{steps * dt:6.2f}s"


def _run_analysis(
    profiles_path: Path,
    n_episodes: int,
    n_steps: int,
    dt: float,
    seed: int,
    good_tiers: Sequence[str],
    window_seconds: Sequence[float],
) -> None:
    """@brief Print the chain summary and recovery-from-bad-start stats."""
    init_weights, P = _load_profiles(profiles_path)
    good_mask = _resolve_good_mask(good_tiers)
    rng = np.random.default_rng(seed)

    pi = _stationary_distribution(P)
    dwell_steps = 1.0 / (1.0 - np.diag(P))

    print(
        f"Chain: 4 tiers, {n_episodes} eps x {n_steps} steps "
        f"({n_steps * dt:.0f}s @ {1.0/dt:.0f} Hz)"
    )
    bad_str = ",".join(t for t, m in zip(_TIER_ORDER, good_mask) if not m)
    good_str = ",".join(good_tiers)
    print(f"Good = {{{good_str}}}   Bad = {{{bad_str}}}")

    rollouts = _simulate_episodes(P, init_weights, n_episodes, n_steps, rng)
    occupation = np.array(
        [(rollouts == i).mean() for i in range(len(_TIER_ORDER))]
    )

    print(
        f"\n{'tier':<12}{'init':>8}{'stationary':>12}{'occupation':>12}"
        f"{'dwell':>10}"
    )
    for i, name in enumerate(_TIER_ORDER):
        print(
            f"{name:<12}{init_weights[i]:>8.3f}{pi[i]:>12.3f}"
            f"{occupation[i]:>12.3f}{dwell_steps[i] * dt:>9.1f}s"
        )

    bad_init_mask = ~good_mask[rollouts[:, 0]]
    n_bad = int(bad_init_mask.sum())
    if n_bad == 0:
        return

    print(
        f"\nEpisodes starting bad: {n_bad}/{n_episodes} "
        f"({n_bad / n_episodes * 100:.1f}%). "
        f"Times below are within-episode, conditional on bad start."
    )
    print(f"\n{'window W':>10}{'reached':>12}{'median':>10}{'p90':>10}")
    for win_seconds in window_seconds:
        win_steps = int(round(win_seconds / dt))
        first_win = _first_good_window(rollouts[bad_init_mask], good_mask, win_steps)
        reached = first_win[first_win >= 0]
        pct = len(reached) / n_bad * 100
        med = _format_seconds(
            float(np.median(reached)) if len(reached) else -1.0, dt
        )
        p90 = _format_seconds(
            float(np.percentile(reached, 90)) if len(reached) else -1.0, dt
        )
        print(f"{win_seconds:>9.1f}s{pct:>11.1f}%{med:>10}{p90:>10}")


def _parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[2]
    default_profiles = (
        project_root / "configs" / "deployment" / "sim" / "gnss_noise_profiles.yaml"
    )
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--profiles", type=Path, default=default_profiles)
    p.add_argument("--n-episodes", type=int, default=10000)
    p.add_argument(
        "--n-steps",
        type=int,
        default=1750,
        help="Episode length in steps (default matches env_config max_steps).",
    )
    p.add_argument(
        "--dt",
        type=float,
        default=0.05,
        help="Per-step duration (seconds), matches carla_timestep.",
    )
    p.add_argument(
        "--good-tiers",
        nargs="+",
        default=["rtk_fixed", "rtk_float"],
        help="Which tiers count as 'good'.",
    )
    p.add_argument(
        "--window-seconds",
        nargs="+",
        type=float,
        default=[3.0, 5.0, 10.0],
        help="Contiguous good-window lengths (seconds) to evaluate.",
    )
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    """@brief CLI entry point."""
    args = _parse_args()
    _run_analysis(
        profiles_path=args.profiles,
        n_episodes=args.n_episodes,
        n_steps=args.n_steps,
        dt=args.dt,
        seed=args.seed,
        good_tiers=args.good_tiers,
        window_seconds=args.window_seconds,
    )


if __name__ == "__main__":
    main()
