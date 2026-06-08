"""
@file gnss_drift.py
@brief Pure helper for the per-episode GNSS mid-episode-drift margin.

Scales a GNSS RTK fix-state Markov transition matrix by the per-stage
`drift_scale` in [0, 1]. Lives in utils/ (no ROS 2 dependency) so it is
importable and unit-testable on the host and in CI, while the ros2-bridge
GnssNoiseRelayNode imports it to rebuild its effective transition matrix.
"""

import numpy as np


def scale_transition_matrix(base: np.ndarray, drift_scale: float) -> np.ndarray:
    """
    @brief Scale a row-stochastic transition matrix's off-diagonals by drift_scale.

    Multiplies every off-diagonal of `base` by `drift_scale` (clipped to [0, 1]) and
    sets each diagonal so the row still sums to 1: drift_scale = 0 yields the identity
    (current tier holds), drift_scale = 1 reproduces `base`.

    @param base: Square row-stochastic matrix (each row sums to 1) - the full chain.
    @param drift_scale: Drift margin in [0, 1]; out-of-range values are clipped.
    @return New float64 row-stochastic matrix (does not mutate `base`).
    """
    scale = float(min(1.0, max(0.0, drift_scale)))
    eff = np.array(base, dtype=np.float64) * scale
    np.fill_diagonal(eff, 0.0)
    off_diag_mass = eff.sum(axis=1)
    np.fill_diagonal(eff, 1.0 - off_diag_mass)
    return eff
