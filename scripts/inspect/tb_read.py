"""
@file tb_read.py
@brief Print scalar trajectories from a TensorBoard event directory.

Ad-hoc diagnostic for inspecting a training run's logged scalars without the
TensorBoard UI. Usage: python scripts/inspect/tb_read.py <log_dir>
"""

import sys

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ea = EventAccumulator(sys.argv[1], size_guidance={"scalars": 0})
ea.Reload()
tags = ea.Tags()["scalars"]
print("ALL TAGS:", sorted(tags))
print()
want = [
    "env/mean_speed_ms",
    "env/mean_progress_reward",
    "env/success_rate",
    "env/collision_rate",
    "env/oob_rate",
    "env/timeout_rate",
    "env/mean_pos_error_m",
    "env/mean_orientation_error_rad",
    "train/entropy_loss",
    "train/policy_gradient_loss",
    "train/value_loss",
    "train/explained_variance",
    "train/ent_coef",
    "train/std",
    "train/clip_fraction",
    "train/approx_kl",
    "rollout/ep_rew_mean",
    "rollout/ep_len_mean",
]
for t in want:
    if t not in tags:
        print(f"{t:38s} MISSING")
        continue
    ev = ea.Scalars(t)
    steps = [e.step for e in ev]
    vals = [e.value for e in ev]
    n = len(vals)
    idx = [0, n // 4, n // 2, 3 * n // 4, n - 1] if n > 4 else list(range(n))
    samp = ", ".join(f"{steps[i]//1000}k:{vals[i]:.4g}" for i in idx)
    print(
        f"{t:38s} n={n:4d} first={vals[0]:.4g} last={vals[-1]:.4g} "
        f"min={min(vals):.4g} max={max(vals):.4g} | {samp}"
    )
