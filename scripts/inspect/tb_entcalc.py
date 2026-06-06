"""
@file tb_entcalc.py
@brief Compute the entropy-vs-policy-gradient loss balance from a run's logs.

Reads train/entropy_loss, train/policy_gradient_loss, train/value_loss and the
logged learning_rate from a TensorBoard event dir, then for each sampled step
reports the entropy term (ent_coef * entropy_loss) against the policy-gradient
loss so the dominant term is visible. Used to size ent_coef. The actual ent_coef
used per update is the linear-decay schedule, recomputed here from the stage
initials and the run's progress.

Usage: python scripts/inspect/tb_entcalc.py <log_dir> <ent_init> <ent_final>
       <stage_timesteps>
"""

import sys

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def main() -> None:
    log_dir = sys.argv[1]
    ent_init = float(sys.argv[2])
    ent_final = float(sys.argv[3])
    stage_steps = float(sys.argv[4])

    ea = EventAccumulator(log_dir, size_guidance={"scalars": 0})
    ea.Reload()

    ent = {e.step: e.value for e in ea.Scalars("train/entropy_loss")}
    pg = {e.step: e.value for e in ea.Scalars("train/policy_gradient_loss")}
    steps = sorted(set(ent) & set(pg))

    print(
        f"ent_coef schedule used in THIS run: {ent_init} -> {ent_final} "
        f"over {stage_steps/1e6:.1f}M steps\n"
    )
    hdr = (
        f"{'step':>8} {'entropy_loss':>13} {'|pg_loss|':>11} "
        f"{'ent_coef':>9} {'ent_term':>10} {'ratio(ent/pg)':>14}"
    )
    print(hdr)
    n = len(steps)
    idx = [0, n // 5, 2 * n // 5, 3 * n // 5, 4 * n // 5, n - 1]
    for i in idx:
        s = steps[i]
        progress_remaining = max(0.0, 1.0 - s / stage_steps)
        ent_coef = ent_final + progress_remaining * (ent_init - ent_final)
        e = ent[s]  # entropy_loss = -entropy (SB3 sign)
        p = abs(pg[s])
        ent_term = ent_coef * e  # signed contribution to the loss
        ratio = abs(ent_term) / p if p > 1e-12 else float("inf")
        print(
            f"{s:>8} {e:>13.4f} {p:>11.5f} {ent_coef:>9.5f} "
            f"{ent_term:>10.5f} {ratio:>14.1f}"
        )

    # Use the LATE-run entropy magnitude (policy near its widest) to size ent_coef
    # for a target ent_term : pg_loss balance.
    late_ent = abs(ent[steps[-1]])
    late_pg = abs(pg[steps[-1]])
    print(
        f"\nLate-run magnitudes: |entropy_loss|={late_ent:.3f}, "
        f"|pg_loss|={late_pg:.5f}"
    )
    for target in (1.0, 0.5, 0.25):
        # ent_coef * late_ent = target * late_pg
        ec = target * late_pg / late_ent
        print(
            f"  ent_coef for ent_term = {target:>4} x pg_loss: {ec:.5f}"
        )


if __name__ == "__main__":
    main()
