from uncertainty_rl.training.train_ppo import linear_schedule
s = linear_schedule(0.02, 0.008)
# progress_remaining goes 1.0 -> 0.0 over the stage
for pr in [1.0, 0.75, 0.5, 0.25, 0.0]:
    print(f"progress_remaining={pr} -> ent_coef={s(pr):.5f}")
