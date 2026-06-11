# Hyperparameter Search

> **Updated 8 June 2026.** Under the single-phase curriculum, HPO tunes STRUCTURAL PPO params
> only (`n_steps`, `batch_size`, `n_epochs`, `clip_range`, `gae_lambda`, `gamma`, `vf_coef`,
> `max_grad_norm`). `learning_rate` and `ent_coef` are NOT tuned - they are stage-owned
> per-policy schedules (`standard_overrides`/`evidential_overrides`). See
> `documentation/detailed_notes/ablation_hpo_methodology.md`.

Search space design rationale and literature references for the Optuna-based
hyperparameter tuning in `uncertainty_rl/training/tune_hyperparams.py`.

---

## Search space design

Eight parameters are actively searched; the rest retain their defaults from
`configs/train_config.yaml`:

| Parameter | Range | Sampling |
|-----------|-------|---------|
| `learning_rate` | [1e-5, 1e-3] | log scale |
| `n_steps` | {1024, 2048, 4096} | categorical |
| `batch_size` | {64, 128, 256} | categorical, constrained <= n_steps |
| `n_epochs` | {3, 5, 10} | categorical |
| `gamma` | [0.98, 0.999] | log scale via 1 - (1-gamma) |
| `ent_coef` | [1e-6, 0.01] | log scale |
| `evidential.lambda_reg` | [1e-5, 0.01] | log scale |
| `evidential.lambda_reg_warmup_steps` | [10000, 100000] | log scale |

Fixed parameters (`clip_range`, `max_grad_norm`, `target_kl`, `net_arch`) are
excluded from the search because their sensitivity is low relative to the
above eight for on-policy RL in parking tasks [1][2].

## Sampler and pruner

**Sampler**: multivariate TPE (`TPESampler(multivariate=True)`). Multivariate
TPE models parameter interactions (e.g. `learning_rate` vs `batch_size`) by
fitting a joint density model rather than independent univariate models [3].
`n_startup_trials=10` gives pure random exploration before TPE builds its
model.

**Pruner**: `MedianPruner` over `HyperbandPruner`. Hyperband creates multiple
brackets each requiring approximately 10 startup trials, exhausting most of
the trial budget on random search for a budget of 35-50 trials [4]. Median
pruner is simpler and more budget-efficient at this scale.

## Gamma sampling

Gamma is sampled as `1 - (1 - gamma)` on a log scale rather than gamma
directly on a linear scale. This gives uniform precision near 1.0, where
small differences in gamma (0.99 vs 0.999) have large effects on effective
horizon length, whereas differences near 0.9 matter less [4].

## References

[1] M. Andrychowicz et al., "What Matters for On-Policy Deep Actor-Critic Methods?
    A Large-Scale Study," in Proc. Int. Conf. Learning Representations (ICLR), May 2021.

[2] T. Eimer, A. Biedenkapp, M. Reimer, S. Adriaensen, F. Hutter, and M. Lindauer,
    "Hyperparameters in Reinforcement Learning and How To Tune Them," in Proc. Int.
    Conf. Machine Learning (ICML), Honolulu, HI, USA, Jul. 2023.

[3] S. Watanabe, "Tree-Structured Parzen Estimator: Understanding Its Algorithm
    Components and Their Roles for Better Empirical Performance," arXiv:2304.11127,
    Apr. 2023.

[4] A. Raffin, "Automatic Hyperparameter Tuning in Practice," Tutorial at IEEE Int.
    Conf. Robotics and Automation (ICRA), Philadelphia, PA, USA, May 2022.
