# Hyperparameter Search

Search space design rationale and literature references for the Optuna study in
`uncertainty_rl/training/tune_hyperparams.py`.

> **This search was never run.** Every reported result uses the committed defaults in
> `configs/train_config.yaml`, for the reason given in Section 3.8.1 of the dissertation.
> The tooling below is retained for future work. See `ablation_hpo_methodology.md` for
> the constraints a later run would have to respect.

---

## Search space design

Eight structural PPO parameters are actively searched; the rest retain their
defaults from `configs/train_config.yaml`. Bounds are read from
`search_space:` in `configs/training/tuning_config.yaml`, the defaults below
being the fallbacks in `tune_hyperparams.py`:

| Parameter | Range | Sampling |
|-----------|-------|---------|
| `n_steps` | {1024, 2048, 4096} | categorical |
| `batch_size` | {64, 128, 256} | categorical, constrained <= n_steps |
| `n_epochs` | {3, 5, 10} | categorical |
| `gamma` | [0.98, 0.999] | log scale via 1 - (1-gamma) |
| `gae_lambda` | [0.90, 0.98] | uniform |
| `clip_range` | [0.1, 0.3] | uniform |
| `vf_coef` | [0.25, 1.0] | uniform |
| `max_grad_norm` | [0.3, 2.0] | uniform |

`learning_rate` and `ent_coef` are deliberately excluded: both are stage-owned
per-policy schedules set through `standard_overrides` / `evidential_overrides`,
so tuning them here would be a no-op. `net_arch`, `activation` and `policy_type`
are architectural and locked across stages, since changing them prevents saved
weights from loading. The `evidential.*` terms are ablation-specific and cannot
be tuned on the vanilla config without breaking fairness between arms [1][2].

## Sampler and pruner

**Sampler**: multivariate TPE (`TPESampler(multivariate=True)`). Multivariate
TPE models parameter interactions (e.g. `n_steps` vs `batch_size`) by fitting a
joint density model rather than independent univariate models [3]. The startup
trials give pure random exploration before TPE builds its model.

**Pruner**: `MedianPruner` over `HyperbandPruner`. Hyperband creates multiple
brackets each requiring its own startup trials, which would exhaust most of the
budget on random search at this scale [4]. Median pruning is simpler and more
budget-efficient for a study of this size.

Trial counts, startup trials and warmup steps are set in
`configs/training/tuning_config.yaml` and are not repeated here.

## Gamma sampling

Gamma is sampled as `1 - (1 - gamma)` on a log scale rather than gamma directly on a
linear scale. This gives uniform precision near 1.0, where small differences (0.99
against 0.999) change the effective horizon substantially, whereas differences near 0.9
matter less. The discount factor is among the parameters whose setting most affects
on-policy performance [1].

## References

[1] M. Andrychowicz, A. Raichuk, P. Stanczyk, M. Orsini, S. Girgin, R. Marinier,
    L. Hussenot, M. Geist, O. Pietquin, M. Michalski, S. Gelly, and O. Bachem, "What
    matters for on-policy deep actor-critic methods? A large-scale study," in *Proc.
    9th Int. Conf. Learning Representations (ICLR)*, 2021.

[2] T. Eimer, M. Lindauer, and R. Raileanu, "Hyperparameters in reinforcement learning
    and how to tune them," in *Proc. 40th Int. Conf. Machine Learning (ICML)*, vol. 202,
    2023, pp. 9104-9149.

[3] T. Akiba, S. Sano, T. Yanase, T. Ohta, and M. Koyama, "Optuna: A next-generation
    hyperparameter optimization framework," in *Proc. 25th ACM SIGKDD Int. Conf.
    Knowledge Discovery and Data Mining*, 2019, pp. 2623-2631.

[4] L. Li, K. Jamieson, G. DeSalvo, A. Rostamizadeh, and A. Talwalkar, "Hyperband: A
    novel bandit-based approach to hyperparameter optimization," *Journal of Machine
    Learning Research*, vol. 18, no. 185, pp. 1-52, 2018.
