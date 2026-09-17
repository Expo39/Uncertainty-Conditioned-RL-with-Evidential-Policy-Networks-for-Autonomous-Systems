# Ablation and Hyperparameter Optimisation

> **No HPO was run.** Every arm and seed uses the committed defaults in
> `configs/train_config.yaml`.

Extracted from `training/train_ppo.py` and `training/tune_hyperparams.py`.

Section 3.8 of the dissertation (`docs/AntonioGaldes_Dissertation.pdf`) is canonical for
the ablation and, in Section 3.8.1, for the decision to fix the hyperparameters rather
than tune them. The search space and Optuna settings are in `hyperparameter_search.md`.

This note records only the code-level constraints behind that decision.

## Architectural settings are neither tuned nor varied

`net_arch`, `activation`, `policy_type`, `include_covariance`, `include_obstacle_obs`
and the observation and action dimensions stay identical across every arm and every
stage. Two separate reasons apply:

- They define the ablation. `policy_type` and the two observation flags ARE the variable
  under test, set per arm in `configs/baselines/`; varying them elsewhere would confound
  the comparison.
- A stage resume loads saved weights, so a change to any of them makes the checkpoint
  unloadable. The stage allowlist in `train_ppo.py` excludes all of them for that reason.

## One set per chain

A hyperparameter set is fixed for a baseline's entire six-stage chain. Training early
stages on one set and resuming later stages on another would give a final model
optimised under neither, so the set is committed before stage 1 and left alone.

## If the pipeline is used later

`BASELINE=` is mandatory. Omitting it tunes the `train_config.yaml` defaults, which are
the full-method configuration rather than vanilla. A per-baseline run leaves
`train_config.yaml` untouched and writes to `logs/tuning/results/`, promotion being
manual by design.

Tuning seeds must stay disjoint from the evaluation seeds 42, 123 and 7, since an
optimum found on a seed it is later scored on overfits that seed [1], and results should
be reported across seeds rather than from the best run [2].

## References

[1] T. Eimer, M. Lindauer, and R. Raileanu, "Hyperparameters in reinforcement learning
    and how to tune them," in *Proc. 40th Int. Conf. Machine Learning (ICML)*, vol. 202,
    2023, pp. 9104-9149.

[2] P. Henderson, R. Islam, P. Bachman, J. Pineau, D. Precup, and D. Meger, "Deep
    reinforcement learning that matters," in *Proc. AAAI Conf. Artificial Intelligence*,
    2018, pp. 3207-3214,
    doi: [10.1609/aaai.v32i1.11694](https://doi.org/10.1609/aaai.v32i1.11694).
