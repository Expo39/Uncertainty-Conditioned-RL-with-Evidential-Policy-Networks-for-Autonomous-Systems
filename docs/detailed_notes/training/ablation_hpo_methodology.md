# Ablation Study and Hyperparameter Optimisation Methodology

> **No HPO was run for the reported experiments.** Every arm and seed uses the
> committed defaults in `configs/train_config.yaml`. The tuning pipeline described
> below is retained for future work and produced no reported result.

Extracted from the training and tuning pipeline (`training/train_ppo.py`,
`training/tune_hyperparams.py`).

Section 3.8 of the dissertation (`docs/AntonioGaldes_Dissertation.pdf`) is canonical for
the ablation itself - the four arms, the seed matrix and, in Section 3.8.1, the decision
to fix hyperparameters rather than tune them. The search space and Optuna machinery are
in `hyperparameter_search.md`.

This note records only what neither covers: the protocol that would apply if tuning were
run later, and the fairness constraints it would have to satisfy.

## The tension

- A FAIR ablation needs each baseline to perform at its own best, so a measured
  gap reflects the method and not unlucky tuning of one condition.
- But if you tune one configuration and apply those hyperparameters to all four,
  whichever configuration you tuned on gets an implicit advantage - the "shared
  hyperparameters overfit to one ablation" problem.

## What the literature says

Eimer et al. [1] is the current best-practice reference for HPO in RL, and two of its
recommendations matter here:

1. **Tune each algorithm individually** rather than forcing a single shared
   hyperparameter set, because the hyperparameter landscape differs by algorithm and a
   shared set disadvantages whichever configuration it was not tuned for.
2. **Separate tuning seeds from evaluation seeds.** The optimum can overfit to the
   tuning seed, so reporting on held-out seeds prevents that overfitting from inflating
   results. This is the single most important control.

Henderson et al. [2] show that hyperparameters and random seeds swing RL results
dramatically, and recommend reporting mean and variance across multiple seeds with
significance testing rather than the best run.

## If tuning were run

Neither option below was taken; Section 3.8.1 records why. Were tuning run later, two
exist.

Tuning the vanilla baseline once and sharing that set to all four arms is the
conservative choice: the set is optimised for the weakest arm, so it biases against the
method, and a win under it cannot be attributed to favourable tuning. Tuning each arm
separately follows Eimer et al. [1] and gives every condition its own best chance, but
costs roughly four times as much and reintroduces the "tuned for your own method"
objection that sharing removes.

Either way the fairness control is the same and non-negotiable: tuning seeds disjoint
from evaluation seeds [1], with multi-seed reporting [2].

### One set for the whole curriculum

A hyperparameter set must be fixed for a baseline's entire six-stage chain. Training
early stages on one set and resuming later stages on another is an incoherent mixed
schedule, and architectural settings cannot change on resume at all because the saved
weights would not load. The question is therefore not "tune per stage" but "obtain one
set per baseline and run the whole chain on it".

Tuning would run at stage 1 on the vanilla config, as a low-cost proxy for the full
chain. Multi-fidelity HPO on a cheaper proxy is a recognised technique, from the
successive-halving and Hyperband lineage [3] to RL-specific practice [1]. That principle
is all the literature supplies; the claim that stage 1 is a good proxy here rests on
this project's configs, where the stages differ only in the range of one axis, not in
reward, architecture or observation.

### Protocol

1. **Establish the task works first, on vanilla PPO**, using the committed
   `train_config.yaml` defaults. Vanilla is the pathfinder: a stall is then
   unambiguously an env or reward bug rather than an evidential interaction. Tuning a
   pipeline that does not yet learn optimises noise.
2. **Tune once, on the vanilla baseline at stage 1.** `BASELINE=` is mandatory;
   omitting it tunes the `train_config.yaml` defaults, which are the full-method
   config. The per-baseline path leaves `train_config.yaml` untouched and writes to
   `logs/tuning/results/`. Promotion into `train_config.yaml` is manual by design.
3. **Seed separation:** tuning uses a dedicated seed, disjoint from the evaluation
   seeds, so a tuned optimum cannot overfit the seeds it is scored on.
4. **Retrain the whole chain** for each baseline on the single fixed set.
5. **Report** mean and spread across evaluation seeds per baseline, with the tuning
   budget (trials, search space, seeds) stated.

### The honest caveat

A set tuned at stage 1 may be slightly suboptimal at later stages. Two mitigations:
PPO's structural parameters mainly govern optimisation stability, which transfers
across stages better than task-specific quantities; and every baseline receives the
same treatment, so any mild suboptimality is uniform across the ablation and cannot
bias the relative comparison. Tuning each trial over a full chain would be more
faithful but is infeasible on a single GPU within an MSc timeline.

### What stays fixed across ALL baselines and both phases

Architecture-defining settings are NOT tuned and NOT varied: `net_arch`,
`activation`, `policy_type` (except where it IS the ablation), observation/action
dims, `include_covariance` / `include_obstacle_obs` (except where they ARE the
ablation). These define the ablation itself; tuning them would confound the
comparison. Resume across phases also requires them constant.

### Why tune per-baseline does NOT reintroduce unfairness

Tuning each baseline to its own optimum is the opposite of unfair: it gives every
condition its best chance, so the comparison is "best vanilla PPO against best full
method", not "full method against a handicapped baseline". The fairness control is the
seed separation, which stops any baseline from winning by overfitting hyperparameters to
the seeds it is scored on.

## References

[1] T. Eimer, M. Lindauer, and R. Raileanu, "Hyperparameters in reinforcement learning
    and how to tune them," in *Proc. 40th Int. Conf. Machine Learning (ICML)*, vol. 202,
    2023, pp. 9104-9149.

[2] P. Henderson, R. Islam, P. Bachman, J. Pineau, D. Precup, and D. Meger, "Deep
    reinforcement learning that matters," in *Proc. AAAI Conf. Artificial Intelligence*,
    2018, pp. 3207-3214,
    doi: [10.1609/aaai.v32i1.11694](https://doi.org/10.1609/aaai.v32i1.11694).

[3] L. Li, K. Jamieson, G. DeSalvo, A. Rostamizadeh, and A. Talwalkar, "Hyperband: A
    novel bandit-based approach to hyperparameter optimization," *Journal of Machine
    Learning Research*, vol. 18, no. 185, pp. 1-52, 2018.
