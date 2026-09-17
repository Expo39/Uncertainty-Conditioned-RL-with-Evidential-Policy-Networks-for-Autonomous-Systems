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

Established practice is to optimise each method under comparison separately, since
values that suit one method rarely suit another. An ablation inverts that requirement:
the arms differ by one component at a time, and every other variable is held constant so
the measured gap can be attributed to that component. Tuning per arm would add the
configuration to the list of things that differ.

The conflict is resolved in favour of the controlled comparison, because the
contribution rests on the ablation. The cost is that each arm is ranked at one operating
point rather than at its best, which is stated as a limitation rather than hidden.

## What the literature says

Eimer et al. [1] is the current best-practice reference for HPO in RL, and two of its
recommendations matter here:

1. **Tune each algorithm individually** rather than forcing a single shared
   hyperparameter set, because the hyperparameter landscape differs by algorithm and a
   shared set disadvantages whichever configuration it was not tuned for. This addresses
   comparisons between algorithms; it does not carry over to an ablation, where adding a
   per-arm search would confound the variable under test.
2. **Separate tuning seeds from evaluation seeds.** The optimum can overfit to the
   tuning seed, so reporting on held-out seeds prevents that overfitting from inflating
   results. This control applies whenever tuning happens at all.

Henderson et al. [2] show that hyperparameters and random seeds swing RL results
dramatically, and recommend reporting mean and variance across multiple seeds with
significance testing rather than the best run.

## If tuning were run

No tuning was run; Section 3.8.1 records why. Were it run later, only one option
preserves the ablation.

Tuning the vanilla baseline once and sharing that set to all four arms keeps the
configuration constant across the comparison, so it stays a held variable rather than
becoming a fifth free one. It is also conservative in direction: the set is optimised
for the weakest arm, so it biases against the method, and a win under it cannot be
attributed to favourable tuning.

Tuning each arm separately follows the general advice of Eimer et al. [1], but that
advice addresses comparisons between algorithms, not an ablation in which every other
variable is pinned. Applied here it would confound the result, as set out below.

Whichever is used, tuning seeds must stay disjoint from evaluation seeds [1], with
multi-seed reporting [2].

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
comparison. Resume across stages also requires them constant, as the saved weights
would not otherwise load.

### Why per-baseline tuning introduces bias

Tuning each arm separately is the standard advice outside an ablation, and it is the
wrong move inside one. The four arms isolate the actor head and the covariance gate by
holding every other variable constant. A separate search per arm makes the configuration
a fifth variable, so any measured difference becomes partly attributable to the four
searches converging unevenly rather than to the architecture under test. Seed separation
does not rescue this: it controls overfitting to the seeds an arm is scored on, not the
confound introduced by four independent searches.

This is why the reported experiments are untuned, and why per-arm tuning would not be
the default even with unlimited compute.

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
