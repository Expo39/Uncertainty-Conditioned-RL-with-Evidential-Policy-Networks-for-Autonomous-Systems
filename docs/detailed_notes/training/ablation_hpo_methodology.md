# Ablation Study and Hyperparameter Optimisation Methodology

> **Updated 8 June 2026 - single-phase curriculum.** The 2-phase (clean -> noisy) framing
> below is superseded. HPO now runs AFTER the curriculum validates on vanilla (post-stage-6
> checkpoint) and tunes STRUCTURAL PPO params only (`n_steps`, `batch_size`, `n_epochs`,
> `clip_range`, `gae_lambda`, `gamma`, `vf_coef`, `max_grad_norm`) - NOT `learning_rate`/
> `ent_coef`, which are stage-owned per-policy (`standard_overrides`/`evidential_overrides`)
> schedules. The fairness arguments (vanilla-and-share, disjoint tuning/eval seeds) are
> unchanged. Canonical: `documentation/detailed_notes/ablation_hpo_methodology.md` and
> `documentation/CURRICULUM_PLAN.md`.

Extracted from the training and tuning pipeline (`training/train_ppo.py`, `training/tune_hyperparams.py`). This document explains how hyperparameter tuning is performed for the four ablation baselines, and why, grounded in the RL methodology literature.

## The four ablation baselines

The contribution is tested as a 2x2 matrix (input uncertainty x output
uncertainty):

1. Vanilla PPO - no covariance obs, standard Gaussian actor.
2. Input-uncertainty only - covariance obs, standard actor.
3. Output-uncertainty only - no covariance obs, evidential actor.
4. Full method - covariance obs + evidential actor.

The variable under test is the architecture/observation, NOT the hyperparameters.

## The tension

- A FAIR ablation needs each baseline to perform at its own best, so a measured
  gap reflects the method and not unlucky tuning of one condition.
- But if you tune one configuration and apply those hyperparameters to all four,
  whichever configuration you tuned on gets an implicit advantage - the "shared
  hyperparameters overfit to one ablation" problem.

## What the literature says

- **Eimer, Lindauer and Raileanu, "Hyperparameters in Reinforcement Learning and
  How To Tune Them", ICML 2023 (arXiv:2306.01324)** - the current best-practice
  reference for HPO in RL. Two recommendations matter here:
  1. **Tune each algorithm individually** rather than forcing a single shared
     hyperparameter set, because the hyperparameter landscape differs by
     algorithm and a shared set disadvantages whichever configuration it was not
     tuned for.
  2. **Separate tuning seeds from testing (evaluation) seeds.** The optimum can
     overfit to the tuning seed; reporting on held-out seeds prevents that
     overfitting from inflating results. This is the single most important
     control.
- **Henderson et al., "Deep Reinforcement Learning that Matters", AAAI 2018
  (arXiv:1709.06560)** - hyperparameters and random seeds swing RL results
  dramatically; report mean +/- variance across MULTIPLE seeds with significance
  testing, never cherry-pick the best run.

## Decision for this project

**Primary protocol (MSc timeline): tune the vanilla PPO baseline once and share
that single set to all four baselines**, with strict tuning/evaluation seed
separation, reported across multiple held-out seeds. This satisfies the
fair-comparison requirement (Henderson et al.: baselines must be tuned to the
greatest extent possible) by tuning the WEAKEST baseline to its best, and it is the
conservative, unattackable choice: the shared set is optimised for vanilla PPO, so
it biases AGAINST the method. If the full method still wins on hyperparameters
tuned for the baseline, nobody can object that the method was favoured by tuning.
It is also the cheapest - one study, not four.

**Stretch goal (compute permitting): tune each of the four baselines separately
(one Optuna study per baseline).** This is the modern best practice under Eimer et
al. (individualised tuning) - it gives every condition its own best chance. But it
costs ~4x as much and is rhetorically WEAKER than vanilla-and-share: per-baseline
tuning reintroduces the "you tuned for your own method" objection that
vanilla-and-share removes outright. Promote per-baseline tuning to primary only if
the vanilla-tuned set visibly handicaps a baseline.

**Fallback within the primary protocol:** if the evidential baselines
(output-uncertainty, full-method) are unstable under vanilla's tuned learning rate
- evidential losses can need a gentler LR - tune those two separately. Trigger this
only on observed instability in the shared-hyperparameter run; do not pre-pay for
it.

Whichever is chosen, the fairness control is the SAME and non-negotiable: tuning
seeds disjoint from evaluation seeds (Eimer et al.), with multi-seed reporting
(Henderson et al.).

### The mid-run-change problem (why ONE set for the whole curriculum)

A hyperparameter set must be FIXED for the entire A -> B curriculum run of a given
baseline. You do NOT train Phase A on one set and then "continue the model with
updated hyperparameters" for Phase B - that is an incoherent mixed schedule (the
final model would have used defaults for A and tuned params for B), and network
architecture cannot change on resume at all (weights would not load). So the
question is not "tune A then tune B" - it is "obtain ONE set per baseline and run
the whole curriculum on it."

The cost-efficient way to obtain that one set is to **tune on the cheaper clean
Phase A as a proxy**, then apply it across the whole run. The justification has TWO
distinct parts - keep them separate and do not let the citation carry the second:

1. **The principle (from the literature):** multi-fidelity HPO legitimately tunes
   on a cheaper proxy task to reduce cost (the Hyperband / successive-halving
   lineage; and RL-HPO best practice, Eimer et al. 2023). This supports "tuning on
   a cheap proxy is a recognised technique" - NOTHING more. No external paper says
   "tune on the clean phase" or "Phase A and B are the same task"; those are claims
   about THIS environment, not the literature.
2. **Why the clean phase is a GOOD proxy here (from THIS project's config, not a
   citation):** in the restructured design, Phase A and Phase B differ ONLY by
   sensor-noise injection. Verified against the configs - the toggles that change
   between phases are exactly: `fixed_gnss_tier` (removed in B),
   `lidar.noise.enabled`, `enable_gnss_noise`, `enable_markov_transitions`,
   `enable_gnss_anisotropy`, `enable_imu_noise` (all false in A, true in B).
   Everything else is identical across both phases: layout, random bays/spawns
   (`use_extra_spawns`), 0.8 bay occupancy, parked cars, reward, and architecture.
   Patrol vehicles and pedestrians are off in BOTH (out of scope). So the
   optimisation landscape differs between A and B only in input-signal QUALITY,
   which makes the clean phase a high-fidelity, low-cost proxy for tuning. This
   claim is evidenced by the config table, which the dissertation should SHOW, not
   by any cited paper.

Defensible sentence for the write-up: "Tuning is performed on the clean phase,
which differs from the noisy phase only by sensor-noise injection (Table X); the
geometry, obstacles, reward and architecture are identical, so the clean phase is
a low-cost, high-fidelity proxy for the noisy optimisation landscape. Multi-fidelity
HPO on a cheaper proxy is a standard technique [Li 2017 Hyperband lineage;
Eimer 2023]."

### Protocol

1. **Establish the task works first - on vanilla PPO.** Prove the clean Phase A
   curriculum on the VANILLA PPO baseline first, on the committed
   `train_config.yaml` defaults (vanilla is the pathfinder - simplest agent, so a
   stall is unambiguously an env/reward bug, not an evidential interaction). Do NOT
   tune before the task is learnable - tuning a broken pipeline optimises noise. If
   Phase A will not converge on defaults, that is an environment/reward problem,
   not a tuning problem.
2. **Tune ONCE, on the vanilla baseline, on the clean Phase A task** (cheap - no
   noise pipeline per Optuna trial; reduced timesteps as a multi-fidelity proxy).
   Run `make docker-tune STAGE=4 BASELINE=configs/baselines/vanilla_ppo.yaml` - the
   `BASELINE=` is mandatory: omitting it tunes the `train_config.yaml` defaults,
   which are the FULL METHOD config, not vanilla. The per-baseline path leaves
   `train_config.yaml` untouched and writes to
   `logs/tuning/results/best_params_vanilla_ppo.yaml`; for share-from-vanilla,
   manually copy those values into `train_config.yaml` once (no auto-promotion flag
   by design). All four baselines then inherit that shared set. (Stretch goal if
   compute allows: one study per baseline. Fallback: tune the two evidential
   baselines separately only if they are unstable under vanilla's tuned LR.)
3. **Seed separation:** tuning studies use dedicated tuning seeds; final training +
   evaluation use a DISJOINT set of evaluation seeds never seen during tuning.
4. **Retrain the WHOLE A -> B curriculum** for each baseline with the single fixed
   shared set, on the evaluation seeds. One set, entire run - no mid-run change.
5. **Report** mean +/- std (or CI) across evaluation seeds per baseline, with a
   significance test between the full method and each baseline. Report the tuning
   budget (trials, search space, seeds).

### The honest caveat (state it, do not hide it)

Hyperparameters tuned on the clean phase MAY be slightly suboptimal for the noisy
phase. Two mitigations: (a) PPO hyperparameters (learning rate, clip range, GAE
lambda, entropy coef) mainly govern OPTIMISATION STABILITY, which transfers across
noise levels far better than task-specific quantities; (b) every baseline gets the
SAME treatment, so any mild suboptimality is UNIFORM across the ablation - it
cannot bias the relative comparison, which is the contribution. The alternative
(tune on the full A -> B task with each Optuna trial running the whole curriculum)
is more faithful but multiplies tuning cost by the full curriculum length x4
baselines - infeasible on a single GPU for an MSc timeline. Tuning on the clean
proxy is the coherent, affordable, citable choice; full-task tuning is the
documented (compute-permitting) alternative.

### What stays fixed across ALL baselines and both phases

Architecture-defining settings are NOT tuned and NOT varied: `net_arch`,
`activation`, `policy_type` (except where it IS the ablation), observation/action
dims, `include_covariance` / `include_obstacle_obs` (except where they ARE the
ablation). These define the ablation itself; tuning them would confound the
comparison. Resume across phases also requires them constant.

### Why tune per-baseline does NOT reintroduce unfairness

Tuning each baseline to its own optimum is the OPPOSITE of unfair: it gives every
condition its best chance, so the comparison is "best vanilla PPO vs best full
method", not "full method vs a handicapped baseline". The fairness control is the
seed separation (tuning != evaluation seeds), which stops any baseline from
winning by overfitting hyperparameters to the seeds it is scored on.
