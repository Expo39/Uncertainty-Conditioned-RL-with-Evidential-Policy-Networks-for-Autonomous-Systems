# Curriculum Stage Files

Single-phase ADR curriculum, six stages. Each `stage<N>.yaml` owns the per-stage
difficulty, selected with `make docker-train STAGE=N` (which passes `--stage N`; omitting
it defaults to stage 1). This README documents the file contract - the per-stage
difficulty keys and the per-policy override blocks each stage owns.

## How a stage is applied

A stage file is consumed in two separate passes, because its difficulty keys are
environment settings while its override blocks are training hyperparameters:

| Pass | Function (in `train_ppo.py`) | What it takes | Precedence |
|---|---|---|---|
| Environment | `load_env_config()` | Everything except the two `*_overrides` blocks | Deep-merged onto `env_config.yaml` **before** the `sensor < agent < env` merge, so a stage key wins over the base env value |
| Training | `_apply_stage_training_overrides()` | `standard_overrides` or `evidential_overrides`, picked by `policy_type` | Applied **after** `merge_configs()` and `apply_baseline()`, so a stage schedule wins over `train_config.yaml` |

`load_env_config()` drops the legacy `training_overrides` key from the env dict so it
never reaches the environment constructor, and a missing `stage<N>.yaml` raises
`FileNotFoundError` rather than silently training on the base config. The full chain is
`sensor_config < agent_config < env_config (+ stage) < train_config (+ baseline)`, with
the stage schedule layered last - see
[configs/README.md](../../../README.md#how-configs-are-loaded).

## The six stages

| Stage | Bays | `bay_margin` (m) | Occupancy | Budget (decisions) | Primary axis |
|---|---|---|---|---|---|
| 1 | 4 (one each from 4 of the 5 clusters) | -0.75 | 0.0-0.1 | 1.5M | bootstrap (from scratch) |
| 2 | 15 (3/3/4/2/3 over the five clusters) | -0.55 | 0.0-0.1 | 1.5M | bay variety |
| 3 | all 47 (`allowed_bay_ids` omitted) | -0.40 | 0.0-0.1 | 2.0M | bay variety -> full lot |
| 4 | all 47 | -0.25 | 0.0-0.1 | 1.5M | margin -> strict (clean) |
| 5 | all 47 | -0.25 | 0.0-0.5 | 2.0M | occupancy |
| 6 | all 47 | -0.25 | 0.0-0.8 | 1.5M | occupancy full (final operating point) |

Stage transitions are scheduled on budget, not gated on a measured success rate: each
stage runs its `stage_timesteps` and the next stage resumes from the checkpoint that
leaves. Section 3.7 of
[`docs/AntonioGaldes_Dissertation.pdf`](../../../../docs/AntonioGaldes_Dissertation.pdf)
is the canonical account.

Design rules behind the table:

- **Every observation channel is live in every stage.** The GNSS Markov chain
  (`../gnss_noise_profiles.yaml`) is fixed and identical across stages, so the
  covariance features never degenerate, occupancy is always above 0 so LiDAR carries
  signal, and every bay set spans multiple approach orientations so `dx/dy/dyaw` vary.
  `tests/test_curriculum_invariants.py` asserts all of this; it parses YAML only, so it
  needs no torch, but it runs under `make docker-test-unit` rather than in CI (CI runs
  lint, typecheck and an import check only).
- **One axis's range ramps per stage** (bays with a co-tightening margin in 1-4,
  occupancy in 5-6), so a stall is attributable to the axis that moved.
- **Stage 1 carries the only from-scratch skill**: its four bays already span all three
  approach orientations (yaw 270, 90 and 180) at the loosest margin. Stages 2-3
  interpolate within those mastered orientations, stage 4 finalises the strict
  `STRICT_BAY_MARGIN` criterion in a clean lot, and stages 5-6 ramp the avoidance task.
  Each stage's bay set is a superset of the previous stage's so the resumed policy keeps
  a success signal.
- **Occupancy is a per-episode range** (`bay_occupancy_min/max`), and the low end stays
  at 0.0 (empty in-distribution) so easy episodes keep the success signal alive
  within a stage, while only the max ramps (0.1 -> 0.5 -> 0.8).

## Key contract

Every stage spells out the same full difficulty key set:

- `use_extra_spawns` - false in every stage (single primary spawn).
- `bay_margin` - success acceptance margin in metres (negative inflates the box
  outward, and -0.25 is `STRICT_BAY_MARGIN`, the evaluation criterion).
- `parking_scenarios.fixed_floor_plan` - `rectangle` (trapezoid/irregular held out
  for OOD evaluation).
- `parking_scenarios.fixed_gnss_tier` - OMITTED in every stage, so the per-episode
  START tier is SAMPLED from the init weights in `gnss_noise_profiles.yaml` (episodes
  can begin in any fix state, the realistic arrival case). The stage-invariant GNSS
  Markov chain then wanders from there.
- `parking_scenarios.allowed_bay_ids` - bay whitelist of `perpendicular_<N>` ids, where
  omitting it allows all eligible bays. `configs/layouts/rectangle.yaml` defines 49 bays,
  of which the two `motorcycle_*` bays are flagged `always_empty` and never targeted,
  leaving **47 eligible**. An id absent from the floor plan raises at env construction.
  Cluster geometry (five spatial clusters, three distinct yaws): 0-11 (yaw 270, centre
  row at y=1.25), 12-24 (yaw 90, south row at y=-16.65), 25-32 (yaw 270, north row west
  block at y=19.15), 33-37 (yaw 270, north row east block at the same y, separated from
  25-32 by an open gap in x), 38-46 (yaw 180, east column).
- `parking_scenarios.bay_occupancy_min/max` - per-episode occupancy sampling range.
- `carla_sensors.lidar.noise.enabled` - true in every stage (constant realism
  floor).

## Override blocks

`standard_overrides` (vanilla_ppo, input_uncertainty) and `evidential_overrides`
(output_uncertainty, full_method) are selected by `policy_type` in
`_apply_stage_training_overrides()`. The two blocks began identical and were split only
where evidential instability was observed, so they now differ in `ent_coef_final` in
every stage, and in the `ent_coef` starting value from stage 4 onward.

Keys are allowlisted - `stage_timesteps`, `learning_rate`, `learning_rate_final`,
`ent_coef`, `ent_coef_final`, `clip_range`, `n_epochs`, `batch_size`, `n_steps`,
`target_kl` - and anything else raises. The allowlist deliberately excludes every
architectural key (`net_arch`, `activation`, `policy_type`, `include_covariance`,
`include_obstacle_obs`, and hence the observation and action dimensions), so a stage can
never change the policy shape and break weight loading on resume. No stage file sets any
of them; `tests/test_curriculum_invariants.py` mirrors the allowlist and fails under
`make docker-test-unit` if one appears.

Only `stage_timesteps`, `learning_rate(_final)` and `ent_coef(_final)` are actually set;
the remaining allowlisted keys stay at their `train_config.yaml` values in every stage.

| Stage | `stage_timesteps` | `learning_rate` -> `_final` | `ent_coef` -> `_final` (standard) | `ent_coef` -> `_final` (evidential) |
|---|---|---|---|---|
| 1 | 1 500 000 | 3.0e-4 -> 5e-5 | 0.003 -> 0.0005 | 0.003 -> 0.0015 |
| 2 | 1 500 000 | 1.5e-4 -> 5e-5 | 0.0015 -> 0.0005 | 0.0015 -> 0.0015 |
| 3 | 2 000 000 | 1.5e-4 -> 4e-5 | 0.0015 -> 0.0005 | 0.0015 -> 0.0015 |
| 4 | 1 500 000 | 1.2e-4 -> 4e-5 | 0.001 -> 0.0005 | 0.002 -> 0.0015 |
| 5 | 2 000 000 | 1.2e-4 -> 3e-5 | 0.001 -> 0.0005 | 0.002 -> 0.0015 |
| 6 | 1 500 000 | 1.0e-4 -> 2e-5 | 0.001 -> 0.0005 | 0.002 -> 0.0015 |

The learning-rate columns are shared: outside the two entropy keys the standard and
evidential blocks are identical in every stage, which the invariants test asserts.

Schedule conventions:

- `stage_timesteps` counts SB3 timesteps, which are policy decisions. Each decision spans
  `action_repeat` sim ticks.
- `learning_rate` and `ent_coef` decay linearly to their `_final` values over the
  stage budget.
- Resumed stages (2+) restart the learning rate at roughly 3x the previous stage's
  final value, not the from-scratch rate, so a near-converged policy is not
  destabilised at a difficulty boundary.
- `ent_coef` decays to a floor of 0.0005 under `standard_overrides` and 0.0015 under
  `evidential_overrides`, in every stage. The standard policy's action std is
  entropy-driven, so its floor bounds exploration noise at convergence. The evidential
  policy's action std is the NIG aleatoric instead, bounded below by the
  stage-invariant `evidential.aleatoric_floor` in `train_config.yaml`, which is why the
  two floors differ.

## Running a stage

Stage 1 trains from scratch, and every later stage resumes from the previous stage's
final checkpoint and must pass the same baseline the checkpoint was trained with
(observation dimensions differ between baselines):

`CHECKPOINT` is the previous stage's run leaf, from which the recipe reconstructs
`checkpoints/<baseline>/<leaf>` and passes it as `--resume-from`. Both variables take
bare names rather than paths, as described in [COMMANDS.md](../../../../COMMANDS.md).
`train_ppo.py` names each leaf `<stage>_<seed>_<DDMMYYYY-HHMM>`, so a stage-1 run on seed
42 leaves a leaf such as `1_42_19062026-0120`.

```bash
make docker-train STAGE=1 BASELINE=vanilla_ppo
make docker-train STAGE=2 BASELINE=vanilla_ppo CHECKPOINT=1_42_19062026-0120
```

Omitting `STAGE` falls back to `DEFAULT_STAGE = 1` in `train_ppo.py`, and omitting
`BASELINE` falls back to `full_method`. Spell both out so a run is unambiguous.
