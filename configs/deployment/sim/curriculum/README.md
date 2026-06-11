# Curriculum Stage Files

Single-phase ADR curriculum, six stages. Each `stage<N>.yaml` owns the per-stage
difficulty and is deep-merged over `configs/deployment/sim/env_config.yaml` when
selected with `make docker-train STAGE=N` (omitting the stage defaults to stage 1).
Strategy and rationale live in `documentation/CURRICULUM_PLAN.md` and
`documentation/detailed_notes/curriculum_design_principles.md`; this README documents
the file contract.

## The six stages

| Stage | Bays | Margin | Occupancy | Budget (decisions) | Primary axis |
|---|---|---|---|---|---|
| 1 | 4 (4 of 5 clusters) | -0.75 | 0.1 | 1.5M | bootstrap (from scratch) |
| 2 | 14 (3 per cluster; 2 from the 5-bay east block) | -0.55 | 0.1 | 1.5M | bay variety |
| 3 | all 47 | -0.40 | 0.1 | 2.0M | bay variety -> full lot |
| 4 | all 47 | -0.25 | 0.1 | 1.5M | margin -> strict (clean) |
| 5 | all 47 | -0.25 | 0.2-0.5 | 2.0M | occupancy |
| 6 | all 47 | -0.25 | 0.2-0.8 | 2.5M | occupancy full (final operating point) |

Design rules behind the table:

- **Every observation channel is live in every stage.** The GNSS Markov chain
  (`../gnss_noise_profiles.yaml`) is fixed and identical across stages, so the
  covariance features never degenerate; occupancy is always > 0 so LiDAR carries
  signal; every bay set spans multiple approach orientations so `dx/dy/dyaw` vary.
  `tests/test_curriculum_invariants.py` enforces this in CI.
- **One axis's range ramps per stage** (bays with a co-tightening margin in 1-4,
  occupancy in 5-6), so a stall is attributable to the axis that moved.
- **Stage 1 carries the only from-scratch skill** (all approach geometries at the
  loosest margin); stages 2-3 interpolate within mastered orientations; stage 4
  finalises the strict `STRICT_BAY_MARGIN` criterion in a clean lot; stages 5-6 ramp
  the avoidance task. Each stage's bay set is a superset of the previous stage's so
  the resumed policy keeps a success signal.
- **Occupancy is a per-episode range** (`bay_occupancy_min/max`); the low end stays
  at 0.2 so easy episodes keep the success signal alive within a stage.

## Key contract

Every stage spells out the same full difficulty key set:

- `use_extra_spawns` - false in every stage (single primary spawn).
- `bay_margin` - success acceptance margin in metres (negative inflates the box
  outward; -0.25 is `STRICT_BAY_MARGIN`, the evaluation criterion).
- `parking_scenarios.fixed_floor_plan` - `rectangle` (trapezoid/irregular held out
  for OOD evaluation).
- `parking_scenarios.fixed_gnss_tier` - `rtk_fixed` in every stage; episodes START
  clean and the stage-invariant GNSS Markov chain wanders from there.
- `parking_scenarios.allowed_bay_ids` - bay whitelist; omitted = all 47 bays.
  Cluster geometry in `configs/layouts/rectangle.yaml` (five spatial clusters,
  three distinct yaws): 0-11 (yaw 270, north row), 12-24 (yaw 90, south row),
  25-32 (yaw 270, upper row west block), 33-37 (yaw 270, upper row east block -
  separated from 25-32 by an open gap), 38-46 (yaw 180, east column).
- `parking_scenarios.bay_occupancy_min/max` - per-episode occupancy sampling range.
- `carla_sensors.lidar.noise.enabled` - true in every stage (constant realism
  floor).

## Override blocks

`standard_overrides` (vanilla_ppo, input_uncertainty) and `evidential_overrides`
(output_uncertainty, full_method) are selected by `policy_type` in
`_apply_stage_training_overrides()` and are initialised identical, splitting only on
observed evidential instability. Keys are allowlisted (`stage_timesteps`,
`learning_rate(_final)`, `ent_coef(_final)`, `clip_range`, `n_epochs`, `batch_size`,
`n_steps`, `target_kl`); architecture keys are rejected so a stage can never break
weight loading on resume.

Schedule conventions:

- `stage_timesteps` counts SB3 timesteps = policy decisions; each decision spans
  `action_repeat` sim ticks.
- `learning_rate` and `ent_coef` decay linearly to their `_final` values over the
  stage budget.
- Resumed stages (2+) restart the learning rate at roughly 3x the previous stage's
  final value, not the from-scratch rate, so a near-converged policy is not
  destabilised at a difficulty boundary.
- `ent_coef` decays to the 0.0005 floor in every stage; for the standard policy the
  action std is entropy-driven, so the floor bounds exploration noise at
  convergence.

## Running a stage

Stage 1 trains from scratch; every later stage resumes from the previous stage's
final checkpoint and must pass the same baseline the checkpoint was trained with
(observation dimensions differ between baselines):

```bash
make docker-train STAGE=2 BASELINE=configs/baselines/vanilla_ppo.yaml \
    CHECKPOINT=checkpoints/<previous_stage_run>/final_model
```
