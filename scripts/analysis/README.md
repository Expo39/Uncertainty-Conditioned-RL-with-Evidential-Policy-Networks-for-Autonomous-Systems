# scripts/analysis/

Analysis tooling that turns an evaluation run into the headline input-covariance and uncertainty results. None of these scripts are imported by the training pipeline - they consume the CSVs that `uncertainty_rl/evaluation/evaluate.py` writes and are invoked exclusively via `make` targets. Run them **after** an eval run has produced results under `outputs/raw/evaluation_results/seed_<N>/<baseline>/<leaf>/`.

The results tree is seed-nested: `make docker-eval` writes into `outputs/raw/evaluation_results/seed_<N>/`, where the seed is parsed from the `CHECKPOINT` leaf (or `SEED=`, default 42). The per-seed analyses therefore read one `seed_<N>/` sub-root and write under `outputs/raw_derived/<analysis>/seed_<N>/`; only `cross_seed.py` and `bundle.py` read the parent root that spans every seed. Three evaluation seeds are used: 42, 123 and 7.

Most scripts are host-side (CPU-only, read the eval CSVs from the project `.venv/`), and only the covariance probe needs torch and runs inside the training container.

**These modules write CSVs only** - none imports matplotlib or seaborn. Figures are rendered separately by [`figures/`](figures/), which reads the CSVs written here. That split is what lets a figure be redrawn without recomputing a statistic, and a statistic be recomputed without redrawing. `trace_tiers.py` and `uncertainty_verdict.py` are console diagnostics that write nothing at all.

Output tiers: raw eval CSVs in `outputs/raw/`, everything written here in `outputs/raw_derived/`, and the curated set assembled into `outputs/main_analysis/` by `bundle.py`.

## Quick reference

| Task | Command |
|------|---------|
| Cross-arm covariance contrast + caution tables | `make analyse-ablation [STAGE=6]` |
| Is the EKF covariance honest (std vs error)? | `make analyse-calibration [ARM=full_method]` |
| EKF-std vs evidential-epistemic safety-gate ROC | `make analyse-gate [STAGE=6]` |
| Causal "does the policy use covariance?" probe | `make docker-covariance-probe BASELINE=full_method CHECKPOINT=6_42_22062026-1502` |
| Epistemic-vs-aleatoric separation verdict | `make uncertainty-verdict EVAL_DIR=outputs/raw/evaluation_results/seed_42/<baseline>/<leaf>/without_wrapper` |
| Handover timing vs degradation onset | `make handover-timing [ARM=full_method]` |
| Pool all seeds into headline + per-seed robustness | `make analyse-cross-seed [STAGE=6]` |
| Per-tier episode breakdown from the demo traces | `make trace-tier-breakdown TRACE_DIR=<dir>` |
| Export TensorBoard scalars for the training-curve figure | `make training-curves [LOGS_ROOT=logs]` |
| Draw the headline figures | `make figures [FIG=gate_roc]` |
| Draw the per-run diagnostic panels | `make run-figures [RUN_DIR=<dir>]` |
| Assemble the curated `outputs/main_analysis/` set | `make analysis-bundle [STAGE=6] [BOOTSTRAP_SEED=42]` |

## Modules

### `ablation.py` (host-side)

Cross-arm contrast, and the home of the shared condition-scope filters every other module imports. Globs `outputs/raw/evaluation_results/seed_<N>/<baseline>/<leaf>/episode_records.csv` for the four arms, joins on condition, and computes:

- The covariance contrast deltas with 95% bootstrap CIs - `input_uncertainty - vanilla_ppo` (standard heads) and `full_method - output_uncertainty` (evidential heads), the single-variable covariance on/off tests. 10000 resamples, treatment and control resampled independently (a two-sample difference, not paired - the arms ran on separate episode draws), percentile interval at [2.5, 97.5]. The RNG seed is `--seed` (default 42) and is a reproducibility knob, not an experiment seed.
- The GNSS degradation slope per arm (`gnss_fixed -> gnss_degraded` success drop + position-error growth). **Only computed with `--keep-held-tiers`** (see below).
- Behaviour-by-std: final pos-error and the caution metrics binned into five quantile bins of EKF position std per arm (the mechanism - precision / caution under uncertainty).
- Caution slopes, contrast and levels: the per-arm Spearman of each caution metric (`mean_brake_cmd`, `mean_speed_moving_ms`, `mean_abs_vyaw_rads`, `mean_action_jerk`) against EKF position std, the cross-arm difference of those slopes (the causal read: caution attributable to *seeing* the covariance rather than to a harder episode), and the absolute operating level with its success / position-error payoff.

**Condition scope is set by three filters defined here**, so no table or figure can
widen it by a side path:

- `drop_held_tiers()` (`_HELD_TIER_CONDITIONS` = `gnss_fixed`, `gnss_degraded`) is
  applied at every load boundary in `ablation`, `cross_seed`, `gate_roc` and
  `calibration`, leaving five conditions in the raw derived CSVs.
- `drop_unreported()` (`_UNREPORTED_CONDITIONS` = `lidar_degraded`) is applied at the
  **figure and bundle** boundary only: it corrupts the obstacle channel alone, and the
  EKF never consumes LiDAR, so localisation std stays pinned at its RTK-fixed floor and
  the condition carries no localisation-uncertainty signal. This leaves the **four**
  reported conditions - `anchor_deployment`, `anchor_empty`, `ood_irregular_rtk_fixed`,
  `gnss_degrade_one_way`.
- `keep_varying()` (`_VARYING_CONDITIONS` = `anchor_deployment`, `anchor_empty`,
  `gnss_degrade_one_way`) narrows further for the behaviour and calibration readings,
  which need a std that moves *while the vehicle drives*. That excludes the held tiers
  and the OOD layout (held at RTK fixed).

Writes `condition_summary.csv`, `covariance_contrasts.csv`, `behaviour_by_std.csv`,
`caution_slopes.csv`, `caution_contrast.csv` and `caution_levels.csv` (plus
`degradation_slope.csv` only when the held tiers are kept) under
`outputs/raw_derived/ablation_analysis/seed_<N>/stage<S>/`. Run with
`make analyse-ablation [STAGE=6] [SEED=42] [CHECKPOINT=<leaf>]`.

### `calibration.py` (host-side)

Is the covariance HONEST? Reads `calibration_records.csv` (per-step predicted std vs actual GT-EKF error, written for every arm, and the EKF is identical across arms so any one suffices) and reports the std-vs-error correlation (Pearson and Spearman, per condition plus a pooled `varying_pooled` scope) and a binned mean-error-per-std-bin table over five quantile bins. A monotone rise means high std really does mean high error, so conditioning on it is justified - the precondition for the whole approach. Spearman is computed as Pearson on ranks, so the script needs no scipy (absent from both the host `.venv/` and the training container).

The headline verdict is read off the pooled **varying** position row only: a held tier
pins the std all episode, so there is no spread to correlate against. Each row carries a
`std_spread` (p90 - p10) column and a `varying` / `held` group tag that expose this
directly; a pooled Spearman at or above 0.2 is reported as honest. Writes
`calibration_correlations.csv` and `calibration_binned.csv` under
`outputs/raw_derived/calibration_analysis/seed_<N>/<baseline>/<leaf>/`. Run with
`make analyse-calibration [ARM=<name>] [CHECKPOINT=<leaf>]`.

### `gate_roc.py` (host-side)

Safety-gate comparison. Sweeps an abort threshold over two signals and scores
failure-catch against false-abort (ROC AUC):

- `ekf_std_pos_max_m` - the EKF position std, available to EVERY arm since the EKF
  always runs.
- `max_epistemic` - the evidential epistemic, evidential arms only.

A higher epistemic AUC on the same arm means the policy's own confidence beats the
EKF-std gate a covariance-blind system could already build.

A failure is any of `collision`, `out_of_bounds`, `near_miss` or `stuck`. `handoff`
episodes are excluded, having already aborted, so there is no ground-truth outcome to
score. The AUC is a trapezoidal area computed directly, needing neither `np.trapz`
(removed in NumPy 2.x) nor sklearn.

Writes `gate_auc.csv` under `outputs/raw_derived/gate_analysis/seed_<N>/stage<S>/`.
Run with `make analyse-gate [STAGE=6]`.

**Note** that `bundle.py` scores a different gate signal for the curated set: the
per-episode max of the per-step epistemic + aleatoric **sum**, since the deployed safety
layer thresholds the total at each step. Summing the two per-episode maxima is a
different quantity, as the channels need not peak on the same step.
It also reports each AUC with a 95% stratified bootstrap interval (failures and
successes resampled separately, so every resample keeps both classes; 10,000 resamples
on `--bootstrap-seed`, default 42), both pooled over the varying conditions and within
each condition. That AUC is the tie-aware Mann-Whitney form (`_bootstrap_auc_ci`), since
the EKF std pins at its floor on many episodes and the trapezoidal sweep would depend on
how tied scores happen to sort.

### `covariance_probe.py` (in-container, torch)

Causal probe. Holds an observation fixed and sweeps ONLY the covariance block, reporting the action delta: a growing delta means the policy conditions on the input, a flat one means it ignores it. Two modes: synthetic (default, one hand-built near-bay obs) and on-manifold (`--real-obs <file.npy>`, replays the real observations the eval loop dumps when `EVAL_DUMP_OBS` is set).

The sweep runs the four GNSS tiers (`rtk_fixed`, `rtk_float`, `standalone`, `degraded`)
through the same normalisation `build_observation()` applies, so the probe drives the
covariance block exactly as the env would. It is the one module here that imports torch,
hence the container. Run with:

```bash
make docker-covariance-probe BASELINE=<name> CHECKPOINT=<leaf> \
    [REAL_OBS=outputs/raw/evaluation_results/seed_42/<baseline>/<leaf>/real_observations.npy]
```

### `uncertainty_verdict.py` (host-side)

Reads an eval run's `per_step_records.csv` and reports, per condition, epistemic / aleatoric / epi-over-ale ratio / implied nu / frac(epi>ale), then a CLEAN-vs-HARD separation verdict (does epistemic rise RELATIVELY on novel / degraded conditions). The verdict is the ratio of the mean HARD ratio to the mean CLEAN one: at or above 1.5x it separates, at or above 1.1x the separation is weak, below that it is flat. On the single-head NIG actor the ratio is flat (epistemic = aleatoric / nu, and RL leaves nu unsupervised, so the two channels are one signal), and this confirms it from data.

It is the only module here that prints to stdout without writing a CSV, and unlike the
others it takes an explicit dir rather than discovering arms. The run needs an evidential
head and `EVAL_PER_STEP_CAP > 0` for `per_step_records.csv` to exist at all:

```bash
make uncertainty-verdict EVAL_DIR=outputs/raw/evaluation_results/seed_42/<baseline>/<leaf>/without_wrapper
```

### `handover_timing.py` (host-side)

When does the safety wrapper hand over, relative to the degradation onset? Turns the per-episode handover-timing columns (`handoff_step`, `degraded_onset_step`) into a per-condition latency table. The claim is about TIMING, not rate, and is only ever compared within an onset regime:

- **spawn** - degraded/novel from episode start, with latency `handoff_step` (steps from spawn).
- **switch** - starts clean and drifts into the degraded tier mid-episode, with latency `handoff_step - degraded_onset_step` (steps after the drift crossing).
- **none** - clean/in-distribution, where a LOW handover fraction is the desired result.

In the switch regime a handover that precedes the drift crossing is not onset-driven, so
it is excluded from the latency summary while still counting towards the handover
fraction. Handovers only fire with the wrapper on, so this is the one analysis that
prefers the `with_wrapper` variant. Writes `handover_timing.csv` (per condition: regime,
episode count, handoff fraction, median and p90 latency) under
`outputs/raw_derived/handover_timing/seed_<N>/<baseline>/<leaf>/`. Run with
`make handover-timing [ARM=<name>] [CHECKPOINT=<leaf>]`.

### `cross_seed.py` (host-side)

Pools EVERY seed's stage-`STAGE` eval into the seed-robust headline. The per-seed analyses each read one `seed_<N>/` tree, and a cross-arm difference on a single seed is indistinguishable from seed luck (Henderson et al. 2017), so this aggregator produces two reads side by side:

- **POOLED** - concatenate every seed's per-episode records into one sample and run the EXISTING statistics (the ablation bootstrap contrast, the gate ROC AUC, the calibration rank correlation, the handover-timing table) on the ~3x larger pool: the precision-of-effect headline.
- **PER-SEED ROBUSTNESS** - per arm and condition, the mean and `[min, max]` of each metric across seeds, plus the same range over each contrast delta. This is the honest cross-seed-stability check: pooling per-episode records and bootstrapping the pool treats seed as a fixed nuisance, so the pooled CI reflects episode-level variance but under-represents between-seed variance. An effect smaller than a rival's range is not robust, whatever the pooled CI says.

It re-implements no statistics: the per-analysis modules expose pure DataFrame functions,
so a pooled frame with an added `seed` column flows through them untouched. Two source
CSVs are not arm-mapped and are therefore pinned to one arm for reproducibility -
calibration to `vanilla_ppo` (the EKF is identical across arms, so any one characterises
the filter) and handover timing to `full_method` with the `with_wrapper` variant (the
wrapper only fires on an evidential head).

Reads `--results-root outputs/raw/evaluation_results` (the PARENT of the `seed_<N>/`
trees, never a `seed_<N>/` dir) and writes the pooled `pooled_*.csv`, `per_seed_summary.csv`
and `seed_robustness*.csv` under
`outputs/raw_derived/cross_seed_analysis/all_seeds/stage<S>/`. CSVs only - the figures over
this pool are drawn separately by `figures/build.py`. Run with
`make analyse-cross-seed [STAGE=6]`.

### `trace_tiers.py` (host-side)

Resolves demo-trace outcomes by localisation tier, separating failure because the task
was hard from failure despite clean localisation.

Reads the `episode_*.csv` files in one demo-trace dump, reduces each to a success flag, a
final position error and the episode's worst TRUE localisation error
`||ekf_xy - gt_xy||`, then bins outcomes into four bands (`<=0.36`, `<=1.80`, `<=5.0`,
`>5.0` m) following the per-tier `metric_stddev_m` design floors.

True error is used in preference to the reported EKF std, because the posterior is damped
by IMU fusion and saturates near 1.3 m even where the estimate is metres off, which
understates the tier. The same table is printed against the reported std as a secondary
view, precisely to expose that gap.

Also reports the Pearson correlation of per-step aleatoric against the reported
`ekf_std_x`: near zero means the head's uncertainty does not track localisation
uncertainty.

Unlike the other modules this one **prints only and writes no CSV**; it is a diagnostic,
not a source of a reported number. It takes a demo-trace dump, not an eval results dir:
`make trace-tier-breakdown TRACE_DIR=outputs/raw/demo_traces/<baseline>/<leaf>/<timestamp>`
(the variable is required - the target fails with a usage message if it is unset).

### `tb_curves.py` (host-side)

The one reading not sourced from the evaluation CSVs. It parses the TensorBoard event
files under `logs/<arm>/<stage>_<seed>_<timestamp>/` directly (via
`scripts/diagnostics/tb_read.py`) and exports `env/success_rate` and
`env/collision_rate`.

Every curriculum stage is laid end to end on one cumulative decision axis, each stage's
offset taken from its longest seed so all arms share an axis and the boundaries align.
Curves are averaged across seeds at matched steps, smoothed once with an exponential
moving average at 0.9 (the TensorBoard UI default) and thinned to an even stride of 330
points.

Writes `training_curves.csv` and `training_stage_bounds.csv` to
`outputs/raw_derived/training/`, which the `training_curves` figure reads. Run with
`make training-curves [LOGS_ROOT=logs]`.

### `bundle.py` (host-side)

Assembles `outputs/main_analysis/` - the curated set. Recomputes each summary from the per-episode records through the shared scope filters (`drop_unreported`, `keep_varying`) rather than copying, so every cell traces back to raw data. Writes nine tables into `summaries/` (success by condition, position error, covariance contrasts, conditioning analysis, seed unanimity, gate AUC with bootstrap intervals, calibration, behaviour bands, per-seed summary), copies the pooled cross-seed CSVs verbatim into `values/`, and writes a `MANIFEST.md` naming the source of each artefact and stating the reported scope.

Two details matter for reading the numbers. Position error is reported as both a mean
(from the pooled summary) and a median (recomputed from the per-episode records, the only
place it exists): they disagree where an arm parks more often and its surviving failures
end further out, so both are carried rather than one standing for the other. The
conditioning correlations are cut WITHIN each condition and then averaged over the three
varying ones, so a differing condition mix cannot masquerade as a behavioural response.

`summaries/` and `values/` are rebuilt from scratch each run; `figures/` is left alone,
because `make figures` writes straight into `main_analysis/figures/`. Run
`make figures` first, then `make analysis-bundle [STAGE=6] [BOOTSTRAP_SEED=42]`, so the manifest lists the
figures that are actually present.

### `figures/` (host-side)

Every rendered figure, and nothing else: these modules read CSVs and draw them, never recomputing a statistic. All of them import `scripts/figure_style.py`, the single source of rcParams, palette and legend treatment - never set those at a call site.

- `build.py` - the single entry point for the headline set, drawn into
  `outputs/main_analysis/figures/`. Nine figures, each id also its output filename stem:
  `ablation_by_condition`, `behaviour_by_std`, `degradation_tiers`, `ekf_calibration`,
  `ekf_sawtooth`, `gate_roc`, `lot_layouts`, `seed_robustness`, `training_curves`. Run
  `make figures` for all of them or `make figures FIG=gate_roc` for one. A figure whose
  input is missing is skipped with a message rather than aborting the run.
- `degradation_tiers.py` - per-arm success across TRUE-localisation tiers, pooled over
  seeds. Episodes are banded into equal-population terciles of their MEAN true error
  (mean, not max, so one transient spike cannot make the axis non-monotone), since the
  live chain rarely holds one integer tier for a whole episode.
- `ekf_sawtooth.py` - the reported EKF sigma_x sawtooth over one episode, with the true
  `gnss_tier` Markov state shaded behind it. Reads one demo-trace CSV, no smoothing.
- `lot_layouts.py` - both lot layouts drawn from the generated layout YAMLs; no geometry
  is recomputed.
- `training_curves.py` - the two stacked training panels, drawn from the CSVs
  `../tb_curves.py` exports. Nothing is re-averaged or re-smoothed here.
- `run_figures.py` - the per-run diagnostic panels (a 2x2 summary and an outcome-share
  breakdown) rebuilt from one run's `evaluation_results.csv`, written into
  `outputs/raw_derived/per_run_figures/`. Distinct from the pooled set: these summarise
  one run in isolation and are never pooled, so they carry every eval condition.
  `make run-figures [RUN_DIR=<dir>]`.

Note that `tb_curves.py` sits one level up in `scripts/analysis/`, not here: it writes a
CSV rather than drawing, so it belongs on the analysis side of the boundary.

### `_discovery.py` (shared helper, not a Make target)

Single source of truth for locating per-run CSVs under the nested `<baseline>/<leaf>/<wrapper_variant>/` results tree, so a layout change is a one-file fix. Maps each CSV back to its arm name, prefers the `without_wrapper` variant for uncertainty/behaviour reads (the wrapper caps throttle and corrupts the free-running signal), and falls back to the legacy two-level layout. Imported by the analysis scripts, never run directly. The public helpers:

- `discover_records()` - every CSV of a given name, sorted by variant preference then
  newest first, optionally restricted to one arm, leaf or curriculum stage.
- `discover_arm_csvs()` - one best CSV per arm, taken as the first hit in that order.
- `arm_leaf_subpath()` - the `<baseline>/<leaf>` sub-path of a discovered CSV, so
  per-arm output mirrors the eval's own nesting.
- `seed_roots()` - the `seed_<N>/` sub-roots of an output root, or the root itself when
  there is no seed nesting, so `cross_seed.py` can loop the seeds and reuse the per-seed
  discovery on each.
- `stage_leaf()` and `pooled_frame()` - direct addressing by (seed, arm, stage, variant),
  used by `bundle.py` and `figures/build.py` to pool one named CSV across every seed and
  arm in a single call.

**Pin the arm and the stage for any quoted quantity.** Unpinned, discovery falls back to
whichever matching file has the newest mtime, so a reported statistic can silently change
which run it describes.

## Conventions

- **Host-side scripts are read-only**: they consume the eval CSVs and write only the report files passed via `--output-dir`. No config/checkpoint mutation.
- **Arm discovery** uses the nested `<baseline>/<leaf>` output layout via `_discovery.py`, and the baseline (arm) name is taken from the directory, so no per-run flag is needed. The most recently modified leaf wins when an arm has several, so pin a specific run with `CHECKPOINT=<leaf>` or, better for a cross-arm read, `STAGE=<N>` - which compares every arm at the same curriculum stage rather than at each arm's own newest leaf.
- **Column schema** comes from `episode_records.csv` (written by `uncertainty_rl/evaluation/evaluate.py`): `condition`, `success`, `final_pos_error_m`, `outcome`, `ekf_std_pos_max_m`, `max_epistemic`, ... Adding a metric there is additive, but do not remove columns these scripts read. The caution columns (`mean_speed_moving_ms`, `mean_brake_cmd`, `mean_abs_vyaw_rads`, `mean_action_jerk`) were added later, so the modules aggregate only those actually present and emit no caution tables for an older CSV.
- **Every script here has a Make target** - add one before adding a script. `_discovery.py` is the one exception: it is an imported helper, not an entry point. The host-side analyses use the project `.venv/`, while `docker-covariance-probe` execs into the training container (where torch lives).
- **No hyperparameter optimisation was run.** Every reported result comes from the committed defaults in `configs/train_config.yaml`, so nothing here reads a tuning study.

## See also

- [scripts/README.md](../README.md) - all Make targets overview
- [scripts/analysis/CLAUDE.md](CLAUDE.md) - the working contract for this directory
- [uncertainty_rl/evaluation/](../../uncertainty_rl/evaluation/) - `evaluate.py`, the condition sweep that writes the CSVs these scripts consume
- [configs/eval_config.yaml](../../configs/eval_config.yaml) - the seven eval conditions, of which four carry reported results
- [docs/AntonioGaldes_Dissertation.pdf](../../docs/AntonioGaldes_Dissertation.pdf) - canonical for the methodology; this README covers the implementation
