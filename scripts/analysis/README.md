# scripts/analysis/

Analysis tooling that turns an evaluation run into the headline input-covariance and uncertainty results. None of these scripts are imported by the training pipeline - they consume the CSVs that `uncertainty_rl/evaluation/evaluate.py` writes and are invoked exclusively via `make` targets. Run them **after** an eval run has produced results under `outputs/raw/evaluation_results/<baseline>/<leaf>/`.

Most scripts are host-side (CPU-only, read the eval CSVs from the project `.venv/`); only the covariance probe needs torch and runs inside the training container.

**These modules write CSVs only** - none imports matplotlib or seaborn. Figures are rendered separately by [`figures/`](figures/), which reads the CSVs written here. That split is what lets a figure be redrawn without recomputing a statistic, and a statistic be recomputed without redrawing.

Output tiers: raw eval CSVs in `outputs/raw/`, everything written here in `outputs/raw_derived/`, and the curated set assembled into `outputs/main_analysis/` by `bundle.py`.

## Quick reference

| Task | Command |
|------|---------|
| Cross-arm covariance contrast + degradation slope | `make analyse-ablation` |
| Is the EKF covariance honest (std vs error)? | `make analyse-calibration [ARM=full_method]` |
| EKF-std vs evidential-epistemic safety-gate ROC | `make analyse-gate` |
| Causal "does the policy use covariance?" probe | `make docker-covariance-probe BASELINE=full_method CHECKPOINT=seed42_11062026-0628` |
| Epistemic-vs-aleatoric separation verdict | `make uncertainty-verdict EVAL_DIR=outputs/raw/evaluation_results/<baseline>/<leaf>/without_wrapper` |
| Handover timing vs degradation onset | `make handover-timing [ARM=full_method]` |
| Pool all seeds into headline + per-seed robustness | `make analyse-cross-seed [STAGE=6]` |

## Modules

### `ablation.py` (host-side)

Cross-arm contrast. Globs `outputs/raw/evaluation_results/<baseline>/<leaf>/episode_records.csv` for the four arms, joins on condition, and computes:

- The covariance contrast deltas with 95% bootstrap CIs - `input_uncertainty - vanilla_ppo` (standard heads) and `full_method - output_uncertainty` (evidential heads), the single-variable covariance on/off tests.
- The GNSS degradation slope per arm (`gnss_fixed -> gnss_degraded` success drop + position-error growth). **Only computed with `--keep-held-tiers`** (see below).
- Behaviour-by-std: final pos-error and approach speed binned by EKF position std per arm (the mechanism - precision / caution under uncertainty).

**Held tiers are dropped by default.** `gnss_fixed` and `gnss_degraded` each pin one
fix state for the whole episode, so neither degrades *within* an episode and the
"slope" between them is a between-condition difference rather than degradation any
arm rides through; the EKF also suppresses a static raw fault, so the two do not
separate at the policy's input and the slope is flat by construction (see
`documentation/detailed_notes/degraded_gnss_is_not_a_blackout.md`). Every table and
analysis therefore covers the **five** retained conditions. The
graceful-degradation evidence instead comes from the live anchor chain banded by true
error (`scripts/analysis/figures/degradation_tiers.py`) and the one-way drift. Pass
`--keep-held-tiers` to restore the old seven-condition behaviour and the slope.

Writes `condition_summary.csv`, `covariance_contrasts.csv`, `behaviour_by_std.csv`
(plus `degradation_slope.csv` only when the held tiers are kept). Run with
`make analyse-ablation`.

### `calibration.py` (host-side)

Is the covariance HONEST? Reads `calibration_records.csv` (per-step predicted std vs actual GT-EKF error, written for every arm; the EKF is identical across arms so any one suffices) and reports the std-vs-error rank correlation (overall + per condition) and a binned mean-error-per-std-bin table. A monotone rise means high std really does mean high error, so conditioning on it is justified - the precondition for the whole approach. Spearman is computed via ranks (no scipy dependency). Writes `calibration_correlations.csv` and `calibration_binned.csv`. Run with `make analyse-calibration [ARM=<name>]`.

### `gate_roc.py` (host-side)

Safety-gate comparison. From the same CSVs, sweeps an abort threshold over two signals and scores failure-catch vs false-abort (ROC AUC): the EKF position std (`ekf_std_pos_max_m`, available to EVERY arm since the EKF always runs) versus the evidential epistemic (`max_epistemic`, evidential arms only). A higher epistemic AUC on the same arm means the policy's own confidence beats the EKF-std gate a covariance-blind system could build. `handoff` episodes are excluded (already aborted). Writes `gate_auc.csv`. Run with `make analyse-gate`.

### `covariance_probe.py` (in-container, torch)

Causal probe. Holds an observation fixed and sweeps ONLY the covariance block, reporting the action delta: a growing delta means the policy conditions on the input, a flat one means it ignores it. Two modes: synthetic (default, one hand-built near-bay obs) and on-manifold (`--real-obs <file.npy>`, replays the real observations the eval loop dumps when `EVAL_DUMP_OBS` is set). Run with:

```bash
make docker-covariance-probe BASELINE=<name> CHECKPOINT=<leaf> \
    [REAL_OBS=outputs/raw/evaluation_results/<baseline>/<leaf>/real_observations.npy]
```

### `uncertainty_verdict.py` (host-side)

Reads an eval run's `per_step_records.csv` and reports, per condition, epistemic / aleatoric / epi-over-ale ratio / implied nu / frac(epi>ale), then a CLEAN-vs-HARD separation verdict (does epistemic rise RELATIVELY on novel / degraded conditions). On the single-head NIG actor the ratio is flat (the two channels are one signal); this confirms it from data. Unlike the other scripts it takes an explicit dir, not arm discovery:

```bash
make uncertainty-verdict EVAL_DIR=outputs/raw/evaluation_results/<baseline>/<leaf>/without_wrapper
```

### `handover_timing.py` (host-side)

When does the safety wrapper hand over, relative to the degradation onset? Turns the per-episode handover-timing columns (`handoff_step`, `degraded_onset_step`) into a per-condition latency table. The claim is about TIMING, not rate, and is only ever compared within an onset regime:

- **spawn** - degraded/novel from episode start; latency = `handoff_step` (steps from spawn).
- **switch** - starts clean and drifts into the degraded tier mid-episode; latency = `handoff_step - degraded_onset_step` (steps after the drift crossing).
- **none** - clean/in-distribution; a LOW handover fraction here is the desired result.

Run with `make handover-timing [ARM=<name>]`.

### `cross_seed.py` (host-side)

Pools EVERY seed's stage-`STAGE` eval into the seed-robust headline. The per-seed analyses each read one `outputs/<analysis>/seed_<N>/` tree; a cross-arm difference on a single seed is indistinguishable from seed luck (Henderson et al. 2017), so this aggregator produces two reads side by side: (a) POOLED - concatenate every seed's per-episode records into one sample and run the EXISTING statistics (the ablation bootstrap contrast, the gate ROC AUC, the calibration rank correlation) on the ~3x larger pool, the precision-of-effect headline; (b) PER-SEED ROBUSTNESS - per arm, the mean and `[min, max]` of each metric across seeds, the honest cross-seed-stability check the curriculum plan mandates (a bootstrap on a fixed pool of three runs under-represents between-seed variance). It re-implements no statistics: the per-analysis modules expose pure DataFrame functions, so a pooled frame with an added `seed` column flows through them untouched. Reads `--results-root outputs/raw/evaluation_results` (the PARENT of the `seed_<N>/` trees) and writes the pooled `pooled_*.csv` + per-seed `seed_robustness*.csv` under `outputs/raw_derived/cross_seed_analysis/all_seeds/stage<S>/` (the cross-seed sibling of the per-seed `seed_<N>/stage<S>/`). Run with `make analyse-cross-seed [STAGE=6]`.

### `bundle.py` (host-side)

Assembles `outputs/main_analysis/` - the curated set. Recomputes each summary from the per-episode records through the shared scope filters (`drop_unreported`, `keep_varying`) rather than copying, so every cell traces back to raw data; copies the pooled CSVs into `values/`; and writes a `MANIFEST.md` naming the source of each artefact. Figures are not copied - `make figures` writes them straight into `main_analysis/figures/`. Run with `make analysis-bundle [STAGE=6]`.

### `figures/` (host-side)

Every rendered figure. `build.py` draws the headline set into `outputs/main_analysis/figures/` (`make figures [FIG=gate_roc]`); `run_figures.py` draws the per-run diagnostic panels into `outputs/raw_derived/per_run_figures/` (`make run-figures`); `tb_curves.py` (one level up) exports the TensorBoard scalars the training-curve figure reads. All of them import `scripts/figure_style.py`, the single source of rcParams, palette and legend treatment.

### `_discovery.py` (shared helper, not a Make target)

Single source of truth for locating per-run CSVs under the nested `<baseline>/<leaf>/<wrapper_variant>/` results tree. Maps each CSV back to its arm name, prefers the `without_wrapper` variant for uncertainty/behaviour reads (the wrapper caps throttle and corrupts the free-running signal), and falls back to the legacy two-level layout. `seed_roots()` returns the `seed_<N>/` sub-roots of an output root (or the root itself when there is no seed nesting) so `cross_seed.py` can loop the seeds and reuse the per-seed discovery on each. Imported by the analysis scripts; never run directly.

## Conventions

- **Host-side scripts are read-only**: they consume the eval CSVs and write only the report files/figures passed via `--output-dir`. No config/checkpoint mutation.
- **Arm discovery** uses the nested `<baseline>/<leaf>` output layout via `_discovery.py`; the baseline (arm) name is taken from the directory, so no per-run flag is needed. The most recently modified leaf wins when an arm has several; pin a specific run with `CHECKPOINT=<leaf>`.
- **Column schema** comes from `episode_records.csv` (written by `uncertainty_rl/evaluation/evaluate.py`): `condition`, `success`, `final_pos_error_m`, `outcome`, `ekf_std_pos_max_m`, `max_epistemic`, ... Adding a metric there is additive; do not remove columns these scripts read.
- **Every script has a Make target** - add one before adding a script. The host-side analyses use the project `.venv/`; `docker-covariance-probe` execs into the training container (where torch lives).

## See also

- [scripts/README.md](../README.md) - all Make targets overview
- [scripts/analysis/CLAUDE.md](CLAUDE.md) - the local working contract for this directory
- [uncertainty_rl/evaluation/](../../uncertainty_rl/evaluation/) - `evaluate.py`, the condition sweep that writes the CSVs these scripts consume
- [configs/eval_config.yaml](../../configs/eval_config.yaml) - the eval conditions
- [scripts/diagnostics/CLAUDE.md](../diagnostics/CLAUDE.md) - TensorBoard + Markov diagnostics (the other host-side analysis tools)
