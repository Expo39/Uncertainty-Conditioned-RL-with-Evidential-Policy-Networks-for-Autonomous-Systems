# Make Command Reference

Every workflow in this project goes through a `make` target - never call `docker
compose`, `pytest`, `python`, or the linters directly. All commands run from the
repository root. GPU-dependent targets require the full Docker stack (CARLA + ROS 2 +
training containers), and the column in each table marks which.

Two conventions run through the whole reference:

- **`BASELINE` and `CHECKPOINT` are bare names, never paths.** The output tree is nested
  by baseline: `<root>/<baseline>/<leaf>/`, where `<leaf>` is
  `<stage>_<seed>_<DDMMYYYY-HHMM>` (or `trial_<N>` for tuning). You pass only the names -
  `BASELINE=input_uncertainty CHECKPOINT=6_42_11062026-0628` - and the recipes
  reconstruct the full `checkpoints/<baseline>/<leaf>/` paths. `BASELINE` defaults to
  `full_method`. The middle token is the seed, and the evaluation targets parse it back
  out to nest their results under `seed_<N>/`, so a malformed leaf sends output to the
  wrong tree.
- **Every run is staged and baselined.** Omitting `STAGE` defaults to stage 1 (the
  curriculum head), and omitting `BASELINE` defaults to `full_method`. No value falls back to
  a hidden Python default - see [configs/README.md](configs/README.md) for the merge
  precedence chain.

A short, self-documenting summary of every target is always available with
`make help`.

---

## Local Development (no Docker, no GPU)

These targets run in the local `.venv/` virtual environment, which the Makefile creates
automatically on first use (`make install`).

> **Note:** `torch`, `stable_baselines3`, and `gymnasium` are **not** installed in the
> local `.venv/`. Anything that imports them must run inside the training container via
> `make docker-test-unit`. The local targets below cover only
> linting, type checking, and import sanity - `make test-unit` and `make verify` fail on
> the host with `ModuleNotFoundError: No module named 'torch'`.

| Command | Purpose |
|---------|---------|
| `make install` | Create `.venv/` and install the package with all dev dependencies |
| `make lint` | Run flake8, isort, and black in check-only mode (read-only) |
| `make format` | Auto-format code with isort and black |
| `make typecheck` | Run mypy type checking |
| `make sanity` | Quick import check - `python -c "import uncertainty_rl"` |
| `make test-unit` | Unit tests only (fails on host - use `docker-test-unit`) |
| `make verify` | The CI checks: lint + typecheck + import. Tests need `docker-test-unit` |

---

## Docker Build

After editing any file **COPYed** into an image - anything under `uncertainty_rl/ros2/`
or any `Dockerfile` - rebuild with a no-cache target. Plain `make docker-build` uses the
layer cache and can silently serve a stale image. It is only safe for `pyproject.toml` or
bind-mounted file changes.

| Command | Purpose |
|---------|---------|
| `make docker-build` | Build all images (core + env-workers + inspect stacks) using the layer cache |
| `make docker-build-no-cache` | Rebuild all images without cache (core + inspect stacks) |
| `make docker-build-no-cache-core` | Rebuild core + env-worker images only (carla, ros2-bridge, training), no cache |
| `make docker-build-no-cache-inspect` | Rebuild the inspect-stack images only (activates every inspect profile), no cache |
| `make docker-build-ros2` | Rebuild only the ros2-bridge images (core + inspect), no cache |

`make docker-build` accepts `SERVICE=<compose-service>` to build a single service.

---

## Container Lifecycle

The full stack is N CARLA env workers (read from `parallel_workers` in
`train_config.yaml`) plus the training stack. The worker scripts under
`scripts/multi_workers/` bring those up and down, and the `make` targets call them for you.

| Command | Purpose | GPU? |
|---------|---------|------|
| `make docker-up` | Start N env workers + the training stack | Yes |
| `make docker-down` | Stop the training stack and all running env workers | No |
| `make docker-restart` | Restart all containers | Yes |
| `make docker-dev` | Start the full stack and drop into the training shell (GPU-machine workflow) | Yes |
| `make docker-ps` | Show running containers | No |
| `make docker-watch` | Watch container health, refreshing every 5 s (Ctrl+C to exit) | No |
| `make docker-top` | Show running processes inside the containers | No |
| `make docker-inspect-down` | Stop every inspect-mode container (all profiles) | No |

---

## Shells

| Command | Purpose | GPU? |
|---------|---------|------|
| `make docker-shell` | Interactive bash in the training container | Yes |
| `make docker-shell-ros2 [WORKER=0]` | Interactive bash in a worker's ROS 2 bridge | Yes |
| `make docker-shell-ros2-inspect` | Interactive bash in the ROS 2 inspect container | Yes |

---

## Container Logs

| Command | Purpose |
|---------|---------|
| `make docker-logs` | Follow logs from the training-stack containers |
| `make docker-logs-training` | Follow the training container only |
| `make docker-logs-carla [WORKER=0]` | Follow a worker's CARLA server |
| `make docker-logs-ros2 [WORKER=0]` | Follow a worker's ROS 2 bridge |
| `make docker-inspect-dryrun-logs` | Follow the dryrun training container (run alongside `docker-inspect-dryrun`) |
| `make docker-logs-ros2-inspect` | Follow the ROS 2 inspect container (run alongside `docker-inspect-dryrun`) |

---

## Training

Each stage of the single-phase ADR curriculum is selected with `STAGE=N` and resumes
from the previous stage's checkpoint via `CHECKPOINT=<leaf>`. The first stage starts from
a random init (no `CHECKPOINT`). See
[configs/deployment/sim/curriculum/README.md](configs/deployment/sim/curriculum/README.md).

| Command | Purpose | GPU? |
|---------|---------|------|
| `make docker-train [STAGE=1] [BASELINE=vanilla_ppo] [CHECKPOINT=<leaf>] [LAYOUT=rectangle]` | Run a training stage | Yes |
| `make docker-train-short [STAGE=1] [BASELINE=...] [CHECKPOINT=<leaf>]` | 10 000-step smoke test before a full run | Yes |
| `make run-seed-leg [DRY_RUN=1]` | The whole experiment: every arm through every stage, then evaluate the final stage | Yes |

```bash
# Stage 1 from scratch, vanilla baseline
make docker-train STAGE=1 BASELINE=vanilla_ppo

# Stage 2 resuming from a stage-1 run leaf (same baseline)
make docker-train STAGE=2 BASELINE=vanilla_ppo CHECKPOINT=6_42_11062026-0628
```

`run-seed-leg` chains all of that automatically: it trains the six arms through stages
1-6 for each seed (`ARMS_OVERRIDE="heteroscedastic heteroscedastic_input"` restricts it
to the listed arms), resume-chaining every stage, then evaluates the final stage only. It is
idempotent, skipping work that is already done, so a crashed leg resumes on re-run. Expect
it to run for days - start it under `tmux`, and use `DRY_RUN=1` first to print the plan
without executing it.

> **Further reading:** [uncertainty_rl/training/README.md](uncertainty_rl/training/README.md) - training loop, callbacks, resume mechanics.

---

## Hyperparameter Tuning

> **Never run.** No reported result used this pipeline: every arm and seed trained on the
> committed defaults in `configs/train_config.yaml`, because a separate search per arm
> would make the configuration an extra experimental variable and confound the 3x2
> ablation. The target is retained for future work.

Were it run, it would tune structural PPO parameters only, leaving `learning_rate` and
`ent_coef` to their stage-owned schedules.

| Command | Purpose | GPU? |
|---------|---------|------|
| `make docker-tune [STAGE=1] [BASELINE=vanilla_ppo] [LAYOUT=rectangle]` | Run the Optuna study. Stage 1 is the recommended tuning stage, being the only one whose ~100k-step trial budget yields a non-zero success-rate objective | Yes |

> **Further reading:** [uncertainty_rl/training/README.md](uncertainty_rl/training/README.md) - Optuna search space, trial budget, callbacks. [configs/training/README.md](configs/training/README.md) - study settings.

---

## Evaluation

| Command | Purpose | GPU? |
|---------|---------|------|
| `make docker-eval [BASELINE=vanilla_ppo] [CHECKPOINT=<leaf>] [LAYOUT=rectangle]` | Run the full condition sweep, writing raw CSVs to `outputs/raw/evaluation_results/` | Yes |
| `make docker-eval-visualise-3d [BASELINE=...] [CHECKPOINT=<leaf>]` | Load a checkpoint with a live CARLA 3D spectator view (needs a display) | Yes |
| `make eval-visualise-2d [BASELINE=...] [CHECKPOINT=<leaf>] [LAYOUT=rectangle] [STAGE=N] [REALTIME=false]` | Start a checkpoint demo drive and open the 2D bird's-eye viewer | No for the viewer, yes for CARLA |

The sweep runs the seven conditions defined in `configs/eval_config.yaml`, spanning the
in-distribution anchor through held fix-state tiers, degraded LiDAR, an unseen layout and
mid-episode drift. Four of them carry reported results; the analysis scripts exclude the
other three.

> **Further reading:** [uncertainty_rl/evaluation/README.md](uncertainty_rl/evaluation/README.md) - condition table, metric definitions, output structure.

---

## Inspection and Demo

These targets bring up a windowed CARLA server (X11 required) to verify geometry,
sensors, and the live pipeline. They tear down the training stack first, so do not run
them while training.

| Command | Purpose | GPU? |
|---------|---------|------|
| `make docker-inspect [INSPECT_LAYOUT=rectangle]` | Spawn a layout for visual inspection (includes perimeter cones) | Yes |
| `make docker-inspect-sensors [INSPECT_LAYOUT=...] [SENSORS_VIEW=birds_eye] [INSPECT_ZOOM=close]` | Visualise sensor FOV on the layout | Yes |
| `make docker-inspect-live [INSPECT_SENSOR=lidar] [INSPECT_LAYOUT=...]` | Live single-sensor view in windowed CARLA | Yes |
| `make docker-inspect-dryrun [STAGE=1] [BASELINE=...] [MANUAL=true] [INSPECT_EPISODES=5] [INSPECT_VIEW=...] [INSPECT_PAUSE=3.0] [INSPECT_OOD=false]` | Full training pipeline in windowed CARLA, built identically to training | Yes |
| `make docker-inspect-eval-dryrun SCENARIO=anchor_deployment [BASELINE=...] [MANUAL=true] [INSPECT_EPISODES=5] [INSPECT_VIEW=...]` | Manually drive one named eval condition, to verify the scenario wiring (no checkpoint) | Yes |
| `make docker-demo MODEL=<path>` | Windowed CARLA demo of a checkpoint | Yes |

> **Further reading:** [scripts/inspect/README.md](scripts/inspect/README.md) - inspector modes, CLI flags, per-mode previews.

---

## Testing and QA

All test targets first ensure the stack is up (`scripts/multi_workers/ensure_stack.sh`)
and then run inside the training container - that is the only place `torch` is installed.

| Command | Purpose | GPU? |
|---------|---------|------|
| `make docker-test` | Full suite in the container (unit + integration) | Yes |
| `make docker-test-unit` | Unit tests only (no CARLA, no ROS 2) | No |
| `make docker-test-integration` | Integration tests (requires the full stack) | Yes |

Use `make docker-test-unit` (not `make test-unit`) to verify code on the host - `torch`
is not installed outside Docker.

---

## Layout Generation, Diagnostics, and Visualisation

| Command | Purpose | GPU? |
|---------|---------|------|
| `make generate-layouts [LAYOUT=trapezoid]` | Regenerate all lot YAMLs and bird's-eye PNGs | No |
| `make analyse-markov [N_EPISODES=10000] [N_STEPS=1750]` | Diagnose the GNSS tier Markov chain from `gnss_noise_profiles.yaml` | No |
| `make tb-scalars LOG=logs/<run_dir> [ARGS="--match success --last 10"]` | Print TensorBoard scalar trajectories for one or more runs | No |
| `make visualise [WORKER=0]` | Open the detachable 2D bird's-eye viewer for a running worker | No |
| `make check-host-deps` | Verify the host-side tools the recording targets need (ffmpeg) | No |
| `make record-screen [DURATION=30] [OUT=...]` | Screen-record the CARLA window to MP4 until Ctrl+C | No |
| `make clip START=00:05 END=00:20 [FORMAT=gif] [VIDEO=...]` | Cut a GIF or MP4 from the newest recording | No |

Layout YAMLs are committed to `configs/layouts/`. Regenerate only when you change the
floor plan geometry in `scripts/layouts/floor_plans/*.py`.

> **Further reading:** [scripts/layouts/README.md](scripts/layouts/README.md) - floor plan modules, LotBuilder DSL, coordinate frame. [scripts/visualise/README.md](scripts/visualise/README.md) - viewer protocol and JSONL schema.

---

## Results Analysis

Run these after an evaluation has written `outputs/raw/evaluation_results/`. They read
raw CSVs and write derived ones. None of them needs a GPU or the simulator.

| Command | Purpose | GPU? |
|---------|---------|------|
| `make analyse-cross-seed [STAGE=6]` | Pool every seed into the headline CSVs + per-seed robustness | No |
| `make analyse-seed-level [STAGE=6] [BOOTSTRAP_SEED=20260927]` | Two-level (seed, episode) bootstrap intervals, seed permutation tests and brake correlations for the 3x2 contrasts | No |
| `make analyse-ablation [STAGE=6] [SEED=42]` | Single-seed cross-arm contrast | No |
| `make analyse-gate [STAGE=6] [SEED=42]` | Single-seed safety-gate ROC | No |
| `make analyse-calibration [ARM=full_method] [SEED=42]` | Is the EKF covariance honest? | No |
| `make handover-timing [ARM=full_method]` | Handover latency vs degradation onset | No |
| `make uncertainty-verdict EVAL_DIR=...` | Epistemic-vs-aleatoric separation | No |
| `make trace-tier-breakdown TRACE_DIR=...` | Demo-trace outcomes resolved by GNSS tier (collapse vs hard task) | No |
| `make docker-covariance-probe BASELINE=<name> CHECKPOINT=<leaf>` | Causal probe: does the policy USE the covariance input? | Yes |
| `make training-curves` | TensorBoard scalars -> CSV | No |
| `make figures [FIG=gate_roc]` | Render the figures into `outputs/main_analysis/figures/` | No |
| `make run-figures [RUN_DIR=...]` | Per-run diagnostic panels | No |
| `make analysis-bundle [STAGE=6] [BOOTSTRAP_SEED=42]` | Assemble summaries (gate AUCs with 95% bootstrap intervals), values and MANIFEST | No |

Typical order after a completed evaluation:

```bash
make analyse-cross-seed STAGE=6   # pooled CSVs -> raw_derived/cross_seed_analysis/
make training-curves              # TB scalars  -> raw_derived/training/
make figures                      # figures     -> main_analysis/figures/
make analysis-bundle STAGE=6 BOOTSTRAP_SEED=42   # summaries + values + MANIFEST
```

The single-seed targets (`analyse-ablation`, `analyse-gate`, `analyse-calibration`,
`handover-timing`) are debugging aids: `analyse-cross-seed` recomputes the same
statistics over the pooled sample and is what the headline set reads.

`docker-covariance-probe` is the one analysis target that needs the training container,
since it loads the policy and runs forward passes; everything else is host-side pandas.
`trace-tier-breakdown` reads a demo-trace dump rather than an evaluation results tree,
and prints to the console without writing a CSV.

> **Further reading:** [scripts/analysis/README.md](scripts/analysis/README.md) - per-script reference and CSV schema. The `outputs/` tier layout is in [USAGE.md](USAGE.md#results-layout).

---

## Maintenance

| Command | Purpose |
|---------|---------|
| `make clean-cache` | Remove build caches and `.pyc` files (preserves checkpoints, logs, outputs, maps) |
| `make clean` | Remove build artefacts, caches, and generated outputs/raw_derived/layouts (preserves checkpoints, logs, `.xodr`, `.venv`) |
| `make clean-all` | Remove everything including checkpoints and logs (preserves `.xodr` and `.venv`) |
| `make clean-venv` | Delete `.venv/` (re-create with `make install`) |
| `make docker-clean [STACK=all]` | Stop containers and remove volumes |
| `make docker-clean-all [STACK=all]` | Remove containers, images, and volumes |
| `make backup-configs` | Pack all `CLAUDE.md`, `TODO.md`, and `documentation/` into `project_configs.tar.gz` |
| `make restore-configs` | Restore those files from `project_configs.tar.gz` |
| `make backup-results [RESULTS_ARCHIVE=...]` | Archive `checkpoints/`, `logs/` and `outputs/` into `project_results.tar.gz` (multi-GB) |
| `make restore-results [RESULTS_ARCHIVE=...] [FORCE=1]` | Restore those trees, refusing to overwrite unless `FORCE=1` |
| `make list-results-archive [RESULTS_ARCHIVE=...]` | List the archive contents without extracting |
| `make help` | Print a one-line summary of every target |

### Backing up run output

`checkpoints/`, `logs/` and `outputs/` are gitignored: a fresh clone has none of them,
and they represent GPU time that cannot be regenerated. `backup-results` packs all three.

```bash
make backup-results                              # -> project_results.tar.gz (~5 GB here)
make list-results-archive                        # check it without extracting
make backup-results RESULTS_ARCHIVE=/mnt/usb/run.tar.gz   # somewhere else
```

The archive is written to the repo root, which neither `make clean` nor `make clean-all`
touches - so a backup survives the very targets that delete what it holds. It is
gitignored (`project_results*.tar.gz`), so it will never be committed.

Restoring refuses to clobber existing trees, so move them aside or pass `FORCE=1`:

```bash
make restore-results          # fails if checkpoints/ logs/ outputs/ already exist
make restore-results FORCE=1  # overwrite them deliberately
```

Model files are already-compressed `.zip`, so gzip gains little - expect the archive to
be roughly the on-disk size, and the run to take a few minutes.

> **Note:** this is distinct from `backup-configs`, which packs the small gitignored
> *text* files (`CLAUDE.md`, `TODO.md`, `documentation/`) rather than run output.

---

## Variables

Pass variables after the target name: `make <target> VAR=value`. Bare-name conventions
for `BASELINE` and `CHECKPOINT` are described at the top of this file.

| Variable | Default | Accepted values | Used by |
|----------|---------|-----------------|---------|
| `STAGE` | `1` | `1`-`6` | `docker-train`, `docker-train-short`, `docker-tune`, `docker-inspect-dryrun`, `eval-visualise-2d` |
| `BASELINE` | `full_method` | `vanilla_ppo`, `input_uncertainty`, `heteroscedastic`, `heteroscedastic_input`, `output_uncertainty`, `full_method` (bare name) | `docker-train`, `docker-train-short`, `docker-tune`, `docker-eval`, `docker-eval-visualise-3d`, `docker-inspect-dryrun`, `eval-visualise-2d` |
| `CHECKPOINT` | _(none)_ | run leaf `<stage>_<seed>_<DDMMYYYY-HHMM>` (bare name), or `trial_<N>` for a tuning trial | `docker-train`, `docker-train-short`, `docker-eval`, `docker-eval-visualise-3d`, `docker-covariance-probe`, `eval-visualise-2d` |
| `LAYOUT` | `rectangle` | `rectangle`, `trapezoid`, `irregular_a` | `docker-train`, `docker-tune`, `docker-eval`, `eval-visualise-2d`, `generate-layouts` |
| `WORKER` | `0` | integer worker index | `docker-shell-ros2`, `docker-logs-carla`, `docker-logs-ros2`, `visualise` |
| `STACK` | `all` | `all`, `training`, `inspect` | `docker-clean`, `docker-clean-all` |
| `SERVICE` | _(all)_ | any compose service name | `docker-build` |
| `MODEL` | `checkpoints/final_model` | path to a checkpoint directory | `docker-demo` |
| `REALTIME` | _(unset)_ | `true`, `false` | `eval-visualise-2d` |
| `N_EPISODES` / `N_STEPS` | from config | integers | `analyse-markov` |
| `MANUAL` | _(unset)_ | `true` | `docker-inspect-dryrun` - manual keyboard drive |
| `INSPECT_LAYOUT` | `rectangle` | `rectangle`, `trapezoid`, `irregular_a` | `docker-inspect`, `docker-inspect-sensors`, `docker-inspect-live` |
| `INSPECT_VIEW` | `third_person` | `third_person`, `side`, `back`, `front`, `free`, `birds_eye` | `docker-inspect-dryrun` |
| `INSPECT_EPISODES` | _(unset)_ | integer | `docker-inspect-dryrun` |
| `INSPECT_PAUSE` | `3.0` | seconds (float) | `docker-inspect-dryrun` - pause between episodes |
| `INSPECT_OOD` | `false` | `true`, `false` | `docker-inspect-dryrun` - use the OOD layout |
| `SENSORS_VIEW` | `birds_eye` | `birds_eye`, `side`, `front` | `docker-inspect-sensors` |
| `INSPECT_ZOOM` | `close` | `close`, `wide` | `docker-inspect-sensors` |
| `INSPECT_SENSOR` | `lidar` | sensor type string | `docker-inspect-live` |
| `SCENARIO` | `anchor_deployment` | any condition name in `configs/eval_config.yaml`; `docker-eval` also accepts a space-separated subset | `docker-inspect-eval-dryrun`, `docker-eval` |
| `PER_STEP_CAP` | `0` | leading steps per episode written to `per_step_records.csv`; `0` disables | `docker-eval` |
| `NO_SAFETY` | _(unset)_ | `1` bypasses the SafetyWrapper; results nest under `without_wrapper/` | `docker-eval` |
| `DRY_RUN` | `0` | `1` prints the plan without running it | `run-seed-leg` |
| `TRACE_DIR` | _(required)_ | path to one demo-trace dump | `trace-tier-breakdown` |
| `REAL_OBS` | _(unset)_ | path to a dumped `.npy` observation set | `docker-covariance-probe` - on-manifold mode |
| `LOG` | _(required)_ | one or more run log directories | `tb-scalars` |
| `ARGS` | _(unset)_ | flags forwarded verbatim, e.g. `"--match success --last 10"` | `tb-scalars` |
