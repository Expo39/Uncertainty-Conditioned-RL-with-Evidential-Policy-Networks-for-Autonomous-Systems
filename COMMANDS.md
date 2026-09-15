# Make Command Reference

Every workflow in this project goes through a `make` target - never call `docker
compose`, `pytest`, `python`, or the linters directly. All commands run from the
repository root. GPU-dependent targets require the full Docker stack (CARLA + ROS 2 +
training containers); the column in each table marks which.

Two conventions run through the whole reference:

- **`BASELINE` and `CHECKPOINT` are bare names, never paths.** The output tree is nested
  by baseline: `<root>/<baseline>/<leaf>/`, where `<leaf>` is `seed<N>_<DDMMYYYY-HHMM>`
  (or `trial_<N>` for tuning). You pass only the names -
  `BASELINE=input_uncertainty CHECKPOINT=seed42_11062026-0628` - and the recipes
  reconstruct the full `checkpoints/<baseline>/<leaf>/` paths. `BASELINE` defaults to
  `full_method`; the seed and timestamp are recoverable from the leaf name alone.
- **Every run is staged and baselined.** Omitting `STAGE` defaults to stage 1 (the
  curriculum head); omitting `BASELINE` defaults to `full_method`. No value falls back to
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
> `make docker-test-unit` or `make docker-verify`. The local targets below cover only
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
| `make verify` | Local CPU-only checks: lint + typecheck + sanity |

---

## Docker Build

After editing any file **COPYed** into an image - anything under `uncertainty_rl/ros2/`
or any `Dockerfile` - rebuild with a no-cache target. Plain `make docker-build` uses the
layer cache and can silently serve a stale image; it is only safe for `pyproject.toml` or
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
`scripts/multi_workers/` bring those up and down; the `make` targets call them for you.

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

```bash
# Stage 1 from scratch, vanilla baseline
make docker-train STAGE=1 BASELINE=vanilla_ppo

# Stage 2 resuming from a stage-1 run leaf (same baseline)
make docker-train STAGE=2 BASELINE=vanilla_ppo CHECKPOINT=seed42_11062026-0628
```

> **Further reading:** [uncertainty_rl/training/README.md](uncertainty_rl/training/README.md) - training loop, callbacks, resume mechanics.

---

## Hyperparameter Tuning

Per project methodology, HPO is run **after** the curriculum validates on the vanilla
baseline and tunes structural PPO parameters only.

| Command | Purpose | GPU? |
|---------|---------|------|
| `make docker-tune [STAGE=4] [BASELINE=full_method] [LAYOUT=rectangle]` | Run the Optuna study | Yes |

> **Further reading:** [uncertainty_rl/training/README.md](uncertainty_rl/training/README.md) - Optuna search space, trial budget, callbacks. [configs/training/README.md](configs/training/README.md) - study settings.

---

## Evaluation

| Command | Purpose | GPU? |
|---------|---------|------|
| `make docker-eval [BASELINE=vanilla_ppo] [CHECKPOINT=<leaf>] [LAYOUT=rectangle]` | Run the full 9-condition sweep; writes metrics + plots to `evaluation_results/` | Yes |
| `make docker-eval-visualise-3d [BASELINE=...] [CHECKPOINT=<leaf>]` | Load a checkpoint with a live CARLA 3D spectator view (needs a display) | Yes |
| `make eval-visualise-2d [BASELINE=...] [CHECKPOINT=<leaf>] [LAYOUT=rectangle] [STAGE=N] [REALTIME=false]` | Start a checkpoint demo drive and open the 2D bird's-eye viewer | No (viewer); Yes (CARLA) |

The 9 conditions span nominal GNSS (RTK fixed, empty lot) through to worst-case
(degraded fix state, high IMU noise, OOD layout).

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
| `make docker-verify` | All checks in the container: unit tests + lint + typecheck + sanity | No |
| `make docker-lint` | flake8 + isort + black in the container | No |
| `make docker-format` | Auto-format in the container | No |
| `make docker-typecheck` | mypy in the container | No |

Use `make docker-test-unit` (not `make test-unit`) to verify code on the host - `torch`
is not installed outside Docker.

---

## Layout Generation, Diagnostics, and Visualisation

| Command | Purpose | GPU? |
|---------|---------|------|
| `make generate-layouts [LAYOUT=trapezoid]` | Regenerate all lot YAMLs and bird's-eye PNGs | No |
| `make analyse-markov [N_EPISODES=10000] [N_STEPS=1750]` | Diagnose the GNSS tier Markov chain from `gnss_noise_profiles.yaml` | No |
| `make visualise [WORKER=0]` | Open the detachable 2D bird's-eye viewer for a running worker | No |

Layout YAMLs are committed to `configs/layouts/`. Regenerate only when you change the
floor plan geometry in `scripts/layouts/floor_plans/*.py`.

> **Further reading:** [scripts/layouts/README.md](scripts/layouts/README.md) - floor plan modules, LotBuilder DSL, coordinate frame. [scripts/visualise/README.md](scripts/visualise/README.md) - viewer protocol and JSONL schema.

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
| `make help` | Print a one-line summary of every target |

---

## Variables

Pass variables after the target name: `make <target> VAR=value`. Bare-name conventions
for `BASELINE` and `CHECKPOINT` are described at the top of this file.

| Variable | Default | Accepted values | Used by |
|----------|---------|-----------------|---------|
| `STAGE` | `1` | `1`-`6` | `docker-train`, `docker-train-short`, `docker-tune`, `docker-inspect-dryrun`, `eval-visualise-2d` |
| `BASELINE` | `full_method` | `vanilla_ppo`, `input_uncertainty`, `output_uncertainty`, `full_method` (bare name) | `docker-train`, `docker-train-short`, `docker-tune`, `docker-eval`, `docker-eval-visualise-3d`, `docker-inspect-dryrun`, `eval-visualise-2d` |
| `CHECKPOINT` | _(none)_ | run leaf `seed<N>_<DDMMYYYY-HHMM>` (bare name) | `docker-train`, `docker-train-short`, `docker-eval`, `docker-eval-visualise-3d`, `eval-visualise-2d` |
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
