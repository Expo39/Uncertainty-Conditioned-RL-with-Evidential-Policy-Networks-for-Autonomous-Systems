# Make Command Reference

All commands are run from the repository root. GPU-dependent targets require the full
Docker stack (CARLA + ROS 2 + training containers) to be running.

---

## Local Development (no Docker, no GPU)

These targets run in the local `.venv/` virtual environment. The Makefile creates
`.venv/` automatically on first use.

> **Note:** `torch`, `stable_baselines3`, and `gymnasium` are not installed in the
> local `.venv/`. Tests that import them must run inside the training container via
> `make docker-test-unit` or `make docker-verify`. The targets below only cover
> lint, type checking, and import sanity.

| Command | Purpose |
|---------|---------|
| `make install` | Create `.venv/` and install the package with all dev dependencies |
| `make lint` | Run flake8, isort, and black checks (read-only) |
| `make format` | Auto-format code with black and isort |
| `make typecheck` | Run mypy type checking |
| `make sanity` | Quick import check - `python -c "import uncertainty_rl"` |
| `make test-unit` | Unit tests only, no CARLA or ROS 2 (will fail on host - use `docker-test-unit`) |
| `make verify` | All CPU-only checks: test-unit + lint + typecheck + sanity |

---

## Docker Build

| Command | Purpose |
|---------|---------|
| `make docker-build` | Build all images using Docker layer cache |
| `make docker-build-no-cache` | Rebuild all images without cache - use after editing COPYed files |
| `make docker-build-no-cache-core` | Rebuild training image only, no cache |
| `make docker-build-no-cache-inspect` | Rebuild inspect image only, no cache |
| `make docker-build-ros2` | Rebuild the ROS 2 bridge image only |

---

## Container Lifecycle

| Command | Purpose | GPU? |
|---------|---------|------|
| `make docker-up` | Start all containers (CARLA + ROS 2 + training) | Yes |
| `make docker-down` | Stop and remove containers | No |
| `make docker-restart` | Stop then start all containers | Yes |
| `make docker-inspect-down` | Stop inspect-mode containers only | No |
| `make docker-ps` | Show container status | No |
| `make docker-watch` | Watch container resource usage (docker stats) | No |
| `make docker-top` | Show processes inside containers | No |
| `make docker-dev` | Start full stack and drop into training container shell | Yes |
| `make docker-shell` | Interactive bash in the training container | Yes |
| `make docker-shell-ros2` | Interactive bash in the ROS 2 bridge container | Yes |
| `make docker-shell-ros2-inspect` | Interactive bash in the ROS 2 inspect container | Yes |
| `make docker-clean` | Stop containers and remove named volumes | No |
| `make docker-clean-all` | Stop containers, remove volumes, and prune images | No |

---

## Container Logs

| Command | Purpose |
|---------|---------|
| `make docker-logs` | Follow logs from all containers |
| `make docker-logs-training` | Follow training container logs |
| `make docker-logs-carla` | Follow CARLA server logs |
| `make docker-logs-ros2` | Follow ROS 2 bridge logs |
| `make docker-inspect-dryrun-logs` | Follow inspect dry-run logs |
| `make docker-logs-ros2-inspect` | Follow ROS 2 inspect container logs |

---

## Training

| Command | Purpose | GPU? | Notes |
|---------|---------|------|-------|
| `make docker-train` | Full training run | Yes | Uses `configs/train_config.yaml` |
| `make docker-train-short` | 10 000-step smoke test | Yes | Quick sanity check before a full run |

---

## Hyperparameter Tuning

| Command | Purpose | GPU? | Notes |
|---------|---------|------|-------|
| `make docker-tune` | Run Optuna tuning study (35 trials, ~8-10 h) | Yes | Updates `train_config.yaml` on completion |
| `make docker-tune LAYOUT=trapezoid` | Tune on a specific layout | Yes | Layouts: `rectangle`, `trapezoid`, `irregular_a` |

After tuning completes, `train_config.yaml` is automatically updated with the best
hyperparameters. Run `make docker-train` immediately after.

> **Further reading:** [uncertainty_rl/training/README.md](uncertainty_rl/training/README.md) - Optuna search space, trial budget, callback descriptions.

---

## Evaluation

| Command | Purpose | GPU? | Notes |
|---------|---------|------|-------|
| `make docker-eval` | Run all 9 evaluation conditions | Yes | Outputs metrics and plots to `outputs/eval/` |
| `make docker-eval-visualise-3d` | Evaluation with live CARLA 3D spectator view | Yes | Requires a display |
| `make eval-visualise-2d` | Post-hoc 2D bird's-eye replay from saved JSONL | No | Reads `outputs/vis/` |

The 9 evaluation conditions range from nominal GNSS (RTK fixed, empty lot) to worst-case
(250x GNSS noise, high IMU noise, OOD layout).

> **Further reading:** [uncertainty_rl/evaluation/README.md](uncertainty_rl/evaluation/README.md) - condition table, metrics definitions, output structure.

---

## Inspection and Demo

These targets launch the lot inspector, which connects to a live CARLA server and renders
the parking lot, sensor feeds, and vehicle state.

| Command | Purpose | GPU? | Notes |
|---------|---------|------|-------|
| `make docker-inspect` | Default layout inspector (static view) | Yes |
| `make docker-inspect-sensors` | Inspector with sensor feed overlay | Yes |
| `make docker-inspect-live` | Live inspector (step-by-step episode) | Yes |
| `make docker-inspect-dryrun` | Dry-run inspector (manual drive) | Yes |
| `make docker-demo` | Demo mode (pre-trained policy, no training) | Yes |

> **Further reading:** [scripts/inspect/README.md](scripts/inspect/README.md) - inspector modes, CLI flags, per-mode GIF previews.

---

## Testing and QA

| Command | Purpose | GPU? |
|---------|---------|------|
| `make docker-test` | Full test suite inside container (unit + integration) | Yes |
| `make docker-test-unit` | Unit tests only inside container (no CARLA, no ROS 2) | No |
| `make docker-test-integration` | Integration tests inside container (requires full stack) | Yes |
| `make docker-verify` | All checks inside container: tests + lint + typecheck + sanity | No |
| `make docker-lint` | Linting (flake8 + isort + black) inside container | No |
| `make docker-format` | Auto-format inside container | No |
| `make docker-typecheck` | mypy type checking inside container | No |

Use `make docker-test-unit` (not `make test-unit`) when verifying code on the host
machine - torch is not installed outside Docker.

---

## Layout Generation and Visualisation

| Command | Purpose | GPU? | Notes |
|---------|---------|------|-------|
| `make generate-layouts` | Regenerate all lot YAMLs and bird's-eye PNGs | No | Reads `scripts/layouts/floor_plans/*.py` |
| `make visualise` | Launch detachable 2D bird's-eye viewer (Pygame) | No | Reads `outputs/vis/*.jsonl` |
| `make eval-visualise-2d` | Replay an evaluation run in the 2D viewer | No | |

Layout YAMLs are committed to `configs/layouts/`. Regenerate only when you change
the floor plan geometry in `scripts/layouts/floor_plans/*.py`.

> **Further reading:** [scripts/layouts/README.md](scripts/layouts/README.md) - floor plan modules, LotBuilder DSL, bay counts, coordinate frame.

---

## Maintenance

| Command | Purpose |
|---------|---------|
| `make clean-cache` | Remove Python `__pycache__` directories and `.pyc` files |
| `make clean` | Remove build artefacts and caches (preserves `.venv/`) |
| `make clean-all` | Full clean including outputs and generated files |
| `make clean-venv` | Delete `.venv/` (re-create with `make install`) |
| `make backup-configs` | Archive `configs/` to a timestamped tarball |
| `make restore-configs` | Restore configs from the latest backup tarball |
| `make help` | Print a short summary of all targets with descriptions |

---

## Environment Variables

Several targets accept optional `make` variables. Pass them after the target name:
`make <target> VAR=value`.

| Variable | Default | Accepted values | Used by |
|----------|---------|-----------------|---------|
| `LAYOUT` | `rectangle` | `rectangle`, `trapezoid`, `irregular_a` | `docker-train`, `docker-train-short`, `docker-tune`, `docker-eval`, `generate-layouts`, `eval-visualise-2d` |
| `WORKER` | `0` | integer worker index | `docker-shell-ros2`, `docker-logs-carla`, `docker-logs-ros2`, `visualise` |
| `STACK` | `all` | `all`, `training`, `inspect` | `docker-clean`, `docker-clean-all` |
| `SERVICE` | _(all services)_ | any compose service name | `docker-build`, `docker-build-no-cache` |
| `CHECKPOINT` | `checkpoints/final_model` | path to checkpoint directory | `docker-eval-visualise-3d`, `eval-visualise-2d` |
| `MODEL` | `checkpoints/final_model` | path to checkpoint directory | `docker-demo` |
| `MANUAL` | _(unset)_ | `true` | `docker-inspect-dryrun` - enables manual keyboard drive |
| `INSPECT_LAYOUT` | `rectangle` | `rectangle`, `trapezoid`, `irregular_a` | `docker-inspect`, `docker-inspect-sensors`, `docker-inspect-live`, `docker-inspect-dryrun` |
| `INSPECT_VIEW` | `third_person` | `third_person`, `side`, `back`, `front`, `free` (dryrun); `birds_eye`, `side`, `front` (sensors) | `docker-inspect-dryrun`, `docker-inspect-sensors` |
| `INSPECT_ZOOM` | `close` | `close`, `wide` | `docker-inspect-sensors` |
| `INSPECT_SENSOR` | `lidar` | sensor type string | `docker-inspect-live` |
| `INSPECT_EPISODES` | _(unset)_ | integer | `docker-inspect-dryrun` |
| `INSPECT_PAUSE` | `3.0` | seconds (float) | `docker-inspect-dryrun` - pause between episodes |
