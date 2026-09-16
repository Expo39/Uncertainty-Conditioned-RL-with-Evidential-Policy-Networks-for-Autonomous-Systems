# Development commands for training, evaluation, testing, and linting.

.PHONY: help install test test-unit test-integration
.PHONY: lint format typecheck verify clean clean-cache clean-all clean-venv
.PHONY: backup-configs restore-configs backup-results restore-results list-results-archive
.PHONY: figures run-figures training-curves analysis-bundle generate-layouts visualise eval-visualise-2d docker-eval-visualise-3d clip record-screen check-host-deps
.PHONY: docker-build docker-build-no-cache docker-build-no-cache-core docker-build-no-cache-inspect docker-build-ros2 docker-up docker-down docker-restart docker-ps docker-watch docker-top
.PHONY: docker-eval docker-covariance-probe
.PHONY: docker-test docker-test-unit docker-test-integration
.PHONY: docker-shell docker-shell-ros2 docker-shell-ros2-inspect docker-logs docker-logs-training docker-logs-carla docker-logs-ros2 docker-inspect-dryrun-logs docker-logs-ros2-inspect
.PHONY: docker-clean docker-clean-all docker-dev docker-demo docker-inspect docker-inspect-down docker-inspect-sensors docker-inspect-live docker-inspect-dryrun docker-inspect-eval-dryrun
.PHONY: docker-train docker-train-short docker-tune run-seed-leg
.PHONY: ensure-dirs

VENV        := .venv
PYTHON      := $(VENV)/bin/python3
PYTEST      := $(VENV)/bin/pytest
CONFIG_DIR  := configs
SRC_DIR     := uncertainty_rl
TESTS_DIR   := tests
SCRIPTS_DIR := scripts
DOCKER_COMPOSE         := docker compose
DOCKER_COMPOSE_INSPECT := docker compose -f docker-compose.yml -f docker-compose.inspect.yml
DOCKER_COMPOSE_WORKERS := docker compose -f docker-compose.env_workers.yml
LAYOUT       ?= rectangle
CHECKPOINT   ?=
BASELINE     ?=
STAGE        ?=
# Which seed's results to READ (a run's RNG seed lives in agent_config.yaml).
# SEED is only needed by analyse-gate / analyse-ablation, which key on STAGE and
# so cannot infer it; EVAL_SEED prefers the checkpoint leaf and owns the output
# tree, where every path nests seed_<N>/ so seeds cannot overwrite each other.
SEED         ?=
EVAL_SEED = $(or $(word 2,$(subst _, ,$(CHECKPOINT))),$(SEED),42)
EVAL_RESULTS_ROOT = outputs/raw/evaluation_results/seed_$(EVAL_SEED)
ABLATION_ROOT     = outputs/raw_derived/ablation_analysis/seed_$(EVAL_SEED)
GATE_ROOT         = outputs/raw_derived/gate_analysis/seed_$(EVAL_SEED)
CALIBRATION_ROOT  = outputs/raw_derived/calibration_analysis/seed_$(EVAL_SEED)
HANDOVER_ROOT     = outputs/raw_derived/handover_timing/seed_$(EVAL_SEED)
# Leading steps/episode into per_step_records.csv; 0 disables.
PER_STEP_CAP ?= 0
# 1 bypasses the SafetyWrapper; results nest with_/without_wrapper.
NO_SAFETY    ?=

# Viewer: RECORD starts capture at once (R toggles either way), UI_SCALE sizes
# fonts for a projector, TRACE=false suppresses the demo's per-step CSVs,
# SHOW_EPISODE exposes internal bookkeeping that means nothing to an audience.
RECORD     ?= false
UI_SCALE   ?= 1.5
TRACE      ?= true
SHOW_EPISODE ?= false
RECORD_DIR ?= outputs/recordings
REC_FPS    ?= 30

# clip parameters (see the clip target).
VIDEO  ?=
START  ?=
END    ?=
FORMAT ?= gif
WIDTH  ?= 800
FPS    ?= 15

# Screen capture; empty DURATION records until Ctrl+C. ultrafast + high CRF keeps
# the encoder ahead of the capture rate, since a slower preset cannot sustain a 4K
# desktop against CARLA on the same GPU and x11grab then drops frames.
DURATION ?=
REC_PRESET ?= ultrafast
REC_CRF    ?= 23
REC_HEIGHT ?= 720
REGION   ?= 800x600
OFFSET   ?= 0,0
OUT      ?=

# BASELINE and CHECKPOINT are bare names (BASELINE=input_uncertainty
# CHECKPOINT=seed42_11062026-0628); the recipes rebuild the nested paths
# <root>/<baseline>/<leaf>/. A full YAML path also works (the stem is taken).
BASELINE_NAME = $(if $(BASELINE),$(notdir $(basename $(BASELINE))),full_method)
BASELINE_YAML = $(CONFIG_DIR)/baselines/$(BASELINE_NAME).yaml
# Resume/run directory (checkpoints/<baseline>/<leaf>) for train --resume-from.
CHECKPOINT_DIR = checkpoints/$(BASELINE_NAME)/$(CHECKPOINT)
# The saved model inside that run dir, for eval/demo --model-path / --checkpoint.
CHECKPOINT_MODEL = $(if $(CHECKPOINT),$(CHECKPOINT_DIR)/final_model,checkpoints/final_model)
# Short label for the echo banners (<baseline>/<leaf>, cosmetic only).
CHECKPOINT_NAME = $(if $(CHECKPOINT),$(BASELINE_NAME)/$(CHECKPOINT),final_model)

# Scripts that bring up/down N env workers (N read from train_config.yaml by default).
WORKERS_UP   = bash scripts/multi_workers/workers_up.sh
WORKERS_DOWN = bash scripts/multi_workers/workers_down.sh

# Ensure the local venv exists and the package is installed.
# Runs automatically before every local target that needs Python.
define ensure-venv
	@if [ ! -f "$(VENV)/bin/python3" ]; then \
		$(MAKE) install; \
	fi
endef


help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}'


install: ## Create .venv and install package + dev dependencies
	python3 -m venv $(VENV)
	$(VENV)/bin/pip install --upgrade pip
	$(VENV)/bin/pip install -e ".[dev]"
	$(VENV)/bin/pre-commit install || true  # Non-fatal: core.hooksPath may be managed externally




# Pre-create host-side bind-mount targets so the Docker daemon (root) does not
# create them as root-owned. Must run before any `docker compose up` call.
ensure-dirs: ## Pre-create host directories for bind mounts (avoids root-owned logs/)
	@mkdir -p logs/ros2 outputs/raw checkpoints

SERVICE ?=
docker-build: ## Build all Docker images (core + env-workers + inspect stacks).
	$(DOCKER_COMPOSE) build $(SERVICE)
	bash scripts/multi_workers/workers_build.sh docker-compose.env_workers.yml $(SERVICE)
	$(DOCKER_COMPOSE_INSPECT) build $(SERVICE)

docker-build-no-cache: docker-build-no-cache-core docker-build-no-cache-inspect ## Build all images without cache (core + inspect stacks)

docker-build-no-cache-core: ## Build core + env-worker images without cache (carla, ros2-bridge, training)
	$(DOCKER_COMPOSE) build --no-cache
	bash scripts/multi_workers/workers_build.sh docker-compose.env_workers.yml --no-cache

docker-build-no-cache-inspect: ## Build inspect-stack images without cache (ros2-bridge-inspect, training-inspect-*)
	@# Inspect services are profile-gated; activate every profile so build --no-cache
	@# does not skip them.
	$(DOCKER_COMPOSE_INSPECT) --profile inspect --profile inspect-sensors --profile inspect-live --profile inspect-dryrun --profile inspect-eval-dryrun build --no-cache

docker-build-ros2: ## Rebuild only the ros2-bridge images without cache
	$(DOCKER_COMPOSE_WORKERS) build --no-cache ros2-bridge
	$(DOCKER_COMPOSE_INSPECT) build --no-cache ros2-bridge-inspect

docker-up: ensure-dirs ## Start all containers (N env workers from train_config.yaml + training stack)
	$(WORKERS_UP)
	$(DOCKER_COMPOSE) up -d

docker-down: ## Stop all containers (training stack + all running env workers)
	$(DOCKER_COMPOSE) down
	$(WORKERS_DOWN)

docker-inspect-down: ## Stop all inspect containers (all profiles)
	$(DOCKER_COMPOSE_INSPECT) --profile inspect --profile inspect-dryrun --profile inspect-eval-dryrun --profile inspect-sensors --profile inspect-live down
	docker rm -f uncertainty-rl-carla-demo uncertainty-rl-ros2-inspect uncertainty-rl-training-inspect uncertainty-rl-training-inspect-dryrun uncertainty-rl-training-inspect-eval-dryrun uncertainty-rl-training-inspect-sensors uncertainty-rl-training-inspect-live 2>/dev/null || true
	xhost -local:docker 2>/dev/null || true

docker-restart: ## Restart all containers
	$(DOCKER_COMPOSE) restart

docker-ps: ## Show running containers
	$(DOCKER_COMPOSE) ps

docker-watch: ## Watch container health status (refreshes every 5s, Ctrl+C to exit)
	watch -n 5 docker compose ps

docker-top: ## Show running processes in containers
	$(DOCKER_COMPOSE) top


run-seed-leg: ensure-dirs ## Multi-seed leg (seeds in the script): train all arms/stages + eval final stage (cap 440, EDL with+without) + suite tables. Idempotent (skips done work); resumes a crash by re-running. Long-running; use tmux. Usage: make run-seed-leg [DRY_RUN=1]
	bash scripts/training/run_seed_leg.sh

docker-train: ensure-dirs ## Run training. Usage: make docker-train [LAYOUT=rectangle] [STAGE=1] [BASELINE=vanilla_ppo] [CHECKPOINT=seed42_11062026-0628]
	@echo "Training: layout=$(LAYOUT) stage=$(or $(STAGE),1) checkpoint=$(CHECKPOINT_NAME) baseline=$(BASELINE_NAME) seed=agent_config"
	$(DOCKER_COMPOSE) down
	$(WORKERS_DOWN)
	$(DOCKER_COMPOSE) up -d --wait
	$(WORKERS_UP)
	$(DOCKER_COMPOSE) exec training bash scripts/training/train.sh $(if $(STAGE),--stage $(STAGE),) $(if $(CHECKPOINT),--resume-from $(CHECKPOINT_DIR),) $(if $(BASELINE),--baseline $(BASELINE_YAML),)

docker-train-short: ensure-dirs ## Quick training (10k steps). Usage: make docker-train-short [LAYOUT=rectangle] [STAGE=1] [BASELINE=vanilla_ppo] [CHECKPOINT=seed42_11062026-0628]
	@echo "Training (10k steps): layout=$(LAYOUT) stage=$(or $(STAGE),1) checkpoint=$(CHECKPOINT_NAME) baseline=$(BASELINE_NAME) seed=agent_config"
	$(DOCKER_COMPOSE) down
	$(WORKERS_DOWN)
	$(DOCKER_COMPOSE) up -d --wait
	$(WORKERS_UP)
	$(DOCKER_COMPOSE) exec training bash scripts/training/train.sh --total-timesteps 10000 $(if $(STAGE),--stage $(STAGE),) $(if $(CHECKPOINT),--resume-from $(CHECKPOINT_DIR),) $(if $(BASELINE),--baseline $(BASELINE_YAML),)

docker-tune: ensure-dirs ## Run Optuna hyperparameter tuning. Usage: make docker-tune [LAYOUT=rectangle] [STAGE=1] [BASELINE=vanilla_ppo]
	@echo "Tuning: layout=$(LAYOUT) stage=$(or $(STAGE),1) baseline=$(BASELINE_NAME)"
	$(DOCKER_COMPOSE) down
	$(WORKERS_DOWN)
	$(DOCKER_COMPOSE) up -d --wait
	$(WORKERS_UP)
	$(DOCKER_COMPOSE) exec training bash scripts/training/tune.sh $(if $(STAGE),--stage $(STAGE),) $(if $(BASELINE),--baseline $(BASELINE_YAML),)

docker-eval: ensure-dirs ## Run evaluation inside container. Usage: make docker-eval [LAYOUT=rectangle] [BASELINE=vanilla_ppo] [CHECKPOINT=seed42_11062026-0628] [SCENARIO="gnss_degraded"|"gnss_fixed gnss_degraded"] [PER_STEP_CAP=20] [NO_SAFETY=1]
	@echo "Evaluation: layout=$(LAYOUT) checkpoint=$(CHECKPOINT_NAME) baseline=$(BASELINE_NAME) seed=$(EVAL_SEED) scenario=$(if $(filter command line,$(origin SCENARIO)),$(SCENARIO),<all>) per_step_cap=$(PER_STEP_CAP) safety_wrapper=$(if $(NO_SAFETY),OFF,ON)"
	$(DOCKER_COMPOSE) down
	$(WORKERS_DOWN)
	$(DOCKER_COMPOSE) up -d --wait
	bash scripts/multi_workers/workers_up.sh 1
	$(DOCKER_COMPOSE) exec -e EVAL_PER_STEP_CAP=$(PER_STEP_CAP) -e EVAL_DISABLE_SAFETY_WRAPPER=$(if $(NO_SAFETY),1,0) -e EVAL_SEED=$(EVAL_SEED) training python $(SRC_DIR)/evaluation/evaluate.py \
		--model-path $(CHECKPOINT_MODEL) \
		--eval-config $(CONFIG_DIR)/eval_config.yaml \
		--env-config $(CONFIG_DIR)/deployment/sim/env_config.yaml \
		--train-config $(CONFIG_DIR)/train_config.yaml \
		$(if $(BASELINE),--baseline $(BASELINE_YAML),) \
		$(if $(filter command line,$(origin SCENARIO)),--conditions $(SCENARIO),) \
		--output-dir $(EVAL_RESULTS_ROOT)

docker-covariance-probe: ## Causal probe - does the policy USE the covariance input? Usage: make docker-covariance-probe BASELINE=full_method CHECKPOINT=seed42_11062026-0628 [REAL_OBS=outputs/raw/evaluation_results/seed_42/full_method/<leaf>/real_observations.npy]
	@echo "Covariance probe: checkpoint=$(CHECKPOINT_NAME) baseline=$(BASELINE_NAME)$(if $(REAL_OBS), (on-manifold),)"
	$(DOCKER_COMPOSE) exec training python $(SCRIPTS_DIR)/analysis/covariance_probe.py \
		--model-path $(CHECKPOINT_MODEL) \
		$(if $(REAL_OBS),--real-obs $(REAL_OBS),)


docker-eval-visualise-3d: ## Load checkpoint + CARLA 3D chase view. Usage: make docker-eval-visualise-3d [BASELINE=full_method] [CHECKPOINT=6_42_22062026-1502] [STAGE=6] [GNSS_TIER=fixed|float|standalone|degraded]
	$(eval _DISPLAY := $(or $(DISPLAY),$(shell ls /tmp/.X11-unix/X* 2>/dev/null | head -1 | sed 's|/tmp/.X11-unix/X|:|'),$(error No display attached!)))
	@echo "Demo drive 3D: checkpoint=$(CHECKPOINT_NAME), stage=$(if $(STAGE),$(STAGE),<base>), gnss_tier=$(if $(GNSS_TIER),$(GNSS_TIER),<sampled>)"
	@# Naming only the demo services keeps --abort-on-container-exit from being
	@# tripped by an unrelated container. Demo CARLA has its own port range but
	@# competes for the GPU, so stop a training run first.
	docker rm -f uncertainty-rl-carla-demo uncertainty-rl-checkpoint-demo uncertainty-rl-ros2-inspect 2>/dev/null || true
	@# Grant on the RESOLVED display: DISPLAY may be unset here, and CARLA maps
	@# its window once at startup, so a late grant does not help.
	DISPLAY=$(_DISPLAY) xhost +local:docker 2>/dev/null || true
	DISPLAY=$(_DISPLAY) CHECKPOINT=$(CHECKPOINT_MODEL) \
		DEMO_BASELINE_ARG="$(if $(BASELINE),--baseline $(BASELINE_YAML),)" \
		DEMO_STAGE_ARG="$(if $(STAGE),--stage $(STAGE),)" \
		DEMO_TIER_ARG="$(if $(GNSS_TIER),--gnss-tier $(GNSS_TIER),)" \
		$(DOCKER_COMPOSE_INSPECT) --profile demo up --build --force-recreate \
			--abort-on-container-exit carla-server-demo ros2-bridge-inspect checkpoint-demo
	DISPLAY=$(_DISPLAY) xhost -local:docker 2>/dev/null || true
	@# The demo container outlives an aborted server and hangs at "Driving.",
	@# so clear it here rather than leaving it for the next run to trip over.
	docker rm -f uncertainty-rl-carla-demo uncertainty-rl-checkpoint-demo uncertainty-rl-ros2-inspect 2>/dev/null || true


docker-test: ## Run full test suite inside container
	@bash scripts/multi_workers/ensure_stack.sh
	$(DOCKER_COMPOSE) exec training pytest $(TESTS_DIR) -v --tb=short

docker-test-unit: ## Run unit tests inside container
	@bash scripts/multi_workers/ensure_stack.sh
	$(DOCKER_COMPOSE) exec training pytest $(TESTS_DIR) -v --tb=short -m "not integration"

docker-test-integration: ## Run integration tests inside container
	@bash scripts/multi_workers/ensure_stack.sh
	$(DOCKER_COMPOSE) exec training pytest $(TESTS_DIR) -v --tb=short -m "integration"


docker-shell: ## Interactive shell in training container
	$(DOCKER_COMPOSE) exec training /bin/bash

WORKER ?= 0
docker-shell-ros2: ## Interactive shell in ROS 2 bridge for a worker. Usage: make docker-shell-ros2 [WORKER=0]
	docker exec -it uncertainty-rl-ros2-$(WORKER) /bin/bash

docker-shell-ros2-inspect: ## Interactive shell in ROS 2 inspect container
	$(DOCKER_COMPOSE_INSPECT) exec ros2-bridge-inspect /bin/bash

docker-logs: ## Follow logs from training stack containers (training, tensorboard)
	$(DOCKER_COMPOSE) logs -f

docker-logs-training: ## Follow logs from training container
	$(DOCKER_COMPOSE) logs -f training

docker-logs-carla: ## Follow logs from CARLA server for a worker. Usage: make docker-logs-carla [WORKER=0]
	docker logs -f uncertainty-rl-carla-$(WORKER)

docker-logs-ros2: ## Follow logs from ROS 2 bridge for a worker. Usage: make docker-logs-ros2 [WORKER=0]
	docker logs -f uncertainty-rl-ros2-$(WORKER)

docker-inspect-dryrun-logs: ## Follow dryrun training container logs (run alongside docker-inspect-dryrun)
	$(DOCKER_COMPOSE_INSPECT) logs -f training-inspect-dryrun

docker-logs-ros2-inspect: ## Follow ROS 2 inspect container logs (run alongside docker-inspect-dryrun)
	$(DOCKER_COMPOSE_INSPECT) logs -f ros2-bridge-inspect


STACK ?= all
docker-clean: ## Stop containers and remove volumes. Usage: make docker-clean [STACK=all|training|inspect]
	bash scripts/cleanup/stack_clean.sh $(STACK)

docker-clean-all: ## Remove all containers, images, and volumes. Usage: make docker-clean-all [STACK=all|training|inspect]
	bash scripts/cleanup/stack_clean.sh $(STACK) --rmi

docker-dev: ## Start N env workers + training stack and drop into training shell (GPU machine workflow)
	$(WORKERS_UP)
	$(DOCKER_COMPOSE) up -d --wait
	$(DOCKER_COMPOSE) exec training /bin/bash


MODEL ?= checkpoints/final_model
docker-demo: ## Windowed CARLA demo with checkpoint (requires X11). Usage: make docker-demo MODEL=<path>
	$(eval _DISPLAY := $(or $(DISPLAY),$(shell ls /tmp/.X11-unix/X* 2>/dev/null | head -1 | sed 's|/tmp/.X11-unix/X|:|'),$(error No display attached!)))
	xhost +local:docker 2>/dev/null || true
	DISPLAY=$(_DISPLAY) MODEL=$(MODEL) $(DOCKER_COMPOSE_INSPECT) --profile demo up --abort-on-container-exit
	xhost -local:docker 2>/dev/null || true


INSPECT_EPISODES ?=
INSPECT_VIEW     ?= third_person
INSPECT_PAUSE    ?= 3.0
INSPECT_OOD      ?= false
docker-inspect-dryrun: ## Full training pipeline in windowed CARLA, built identically to training. Usage: make docker-inspect-dryrun [STAGE=1] [BASELINE=configs/baselines/vanilla_ppo.yaml] [MANUAL=true] [INSPECT_EPISODES=5] [INSPECT_VIEW=third_person|side|back|front|free|birds_eye] [INSPECT_PAUSE=3.0] [INSPECT_OOD=true|false]
	@echo "Dryrun: stage=$(or $(STAGE),1) baseline=$(BASELINE_NAME) manual=$(MANUAL) ood=$(INSPECT_OOD)"
	$(eval _DISPLAY := $(or $(DISPLAY),$(shell ls /tmp/.X11-unix/X* 2>/dev/null | head -1 | sed 's|/tmp/.X11-unix/X|:|'),$(error No display attached!)))
	$(DOCKER_COMPOSE) down 2>/dev/null || true
	docker rm -f uncertainty-rl-carla-demo uncertainty-rl-ros2-inspect uncertainty-rl-training-inspect-dryrun 2>/dev/null || true
	docker network prune -f 2>/dev/null || true
	@# Stale signal files would be read as this run's data on bridge startup.
	rm -f outputs/initial_pose.json outputs/ekf_state.json outputs/ekf_state.json.tmp 2>/dev/null || true
	xhost +local:docker 2>/dev/null || true
	DISPLAY=$(_DISPLAY) EPISODES=$(INSPECT_EPISODES) \
		INSPECT_VIEW=$(INSPECT_VIEW) INSPECT_PAUSE=$(INSPECT_PAUSE) \
		INSPECT_MANUAL=$(MANUAL) INSPECT_OOD=$(INSPECT_OOD) \
		INSPECT_STAGE=$(STAGE) INSPECT_BASELINE=$(if $(BASELINE),$(BASELINE_YAML),) \
		bash scripts/inspect/dryrun.sh

SCENARIO ?= anchor_deployment
docker-inspect-eval-dryrun: ## Manually drive a named eval condition in windowed CARLA (no checkpoint), to verify the eval scenario wiring. Usage: make docker-inspect-eval-dryrun SCENARIO=anchor_deployment [BASELINE=full_method] [MANUAL=true] [INSPECT_EPISODES=5] [INSPECT_VIEW=third_person|side|back|front|free|birds_eye] [INSPECT_PAUSE=3.0]
	@echo "Eval dryrun: scenario=$(SCENARIO) baseline=$(BASELINE_NAME) manual=$(MANUAL)"
	$(eval _DISPLAY := $(or $(DISPLAY),$(shell ls /tmp/.X11-unix/X* 2>/dev/null | head -1 | sed 's|/tmp/.X11-unix/X|:|'),$(error No display attached!)))
	$(DOCKER_COMPOSE) down 2>/dev/null || true
	docker rm -f uncertainty-rl-carla-demo uncertainty-rl-ros2-inspect uncertainty-rl-training-inspect-eval-dryrun 2>/dev/null || true
	docker network prune -f 2>/dev/null || true
	@# Stale signal files would be read as this run's data on bridge startup.
	rm -f outputs/initial_pose.json outputs/ekf_state.json outputs/ekf_state.json.tmp 2>/dev/null || true
	xhost +local:docker 2>/dev/null || true
	DISPLAY=$(_DISPLAY) EPISODES=$(INSPECT_EPISODES) \
		INSPECT_VIEW=$(INSPECT_VIEW) INSPECT_PAUSE=$(INSPECT_PAUSE) \
		INSPECT_MANUAL=$(MANUAL) \
		INSPECT_SCENARIO=$(SCENARIO) INSPECT_BASELINE=$(if $(BASELINE),$(BASELINE_YAML),) \
		INSPECT_PROFILE=inspect-eval-dryrun \
		INSPECT_SERVICE=training-inspect-eval-dryrun \
		INSPECT_CONTAINER=uncertainty-rl-training-inspect-eval-dryrun \
		bash scripts/inspect/dryrun.sh

docker-inspect: ## Spawn a layout in windowed CARLA for visual inspection (includes perimeter cones). Usage: make docker-inspect [INSPECT_LAYOUT=rectangle]
	$(eval _DISPLAY := $(or $(DISPLAY),$(shell ls /tmp/.X11-unix/X* 2>/dev/null | head -1 | sed 's|/tmp/.X11-unix/X|:|'),$(error No display attached!)))
	docker rm -f uncertainty-rl-carla-demo uncertainty-rl-training-inspect 2>/dev/null || true
	$(DOCKER_COMPOSE) down 2>/dev/null || true
	docker network prune -f 2>/dev/null || true
	xhost +local:docker 2>/dev/null || true
	DISPLAY=$(_DISPLAY) LAYOUT=$(INSPECT_LAYOUT) $(DOCKER_COMPOSE_INSPECT) --profile inspect up --force-recreate --abort-on-container-exit carla-server-demo training-inspect
	xhost -local:docker 2>/dev/null || true

SENSORS_VIEW    ?= birds_eye
INSPECT_ZOOM    ?= close
docker-inspect-sensors: ## Visualise sensor FOV on the parking lot layout in windowed CARLA. Usage: make docker-inspect-sensors [INSPECT_LAYOUT=rectangle|trapezoid|irregular_a] [SENSORS_VIEW=birds_eye|side|front] [INSPECT_ZOOM=close|wide]
	$(eval _DISPLAY := $(or $(DISPLAY),$(shell ls /tmp/.X11-unix/X* 2>/dev/null | head -1 | sed 's|/tmp/.X11-unix/X|:|'),$(error No display attached!)))
	docker rm -f uncertainty-rl-carla-demo uncertainty-rl-training-inspect-sensors 2>/dev/null || true
	$(DOCKER_COMPOSE) down 2>/dev/null || true
	docker network prune -f 2>/dev/null || true
	xhost +local:docker 2>/dev/null || true
	DISPLAY=$(_DISPLAY) LAYOUT=$(INSPECT_LAYOUT) VIEW=$(SENSORS_VIEW) ZOOM=$(INSPECT_ZOOM) $(DOCKER_COMPOSE_INSPECT) --profile inspect-sensors up --force-recreate --abort-on-container-exit carla-server-demo training-inspect-sensors
	xhost -local:docker 2>/dev/null || true

INSPECT_SENSOR  ?= lidar
docker-inspect-live: ## Live sensor mode in windowed CARLA. Usage: make docker-inspect-live [INSPECT_SENSOR=lidar] [INSPECT_LAYOUT=rectangle|trapezoid|irregular_a]
	$(eval _DISPLAY := $(or $(DISPLAY),$(shell ls /tmp/.X11-unix/X* 2>/dev/null | head -1 | sed 's|/tmp/.X11-unix/X|:|'),$(error No display attached!)))
	docker rm -f uncertainty-rl-carla-demo uncertainty-rl-training-inspect-live 2>/dev/null || true
	$(DOCKER_COMPOSE) down 2>/dev/null || true
	docker network prune -f 2>/dev/null || true
	xhost +local:docker 2>/dev/null || true
	DISPLAY=$(_DISPLAY) LAYOUT=$(INSPECT_LAYOUT) SENSOR=$(INSPECT_SENSOR) $(DOCKER_COMPOSE_INSPECT) --profile inspect-live up --force-recreate --abort-on-container-exit carla-server-demo training-inspect-live
	xhost -local:docker 2>/dev/null || true





figures: ## Regenerate the analysed-data figures into outputs/main_analysis/figures. Usage: make figures [FIG=gate_roc]
	$(call ensure-venv)
	mkdir -p outputs/main_analysis/figures
	$(PYTHON) scripts/analysis/figures/build.py \
		$(if $(filter command line,$(origin FIG)),--only $(FIG),--all)

run-figures: ## Redraw the per-run eval panels into outputs/raw_derived/figures/per_run. Usage: make run-figures [RUN_DIR=outputs/raw/evaluation_results/seed_42/full_method/<leaf>/without_wrapper]
	$(call ensure-venv)
	$(PYTHON) $(SCRIPTS_DIR)/analysis/figures/run_figures.py \
		--root $(or $(RESULTS_ROOT),outputs/raw/evaluation_results) \
		--output-dir $(or $(OUTPUT_DIR),outputs/raw_derived/per_run_figures) \
		$(if $(filter command line,$(origin RUN_DIR)),--run-dir $(RUN_DIR),)

generate-layouts: ## Generate lot layout YAMLs + bird's-eye PNGs. Usage: make generate-layouts [LAYOUT=trapezoid]
	$(call ensure-venv)
	mkdir -p configs/layouts outputs/raw_derived/layouts
	$(PYTHON) scripts/layouts/generate_layouts.py \
		--output-dir configs/layouts \
		--plot-dir outputs/raw_derived/layouts \
		$(if $(filter command line,$(origin LAYOUT)),--layout $(LAYOUT),)

# Everything below is host-side and CPU-only: reads the raw CSVs (or TensorBoard
# event files) and writes derived CSVs. No CARLA, no ROS 2, no GPU.
analyse-markov: ## Diagnose GNSS tier Markov chain from gnss_noise_profiles.yaml. Usage: make analyse-markov [N_EPISODES=10000] [N_STEPS=1750]
	$(call ensure-venv)
	$(PYTHON) scripts/diagnostics/markov_analyser.py \
		$(if $(filter command line,$(origin N_EPISODES)),--n-episodes $(N_EPISODES),) \
		$(if $(filter command line,$(origin N_STEPS)),--n-steps $(N_STEPS),)

trace-tier-breakdown: ## Resolve demo-trace success/pos-error by GNSS tier (collapse vs hard-task). Usage: make trace-tier-breakdown TRACE_DIR=outputs/raw/demo_traces/<baseline>/<leaf>/<timestamp>
	$(call ensure-venv)
	@if [ -z "$(TRACE_DIR)" ]; then echo "Set TRACE_DIR=outputs/raw/demo_traces/<baseline>/<leaf>/<timestamp>"; exit 1; fi
	$(PYTHON) scripts/analysis/trace_tiers.py --trace-dir $(TRACE_DIR)

analyse-ablation: ## Cross-arm covariance contrast + degradation slope from eval CSVs. Usage: make analyse-ablation [STAGE=1] [SEED=42] [CHECKPOINT=1_42_19062026-0120] [SLOPE_CLEAN=gnss_fixed SLOPE_DEGRADED=gnss_degraded]
	$(call ensure-venv)
	$(PYTHON) scripts/analysis/ablation.py \
		--results-root $(or $(RESULTS_ROOT),$(EVAL_RESULTS_ROOT)) \
		--output-dir $(or $(OUTPUT_DIR),$(ABLATION_ROOT)) \
		--slope-clean $(or $(SLOPE_CLEAN),gnss_fixed) \
		--slope-degraded $(or $(SLOPE_DEGRADED),gnss_degraded) \
		$(if $(CHECKPOINT),--checkpoint $(CHECKPOINT),) \
		$(if $(STAGE),--stage $(STAGE),)

analyse-gate: ## EKF-std vs evidential-epistemic safety-gate ROC from eval CSVs. Usage: make analyse-gate [STAGE=1] [SEED=42]
	$(call ensure-venv)
	$(PYTHON) scripts/analysis/gate_roc.py \
		--results-root $(or $(RESULTS_ROOT),$(EVAL_RESULTS_ROOT)) \
		--output-dir $(or $(OUTPUT_DIR),$(GATE_ROOT)) \
		$(if $(STAGE),--stage $(STAGE),)

analyse-calibration: ## Is the EKF covariance an honest signal (std vs actual error)? Usage: make analyse-calibration [ARM=full_method] [CHECKPOINT=1_42_19062026-0120]
	$(call ensure-venv)
	$(PYTHON) scripts/analysis/calibration.py \
		--results-root $(or $(RESULTS_ROOT),$(EVAL_RESULTS_ROOT)) \
		--output-dir $(or $(OUTPUT_DIR),$(CALIBRATION_ROOT)) \
		$(if $(ARM),--arm $(ARM),) \
		$(if $(CHECKPOINT),--checkpoint $(CHECKPOINT),)

handover-timing: ## When does the wrapper hand over vs degradation onset? Usage: make handover-timing [ARM=full_method] [CHECKPOINT=1_42_19062026-0120]
	$(call ensure-venv)
	$(PYTHON) scripts/analysis/handover_timing.py \
		--results-root $(or $(RESULTS_ROOT),$(EVAL_RESULTS_ROOT)) \
		--output-dir $(or $(OUTPUT_DIR),$(HANDOVER_ROOT)) \
		$(if $(ARM),--arm $(ARM),) \
		$(if $(CHECKPOINT),--checkpoint $(CHECKPOINT),)

analyse-cross-seed: ## Pool all seeds into headline tables + per-seed robustness. Usage: make analyse-cross-seed [STAGE=6] [SLOPE_CLEAN=gnss_fixed SLOPE_DEGRADED=gnss_degraded]
	$(call ensure-venv)
	# Cross-seed spans seeds: parent root, NEVER EVAL_RESULTS_ROOT (which is seed_<N>/).
	$(PYTHON) scripts/analysis/cross_seed.py \
		--results-root $(or $(RESULTS_ROOT),outputs/raw/evaluation_results) \
		--output-dir $(or $(OUTPUT_DIR),outputs/raw_derived/cross_seed_analysis) \
		--stage $(or $(STAGE),6) \
		--slope-clean $(or $(SLOPE_CLEAN),gnss_fixed) \
		--slope-degraded $(or $(SLOPE_DEGRADED),gnss_degraded)

training-curves: ## Export seed-averaged training curves from the TensorBoard logs to CSV. Usage: make training-curves [LOGS_ROOT=logs]
	$(call ensure-venv)
	$(PYTHON) $(SCRIPTS_DIR)/analysis/tb_curves.py \
		--logs-root $(or $(LOGS_ROOT),logs) \
		--output-dir $(or $(OUTPUT_DIR),outputs/raw_derived/training)

analysis-bundle: ## Assemble the summaries and values into outputs/main_analysis. Run after `make figures`. Usage: make analysis-bundle [STAGE=6]
	$(call ensure-venv)
	$(PYTHON) $(SCRIPTS_DIR)/analysis/bundle.py \
		--frozen $(or $(FROZEN),outputs/raw_derived/cross_seed_analysis/all_seeds/stage$(or $(STAGE),6)) \
		--raw $(or $(RESULTS_ROOT),outputs/raw/evaluation_results) \
		--output-dir $(or $(OUTPUT_DIR),outputs/main_analysis)

uncertainty-verdict: ## Judge epistemic-vs-aleatoric separation. Usage: make uncertainty-verdict EVAL_DIR=outputs/raw/evaluation_results/seed_42/<baseline>/<leaf>/without_wrapper
	$(call ensure-venv)
	@if [ -z "$(EVAL_DIR)" ]; then echo "Set EVAL_DIR=<eval run dir with per_step_records.csv>"; exit 1; fi
	$(PYTHON) $(SCRIPTS_DIR)/analysis/uncertainty_verdict.py $(EVAL_DIR)

tb-scalars: ## Print TB scalar trajectories. Usage: make tb-scalars LOG=logs/<run_dir> [ARGS="--match success --points 20 --last 10"]
	$(call ensure-venv)
	@if [ -z "$(LOG)" ]; then echo "Set LOG=logs/<run_dir>"; exit 1; fi
	$(PYTHON) $(SCRIPTS_DIR)/diagnostics/tb_read.py $(LOG) $(ARGS)


# WORKER selects which env worker's vis_history file to watch.
# Worker 0 (default): outputs/vis_history.jsonl
# Worker N: outputs/vis_history_N.jsonl
_VIS_FILE = $(if $(filter 0,$(WORKER)),outputs/vis_history.jsonl,outputs/vis_history_$(WORKER).jsonl)

# Recording flags shared by the two viewer targets. RECORD=true adds --record;
# the viewer's R key works either way.
_VIS_FLAGS = --ui-scale $(UI_SCALE) --record-dir $(RECORD_DIR) --fps $(REC_FPS) \
	$(if $(filter true,$(RECORD)),--record,) \
	$(if $(filter true,$(SHOW_EPISODE)),--show-episode,)

visualise: ## Open 2D bird's-eye viewer. Usage: make visualise [WORKER=0] [RECORD=false] [UI_SCALE=1.5]
	$(call ensure-venv)
	@if [ "$(RECORD)" = "true" ]; then $(MAKE) --no-print-directory check-host-deps; fi
	$(eval _DISPLAY := $(or $(DISPLAY),$(shell ls /tmp/.X11-unix/X* 2>/dev/null | head -1 | sed 's|/tmp/.X11-unix/X|:|')))
	@if [ -z "$(_DISPLAY)" ]; then echo "No display attached!"; exit 1; fi
	PYTHONPATH=$(CURDIR) DISPLAY=$(_DISPLAY) \
		$(PYTHON) scripts/visualise/visualiser.py --history-file $(_VIS_FILE) $(_VIS_FLAGS)

eval-visualise-2d: ## Load checkpoint, start demo drive, open 2D viewer. Usage: make eval-visualise-2d [LAYOUT=rectangle] [BASELINE=vanilla_ppo] [CHECKPOINT=seed42_11062026-0628] [REALTIME=false] [STAGE=N] [GNSS_TIER=fixed|float|standalone|degraded] [RECORD=false] [TRACE=true] [UI_SCALE=1.5]
	$(call ensure-venv)
	@if [ "$(RECORD)" = "true" ]; then $(MAKE) --no-print-directory check-host-deps; fi
	$(eval _DISPLAY := $(or $(DISPLAY),$(shell ls /tmp/.X11-unix/X* 2>/dev/null | head -1 | sed 's|/tmp/.X11-unix/X|:|')))
	@if [ -z "$(_DISPLAY)" ]; then echo "No display attached!"; exit 1; fi
	@echo "Demo drive 2D: layout=$(LAYOUT), checkpoint=$(CHECKPOINT_NAME), gnss_tier=$(if $(GNSS_TIER),$(GNSS_TIER),<sampled>), record=$(RECORD), trace=$(TRACE)"
	@# This tears the WHOLE stack down first (workers + compose, orphans
	@# included), so it will kill a training run in progress. TRACE=false only
	@# suppresses the demo's own CSVs; it does not make this target read-only.
	@echo "NOTE: this stops any running training stack before starting the demo."
	@# Tear down any pre-existing stack first (orphans included).
	$(WORKERS_DOWN)
	$(DOCKER_COMPOSE) down --remove-orphans
	@# Clear stale viewer history (may be root-owned, hence sudo).
	sudo rm -f $(_VIS_FILE) outputs/.vis_active
	@# Training writes bay_successes/ as root; pre-own eval/ or the demo CSV dump fails.
	sudo mkdir -p outputs/raw/bay_successes/eval && sudo chown $$(id -u):$$(id -g) outputs/raw/bay_successes/eval
	$(DOCKER_COMPOSE) up -d --wait
	bash scripts/multi_workers/workers_up.sh 1
	@# Demo + log stream + viewer + graceful teardown live in the helper script.
	DISPLAY=$(_DISPLAY) \
		DEMO_CHECKPOINT=$(CHECKPOINT_MODEL) \
		DEMO_BASELINE_YAML=$(if $(BASELINE),$(BASELINE_YAML),) \
		DEMO_STAGE=$(STAGE) \
		DEMO_GNSS_TIER=$(GNSS_TIER) \
		DEMO_REALTIME=$(REALTIME) \
		DEMO_TRACE=$(TRACE) \
		DEMO_VIS_FILE=$(_VIS_FILE) \
		DEMO_VIS_FLAGS="$(_VIS_FLAGS)" \
		bash scripts/visualise/eval_visualise_2d.sh

clip: ## Cut a GIF/MP4 from the newest recording (or VIDEO=<path>). Usage: make clip START=00:05 END=00:20 [VIDEO=...] [FORMAT=gif] [WIDTH=800] [FPS=15] [OUT=docs/media/<name>.gif]
	@$(MAKE) --no-print-directory check-host-deps
	@# Defaults to the newest recording: the timestamped names are awkward to
	@# type and it is nearly always the clip just captured.
	$(eval _VIDEO := $(if $(VIDEO),$(VIDEO),$(shell ls -t $(RECORD_DIR)/*.mp4 2>/dev/null | head -1)))
	@if [ -z "$(_VIDEO)" ]; then \
		echo "No recordings in $(RECORD_DIR)/ - set VIDEO=<path>.mp4"; exit 1; fi
	@if [ ! -f "$(_VIDEO)" ]; then echo "No such video: $(_VIDEO)"; exit 1; fi
	@if [ -z "$(VIDEO)" ]; then echo "Using newest recording: $(_VIDEO)"; fi
	@if [ -z "$(START)" ] || [ -z "$(END)" ]; then \
		echo "Set START and END (MM:SS or seconds), e.g. START=00:05 END=00:20"; exit 1; fi
	@mkdir -p $(dir $(_VIDEO))
	$(eval _CLIP_OUT := $(if $(OUT),$(OUT),$(basename $(_VIDEO))_$(subst :,,$(START))-$(subst :,,$(END)).$(FORMAT)))
	@# -ss/-to before -i seeks on keyframes; re-encoding keeps the cut accurate.
	@# GIF uses the two-pass palette pipeline, far better on flat graphics.
	@if [ "$(FORMAT)" = "gif" ]; then \
		ffmpeg -hide_banner -loglevel error -y -ss $(START) -to $(END) -i "$(_VIDEO)" \
			-vf "fps=$(FPS),scale=$(WIDTH):-1:flags=lanczos,split[a][b];[a]palettegen[p];[b][p]paletteuse" \
			"$(_CLIP_OUT)"; \
	else \
		ffmpeg -hide_banner -loglevel error -y -ss $(START) -to $(END) -i "$(_VIDEO)" \
			-vf "scale=$(WIDTH):-2:flags=lanczos" -c:v libx264 -crf 18 -pix_fmt yuv420p \
			"$(_CLIP_OUT)"; \
	fi
	@echo "Wrote $(_CLIP_OUT) ($$(du -h '$(_CLIP_OUT)' | cut -f1))"

record-screen: ## Screen-record the CARLA window to MP4 until Ctrl+C. Usage: make record-screen [DURATION=30] [REC_HEIGHT=720] [REGION=WxH] [OFFSET=X,Y] [OUT=...]
	@$(MAKE) --no-print-directory check-host-deps
	@# CARLA draws its window server-side, so unlike the 2D viewer it cannot
	@# record itself; x11grab captures it from the host X display instead.
	$(eval _DISPLAY := $(or $(DISPLAY),$(shell ls /tmp/.X11-unix/X* 2>/dev/null | head -1 | sed 's|/tmp/.X11-unix/X|:|')))
	@if [ -z "$(_DISPLAY)" ]; then echo "No display attached!"; exit 1; fi
	@# Match the WM class: CARLA's title has trailing spaces, so -name fails.
	$(eval _WIN := $(shell DISPLAY=$(_DISPLAY) xwininfo -root -tree 2>/dev/null | grep -m1 CarlaUE4-Linux-Shipping))
	$(eval _AUTO_REGION := $(shell echo '$(_WIN)' | awk '{print $$(NF-1)}' | grep -oE '^[0-9]+x[0-9]+'))
	$(eval _AUTO_OFFSET := $(shell echo '$(_WIN)' | awk '{print $$NF}' | tr '+' ' ' | awk 'NF>=2 {print $$1","$$2}'))
	@# Both or neither: a region with an empty offset gives ffmpeg "-i :1+,".
	$(eval _REGION := $(if $(and $(_AUTO_REGION),$(_AUTO_OFFSET)),$(_AUTO_REGION),$(REGION)))
	$(eval _OFFSET := $(if $(and $(_AUTO_REGION),$(_AUTO_OFFSET)),$(_AUTO_OFFSET),$(OFFSET)))
	@if [ -z "$(_REGION)" ] || [ -z "$(_OFFSET)" ]; then \
		echo "Could not resolve a capture region. Is the CARLA window open?"; \
		echo "  Pass one explicitly: make record-screen REGION=1920x1080 OFFSET=0,0"; \
		exit 1; \
	fi
	@mkdir -p $(RECORD_DIR)
	$(eval _SCREEN_OUT := $(if $(OUT),$(OUT),$(RECORD_DIR)/screen-$(shell date +%d-%m-%Y-%H%M%S).mp4))
	@echo "Recording $(_REGION) at +$(_OFFSET) on $(_DISPLAY)$(if $(DURATION), for $(DURATION)s, until Ctrl+C) -> $(_SCREEN_OUT)"
	@# Foreground with the terminal attached, so Ctrl+C reaches ffmpeg directly;
	@# backgrounding or redirecting stdin detaches it and kills x11grab. `pad`
	@# rounds up because libx264 needs even dimensions; the fragmented-MP4
	@# movflags write the index incrementally so a Ctrl+C still leaves it playable.
	@ffmpeg -hide_banner -loglevel error -y -f x11grab \
		-video_size $(_REGION) -framerate $(REC_FPS) -i "$(_DISPLAY)+$(_OFFSET)" \
		$(if $(DURATION),-t $(DURATION),) \
		-vf "scale=-2:$(REC_HEIGHT),pad=ceil(iw/2)*2:ceil(ih/2)*2" \
		-c:v libx264 -preset $(REC_PRESET) -crf $(REC_CRF) -tune zerolatency \
		-threads 0 -pix_fmt yuv420p \
		-movflags +frag_keyframe+empty_moov+default_base_moof \
		"$(_SCREEN_OUT)"; rc=$$?; \
	if [ ! -s "$(_SCREEN_OUT)" ]; then \
		echo "Recording failed - no data captured (ffmpeg exit $$rc)."; exit 1; \
	fi; \
	if [ $$rc -ne 0 ] && [ $$rc -ne 255 ] && \
	   ! xwininfo -root -tree 2>/dev/null | grep -q CarlaUE4-Linux-Shipping; then \
		echo ""; \
		echo "NOTE: the CARLA window disappeared, so the capture ended early."; \
		echo "  Stop the recording (Ctrl+C here) BEFORE stopping the demo."; \
		echo ""; \
	fi; \
	echo "Recording stopped."; \
	echo "Finalising..."; \
	if ffmpeg -hide_banner -loglevel error -y -i "$(_SCREEN_OUT)" -c copy \
		-movflags +faststart "$(_SCREEN_OUT).tmp.mp4" 2>/dev/null \
		&& [ -s "$(_SCREEN_OUT).tmp.mp4" ]; then \
		mv -f "$(_SCREEN_OUT).tmp.mp4" "$(_SCREEN_OUT)"; \
	else \
		rm -f "$(_SCREEN_OUT).tmp.mp4"; \
	fi
	@# Prove the result is readable before claiming success, so a broken capture
	@# is reported now rather than discovered when a player refuses to open it.
	@if ! ffprobe -v error -show_entries format=duration -of csv=p=0 \
		"$(_SCREEN_OUT)" >/dev/null 2>&1; then \
		echo "WARNING: $(_SCREEN_OUT) is not readable - the capture was cut"; \
		echo "  short before any video was written. Record again."; \
		exit 1; \
	fi
	@echo "Wrote $(_SCREEN_OUT) ($$(du -h '$(_SCREEN_OUT)' | cut -f1))"
	@echo "Cut a GIF with: make clip VIDEO=$(_SCREEN_OUT) START=00:02 END=00:12"

check-host-deps: ## Verify host-side tools the recording targets need (ffmpeg)
	@if command -v ffmpeg >/dev/null 2>&1; then \
		echo "ffmpeg: $$(ffmpeg -version | head -1)"; \
	else \
		echo "ffmpeg: MISSING."; \
		echo "  The 2D viewer runs on the host, so recording and 'make clip'"; \
		echo "  need the host ffmpeg binary (it is not in any container)."; \
		echo "  Install it with: sudo apt-get install ffmpeg"; \
		exit 1; \
	fi


test-unit: ## Run unit tests only (no GPU, no CARLA, no ROS 2)
	$(call ensure-venv)
	$(PYTEST) $(TESTS_DIR) -v --tb=short -m "not integration"


lint: ## Run all linters (flake8 + isort + black)
	$(call ensure-venv)
	$(VENV)/bin/flake8 $(SRC_DIR) $(TESTS_DIR) $(SCRIPTS_DIR)
	$(VENV)/bin/isort --check-only --diff $(SRC_DIR) $(TESTS_DIR) $(SCRIPTS_DIR)
	$(VENV)/bin/black --check $(SRC_DIR) $(TESTS_DIR) $(SCRIPTS_DIR)

format: ## Auto-format code with black + isort
	$(call ensure-venv)
	$(VENV)/bin/isort $(SRC_DIR) $(TESTS_DIR) $(SCRIPTS_DIR)
	$(VENV)/bin/black $(SRC_DIR) $(TESTS_DIR) $(SCRIPTS_DIR)

typecheck: ## Run mypy type checking
	$(call ensure-venv)
	$(VENV)/bin/mypy $(SRC_DIR) --ignore-missing-imports


sanity: ## Quick import check
	$(call ensure-venv)
	$(PYTHON) -c "import uncertainty_rl; print('Package imports OK')"

# Mirrors CI exactly: only .[dev] is installed (no torch), so lint + typecheck
# + import and no tests. The tests need torch - see docker-test-unit.
verify: lint typecheck sanity ## Run the CI checks locally (lint + typecheck + import). Tests: docker-test-unit.


clean-cache: ## Remove only build caches and .pyc files (preserves checkpoints, logs, outputs, maps)
	rm -rf __pycache__ .pytest_cache htmlcov .mypy_cache
	find . -path ./$(VENV) -prune -o -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -path ./$(VENV) -prune -o -type f -name "*.pyc" -delete 2>/dev/null || true

clean: ## Remove build artefacts, caches, generated outputs, maps, and layouts (preserves checkpoints, logs, .xodr, .venv)
	rm -rf __pycache__ .pytest_cache htmlcov .mypy_cache
	sudo rm -rf evaluation_results/ experiments/ results/
	sudo rm -rf outputs/
	find configs/layouts/ -type f ! -name "*.xodr" -delete 2>/dev/null || true
	find . -path ./$(VENV) -prune -o -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -path ./$(VENV) -prune -o -type f -name "*.pyc" -delete 2>/dev/null || true

clean-all: ## Remove everything including checkpoints and logs (preserves .xodr and .venv)
	rm -rf __pycache__ .pytest_cache htmlcov .mypy_cache
	sudo rm -rf logs/ checkpoints/ evaluation_results/ experiments/ results/
	sudo rm -rf outputs/
	find configs/layouts/ -type f ! -name "*.xodr" -delete 2>/dev/null || true
	find . -path ./$(VENV) -prune -o -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -path ./$(VENV) -prune -o -type f -name "*.pyc" -delete 2>/dev/null || true

clean-venv: ## Remove the local virtual environment (re-create with make install)
	rm -rf $(VENV)


backup-configs: ## Pack CLAUDE.md, TODO.md, documentation/, and the real-world datum into project_configs.tar.gz
	@find . -name "CLAUDE.md" -not -path "./.venv/*" > /tmp/_backup_files.txt
	@echo "TODO.md" >> /tmp/_backup_files.txt
	@find ./documentation -type f >> /tmp/_backup_files.txt 2>/dev/null || true
	@# Site-specific RTK datum (gitignored, hand-measured, not reproducible from code).
	@[ -f configs/deployment/real/real_world_datum.yaml ] && \
		echo "configs/deployment/real/real_world_datum.yaml" >> /tmp/_backup_files.txt || true
	tar -czf project_configs.tar.gz -T /tmp/_backup_files.txt
	@rm -f /tmp/_backup_files.txt
	@echo "Backed up to project_configs.tar.gz ($$(du -h project_configs.tar.gz | cut -f1))"

restore-configs: ## Restore CLAUDE.md, TODO.md, documentation/, and the real-world datum from project_configs.tar.gz
	tar -xzf project_configs.tar.gz
	@echo "Restored configs from project_configs.tar.gz"

# Gitignored trees holding GPU time a fresh clone cannot regenerate. The archive
# lands in the repo root, which `make clean` never touches, so a backup survives
# the very targets that delete what it holds.
RESULTS_TREES = checkpoints logs outputs
RESULTS_ARCHIVE ?= project_results.tar.gz

backup-results: ## Archive checkpoints/, logs/ and outputs/ (multi-GB). Usage: make backup-results [RESULTS_ARCHIVE=project_results.tar.gz]
	@present="$$(for d in $(RESULTS_TREES); do [ -d "$$d" ] && echo $$d; done)"; \
	if [ -z "$$present" ]; then echo "Nothing to back up - none of $(RESULTS_TREES) exist."; exit 1; fi; \
	echo "Archiving: $$(echo $$present | tr '\n' ' ')"; \
	echo "On disk: $$(du -csh $$present | tail -1 | cut -f1) (model .zip files are already compressed, so gzip gains little)"; \
	tar -czf $(RESULTS_ARCHIVE) $$present
	@echo "Wrote $(RESULTS_ARCHIVE) ($$(du -h $(RESULTS_ARCHIVE) | cut -f1))"

restore-results: ## Restore checkpoints/, logs/ and outputs/ from the archive. Usage: make restore-results [RESULTS_ARCHIVE=...] [FORCE=1]
	@[ -f $(RESULTS_ARCHIVE) ] || { echo "$(RESULTS_ARCHIVE) not found."; exit 1; }
	@# Refuse to overwrite existing trees unless asked: restoring over a newer
	@# run would silently mix two sets of results.
	@if [ -z "$(FORCE)" ]; then \
		for d in $(RESULTS_TREES); do \
			if [ -d "$$d" ]; then \
				echo "$$d/ already exists. Move it aside, or pass FORCE=1 to overwrite."; exit 1; \
			fi; \
		done; \
	fi
	tar -xzf $(RESULTS_ARCHIVE)
	@echo "Restored from $(RESULTS_ARCHIVE):"
	@for d in $(RESULTS_TREES); do [ -d "$$d" ] && printf "  %-14s %s\n" "$$d" "$$(du -sh $$d | cut -f1)"; done || true

list-results-archive: ## Show what is inside the results archive without extracting. Usage: make list-results-archive [RESULTS_ARCHIVE=...]
	@[ -f $(RESULTS_ARCHIVE) ] || { echo "$(RESULTS_ARCHIVE) not found."; exit 1; }
	@echo "$(RESULTS_ARCHIVE) ($$(du -h $(RESULTS_ARCHIVE) | cut -f1)), top-level entries:"
	@tar -tzf $(RESULTS_ARCHIVE) | awk -F/ '{print $$1"/"$$2}' | sort -u | head -30
	@echo "total entries: $$(tar -tzf $(RESULTS_ARCHIVE) | wc -l)"
