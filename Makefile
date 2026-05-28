# Development commands for training, evaluation, testing, and linting.

.PHONY: help install test test-unit test-integration
.PHONY: lint format typecheck verify clean clean-cache clean-all clean-venv
.PHONY: backup-configs restore-configs
.PHONY: generate-layouts visualise eval-visualise-2d docker-eval-visualise-3d
.PHONY: docker-build docker-build-no-cache docker-build-no-cache-core docker-build-no-cache-inspect docker-build-ros2 docker-up docker-down docker-restart docker-ps docker-watch docker-top
.PHONY: docker-eval
.PHONY: docker-test docker-test-unit docker-test-integration docker-verify docker-lint docker-format docker-typecheck
.PHONY: docker-shell docker-shell-ros2 docker-shell-ros2-inspect docker-logs docker-logs-training docker-logs-carla docker-logs-ros2 docker-inspect-dryrun-logs docker-logs-ros2-inspect
.PHONY: docker-clean docker-clean-all docker-dev docker-demo docker-inspect docker-inspect-down docker-inspect-sensors docker-inspect-live docker-inspect-dryrun
.PHONY: docker-train docker-train-short docker-tune
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

# ----------------------------------------------------------------------
# Help
# ----------------------------------------------------------------------

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}'

# ----------------------------------------------------------------------
# Setup
# ----------------------------------------------------------------------

install: ## Create .venv and install package + dev dependencies
	python3 -m venv $(VENV)
	$(VENV)/bin/pip install --upgrade pip
	$(VENV)/bin/pip install -e ".[dev]"
	$(VENV)/bin/pre-commit install || true  # Non-fatal: core.hooksPath may be managed externally


# ======================================================================
# DOCKER - commands that run inside containers
# ======================================================================

# ----------------------------------------------------------------------
# Docker: Lifecycle
# ----------------------------------------------------------------------

# Pre-create host-side bind-mount targets so the Docker daemon (root) does not
# create them as root-owned. Must run before any `docker compose up` call.
ensure-dirs: ## Pre-create host directories for bind mounts (avoids root-owned logs/)
	@mkdir -p logs/ros2 outputs checkpoints

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
	$(DOCKER_COMPOSE_INSPECT) build --no-cache

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
	$(DOCKER_COMPOSE_INSPECT) --profile inspect --profile inspect-dryrun --profile inspect-sensors --profile inspect-live down
	docker rm -f uncertainty-rl-carla-demo uncertainty-rl-ros2-inspect uncertainty-rl-training-inspect uncertainty-rl-training-inspect-dryrun uncertainty-rl-training-inspect-sensors uncertainty-rl-training-inspect-live 2>/dev/null || true
	xhost -local:docker 2>/dev/null || true

docker-restart: ## Restart all containers
	$(DOCKER_COMPOSE) restart

docker-ps: ## Show running containers
	$(DOCKER_COMPOSE) ps

docker-watch: ## Watch container health status (refreshes every 5s, Ctrl+C to exit)
	watch -n 5 docker compose ps

docker-top: ## Show running processes in containers
	$(DOCKER_COMPOSE) top

# ----------------------------------------------------------------------
# Docker: Training & Evaluation
# ----------------------------------------------------------------------

docker-train: ensure-dirs ## Run training. Usage: make docker-train [LAYOUT=rectangle] [CHECKPOINT=path/to/checkpoint]
	@echo "Training: layout=$(LAYOUT) checkpoint=$(CHECKPOINT)"
	$(DOCKER_COMPOSE) down
	$(WORKERS_DOWN)
	$(DOCKER_COMPOSE) up -d --wait
	$(WORKERS_UP)
	$(DOCKER_COMPOSE) exec training bash scripts/training/train.sh $(if $(CHECKPOINT),--resume-from $(CHECKPOINT),)

docker-train-short: ensure-dirs ## Quick training (10k steps). Usage: make docker-train-short [LAYOUT=rectangle] [CHECKPOINT=path/to/checkpoint]
	@echo "Training (10k steps): layout=$(LAYOUT) checkpoint=$(CHECKPOINT)"
	$(DOCKER_COMPOSE) down
	$(WORKERS_DOWN)
	$(DOCKER_COMPOSE) up -d --wait
	$(WORKERS_UP)
	$(DOCKER_COMPOSE) exec training bash scripts/training/train.sh --total-timesteps 10000 $(if $(CHECKPOINT),--resume-from $(CHECKPOINT),)

docker-tune: ensure-dirs ## Run Optuna hyperparameter tuning. Usage: make docker-tune [LAYOUT=rectangle]
	@echo "Tuning: layout=$(LAYOUT)"
	$(DOCKER_COMPOSE) down
	$(WORKERS_DOWN)
	$(DOCKER_COMPOSE) up -d --wait
	$(WORKERS_UP)
	$(DOCKER_COMPOSE) exec training bash scripts/training/tune.sh

docker-eval: ensure-dirs ## Run evaluation inside container. Usage: make docker-eval [LAYOUT=rectangle] [CHECKPOINT=path]
	@echo "Evaluation: layout=$(LAYOUT) checkpoint=$(or $(CHECKPOINT),checkpoints/final_model)"
	$(DOCKER_COMPOSE) down
	$(WORKERS_DOWN)
	$(DOCKER_COMPOSE) up -d --wait
	bash scripts/multi_workers/workers_up.sh 1
	$(DOCKER_COMPOSE) exec training python $(SRC_DIR)/evaluation/evaluate.py \
		--model-path $(or $(CHECKPOINT),checkpoints/final_model) \
		--eval-config $(CONFIG_DIR)/eval_config.yaml \
		--env-config $(CONFIG_DIR)/deployment/sim/env_config.yaml \
		--train-config $(CONFIG_DIR)/train_config.yaml \
		--output-dir evaluation_results

docker-eval-visualise-3d: ## Load checkpoint + CARLA 3D spectator view. Usage: make docker-eval-visualise-3d [CHECKPOINT=path]
	$(eval _DISPLAY := $(or $(DISPLAY),$(shell ls /tmp/.X11-unix/X* 2>/dev/null | head -1 | sed 's|/tmp/.X11-unix/X|:|'),$(error No display attached!)))
	DISPLAY=$(_DISPLAY) CHECKPOINT=$(or $(CHECKPOINT),checkpoints/final_model) \
		$(DOCKER_COMPOSE_INSPECT) --profile demo up --build --abort-on-container-exit

# ----------------------------------------------------------------------
# Docker: Testing & Linting
# ----------------------------------------------------------------------

docker-test: ## Run full test suite inside container 
	@bash scripts/multi_workers/ensure_stack.sh
	$(DOCKER_COMPOSE) exec training pytest $(TESTS_DIR) -v --tb=short

docker-test-unit: ## Run unit tests inside container 
	@bash scripts/multi_workers/ensure_stack.sh
	$(DOCKER_COMPOSE) exec training pytest $(TESTS_DIR) -v --tb=short -m "not integration"

docker-test-integration: ## Run integration tests inside container 
	@bash scripts/multi_workers/ensure_stack.sh
	$(DOCKER_COMPOSE) exec training pytest $(TESTS_DIR) -v --tb=short -m "integration"

docker-verify: ## Run all checks inside container 
	@bash scripts/multi_workers/ensure_stack.sh
	$(DOCKER_COMPOSE) exec training bash -c "pytest $(TESTS_DIR) -v --tb=short -m 'not integration' && flake8 $(SRC_DIR) $(TESTS_DIR) $(SCRIPTS_DIR) --max-line-length 88 --extend-ignore E203,W503 && isort --check-only --diff $(SRC_DIR) $(TESTS_DIR) $(SCRIPTS_DIR) && black --check $(SRC_DIR) $(TESTS_DIR) $(SCRIPTS_DIR) && mypy $(SRC_DIR) --ignore-missing-imports && python -c 'import uncertainty_rl; print(\"All checks passed.\")'"

docker-lint: ## Run linters inside container 
	@bash scripts/multi_workers/ensure_stack.sh
	$(DOCKER_COMPOSE) exec training make lint

docker-format: ## Format code inside container 
	@bash scripts/multi_workers/ensure_stack.sh
	$(DOCKER_COMPOSE) exec training make format

docker-typecheck: ## Run mypy inside container 
	@bash scripts/multi_workers/ensure_stack.sh
	$(DOCKER_COMPOSE) exec training make typecheck

# ----------------------------------------------------------------------
# Docker: Shells & Logs
# ----------------------------------------------------------------------

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

# ----------------------------------------------------------------------
# Docker: Cleanup
# ----------------------------------------------------------------------

STACK ?= all
docker-clean: ## Stop containers and remove volumes. Usage: make docker-clean [STACK=all|training|inspect]
	bash scripts/cleanup/stack_clean.sh $(STACK)

docker-clean-all: ## Remove all containers, images, and volumes. Usage: make docker-clean-all [STACK=all|training|inspect]
	bash scripts/cleanup/stack_clean.sh $(STACK) --rmi

docker-dev: ## Start N env workers + training stack and drop into training shell (GPU machine workflow)
	$(WORKERS_UP)
	$(DOCKER_COMPOSE) up -d --wait
	$(DOCKER_COMPOSE) exec training /bin/bash

# ----------------------------------------------------------------------
# Docker: Demo to evaluate model in windowed mode
# ----------------------------------------------------------------------

MODEL ?= checkpoints/final_model
docker-demo: ## Windowed CARLA demo with checkpoint (requires X11). Usage: make docker-demo MODEL=<path>
	$(eval _DISPLAY := $(or $(DISPLAY),$(shell ls /tmp/.X11-unix/X* 2>/dev/null | head -1 | sed 's|/tmp/.X11-unix/X|:|'),$(error No display attached!)))
	xhost +local:docker 2>/dev/null || true
	DISPLAY=$(_DISPLAY) MODEL=$(MODEL) $(DOCKER_COMPOSE_INSPECT) --profile demo up --abort-on-container-exit
	xhost -local:docker 2>/dev/null || true

# ----------------------------------------------------------------------
# Docker: Inspection Tools to confirm all is good in the simulator
# ----------------------------------------------------------------------

INSPECT_EPISODES ?=
INSPECT_VIEW     ?= third_person
INSPECT_PAUSE    ?= 3.0
INSPECT_OOD      ?= false
docker-inspect-dryrun: ## Full training pipeline in windowed CARLA. Default: constant forward drive. Usage: make docker-inspect-dryrun [MANUAL=true] [INSPECT_EPISODES=5] [INSPECT_VIEW=third_person|side|back|front|free|birds_eye] [INSPECT_PAUSE=3.0] [INSPECT_OOD=true|false]
	$(eval _DISPLAY := $(or $(DISPLAY),$(shell ls /tmp/.X11-unix/X* 2>/dev/null | head -1 | sed 's|/tmp/.X11-unix/X|:|'),$(error No display attached!)))
	$(DOCKER_COMPOSE) down 2>/dev/null || true
	docker rm -f uncertainty-rl-carla-demo uncertainty-rl-ros2-inspect uncertainty-rl-training-inspect-dryrun 2>/dev/null || true
	docker network prune -f 2>/dev/null || true
	@# Clean stale signal files from previous runs to prevent the ros2-bridge
	@# from processing leftover initial_pose or ekf_state data on startup.
	rm -f outputs/initial_pose.json outputs/ekf_state.json outputs/ekf_state.json.tmp 2>/dev/null || true
	xhost +local:docker 2>/dev/null || true
	DISPLAY=$(_DISPLAY) EPISODES=$(INSPECT_EPISODES) \
		INSPECT_VIEW=$(INSPECT_VIEW) INSPECT_PAUSE=$(INSPECT_PAUSE) \
		INSPECT_MANUAL=$(MANUAL) INSPECT_OOD=$(INSPECT_OOD) \
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



# ======================================================================
# LOCAL - commands that run on the host machine
# ======================================================================

# ----------------------------------------------------------------------
# Layout Generation
# ----------------------------------------------------------------------

generate-layouts: ## Generate lot layout YAMLs + bird's-eye PNGs. Usage: make generate-layouts [LAYOUT=trapezoid]
	$(call ensure-venv)
	mkdir -p configs/layouts outputs/layouts
	$(PYTHON) scripts/layouts/generate_layouts.py \
		--output-dir configs/layouts \
		--plot-dir outputs/layouts \
		$(if $(filter command line,$(origin LAYOUT)),--layout $(LAYOUT),)

# ----------------------------------------------------------------------
# Markov Chain Analysis
# ----------------------------------------------------------------------

analyse-markov: ## Diagnose GNSS tier Markov chain from gnss_noise_profiles.yaml. Usage: make analyse-markov [N_EPISODES=10000] [N_STEPS=1750]
	$(call ensure-venv)
	$(PYTHON) scripts/inspect/markov_analyser.py \
		$(if $(filter command line,$(origin N_EPISODES)),--n-episodes $(N_EPISODES),) \
		$(if $(filter command line,$(origin N_STEPS)),--n-steps $(N_STEPS),)

# ----------------------------------------------------------------------
# Visualisation (host-side viewer + Docker driver)
# Two use cases:
#   make visualise          - training already running, just open the viewer
#   make eval-visualise-2d  - start checkpoint demo drive + open viewer
# ----------------------------------------------------------------------

# WORKER selects which env worker's vis_history file to watch.
# Worker 0 (default): outputs/vis_history.jsonl
# Worker N: outputs/vis_history_N.jsonl
_VIS_FILE = $(if $(filter 0,$(WORKER)),outputs/vis_history.jsonl,outputs/vis_history_$(WORKER).jsonl)

visualise: ## Open 2D bird's-eye viewer. Usage: make visualise [WORKER=0]
	$(call ensure-venv)
	$(eval _DISPLAY := $(or $(DISPLAY),$(shell ls /tmp/.X11-unix/X* 2>/dev/null | head -1 | sed 's|/tmp/.X11-unix/X|:|')))
	@if [ -z "$(_DISPLAY)" ]; then echo "No display attached!"; exit 1; fi
	PYTHONPATH=$(CURDIR) DISPLAY=$(_DISPLAY) \
		$(PYTHON) scripts/visualise/visualiser.py --history-file $(_VIS_FILE)

eval-visualise-2d: ## Load checkpoint, start demo drive, open 2D viewer. Usage: make eval-visualise-2d [LAYOUT=rectangle] [CHECKPOINT=path] [REALTIME=false]
	$(call ensure-venv)
	$(eval _DISPLAY := $(or $(DISPLAY),$(shell ls /tmp/.X11-unix/X* 2>/dev/null | head -1 | sed 's|/tmp/.X11-unix/X|:|')))
	@if [ -z "$(_DISPLAY)" ]; then echo "No display attached!"; exit 1; fi
	@echo "Demo drive 2D: layout=$(LAYOUT), checkpoint=$(or $(CHECKPOINT),checkpoints/final_model)"
	@# Tear down any pre-existing stack first. 
	$(WORKERS_DOWN)
	$(DOCKER_COMPOSE) down --remove-orphans
	@# Also clear the visualisation history so the viewer starts on this
	@# run's frames, not stale ones left over from a previous session.
	$(DOCKER_COMPOSE) up -d --wait
	bash scripts/multi_workers/workers_up.sh 1
	@# Start the demo container detached, then run the viewer in the foreground.
	@set -e; \
	demo_cid=$$($(DOCKER_COMPOSE) --profile demo run --rm -d demo \
		python $(SCRIPTS_DIR)/visualise/demo_drive.py \
		--checkpoint $(or $(CHECKPOINT),checkpoints/final_model) \
		--env-config $(CONFIG_DIR)/deployment/sim/env_config.yaml \
		--train-config $(CONFIG_DIR)/train_config.yaml \
		$(if $(filter false,$(REALTIME)),--no-realtime,) | tail -n1); \
	echo "Demo container: $$demo_cid"; \
	trap 'echo "Stopping demo container..."; docker rm -f $$demo_cid >/dev/null 2>&1 || true' EXIT INT TERM; \
	PYTHONPATH=$(CURDIR) DISPLAY=$(_DISPLAY) \
		$(PYTHON) scripts/visualise/visualiser.py --history-file $(_VIS_FILE)

# ----------------------------------------------------------------------
# Testing
# ----------------------------------------------------------------------

test-unit: ## Run unit tests only (no GPU, no CARLA, no ROS 2)
	$(call ensure-venv)
	$(PYTEST) $(TESTS_DIR) -v --tb=short -m "not integration"

# ----------------------------------------------------------------------
# Linting & Formatting
# ----------------------------------------------------------------------

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

# ----------------------------------------------------------------------
# Sanity Check
# ----------------------------------------------------------------------

sanity: ## Quick import check
	$(call ensure-venv)
	$(PYTHON) -c "import uncertainty_rl; print('Package imports OK')"

verify: lint typecheck sanity ## Run all local checks (lint + typecheck + sanity).

# ----------------------------------------------------------------------
# Cleanup
# ----------------------------------------------------------------------

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

# ----------------------------------------------------------------------
# Config Backup
# ----------------------------------------------------------------------

backup-configs: ## Pack all CLAUDE.md, TODO.md and documentation/ into project_configs.tar.gz
	@find . -name "CLAUDE.md" -not -path "./.venv/*" > /tmp/_backup_files.txt
	@echo "TODO.md" >> /tmp/_backup_files.txt
	@find ./documentation -type f >> /tmp/_backup_files.txt 2>/dev/null || true
	tar -czf project_configs.tar.gz -T /tmp/_backup_files.txt
	@rm -f /tmp/_backup_files.txt
	@echo "Backed up to project_configs.tar.gz ($$(du -h project_configs.tar.gz | cut -f1))"

restore-configs: ## Restore CLAUDE.md, TODO.md, documentation/, and .github/ from project_configs.tar.gz
	tar -xzf project_configs.tar.gz
	@echo "Restored configs from project_configs.tar.gz"