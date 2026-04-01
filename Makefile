# Makefile - Uncertainty-Conditioned RL
# Development commands for training, evaluation, testing, and linting.

.PHONY: help install test test-unit test-integration verify
.PHONY: lint format typecheck clean clean-venv
.PHONY: backup-configs restore-configs
.PHONY: generate-layouts visualise eval-visualise-2d docker-eval-visualise-3d
.PHONY: docker-build docker-build-no-cache docker-up docker-down docker-restart docker-ps docker-watch docker-top
.PHONY: docker-eval
.PHONY: docker-test docker-test-unit docker-test-integration docker-verify docker-lint docker-format docker-typecheck
.PHONY: docker-shell docker-shell-ros2 docker-shell-ros2-inspect docker-logs docker-logs-training docker-logs-carla docker-logs-ros2 docker-inspect-dryrun-logs docker-logs-ros2-inspect
.PHONY: docker-clean docker-clean-all docker-dev docker-demo docker-inspect docker-inspect-down docker-inspect-sensors docker-inspect-live docker-inspect-dryrun
.PHONY: docker-map docker-train-loc docker-train-loc-short

VENV        := .venv
PYTHON      := $(VENV)/bin/python3
PYTEST      := $(VENV)/bin/pytest
CONFIG_DIR  := configs
SRC_DIR     := uncertainty_rl
TESTS_DIR   := tests
SCRIPTS_DIR := scripts
DOCKER_COMPOSE         := docker compose
DOCKER_COMPOSE_INSPECT := docker compose -f docker-compose.yml -f docker-compose.inspect.yml

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
	$(VENV)/bin/pre-commit install


# ======================================================================
# DOCKER - commands that run inside containers
# ======================================================================

# ----------------------------------------------------------------------
# Docker: Lifecycle
# ----------------------------------------------------------------------
SERVICE ?=
docker-build: ## Build all Docker images (core + inspect stacks). Usage: make docker-build
	$(DOCKER_COMPOSE) build $(SERVICE)
	$(DOCKER_COMPOSE_INSPECT) build $(SERVICE)

docker-build-no-cache: ## Build images without cache (clean rebuild)
	$(DOCKER_COMPOSE) build --no-cache
	$(DOCKER_COMPOSE_INSPECT) build --no-cache

docker-up: ## Start all containers
	$(DOCKER_COMPOSE) up -d

docker-down: ## Stop all containers
	$(DOCKER_COMPOSE) down

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
# Cartographer SLAM mapping workflow (one-time per floor plan)
# ----------------------------------------------------------------------
# Two-step process:
#   1. make docker-up              (start stack in SLAM mode, default)
#   2. make docker-map LAYOUT=...  (drive patrol loop + serialise .pbstream in one step)
# Then for training use make docker-train-loc LAYOUT=...
# ----------------------------------------------------------------------

LAYOUT ?= rectangle

# Read sensor_suite from carla/env_config.yaml (single source of truth for env settings).
# suite_a -> 2d maps, suite_b/suite_c -> 3d maps.
SENSOR_SUITE := $(shell grep '^sensor_suite:' $(CONFIG_DIR)/carla/env_config.yaml | awk '{print $$2}')
MAP_DIM := $(if $(filter suite_a,$(SENSOR_SUITE)),2d,3d)

docker-map: ## Drive patrol loop + serialise Cartographer map. Usage: make docker-map [LAYOUT=rectangle]
	mkdir -p outputs/maps/$(MAP_DIM) configs/maps/2d configs/maps/3d
	@echo "Mapping: layout=$(LAYOUT), suite=$(SENSOR_SUITE), map_dim=$(MAP_DIM)"
	@# Restart the full stack so CARLA has a clean world (no stale actors from
	@# previous runs) and the bridge starts fresh with Cartographer in SLAM mode.
	@# --wait blocks until all healthchecks pass (CARLA ~60s, bridge ~30s).
	SENSOR_SUITE=$(SENSOR_SUITE) $(DOCKER_COMPOSE) down
	SENSOR_SUITE=$(SENSOR_SUITE) $(DOCKER_COMPOSE) up -d --wait
	SENSOR_SUITE=$(SENSOR_SUITE) $(DOCKER_COMPOSE) exec training python -m scripts.mapping.mapping_drive \
		--layout $(LAYOUT) \
		--carla-host carla-server \
		--carla-port 2000
	bash scripts/mapping/save_map.sh $(LAYOUT) $(MAP_DIM)

# ----------------------------------------------------------------------
# Docker: Training & Evaluation
# ----------------------------------------------------------------------

# Shared env vars for localisation mode (used by train/eval targets below).
# export ensures they persist across chained commands in a single shell recipe.
LOC_ENV = export CARTOGRAPHER_MODE=loc \
	CARTOGRAPHER_MAP=/workspace/configs/maps/$(MAP_DIM)/$(LAYOUT).pbstream \
	SENSOR_SUITE=$(SENSOR_SUITE)

docker-train-loc: ## Run training in pure localisation mode. Usage: make docker-train-loc [LAYOUT=rectangle]
	@echo "Training (loc): layout=$(LAYOUT), suite=$(SENSOR_SUITE), map_dim=$(MAP_DIM)"
	$(LOC_ENV) && $(DOCKER_COMPOSE) down && $(DOCKER_COMPOSE) up -d --wait
	$(LOC_ENV) && $(DOCKER_COMPOSE) exec training bash scripts/training/train.sh

docker-train-loc-short: ## Quick training (10k steps) in pure localisation mode. Usage: make docker-train-loc-short [LAYOUT=rectangle]
	@echo "Training (loc, 10k steps): layout=$(LAYOUT), suite=$(SENSOR_SUITE), map_dim=$(MAP_DIM)"
	$(LOC_ENV) && $(DOCKER_COMPOSE) down && $(DOCKER_COMPOSE) up -d --wait
	$(LOC_ENV) && $(DOCKER_COMPOSE) exec training bash scripts/training/train.sh --total-timesteps 10000


docker-eval: ## Run evaluation inside container. Usage: make docker-eval [LAYOUT=rectangle]
	@echo "Evaluation (loc): layout=$(LAYOUT), suite=$(SENSOR_SUITE), map_dim=$(MAP_DIM)"
	$(LOC_ENV) && $(DOCKER_COMPOSE) down && $(DOCKER_COMPOSE) up -d --wait
	$(LOC_ENV) && $(DOCKER_COMPOSE) exec training python $(SRC_DIR)/evaluation/evaluate.py \
		--model-path checkpoints/final_model \
		--eval-config $(CONFIG_DIR)/eval_config.yaml \
		--env-config $(CONFIG_DIR)/carla/env_config.yaml \
		--train-config $(CONFIG_DIR)/train_config.yaml \
		--output-dir evaluation_results

docker-eval-visualise-3d: ## Load checkpoint + CARLA 3D spectator view. Usage: make docker-eval-visualise-3d [CHECKPOINT=path]
	@echo "Demo drive 3D: checkpoint=$(or $(CHECKPOINT),checkpoints/final_model)"
	DISPLAY=$(or $(DISPLAY),:0) CHECKPOINT=$(or $(CHECKPOINT),checkpoints/final_model) \
		$(DOCKER_COMPOSE_INSPECT) --profile demo up --build --abort-on-container-exit

# ----------------------------------------------------------------------
# Docker: Testing & Linting
# ----------------------------------------------------------------------

# Ensure the core training stack is running.
# Only starts containers if the training service is not already up.
# Never touches the inspect stack (docker-compose.inspect.yml).
define ensure-core-stack-running
	@if ! $(DOCKER_COMPOSE) ps --status running training 2>/dev/null | grep -q training; then \
		echo "Core stack not running -- starting (this may take up to 90s)..."; \
		$(DOCKER_COMPOSE) up -d --wait; \
	fi
endef

docker-test: ## Run full test suite inside container (auto-starts core stack if needed)
	$(call ensure-core-stack-running)
	$(DOCKER_COMPOSE) exec training pytest $(TESTS_DIR) -v --tb=short

docker-test-unit: ## Run unit tests inside container (auto-starts core stack if needed)
	$(call ensure-core-stack-running)
	$(DOCKER_COMPOSE) exec training pytest $(TESTS_DIR) -v --tb=short -m "not integration"

docker-test-integration: ## Run integration tests inside container (auto-starts core stack if needed)
	$(call ensure-core-stack-running)
	$(DOCKER_COMPOSE) exec training pytest $(TESTS_DIR) -v --tb=short -m "integration"

docker-verify: ## Run all checks inside container (auto-starts core stack if needed)
	$(call ensure-core-stack-running)
	$(DOCKER_COMPOSE) exec training make verify

docker-lint: ## Run linters inside container (auto-starts core stack if needed)
	$(call ensure-core-stack-running)
	$(DOCKER_COMPOSE) exec training make lint

docker-format: ## Format code inside container (auto-starts core stack if needed)
	$(call ensure-core-stack-running)
	$(DOCKER_COMPOSE) exec training make format

docker-typecheck: ## Run mypy inside container (auto-starts core stack if needed)
	$(call ensure-core-stack-running)
	$(DOCKER_COMPOSE) exec training make typecheck

# ----------------------------------------------------------------------
# Docker: Shells & Logs
# ----------------------------------------------------------------------

docker-shell: ## Interactive shell in training container
	$(DOCKER_COMPOSE) exec training /bin/bash

docker-shell-ros2: ## Interactive shell in ROS 2 container
	$(DOCKER_COMPOSE) exec ros2-bridge /bin/bash

docker-shell-ros2-inspect: ## Interactive shell in ROS 2 inspect container (use while docker-inspect-dryrun is running)
	$(DOCKER_COMPOSE_INSPECT) exec ros2-bridge-inspect /bin/bash

docker-logs: ## Follow logs from all containers
	$(DOCKER_COMPOSE) logs -f

docker-logs-training: ## Follow logs from training container
	$(DOCKER_COMPOSE) logs -f training

docker-logs-carla: ## Follow logs from CARLA server
	$(DOCKER_COMPOSE) logs -f carla-server

docker-logs-ros2: ## Follow logs from ROS 2 bridge
	$(DOCKER_COMPOSE) logs -f ros2-bridge

docker-inspect-dryrun-logs: ## Follow dryrun training container logs (run alongside docker-inspect-dryrun)
	$(DOCKER_COMPOSE_INSPECT) logs -f training-inspect-dryrun

docker-logs-ros2-inspect: ## Follow ROS 2 inspect container logs (run alongside docker-inspect-dryrun)
	$(DOCKER_COMPOSE_INSPECT) logs -f ros2-bridge-inspect

# ----------------------------------------------------------------------
# Docker: Cleanup
# ----------------------------------------------------------------------

docker-clean: ## Stop containers and remove volumes
	$(DOCKER_COMPOSE) down -v
	docker volume prune -f

docker-clean-all: ## Remove all containers, images, and volumes
	$(DOCKER_COMPOSE) down -v --rmi all
	docker system prune -af

docker-dev: ## Start full stack and drop into training shell (GPU machine workflow)
	$(DOCKER_COMPOSE) up -d --wait
	$(DOCKER_COMPOSE) exec training /bin/bash

# ----------------------------------------------------------------------
# Docker: Demo to evaluate model in windowed mode
# ----------------------------------------------------------------------

MODEL ?= checkpoints/final_model
docker-demo: ## Windowed CARLA demo with checkpoint (requires X11). Usage: make docker-demo MODEL=<path>
	$(eval _DISPLAY := $(or $(DISPLAY),$(shell ls /tmp/.X11-unix/X* 2>/dev/null | head -1 | sed 's|/tmp/.X11-unix/X|:|'),$(error No X11 display found. Set DISPLAY manually: export DISPLAY=:0)))
	@echo "Using DISPLAY=$(_DISPLAY)"
	xhost +local:docker 2>/dev/null || true
	DISPLAY=$(_DISPLAY) MODEL=$(MODEL) $(DOCKER_COMPOSE_INSPECT) --profile demo up --abort-on-container-exit
	xhost -local:docker 2>/dev/null || true

# ----------------------------------------------------------------------
# Docker: Inspection Tools to confirm all is good in the simulator
# ----------------------------------------------------------------------

INSPECT_LAYOUT   ?= rectangle
INSPECT_EPISODES ?=
INSPECT_VIEW     ?= third_person
INSPECT_PAUSE    ?= 3.0
docker-inspect-dryrun: ## Full training pipeline with random actions in windowed CARLA. Usage: make docker-inspect-dryrun [INSPECT_LAYOUT=rectangle] [INSPECT_EPISODES=5] [INSPECT_VIEW=third_person|side|back|front|free] [INSPECT_PAUSE=3.0]
	$(eval _DISPLAY := $(or $(DISPLAY),$(shell ls /tmp/.X11-unix/X* 2>/dev/null | head -1 | sed 's|/tmp/.X11-unix/X|:|'),$(error No X11 display found. Set DISPLAY manually: export DISPLAY=:0)))
	$(eval LAYOUT := $(INSPECT_LAYOUT))
	@echo "Using DISPLAY=$(_DISPLAY)  LAYOUT=$(LAYOUT)  VIEW=$(INSPECT_VIEW)  PAUSE=$(INSPECT_PAUSE)  EPISODES=$(INSPECT_EPISODES)"
	$(DOCKER_COMPOSE) down 2>/dev/null || true
	docker rm -f uncertainty-rl-carla-demo uncertainty-rl-ros2-inspect uncertainty-rl-training-inspect-dryrun 2>/dev/null || true
	docker network prune -f 2>/dev/null || true
	xhost +local:docker 2>/dev/null || true
	DISPLAY=$(_DISPLAY) LAYOUT=$(LAYOUT) EPISODES=$(INSPECT_EPISODES) \
		INSPECT_VIEW=$(INSPECT_VIEW) INSPECT_PAUSE=$(INSPECT_PAUSE) \
		CARTOGRAPHER_MODE=loc \
		CARTOGRAPHER_MAP=/workspace/configs/maps/$(MAP_DIM)/$(LAYOUT).pbstream \
		SENSOR_SUITE=$(SENSOR_SUITE) \
		$(DOCKER_COMPOSE_INSPECT) --profile inspect-dryrun up \
		--force-recreate --detach \
		carla-server-demo ros2-bridge-inspect training-inspect-dryrun
	bash scripts/inspect/dryrun.sh

docker-inspect: ## Spawn a layout in windowed CARLA for visual inspection (includes perimeter cones). Usage: make docker-inspect [INSPECT_LAYOUT=rectangle]
	$(eval _DISPLAY := $(or $(DISPLAY),$(shell ls /tmp/.X11-unix/X* 2>/dev/null | head -1 | sed 's|/tmp/.X11-unix/X|:|'),$(error No X11 display found. Set DISPLAY manually: export DISPLAY=:0)))
	@echo "Using DISPLAY=$(_DISPLAY)"
	docker rm -f uncertainty-rl-carla-demo uncertainty-rl-training-inspect 2>/dev/null || true
	$(DOCKER_COMPOSE) down 2>/dev/null || true
	docker network prune -f 2>/dev/null || true
	xhost +local:docker 2>/dev/null || true
	DISPLAY=$(_DISPLAY) LAYOUT=$(INSPECT_LAYOUT) $(DOCKER_COMPOSE_INSPECT) --profile inspect up --force-recreate --abort-on-container-exit carla-server-demo training-inspect
	xhost -local:docker 2>/dev/null || true

INSPECT_SUITE   ?= suite_a
INSPECT_VIEW    ?= birds_eye
INSPECT_ZOOM    ?= close
docker-inspect-sensors: ## Visualise sensor FOV on the parking lot layout in windowed CARLA. Usage: make docker-inspect-sensors [INSPECT_SUITE=suite_a|suite_b|suite_c] [INSPECT_LAYOUT=rectangle|trapezoid|irregular_a] [INSPECT_VIEW=birds_eye|side|front] [INSPECT_ZOOM=close|wide]
	$(eval _DISPLAY := $(or $(DISPLAY),$(shell ls /tmp/.X11-unix/X* 2>/dev/null | head -1 | sed 's|/tmp/.X11-unix/X|:|'),$(error No X11 display found. Set DISPLAY manually: export DISPLAY=:0)))
	@echo "Using DISPLAY=$(_DISPLAY)"
	docker rm -f uncertainty-rl-carla-demo uncertainty-rl-training-inspect-sensors 2>/dev/null || true
	$(DOCKER_COMPOSE) down 2>/dev/null || true
	docker network prune -f 2>/dev/null || true
	xhost +local:docker 2>/dev/null || true
	DISPLAY=$(_DISPLAY) SUITE=$(INSPECT_SUITE) LAYOUT=$(INSPECT_LAYOUT) VIEW=$(INSPECT_VIEW) ZOOM=$(INSPECT_ZOOM) $(DOCKER_COMPOSE_INSPECT) --profile inspect-sensors up --force-recreate --abort-on-container-exit carla-server-demo training-inspect-sensors
	xhost -local:docker 2>/dev/null || true

INSPECT_SENSOR  ?= lidar
docker-inspect-live: ## Live sensor mode in windowed CARLA. INSPECT_SENSOR=camera forces suite_c automatically. Usage: make docker-inspect-live [INSPECT_SENSOR=lidar|camera] [INSPECT_SUITE=suite_a|suite_b|suite_c] [INSPECT_LAYOUT=rectangle|trapezoid|irregular_a]
	$(eval _DISPLAY := $(or $(DISPLAY),$(shell ls /tmp/.X11-unix/X* 2>/dev/null | head -1 | sed 's|/tmp/.X11-unix/X|:|'),$(error No X11 display found. Set DISPLAY manually: export DISPLAY=:0)))
	$(eval _SUITE := $(if $(filter camera,$(INSPECT_SENSOR)),suite_c,$(INSPECT_SUITE)))
	@echo "Using DISPLAY=$(_DISPLAY)  SUITE=$(_SUITE)  SENSOR=$(INSPECT_SENSOR)"
	docker rm -f uncertainty-rl-carla-demo uncertainty-rl-training-inspect-live 2>/dev/null || true
	$(DOCKER_COMPOSE) down 2>/dev/null || true
	docker network prune -f 2>/dev/null || true
	xhost +local:docker 2>/dev/null || true
	DISPLAY=$(_DISPLAY) SUITE=$(_SUITE) LAYOUT=$(INSPECT_LAYOUT) SENSOR=$(INSPECT_SENSOR) $(DOCKER_COMPOSE_INSPECT) --profile inspect-live up --force-recreate --abort-on-container-exit carla-server-demo training-inspect-live
	xhost -local:docker 2>/dev/null || true



# ======================================================================
# LOCAL - commands that run on the host machine
# ======================================================================

# ----------------------------------------------------------------------
# Layout Generation (no CARLA needed)
# ----------------------------------------------------------------------

generate-layouts: ## Generate lot layout YAMLs + bird's-eye PNGs (no CARLA needed). Usage: make generate-layouts [LAYOUT=trapezoid]
	$(call ensure-venv)
	mkdir -p configs/layouts outputs/layouts
	$(PYTHON) scripts/layouts/generate_layouts.py \
		--output-dir configs/layouts \
		--plot-dir outputs/layouts \
		$(if $(filter command line,$(origin LAYOUT)),--layout $(LAYOUT),)


# ----------------------------------------------------------------------
# Visualisation (host-side viewer + Docker driver)
# Two use cases:
#   make visualise          -- training already running, just open the viewer
#   make eval-visualise-2d  -- start checkpoint demo drive + open viewer
# ----------------------------------------------------------------------

visualise: ## Open 2D bird's-eye viewer (use while training is running). Usage: make visualise
	$(call ensure-venv)
	PYTHONPATH=$(CURDIR) DISPLAY=$(or $(DISPLAY),:0) $(PYTHON) scripts/visualise/visualiser.py

eval-visualise-2d: ## Load checkpoint, start demo drive, open 2D viewer. Usage: make eval-visualise-2d [LAYOUT=rectangle] [CHECKPOINT=path]
	$(call ensure-venv)
	@echo "Demo drive 2D: layout=$(LAYOUT), checkpoint=$(or $(CHECKPOINT),checkpoints/final_model)"
	$(LOC_ENV) && $(DOCKER_COMPOSE) up -d --wait
	$(LOC_ENV) && $(DOCKER_COMPOSE) --profile demo run --rm -d demo \
		python $(SCRIPTS_DIR)/visualise/demo_drive.py \
		--checkpoint $(or $(CHECKPOINT),checkpoints/final_model) \
		--env-config $(CONFIG_DIR)/carla/env_config.yaml \
		--train-config $(CONFIG_DIR)/train_config.yaml
	PYTHONPATH=$(CURDIR) DISPLAY=$(or $(DISPLAY),:0) $(PYTHON) scripts/visualise/visualiser.py

# ----------------------------------------------------------------------
# Testing
# ----------------------------------------------------------------------

test: ## Run full test suite (unit + integration)
	$(call ensure-venv)
	$(PYTEST) $(TESTS_DIR) -v --tb=short

test-unit: ## Run unit tests only (no GPU, no CARLA, no ROS 2)
	$(call ensure-venv)
	$(PYTEST) $(TESTS_DIR) -v --tb=short -m "not integration"

test-integration: ## Run integration tests (requires CARLA + ROS 2 + GPU)
	$(call ensure-venv)
	$(PYTEST) $(TESTS_DIR) -v --tb=short -m "integration"

verify: ## Run all CPU-only checks (tests + lint + typecheck + import sanity)
	$(call ensure-venv)
	$(PYTEST) $(TESTS_DIR) -v --tb=short -m "not integration"
	$(VENV)/bin/flake8 $(SRC_DIR) $(TESTS_DIR) $(SCRIPTS_DIR) --max-line-length 88 --extend-ignore E203,W503
	$(VENV)/bin/isort --check-only --diff $(SRC_DIR) $(TESTS_DIR) $(SCRIPTS_DIR)
	$(VENV)/bin/black --check $(SRC_DIR) $(TESTS_DIR) $(SCRIPTS_DIR)
	$(VENV)/bin/mypy $(SRC_DIR) --ignore-missing-imports
	$(PYTHON) -c "import uncertainty_rl; print('All checks passed.')"

# ----------------------------------------------------------------------
# Linting & Formatting
# ----------------------------------------------------------------------

lint: ## Run all linters (flake8 + isort + black)
	$(call ensure-venv)
	$(VENV)/bin/flake8 $(SRC_DIR) $(TESTS_DIR) $(SCRIPTS_DIR) --max-line-length 88 --extend-ignore E203,W503
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

# ----------------------------------------------------------------------
# Cleanup
# ----------------------------------------------------------------------

clean: ## Remove build artefacts, caches, generated outputs, maps, and layouts (preserves .xodr and .venv)
	rm -rf __pycache__ .pytest_cache htmlcov .mypy_cache
	sudo rm -rf logs/ checkpoints/ evaluation_results/ experiments/ results/
	sudo rm -rf outputs/
	sudo rm -rf configs/maps/
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