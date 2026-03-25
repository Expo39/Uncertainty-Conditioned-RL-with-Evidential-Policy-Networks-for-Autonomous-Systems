# Makefile - Uncertainty-Conditioned RL
# Development commands for training, evaluation, testing, and linting.

.PHONY: help install test test-unit test-integration verify
.PHONY: lint format typecheck clean syntax-check
.PHONY: backup-configs restore-configs
.PHONY: train-loc train-loc-short evaluate
.PHONY: generate-layouts visualise eval-visualise-2d docker-eval-visualise-3d
.PHONY: docker-build docker-build-prod docker-build-no-cache docker-up docker-down docker-restart docker-ps docker-watch docker-top
.PHONY: docker-eval
.PHONY: docker-test docker-test-unit docker-test-integration docker-verify docker-lint docker-format docker-typecheck
.PHONY: docker-shell docker-shell-ros2 docker-logs docker-logs-training docker-logs-carla docker-logs-ros2 docker-inspect-dryrun-logs
.PHONY: docker-clean docker-clean-all docker-full-build docker-dev docker-demo docker-inspect docker-inspect-sensors docker-inspect-live docker-inspect-dryrun docker-inspect-zcheck
.PHONY: docker-generate-layouts docker-map docker-train-loc docker-train-loc-short docker-watch-actors docker-watch-actors

PYTHON := python3
PYTHON_VIS := .venv-vis/bin/python3
PYTEST := pytest
CONFIG_DIR := configs
SRC_DIR := uncertainty_rl
TESTS_DIR := tests
SCRIPTS_DIR := scripts
DOCKER_COMPOSE         := docker compose
DOCKER_COMPOSE_INSPECT := docker compose -f docker-compose.yml -f docker-compose.inspect.yml

# ----------------------------------------------------------------------
# Help
# ----------------------------------------------------------------------

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}'

# ----------------------------------------------------------------------
# Setup
# ----------------------------------------------------------------------

install: ## Install package and dev dependencies
	pip install -e ".[dev]"
	pre-commit install


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

docker-build-prod: ## Build training image without dev dependencies (lighter)
	$(DOCKER_COMPOSE) build --build-arg DEV_INSTALL=false training

docker-build-no-cache: ## Build images without cache (clean rebuild)
	$(DOCKER_COMPOSE) build --no-cache
	$(DOCKER_COMPOSE_INSPECT) build --no-cache

docker-up: ## Start all containers
	$(DOCKER_COMPOSE) up -d

docker-down: ## Stop all containers
	$(DOCKER_COMPOSE) down

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

# Read sensor_suite from train_config.yaml (single source of truth).
# suite_a -> 2d maps, suite_b/suite_c -> 3d maps.
SENSOR_SUITE := $(shell grep '^sensor_suite:' $(CONFIG_DIR)/train_config.yaml | awk '{print $$2}')
MAP_DIM := $(if $(filter suite_a,$(SENSOR_SUITE)),2d,3d)

docker-map: ## Drive patrol loop + serialise Cartographer map. Usage: make docker-map [LAYOUT=rectangle]
	mkdir -p outputs/maps/$(MAP_DIM) configs/maps/2d configs/maps/3d
	@echo "Mapping: layout=$(LAYOUT), suite=$(SENSOR_SUITE), map_dim=$(MAP_DIM)"
	@# Restart the full stack so CARLA has a clean world (no stale actors from
	@# previous runs) and the bridge starts fresh with Cartographer in SLAM mode.
	@# --wait blocks until all healthchecks pass (CARLA ~60s, bridge ~30s).
	SENSOR_SUITE=$(SENSOR_SUITE) $(DOCKER_COMPOSE) down
	SENSOR_SUITE=$(SENSOR_SUITE) $(DOCKER_COMPOSE) up -d --wait
	SENSOR_SUITE=$(SENSOR_SUITE) $(DOCKER_COMPOSE) exec training python -m scripts.mapping_drive \
		--layout $(LAYOUT) \
		--carla-host carla-server \
		--carla-port 2000
	@echo "Serialising Cartographer state to .pbstream..."
	SENSOR_SUITE=$(SENSOR_SUITE) $(DOCKER_COMPOSE) exec ros2-bridge bash -c \
		"source /opt/ros/jazzy/setup.bash && \
		 source /workspace/install/setup.bash && \
		 ros2 service call /write_state cartographer_ros_msgs/srv/WriteState \
		 '{filename: \"/workspace/configs/maps/$(MAP_DIM)/$(LAYOUT).pbstream\", include_unfinished_submaps: true}'"
	@echo "Saved configs/maps/$(MAP_DIM)/$(LAYOUT).pbstream"
	@# Occupancy grid PNG is optional (Cairo can fail on small maps).
	@# The PGM is written to configs/maps/ (rw mount in ros2-bridge), then
	@# converted to PNG via the training container (which has PIL + outputs mount).
	-$(DOCKER_COMPOSE) exec ros2-bridge bash -c \
		"source /opt/ros/jazzy/setup.bash && \
		 source /workspace/install/setup.bash && \
		 ros2 run cartographer_ros cartographer_pbstream_to_ros_map \
		   --pbstream_filename /workspace/configs/maps/$(MAP_DIM)/$(LAYOUT).pbstream \
		   --map_filestem /workspace/configs/maps/$(MAP_DIM)/$(LAYOUT)_grid \
		   --resolution 0.05"
	@if [ -f configs/maps/$(MAP_DIM)/$(LAYOUT)_grid.pgm ]; then \
		$(DOCKER_COMPOSE) exec training python -c \
			"from PIL import Image; Image.open('/workspace/configs/maps/$(MAP_DIM)/$(LAYOUT)_grid.pgm').convert('RGB').save('/workspace/outputs/maps/$(MAP_DIM)/$(LAYOUT).png')"; \
		rm -f configs/maps/$(MAP_DIM)/$(LAYOUT)_grid.pgm configs/maps/$(MAP_DIM)/$(LAYOUT)_grid.yaml; \
		echo "Saved outputs/maps/$(MAP_DIM)/$(LAYOUT).png"; \
	fi

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
	$(LOC_ENV) && $(DOCKER_COMPOSE) exec training \
		bash -c "python $(SRC_DIR)/training/train_ppo.py \
		--config $(CONFIG_DIR)/train_config.yaml \
		--log-dir logs \
		--checkpoint-dir checkpoints \
		2> >(grep -Ev '^(>>>|<<<|$$|This error state|with this new error|rcutils_reset_error|rcutils_set_error_state|error_handling\.c|serdata\.cpp|should be called after|.*serdata.*)' >&2)"

docker-train-loc-short: ## Quick training (10k steps) in pure localisation mode. Usage: make docker-train-loc-short [LAYOUT=rectangle]
	@echo "Training (loc, 10k steps): layout=$(LAYOUT), suite=$(SENSOR_SUITE), map_dim=$(MAP_DIM)"
	$(LOC_ENV) && $(DOCKER_COMPOSE) down && $(DOCKER_COMPOSE) up -d --wait
	$(LOC_ENV) && $(DOCKER_COMPOSE) exec training \
		bash -c "python $(SRC_DIR)/training/train_ppo.py \
		--config $(CONFIG_DIR)/train_config.yaml \
		--total-timesteps 10000 \
		--log-dir logs \
		--checkpoint-dir checkpoints \
		2> >(grep -Ev '^(>>>|<<<|$$|This error state|with this new error|rcutils_reset_error|rcutils_set_error_state|error_handling\.c|serdata\.cpp|should be called after|.*serdata.*)' >&2)"


docker-eval: ## Run evaluation inside container. Usage: make docker-eval [LAYOUT=rectangle]
	@echo "Evaluation (loc): layout=$(LAYOUT), suite=$(SENSOR_SUITE), map_dim=$(MAP_DIM)"
	$(LOC_ENV) && $(DOCKER_COMPOSE) down && $(DOCKER_COMPOSE) up -d --wait
	$(LOC_ENV) && $(DOCKER_COMPOSE) exec training python $(SRC_DIR)/evaluation/evaluate.py \
		--model-path checkpoints/final_model \
		--eval-config $(CONFIG_DIR)/eval_config.yaml \
		--train-config $(CONFIG_DIR)/train_config.yaml \
		--output-dir evaluation_results


# ----------------------------------------------------------------------
# Docker: Layout Generation (no CARLA needed)
# ----------------------------------------------------------------------

docker-generate-layouts: ## Generate lot layout YAMLs + bird's-eye PNGs inside training container. Usage: make docker-generate-layouts [LAYOUT=trapezoid]
	$(DOCKER_COMPOSE) exec training bash -c \
		"mkdir -p configs/layouts outputs/layouts && \
		 python scripts/layouts/generate_layouts.py \
		   --output-dir configs/layouts \
		   --plot-dir outputs/layouts \
		   $(if $(filter command line,$(origin LAYOUT)),--layout $(LAYOUT),)"


# ----------------------------------------------------------------------
# Docker: Testing & Linting
# ----------------------------------------------------------------------

docker-test: ## Run full test suite inside container
	$(DOCKER_COMPOSE) exec training pytest $(TESTS_DIR) -v --tb=short

docker-test-unit: ## Run unit tests inside container (no CARLA/ROS 2 needed)
	$(DOCKER_COMPOSE) exec training pytest $(TESTS_DIR) -v --tb=short -m "not integration"

docker-test-integration: ## Run integration tests inside container (requires full stack)
	$(DOCKER_COMPOSE) exec training pytest $(TESTS_DIR) -v --tb=short -m "integration"

docker-verify: ## Run all checks inside container (tests + lint + typecheck)
	$(DOCKER_COMPOSE) exec training make verify

docker-lint: ## Run linters inside container
	$(DOCKER_COMPOSE) exec training make lint

docker-format: ## Format code inside container
	$(DOCKER_COMPOSE) exec training make format

docker-typecheck: ## Run mypy inside container
	$(DOCKER_COMPOSE) exec training make typecheck

# ----------------------------------------------------------------------
# Docker: Shells & Logs
# ----------------------------------------------------------------------

docker-shell: ## Interactive shell in training container
	$(DOCKER_COMPOSE) exec training /bin/bash

docker-shell-ros2: ## Interactive shell in ROS 2 container
	$(DOCKER_COMPOSE) exec ros2-bridge /bin/bash

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

# ----------------------------------------------------------------------
# Docker: Cleanup
# ----------------------------------------------------------------------

docker-clean: ## Stop containers and remove volumes
	$(DOCKER_COMPOSE) down -v
	docker volume prune -f

docker-clean-all: ## Remove all containers, images, and volumes
	$(DOCKER_COMPOSE) down -v --rmi all
	docker system prune -af

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

INSPECT_LAYOUT  ?= rectangle
INSPECT_EPISODES ?=
docker-inspect-dryrun: ## Full training pipeline with random actions in windowed CARLA. Spectator follows ego; EKF covariance printed every 50 steps. Usage: make docker-inspect-dryrun [INSPECT_LAYOUT=rectangle] [INSPECT_EPISODES=5]
	$(eval _DISPLAY := $(or $(DISPLAY),$(shell ls /tmp/.X11-unix/X* 2>/dev/null | head -1 | sed 's|/tmp/.X11-unix/X|:|'),$(error No X11 display found. Set DISPLAY manually: export DISPLAY=:0)))
	$(eval LAYOUT := $(INSPECT_LAYOUT))
	@echo "Using DISPLAY=$(_DISPLAY)  LAYOUT=$(LAYOUT)  EPISODES=$(INSPECT_EPISODES)"
	$(DOCKER_COMPOSE) down 2>/dev/null || true
	docker rm -f uncertainty-rl-carla-demo uncertainty-rl-ros2-inspect uncertainty-rl-training-inspect-dryrun 2>/dev/null || true
	docker network prune -f 2>/dev/null || true
	xhost +local:docker 2>/dev/null || true
	DISPLAY=$(_DISPLAY) LAYOUT=$(LAYOUT) EPISODES=$(INSPECT_EPISODES) \
		CARTOGRAPHER_MODE=loc \
		CARTOGRAPHER_MAP=/workspace/configs/maps/$(MAP_DIM)/$(LAYOUT).pbstream \
		SENSOR_SUITE=$(SENSOR_SUITE) \
		$(DOCKER_COMPOSE_INSPECT) --profile inspect-dryrun up \
		--force-recreate --detach \
		carla-server-demo ros2-bridge-inspect training-inspect-dryrun
	@echo "Containers started. Streaming training output (Ctrl+C to abort)..."
	@docker logs -f uncertainty-rl-training-inspect-dryrun 2>&1 | \
		grep -Ev '^(>>>|<<<|$$|This error state|with this new error|rcutils_reset_error|rcutils_set_error_state|error_handling\.c|serdata\.cpp|should be called after|.*serdata.*)' || true
	$(DOCKER_COMPOSE_INSPECT) --profile inspect-dryrun down 2>/dev/null || true
	xhost -local:docker 2>/dev/null || true

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

INSPECT_DURATION ?= 120
docker-inspect-zcheck: ## Spawn one of each actor type in a row and print their actual z coords (verifies shared ground plane). Usage: make docker-inspect-zcheck [INSPECT_LAYOUT=rectangle] [INSPECT_DURATION=120]
	$(eval _DISPLAY := $(or $(DISPLAY),$(shell ls /tmp/.X11-unix/X* 2>/dev/null | head -1 | sed 's|/tmp/.X11-unix/X|:|'),$(error No X11 display found. Set DISPLAY manually: export DISPLAY=:0)))
	@echo "Using DISPLAY=$(_DISPLAY)"
	docker rm -f uncertainty-rl-carla-demo uncertainty-rl-training-inspect-zcheck 2>/dev/null || true
	$(DOCKER_COMPOSE) down 2>/dev/null || true
	docker network prune -f 2>/dev/null || true
	xhost +local:docker 2>/dev/null || true
	DISPLAY=$(_DISPLAY) LAYOUT=$(INSPECT_LAYOUT) DURATION=$(INSPECT_DURATION) $(DOCKER_COMPOSE_INSPECT) --profile inspect-zcheck up --force-recreate --abort-on-container-exit carla-server-demo training-inspect-zcheck
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
	mkdir -p configs/layouts outputs/layouts
	$(PYTHON_VIS) scripts/layouts/generate_layouts.py \
		--output-dir configs/layouts \
		--plot-dir outputs/layouts \
		$(if $(filter command line,$(origin LAYOUT)),--layout $(LAYOUT),)


# ----------------------------------------------------------------------
# Training & Evaluation
# ----------------------------------------------------------------------

train-loc: ## Train PPO agent in pure localisation mode against a pre-built .pbstream map. Usage: make train-loc [LAYOUT=rectangle]
	CARTOGRAPHER_MODE=loc \
	CARTOGRAPHER_MAP=configs/maps/$(MAP_DIM)/$(LAYOUT).pbstream \
	$(PYTHON) $(SRC_DIR)/training/train_ppo.py \
		--config $(CONFIG_DIR)/train_config.yaml \
		--log-dir ./logs \
		--checkpoint-dir ./checkpoints

train-loc-short: ## Quick training run (10k steps) in pure localisation mode. Usage: make train-loc-short [LAYOUT=rectangle]
	CARTOGRAPHER_MODE=loc \
	CARTOGRAPHER_MAP=configs/maps/$(MAP_DIM)/$(LAYOUT).pbstream \
	$(PYTHON) $(SRC_DIR)/training/train_ppo.py \
		--config $(CONFIG_DIR)/train_config.yaml \
		--total-timesteps 10000 \
		--log-dir ./logs \
		--checkpoint-dir ./checkpoints

evaluate: ## Evaluate trained agent in pure localisation mode. Usage: make evaluate [LAYOUT=rectangle]
	CARTOGRAPHER_MODE=loc \
	CARTOGRAPHER_MAP=configs/maps/$(MAP_DIM)/$(LAYOUT).pbstream \
	$(PYTHON) $(SRC_DIR)/evaluation/evaluate.py \
		--model-path checkpoints/final_model \
		--config $(CONFIG_DIR)/eval_config.yaml \
		--output-dir ./evaluation_results

# ----------------------------------------------------------------------
# Visualisation (host-side, detachable from training) 
# IMPORTANT: Containers MUST be running to see live data !
# ----------------------------------------------------------------------

visualise: ## Open 2D bird's-eye live visualiser (tails outputs/vis_history.jsonl, detachable)
	PYTHONPATH=$(CURDIR) DISPLAY=$(or $(DISPLAY),:0) $(PYTHON_VIS) -m scripts.visualise

eval-visualise-2d: ## Load checkpoint + headless CARLA + 2D bird's-eye. Usage: make eval-visualise-2d [LAYOUT=rectangle] [CHECKPOINT=path]
	@echo "Demo drive 2D: layout=$(LAYOUT), checkpoint=$(or $(CHECKPOINT),checkpoints/final_model)"
	$(LOC_ENV) && $(DOCKER_COMPOSE) down && $(DOCKER_COMPOSE) up -d --wait
	$(LOC_ENV) && $(DOCKER_COMPOSE) exec -d training python $(SCRIPTS_DIR)/demo_drive.py \
		--checkpoint $(or $(CHECKPOINT),checkpoints/final_model) \
		--train-config $(CONFIG_DIR)/train_config.yaml
	@echo "Model driving in background. Open the 2D visualiser with: make visualise"

docker-eval-visualise-3d: ## Load checkpoint + CARLA 3D spectator view. Usage: make docker-eval-visualise-3d [CHECKPOINT=path]
	@echo "Demo drive 3D: checkpoint=$(or $(CHECKPOINT),checkpoints/final_model)"
	DISPLAY=$(or $(DISPLAY),:0) CHECKPOINT=$(or $(CHECKPOINT),checkpoints/final_model) \
		$(DOCKER_COMPOSE_INSPECT) --profile demo up --build --abort-on-container-exit

# ----------------------------------------------------------------------
# Testing
# ----------------------------------------------------------------------

test: ## Run full test suite (unit + integration)
	$(PYTEST) $(TESTS_DIR) -v --tb=short

test-unit: ## Run unit tests only (no GPU, no CARLA, no ROS 2)
	$(PYTEST) $(TESTS_DIR) -v --tb=short -m "not integration"

test-integration: ## Run integration tests (requires CARLA + ROS 2 + GPU)
	$(PYTEST) $(TESTS_DIR) -v --tb=short -m "integration"

verify: ## Run all CPU-only checks (tests + lint + typecheck + import sanity)
	$(PYTEST) $(TESTS_DIR) -v --tb=short -m "not integration"
	flake8 $(SRC_DIR) $(TESTS_DIR) $(SCRIPTS_DIR) --max-line-length 88 --extend-ignore E203,W503
	isort --check-only --diff $(SRC_DIR) $(TESTS_DIR) $(SCRIPTS_DIR)
	black --check $(SRC_DIR) $(TESTS_DIR) $(SCRIPTS_DIR)
	mypy $(SRC_DIR) --ignore-missing-imports
	$(PYTHON) -c "import uncertainty_rl; print('All checks passed.')"

# ----------------------------------------------------------------------
# Linting & Formatting
# ----------------------------------------------------------------------

lint: ## Run all linters (flake8 + isort + black)
	flake8 $(SRC_DIR) $(TESTS_DIR) $(SCRIPTS_DIR) --max-line-length 88 --extend-ignore E203,W503
	isort --check-only --diff $(SRC_DIR) $(TESTS_DIR) $(SCRIPTS_DIR)
	black --check $(SRC_DIR) $(TESTS_DIR) $(SCRIPTS_DIR)

format: ## Auto-format code with black + isort
	isort $(SRC_DIR) $(TESTS_DIR) $(SCRIPTS_DIR)
	black $(SRC_DIR) $(TESTS_DIR) $(SCRIPTS_DIR)

typecheck: ## Run mypy type checking
	mypy $(SRC_DIR) --ignore-missing-imports

# ----------------------------------------------------------------------
# Sanity Check
# ----------------------------------------------------------------------

sanity: ## Quick import check
	$(PYTHON) -c "import uncertainty_rl; print('Package imports OK')"

syntax-check: ## Check Python syntax with py_compile (no execution)
	$(PYTHON) -m py_compile uncertainty_rl/envs/carla_parking.py && echo "Syntax OK: carla_parking.py"

# ----------------------------------------------------------------------
# Cleanup
# ----------------------------------------------------------------------

clean: ## Remove build artefacts, caches, generated outputs, maps, and layouts (preserves .xodr)
	rm -rf __pycache__ .pytest_cache htmlcov .mypy_cache
	sudo rm -rf logs/ checkpoints/ evaluation_results/ experiments/ results/
	sudo rm -rf outputs/
	sudo rm -rf configs/maps/
	find configs/layouts/ -type f ! -name "*.xodr" -delete 2>/dev/null || true
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

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