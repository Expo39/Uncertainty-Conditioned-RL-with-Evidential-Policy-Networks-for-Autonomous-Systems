# Makefile - Uncertainty-Conditioned RL
# Development commands for training, evaluation, testing, and linting.

.PHONY: help install test test-unit test-integration verify
.PHONY: lint format typecheck clean syntax-check
.PHONY: backup-configs restore-configs
.PHONY: train train-short evaluate ros2 experiment-dry
.PHONY: generate-layouts visualise visualise-record
.PHONY: docker-build docker-build-prod docker-build-no-cache docker-up docker-down docker-restart docker-ps docker-top
.PHONY: docker-train docker-train-short docker-eval docker-experiment docker-experiment-dry
.PHONY: docker-test docker-test-unit docker-test-integration docker-verify docker-lint docker-format docker-typecheck
.PHONY: docker-shell docker-shell-ros2 docker-logs docker-logs-training docker-logs-carla docker-logs-ros2
.PHONY: docker-clean docker-clean-all docker-full-build docker-dev docker-demo docker-inspect docker-inspect-sensors
.PHONY: docker-generate-layouts

PYTHON := python3
PYTHON_VIS := .venv-vis/bin/python3
PYTEST := pytest
CONFIG_DIR := configs
SRC_DIR := uncertainty_rl
TESTS_DIR := tests
SCRIPTS_DIR := scripts
DOCKER_COMPOSE := docker compose

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

# ----------------------------------------------------------------------
# Layout Generation (no CARLA needed)
# ----------------------------------------------------------------------

generate-layouts: ## Generate lot layout YAMLs + bird's-eye PNGs (no CARLA needed). Usage: make generate-layouts [LAYOUT=trapezoid]
	mkdir -p configs/layouts outputs/layouts
	$(PYTHON_VIS) scripts/layouts/generate_layouts.py \
		--output-dir configs/layouts \
		--plot-dir outputs/layouts \
		$(if $(LAYOUT),--layout $(LAYOUT),)

docker-generate-layouts: ## Generate lot layout YAMLs + bird's-eye PNGs inside training container. Usage: make docker-generate-layouts [LAYOUT=trapezoid]
	$(DOCKER_COMPOSE) exec training bash -c \
		"mkdir -p configs/layouts outputs/layouts && \
		 python scripts/layouts/generate_layouts.py \
		   --output-dir configs/layouts \
		   --plot-dir outputs/layouts \
		   $(if $(LAYOUT),--layout $(LAYOUT),)"

# ----------------------------------------------------------------------
# Visualisation (host-side, detachable from training)
# ----------------------------------------------------------------------

visualise: ## Open 2D bird's-eye visualiser (polls outputs/vis_state.json, detachable)
	$(PYTHON_VIS) scripts/visualise_training.py

visualise-record: ## Open 2D visualiser + save MP4 on window close
	$(PYTHON_VIS) scripts/visualise_training.py --record




# ======================================================================
# DOCKER - commands that run inside containers
# ======================================================================

# ----------------------------------------------------------------------
# Docker: Lifecycle
# ----------------------------------------------------------------------

docker-build: ## Build all Docker images
	$(DOCKER_COMPOSE) build

docker-build-prod: ## Build training image without dev dependencies (lighter)
	$(DOCKER_COMPOSE) build --build-arg DEV_INSTALL=false training

docker-build-no-cache: ## Build images without cache (clean rebuild)
	$(DOCKER_COMPOSE) build --no-cache

docker-up: ## Start all containers
	$(DOCKER_COMPOSE) up -d

docker-down: ## Stop all containers
	$(DOCKER_COMPOSE) down

docker-restart: ## Restart all containers
	$(DOCKER_COMPOSE) restart

docker-ps: ## Show running containers
	$(DOCKER_COMPOSE) ps

docker-top: ## Show running processes in containers
	$(DOCKER_COMPOSE) top

# ----------------------------------------------------------------------
# Docker: Training & Evaluation
# ----------------------------------------------------------------------

docker-train: ## Run training inside container
	$(DOCKER_COMPOSE) exec training python $(SRC_DIR)/training/train_ppo.py \
		--config $(CONFIG_DIR)/train_config.yaml \
		--log-dir logs \
		--checkpoint-dir checkpoints

docker-train-short: ## Quick training (10k steps) inside container
	$(DOCKER_COMPOSE) exec training python $(SRC_DIR)/training/train_ppo.py \
		--config $(CONFIG_DIR)/train_config.yaml \
		--total-timesteps 10000 \
		--log-dir logs \
		--checkpoint-dir checkpoints

docker-eval: ## Run evaluation inside container
	$(DOCKER_COMPOSE) exec training python $(SRC_DIR)/evaluation/evaluate.py \
		--model-path checkpoints/final_model \
		--config $(CONFIG_DIR)/eval_config.yaml \
		--output-dir evaluation_results

docker-experiment: ## Run full ablation study inside container (4 baselines x 10 seeds)
	$(DOCKER_COMPOSE) exec training python scripts/run_experiment.py \
		--base-config $(CONFIG_DIR)/train_config.yaml \
		--configs $(CONFIG_DIR)/baselines/vanilla_ppo.yaml \
		          $(CONFIG_DIR)/baselines/input_uncertainty.yaml \
		          $(CONFIG_DIR)/baselines/output_uncertainty.yaml \
		          $(CONFIG_DIR)/baselines/full_method.yaml

docker-experiment-dry: ## Dry-run ablation study inside container (plan without training)
	$(DOCKER_COMPOSE) exec training python scripts/run_experiment.py \
		--base-config $(CONFIG_DIR)/train_config.yaml \
		--configs $(CONFIG_DIR)/baselines/vanilla_ppo.yaml \
		          $(CONFIG_DIR)/baselines/input_uncertainty.yaml \
		          $(CONFIG_DIR)/baselines/output_uncertainty.yaml \
		          $(CONFIG_DIR)/baselines/full_method.yaml \
		--dry-run

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
# Docker: Combined Workflows
# ----------------------------------------------------------------------

docker-full-build: ## Build and start full stack
	$(DOCKER_COMPOSE) build && $(DOCKER_COMPOSE) up -d
	@echo "Waiting for services to be healthy..."
	sleep 30
	$(DOCKER_COMPOSE) ps

docker-dev: ## Start stack + open training shell (development mode)
	$(DOCKER_COMPOSE) up -d
	@echo "Waiting for services to be healthy..."
	sleep 15
	$(DOCKER_COMPOSE) exec training /bin/bash

MODEL ?= checkpoints/final_model
docker-demo: ## Windowed CARLA demo with checkpoint (requires X11). Usage: make docker-demo MODEL=<path>
	$(eval _DISPLAY := $(or $(DISPLAY),$(shell ls /tmp/.X11-unix/X* 2>/dev/null | head -1 | sed 's|/tmp/.X11-unix/X|:|'),$(error No X11 display found. Set DISPLAY manually: export DISPLAY=:0)))
	@echo "Using DISPLAY=$(_DISPLAY)"
	xhost +local:docker 2>/dev/null || true
	DISPLAY=$(_DISPLAY) MODEL=$(MODEL) $(DOCKER_COMPOSE) --profile demo up --abort-on-container-exit
	xhost -local:docker 2>/dev/null || true

INSPECT_LAYOUT ?= trapezoid
docker-inspect: ## Spawn a layout in windowed CARLA for visual inspection. Usage: make docker-inspect [INSPECT_LAYOUT=trapezoid]
	$(eval _DISPLAY := $(or $(DISPLAY),$(shell ls /tmp/.X11-unix/X* 2>/dev/null | head -1 | sed 's|/tmp/.X11-unix/X|:|'),$(error No X11 display found. Set DISPLAY manually: export DISPLAY=:0)))
	@echo "Using DISPLAY=$(_DISPLAY)"
	docker rm -f uncertainty-rl-carla-demo uncertainty-rl-training-inspect 2>/dev/null || true
	$(DOCKER_COMPOSE) down 2>/dev/null || true
	docker network prune -f 2>/dev/null || true
	xhost +local:docker 2>/dev/null || true
	DISPLAY=$(_DISPLAY) LAYOUT=$(INSPECT_LAYOUT) $(DOCKER_COMPOSE) --profile inspect up --force-recreate --abort-on-container-exit carla-server-demo training-inspect
	xhost -local:docker 2>/dev/null || true

INSPECT_SUITE   ?= suite_a
INSPECT_VIEW    ?= birds_eye
docker-inspect-sensors: ## Visualise sensor FOV on the parking lot layout in windowed CARLA. Usage: make docker-inspect-sensors [INSPECT_SUITE=suite_a|suite_b|suite_c] [INSPECT_LAYOUT=rectangle|trapezoid|irregular_a] [INSPECT_VIEW=birds_eye|side]
	$(eval _DISPLAY := $(or $(DISPLAY),$(shell ls /tmp/.X11-unix/X* 2>/dev/null | head -1 | sed 's|/tmp/.X11-unix/X|:|'),$(error No X11 display found. Set DISPLAY manually: export DISPLAY=:0)))
	@echo "Using DISPLAY=$(_DISPLAY)"
	docker rm -f uncertainty-rl-carla-demo uncertainty-rl-training-inspect-sensors 2>/dev/null || true
	$(DOCKER_COMPOSE) down 2>/dev/null || true
	docker network prune -f 2>/dev/null || true
	xhost +local:docker 2>/dev/null || true
	DISPLAY=$(_DISPLAY) SUITE=$(INSPECT_SUITE) LAYOUT=$(INSPECT_LAYOUT) VIEW=$(INSPECT_VIEW) $(DOCKER_COMPOSE) --profile inspect-sensors up --force-recreate --abort-on-container-exit carla-server-demo training-inspect-sensors
	xhost -local:docker 2>/dev/null || true






# ======================================================================
# LOCAL - commands that run on the host machine
# ======================================================================

# ----------------------------------------------------------------------
# Training & Evaluation
# ----------------------------------------------------------------------

train: ## Train PPO agent (requires CARLA running)
	$(PYTHON) $(SRC_DIR)/training/train_ppo.py \
		--config $(CONFIG_DIR)/train_config.yaml \
		--log-dir ./logs \
		--checkpoint-dir ./checkpoints

train-short: ## Quick training run (10k steps) for smoke testing
	$(PYTHON) $(SRC_DIR)/training/train_ppo.py \
		--config $(CONFIG_DIR)/train_config.yaml \
		--total-timesteps 10000 \
		--log-dir ./logs \
		--checkpoint-dir ./checkpoints

evaluate: ## Evaluate trained agent across uncertainty levels
	$(PYTHON) $(SRC_DIR)/evaluation/evaluate.py \
		--model-path checkpoints/final_model \
		--config $(CONFIG_DIR)/eval_config.yaml \
		--output-dir ./evaluation_results

experiment-dry: ## Dry-run ablation study locally (plan without training, no Docker needed)
	$(PYTHON) scripts/run_experiment.py \
		--base-config $(CONFIG_DIR)/train_config.yaml \
		--configs $(CONFIG_DIR)/baselines/vanilla_ppo.yaml \
		          $(CONFIG_DIR)/baselines/input_uncertainty.yaml \
		          $(CONFIG_DIR)/baselines/output_uncertainty.yaml \
		          $(CONFIG_DIR)/baselines/full_method.yaml \
		--dry-run

ros2: ## Launch covariance extractor node
	$(PYTHON) $(SRC_DIR)/ros2/covariance_extractor.py

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

clean: ## Remove build artefacts, caches, and generated outputs
	rm -rf __pycache__ .pytest_cache htmlcov .mypy_cache
	rm -rf logs/ checkpoints/ evaluation_results/ experiments/ results/
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