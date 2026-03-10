# Makefile - Uncertainty-Conditioned RL
# Development commands for training, evaluation, testing, and linting.

.PHONY: help install test test-unit test-integration verify
.PHONY: lint format typecheck clean
.PHONY: backup-configs restore-configs
.PHONY: train train-short evaluate ros2 experiment-dry
.PHONY: generate-layouts visualise visualise-record
.PHONY: docker-build docker-build-prod docker-build-no-cache docker-up docker-down docker-restart docker-ps docker-top display-info docker-carla-windowed
.PHONY: docker-train docker-train-short docker-eval docker-experiment docker-experiment-dry
.PHONY: docker-test docker-test-unit docker-test-integration docker-verify docker-lint docker-format docker-typecheck
.PHONY: docker-shell docker-shell-ros2 docker-logs docker-logs-training docker-logs-carla docker-logs-ros2
.PHONY: docker-clean docker-clean-all docker-full-build docker-dev docker-demo
.PHONY: docker-explore-map docker-explore-map-mark

PYTHON := python3
PYTEST := pytest
CONFIG_DIR := configs
SRC_DIR := uncertainty_rl
TESTS_DIR := tests
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

generate-layouts: ## Generate lot layout YAMLs + bird's-eye PNGs (no CARLA needed)
	mkdir -p configs/layouts outputs/layouts
	$(PYTHON) scripts/generate_lot_layout.py \
		--output-dir configs/layouts \
		--plot-dir outputs/layouts

# ----------------------------------------------------------------------
# Visualisation (host-side, detachable from training)
# ----------------------------------------------------------------------

visualise: ## Open 2D bird's-eye visualiser (polls outputs/vis_state.json, detachable)
	$(PYTHON) scripts/visualise_training.py

visualise-record: ## Open 2D visualiser + save MP4 on window close
	$(PYTHON) scripts/visualise_training.py --record




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

display-info: ## Show active X11 displays and current DISPLAY variable (use before docker-carla-windowed)
	@echo "=== Current DISPLAY variable ==="
	@echo "DISPLAY=${DISPLAY}"
	@echo ""
	@echo "=== X11 sockets in /tmp/.X11-unix ==="
	@ls /tmp/.X11-unix/ 2>/dev/null || echo "(none found)"
	@echo ""
	@echo "=== Active Xorg processes ==="
	@ps aux | grep -E '[X]org' | awk '{print $$11, $$12, $$13}' || echo "(none)"
	@echo ""
	@echo "=== Hint: pass the display to docker-carla-windowed ==="
	@echo "  make docker-carla-windowed CARLA_DISPLAY=:1"

# Usage: make docker-carla-windowed CARLA_DISPLAY=:1
# Run make display-info first to find the correct display value.
CARLA_DISPLAY ?= :1
EXPLORE_TOWN  ?= Town10HD
EXPLORE_PAUSE ?= 3.0
EXPLORE_START ?= 0
EXPLORE_STEP  ?= 1
docker-explore-map: ## Cycle through map spawn points, printing coordinates. Requires full stack running (make docker-up).
	$(DOCKER_COMPOSE) exec training python scripts/explore_map.py \
		--host carla-server \
		--port 2000 \
		--town $(EXPLORE_TOWN) \
		--pause $(EXPLORE_PAUSE) \
		--start $(EXPLORE_START) \
		--step $(EXPLORE_STEP)

docker-carla-windowed: ## Start CARLA in windowed mode. Set CARLA_DISPLAY (default :1).
	@echo "Starting CARLA windowed on DISPLAY=$(CARLA_DISPLAY)"
	DISPLAY=$(CARLA_DISPLAY) xhost +local:docker 2>/dev/null || true
	$(DOCKER_COMPOSE) run --rm \
		-e DISPLAY=$(CARLA_DISPLAY) \
		carla-server \
		/bin/bash CarlaUE4.sh -windowed -ResX=1280 -ResY=720 -world-port=2000 -quality-level=Low

# Usage: make docker-demo MODEL=checkpoints/final_model
# Starts windowed CARLA on separate ports (2100-2102) and runs evaluation with the given model.
# Requires X11 on host. Does not affect the training stack on ports 2000-2002.
MODEL ?= checkpoints/final_model
docker-demo: ## Windowed CARLA demo with checkpoint (requires X11). Usage: make docker-demo MODEL=<path>
	@echo "Starting demo stack (windowed CARLA on ports 2100-2102)..."
	xhost +local:docker 2>/dev/null || true
	DISPLAY=$(DISPLAY) MODEL=$(MODEL) $(DOCKER_COMPOSE) --profile demo up --abort-on-container-exit
	xhost -local:docker 2>/dev/null || true

EXPLORE_MARK_TOWN ?= Town05_Opt
docker-explore-map-mark: ## Interactive spectator + ENTER to record lot origins. Usage: make docker-explore-map-mark [EXPLORE_MARK_TOWN=Town05_Opt]
	$(DOCKER_COMPOSE) exec training python scripts/explore_map.py \
		--host carla-server \
		--port 2000 \
		--mark \
		--town $(EXPLORE_MARK_TOWN)




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
	flake8 $(SRC_DIR) $(TESTS_DIR) --max-line-length 88 --extend-ignore E203,W503
	isort --check-only --diff $(SRC_DIR) $(TESTS_DIR)
	black --check $(SRC_DIR) $(TESTS_DIR)
	mypy $(SRC_DIR) --ignore-missing-imports
	$(PYTHON) -c "import uncertainty_rl; print('All checks passed.')"

# ----------------------------------------------------------------------
# Linting & Formatting
# ----------------------------------------------------------------------

lint: ## Run all linters (flake8 + isort + black)
	flake8 $(SRC_DIR) $(TESTS_DIR) --max-line-length 88 --extend-ignore E203,W503
	isort --check-only --diff $(SRC_DIR) $(TESTS_DIR)
	black --check $(SRC_DIR) $(TESTS_DIR)

format: ## Auto-format code with black + isort
	isort $(SRC_DIR) $(TESTS_DIR)
	black $(SRC_DIR) $(TESTS_DIR)

typecheck: ## Run mypy type checking
	mypy $(SRC_DIR) --ignore-missing-imports

# ----------------------------------------------------------------------
# Sanity Check
# ----------------------------------------------------------------------

sanity: ## Quick import check
	$(PYTHON) -c "import uncertainty_rl; print('Package imports OK')"

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

backup-configs: ## Pack all CLAUDE.md, TODO.md, documentation/, and .github/ into project_configs.tar.gz
	@find . -name "CLAUDE.md" -not -path "./.venv/*" > /tmp/_backup_files.txt
	@echo "TODO.md" >> /tmp/_backup_files.txt
	@find ./documentation -type f >> /tmp/_backup_files.txt 2>/dev/null || true
	tar -czf project_configs.tar.gz -T /tmp/_backup_files.txt
	@rm -f /tmp/_backup_files.txt
	@echo "Backed up to project_configs.tar.gz ($$(du -h project_configs.tar.gz | cut -f1))"

restore-configs: ## Restore CLAUDE.md, TODO.md, documentation/, and .github/ from project_configs.tar.gz
	tar -xzf project_configs.tar.gz
	@echo "Restored configs from project_configs.tar.gz"