# Makefile - Uncertainty-Conditioned RL
# Development commands for training, evaluation, testing, and linting.

.PHONY: help install test
.PHONY: lint format typecheck clean
.PHONY: train train-short evaluate ros2
.PHONY: docker-build docker-build-prod docker-build-no-cache docker-up docker-down docker-restart docker-ps docker-top
.PHONY: docker-train docker-train-short docker-eval
.PHONY: docker-test docker-test-fast docker-lint docker-format
.PHONY: docker-shell docker-shell-ros2 docker-logs docker-logs-training docker-logs-carla docker-logs-ros2
.PHONY: docker-clean docker-clean-all docker-full-build docker-dev

PYTHON := python
PYTEST := pytest
CONFIG_DIR := configs
SRC_DIR := uncertainty_rl
TESTS_DIR := tests
DOCKER_COMPOSE := docker-compose

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

# ----------------------------------------------------------------------
# Docker: Testing & Linting
# ----------------------------------------------------------------------

docker-test: ## Run tests inside container
	$(DOCKER_COMPOSE) exec training pytest $(TESTS_DIR) -v

docker-lint: ## Run linters inside container
	$(DOCKER_COMPOSE) exec training make lint

docker-format: ## Format code inside container
	$(DOCKER_COMPOSE) exec training make format

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

ros2: ## Launch covariance extractor node
	$(PYTHON) $(SRC_DIR)/ros2/covariance_extractor.py

# ----------------------------------------------------------------------
# Testing
# ----------------------------------------------------------------------

test: ## Run full test suite
	$(PYTEST) $(TESTS_DIR) -v --tb=short

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
# Santiy Check
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