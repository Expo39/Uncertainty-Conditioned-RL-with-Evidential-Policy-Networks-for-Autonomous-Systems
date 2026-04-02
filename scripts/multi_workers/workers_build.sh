#!/usr/bin/env bash
# @file workers_build.sh
# @brief Build the env-workers Docker image.
#
# All N workers share the same image so it only needs to be built once.
# Dummy rank-0 values are passed purely so Compose can resolve variable
# references in docker-compose.env_workers.yml without erroring.
#
# Usage:
#   bash scripts/multi_workers/workers_build.sh [compose_file] [--no-cache]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

COMPOSE_FILE="${1:-docker-compose.env_workers.yml}"
shift || true

WORKER_RANK=0 CARLA_PORT=2000 CARLA_PORT_END=2002 \
    ROS_DOMAIN_ID=42 EKF_STATE_FILE=/workspace/outputs/ekf_state.json \
    docker compose -f "$REPO_ROOT/$COMPOSE_FILE" build "$@"
