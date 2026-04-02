#!/usr/bin/env bash
# @file stack_clean.sh
# @brief Stop all containers for a given stack, then remove volumes/images.
#
# Usage:
#   bash scripts/cleanup/stack_clean.sh [STACK] [--rmi]
#
#   STACK : all (default) | training | inspect
#   --rmi : also remove images (for docker-clean-all behaviour)
#
# Examples:
#   bash scripts/cleanup/stack_clean.sh
#   bash scripts/cleanup/stack_clean.sh training
#   bash scripts/cleanup/stack_clean.sh inspect --rmi

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

STACK="${1:-all}"
RMI_FLAG=""
if [[ "${2:-}" == "--rmi" ]]; then
    RMI_FLAG="--rmi all"
fi

COMPOSE_MAIN="docker compose -f $REPO_ROOT/docker-compose.yml"
COMPOSE_INSPECT="DISPLAY=${DISPLAY:-} EPISODES= docker compose -f $REPO_ROOT/docker-compose.yml -f $REPO_ROOT/docker-compose.inspect.yml"
INSPECT_PROFILES="--profile inspect --profile inspect-dryrun --profile inspect-sensors --profile inspect-live"

# Force-stop any containers not managed by compose (e.g. orphaned workers).
stop_workers() {
    docker ps -q --filter "name=uncertainty-rl-carla-" --filter "name=uncertainty-rl-ros2-" | \
        xargs -r docker stop 2>/dev/null || true
    docker ps -aq --filter "name=uncertainty-rl-carla-" --filter "name=uncertainty-rl-ros2-" | \
        xargs -r docker rm -f 2>/dev/null || true
}

stop_inspect() {
    docker ps -aq --filter "name=uncertainty-rl-carla-demo" \
                  --filter "name=uncertainty-rl-ros2-inspect" \
                  --filter "name=uncertainty-rl-training-inspect" | \
        xargs -r docker rm -f 2>/dev/null || true
}

case "$STACK" in
    inspect)
        stop_inspect
        $COMPOSE_INSPECT $INSPECT_PROFILES down -v $RMI_FLAG 2>/dev/null || true
        ;;
    training)
        stop_workers
        $COMPOSE_MAIN down -v $RMI_FLAG 2>/dev/null || true
        ;;
    all|*)
        stop_workers
        stop_inspect
        $COMPOSE_MAIN down -v $RMI_FLAG 2>/dev/null || true
        $COMPOSE_INSPECT $INSPECT_PROFILES down -v $RMI_FLAG 2>/dev/null || true
        ;;
esac

docker volume prune -f

if [[ -n "$RMI_FLAG" ]]; then
    docker system prune -af
fi
