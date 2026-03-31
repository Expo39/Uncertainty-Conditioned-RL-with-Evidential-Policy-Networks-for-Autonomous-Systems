#!/usr/bin/env bash
# @file dryrun.sh
# @brief Stream logs from the dryrun inspect container and tear down on exit.
#
# Called by `make docker-inspect-dryrun` after the inspect containers are
# started in detached mode. Follows the training-inspect-dryrun container log
# and cleans up all inspect containers on Ctrl+C or normal exit.
#
# Usage:
#   scripts/inspect/dryrun.sh

set -euo pipefail

DOCKER_COMPOSE_INSPECT="docker compose -f docker-compose.yml -f docker-compose.inspect.yml"

cleanup() {
    echo ""
    echo "Stopping inspect containers..."
    ${DOCKER_COMPOSE_INSPECT} --profile inspect-dryrun down 2>/dev/null || true
    xhost -local:docker 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "Containers started. Streaming training output (Ctrl+C to abort)..."
docker logs -f uncertainty-rl-training-inspect-dryrun 2>&1
