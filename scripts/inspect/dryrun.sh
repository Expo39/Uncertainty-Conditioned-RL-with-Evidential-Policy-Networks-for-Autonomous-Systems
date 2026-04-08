#!/usr/bin/env bash
# @file dryrun.sh
# @brief Start the dryrun inspect stack and stream output.
#
# Starts carla-server-demo and ros2-bridge-inspect detached, waits for them
# to be healthy, then runs the training container.
#
# Normal mode:  training container starts detached, logs are followed.
# Manual mode (INSPECT_MANUAL=true): training container starts in the
#   foreground with -it so the TTY keyboard controller inside the container
#   receives arrow key presses from this terminal directly.
#
# All environment variables are passed in by the Makefile docker-inspect-dryrun
# target.  Do not call this script directly.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DOCKER_COMPOSE_INSPECT="docker compose \
    -f ${REPO_ROOT}/docker-compose.yml \
    -f ${REPO_ROOT}/docker-compose.inspect.yml"
CONTAINER="uncertainty-rl-training-inspect-dryrun"

cleanup() {
    echo ""
    echo "Stopping inspect containers..."
    ${DOCKER_COMPOSE_INSPECT} --profile inspect-dryrun down 2>/dev/null || true
    xhost -local:docker 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# Start CARLA + ROS 2 bridge detached and wait for health checks
echo "Starting carla-server-demo and ros2-bridge-inspect..."
DISPLAY="${DISPLAY}" LAYOUT="${LAYOUT:-rectangle}" \
    INSPECT_VIEW="${INSPECT_VIEW:-third_person}" \
    INSPECT_PAUSE="${INSPECT_PAUSE:-3.0}" \
    INSPECT_MANUAL="${INSPECT_MANUAL:-false}" \
    EPISODES="${EPISODES:-}" \
    ${DOCKER_COMPOSE_INSPECT} --profile inspect-dryrun up \
    --force-recreate --detach --wait \
    carla-server-demo ros2-bridge-inspect

if [ "${INSPECT_MANUAL:-false}" = "true" ]; then
    echo "Starting training container in foreground (keyboard control active)..."
    echo "  Arrow keys: Up=throttle  Down=brake  Left/Right=steer  Ctrl+C=stop"
    # Run the training container interactively so its stdin is this terminal.
    # --no-deps: carla + ros2 already up. --rm: clean up on exit.
    DISPLAY="${DISPLAY}" LAYOUT="${LAYOUT:-rectangle}" \
        INSPECT_VIEW="${INSPECT_VIEW:-third_person}" \
        INSPECT_PAUSE="${INSPECT_PAUSE:-3.0}" \
        INSPECT_MANUAL="${INSPECT_MANUAL:-false}" \
        EPISODES="${EPISODES:-}" \
        ${DOCKER_COMPOSE_INSPECT} --profile inspect-dryrun \
        run --rm -it --name "${CONTAINER}" \
        training-inspect-dryrun
else
    echo "Starting training container..."
    DISPLAY="${DISPLAY}" LAYOUT="${LAYOUT:-rectangle}" \
        INSPECT_VIEW="${INSPECT_VIEW:-third_person}" \
        INSPECT_PAUSE="${INSPECT_PAUSE:-3.0}" \
        INSPECT_MANUAL="${INSPECT_MANUAL:-false}" \
        EPISODES="${EPISODES:-}" \
        ${DOCKER_COMPOSE_INSPECT} --profile inspect-dryrun up \
        --force-recreate --detach \
        training-inspect-dryrun
    echo "Streaming training output (Ctrl+C to abort)..."
    docker logs -f "${CONTAINER}" 2>&1
fi
