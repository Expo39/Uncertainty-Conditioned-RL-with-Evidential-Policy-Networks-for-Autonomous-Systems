#!/usr/bin/env bash
# @file dryrun.sh
# @brief Start an inspect dry-run stack (training or eval) and stream output.
#
# Starts carla-server-demo and ros2-bridge-inspect detached, waits for them
# to be healthy, then runs the dry-run training container.
#
# The same script drives two dry-runs, selected by the caller via env vars:
#   - the curriculum dry-run (make docker-inspect-dryrun): profile
#     inspect-dryrun, service training-inspect-dryrun, env INSPECT_STAGE.
#   - the eval-scenario dry-run (make docker-inspect-eval-dryrun): profile
#     inspect-eval-dryrun, service training-inspect-eval-dryrun, env
#     INSPECT_SCENARIO.
# The Makefile sets INSPECT_PROFILE / INSPECT_SERVICE / INSPECT_CONTAINER so this
# script stays mode-agnostic; defaults keep the original curriculum behaviour.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DOCKER_COMPOSE_INSPECT="docker compose \
    -f ${REPO_ROOT}/docker-compose.yml \
    -f ${REPO_ROOT}/docker-compose.inspect.yml"

PROFILE="${INSPECT_PROFILE:-inspect-dryrun}"
SERVICE="${INSPECT_SERVICE:-training-inspect-dryrun}"
CONTAINER="${INSPECT_CONTAINER:-uncertainty-rl-training-inspect-dryrun}"

cleanup() {
    echo ""
    echo "Stopping inspect containers..."
    ${DOCKER_COMPOSE_INSPECT} --profile "${PROFILE}" down --timeout 3 2>/dev/null || true
    xhost -local:docker 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# Determine LAYOUT for CARLA server. The training container (lot_inspector.py)
# will randomise the actual floor plan per episode based on INSPECT_OOD (dryrun)
# or pin it from the eval condition's floor_plan (eval_dryrun). CARLA server just
# needs a dummy layout for the OpenDRIVE world.
LAYOUT="rectangle"

# Start CARLA + ROS 2 bridge detached and wait for health checks
echo "Starting carla-server-demo and ros2-bridge-inspect..."
DISPLAY="${DISPLAY}" LAYOUT="${LAYOUT}" \
    INSPECT_VIEW="${INSPECT_VIEW:-third_person}" \
    INSPECT_PAUSE="${INSPECT_PAUSE:-3.0}" \
    INSPECT_MANUAL="${INSPECT_MANUAL:-false}" \
    EPISODES="${EPISODES:-}" \
    INSPECT_OOD="${INSPECT_OOD:-false}" \
    ${DOCKER_COMPOSE_INSPECT} --profile "${PROFILE}" up \
    --force-recreate --detach --wait \
    carla-server-demo ros2-bridge-inspect

if [ "${INSPECT_MANUAL:-false}" = "true" ]; then
    echo "Starting dry-run container in foreground (keyboard control active)..."
    echo "  Arrow keys: Up=throttle  Down=brake  Left/Right=steer  Ctrl+C=stop"
    # Run the training container interactively so its stdin is this terminal.
    # --no-deps: carla + ros2 already up. --rm: clean up on exit.
    DISPLAY="${DISPLAY}" \
        INSPECT_VIEW="${INSPECT_VIEW:-third_person}" \
        INSPECT_PAUSE="${INSPECT_PAUSE:-3.0}" \
        INSPECT_MANUAL="${INSPECT_MANUAL:-false}" \
        EPISODES="${EPISODES:-}" \
        INSPECT_OOD="${INSPECT_OOD:-false}" \
        INSPECT_STAGE="${INSPECT_STAGE:-}" \
        INSPECT_SCENARIO="${INSPECT_SCENARIO:-}" \
        INSPECT_BASELINE="${INSPECT_BASELINE:-}" \
        ${DOCKER_COMPOSE_INSPECT} --profile "${PROFILE}" \
        run --rm -it --name "${CONTAINER}" \
        "${SERVICE}"
else
    echo "Starting dry-run container..."
    DISPLAY="${DISPLAY}" \
        INSPECT_VIEW="${INSPECT_VIEW:-third_person}" \
        INSPECT_PAUSE="${INSPECT_PAUSE:-3.0}" \
        INSPECT_MANUAL="${INSPECT_MANUAL:-false}" \
        EPISODES="${EPISODES:-}" \
        INSPECT_OOD="${INSPECT_OOD:-false}" \
        INSPECT_STAGE="${INSPECT_STAGE:-}" \
        INSPECT_SCENARIO="${INSPECT_SCENARIO:-}" \
        INSPECT_BASELINE="${INSPECT_BASELINE:-}" \
        ${DOCKER_COMPOSE_INSPECT} --profile "${PROFILE}" up \
        --force-recreate --detach \
        "${SERVICE}"
    echo "Streaming dry-run output (Ctrl+C to abort)..."
    docker logs -f "${CONTAINER}" 2>&1
fi
