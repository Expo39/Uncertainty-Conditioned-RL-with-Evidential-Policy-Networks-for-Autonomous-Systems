#!/usr/bin/env bash
# @file eval_visualise_2d.sh
# @brief Run a checkpoint demo drive and watch it in the host 2D viewer.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

# Run parameters, supplied by the make recipe (defaults for direct use).
DEMO_CHECKPOINT="${DEMO_CHECKPOINT:?set DEMO_CHECKPOINT to the model path}"
DEMO_BASELINE_YAML="${DEMO_BASELINE_YAML:-}"  # empty -> demo_drive.py default
DEMO_STAGE="${DEMO_STAGE:-}"                  # empty -> demo_drive.py default
DEMO_REALTIME="${DEMO_REALTIME:-true}"        # false -> step flat out
DEMO_VIS_FILE="${DEMO_VIS_FILE:-outputs/vis_history.jsonl}"
DISPLAY="${DISPLAY:?no display attached}"

# UID/GID keep the demo container from falling back to root and leaving
# root-owned files in outputs/. UID is bash-readonly (already the invoking
# user); it only needs the export attribute so docker compose can
# interpolate it into the demo service's user: field.
export UID
export GID="$(id -g)"

demo_args=(
    --checkpoint "${DEMO_CHECKPOINT}"
    --env-config configs/deployment/sim/env_config.yaml
    --train-config configs/train_config.yaml
)
if [ -n "${DEMO_BASELINE_YAML}" ]; then
    demo_args+=(--baseline "${DEMO_BASELINE_YAML}")
fi
if [ -n "${DEMO_STAGE}" ]; then
    demo_args+=(--stage "${DEMO_STAGE}")
fi
if [ "${DEMO_REALTIME}" = "false" ]; then
    demo_args+=(--no-realtime)
fi

demo_cid=""
logs_pid=""

cleanup() {
    echo ""
    echo "Stopping demo container..."
    # Graceful stop first so the demo flushes bay_successes.csv; the --rm
    # container removes itself on exit, so rm -f is only the fallback.
    if [ -n "${demo_cid}" ]; then
        docker stop -t 15 "${demo_cid}" >/dev/null 2>&1 || true
    fi
    if [ -n "${logs_pid}" ]; then
        kill "${logs_pid}" 2>/dev/null || true
    fi
    if [ -n "${demo_cid}" ]; then
        docker rm -f "${demo_cid}" >/dev/null 2>&1 || true
    fi
}
trap cleanup EXIT INT TERM

echo "Starting demo container..."
demo_cid="$(docker compose --profile demo run --rm -d demo \
    python scripts/visualise/demo_drive.py "${demo_args[@]}" | tail -n1)"
echo "Demo container: ${demo_cid}"

# Stream demo logs in the background so failures (checkpoint load errors,
# CARLA connection issues, etc.) are visible alongside the viewer.
docker logs -f "${demo_cid}" 2>&1 | sed 's/^/[demo] /' &
logs_pid=$!

PYTHONPATH="${REPO_ROOT}" DISPLAY="${DISPLAY}" \
    "${REPO_ROOT}/.venv/bin/python3" scripts/visualise/visualiser.py \
    --history-file "${DEMO_VIS_FILE}"
