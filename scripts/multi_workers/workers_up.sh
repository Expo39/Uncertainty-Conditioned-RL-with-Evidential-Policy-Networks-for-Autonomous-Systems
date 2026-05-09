#!/usr/bin/env bash
# @file workers_up.sh
# @brief Bring up N env workers (CARLA server + ros2-bridge pairs).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# N: from argument, or read parallel_workers from train_config.yaml, default 1.
if [ -n "${1:-}" ]; then
    N=$1
else
    CONFIG="$REPO_ROOT/configs/train_config.yaml"
    N=$(grep '^parallel_workers:' "$CONFIG" 2>/dev/null | awk '{print $2}')
    N=${N:-1}
fi

COMPOSE_FILE="${2:-docker-compose.env_workers.yml}"

mkdir -p "$REPO_ROOT/outputs" "$REPO_ROOT/logs/ros2"

for r in $(seq 0 $((N - 1))); do
    port=$((2000 + r * 1000))
    port_end=$((2002 + r * 1000))
    domain=$((42 + r))
    if [ "$r" -eq 0 ]; then
        ekf=/workspace/outputs/ekf_state.json
    else
        ekf=/workspace/outputs/ekf_state_${r}.json
    fi

    WORKER_RANK=$r CARLA_PORT=$port CARLA_PORT_END=$port_end \
        ROS_DOMAIN_ID=$domain EKF_STATE_FILE=$ekf \
        docker compose -f "$REPO_ROOT/$COMPOSE_FILE" up -d --wait
done
