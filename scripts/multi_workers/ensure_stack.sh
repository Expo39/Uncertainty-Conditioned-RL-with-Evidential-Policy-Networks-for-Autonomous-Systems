#!/usr/bin/env bash
# @file ensure_stack.sh
# @brief Start the core training stack if it is not already running.
#
# Only starts containers if the training service is not already up.
# Never touches the inspect stack (docker-compose.inspect.yml).
#
# Usage:
#   bash scripts/multi_workers/ensure_stack.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

if ! docker compose -f "$REPO_ROOT/docker-compose.yml" ps --status running training 2>/dev/null | grep -q training; then
    echo "Core stack not running - starting (this may take up to 90s)..."
    bash "$SCRIPT_DIR/workers_up.sh" 1
    docker compose -f "$REPO_ROOT/docker-compose.yml" up -d --wait
fi
