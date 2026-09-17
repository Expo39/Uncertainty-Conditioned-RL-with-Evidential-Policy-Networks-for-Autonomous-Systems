#!/usr/bin/env bash
# @file ensure_stack.sh
# @brief Start the core training stack if it is not already running.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

if ! docker compose -f "$REPO_ROOT/docker-compose.yml" ps --status running training 2>/dev/null | grep -q training; then
    echo "Core stack not running - starting (this may take up to 90s)..."
    # Base stack FIRST: it owns uncertainty-rl-network and creates it with the
    # compose ownership label. workers_up.sh treats that network as external, so
    # it must already exist and be labelled, or the base stack later refuses to
    # adopt a hand-made one. This ordering self-heals after a teardown.
    docker compose -f "$REPO_ROOT/docker-compose.yml" up -d --wait
    bash "$SCRIPT_DIR/workers_up.sh" 1
fi
