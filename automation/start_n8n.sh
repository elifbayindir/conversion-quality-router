#!/usr/bin/env bash
# Start a project-local n8n instance via npx, pinned to an exact version.
#
# No global npm install is performed or required -- npx downloads/caches the
# pinned version on first run. All n8n runtime state (user account,
# credentials DB, execution DB, cache) is isolated to automation/.n8n-runtime/
# (git-ignored; see decision D11 in .project-control/DECISIONS.md), never the
# default ~/.n8n.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

N8N_VERSION="2.41.3"
export N8N_USER_FOLDER="$PROJECT_ROOT/automation/.n8n-runtime"
export N8N_PORT="${N8N_PORT:-5678}"
# Bind to loopback only -- never 0.0.0.0/*/::: -- this local PoC instance
# must never be reachable from outside this machine.
export N8N_LISTEN_ADDRESS="127.0.0.1"
export N8N_DIAGNOSTICS_ENABLED="false"
export N8N_VERSION_NOTIFICATIONS_ENABLED="false"
export N8N_TEMPLATES_ENABLED="false"
# This local PoC instance is never exposed beyond localhost. Workflow nodes
# reference N8N_NOTIFY_WEBHOOK_URL and FASTAPI_BASE_URL via {{$env.*}}
# expressions instead of hardcoding them, so this must be off for those
# expressions to resolve. Do not set this on a shared/exposed n8n instance.
export N8N_BLOCK_ENV_ACCESS_IN_NODE="false"
export FASTAPI_BASE_URL="${FASTAPI_BASE_URL:-http://127.0.0.1:8000}"

# Load N8N_NOTIFY_WEBHOOK_URL (and anything else) from the project-root
# .env without ever echoing it. Never commit .env. A variable already
# present in the calling environment (e.g. N8N_NOTIFY_WEBHOOK_URL=http://
# 127.0.0.1:8090/ for mock-only testing) always wins over .env -- this must
# NOT unconditionally source-and-override, or repeated test runs would keep
# hitting the real webhook.
if [ -f "$PROJECT_ROOT/.env" ]; then
  while IFS='=' read -r key value; do
    case "$key" in ''|'#'*) continue ;; esac
    if [ -z "${!key:-}" ]; then
      export "$key=$value"
    fi
  done < "$PROJECT_ROOT/.env"
fi

mkdir -p "$N8N_USER_FOLDER"

echo "Starting n8n@$N8N_VERSION (local, project-scoped runtime at $N8N_USER_FOLDER)..."
exec npx --yes "n8n@$N8N_VERSION" start
