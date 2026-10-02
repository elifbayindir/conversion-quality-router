#!/usr/bin/env bash
# T803: fail-closed local environment for demo recording.
#
# Starts exactly three local processes -- a mock notification receiver, the
# real FastAPI app (real model/inference layer) with a deterministic
# non-LLM fake provider standing in for Anthropic, and the real, pinned
# n8n 2.41.3 instance (its already-set-up owner account and already-
# imported/active workflow -- this script never touches owner setup) -- and
# verifies all three are healthy before returning.
#
# SAFETY: never reads a real Slack/Discord webhook from .env for this run.
# N8N_NOTIFY_WEBHOOK_URL is forced to the local mock receiver *before*
# automation/start_n8n.sh's own precedence-safe .env loader ever runs, so a
# real value in .env can never silently win. ANTHROPIC_API_KEY is unset so
# any accidental real-provider code path fails closed. Makes ZERO real
# Slack/Discord or Anthropic calls.
#
# PIDs and logs are written to logs/demo-runtime/ (git-ignored). Stop with
# scripts/stop_demo_environment.sh -- it only stops what this script
# started, by PID, never a broad pkill.
set -uo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON="${PYTHON:-$PROJECT_ROOT/venv/bin/python}"
RUNTIME_DIR="$PROJECT_ROOT/logs/demo-runtime"
mkdir -p "$RUNTIME_DIR"

FASTAPI_PORT=8000
N8N_PORT=5678
NOTIFY_PORT=8090
MOCK_NOTIFY_URL="http://127.0.0.1:${NOTIFY_PORT}/"
NOTIFICATIONS_LOG="$RUNTIME_DIR/demo_notifications.jsonl"

port_open() {
  local port="$1"
  "$PYTHON" -c "
import socket, sys
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.settimeout(0.5)
sys.exit(0 if s.connect_ex(('127.0.0.1', $port)) == 0 else 1)
" 2>/dev/null
}

echo "=== [1] Verify ports are empty before starting anything ==="
FAILED=0
for port in "$FASTAPI_PORT" "$N8N_PORT" "$NOTIFY_PORT"; do
  if port_open "$port"; then
    echo "REFUSING TO START: port $port is already in use." >&2
    FAILED=1
  else
    echo "PASS: port $port is free."
  fi
done
if [ "$FAILED" -eq 1 ]; then
  echo "Stop whatever is already listening on the port(s) above, or run" >&2
  echo "scripts/stop_demo_environment.sh first, then retry." >&2
  exit 1
fi

echo ""
echo "=== [2] Force safe environment (never the real .env Slack webhook) ==="
export N8N_NOTIFY_WEBHOOK_URL="$MOCK_NOTIFY_URL"
export FASTAPI_BASE_URL="http://127.0.0.1:${FASTAPI_PORT}"
unset ANTHROPIC_API_KEY
echo "N8N_NOTIFY_WEBHOOK_URL forced to the local mock receiver (a real .env"
echo "value can never win -- start_n8n.sh's loader only fills unset vars)."
echo "ANTHROPIC_API_KEY unset -- any accidental real-provider path fails closed."

echo ""
echo "=== [3] Start mock notification receiver (:$NOTIFY_PORT) ==="
: > "$NOTIFICATIONS_LOG"
nohup "$PYTHON" scripts/mock_notification_receiver.py \
  --port "$NOTIFY_PORT" --log "$NOTIFICATIONS_LOG" \
  > "$RUNTIME_DIR/mock_notify.log" 2>&1 &
echo $! > "$RUNTIME_DIR/mock_notify.pid"
sleep 1
if ! port_open "$NOTIFY_PORT"; then
  echo "FAILED: mock notification receiver did not come up on :$NOTIFY_PORT." >&2
  exit 1
fi
echo "PASS: mock notification receiver up (PID $(cat "$RUNTIME_DIR/mock_notify.pid"))."

echo ""
echo "=== [4] Start real API + deterministic fake-agent provider (:$FASTAPI_PORT) ==="
nohup "$PYTHON" scripts/qa_fake_agent_api_server.py \
  --mode rule_based --port "$FASTAPI_PORT" \
  > "$RUNTIME_DIR/fake_api.log" 2>&1 &
echo $! > "$RUNTIME_DIR/fake_api.pid"
deadline=$((SECONDS + 30))
while [ $SECONDS -lt "$deadline" ]; do
  if grep -q "QA_FAKE_API_READY" "$RUNTIME_DIR/fake_api.log" 2>/dev/null; then break; fi
  sleep 0.3
done
if ! grep -q "QA_FAKE_API_READY" "$RUNTIME_DIR/fake_api.log" 2>/dev/null; then
  echo "FAILED: real API (fake-agent provider) did not become ready on :$FASTAPI_PORT." >&2
  exit 1
fi
echo "PASS: real API (real model, deterministic fake agent provider) up (PID $(cat "$RUNTIME_DIR/fake_api.pid"))."

echo ""
echo "=== [5] Start n8n 2.41.3 (existing owner account, existing active workflow) ==="
nohup ./automation/start_n8n.sh > "$RUNTIME_DIR/n8n.log" 2>&1 &
echo $! > "$RUNTIME_DIR/n8n.pid"
deadline=$((SECONDS + 90))
while [ $SECONDS -lt "$deadline" ]; do
  if port_open "$N8N_PORT"; then break; fi
  sleep 1
done
if ! port_open "$N8N_PORT"; then
  echo "FAILED: n8n did not bind :$N8N_PORT within 90s." >&2
  exit 1
fi
sleep 2  # let the production webhook trigger finish registering
echo "PASS: n8n up (PID $(cat "$RUNTIME_DIR/n8n.pid"))."

echo ""
echo "=== [6] Health check all three services ==="
if ! "$PYTHON" -c "
import urllib.error
import urllib.request
# n8n's bare '/' can legitimately answer with a non-200 (e.g. a client-side
# route) -- any real HTTP response (HTTPError included) proves the server
# is up and listening; only a connection-level failure counts as down.
for name, url in [('FastAPI /health', 'http://127.0.0.1:${FASTAPI_PORT}/health'),
                   ('n8n /', 'http://127.0.0.1:${N8N_PORT}/')]:
    try:
        with urllib.request.urlopen(url, timeout=3) as resp:
            print(f'PASS: {name} -> {resp.status}')
    except urllib.error.HTTPError as exc:
        print(f'PASS: {name} -> {exc.code} (server responded)')
    except Exception as exc:
        print(f'FAIL: {name} -> {exc}')
        raise SystemExit(1)
"; then
  echo "FAILED: health check did not pass for all services." >&2
  exit 1
fi

echo ""
echo "DEMO ENVIRONMENT READY. 0 real Slack/Discord or Anthropic calls made."
echo "PIDs recorded in $RUNTIME_DIR/*.pid -- stop with scripts/stop_demo_environment.sh."
