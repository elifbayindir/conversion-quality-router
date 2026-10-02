#!/usr/bin/env bash
# Start the local decision-support UI in one command, fail-closed.
#
# Starts exactly two local processes, both bound to 127.0.0.1:
#   1. the REAL FastAPI app (real frozen model, real agent validation/retry/
#      fallback chain) with a deterministic demo provider injected explicitly
#      (scripts/qa_fake_agent_api_server.py) on :8000;
#   2. the Streamlit UI (src/conversion_router/ui/app.py) on :8501.
#
# SAFETY: makes ZERO Anthropic, Slack/Discord, or other external calls.
# ANTHROPIC_API_KEY is exported as an EMPTY string: python-dotenv never
# overrides a variable that is already present, so a real key in .env can
# never be loaded and any accidental real-provider path fails closed.
# This script never reads .env itself. Streamlit usage statistics are off.
#
# Usage:
#   scripts/start_ui.sh                  # rule-based demo provider
#   scripts/start_ui.sh --fallback-demo  # provider always invalid -> SYSTEM_FALLBACK
# Stop with scripts/stop_ui.sh (PID-based only, never a broad pkill).
set -uo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON="${PYTHON:-$PROJECT_ROOT/venv/bin/python}"
RUNTIME_DIR="$PROJECT_ROOT/logs/ui-runtime"
API_PORT=8000
UI_PORT=8501
API_MODE="rule_based"
RUN_MODE="local_demo"

case "${1:-}" in
  "") ;;
  --fallback-demo) API_MODE="invalid"; RUN_MODE="local_demo_fallback" ;;
  *) echo "Unknown option: $1 (expected --fallback-demo or nothing)" >&2; exit 2 ;;
esac

port_open() {
  "$PYTHON" -c "
import socket, sys
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.settimeout(0.5)
sys.exit(0 if s.connect_ex(('127.0.0.1', $1)) == 0 else 1)
" 2>/dev/null
}

if ! "$PYTHON" -c "import streamlit" 2>/dev/null; then
  echo "Streamlit is not installed. Run: venv/bin/python -m pip install -e \".[dev,ui]\"" >&2
  exit 1
fi

echo "=== [1] Verify ports are free ==="
FAILED=0
for port in "$API_PORT" "$UI_PORT"; do
  if port_open "$port"; then
    echo "REFUSING TO START: port $port is already in use." >&2
    FAILED=1
  else
    echo "PASS: port $port is free."
  fi
done
[ "$FAILED" -eq 0 ] || { echo "Run scripts/stop_ui.sh or free the port(s), then retry." >&2; exit 1; }

# On any failure after this point, stop only the processes this run started.
API_PID=""
UI_PID=""
cleanup_on_failure() {
  local rc=$?
  if [ "$rc" -ne 0 ]; then
    for pid in $UI_PID $API_PID; do
      kill "$pid" 2>/dev/null && echo "CLEANUP: stopped PID $pid started by this run." >&2
    done
    rm -f "$RUNTIME_DIR/api.pid" "$RUNTIME_DIR/ui.pid"
  fi
}
trap cleanup_on_failure EXIT

echo ""
echo "=== [2] Force a safe environment ==="
mkdir -p "$RUNTIME_DIR"
export ANTHROPIC_API_KEY=""
export N8N_NOTIFY_WEBHOOK_URL="http://127.0.0.1:8090/"
export CQR_API_BASE_URL="http://127.0.0.1:${API_PORT}"
export STREAMLIT_BROWSER_GATHER_USAGE_STATS="false"
# Tells the UI which decision provider this launcher started, so it can label
# the run as a local demo. Without it the UI never claims to be a demo.
export CQR_DECISION_PROVIDER_MODE="$RUN_MODE"
echo "ANTHROPIC_API_KEY forced empty (fail-closed); API base URL is loopback."
echo "UI run mode: $RUN_MODE (deterministic demo provider, no external calls)."

echo ""
echo "=== [3] $(date +%H:%M:%S) Start API with the '$API_MODE' demo provider (:$API_PORT) ==="
nohup "$PYTHON" scripts/qa_fake_agent_api_server.py --mode "$API_MODE" --port "$API_PORT" \
  > "$RUNTIME_DIR/api.log" 2>&1 &
API_PID=$!
echo "$API_PID" > "$RUNTIME_DIR/api.pid"
deadline=$((SECONDS + 30))
until grep -q "QA_FAKE_API_READY" "$RUNTIME_DIR/api.log" 2>/dev/null; do
  if [ $SECONDS -ge "$deadline" ]; then
    echo "FAILED: API did not become ready. See $RUNTIME_DIR/api.log" >&2
    exit 1
  fi
  sleep 0.3
done
if [ "$(lsof -nP -t -iTCP:"$API_PORT" -sTCP:LISTEN 2>/dev/null | head -1)" != "$API_PID" ]; then
  echo "FAILED: the API listener on :$API_PORT is not PID $API_PID." >&2
  exit 1
fi
echo "PASS: API ready (PID $API_PID, verified as the :$API_PORT listener)."

echo ""
echo "=== [4] $(date +%H:%M:%S) Start Streamlit UI (127.0.0.1:$UI_PORT) ==="
nohup "$PYTHON" -m streamlit run src/conversion_router/ui/app.py \
  --server.address 127.0.0.1 --server.port "$UI_PORT" --server.headless true \
  --browser.gatherUsageStats false --server.fileWatcherType none \
  --client.toolbarMode minimal \
  > "$RUNTIME_DIR/ui.log" 2>&1 &
UI_PID=$!
echo "$UI_PID" > "$RUNTIME_DIR/ui.pid"
deadline=$((SECONDS + 60))
until "$PYTHON" -c "
import urllib.request, sys
try:
    with urllib.request.urlopen('http://127.0.0.1:${UI_PORT}/_stcore/health', timeout=2) as r:
        sys.exit(0 if r.status == 200 else 1)
except Exception:
    sys.exit(1)
" 2>/dev/null; do
  if [ $SECONDS -ge "$deadline" ]; then
    echo "FAILED: UI did not become healthy. See $RUNTIME_DIR/ui.log" >&2
    exit 1
  fi
  sleep 0.5
done
if [ "$(lsof -nP -t -iTCP:"$UI_PORT" -sTCP:LISTEN 2>/dev/null | head -1)" != "$UI_PID" ]; then
  echo "FAILED: the UI listener on :$UI_PORT is not PID $UI_PID." >&2
  exit 1
fi
for name in api ui; do
  expected="$API_PID"; [ "$name" = "ui" ] && expected="$UI_PID"
  if [ "$(cat "$RUNTIME_DIR/$name.pid" 2>/dev/null)" != "$expected" ]; then
    echo "WARNING: $name.pid does not read back as $expected; rewriting it." >&2
    echo "$expected" > "$RUNTIME_DIR/$name.pid"
  fi
done
echo "PASS: UI healthy (PID $UI_PID, verified as the :$UI_PORT listener)."

echo ""
echo "UI READY: http://127.0.0.1:${UI_PORT}  (0 external calls; stop with scripts/stop_ui.sh)"
