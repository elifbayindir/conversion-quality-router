#!/usr/bin/env bash
# Stop exactly what scripts/start_ui.sh started, by PID only (never a broad
# pkill): the recorded PID and the port listener, each killed only when its
# command line matches start_ui.sh's launch signature. Then verify every local
# port this project uses is clean: 8000 (API), 8090 (mock notifications),
# 5678 (n8n), 8501 (UI).
set -uo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON="${PYTHON:-$PROJECT_ROOT/venv/bin/python}"
RUNTIME_DIR="$PROJECT_ROOT/logs/ui-runtime"

port_open() {
  "$PYTHON" -c "
import socket, sys
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.settimeout(0.5)
sys.exit(0 if s.connect_ex(('127.0.0.1', $1)) == 0 else 1)
" 2>/dev/null
}

# Exact launch signatures written by start_ui.sh. A PID is stopped only when
# its command line matches one of these, so nothing unrelated is ever killed.
API_SIGNATURE="$PROJECT_ROOT/venv/bin/python scripts/qa_fake_agent_api_server.py"
UI_SIGNATURE="$PROJECT_ROOT/venv/bin/python -m streamlit run src/conversion_router/ui/app.py"

listener_pid() {
  lsof -nP -t -iTCP:"$1" -sTCP:LISTEN 2>/dev/null | head -1
}

stop_service() {
  local name="$1" port="$2" signature="$3"
  local pid_file="$RUNTIME_DIR/$name.pid"
  local candidates=""
  [ -f "$pid_file" ] && candidates="$(cat "$pid_file")"
  # The recorded PID can be stale (for example after a file sync restored an
  # old copy), so the current listener on this service's port is also checked.
  candidates="$candidates $(listener_pid "$port")"
  local stopped=0
  for pid in $(echo "$candidates" | tr ' ' '\n' | sort -u); do
    [ -n "$pid" ] || continue
    kill -0 "$pid" 2>/dev/null || continue
    local cmd
    cmd="$(ps -o command= -p "$pid" 2>/dev/null)"
    case "$cmd" in
      "$signature"*)
        kill "$pid" 2>/dev/null
        for _ in $(seq 1 20); do
          kill -0 "$pid" 2>/dev/null || break
          sleep 0.5
        done
        kill -0 "$pid" 2>/dev/null && kill -9 "$pid" 2>/dev/null
        echo "STOPPED: $name (PID $pid)."
        stopped=1
        ;;
      *)
        echo "LEFT RUNNING: PID $pid (recorded for $name or listening on :$port) was not started by start_ui.sh." >&2
        ;;
    esac
  done
  [ "$stopped" -eq 1 ] || echo "SKIP: $name was not running."
  rm -f "$pid_file"
}

echo "=== Stopping UI environment (PID-based, signature-checked) ==="
stop_service "ui" 8501 "$UI_SIGNATURE"
stop_service "api" 8000 "$API_SIGNATURE"

echo ""
echo "=== Verifying ports are clean ==="
FAILED=0
for port in 8000 8090 5678 8501; do
  if port_open "$port"; then
    echo "WARNING: port $port is still in use." >&2
    FAILED=1
  else
    echo "PASS: port $port is clean."
  fi
done
[ "$FAILED" -eq 0 ] && echo "" && echo "UI ENVIRONMENT STOPPED. All ports clean."
exit $FAILED
