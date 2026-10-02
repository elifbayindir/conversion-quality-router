#!/usr/bin/env bash
# T803: stop exactly what scripts/start_demo_environment.sh started, by PID
# only -- never a broad pkill (which could kill an unrelated process this
# script has no business touching). Verifies ports 8000/5678/8090 are clean
# afterward.
set -uo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON="${PYTHON:-$PROJECT_ROOT/venv/bin/python}"
RUNTIME_DIR="$PROJECT_ROOT/logs/demo-runtime"

port_open() {
  local port="$1"
  "$PYTHON" -c "
import socket, sys
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.settimeout(0.5)
sys.exit(0 if s.connect_ex(('127.0.0.1', $port)) == 0 else 1)
" 2>/dev/null
}

stop_pid_file() {
  local name="$1"
  local pid_file="$RUNTIME_DIR/$name.pid"
  if [ ! -f "$pid_file" ]; then
    echo "SKIP: no $pid_file (never started by this session, or already stopped)."
    return
  fi
  local pid
  pid="$(cat "$pid_file")"
  if kill -0 "$pid" 2>/dev/null; then
    kill "$pid" 2>/dev/null
    for _ in $(seq 1 20); do
      kill -0 "$pid" 2>/dev/null || break
      sleep 0.5
    done
    if kill -0 "$pid" 2>/dev/null; then
      kill -9 "$pid" 2>/dev/null
    fi
    echo "STOPPED: $name (PID $pid)."
  else
    echo "SKIP: $name (PID $pid) was not running."
  fi
  rm -f "$pid_file"
}

echo "=== Stopping demo environment (PID-based, no broad pkill) ==="
stop_pid_file "n8n"
stop_pid_file "fake_api"
stop_pid_file "mock_notify"

echo ""
echo "=== Verifying ports are clean ==="
FAILED=0
for port in 8000 5678 8090; do
  if port_open "$port"; then
    echo "WARNING: port $port is still in use (something else may be listening)." >&2
    FAILED=1
  else
    echo "PASS: port $port is clean."
  fi
done

if [ "$FAILED" -eq 0 ]; then
  echo ""
  echo "DEMO ENVIRONMENT STOPPED. All ports clean."
fi
exit $FAILED
