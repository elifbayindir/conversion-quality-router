#!/usr/bin/env bash
# T703: real clean-room setup verification.
#
# Builds a clean source snapshot (tracked + non-ignored-untracked files
# only -- never .env, venv, data, model binaries, n8n runtime, logs, cache,
# .git, .project-control), installs it into a fresh Python 3.11 venv in a
# temporary directory outside the project, regenerates every inference-
# required artifact from scratch via the documented pipeline (never
# touching the test set for tuning/selection), runs the real API's
# health/ready/predict smoke test, then starts a second, independent,
# freshly-owned local n8n 2.41.3 instance (its own throwaway synthetic
# owner account -- never the project's real one), imports the committed,
# sanitized workflow export, and runs one production webhook fixture
# against local mocks. Tears down every temporary process at the end.
#
# Never reads the project's root .env. Never calls real Slack/Anthropic.
set -uo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FAILURES=0
STEP=0

step() {
  STEP=$((STEP + 1))
  echo ""
  echo "=== [$STEP] $1 ==="
}

check() {
  if [ "$1" -eq 0 ]; then
    echo "PASS: $2"
  else
    echo "FAIL: $2" >&2
    FAILURES=$((FAILURES + 1))
  fi
}

WORKDIR="$(mktemp -d -t cqr-clean-room)"
SRC_DIR="$WORKDIR/src"
VENV_DIR="$WORKDIR/venv"
N8N_USER_FOLDER_CLEAN="$WORKDIR/n8n-runtime"
CLEAN_N8N_PORT=5680
CLEAN_FASTAPI_PORT=8010
CLEAN_NOTIFY_PORT=8091

N8N_PID=""
FASTAPI_PID=""
NOTIFY_PID=""
ROUTE_MOCK_PID=""

cleanup() {
  step "Teardown"
  for pid in "$N8N_PID" "$FASTAPI_PID" "$NOTIFY_PID" "$ROUTE_MOCK_PID"; do
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null
      wait "$pid" 2>/dev/null
    fi
  done
  echo "Removing $WORKDIR ..."
  rm -rf "$WORKDIR"
}
trap cleanup EXIT

echo "Clean-room workdir: $WORKDIR"

# ---------------------------------------------------------------------------
step "Build clean source snapshot (tracked + non-ignored-untracked only)"
# ---------------------------------------------------------------------------
mkdir -p "$SRC_DIR"
cd "$PROJECT_ROOT"
{ git ls-files; git ls-files --others --exclude-standard; } | sort -u > "$WORKDIR/manifest.txt"
n_files=$(wc -l < "$WORKDIR/manifest.txt" | tr -d ' ')
echo "Copying $n_files files into the clean snapshot ..."
while IFS= read -r rel_path; do
  mkdir -p "$SRC_DIR/$(dirname "$rel_path")"
  cp "$PROJECT_ROOT/$rel_path" "$SRC_DIR/$rel_path"
done < "$WORKDIR/manifest.txt"

for forbidden in .env .project-control automation/.n8n-runtime logs data/raw data/processed \
  artifacts/models artifacts/preprocessors venv .venv .git; do
  if [ -e "$SRC_DIR/$forbidden" ]; then
    echo "FAIL: forbidden path leaked into clean snapshot: $forbidden" >&2
    FAILURES=$((FAILURES + 1))
  fi
done
check "$([ -f "$SRC_DIR/pyproject.toml" ] && echo 0 || echo 1)" "clean snapshot contains pyproject.toml"
check "$([ -f "$SRC_DIR/automation/conversion_quality_router.json" ] && echo 0 || echo 1)" \
  "clean snapshot contains the committed n8n workflow export"

# ---------------------------------------------------------------------------
step "Create a fresh Python 3.11 venv and install the package (editable, dev extras)"
# ---------------------------------------------------------------------------
PY311="$(command -v python3.11)"
if [ -z "$PY311" ]; then
  echo "FAIL: python3.11 not found on PATH -- cannot run the clean-room install." >&2
  exit 1
fi
"$PY311" -m venv "$VENV_DIR"
CLEAN_PY="$VENV_DIR/bin/python"
"$CLEAN_PY" -m pip install --quiet --upgrade pip
if ! "$CLEAN_PY" -m pip install --quiet -e "$SRC_DIR[dev]"; then
  echo "FAIL: pip install -e \".[dev]\" failed in the clean venv." >&2
  FAILURES=$((FAILURES + 1))
else
  echo "PASS: pip install -e \".[dev]\" succeeded in the clean venv."
fi

# ---------------------------------------------------------------------------
step "Data download + checksum verification (real network call to UCI)"
# ---------------------------------------------------------------------------
( cd "$SRC_DIR" && "$CLEAN_PY" scripts/download_data.py )
check $? "download_data.py (checksum-verified download)"

# ---------------------------------------------------------------------------
step "Regenerate every inference-required artifact from scratch"
# ---------------------------------------------------------------------------
for script in make_split.py train_baseline.py select_baseline_threshold.py \
  report_baseline.py train_mlp.py calibrate_mlp.py finalize_mlp.py; do
  ( cd "$SRC_DIR" && "$CLEAN_PY" "scripts/$script" )
  check $? "scripts/$script"
done
check "$([ -f "$SRC_DIR/artifacts/metadata/mlp_metadata.json" ] && echo 0 || echo 1)" \
  "artifacts/metadata/mlp_metadata.json regenerated"
check "$([ -f "$SRC_DIR/artifacts/models/mlp_checkpoint.pt" ] && echo 0 || echo 1)" \
  "artifacts/models/mlp_checkpoint.pt regenerated"

# ---------------------------------------------------------------------------
step "API health/ready/predict smoke test against the freshly regenerated artifacts"
# ---------------------------------------------------------------------------
( cd "$SRC_DIR" && "$CLEAN_PY" scripts/smoke_test.py )
check $? "scripts/smoke_test.py (real uvicorn subprocess, clean venv, clean artifacts)"

# ---------------------------------------------------------------------------
step "Start a second, independent, freshly-owned local n8n 2.41.3 instance"
# ---------------------------------------------------------------------------
mkdir -p "$N8N_USER_FOLDER_CLEAN"
N8N_USER_FOLDER="$N8N_USER_FOLDER_CLEAN" \
  N8N_PORT="$CLEAN_N8N_PORT" \
  N8N_LISTEN_ADDRESS="127.0.0.1" \
  N8N_DIAGNOSTICS_ENABLED="false" \
  N8N_VERSION_NOTIFICATIONS_ENABLED="false" \
  N8N_TEMPLATES_ENABLED="false" \
  N8N_BLOCK_ENV_ACCESS_IN_NODE="false" \
  FASTAPI_BASE_URL="http://127.0.0.1:$CLEAN_FASTAPI_PORT" \
  N8N_NOTIFY_WEBHOOK_URL="http://127.0.0.1:$CLEAN_NOTIFY_PORT/" \
  npx --yes n8n@2.41.3 start \
  > "$WORKDIR/n8n.log" 2>&1 &
N8N_PID=$!

deadline=$((SECONDS + 90))
while [ $SECONDS -lt "$deadline" ]; do
  if curl -s -o /dev/null "http://127.0.0.1:$CLEAN_N8N_PORT/"; then
    break
  fi
  sleep 1
done
if curl -s -o /dev/null "http://127.0.0.1:$CLEAN_N8N_PORT/"; then
  echo "PASS: clean n8n instance is listening on :$CLEAN_N8N_PORT"
else
  echo "FAIL: clean n8n instance did not come up." >&2
  FAILURES=$((FAILURES + 1))
fi

# ---------------------------------------------------------------------------
step "Bootstrap a throwaway synthetic owner account (never the project's real one)"
# ---------------------------------------------------------------------------
# The HTTP port accepts connections (serving the static frontend) noticeably
# before the REST API backend finishes initializing; a request in that
# window gets a bare "n8n is starting up. Please wait" text body back with
# HTTP 200 (not an error status), so status-code-only polling is not
# sufficient here -- retry until the body is actually valid JSON.
OWNER_EMAIL="qa-clean-room@example.invalid"
OWNER_PASSWORD="Qa-CleanRoom-$(date +%s)!"
setup_resp="000"
rest_deadline=$((SECONDS + 60))
while [ $SECONDS -lt "$rest_deadline" ]; do
  setup_resp=$(curl -s -o "$WORKDIR/owner_setup.json" -w "%{http_code}" \
    -X POST "http://127.0.0.1:$CLEAN_N8N_PORT/rest/owner/setup" \
    -H "Content-Type: application/json" \
    -c "$WORKDIR/cookies.txt" \
    -d "{\"email\":\"$OWNER_EMAIL\",\"firstName\":\"QA\",\"lastName\":\"CleanRoom\",\"password\":\"$OWNER_PASSWORD\"}")
  if [ "$setup_resp" = "200" ] && "$CLEAN_PY" -c "import json; json.load(open('$WORKDIR/owner_setup.json'))" 2>/dev/null; then
    break
  fi
  sleep 1
done
check "$([ "$setup_resp" = "200" ] && "$CLEAN_PY" -c "import json; json.load(open('$WORKDIR/owner_setup.json'))" 2>/dev/null && echo 0 || echo 1)" \
  "owner/setup -> 200 with valid JSON (synthetic local credentials, REST API fully ready)"

# ---------------------------------------------------------------------------
step "Import the committed, sanitized workflow export and activate it"
# ---------------------------------------------------------------------------
import_resp=$(curl -s -o "$WORKDIR/import.json" -w "%{http_code}" \
  -X POST "http://127.0.0.1:$CLEAN_N8N_PORT/rest/workflows" \
  -H "Content-Type: application/json" \
  -b "$WORKDIR/cookies.txt" \
  --data-binary @"$SRC_DIR/automation/conversion_quality_router.json")
check "$([ "$import_resp" = "200" ] && echo 0 || echo 1)" "POST /rest/workflows (import committed export) -> 200"

WORKFLOW_ID=$("$CLEAN_PY" -c "
import json
body = json.load(open('$WORKDIR/import.json'))
node = body['data'] if isinstance(body, dict) and 'data' in body else body
print(node['id'])
" 2>/dev/null)
if [ -z "${WORKFLOW_ID:-}" ]; then
  echo "FAIL: could not parse imported workflow id. Response body:" >&2
  cat "$WORKDIR/import.json" >&2
  FAILURES=$((FAILURES + 1))
else
  # Activation requires the current versionId in the body (n8n 2.41.3 REST
  # API contract, confirmed empirically -- a bare POST with no body 400s,
  # and PATCH .../activate is 404; PATCH the workflow's own `active` field
  # directly also silently no-ops).
  VERSION_ID=$("$CLEAN_PY" -c "
import json
body = json.load(open('$WORKDIR/import.json'))
node = body['data'] if isinstance(body, dict) and 'data' in body else body
print(node['versionId'])
" 2>/dev/null)
  activate_resp=$(curl -s -o "$WORKDIR/activate.json" -w "%{http_code}" \
    -X POST "http://127.0.0.1:$CLEAN_N8N_PORT/rest/workflows/$WORKFLOW_ID/activate" \
    -H "Content-Type: application/json" \
    -b "$WORKDIR/cookies.txt" \
    -d "{\"versionId\": \"$VERSION_ID\"}")
  is_active=$("$CLEAN_PY" -c "
import json
body = json.load(open('$WORKDIR/activate.json'))
node = body['data'] if isinstance(body, dict) and 'data' in body else body
print(node.get('active'))
" 2>/dev/null)
  check "$([ "$activate_resp" = "200" ] && [ "$is_active" = "True" ] && echo 0 || echo 1)" \
    "POST /rest/workflows/$WORKFLOW_ID/activate -> 200, active=true"
  sleep 2  # let the production webhook trigger finish registering
fi

# ---------------------------------------------------------------------------
step "Start local mocks (API route + notification) for the fixture run"
# ---------------------------------------------------------------------------
"$CLEAN_PY" "$SRC_DIR/scripts/mock_route_server.py" --port "$CLEAN_FASTAPI_PORT" \
  > "$WORKDIR/mock_route.log" 2>&1 &
ROUTE_MOCK_PID=$!
"$CLEAN_PY" "$SRC_DIR/scripts/mock_notification_receiver.py" --port "$CLEAN_NOTIFY_PORT" \
  --log "$WORKDIR/mock_notifications.jsonl" \
  > "$WORKDIR/mock_notify.log" 2>&1 &
NOTIFY_PID=$!
sleep 1.5

# ---------------------------------------------------------------------------
step "Run one production webhook fixture against the clean instance"
# ---------------------------------------------------------------------------
# The fixture file wraps the payload under an "input" key; extract it first.
"$CLEAN_PY" -c "
import json
fixture = json.load(open('$SRC_DIR/automation/fixtures/valid_log_only.json'))
json.dump(fixture['input'], open('$WORKDIR/fixture_input.json', 'w'))
"
fixture_resp=$(curl -s -o "$WORKDIR/fixture_response.json" -w "%{http_code}" \
  -X POST "http://127.0.0.1:$CLEAN_N8N_PORT/webhook/conversion-quality-router" \
  -H "Content-Type: application/json" \
  -d @"$WORKDIR/fixture_input.json")
check "$([ "$fixture_resp" = "200" ] && echo 0 || echo 1)" \
  "webhook fixture (valid_log_only) against the clean instance -> 200"
if [ "$fixture_resp" = "200" ]; then
  decision=$("$CLEAN_PY" -c "import json; print(json.load(open('$WORKDIR/fixture_response.json')).get('decision'))")
  check "$([ "$decision" = "LOG_ONLY" ] && echo 0 || echo 1)" "clean instance response decision=LOG_ONLY (got: $decision)"
fi

echo ""
if [ "$FAILURES" -eq 0 ]; then
  echo "CLEAN-ROOM VERIFICATION: PASSED (0 real external calls beyond the dataset download)."
  exit 0
else
  echo "CLEAN-ROOM VERIFICATION: FAILED -- $FAILURES check(s) failed." >&2
  exit 1
fi
