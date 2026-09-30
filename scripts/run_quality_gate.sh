#!/usr/bin/env bash
# Single, documented QA gate. Runs: tracker validate, ruff, the full pytest
# suite (including agent adversarial tests and n8n workflow structural
# tests), a clean re-execution of the analysis notebook, a live local API
# smoke test, and a standalone n8n workflow export validation.
#
# SAFETY: this script makes ZERO real Anthropic, Slack/Discord, or other
# external calls by default. It never sources the project's .env. External
# service variables are forced to explicit loopback mock addresses before
# anything runs, and ANTHROPIC_API_KEY is unset so any accidental
# real-provider code path fails closed (AgentCredentialError) instead of
# succeeding with a real credential. A non-loopback N8N_NOTIFY_WEBHOOK_URL
# already present in the environment is refused unless
# ALLOW_LIVE_EXTERNAL_ACTION=1 is explicitly set by the caller.
#
# Usage: venv/bin/python -m pytest ... is NOT how this is invoked -- run:
#   scripts/run_quality_gate.sh
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON="${PYTHON:-$PROJECT_ROOT/venv/bin/python}"
FAILURES=0
STEP=0

step() {
  STEP=$((STEP + 1))
  echo ""
  echo "=== [$STEP] $1 ==="
}

run() {
  if ! "$@"; then
    echo "FAILED: $*" >&2
    FAILURES=$((FAILURES + 1))
  fi
}

# ---------------------------------------------------------------------------
# Safety guard: refuse a non-loopback notification URL unless explicitly
# allowed. Must run BEFORE forcing the mock defaults below.
# ---------------------------------------------------------------------------
if [ -n "${N8N_NOTIFY_WEBHOOK_URL:-}" ]; then
  case "$N8N_NOTIFY_WEBHOOK_URL" in
    http://127.0.0.1:*|http://localhost:*) ;;
    *)
      if [ "${ALLOW_LIVE_EXTERNAL_ACTION:-0}" != "1" ]; then
        echo "REFUSING TO RUN: N8N_NOTIFY_WEBHOOK_URL is not a loopback address" >&2
        echo "and ALLOW_LIVE_EXTERNAL_ACTION is not '1'. This QA gate must never" >&2
        echo "be able to trigger a real external action." >&2
        exit 1
      fi
      ;;
  esac
fi

# Force safe defaults regardless of what the calling shell/.env would provide.
# This script never sources .env.
export N8N_NOTIFY_WEBHOOK_URL="http://127.0.0.1:8090/"
export FASTAPI_BASE_URL="http://127.0.0.1:8000"
unset ANTHROPIC_API_KEY

# Regression check (T701 requirement): a shell-set mock value must survive
# and win over whatever a real .env might contain, using the SAME shared
# loader automation/start_n8n.sh uses -- not a reimplementation.
step "Regression: mock env var takes precedence over .env (fail-closed check)"
TMP_ENV_CHECK="$(mktemp)"
echo "N8N_NOTIFY_WEBHOOK_URL=https://hooks.slack.com/services/SHOULD/NEVER/WIN" > "$TMP_ENV_CHECK"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/load_env_with_precedence.sh"
load_env_with_precedence "$TMP_ENV_CHECK"
rm -f "$TMP_ENV_CHECK"
if [ "$N8N_NOTIFY_WEBHOOK_URL" != "http://127.0.0.1:8090/" ]; then
  echo "FAIL-CLOSED: N8N_NOTIFY_WEBHOOK_URL was overridden by a fake .env value!" >&2
  echo "This must never happen -- aborting the entire QA gate." >&2
  exit 1
fi
echo "PASS: mock value survived; a real .env value can never silently win."

step "tracker.py validate"
run "$PYTHON" .project-control/tracker.py validate

step "ruff check ."
run "$PYTHON" -m ruff check .

step "full pytest suite (includes agent adversarial + n8n structural tests)"
run "$PYTHON" -m pytest -q

step "agent adversarial/mock tests (explicit re-run for a clear standalone count)"
run "$PYTHON" -m pytest -q \
  tests/unit/test_agent_client.py \
  tests/unit/test_agent_validator.py \
  tests/unit/test_agent_fallback.py \
  tests/integration/test_api_decide_route.py

step "n8n workflow export structural validation"
run "$PYTHON" scripts/validate_workflow_export.py

step "notebook clean re-execution (local computation only, no external calls)"
if [ -f notebooks/01_end_to_end_analysis.ipynb ]; then
  run "$PYTHON" -m nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=300 \
    notebooks/01_end_to_end_analysis.ipynb
else
  echo "SKIP: notebooks/01_end_to_end_analysis.ipynb not present yet."
fi

step "API smoke test (real local uvicorn subprocess; /health /ready /predict only)"
run "$PYTHON" scripts/smoke_test.py

echo ""
echo "=== Cleanup ==="
find "$PROJECT_ROOT" -type d -name "__pycache__" \
  -not -path "*/.git/*" -not -path "*/venv/*" -not -path "*/.project-control/*" \
  -exec rm -rf {} + 2>/dev/null
rm -rf "$PROJECT_ROOT/.pytest_cache" "$PROJECT_ROOT/.ruff_cache"

echo ""
if [ "$FAILURES" -eq 0 ]; then
  echo "QUALITY GATE: PASSED (0 real external calls made)."
  exit 0
else
  echo "QUALITY GATE: FAILED -- $FAILURES step(s) failed. See output above." >&2
  exit 1
fi
