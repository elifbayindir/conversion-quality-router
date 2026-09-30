#!/usr/bin/env bash
# Load KEY=VALUE pairs from a .env-style file into the environment, WITHOUT
# ever overriding a variable already set in the calling shell. Never echoes
# values.
#
# Used by automation/start_n8n.sh so a caller can force a mock URL (e.g.
# N8N_NOTIFY_WEBHOOK_URL=http://127.0.0.1:8090/) and be certain .env's real
# value can never silently replace it. Regression-tested directly (not
# reimplemented) by tests/unit/test_env_precedence.py.
#
# Usage:
#   source scripts/load_env_with_precedence.sh
#   load_env_with_precedence /path/to/.env

load_env_with_precedence() {
  local env_file="$1"
  if [ -f "$env_file" ]; then
    while IFS='=' read -r key value; do
      case "$key" in ''|'#'*) continue ;; esac
      if [ -z "${!key:-}" ]; then
        export "$key=$value"
      fi
    done < "$env_file"
  fi
}
