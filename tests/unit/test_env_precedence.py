"""Regression test for scripts/load_env_with_precedence.sh.

This exercises the REAL shared script used by automation/start_n8n.sh (not
a reimplementation), so it can never silently drift out of sync with the
actual behavior that protects against a mock test run accidentally picking
up the real N8N_NOTIFY_WEBHOOK_URL from .env.
"""

import shlex
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOADER_SCRIPT = PROJECT_ROOT / "scripts" / "load_env_with_precedence.sh"


def _run_loader(env_file: Path, preset_value: str | None) -> str:
    # Always start from a clean slate: unset first, regardless of whatever
    # this test process's own environment happens to have (e.g. when this
    # suite runs under scripts/run_quality_gate.sh, which pre-sets a mock
    # N8N_NOTIFY_WEBHOOK_URL for its own safety -- that must not leak in
    # and change what these specific sub-scenarios are testing).
    preset = "unset N8N_NOTIFY_WEBHOOK_URL;"
    if preset_value:
        preset += f" export N8N_NOTIFY_WEBHOOK_URL={shlex.quote(preset_value)};"
    command = (
        f"{preset} "
        f"source {shlex.quote(str(LOADER_SCRIPT))} && "
        f"load_env_with_precedence {shlex.quote(str(env_file))} && "
        f'printf "%s" "${{N8N_NOTIFY_WEBHOOK_URL:-__UNSET__}}"'
    )
    result = subprocess.run(
        ["bash", "-c", command],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout


def test_shell_preset_value_wins_over_dotenv(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "N8N_NOTIFY_WEBHOOK_URL=https://hooks.slack.com/services/REAL/SECRET/VALUE\n"
    )

    output = _run_loader(env_file, preset_value="http://127.0.0.1:8090/")

    assert output == "http://127.0.0.1:8090/"
    assert "hooks.slack.com" not in output


def test_dotenv_value_is_used_when_shell_var_is_unset(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("N8N_NOTIFY_WEBHOOK_URL=http://example.invalid/placeholder\n")

    output = _run_loader(env_file, preset_value=None)

    assert output == "http://example.invalid/placeholder"


def test_missing_env_file_leaves_variable_unset(tmp_path):
    missing_env_file = tmp_path / "does_not_exist.env"

    output = _run_loader(missing_env_file, preset_value=None)

    assert output == "__UNSET__"


def test_comments_and_blank_lines_in_env_file_are_ignored(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "# a comment\n\nN8N_NOTIFY_WEBHOOK_URL=http://example.invalid/from-file\n"
    )

    output = _run_loader(env_file, preset_value=None)

    assert output == "http://example.invalid/from-file"
