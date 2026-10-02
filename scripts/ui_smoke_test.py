#!/usr/bin/env python3
"""Live smoke test for the local UI environment.

Starts `scripts/start_ui.sh` (optionally `--fallback-demo`), then over real
loopback HTTP: checks the Streamlit page and health endpoint, routes the three
presets through the UI's own `ApiClient` (/predict -> /decide), verifies both
processes listen on 127.0.0.1 only and hold no non-loopback connections
(`lsof`), and always stops the environment with `scripts/stop_ui.sh`.

Usage:
    venv/bin/python scripts/ui_smoke_test.py
    venv/bin/python scripts/ui_smoke_test.py --fallback-demo
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

from streamlit.testing.v1 import AppTest

from conversion_router.feedback.schemas import HumanResponse, OverrideReason
from conversion_router.feedback.store import FeedbackStore
from conversion_router.schemas import Decision
from conversion_router.ui.api_client import ApiClient
from conversion_router.ui.features import load_presets
from conversion_router.ui.settings import RunMode

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUNTIME_DIR = PROJECT_ROOT / "logs" / "ui-runtime"
APP_PATH = PROJECT_ROOT / "src" / "conversion_router" / "ui" / "app.py"
LOOPBACK = ("127.0.0.1", "localhost", "[::1]", "*")

EXPECTED_ROUTES = {
    "real_model_priority_review": Decision.PRIORITY_REVIEW,
    "real_model_human_review": Decision.HUMAN_REVIEW,
    "real_model_log_only": Decision.LOG_ONLY,
}

failures: list[str] = []


def check(condition: bool, label: str) -> None:
    print(("PASS: " if condition else "FAIL: ") + label)
    if not condition:
        failures.append(label)


def network_rows(pid: str) -> list[str]:
    out = subprocess.run(
        ["lsof", "-nP", "-a", "-p", pid, "-i"], capture_output=True, text=True, check=False
    ).stdout
    return out.splitlines()[1:]


def listener_pid(port: int) -> str:
    out = subprocess.run(
        ["lsof", "-nP", "-t", f"-iTCP:{port}", "-sTCP:LISTEN"],
        capture_output=True, text=True, check=False,
    ).stdout.split()
    return out[0] if out else ""


def non_loopback(rows: list[str]) -> list[str]:
    bad = []
    for row in rows:
        name = row.split()[8] if len(row.split()) > 8 else ""
        endpoints = name.replace("->", " ").split()
        hosts = [ep.rsplit(":", 1)[0] for ep in endpoints]
        if any(host not in LOOPBACK for host in hosts):
            bad.append(row)
    return bad


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fallback-demo", action="store_true")
    args = parser.parse_args()

    start_cmd = [str(PROJECT_ROOT / "scripts" / "start_ui.sh")]
    if args.fallback_demo:
        start_cmd.append("--fallback-demo")
    started = subprocess.run(start_cmd, cwd=PROJECT_ROOT, check=False)
    smoke_dir = tempfile.TemporaryDirectory()
    smoke_db = Path(smoke_dir.name) / "smoke_feedback.sqlite3"
    try:
        check(started.returncode == 0, "start_ui.sh brought up API and UI")
        if started.returncode != 0:
            print("UI SMOKE: FAILED (environment did not start)")
            return 1

        with urllib.request.urlopen("http://127.0.0.1:8501/", timeout=5) as resp:
            check(resp.status == 200 and b"<html" in resp.read().lower(), "UI page served")
        client = ApiClient()
        check(client.health(), "API /health over loopback")
        ready = client.ready()
        check(ready.model_name == "conversion_mlp", f"API /ready model {ready.model_version}")

        for preset in load_presets():
            result = client.route(preset.session)
            expected = (
                Decision.SYSTEM_FALLBACK if args.fallback_demo else EXPECTED_ROUTES[preset.key]
            )
            check(
                result.decision.decision == expected
                and result.decision.request_id == result.prediction.request_id,
                f"{preset.label}: p={result.prediction.prediction.purchase_probability} "
                f"uncertain={result.prediction.prediction.uncertain} -> "
                f"{result.decision.decision} (request {result.request_id})",
            )

        # Run the real UI script against the live API over loopback HTTP.
        mode = RunMode.LOCAL_DEMO_FALLBACK if args.fallback_demo else RunMode.LOCAL_DEMO
        app = AppTest.from_file(str(APP_PATH), default_timeout=60)
        app.session_state["api_client"] = ApiClient()
        app.session_state["feedback_store"] = FeedbackStore(smoke_db)
        app.session_state["run_mode"] = mode
        app.run()
        sidebar = " ".join(m.value for m in app.sidebar.markdown)
        check("No external LLM or messaging calls" in sidebar, f"sidebar shows run mode {mode}")

        app.selectbox(key="preset_label").set_value("Strong purchase signal").run()
        app.radio(key="evaluation_mode").set_value("Model only").run()
        app.button(key="evaluate_button").click().run()
        page = " ".join(m.value for m in app.markdown)
        check(
            not app.exception and "Model-only evaluation" in page
            and 'data-role="probability"' in page,
            "model-only result rendered from the live API (no route requested)",
        )

        app.radio(key="evaluation_mode").set_value("Full decision routing").run()
        app.button(key="evaluate_button").click().run()
        expected_route = "SYSTEM_FALLBACK" if args.fallback_demo else "PRIORITY_REVIEW"
        page = " ".join(m.value for m in app.markdown)
        check(
            not app.exception and f'data-route="{expected_route}"' in page,
            f"full-routing result card {expected_route} rendered from the live API",
        )

        # Record reviewer feedback for that live decision in a temporary store,
        # then reopen the store as a new object (restart) and read it back.
        if args.fallback_demo:
            app.radio(key="fb_response").set_value(HumanResponse.OVERRIDE).run()
            app.selectbox(key="fb_decision").set_value(Decision.HUMAN_REVIEW).run()
            app.selectbox(key="fb_reason").set_value(OverrideReason.FALLBACK_RESOLVED_BY_REVIEW).run()
        else:
            app.radio(key="fb_response").set_value(HumanResponse.AGREE).run()
        app.button(key="fb_submit").click().run()
        stored = FeedbackStore(smoke_db).records()
        check(
            not app.exception and len(stored) == 1,
            f"feedback recorded for the live decision and read back after reopen "
            f"({stored[0].human_response if stored else 'none'} -> "
            f"{stored[0].final_decision if stored else 'none'})",
        )

        ui_env = subprocess.run(
            ["ps", "eww", "-o", "command=", "-p", listener_pid(8501)],
            capture_output=True, text=True, check=False,
        ).stdout
        check(
            f"CQR_DECISION_PROVIDER_MODE={mode.value}" in ui_env,
            f"served UI process was started with run mode {mode.value}",
        )

        for name, port in (("api", 8000), ("ui", 8501)):
            pid = listener_pid(port)
            recorded = (RUNTIME_DIR / f"{name}.pid").read_text().strip()
            check(pid == recorded, f"{name}.pid ({recorded}) matches the :{port} listener ({pid})")
            rows = network_rows(pid)
            listens = [r for r in rows if "(LISTEN)" in r]
            check(
                bool(listens) and all("127.0.0.1:" in r for r in listens),
                f"{name} (PID {pid}) listens on 127.0.0.1 only",
            )
            bad = non_loopback(rows)
            check(not bad, f"{name} (PID {pid}) has 0 non-loopback connections")
            for row in bad:
                print("   ", row)
    finally:
        stopped = subprocess.run(
            [str(PROJECT_ROOT / "scripts" / "stop_ui.sh")], cwd=PROJECT_ROOT, check=False
        )
        check(stopped.returncode == 0, "stop_ui.sh stopped by PID and ports are clean")

    print("")
    print("UI SMOKE: " + ("PASSED" if not failures else f"FAILED ({len(failures)})"))
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
