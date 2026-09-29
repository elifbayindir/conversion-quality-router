#!/usr/bin/env python3
"""Live local smoke test: boots the real API as a subprocess and hits it
over real HTTP (not FastAPI's in-process TestClient), against the real
frozen artifacts.

Covers: valid request, missing field, unknown field, out-of-range value.
The "unavailable" scenario (missing/mismatched artifacts) is covered by the
automated, in-process test suite (tests/integration/test_api_routes.py),
which simulates it safely without touching real artifact files; this script
deliberately does not tamper with real artifacts on disk.

Exit code 0 = all checks passed. Non-zero = failure (see printed detail).
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import httpx

PROJECT_ROOT = Path(__file__).resolve().parent.parent
HOST = "127.0.0.1"
PORT = 8099
BASE_URL = f"http://{HOST}:{PORT}"
STARTUP_TIMEOUT_SECONDS = 30

VALID_PAYLOAD = {
    "Administrative": 2,
    "Administrative_Duration": 50.0,
    "Informational": 1,
    "Informational_Duration": 20.0,
    "ProductRelated": 25,
    "ProductRelated_Duration": 800.0,
    "BounceRates": 0.01,
    "ExitRates": 0.02,
    "SpecialDay": 0.0,
    "Month": "Nov",
    "OperatingSystems": 2,
    "Browser": 2,
    "Region": 1,
    "TrafficType": 2,
    "VisitorType": "Returning_Visitor",
    "Weekend": True,
}


def wait_for_health(client: httpx.Client) -> None:
    deadline = time.monotonic() + STARTUP_TIMEOUT_SECONDS
    last_error = None
    while time.monotonic() < deadline:
        try:
            response = client.get("/health", timeout=2.0)
            if response.status_code == 200:
                return
        except httpx.HTTPError as exc:
            last_error = exc
        time.sleep(0.3)
    raise RuntimeError(f"Server did not become healthy in time. Last error: {last_error}")


def check(condition: bool, description: str, failures: list[str]) -> None:
    print(f"{'PASS' if condition else 'FAIL'}: {description}")
    if not condition:
        failures.append(description)


def main() -> int:
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "conversion_router.api.app:app",
            "--host",
            HOST,
            "--port",
            str(PORT),
            "--log-level",
            "warning",
        ],
        cwd=PROJECT_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    failures: list[str] = []

    try:
        with httpx.Client(base_url=BASE_URL) as client:
            wait_for_health(client)

            health_resp = client.get("/health")
            check(health_resp.status_code == 200, "GET /health -> 200", failures)
            check(
                health_resp.json() == {"status": "ok"},
                "GET /health body == {status: ok}",
                failures,
            )

            ready_resp = client.get("/ready")
            check(
                ready_resp.status_code == 200,
                "GET /ready -> 200 (real artifacts present)",
                failures,
            )
            check(
                ready_resp.json().get("model_name") == "conversion_mlp",
                "GET /ready reports model_name=conversion_mlp",
                failures,
            )

            valid_resp = client.post("/predict", json=VALID_PAYLOAD)
            check(valid_resp.status_code == 200, "POST /predict valid payload -> 200", failures)
            body = valid_resp.json()
            check(
                "prediction" in body and "purchase_probability" in body["prediction"],
                "POST /predict valid payload has prediction.purchase_probability",
                failures,
            )

            missing = {k: v for k, v in VALID_PAYLOAD.items() if k != "Weekend"}
            missing_resp = client.post("/predict", json=missing)
            check(missing_resp.status_code == 422, "POST /predict missing field -> 422", failures)

            unknown = {**VALID_PAYLOAD, "UnexpectedField": 1}
            unknown_resp = client.post("/predict", json=unknown)
            check(unknown_resp.status_code == 422, "POST /predict unknown field -> 422", failures)

            out_of_range = {**VALID_PAYLOAD, "BounceRates": 5.0}
            oor_resp = client.post("/predict", json=out_of_range)
            check(oor_resp.status_code == 422, "POST /predict out-of-range value -> 422", failures)

    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()

    print()
    if failures:
        print(f"SMOKE TEST FAILED: {len(failures)} check(s) failed: {failures}")
        return 1
    print("SMOKE TEST PASSED: all checks green.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
