"""Explicit UI configuration: run mode and display-only policy parameters.

The run mode is declared by the launcher (`scripts/start_ui.sh` sets
`CQR_DECISION_PROVIDER_MODE`). When it is not declared, the UI does not claim
to be a demo; it says the mode is unknown. No credential or provider secret is
read or shown here.

The uncertainty band is not part of the API response, so for the probability
rail it is read from the frozen model metadata file. It is used for drawing
only, and only when that file's threshold equals the threshold the service
returned; the `uncertain` flag shown to the user always comes from the API.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODE_ENV_VAR = "CQR_DECISION_PROVIDER_MODE"
METADATA_ENV_VAR = "CQR_MODEL_METADATA"
DEFAULT_METADATA_PATH = PROJECT_ROOT / "artifacts" / "metadata" / "mlp_metadata.json"


class RunMode(StrEnum):
    LOCAL_DEMO = "local_demo"
    LOCAL_DEMO_FALLBACK = "local_demo_fallback"
    UNDECLARED = "undeclared"


RUN_MODE_TEXT: dict[RunMode, str] = {
    RunMode.LOCAL_DEMO: "Local demo mode — trained model with a deterministic decision "
    "provider. No external LLM or messaging calls.",
    RunMode.LOCAL_DEMO_FALLBACK: "Local demo mode (safety-fallback demonstration) — trained "
    "model; the deterministic decision provider deliberately returns invalid output so the "
    "safety fallback is shown. No external LLM or messaging calls.",
    RunMode.UNDECLARED: "Run mode not declared by the launcher. Decisions come from whichever "
    "provider the connected service is configured with.",
}

AUTOMATION_PREVIEW_NOTE = (
    " This interface previews the recommended operational action. It does not trigger the "
    "n8n workflow or send Slack notifications."
)

AUTOMATION_BOUNDARY = (
    ("Agree / Override here", "Local audit feedback. It does not resume an n8n execution."),
    ("n8n human approval", "Happens in the separate automation workflow."),
    ("Slack", "Operational notification channel only; it does not produce predictions or "
     "decisions."),
)

ROUTING_HELP: dict[RunMode, str] = {
    RunMode.LOCAL_DEMO: "Runs the trained model, then the decision policy. In local demo "
    "mode the policy is applied by a deterministic provider that follows the written decision "
    "rules instead of a language model; its output still passes the same validation."
    + AUTOMATION_PREVIEW_NOTE,
    RunMode.LOCAL_DEMO_FALLBACK: "Runs the trained model, then the decision policy. In this "
    "demonstration the decision provider always fails validation, so every session is routed "
    "by the safety fallback." + AUTOMATION_PREVIEW_NOTE,
    RunMode.UNDECLARED: "Runs the model, then the decision policy configured on the service."
    + AUTOMATION_PREVIEW_NOTE,
}


def run_mode(environ: dict | None = None) -> RunMode:
    value = (environ if environ is not None else os.environ).get(MODE_ENV_VAR, "")
    try:
        return RunMode(value)
    except ValueError:
        return RunMode.UNDECLARED


@dataclass(frozen=True)
class DisplayPolicy:
    threshold: float
    uncertainty_band: float


def load_display_policy(path: Path | None = None) -> DisplayPolicy | None:
    path = path or Path(os.environ.get(METADATA_ENV_VAR, DEFAULT_METADATA_PATH))
    try:
        metadata = json.loads(path.read_text(encoding="utf-8"))
        return DisplayPolicy(
            threshold=float(metadata["threshold_selection"]["threshold"]),
            uncertainty_band=float(metadata["calibration"]["uncertainty_band"]),
        )
    except (OSError, ValueError, KeyError, TypeError):
        return None


def band_for(threshold_from_api: float, policy: DisplayPolicy | None) -> float | None:
    """The band to draw, or None when the display file does not match the service."""
    if policy is None or abs(policy.threshold - threshold_from_api) > 1e-9:
        return None
    return policy.uncertainty_band
