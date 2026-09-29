"""Provider-neutral LLM client and decision-agent orchestration.

Only this module talks to an LLM provider. It sends exactly one thing to
the model: the system prompt from `prompts/decision_agent_v1.md` and the
JSON-serialized, already-validated `PredictionResponse` -- never raw or
free-text user input. `get_decision()` runs the full policy: call, schema
validation, cross-field validation, one bounded retry on any failure, then
a deterministic (non-LLM) fallback. It always returns a valid
`DecisionResponse`; it never raises out to its caller.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Protocol

import httpx
from dotenv import load_dotenv
from pydantic import ValidationError

from conversion_router.agent.fallback import AGENT_VERSION, build_fallback_decision
from conversion_router.agent.validator import DecisionValidationError, validate_decision_core
from conversion_router.schemas import (
    Decision,
    DecisionResponse,
    LLMDecisionCore,
    PredictionResponse,
    ReasonCode,
)

logger = logging.getLogger("conversion_router.agent")

PROJECT_ROOT = Path(__file__).resolve().parents[3]
PROMPT_PATH = PROJECT_ROOT / "prompts" / "decision_agent_v1.md"

ANTHROPIC_MODEL = "claude-haiku-4-5-20251001"
ANTHROPIC_ENDPOINT = "https://api.anthropic.com/v1/messages"
ANTHROPIC_API_VERSION = "2023-06-01"
MAX_TOKENS = 300
TEMPERATURE = 0
CONNECT_TIMEOUT_SECONDS = 5.0
READ_TIMEOUT_SECONDS = 15.0
MAX_ATTEMPTS = 2  # one initial call + at most one controlled retry

_dotenv_loaded = False


def _ensure_dotenv_loaded() -> None:
    global _dotenv_loaded
    if not _dotenv_loaded:
        load_dotenv(PROJECT_ROOT / ".env")
        _dotenv_loaded = True


class AgentTransportError(RuntimeError):
    """Network, timeout, or HTTP-status failure calling the provider."""


class AgentOutputError(RuntimeError):
    """Provider responded, but content was empty, a refusal, or unusable."""


class AgentCredentialError(RuntimeError):
    """ANTHROPIC_API_KEY is not set."""


def load_system_prompt() -> str:
    """Extract the fenced 'System prompt (verbatim)' block from the prompt file."""
    text = PROMPT_PATH.read_text(encoding="utf-8")
    marker = "## System prompt (verbatim)\n\n```\n"
    start = text.index(marker) + len(marker)
    end = text.index("\n```", start)
    return text[start:end]


def build_user_content(prediction: PredictionResponse) -> str:
    """The only content sent as the user message: the validated prediction, nothing else."""
    return prediction.model_dump_json()


def decision_core_json_schema() -> dict:
    return LLMDecisionCore.model_json_schema()


class LLMProvider(Protocol):
    def complete_decision(self, system_prompt: str, user_content: str) -> str:
        """Return the raw text content of the model's response (expected to be JSON)."""
        ...


class AnthropicProvider:
    """httpx-based client for the Anthropic Messages API."""

    def __init__(self, api_key: str | None = None, client: httpx.Client | None = None) -> None:
        _ensure_dotenv_loaded()
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self._api_key:
            raise AgentCredentialError("ANTHROPIC_API_KEY is not set.")
        self._client = client or httpx.Client(
            timeout=httpx.Timeout(
                connect=CONNECT_TIMEOUT_SECONDS,
                read=READ_TIMEOUT_SECONDS,
                write=READ_TIMEOUT_SECONDS,
                pool=CONNECT_TIMEOUT_SECONDS,
            )
        )

    def complete_decision(self, system_prompt: str, user_content: str) -> str:
        body = {
            "model": ANTHROPIC_MODEL,
            "max_tokens": MAX_TOKENS,
            "temperature": TEMPERATURE,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_content}],
            "output_config": {
                "format": {
                    "type": "json_schema",
                    "schema": decision_core_json_schema(),
                }
            },
        }
        headers = {
            "x-api-key": self._api_key,
            "anthropic-version": ANTHROPIC_API_VERSION,
            "content-type": "application/json",
        }
        try:
            response = self._client.post(ANTHROPIC_ENDPOINT, json=body, headers=headers)
        except httpx.TimeoutException as exc:
            raise AgentTransportError(f"timeout calling Anthropic API: {exc}") from exc
        except httpx.HTTPError as exc:
            raise AgentTransportError(f"transport error calling Anthropic API: {exc}") from exc

        if response.status_code == 429:
            raise AgentTransportError("Anthropic API rate limited (429)")
        if response.status_code >= 500:
            raise AgentTransportError(f"Anthropic API server error ({response.status_code})")
        if response.status_code != 200:
            raise AgentTransportError(f"Anthropic API returned status {response.status_code}")

        payload = response.json()
        if payload.get("stop_reason") == "refusal":
            raise AgentOutputError("Anthropic API refused to respond")

        content_blocks = payload.get("content") or []
        text = "".join(
            block.get("text", "") for block in content_blocks if block.get("type") == "text"
        ).strip()
        if not text:
            raise AgentOutputError("Anthropic API returned empty content")
        return text


def get_decision(
    prediction: PredictionResponse,
    request_id: str,
    provider: LLMProvider | None = None,
) -> DecisionResponse:
    """Run the full agent policy. Always returns a valid DecisionResponse; never raises."""
    try:
        active_provider = provider or AnthropicProvider()
    except AgentCredentialError as exc:
        logger.warning("Agent unavailable: %s", exc)
        return build_fallback_decision(request_id, reason=ReasonCode.AGENT_UNAVAILABLE)

    system_prompt = load_system_prompt()
    user_content = build_user_content(prediction)

    last_reason = ReasonCode.AGENT_UNAVAILABLE
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            raw_text = active_provider.complete_decision(system_prompt, user_content)
        except AgentTransportError as exc:
            logger.warning(
                "Agent transport failure (attempt %d/%d): %s", attempt, MAX_ATTEMPTS, exc
            )
            last_reason = ReasonCode.AGENT_UNAVAILABLE
            continue
        except AgentOutputError as exc:
            logger.warning("Agent output failure (attempt %d/%d): %s", attempt, MAX_ATTEMPTS, exc)
            last_reason = ReasonCode.AGENT_OUTPUT_INVALID
            continue

        try:
            parsed = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            logger.warning(
                "Agent returned invalid JSON (attempt %d/%d): %s", attempt, MAX_ATTEMPTS, exc
            )
            last_reason = ReasonCode.AGENT_OUTPUT_INVALID
            continue

        try:
            core = LLMDecisionCore.model_validate(parsed)
        except ValidationError as exc:
            logger.warning(
                "Agent output failed schema validation (attempt %d/%d): %s",
                attempt, MAX_ATTEMPTS, exc,
            )
            last_reason = ReasonCode.AGENT_OUTPUT_INVALID
            continue

        try:
            validate_decision_core(core, prediction)
        except DecisionValidationError as exc:
            logger.warning(
                "Agent output failed cross-field validation (attempt %d/%d): %s",
                attempt, MAX_ATTEMPTS, exc,
            )
            last_reason = ReasonCode.AGENT_OUTPUT_INVALID
            continue

        return DecisionResponse(
            request_id=request_id,
            decision=Decision(core.decision.value),
            priority=core.priority,
            allowed_action=core.allowed_action,
            requires_human_approval=core.requires_human_approval,
            reason_codes=[ReasonCode(rc.value) for rc in core.reason_codes],
            explanation=core.explanation,
            agent_version=AGENT_VERSION,
        )

    return build_fallback_decision(request_id, reason=last_reason)
