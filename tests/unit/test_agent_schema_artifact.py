"""Verify prompts/decision_agent_schema_v1.json matches LLMDecisionCore at runtime."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from conversion_router.schemas import LLMDecisionCore

SCHEMA_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "prompts"
    / "decision_agent_schema_v1.json"
)


@pytest.fixture()
def committed_schema() -> dict:
    assert SCHEMA_PATH.exists(), f"{SCHEMA_PATH} not found"
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


@pytest.fixture()
def runtime_schema() -> dict:
    return LLMDecisionCore.model_json_schema()


def test_file_exists():
    assert SCHEMA_PATH.exists()


def test_file_parses_as_json():
    json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


def test_matches_runtime(committed_schema, runtime_schema):
    assert committed_schema == runtime_schema


def test_decision_enum_is_closed(committed_schema):
    decision_def = committed_schema["$defs"]["LLMDecision"]
    allowed = set(decision_def["enum"])
    assert allowed == {"LOG_ONLY", "PRIORITY_REVIEW", "HUMAN_REVIEW"}


def test_system_fallback_excluded(committed_schema):
    decision_def = committed_schema["$defs"]["LLMDecision"]
    assert "SYSTEM_FALLBACK" not in decision_def["enum"]
