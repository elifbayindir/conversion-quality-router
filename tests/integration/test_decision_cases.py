"""The two worked decision cases reproduce end to end (real model, real agent
chain with the deterministic demo provider, real feedback module) and the
committed case documents match a fresh build apart from per-run IDs."""

import json
import sys
from pathlib import Path

import pytest

from conversion_router.modeling.inference import DEFAULT_METADATA_PATH

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
import build_decision_cases as bdc  # noqa: E402

pytestmark = pytest.mark.skipif(
    not DEFAULT_METADATA_PATH.exists() or not bdc.VALIDATION_PREDICTIONS.exists(),
    reason="Frozen artifacts or validation predictions not present.",
)


@pytest.fixture(scope="module")
def built():
    return bdc.build_cases()


def test_cases_follow_the_expected_routes_and_feedback(built):
    strong, uncertain = built["cases"]
    assert strong["agent_decision"]["decision"] == "PRIORITY_REVIEW"
    assert strong["uncertainty"]["uncertain"] is False
    assert strong["human_feedback"]["response"] == "AGREE"
    assert strong["audit_record"]["final_decision"] == "PRIORITY_REVIEW"

    assert uncertain["agent_decision"]["decision"] == "HUMAN_REVIEW"
    assert uncertain["uncertainty"]["uncertain"] is True
    assert uncertain["operational_action"]["requires_human_approval"] is True
    assert uncertain["human_feedback"]["response"] == "OVERRIDE"
    assert uncertain["audit_record"]["final_decision"] == "LOG_ONLY"


def test_every_case_has_all_flow_sections_and_is_marked_scripted(built):
    sections = ("session_context", "model_evidence", "uncertainty", "agent_decision",
                "operational_action", "human_feedback", "audit_record", "known_outcome")
    for case in built["cases"]:
        for section in sections:
            assert case[section] not in (None, {}, [])
        assert case["human_feedback"]["scripted"] is True
        assert "validation split" in case["source"]
        record = case["audit_record"]
        assert record["request_id"].startswith("req_")
        assert record["agent_version"] and record["model_version"]


def test_committed_case_documents_are_current(built):
    committed_md = bdc.OUT_MD.read_text(encoding="utf-8")
    assert bdc._stable(committed_md) == bdc._stable(bdc.render_markdown(built))
    committed = json.loads(bdc.OUT_JSON.read_text(encoding="utf-8"))
    assert bdc._stable(json.dumps(committed, indent=2)) == bdc._stable(
        json.dumps(built, indent=2)
    )


def test_building_cases_never_touches_the_real_feedback_store(built):
    # The builder only ever uses a temporary store.
    source = (PROJECT_ROOT / "scripts" / "build_decision_cases.py").read_text(encoding="utf-8")
    assert "TemporaryDirectory" in source
    assert "DEFAULT_DB_PATH" not in source and "from_env" not in source
