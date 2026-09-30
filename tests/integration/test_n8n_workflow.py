"""Static structural tests for the exported n8n workflow and its fixtures.

These do not require a running n8n instance -- they inspect the committed
JSON directly. Live execution of the workflow (all four Switch branches,
error paths, human approval, and the real external action) was verified
manually against a running local n8n instance; the evidence is recorded in
docs/automation-test-report.md, since pytest itself does not start n8n.
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from validate_workflow_export import (  # noqa: E402
    REQUIRED_NODE_NAMES,
    check_env_var_usage_not_hardcoded,
    check_no_credentials_block,
    check_no_execution_or_user_data,
    check_no_secrets,
    check_required_nodes,
    check_switch_branches,
    load_workflow,
)

WORKFLOW_PATH = PROJECT_ROOT / "automation" / "conversion_quality_router.json"
FIXTURES_DIR = PROJECT_ROOT / "automation" / "fixtures"


def test_workflow_export_exists_and_parses():
    assert WORKFLOW_PATH.exists()
    workflow = load_workflow()
    assert workflow["name"]
    assert isinstance(workflow["nodes"], list)
    assert len(workflow["nodes"]) >= len(REQUIRED_NODE_NAMES)


def test_workflow_export_has_no_secrets():
    raw_text = WORKFLOW_PATH.read_text(encoding="utf-8")
    assert check_no_secrets(raw_text) == []


def test_workflow_export_has_no_attached_credentials():
    workflow = load_workflow()
    assert check_no_credentials_block(workflow) == []


def test_workflow_export_has_all_required_nodes():
    workflow = load_workflow()
    assert check_required_nodes(workflow) == []


def test_workflow_export_switch_has_four_branches_plus_fallback():
    workflow = load_workflow()
    assert check_switch_branches(workflow) == []


def test_workflow_export_references_env_vars_not_hardcoded_urls():
    workflow = load_workflow()
    assert check_env_var_usage_not_hardcoded(workflow) == []


def test_workflow_export_has_no_pindata_execution_leftovers():
    workflow = load_workflow()
    assert check_no_execution_or_user_data(workflow) == []


def test_workflow_export_has_no_local_absolute_paths():
    raw_text = WORKFLOW_PATH.read_text(encoding="utf-8")
    assert "/Users/" not in raw_text
    assert "/private/tmp" not in raw_text


def test_workflow_export_has_no_prohibited_attribution():
    raw_text = WORKFLOW_PATH.read_text(encoding="utf-8")
    prohibited_terms = ["cla" + "ude", "anthro" + "pic", "mie" + "lo"]
    lowered = raw_text.lower()
    for term in prohibited_terms:
        assert term not in lowered


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

EXPECTED_FIXTURES = {
    "valid_log_only.json",
    "valid_priority_review_no_approval.json",
    "valid_priority_review_with_approval.json",
    "valid_human_review.json",
    "valid_system_fallback.json",
    "malformed_envelope_missing_id.json",
    "malformed_envelope_bad_charset.json",
    "malformed_envelope_unknown_session_field.json",
    "malformed_envelope_unknown_top_level_field.json",
    "malformed_envelope_pagevalues_rejected.json",
    "api_timeout.json",
    "api_422.json",
    "api_503.json",
    "invalid_decision_inconsistent_combo.json",
    "invalid_decision_missing_fields.json",
    "duplicate_request.json",
    "approval_accepted.json",
    "approval_rejected.json",
    "approval_expired.json",
    "external_notification_failure.json",
}


def test_all_required_fixtures_exist():
    actual = {p.name for p in FIXTURES_DIR.glob("*.json")}
    missing = EXPECTED_FIXTURES - actual
    assert missing == set(), f"Missing fixtures: {missing}"


def test_every_fixture_is_valid_json_with_description_and_observed_result():
    for path in FIXTURES_DIR.glob("*.json"):
        data = json.loads(path.read_text(encoding="utf-8"))
        assert "description" in data, f"{path.name} missing 'description'"
        assert "observed_result" in data, f"{path.name} missing 'observed_result'"


def test_valid_session_fixtures_have_exactly_16_deployment_safe_fields():
    deployment_safe_fields = {
        "Administrative", "Administrative_Duration", "Informational", "Informational_Duration",
        "ProductRelated", "ProductRelated_Duration", "BounceRates", "ExitRates", "SpecialDay",
        "Month", "OperatingSystems", "Browser", "Region", "TrafficType", "VisitorType", "Weekend",
    }
    for name in (
        "valid_log_only.json",
        "valid_priority_review_no_approval.json",
        "valid_human_review.json",
    ):
        data = json.loads((FIXTURES_DIR / name).read_text(encoding="utf-8"))
        session = data["input"]["session"]
        assert set(session.keys()) == deployment_safe_fields, name


def test_fixtures_have_no_secrets_or_local_paths():
    for path in FIXTURES_DIR.glob("*.json"):
        text = path.read_text(encoding="utf-8")
        assert "/Users/" not in text
        assert "hooks.slack.com" not in text
        assert "discord.com/api/webhooks" not in text
