"""Tests for the human-readable notification format in the n8n workflow.

These tests inspect the workflow JSON statically — no running n8n required.
They verify that notification payloads use human-readable templates and do
not expose raw enum duplicates, full request IDs, resume URLs in text, or
secrets.
"""

import json
import re
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW_PATH = PROJECT_ROOT / "automation" / "conversion_quality_router.json"


@pytest.fixture()
def workflow():
    return json.loads(WORKFLOW_PATH.read_text(encoding="utf-8"))


def _get_node(workflow, name):
    for node in workflow["nodes"]:
        if node["name"] == name:
            return node
    raise KeyError(f"Node {name!r} not found")


def _get_json_body(workflow, node_name):
    node = _get_node(workflow, node_name)
    return node["parameters"]["jsonBody"]


class TestPriorityNotification:
    def test_has_human_readable_title(self, workflow):
        body = _get_json_body(workflow, "Send Priority Notification")
        assert "Priority review recommended" in body

    def test_has_priority_label(self, workflow):
        body = _get_json_body(workflow, "Send Priority Notification")
        assert "*Priority:*" in body

    def test_has_action_label(self, workflow):
        body = _get_json_body(workflow, "Send Priority Notification")
        assert "*Recommended action:*" in body

    def test_has_approval_status(self, workflow):
        body = _get_json_body(workflow, "Send Priority Notification")
        assert "*Human approval:*" in body

    def test_has_reason_label(self, workflow):
        body = _get_json_body(workflow, "Send Priority Notification")
        assert "*Reason:*" in body

    def test_has_short_reference(self, workflow):
        body = _get_json_body(workflow, "Send Priority Notification")
        assert "*Case reference:*" in body
        assert "slice(-8)" in body

    def test_has_context_line(self, workflow):
        body = _get_json_body(workflow, "Send Priority Notification")
        assert "Slack delivers the alert" in body
        assert "does not make the decision" in body

    def test_no_raw_duplicate_format(self, workflow):
        body = _get_json_body(workflow, "Send Priority Notification")
        assert '("[" + "PRIORITY_REVIEW" + "]"' not in body

    def test_no_full_request_id_in_text(self, workflow):
        body = _get_json_body(workflow, "Send Priority Notification")
        text_section = body.split('"workflow_request_id"')[0]
        for ref in re.finditer(
            r"\$json\.workflow_request_id", text_section,
        ):
            context = text_section[ref.start():ref.start() + 80]
            assert "slice(-8)" in context, (
                "Full workflow_request_id rendered without truncation"
            )

    def test_no_resume_url_in_text(self, workflow):
        body = _get_json_body(workflow, "Send Priority Notification")
        text_section = body.split('"workflow_request_id"')[0]
        assert "resumeUrl" not in text_section
        assert "approval_resume_url" not in text_section


class TestHumanReviewNotification:
    def test_has_human_readable_title(self, workflow):
        body = _get_json_body(workflow, "Send Review Notification")
        assert "Human review required" in body

    def test_approval_is_required(self, workflow):
        body = _get_json_body(workflow, "Send Review Notification")
        assert "Human approval:* Required" in body

    def test_no_raw_duplicate_format(self, workflow):
        body = _get_json_body(workflow, "Send Review Notification")
        assert '("[" + "HUMAN_REVIEW" + "]"' not in body

    def test_has_context_line(self, workflow):
        body = _get_json_body(workflow, "Send Review Notification")
        assert "does not make the decision" in body


class TestFallbackNotification:
    def test_has_human_readable_title(self, workflow):
        body = _get_json_body(workflow, "Send Fallback Notification")
        assert "Safety fallback" in body

    def test_describes_fail_to_human(self, workflow):
        body = _get_json_body(workflow, "Send Fallback Notification")
        assert "could not be validated" in body
        assert "routed to a reviewer" in body

    def test_no_raw_error_leakage(self, workflow):
        body = _get_json_body(workflow, "Send Fallback Notification")
        assert "stack" not in body.lower()
        assert "traceback" not in body.lower()
        assert "exception" not in body.lower()

    def test_no_raw_duplicate_format(self, workflow):
        body = _get_json_body(workflow, "Send Fallback Notification")
        assert '("[" + "SYSTEM_FALLBACK" + "]"' not in body

    def test_approval_is_required(self, workflow):
        body = _get_json_body(workflow, "Send Fallback Notification")
        assert "Human approval:* Required" in body


class TestFinalOutcomeNotification:
    def test_has_human_readable_title(self, workflow):
        body = _get_json_body(workflow, "Send Final Outcome Notification")
        assert "Review outcome recorded" in body

    def test_distinguishes_approved(self, workflow):
        body = _get_json_body(workflow, "Send Final Outcome Notification")
        assert "Approved" in body

    def test_distinguishes_rejected(self, workflow):
        body = _get_json_body(workflow, "Send Final Outcome Notification")
        assert "Rejected" in body

    def test_distinguishes_expired(self, workflow):
        body = _get_json_body(workflow, "Send Final Outcome Notification")
        assert "Expired" in body

    def test_has_original_route(self, workflow):
        body = _get_json_body(workflow, "Send Final Outcome Notification")
        assert "*Original route:*" in body

    def test_has_short_reference(self, workflow):
        body = _get_json_body(workflow, "Send Final Outcome Notification")
        assert "*Case reference:*" in body
        assert "slice(-8)" in body

    def test_no_full_request_id_in_text(self, workflow):
        body = _get_json_body(workflow, "Send Final Outcome Notification")
        text_section = body.split('"workflow_request_id"')[0]
        for ref in re.finditer(
            r"\$json\.workflow_request_id", text_section,
        ):
            context = text_section[ref.start():ref.start() + 80]
            assert "slice(-8)" in context, (
                "Full workflow_request_id rendered without truncation"
            )

    def test_no_resume_url_in_text_or_payload(self, workflow):
        body = _get_json_body(workflow, "Send Final Outcome Notification")
        assert "resumeUrl" not in body
        assert "approval_resume_url" not in body

    def test_no_raw_format(self, workflow):
        body = _get_json_body(workflow, "Send Final Outcome Notification")
        assert "Approval outcome for" not in body


class TestLogOnlySendsNoNotification:
    def test_log_only_branch_has_no_send_node(self, workflow):
        connections = workflow.get("connections", {})
        log_path_nodes = {"Set Log Only Complete", "Respond Log Only"}
        send_nodes = {
            "Send Priority Notification",
            "Send Review Notification",
            "Send Fallback Notification",
            "Send Final Outcome Notification",
        }
        for log_node in log_path_nodes:
            downstream = set()
            if log_node in connections:
                for outputs in connections[log_node].values():
                    for conns in outputs:
                        for conn in conns:
                            downstream.add(conn["node"])
            assert downstream.isdisjoint(send_nodes), (
                f"LOG_ONLY path node {log_node} connects to a send node"
            )


class TestResumeUrlBoundary:
    def test_initial_notifications_have_resume_url_as_machine_field(
        self, workflow,
    ):
        for name in (
            "Send Priority Notification",
            "Send Review Notification",
            "Send Fallback Notification",
        ):
            body = _get_json_body(workflow, name)
            assert "approval_resume_url" in body
            assert "$execution.resumeUrl" in body

    def test_final_outcome_has_no_resume_url(self, workflow):
        body = _get_json_body(
            workflow, "Send Final Outcome Notification",
        )
        assert "approval_resume_url" not in body
        assert "resumeUrl" not in body

    def test_resume_url_not_in_rendered_text(self, workflow):
        for name in (
            "Send Priority Notification",
            "Send Review Notification",
            "Send Fallback Notification",
        ):
            body = _get_json_body(workflow, name)
            text_end = body.index('"workflow_request_id"')
            text_section = body[:text_end]
            assert "resumeUrl" not in text_section
            assert "approval_resume_url" not in text_section


class TestNoSecretsInNotifications:
    def test_no_webhook_url_in_text(self, workflow):
        for name in (
            "Send Priority Notification",
            "Send Review Notification",
            "Send Fallback Notification",
            "Send Final Outcome Notification",
        ):
            body = _get_json_body(workflow, name)
            assert "hooks.slack.com" not in body
            assert "N8N_NOTIFY_WEBHOOK_URL" not in body.split('"url"')[0]

    def test_slack_not_described_as_making_decision(self, workflow):
        for name in (
            "Send Priority Notification",
            "Send Review Notification",
            "Send Fallback Notification",
        ):
            body = _get_json_body(workflow, name)
            assert "does not make the decision" in body
