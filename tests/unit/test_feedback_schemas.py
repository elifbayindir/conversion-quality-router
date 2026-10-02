from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from conversion_router.feedback.schemas import (
    AuditRecord,
    FeedbackError,
    FeedbackSubmission,
    HumanResponse,
    OverrideReason,
    build_audit_record,
    input_fingerprint,
)
from conversion_router.schemas import Decision, SessionFeaturesRequest

from .feedback_factories import SESSION, decision, prediction


def test_agree_record_keeps_the_full_lineage():
    record = build_audit_record(
        SESSION, prediction(), decision(),
        FeedbackSubmission(response=HumanResponse.AGREE),
        now=datetime(2026, 10, 1, 9, 0, tzinfo=UTC), event_id="fb_fixed",
    )
    assert record.event_id == "fb_fixed"
    assert record.created_at == "2026-10-01T09:00:00+00:00"
    assert record.request_id == "req_fb"
    assert record.input_fingerprint == input_fingerprint(SESSION)
    assert (record.purchase_probability, record.decision_threshold) == (0.1111, 0.12)
    assert record.uncertain is True
    assert record.agent_decision == Decision.HUMAN_REVIEW
    assert record.recommended_action.value == "ADD_TO_REVIEW_QUEUE"
    assert (record.model_version, record.agent_version, record.decision_schema_version) == (
        "1.0.0", "1.0.0", "1.0.0"
    )
    assert record.final_decision == Decision.HUMAN_REVIEW
    assert record.override_reason is None


def test_override_record_stores_reason_note_and_final_decision():
    record = build_audit_record(
        SESSION, prediction(), decision(),
        FeedbackSubmission(
            response=HumanResponse.OVERRIDE,
            override_decision=Decision.PRIORITY_REVIEW,
            override_reason=OverrideReason.UNCERTAINTY_RESOLVED_BY_REVIEW,
            note="  Repeat buyer pattern.  ",
        ),
    )
    assert record.final_decision == Decision.PRIORITY_REVIEW
    assert record.override_reason == OverrideReason.UNCERTAINTY_RESOLVED_BY_REVIEW
    assert record.reviewer_note == "Repeat buyer pattern."
    assert record.event_id.startswith("fb_")


def test_fingerprint_is_deterministic_and_input_sensitive():
    other = SESSION.model_copy(update={"ProductRelated": 11})
    assert input_fingerprint(SESSION) == input_fingerprint(SESSION)
    assert input_fingerprint(SESSION) != input_fingerprint(other)
    assert len(input_fingerprint(SESSION)) == 64


@pytest.mark.parametrize(
    "kwargs",
    [
        {"response": "AGREE", "override_decision": "LOG_ONLY"},
        {"response": "OVERRIDE"},
        {"response": "OVERRIDE", "override_decision": "LOG_ONLY"},
        {"response": "OVERRIDE", "override_decision": "SYSTEM_FALLBACK",
         "override_reason": "OTHER", "note": "x"},
        {"response": "OVERRIDE", "override_decision": "LOG_ONLY", "override_reason": "OTHER"},
        {"response": "AGREE", "note": "x" * 501},
        {"response": "AGREE", "unexpected": True},
    ],
)
def test_invalid_submissions_are_rejected(kwargs):
    with pytest.raises(ValidationError):
        FeedbackSubmission(**kwargs)


def test_override_to_the_same_decision_is_rejected():
    submission = FeedbackSubmission(
        response="OVERRIDE", override_decision="HUMAN_REVIEW",
        override_reason="CONTEXT_NOT_IN_FEATURES",
    )
    with pytest.raises(FeedbackError):
        build_audit_record(SESSION, prediction(), decision(), submission)


def test_agreeing_with_a_safety_fallback_is_rejected():
    with pytest.raises(FeedbackError):
        build_audit_record(
            SESSION, prediction(), decision(value="SYSTEM_FALLBACK"),
            FeedbackSubmission(response="AGREE"),
        )


def test_fallback_can_be_resolved_by_an_explicit_reviewer_decision():
    record = build_audit_record(
        SESSION, prediction(), decision(value="SYSTEM_FALLBACK"),
        FeedbackSubmission(response="OVERRIDE", override_decision="HUMAN_REVIEW",
                           override_reason="FALLBACK_RESOLVED_BY_REVIEW"),
    )
    assert record.agent_decision == Decision.SYSTEM_FALLBACK
    assert record.final_decision == Decision.HUMAN_REVIEW


def test_mismatched_prediction_and_decision_are_rejected():
    with pytest.raises(FeedbackError):
        build_audit_record(
            SESSION, prediction("req_a"), decision("req_b"),
            FeedbackSubmission(response="AGREE"),
        )


def test_audit_record_rejects_unknown_fields():
    record = build_audit_record(SESSION, prediction(), decision(),
                                FeedbackSubmission(response="AGREE"))
    payload = record.model_dump(mode="json") | {"raw_session": {}}
    with pytest.raises(ValidationError):
        AuditRecord.model_validate(payload)


def test_audit_record_does_not_contain_raw_feature_values():
    record = build_audit_record(SESSION, prediction(), decision(),
                                FeedbackSubmission(response="AGREE"))
    dumped = record.model_dump()
    for feature in SessionFeaturesRequest.model_fields:
        assert feature not in dumped
