from conversion_router.feedback.schemas import (
    FeedbackSubmission,
    OverrideReason,
    build_audit_record,
)
from conversion_router.feedback.summary import summarize
from conversion_router.schemas import Decision

from .feedback_factories import SESSION, decision, prediction


def _record(request_id, route, response="AGREE", **override):
    return build_audit_record(
        SESSION, prediction(request_id), decision(request_id, route),
        FeedbackSubmission(response=response, **override),
    )


def test_empty_summary():
    summary = summarize([])
    assert summary.total_reviewed == 0
    assert summary.agree_rate == summary.override_rate == 0.0
    assert summary.by_route == []


def test_counts_rates_routes_reasons_and_transitions():
    records = [
        _record("r1", "HUMAN_REVIEW"),
        _record("r2", "HUMAN_REVIEW", "OVERRIDE", override_decision="PRIORITY_REVIEW",
                override_reason="UNCERTAINTY_RESOLVED_BY_REVIEW"),
        _record("r3", "HUMAN_REVIEW", "OVERRIDE", override_decision="LOG_ONLY",
                override_reason="UNCERTAINTY_RESOLVED_BY_REVIEW"),
        _record("r4", "PRIORITY_REVIEW"),
        _record("r5", "SYSTEM_FALLBACK", "OVERRIDE", override_decision="HUMAN_REVIEW",
                override_reason="FALLBACK_RESOLVED_BY_REVIEW"),
    ]
    summary = summarize(records)

    assert (summary.total_reviewed, summary.agreed, summary.overridden) == (5, 2, 3)
    assert summary.agree_rate == 0.4 and summary.override_rate == 0.6
    by_route = {r.route: r for r in summary.by_route}
    assert by_route[Decision.HUMAN_REVIEW].reviewed == 3
    assert by_route[Decision.HUMAN_REVIEW].overridden == 2
    assert round(by_route[Decision.HUMAN_REVIEW].override_rate, 4) == 0.6667
    assert by_route[Decision.PRIORITY_REVIEW].override_rate == 0.0
    assert Decision.LOG_ONLY not in by_route
    assert summary.override_reasons[0] == (OverrideReason.UNCERTAINTY_RESOLVED_BY_REVIEW, 2)
    assert (Decision.SYSTEM_FALLBACK, Decision.HUMAN_REVIEW, 1) in summary.transitions


def test_summary_is_read_only():
    records = [_record("r1", "HUMAN_REVIEW")]
    snapshot = [r.model_copy() for r in records]
    summarize(records)
    assert records == snapshot
