"""Read-only aggregation of recorded feedback for later human policy review."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from conversion_router.feedback.schemas import AuditRecord, HumanResponse, OverrideReason
from conversion_router.schemas import Decision


@dataclass(frozen=True)
class RouteSummary:
    route: Decision
    reviewed: int
    agreed: int
    overridden: int

    @property
    def override_rate(self) -> float:
        return self.overridden / self.reviewed if self.reviewed else 0.0


@dataclass(frozen=True)
class FeedbackSummary:
    total_reviewed: int
    agreed: int
    overridden: int
    by_route: list[RouteSummary]
    override_reasons: list[tuple[OverrideReason, int]]
    transitions: list[tuple[Decision, Decision, int]]

    @property
    def agree_rate(self) -> float:
        return self.agreed / self.total_reviewed if self.total_reviewed else 0.0

    @property
    def override_rate(self) -> float:
        return self.overridden / self.total_reviewed if self.total_reviewed else 0.0


def summarize(records: list[AuditRecord]) -> FeedbackSummary:
    agreed = sum(r.human_response == HumanResponse.AGREE for r in records)
    by_route = []
    for route in Decision:
        group = [r for r in records if r.agent_decision == route]
        if group:
            route_agreed = sum(r.human_response == HumanResponse.AGREE for r in group)
            by_route.append(
                RouteSummary(route, len(group), route_agreed, len(group) - route_agreed)
            )
    reasons = Counter(r.override_reason for r in records if r.override_reason is not None)
    transitions = Counter(
        (r.agent_decision, r.final_decision)
        for r in records
        if r.human_response == HumanResponse.OVERRIDE
    )
    return FeedbackSummary(
        total_reviewed=len(records),
        agreed=agreed,
        overridden=len(records) - agreed,
        by_route=by_route,
        override_reasons=reasons.most_common(),
        transitions=[(src, dst, n) for (src, dst), n in transitions.most_common()],
    )
