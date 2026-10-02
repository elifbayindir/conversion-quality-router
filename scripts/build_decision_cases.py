#!/usr/bin/env python3
"""Build two end-to-end decision cases from real validation-split sessions.

Each case follows the path the UI uses: the REAL FastAPI app (real frozen
model, real agent validation/retry/fallback chain) runs in-process with the
deterministic `RuleBasedFakeProvider` injected explicitly; the UI's own
`ApiClient` calls `/predict` then `/decide`; the real feedback module builds
the audit record and writes it to a temporary SQLite store, from which it is
read back. No network, no LLM call, no test-split row, no write to the real
feedback store or to any model artifact.

The reviewer actions are scripted to show both the agree and the override
path. They are illustrations of the audit loop, not evidence that reviewers
improve outcomes; the known validation labels are printed next to them.

Outputs:
    docs/decision_cases.json
    docs/decision-cases.md

Request IDs, event IDs, and timestamps are real but differ per run; `--check`
compares everything else.

Usage:
    venv/bin/python scripts/build_decision_cases.py
    venv/bin/python scripts/build_decision_cases.py --check
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient

from conversion_router.agent.demo_providers import RuleBasedFakeProvider
from conversion_router.api.app import create_app
from conversion_router.feedback.schemas import (
    FeedbackSubmission,
    HumanResponse,
    OverrideReason,
    build_audit_record,
)
from conversion_router.feedback.store import FeedbackStore
from conversion_router.schemas import Decision, SessionFeaturesRequest
from conversion_router.ui import features, presentation
from conversion_router.ui.api_client import ApiClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIXTURES_DIR = PROJECT_ROOT / "automation" / "fixtures"
VALIDATION_PREDICTIONS = PROJECT_ROOT / "data" / "processed" / "mlp_validation_predictions.csv"
OUT_JSON = PROJECT_ROOT / "docs" / "decision_cases.json"
OUT_MD = PROJECT_ROOT / "docs" / "decision-cases.md"

VOLATILE = re.compile(r"req_\d{14}_[0-9a-f]{8}|fb_[0-9a-f]{32}|\d{4}-\d{2}-\d{2}T[\d:]{8}\+00:00")

# Each fixture holds a real validation-split session; reviewer actions are scripted.
CASES = (
    {
        "key": "strong_signal",
        "title": "Strong signal routed to priority review, reviewer agrees",
        "fixture": "real_model_priority_review.json",
        "row_index": 6797,
        "expected_route": Decision.PRIORITY_REVIEW,
        "submission": FeedbackSubmission(response=HumanResponse.AGREE),
    },
    {
        "key": "uncertain_signal",
        "title": "Uncertain signal routed to human review, reviewer overrides",
        "fixture": "real_model_human_review.json",
        "row_index": 214,
        "expected_route": Decision.HUMAN_REVIEW,
        "submission": FeedbackSubmission(
            response=HumanResponse.OVERRIDE,
            override_decision=Decision.LOG_ONLY,
            override_reason=OverrideReason.SIGNAL_WEAKER_THAN_SCORED,
            note="Only 10 product pages and a high exit rate.",
        ),
    },
)

CONTEXT_FIELDS = (
    "VisitorType", "Month", "Weekend", "ProductRelated", "ProductRelated_Duration",
    "Informational", "Informational_Duration", "BounceRates", "ExitRates",
)


def _session_context(session) -> list[dict]:
    values = session.model_dump()
    rows = []
    for name in CONTEXT_FIELDS:
        spec = features.FIELD_BY_NAME[name]
        value = values[name]
        if name in features.CATEGORICAL_WHITELISTS:
            shown = features.option_label(name, value)
        elif isinstance(value, bool):
            shown = "Yes" if value else "No"
        else:
            shown = f"{value:g}"
        rows.append({"input": spec.label, "value": shown})
    return rows


def build_cases() -> dict:
    labels = pd.read_csv(VALIDATION_PREDICTIONS).set_index("row_index")["y_true"]
    built = []
    with tempfile.TemporaryDirectory() as tmp, TestClient(create_app()) as http:
        http.app.state.llm_provider = RuleBasedFakeProvider()
        client = ApiClient(base_url="http://testserver", http_client=http)
        store = FeedbackStore(Path(tmp) / "cases.sqlite3")
        for case in CASES:
            payload = json.loads((FIXTURES_DIR / case["fixture"]).read_text(encoding="utf-8"))
            session = SessionFeaturesRequest.model_validate(payload["input"]["session"])
            result = client.route(session)
            if result.decision.decision != case["expected_route"]:
                raise RuntimeError(
                    f"{case['key']}: expected {case['expected_route']}, "
                    f"got {result.decision.decision}"
                )
            record = build_audit_record(
                session, result.prediction, result.decision, case["submission"]
            )
            store.append(record)
            stored = FeedbackStore(store.path).get_by_request(result.request_id)
            if stored != record:
                raise RuntimeError(f"{case['key']}: audit record did not read back intact")
            detail = result.prediction.prediction
            model = result.prediction.model
            decision = result.decision
            built.append(
                {
                    "key": case["key"],
                    "title": case["title"],
                    "source": f"validation split, row_index {case['row_index']} "
                    f"(automation/fixtures/{case['fixture']})",
                    "session_context": _session_context(session),
                    "model_evidence": {
                        "purchase_probability": detail.purchase_probability,
                        "decision_threshold": detail.decision_threshold,
                        "predicted_class": detail.predicted_class.value,
                        "signals": [
                            presentation.SIGNAL_LABELS[s] for s in result.prediction.signals
                        ],
                        "model": f"{model.name} {model.version}",
                    },
                    "uncertainty": {
                        "uncertain": detail.uncertain,
                        "decision_margin": detail.decision_margin,
                        "text": presentation.uncertainty_text(
                            detail.uncertain, detail.decision_margin
                        ),
                    },
                    "agent_decision": {
                        "decision": decision.decision.value,
                        "priority": decision.priority.value,
                        "reason_codes": [r.value for r in decision.reason_codes],
                        "explanation": decision.explanation,
                        "agent_version": decision.agent_version,
                        "schema_version": decision.schema_version,
                    },
                    "operational_action": {
                        "allowed_action": decision.allowed_action.value,
                        "label": presentation.ACTION_LABELS[decision.allowed_action],
                        "requires_human_approval": decision.requires_human_approval,
                    },
                    "human_feedback": {
                        "response": record.human_response.value,
                        "override_reason": record.override_reason.value
                        if record.override_reason else None,
                        "note": record.reviewer_note,
                        "final_decision": record.final_decision.value,
                        "scripted": True,
                    },
                    "audit_record": record.model_dump(mode="json"),
                    "known_outcome": {
                        "converted": bool(labels.loc[case["row_index"]]),
                        "note": "Validation label, unavailable at decision time; shown for "
                        "transparency. The scripted reviewer action is not a claim of accuracy.",
                    },
                }
            )
    return {
        "generated_by": "scripts/build_decision_cases.py",
        "decision_provider": "RuleBasedFakeProvider via the real agent chain (no LLM call)",
        "feedback_store": "temporary SQLite file (the real local store is not touched)",
        "cases": built,
    }


def _flow(case: dict) -> list[str]:
    me, unc, ad = case["model_evidence"], case["uncertainty"], case["agent_decision"]
    op, hf, ko = case["operational_action"], case["human_feedback"], case["known_outcome"]
    ar = case["audit_record"]
    context = "; ".join(f"{row['input']}: {row['value']}" for row in case["session_context"])
    signals = ", ".join(me["signals"]) or "none"
    feedback = (
        "Agree with recommendation"
        if hf["response"] == HumanResponse.AGREE
        else f"Override to {presentation.ROUTE_DISPLAY[Decision(hf['final_decision'])].title} "
        f"(reason: {presentation.OVERRIDE_REASON_LABELS[OverrideReason(hf['override_reason'])]}; "
        f"note: \"{hf['note']}\")"
    )
    return [
        f"## {case['title']}",
        "",
        f"Source: {case['source']}.",
        "",
        f"1. **Session context** — {context}.",
        f"2. **Model evidence** — calibrated purchase probability "
        f"{presentation.format_probability(me['purchase_probability'])} against a decision "
        f"threshold of {presentation.format_probability(me['decision_threshold'])} "
        f"({me['model']}). Supporting signals: {signals}.",
        f"3. **Uncertainty** — {unc['text']}",
        f"4. **Agent decision** — `{ad['decision']}`, priority `{ad['priority']}`, reasons "
        f"`{', '.join(ad['reason_codes'])}`: \"{ad['explanation']}\" "
        f"(agent {ad['agent_version']}, schema {ad['schema_version']}).",
        f"5. **Operational action** — {op['label']} (`{op['allowed_action']}`); human approval "
        f"{'required' if op['requires_human_approval'] else 'not required'}.",
        f"6. **Human feedback** (scripted) — {feedback}.",
        f"7. **Audit record** — event `{ar['event_id']}` for request `{ar['request_id']}`, "
        f"recorded {ar['created_at']}; input fingerprint `{ar['input_fingerprint'][:16]}…`; "
        f"final decision `{ar['final_decision']}`; read back intact from the append-only store.",
        "",
        f"Known outcome: the session {'converted' if ko['converted'] else 'did not convert'}. "
        f"{ko['note']}",
        "",
    ]


def render_markdown(results: dict) -> str:
    lines = [
        "# Decision Cases",
        "",
        "Generated by `scripts/build_decision_cases.py`; do not edit by hand. Full records "
        "are in `docs/decision_cases.json`. See `docs/decision-loop.md` for how the loop "
        "works.",
        "",
        "Both sessions are real validation-split rows (never the test split). Each one went "
        "through the real model and the real decision-agent chain with the deterministic "
        "demo provider (no language-model call), then through the real feedback module. "
        "Reviewer actions are scripted to show the agree and override paths; they are "
        "illustrations, not evidence that reviewers improve outcomes. Request IDs, event "
        "IDs, and timestamps change on every run.",
        "",
    ]
    for case in results["cases"]:
        lines += _flow(case)
    return "\n".join(lines)


def _stable(text: str) -> str:
    return VOLATILE.sub("<volatile>", text)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    results = build_cases()
    outputs = {OUT_JSON: json.dumps(results, indent=2) + "\n", OUT_MD: render_markdown(results)}
    if args.check:
        stale = [
            str(path.relative_to(PROJECT_ROOT))
            for path, text in outputs.items()
            if not path.exists() or _stable(path.read_text(encoding="utf-8")) != _stable(text)
        ]
        print("STALE: " + ", ".join(stale) if stale else "Decision case outputs are current.")
        return 1 if stale else 0
    for path, text in outputs.items():
        path.write_text(text, encoding="utf-8")
        print(f"Wrote {path.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
