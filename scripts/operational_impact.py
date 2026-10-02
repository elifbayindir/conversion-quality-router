#!/usr/bin/env python3
"""Operational impact of the routing policy on frozen predictions.

Reads the FROZEN test predictions written once by `scripts/evaluate_test.py`
(`data/processed/test_predictions.csv`), read-only. The model is not run, no
test artifact is written, and nothing here selects a model, threshold, band,
or policy: those are fixed in `artifacts/metadata/mlp_metadata.json` and
`prompts/decision_agent_v1.md`. If the frozen test predictions are absent, the
validation predictions are used instead and every output is labelled
"validation operational analysis".

Each session's route comes from the existing agent chain
(`conversion_router.agent.client.get_decision`, including schema and
cross-field validation and the deterministic fallback) with the deterministic
`RuleBasedFakeProvider`, which reproduces the written policy without an LLM
call. The policy is not re-implemented here.

Outputs (deterministic, no timestamps):
    docs/operational_impact_results.json
    docs/operational-impact.md
    docs/figures/operational_route_shares.png

Usage:
    venv/bin/python scripts/operational_impact.py          # write outputs
    venv/bin/python scripts/operational_impact.py --check  # verify outputs are current
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

from conversion_router.agent.client import PROMPT_PATH, get_decision
from conversion_router.agent.demo_providers import RuleBasedFakeProvider
from conversion_router.agent.fallback import AGENT_VERSION
from conversion_router.modeling.calibrate import decision_margin, is_uncertain
from conversion_router.schemas import (
    Decision,
    ModelInfo,
    PredictedClass,
    PredictionDetail,
    PredictionResponse,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEST_PREDICTIONS = PROJECT_ROOT / "data" / "processed" / "test_predictions.csv"
TEST_EVALUATION = PROJECT_ROOT / "artifacts" / "metadata" / "test_evaluation.json"
VALIDATION_PREDICTIONS = PROJECT_ROOT / "data" / "processed" / "mlp_validation_predictions.csv"
MODEL_METADATA = PROJECT_ROOT / "artifacts" / "metadata" / "mlp_metadata.json"

OUT_JSON = PROJECT_ROOT / "docs" / "operational_impact_results.json"
OUT_MD = PROJECT_ROOT / "docs" / "operational-impact.md"
OUT_FIGURE = PROJECT_ROOT / "docs" / "figures" / "operational_route_shares.png"

ANALYSIS_FROZEN_TEST = "frozen_test_operational_analysis"
ANALYSIS_VALIDATION = "validation_operational_analysis"

ROUTE_ORDER = (
    Decision.PRIORITY_REVIEW,
    Decision.HUMAN_REVIEW,
    Decision.SYSTEM_FALLBACK,
    Decision.LOG_ONLY,
)
REVIEW_ROUTES = (Decision.PRIORITY_REVIEW, Decision.HUMAN_REVIEW, Decision.SYSTEM_FALLBACK)
HUMAN_ROUTES = (Decision.HUMAN_REVIEW, Decision.SYSTEM_FALLBACK)
ROUTE_LABELS = {
    Decision.PRIORITY_REVIEW: "Priority review",
    Decision.HUMAN_REVIEW: "Human review",
    Decision.SYSTEM_FALLBACK: "Safety fallback",
    Decision.LOG_ONLY: "Logged automatically",
}


class SourceIntegrityError(RuntimeError):
    """The frozen predictions do not match the frozen evaluation record."""


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _relative(path: Path) -> str:
    return str(path.relative_to(PROJECT_ROOT))


def load_source(metadata: dict) -> tuple[pd.DataFrame, dict]:
    """Return one row per session with y_true, probability, predicted flag,
    and the frozen uncertainty flag, plus a provenance record."""
    threshold = metadata["threshold_selection"]["threshold"]
    band = metadata["calibration"]["uncertainty_band"]

    if TEST_PREDICTIONS.exists():
        frame = pd.read_csv(TEST_PREDICTIONS)
        evaluation = json.loads(TEST_EVALUATION.read_text(encoding="utf-8"))
        cm = evaluation["mlp"]["metrics"]["confusion_matrix"]
        positives = frame["y_true"] == 1
        checks = {
            "row_count_matches_evaluation": len(frame) == evaluation["n_test_rows"],
            "true_positives_match_evaluation": int((frame["mlp_pred"] & positives).sum())
            == cm["true_positive"],
            "false_positives_match_evaluation": int((frame["mlp_pred"] & ~positives).sum())
            == cm["false_positive"],
            "uncertain_count_matches_evaluation": int(frame["mlp_uncertain"].sum())
            == evaluation["mlp"]["n_uncertain"],
            "threshold_matches_evaluation": evaluation["mlp"]["metrics"]["threshold"]
            == threshold,
            "uncertain_flags_match_frozen_rule": bool(
                (is_uncertain(frame["mlp_calibrated_proba"].to_numpy(), threshold, band)
                 == frame["mlp_uncertain"].to_numpy()).all()
            ),
        }
        if not all(checks.values()):
            failed = [name for name, ok in checks.items() if not ok]
            raise SourceIntegrityError(f"Frozen test predictions failed checks: {failed}")
        sessions = pd.DataFrame(
            {
                "row_index": frame["row_index"],
                "y_true": frame["y_true"].astype(int),
                "probability": frame["mlp_calibrated_proba"],
                "predicted_positive": frame["mlp_pred"].astype(bool),
                "uncertain": frame["mlp_uncertain"].astype(bool),
            }
        )
        source = {
            "analysis_type": ANALYSIS_FROZEN_TEST,
            "split": "test (frozen single-pass predictions, read-only)",
            "predictions_file": _relative(TEST_PREDICTIONS),
            "predictions_sha256": sha256_of(TEST_PREDICTIONS),
            "evaluation_file": _relative(TEST_EVALUATION),
            "evaluation_sha256": sha256_of(TEST_EVALUATION),
            "integrity_checks": checks,
        }
    else:
        frame = pd.read_csv(VALIDATION_PREDICTIONS)
        proba = frame["calibrated_proba"].to_numpy()
        sessions = pd.DataFrame(
            {
                "row_index": frame["row_index"],
                "y_true": frame["y_true"].astype(int),
                "probability": frame["calibrated_proba"],
                "predicted_positive": frame["calibrated_proba"] >= threshold,
                "uncertain": is_uncertain(proba, threshold, band),
            }
        )
        source = {
            "analysis_type": ANALYSIS_VALIDATION,
            "split": "validation (not final/test evidence)",
            "predictions_file": _relative(VALIDATION_PREDICTIONS),
            "predictions_sha256": sha256_of(VALIDATION_PREDICTIONS),
            "integrity_checks": {},
        }
    return sessions, source


def route_sessions(sessions: pd.DataFrame, metadata: dict) -> list[Decision]:
    """Route each frozen prediction through the existing agent chain."""
    threshold = metadata["threshold_selection"]["threshold"]
    model = ModelInfo(
        name=metadata["model_name"],
        version=metadata["model_version"],
        feature_contract_version=metadata["feature_contract_version"],
    )
    margins = decision_margin(sessions["probability"].to_numpy(), threshold)
    provider = RuleBasedFakeProvider()
    routes = []
    for row, margin in zip(sessions.itertuples(index=False), margins, strict=True):
        request_id = f"offline_{row.row_index}"
        prediction = PredictionResponse(
            request_id=request_id,
            prediction=PredictionDetail(
                purchase_probability=round(float(row.probability), 4),
                decision_threshold=threshold,
                predicted_class=PredictedClass.LIKELY_TO_CONVERT
                if row.predicted_positive
                else PredictedClass.UNLIKELY_TO_CONVERT,
                decision_margin=round(float(margin), 4),
                uncertain=bool(row.uncertain),
            ),
            signals=[],
            model=model,
        )
        routes.append(get_decision(prediction, request_id, provider=provider).decision)
    return routes


def _ratio(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 4) if denominator else 0.0


def compute_metrics(sessions: pd.DataFrame, routes: list[Decision]) -> dict:
    frame = sessions.assign(route=[r.value for r in routes])
    n_sessions = len(frame)
    n_conversions = int(frame["y_true"].sum())

    per_route = []
    for route in ROUTE_ORDER:
        group = frame[frame["route"] == route.value]
        conversions = int(group["y_true"].sum())
        per_route.append(
            {
                "route": route.value,
                "label": ROUTE_LABELS[route],
                "sessions": len(group),
                "share_of_sessions": _ratio(len(group), n_sessions),
                "conversions": conversions,
                "route_conversion_rate": _ratio(conversions, len(group)),
                "share_of_all_conversions": _ratio(conversions, n_conversions),
            }
        )

    def count(routes_subset, column=None):
        mask = frame["route"].isin([r.value for r in routes_subset])
        return int(frame.loc[mask, column].sum()) if column else int(mask.sum())

    review_sessions = count(REVIEW_ROUTES)
    human_sessions = count(HUMAN_ROUTES)
    priority_sessions = count((Decision.PRIORITY_REVIEW,))
    logged_sessions = count((Decision.LOG_ONLY,))
    review_conversions = count(REVIEW_ROUTES, "y_true")
    logged_conversions = count((Decision.LOG_ONLY,), "y_true")

    uncertain_rows = frame[frame["uncertain"]]
    return {
        "n_sessions": n_sessions,
        "n_conversions": n_conversions,
        "base_conversion_rate": _ratio(n_conversions, n_sessions),
        "routes": per_route,
        "summary": {
            "automatically_logged_share": _ratio(logged_sessions, n_sessions),
            "review_queue_share": _ratio(review_sessions, n_sessions),
            "priority_review_share": _ratio(priority_sessions, n_sessions),
            "human_review_share": _ratio(human_sessions, n_sessions),
            "priority_plus_human_review_conversion_coverage": _ratio(
                review_conversions, n_conversions
            ),
            "conversions_in_automatic_log": logged_conversions,
            "share_of_conversions_in_automatic_log": _ratio(logged_conversions, n_conversions),
            "random_selection_of_same_size_expected_coverage": _ratio(
                review_sessions, n_sessions
            ),
        },
        "workload_per_1000_sessions": {
            "priority_review": round(1000 * priority_sessions / n_sessions, 1),
            "human_review": round(1000 * human_sessions / n_sessions, 1),
            "total_review_queue": round(1000 * review_sessions / n_sessions, 1),
        },
        "policy_consistency": {
            "uncertain_sessions": len(uncertain_rows),
            "uncertain_sessions_routed_to_human_review": int(
                (uncertain_rows["route"] == Decision.HUMAN_REVIEW.value).sum()
            ),
            "system_fallbacks": count((Decision.SYSTEM_FALLBACK,)),
        },
    }


def build_results() -> dict:
    metadata = json.loads(MODEL_METADATA.read_text(encoding="utf-8"))
    sessions, source = load_source(metadata)
    routes = route_sessions(sessions, metadata)
    return {
        "analysis_type": source["analysis_type"],
        "source": source
        | {
            "model_metadata_file": _relative(MODEL_METADATA),
            "model_metadata_sha256": sha256_of(MODEL_METADATA),
            "decision_prompt_file": _relative(PROMPT_PATH),
            "decision_prompt_sha256": sha256_of(PROMPT_PATH),
        },
        "model": {
            "name": metadata["model_name"],
            "version": metadata["model_version"],
            "feature_contract_version": metadata["feature_contract_version"],
        },
        "policy": {
            "decision_threshold": metadata["threshold_selection"]["threshold"],
            "false_negative_cost": metadata["threshold_selection"]["false_negative_cost"],
            "false_positive_cost": metadata["threshold_selection"]["false_positive_cost"],
            "uncertainty_band": metadata["calibration"]["uncertainty_band"],
            "agent_version": AGENT_VERSION,
            "route_source": "conversion_router.agent.client.get_decision with the "
            "deterministic RuleBasedFakeProvider (no LLM call)",
            "parameters_chosen_here": "none (threshold, band, and policy are frozen inputs)",
        },
        "metrics": compute_metrics(sessions, routes),
        "non_claims": [
            "No revenue, profit, uplift, or ROI is estimated.",
            "Routes reflect the written policy as reproduced by the deterministic "
            "provider. A live language model could choose a different route for a "
            "clear-cut session; the validator still forces every uncertain session to "
            "human review.",
            "Public dataset, offline analysis; not connected to a live event stream.",
        ],
    }


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def render_markdown(results: dict) -> str:
    m = results["metrics"]
    s = m["summary"]
    w = m["workload_per_1000_sessions"]
    c = m["policy_consistency"]
    priority = next(r for r in m["routes"] if r["route"] == Decision.PRIORITY_REVIEW.value)
    policy = results["policy"]
    cost_ratio = f"{policy['false_negative_cost']:g}:{policy['false_positive_cost']:g}"
    src = results["source"]
    frozen = results["analysis_type"] == ANALYSIS_FROZEN_TEST
    title = (
        "Operational Impact on Frozen Test Predictions"
        if frozen
        else "Validation Operational Analysis (not final test evidence)"
    )
    lines = [
        f"# {title}",
        "",
        "Generated by `scripts/operational_impact.py`; do not edit by hand. "
        "Numbers are in `docs/operational_impact_results.json`.",
        "",
        "## What this measures",
        "",
        "How the frozen routing policy would distribute reviewer attention across "
        f"{m['n_sessions']:,} sessions, and where the {m['n_conversions']:,} converting "
        f"sessions (base rate {_pct(m['base_conversion_rate'])}) would land. It answers an "
        "operations question (how much review work, and how many conversions sit in the "
        "queue versus the automatic log), not a revenue question.",
        "",
        "## Source and integrity",
        "",
        f"- Analysis type: `{results['analysis_type']}` ({src['split']}).",
        f"- Predictions: `{src['predictions_file']}` "
        f"(SHA-256 `{src['predictions_sha256'][:16]}…`), read-only. "
        "The model was not re-run and no test artifact was written.",
    ]
    if frozen:
        lines.append(
            f"- Cross-checked against `{src['evaluation_file']}`: "
            + ", ".join(f"{k.replace('_', ' ')}: {'pass' if v else 'FAIL'}"
                        for k, v in src["integrity_checks"].items())
            + "."
        )
    lines += [
        f"- Model `{results['model']['name']}` {results['model']['version']}; "
        f"decision threshold {results['policy']['decision_threshold']}, uncertainty band "
        f"±{results['policy']['uncertainty_band']} (both frozen in "
        f"`{src['model_metadata_file']}`).",
        f"- Routes: {results['policy']['route_source']}; policy prompt "
        f"`{src['decision_prompt_file']}`. No parameter was chosen in this analysis.",
        "",
        "## Results by route",
        "",
        "| Route | Sessions | Share of sessions | Conversions | Route conversion rate "
        "| Share of all conversions |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in m["routes"]:
        lines.append(
            f"| {r['label']} | {r['sessions']:,} | {_pct(r['share_of_sessions'])} | "
            f"{r['conversions']:,} | {_pct(r['route_conversion_rate'])} | "
            f"{_pct(r['share_of_all_conversions'])} |"
        )
    lines += [
        "",
        "![Share of sessions and share of conversions by route]"
        "(figures/operational_route_shares.png)",
        "",
        "## Workload and coverage",
        "",
        f"- Automatically logged: {_pct(s['automatically_logged_share'])} of sessions need "
        "no reviewer time.",
        f"- Review queue (priority + human review + fallback): "
        f"{_pct(s['review_queue_share'])} of sessions, i.e. {w['total_review_queue']} per "
        f"1,000 sessions ({w['priority_review']} priority, {w['human_review']} human review).",
        f"- That queue contains {_pct(s['priority_plus_human_review_conversion_coverage'])} "
        "of all converting sessions. A random selection of the same size would be expected "
        f"to contain {_pct(s['random_selection_of_same_size_expected_coverage'])}.",
        f"- {s['conversions_in_automatic_log']} converting sessions "
        f"({_pct(s['share_of_conversions_in_automatic_log'])}) end up in the automatic log: "
        "this is the cost of not reviewing low-scored sessions.",
        f"- Policy consistency: {c['uncertain_sessions_routed_to_human_review']} of "
        f"{c['uncertain_sessions']} uncertain sessions were routed to human review; "
        f"{c['system_fallbacks']} safety fallbacks occurred.",
        "",
        "## How to read this",
        "",
        f"- The threshold was chosen on validation data with a {cost_ratio} "
        "false-negative to false-positive cost ratio, so the policy deliberately sends many "
        "sessions to "
        "review in order to miss few conversions. The queue is large by design; a team with "
        "less capacity would need a different, separately validated threshold.",
        "- \"Route conversion rate\" is the share of sessions in that route that converted in "
        f"the dataset. {_pct(priority['route_conversion_rate'])} of priority-review sessions "
        f"converted, against a base rate of {_pct(m['base_conversion_rate'])}: priority means "
        "\"look first\", not \"will buy\".",
        "",
        "## Non-claims and limitations",
        "",
    ]
    lines += [f"- {item}" for item in results["non_claims"]]
    lines += [
        "- Test numbers are reported once, from frozen predictions; they were not used to "
        "choose the model, threshold, band, or policy."
        if frozen
        else "- Validation data was also used to choose the threshold, so these numbers are "
        "optimistic and are not final evidence.",
        "",
        "## Reproduce",
        "",
        "```bash",
        "venv/bin/python scripts/operational_impact.py          # regenerate",
        "venv/bin/python scripts/operational_impact.py --check  # verify outputs are current",
        "```",
        "",
    ]
    return "\n".join(lines)


def render_figure(results: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    surface, ink, muted, grid = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3dd"
    sessions_color, conversions_color = "#2a78d6", "#eb6834"
    routes = [r for r in results["metrics"]["routes"] if r["sessions"] or r["conversions"]]
    labels = [r["label"] for r in routes]
    session_share = [100 * r["share_of_sessions"] for r in routes]
    conversion_share = [100 * r["share_of_all_conversions"] for r in routes]

    fig, ax = plt.subplots(figsize=(8, 0.9 + 0.9 * len(routes)), dpi=150)
    fig.patch.set_facecolor(surface)
    ax.set_facecolor(surface)
    height = 0.34
    positions = range(len(routes))
    for offset, values, color, name in (
        (-height / 2 - 0.01, session_share, sessions_color, "Share of sessions"),
        (height / 2 + 0.01, conversion_share, conversions_color, "Share of all conversions"),
    ):
        ys = [p + offset for p in positions]
        ax.barh(ys, values, height=height, color=color, label=name)
        for y, value in zip(ys, values, strict=True):
            ax.text(value + 0.8, y, f"{value:.1f}%", va="center", fontsize=8, color=ink)
    ax.set_yticks(list(positions), labels, color=ink, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, max(session_share + conversion_share) * 1.15)
    ax.set_xlabel("Percent", color=muted, fontsize=8)
    ax.tick_params(axis="x", colors=muted, labelsize=8)
    ax.tick_params(axis="y", length=0)
    ax.xaxis.grid(True, color=grid, linewidth=0.6)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)
    title = (
        "Frozen test predictions" if results["analysis_type"] == ANALYSIS_FROZEN_TEST
        else "Validation predictions (not final evidence)"
    )
    ax.set_title(
        f"Where sessions and conversions land by route — {title}",
        loc="left", fontsize=10, color=ink,
    )
    ax.legend(frameon=False, fontsize=8, loc="lower right", labelcolor=ink)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=surface, metadata={"Software": None})
    plt.close(fig)


def results_json(results: dict) -> str:
    return json.dumps(results, indent=2) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true",
                        help="verify committed outputs match a fresh computation")
    args = parser.parse_args()

    results = build_results()
    expected = {OUT_JSON: results_json(results), OUT_MD: render_markdown(results)}
    if args.check:
        stale = [
            _relative(path)
            for path, text in expected.items()
            if not path.exists() or path.read_text(encoding="utf-8") != text
        ]
        if not OUT_FIGURE.exists():
            stale.append(_relative(OUT_FIGURE))
        print("STALE: " + ", ".join(stale) if stale else "Operational impact outputs are current.")
        return 1 if stale else 0

    for path, text in expected.items():
        path.write_text(text, encoding="utf-8")
    render_figure(results, OUT_FIGURE)
    for path in (*expected, OUT_FIGURE):
        print(f"Wrote {_relative(path)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
