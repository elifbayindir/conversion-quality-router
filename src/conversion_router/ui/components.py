"""Pure HTML builders for the UI's small decision visuals.

Each function formats values the API already returned; none computes a
probability, threshold, uncertainty flag, or route. All text is HTML-escaped.
Output is a single line so Streamlit's markdown renderer treats it as HTML.
"""

from __future__ import annotations

from html import escape

from conversion_router.schemas import Decision, DecisionResponse, PredictionResponse
from conversion_router.ui import presentation, theme

QUEUE_SEGMENTS = (
    (Decision.PRIORITY_REVIEW, "Priority review"),
    (Decision.HUMAN_REVIEW, "Human review"),
    (Decision.SYSTEM_FALLBACK, "Safety fallback"),
    (Decision.LOG_ONLY, "Logged automatically"),
)


def _clamp(value: float, low: float = 4.0, high: float = 96.0) -> float:
    return max(low, min(high, value))


def pill(text: str, tone: theme.Tone) -> str:
    return (
        f'<span class="cqr-pill" style="color:{tone.fg};background:{tone.bg};'
        f'border-color:{tone.fg}33">{escape(text)}</span>'
    )


def decision_card(decision: DecisionResponse) -> str:
    route = presentation.route_display(decision)
    tone = theme.ROUTE_TONES[decision.decision]
    facts = (
        ("Priority", presentation.PRIORITY_LABELS[decision.priority]),
        ("Recommended action", presentation.ACTION_LABELS[decision.allowed_action]),
        ("Human approval", "Required" if decision.requires_human_approval else "Not required"),
    )
    facts_html = "".join(
        f'<div class="cqr-fact"><div class="cqr-fact-label">{escape(label)}</div>'
        f'<div class="cqr-fact-value">{escape(value)}</div></div>'
        for label, value in facts
    )
    return (
        f'<div class="cqr-card" data-route="{decision.decision.value}" '
        f'style="background:{tone.bg};border-left:4px solid {tone.fg}">'
        f'<div class="cqr-eyebrow">Recommended route</div>'
        f'<div class="cqr-title" style="color:{tone.fg}">{escape(route.title)}</div>'
        f'<div class="cqr-text">{escape(route.summary)}</div>'
        f'<div class="cqr-facts">{facts_html}</div></div>'
    )


def why_this_route(decision: DecisionResponse) -> str:
    reasons = "; ".join(presentation.REASON_LABELS[r] for r in decision.reason_codes)
    return (
        '<div class="cqr-card"><div class="cqr-eyebrow">Why this route?</div>'
        f'<div class="cqr-text"><strong>{escape(reasons)}.</strong> '
        f"{escape(decision.explanation)}</div></div>"
    )


def rail_sentence(probability: float, threshold: float, uncertain: bool) -> str:
    if uncertain:
        return "Inside the uncertainty band: too close to the threshold to call either way."
    if probability >= threshold:
        return "Clearly above the decision threshold."
    return "Clearly below the decision threshold."


def probability_rail(
    probability: float,
    threshold: float,
    uncertain: bool,
    band: float | None,
    tone: theme.Tone,
) -> str:
    """0-100% rail with the probability marker, threshold line, and (if known) band."""
    p, t = probability * 100, threshold * 100
    band_html = ""
    if band is not None:
        left, right = max(0.0, t - band * 100), min(100.0, t + band * 100)
        band_html = (
            f'<div class="cqr-rail-band" data-role="band" '
            f'style="left:{left:.2f}%;width:{right - left:.2f}%"></div>'
        )
    # Probability label above the rail, threshold label below: they never
    # overlap, even when the two values are close.
    return (
        '<div class="cqr-rail" role="img" aria-label="'
        f"Calibrated purchase probability {p:.1f} percent; decision threshold {t:.1f} percent"
        '">'
        '<div class="cqr-rail-track"></div>'
        f"{band_html}"
        f'<div class="cqr-rail-threshold" data-role="threshold" style="left:{t:.2f}%"></div>'
        f'<div class="cqr-rail-dot" data-role="probability" '
        f'style="left:{p:.2f}%;background:{tone.fg}"></div>'
        f'<div class="cqr-rail-label" style="left:{_clamp(p):.2f}%;top:-8px;color:{tone.fg}">'
        f"<strong>Probability {p:.1f}%</strong></div>"
        f'<div class="cqr-rail-label" style="left:{_clamp(t):.2f}%;top:42px">'
        f"Threshold {t:.1f}%</div>"
        "</div>"
        '<div class="cqr-rail-scale"><span>0%</span><span>25%</span><span>50%</span>'
        "<span>75%</span><span>100%</span></div>"
        '<div class="cqr-muted" style="margin-top:6px">'
        + escape(rail_sentence(probability, threshold, uncertain))
        + (
            f" Shaded area: uncertainty band ({t - band * 100:.1f}%–{t + band * 100:.1f}%)."
            if band is not None
            else " Uncertainty band not drawn: display settings do not match the service."
        )
        + " The probability is a calibrated estimate, not a certainty or a causal effect.</div>"
    )


def signal_chips(prediction: PredictionResponse) -> str:
    if not prediction.signals:
        return '<div class="cqr-muted">No supporting signals returned.</div>'
    chips = "".join(
        f'<span class="cqr-chip">{escape(presentation.SIGNAL_LABELS[s])}</span>'
        for s in prediction.signals
    )
    return f'<div class="cqr-chips" aria-label="Supporting signals">{chips}</div>'


def decision_path(decision: DecisionResponse | None, uncertain: bool) -> str:
    """Session signals -> probability -> uncertainty check -> route -> action."""
    steps = [
        ("Session signals", None),
        ("Calibrated probability", None),
        ("Uncertainty check: " + ("uncertain" if uncertain else "clear"), None),
    ]
    if decision is None:
        steps += [("Route not requested (model only)", None)]
    else:
        tone = theme.ROUTE_TONES[decision.decision]
        steps += [
            (presentation.route_display(decision).title, tone),
            (presentation.ACTION_LABELS[decision.allowed_action], tone),
        ]
    parts = []
    for i, (label, tone) in enumerate(steps):
        style = (
            f' style="color:{tone.fg};background:{tone.bg};border-color:{tone.fg}55"'
            if tone
            else ""
        )
        parts.append(f'<span class="cqr-step"{style}>{escape(label)}</span>')
        if i < len(steps) - 1:
            parts.append('<span class="cqr-arrow" aria-hidden="true">→</span>')
    return f'<div class="cqr-path" aria-label="Decision path">{"".join(parts)}</div>'


def queue_composition(counts: dict[Decision, int]) -> str:
    total = sum(counts.values())
    if total == 0:
        return '<div class="cqr-muted">The queue is empty: no route to show yet.</div>'
    segments, legend = [], []
    for route, label in QUEUE_SEGMENTS:
        n = counts.get(route, 0)
        if n == 0 and route == Decision.SYSTEM_FALLBACK:
            continue
        tone = theme.ROUTE_TONES[route]
        share = 100 * n / total
        if n:
            segments.append(
                f'<div class="cqr-seg" data-route="{route.value}" data-count="{n}" '
                f'style="flex:{n} 1 0;background:{tone.fg};color:{theme.SURFACE}" '
                f'title="{escape(label)}: {n} ({share:.0f}%)">{n}</div>'
            )
        legend.append(
            f'<span><span class="cqr-swatch" style="background:{tone.fg}"></span>'
            f"{escape(label)}: {n} ({share:.0f}%)</span>"
        )
    return (
        f'<div class="cqr-stack" role="img" aria-label="Queue composition, {total} sessions">'
        f'{"".join(segments)}</div><div class="cqr-legend">{"".join(legend)}</div>'
    )


def feedback_balance(agreed: int, overridden: int) -> str:
    total = agreed + overridden
    agree_tone, override_tone = theme.PRIMARY_TONE, theme.NEUTRAL
    segments = "".join(
        f'<div class="cqr-seg" style="flex:{n} 1 0;background:{tone.fg};color:{theme.SURFACE}">'
        f"{n}</div>"
        for n, tone in ((agreed, agree_tone), (overridden, override_tone))
        if n
    )
    return (
        f'<div class="cqr-eyebrow">Illustrative local feedback, n={total}</div>'
        f'<div class="cqr-stack" role="img" aria-label="Agreed {agreed}, overridden {overridden}">'
        f"{segments}</div>"
        '<div class="cqr-legend">'
        f'<span><span class="cqr-swatch" style="background:{agree_tone.fg}"></span>'
        f"Agreed: {agreed}</span>"
        f'<span><span class="cqr-swatch" style="background:{override_tone.fg}"></span>'
        f"Overridden: {overridden}</span></div>"
    )


def status_line(text: str, tone: theme.Tone) -> str:
    return (
        f'<div class="cqr-card" style="background:{tone.bg};border-color:{tone.fg}33;'
        f'padding:10px 12px"><div class="cqr-text" style="color:{tone.fg}">{escape(text)}'
        "</div></div>"
    )


def empty_state(title: str, body: str) -> str:
    return (
        f'<div class="cqr-card" data-empty="true"><div class="cqr-text"><strong>{escape(title)}'
        f'</strong></div><div class="cqr-muted">{escape(body)}</div></div>'
    )
