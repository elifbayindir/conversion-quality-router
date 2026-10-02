"""Central colour tokens and the scoped stylesheet for the UI.

Every colour the UI uses comes from here, so a route looks the same on every
screen. Route colours carry meaning only alongside the route's name in text.
All foreground/background pairs meet WCAG AA contrast for normal text
(checked in `tests/unit/test_ui_components.py`).

The stylesheet only targets this package's own `cqr-*` classes; it never
restyles Streamlit's internal elements. Page-level colours (background,
primary interaction colour) are set through `.streamlit/config.toml`.
"""

from __future__ import annotations

from dataclasses import dataclass

from conversion_router.schemas import Decision


@dataclass(frozen=True)
class Tone:
    fg: str
    bg: str


PAGE_BG = "#F7F8FC"
SURFACE = "#FFFFFF"
TEXT = "#1F2937"
TEXT_MUTED = "#667085"
BORDER = "#E4E7EC"
PRIMARY = "#4F46E5"
PRIMARY_SOFT = "#EEF2FF"
TRACK = "#EEF0F4"

PRIMARY_TONE = Tone(PRIMARY, PRIMARY_SOFT)
SUCCESS = Tone("#027A48", "#ECFDF3")
DANGER = Tone("#B42318", "#FDECEC")
NEUTRAL = Tone(TEXT_MUTED, "#F2F4F7")

ROUTE_TONES: dict[Decision, Tone] = {
    Decision.PRIORITY_REVIEW: Tone("#4338CA", "#EEF2FF"),
    Decision.HUMAN_REVIEW: Tone("#B54708", "#FFF4E5"),
    Decision.LOG_ONLY: Tone("#175CD3", "#EAF4FF"),
    Decision.SYSTEM_FALLBACK: DANGER,
}

UNCERTAINTY_BAND_FILL = "#FDE7C8"


def stylesheet() -> str:
    """Scoped CSS for the `cqr-*` components (one line, safe for st.markdown)."""
    rules = f"""
.cqr-card{{background:{SURFACE};border:1px solid {BORDER};border-radius:8px;
padding:14px 16px;margin:4px 0 12px;box-shadow:0 1px 2px rgba(16,24,40,.04);color:{TEXT}}}
.cqr-eyebrow{{font-size:.78rem;letter-spacing:.02em;text-transform:uppercase;color:{TEXT_MUTED}}}
.cqr-title{{font-size:1.25rem;font-weight:600;margin:2px 0 4px}}
.cqr-text{{font-size:.95rem;line-height:1.45;color:{TEXT}}}
.cqr-muted{{font-size:.85rem;color:{TEXT_MUTED}}}
.cqr-facts{{display:flex;flex-wrap:wrap;gap:8px 24px;margin-top:10px}}
.cqr-fact{{min-width:140px;flex:1 1 160px}}
.cqr-fact-label{{font-size:.78rem;color:{TEXT_MUTED}}}
.cqr-fact-value{{font-size:.98rem;font-weight:600;overflow-wrap:anywhere}}
.cqr-pill{{display:inline-block;padding:2px 10px;border-radius:999px;font-size:.82rem;
font-weight:500;margin:2px 6px 2px 0;border:1px solid transparent}}
.cqr-chips{{display:flex;flex-wrap:wrap;gap:6px;margin:6px 0}}
.cqr-chip{{display:inline-block;padding:2px 10px;border-radius:6px;font-size:.84rem;
background:#F2F4F7;color:{TEXT};border:1px solid {BORDER}}}
.cqr-rail{{position:relative;height:58px;margin:26px 4px 8px}}
.cqr-rail-track{{position:absolute;left:0;right:0;top:22px;height:10px;border-radius:5px;
background:{TRACK}}}
.cqr-rail-band{{position:absolute;top:18px;height:18px;border-radius:4px;
background:{UNCERTAINTY_BAND_FILL};border:1px dashed #B54708}}
.cqr-rail-threshold{{position:absolute;top:12px;width:2px;height:30px;background:{TEXT}}}
.cqr-rail-dot{{position:absolute;top:17px;width:20px;height:20px;border-radius:50%;
transform:translateX(-50%);border:3px solid {SURFACE};box-shadow:0 0 0 1px {BORDER}}}
.cqr-rail-label{{position:absolute;font-size:.78rem;white-space:nowrap;transform:translateX(-50%);
color:{TEXT}}}
.cqr-rail-scale{{display:flex;justify-content:space-between;font-size:.72rem;color:{TEXT_MUTED};
margin:0 4px}}
.cqr-path{{display:flex;flex-wrap:wrap;align-items:center;gap:6px;margin:6px 0 10px}}
.cqr-step{{padding:4px 10px;border-radius:6px;font-size:.84rem;background:#F2F4F7;color:{TEXT};
border:1px solid {BORDER}}}
.cqr-arrow{{color:{TEXT_MUTED};font-size:.84rem}}
.cqr-stack{{display:flex;width:100%;height:28px;border-radius:6px;overflow:hidden;gap:2px;
background:{SURFACE};margin:6px 0}}
.cqr-seg{{display:flex;align-items:center;justify-content:center;font-size:.8rem;font-weight:600;
min-width:28px;white-space:nowrap;overflow:hidden}}
.cqr-legend{{display:flex;flex-wrap:wrap;gap:4px 16px;font-size:.84rem;color:{TEXT}}}
.cqr-swatch{{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:6px}}
"""
    return "<style>" + " ".join(line.strip() for line in rules.strip().splitlines()) + "</style>"
