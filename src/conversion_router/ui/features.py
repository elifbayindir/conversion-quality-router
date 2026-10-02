"""Human-readable input form specification and session presets.

Field names come from `SessionFeaturesRequest.model_fields` and categorical
options from `CATEGORICAL_WHITELISTS`; this module only adds display labels,
help text, grouping, and the Basic/Advanced split. Presets are the real
validation-split sessions already used by the QA acceptance matrix
(`automation/fixtures/real_model_*.json`); they are validated against the
request contract on load.
"""

from __future__ import annotations

import calendar
import json
from dataclasses import dataclass
from pathlib import Path

from conversion_router.data.validate import CATEGORICAL_WHITELISTS
from conversion_router.schemas import SessionFeaturesRequest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
FIXTURES_DIR = PROJECT_ROOT / "automation" / "fixtures"


@dataclass(frozen=True)
class FieldSpec:
    name: str
    label: str
    help: str
    group: str
    basic: bool
    step: float | None = None


GROUP_ENGAGEMENT = "Browsing engagement"
GROUP_QUALITY = "Session quality"
GROUP_TIMING = "Timing"
GROUP_CONTEXT = "Visitor and encoded technical context"

# The source dataset publishes these four fields as integer identifiers without
# a reliable semantic mapping, so no real-world name is ever shown or guessed.
ENCODED_CATEGORY_HELP = (
    "Encoded category from the source dataset. A stable semantic label is not provided, "
    "so the identifier is passed to the model without assigning a real-world name."
)
GROUPS = (GROUP_ENGAGEMENT, GROUP_QUALITY, GROUP_TIMING, GROUP_CONTEXT)

FIELD_SPECS: tuple[FieldSpec, ...] = (
    FieldSpec("ProductRelated", "Product pages viewed",
              "Number of product pages opened in the session.", GROUP_ENGAGEMENT, True, 1),
    FieldSpec("ProductRelated_Duration", "Time on product pages (seconds)",
              "Total seconds spent on product pages.", GROUP_ENGAGEMENT, True, 10.0),
    FieldSpec("Administrative", "Account pages viewed",
              "Number of account/administrative pages opened (for example login or "
              "account settings).", GROUP_ENGAGEMENT, False, 1),
    FieldSpec("Administrative_Duration", "Time on account pages (seconds)",
              "Total seconds spent on account/administrative pages.", GROUP_ENGAGEMENT,
              False, 10.0),
    FieldSpec("Informational", "Information pages viewed",
              "Number of information pages opened (for example shipping or help pages).",
              GROUP_ENGAGEMENT, False, 1),
    FieldSpec("Informational_Duration", "Time on information pages (seconds)",
              "Total seconds spent on information pages.", GROUP_ENGAGEMENT, False, 10.0),
    FieldSpec("BounceRates", "Average bounce rate (0-1)",
              "Average share of visitors who left right after landing on the pages this "
              "session visited. Higher means weaker engagement.", GROUP_QUALITY, True, 0.001),
    FieldSpec("ExitRates", "Average exit rate (0-1)",
              "Average share of page views that were the last in a session, over the pages "
              "this session visited.", GROUP_QUALITY, True, 0.001),
    FieldSpec("Month", "Session month", "Calendar month of the session.", GROUP_TIMING, True),
    FieldSpec("Weekend", "Weekend session", "Whether the session happened on a weekend.",
              GROUP_TIMING, True),
    FieldSpec("SpecialDay", "Closeness to a special shopping day (0-1)",
              "0 means no nearby special day; values closer to 1 mean the session is close "
              "to an event such as a holiday.", GROUP_TIMING, False, 0.2),
    FieldSpec("VisitorType", "Visitor type", "New, returning, or other visitor.",
              GROUP_CONTEXT, True),
    FieldSpec("OperatingSystems", "Operating-system category ID", ENCODED_CATEGORY_HELP,
              GROUP_CONTEXT, False),
    FieldSpec("Browser", "Browser category ID", ENCODED_CATEGORY_HELP, GROUP_CONTEXT, False),
    FieldSpec("Region", "Region category ID", ENCODED_CATEGORY_HELP, GROUP_CONTEXT, False),
    FieldSpec("TrafficType", "Traffic-source category ID", ENCODED_CATEGORY_HELP,
              GROUP_CONTEXT, False),
)

FIELD_BY_NAME = {spec.name: spec for spec in FIELD_SPECS}

VISITOR_TYPE_LABELS = {
    "New_Visitor": "New visitor",
    "Returning_Visitor": "Returning visitor",
    "Other": "Other",
}

_MONTH_ORDER = {abbr: i for i, abbr in enumerate(calendar.month_abbr) if abbr}


def categorical_options(field_name: str) -> list:
    """Allowed values for a categorical field, from the shared whitelist."""
    values = CATEGORICAL_WHITELISTS[field_name]
    if field_name == "Month":
        return sorted(values, key=lambda m: _MONTH_ORDER[m[:3]])
    return sorted(values, key=str)


def option_label(field_name: str, value) -> str:
    if field_name == "VisitorType":
        return VISITOR_TYPE_LABELS.get(value, str(value))
    if field_name == "Month":
        return calendar.month_name[_MONTH_ORDER[value[:3]]]
    return str(value)


# -- Basic-mode units ---------------------------------------------------------
# Basic mode shows durations in minutes and rates in percent. Conversions are
# applied only to values the user edits; untouched values reach the API exactly
# as stored (the canonical request contract is unchanged).

MINUTE_FIELDS = frozenset(
    {"ProductRelated_Duration", "Administrative_Duration", "Informational_Duration"}
)
PERCENT_FIELDS = frozenset({"BounceRates", "ExitRates"})
CODE_FIELDS = frozenset({"OperatingSystems", "Browser", "Region", "TrafficType"})


def to_display(field_name: str, value):
    if field_name in MINUTE_FIELDS:
        return value / 60.0
    if field_name in PERCENT_FIELDS:
        return value * 100.0
    return value


def to_canonical(field_name: str, value):
    if field_name in MINUTE_FIELDS:
        return value * 60.0
    if field_name in PERCENT_FIELDS:
        return value / 100.0
    return value


def basic_label(spec: FieldSpec) -> str:
    if spec.name in MINUTE_FIELDS:
        return spec.label.replace("(seconds)", "(minutes)")
    if spec.name in PERCENT_FIELDS:
        return spec.label.replace("(0-1)", "(%)")
    return spec.label


def readable_duration(seconds: float) -> str:
    total = int(round(seconds))
    hours, rest = divmod(total, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours} h {minutes} min"
    if minutes:
        return f"{minutes} min {secs} s"
    return f"{secs} s"


def readable_value(field_name: str, value) -> str:
    """Analyst-facing text for a canonical value (used in tables and captions)."""
    if field_name in CATEGORICAL_WHITELISTS and field_name not in CODE_FIELDS:
        return option_label(field_name, value)
    if field_name in CODE_FIELDS:
        return f"Category {value}"
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if field_name in MINUTE_FIELDS:
        return f"{readable_duration(value)} ({value:g} s)"
    if field_name in PERCENT_FIELDS:
        return f"{value * 100:.2f}%"
    if field_name == "SpecialDay":
        return f"{value:g}"
    return str(value)


def basic_fields() -> list[FieldSpec]:
    return [spec for spec in FIELD_SPECS if spec.basic]


def advanced_only_fields() -> list[FieldSpec]:
    return [spec for spec in FIELD_SPECS if not spec.basic]


def preset_business_fields() -> list[FieldSpec]:
    """Preset values that are not edited in Basic mode, excluding encoded categories."""
    return [spec for spec in advanced_only_fields() if spec.name not in CODE_FIELDS]


def encoded_category_fields() -> list[FieldSpec]:
    return [spec for spec in FIELD_SPECS if spec.name in CODE_FIELDS]


@dataclass(frozen=True)
class Preset:
    key: str
    label: str
    description: str
    session: SessionFeaturesRequest


# Fixture file -> display text. The fixtures carry the real validation rows.
_PRESET_SOURCES = (
    ("real_model_priority_review.json", "Strong purchase signal",
     "Long product browsing with low bounce rate (real validation-split session)."),
    ("real_model_human_review.json", "Borderline signal",
     "Probability close to the decision threshold (real validation-split session)."),
    ("real_model_log_only.json", "Weak purchase signal",
     "Short session with little product engagement (real validation-split session)."),
)


def load_presets(fixtures_dir: Path = FIXTURES_DIR) -> list[Preset]:
    presets = []
    for filename, label, description in _PRESET_SOURCES:
        payload = json.loads((fixtures_dir / filename).read_text(encoding="utf-8"))
        session = SessionFeaturesRequest.model_validate(payload["input"]["session"])
        presets.append(
            Preset(key=filename.removesuffix(".json"), label=label,
                   description=description, session=session)
        )
    return presets
