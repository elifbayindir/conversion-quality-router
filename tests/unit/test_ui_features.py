import ast
from pathlib import Path

from conversion_router.data.validate import CATEGORICAL_WHITELISTS
from conversion_router.schemas import (
    AllowedAction,
    Decision,
    Priority,
    ReasonCode,
    SessionFeaturesRequest,
    Signal,
)
from conversion_router.ui import features, presentation

UI_DIR = Path(__file__).resolve().parents[2] / "src" / "conversion_router" / "ui"


def test_form_spec_covers_exactly_the_request_contract_fields():
    names = [spec.name for spec in features.FIELD_SPECS]
    assert sorted(names) == sorted(SessionFeaturesRequest.model_fields)
    assert len(names) == len(set(names)) == 16


def test_basic_and_advanced_partition_all_fields():
    basic = {s.name for s in features.basic_fields()}
    advanced_only = {s.name for s in features.advanced_only_fields()}
    assert basic.isdisjoint(advanced_only)
    assert basic | advanced_only == set(SessionFeaturesRequest.model_fields)


def test_every_field_has_a_human_label_and_help_text():
    for spec in features.FIELD_SPECS:
        assert spec.label and spec.label != spec.name
        assert spec.help
        assert spec.group in features.GROUPS


def test_categorical_options_come_from_the_shared_whitelist():
    for name, whitelist in CATEGORICAL_WHITELISTS.items():
        assert set(features.categorical_options(name)) == whitelist
    months = features.categorical_options("Month")
    assert months[0] == "Jan" and months[-1] == "Dec"
    assert features.option_label("Month", "June") == "June"
    assert features.option_label("VisitorType", "Returning_Visitor") == "Returning visitor"


def test_ui_does_not_redeclare_whitelist_values():
    """Categorical values must be read from CATEGORICAL_WHITELISTS, not re-typed."""
    for path in UI_DIR.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, (ast.Set, ast.List, ast.Tuple)):
                literals = {
                    elt.value for elt in node.elts
                    if isinstance(elt, ast.Constant) and isinstance(elt.value, str)
                }
                assert not {"Jan", "Feb", "Mar"} <= literals, path


def test_presets_are_valid_full_sessions_from_validation_fixtures():
    presets = features.load_presets()
    assert len(presets) == 3
    for preset in presets:
        assert isinstance(preset.session, SessionFeaturesRequest)
        assert len(preset.session.model_dump()) == 16
        assert preset.key.startswith("real_model_")
    assert len({p.label for p in presets}) == 3


def test_every_contract_enum_value_has_a_display_label():
    assert set(presentation.ROUTE_DISPLAY) == set(Decision)
    assert set(presentation.ACTION_LABELS) == set(AllowedAction)
    assert set(presentation.PRIORITY_LABELS) == set(Priority)
    assert set(presentation.REASON_LABELS) == set(ReasonCode)
    assert set(presentation.SIGNAL_LABELS) == set(Signal)


def test_display_formatting_only_reformats_returned_values():
    assert presentation.format_probability(0.1111) == "11.1%"
    assert "Uncertain" in presentation.uncertainty_text(True, 0.0089)
    assert "0.9 percentage points" in presentation.uncertainty_text(True, 0.0089)
    assert "Clear signal" in presentation.uncertainty_text(False, 0.5)
