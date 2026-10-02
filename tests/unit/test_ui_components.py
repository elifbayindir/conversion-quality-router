import re

import pytest

from conversion_router.schemas import Decision, DecisionResponse, PredictionResponse
from conversion_router.ui import components, features, presentation, settings, theme


def _contrast(fg: str, bg: str) -> float:
    def lum(hex_color: str) -> float:
        rgb = [int(hex_color.lstrip("#")[i:i + 2], 16) / 255 for i in (0, 2, 4)]
        lin = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in rgb]
        return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]

    high, low = sorted((lum(fg), lum(bg)), reverse=True)
    return (high + 0.05) / (low + 0.05)


def _decision(value: str) -> DecisionResponse:
    spec = {
        "PRIORITY_REVIEW": ("HIGH", "ADD_TO_REVIEW_QUEUE_PRIORITY", False,
                            "HIGH_CONFIDENCE_POSITIVE"),
        "HUMAN_REVIEW": ("MEDIUM", "ADD_TO_REVIEW_QUEUE", True, "MODEL_UNCERTAIN"),
        "LOG_ONLY": ("LOW", "ADD_TO_LOG", False, "HIGH_CONFIDENCE_NEGATIVE"),
        "SYSTEM_FALLBACK": ("HIGH", "ADD_TO_REVIEW_QUEUE", True, "AGENT_OUTPUT_INVALID"),
    }[value]
    return DecisionResponse(
        request_id="req_x", decision=value, priority=spec[0], allowed_action=spec[1],
        requires_human_approval=spec[2], reason_codes=[spec[3]],
        explanation="<b>agent</b> note", agent_version="1.0.0",
    )


# -- theme --------------------------------------------------------------------------------


def test_every_route_has_one_distinct_accessible_tone():
    assert set(theme.ROUTE_TONES) == set(Decision)
    assert len({tone.fg for tone in theme.ROUTE_TONES.values()}) == len(Decision)
    for tone in (*theme.ROUTE_TONES.values(), theme.SUCCESS, theme.PRIMARY_TONE, theme.NEUTRAL):
        assert _contrast(tone.fg, tone.bg) >= 4.5, tone
        assert _contrast(theme.SURFACE, tone.fg) >= 4.5, tone  # white text on filled segments
    assert _contrast(theme.TEXT, theme.PAGE_BG) >= 4.5
    assert _contrast(theme.TEXT_MUTED, theme.PAGE_BG) >= 4.5


def test_stylesheet_only_targets_own_classes():
    css = theme.stylesheet()
    assert css.startswith("<style>") and "\n" not in css
    selectors = re.findall(r"(?:^|})\s*([^{}]+)\{", css.removeprefix("<style>"))
    assert selectors and all(s.strip().startswith(".cqr-") for s in selectors)


# -- probability rail ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("probability", "uncertain", "sentence"),
    [
        (0.6667, False, "Clearly above the decision threshold."),
        (0.0, False, "Clearly below the decision threshold."),
        (0.1111, True, "Inside the uncertainty band"),
    ],
)
def test_rail_states_clear_and_uncertain_positions(probability, uncertain, sentence):
    rail = components.probability_rail(probability, 0.12, uncertain, 0.05, theme.PRIMARY_TONE)
    assert sentence in rail
    assert f'data-role="probability" style="left:{probability * 100:.2f}%' in rail
    assert 'data-role="threshold" style="left:12.00%"' in rail
    assert 'data-role="band" style="left:7.00%;width:10.00%"' in rail
    assert "not a certainty or a causal effect" in rail
    assert "0%" in rail and "100%" in rail


def test_rail_without_matching_band_says_so_and_draws_no_band():
    rail = components.probability_rail(0.4, 0.12, False, None, theme.PRIMARY_TONE)
    assert 'data-role="band"' not in rail
    assert "Uncertainty band not drawn" in rail


def test_rail_labels_stay_inside_the_track_at_the_extremes():
    rail = components.probability_rail(0.0, 0.12, False, 0.05, theme.PRIMARY_TONE)
    assert "left:4.00%;top:-8px" in rail  # probability label clamped, marker itself at 0%
    assert 'data-role="probability" style="left:0.00%' in rail
    rail = components.probability_rail(1.0, 0.12, False, 0.05, theme.PRIMARY_TONE)
    assert "left:96.00%;top:-8px" in rail


# -- cards, path, chips -----------------------------------------------------------------


@pytest.mark.parametrize("route", [d.value for d in Decision])
def test_decision_card_shows_route_name_full_action_and_route_colour(route):
    decision = _decision(route)
    card = components.decision_card(decision)
    tone = theme.ROUTE_TONES[decision.decision]
    assert f'data-route="{route}"' in card
    assert tone.fg in card and tone.bg in card
    assert "Recommended route" in card and "Recommended action" in card
    # Colour is never the only carrier: the route title is always present as text.
    assert presentation.ROUTE_DISPLAY[decision.decision].title in card
    assert presentation.ACTION_LABELS[decision.allowed_action] in card


def test_why_this_route_merges_reasons_and_escaped_explanation():
    html = components.why_this_route(_decision("HUMAN_REVIEW"))
    assert "Why this route?" in html and "Model is uncertain" in html
    assert "&lt;b&gt;agent&lt;/b&gt;" in html and "<b>agent</b>" not in html


def test_decision_path_highlights_only_route_steps():
    routed = components.decision_path(_decision("PRIORITY_REVIEW"), uncertain=False)
    for step in ("Session signals", "Calibrated probability", "Uncertainty check: clear",
                 "Priority review", "Add to priority review queue"):
        assert step in routed
    tone = theme.ROUTE_TONES[Decision.PRIORITY_REVIEW]
    assert routed.count(f'class="cqr-step" style="color:{tone.fg}') == 2  # route + action only
    assert routed.count('class="cqr-step"') == 5
    model_only = components.decision_path(None, uncertain=True)
    assert "Route not requested (model only)" in model_only
    assert "Uncertainty check: uncertain" in model_only


def test_signal_chips_come_only_from_returned_signals():
    prediction = PredictionResponse.model_validate(
        {
            "request_id": "req_x",
            "prediction": {"purchase_probability": 0.5, "decision_threshold": 0.12,
                           "predicted_class": "likely_to_convert", "decision_margin": 0.38,
                           "uncertain": False},
            "signals": ["low_bounce_rate", "returning_visitor"],
            "model": {"name": "m", "version": "1", "feature_contract_version": "1"},
        }
    )
    chips = components.signal_chips(prediction)
    assert chips.count('class="cqr-chip"') == 2
    assert "Low bounce rate" in chips and "Returning visitor" in chips
    empty = prediction.model_copy(update={"signals": []})
    assert "No supporting signals" in components.signal_chips(empty)


# -- queue and feedback bars ---------------------------------------------------------------


def test_queue_composition_counts_and_percentages():
    counts = {d: 0 for d in Decision} | {Decision.PRIORITY_REVIEW: 2, Decision.HUMAN_REVIEW: 1,
                                         Decision.LOG_ONLY: 1}
    bar = components.queue_composition(counts)
    segments = dict(re.findall(r'data-route="(\w+)" data-count="(\d+)"', bar))
    assert segments == {"PRIORITY_REVIEW": "2", "HUMAN_REVIEW": "1", "LOG_ONLY": "1"}
    assert "Priority review: 2 (50%)" in bar and "Logged automatically: 1 (25%)" in bar
    assert "Safety fallback" not in bar


def test_queue_composition_shows_fallback_only_when_present_and_empty_state():
    counts = {d: 0 for d in Decision} | {Decision.SYSTEM_FALLBACK: 3}
    assert "Safety fallback: 3 (100%)" in components.queue_composition(counts)
    assert "queue is empty" in components.queue_composition({d: 0 for d in Decision})


def test_override_reason_text_for_agree_and_override():
    from conversion_router.feedback.schemas import OverrideReason

    assert presentation.override_reason_text(None) == "Not applicable — reviewer agreed"
    assert (
        presentation.override_reason_text(OverrideReason.SIGNAL_WEAKER_THAN_SCORED)
        == "Session looks weaker than the score suggests"
    )


def test_routing_help_states_the_automation_boundary_in_every_mode():
    for text in settings.ROUTING_HELP.values():
        assert text.endswith(
            "It does not trigger the n8n workflow or send Slack notifications."
        )


def test_feedback_balance_is_labelled_illustrative_with_n():
    bar = components.feedback_balance(2, 1)
    assert "Illustrative local feedback, n=3" in bar
    assert "Agreed: 2" in bar and "Overridden: 1" in bar


# -- settings -----------------------------------------------------------------------------


def test_run_mode_is_explicit_and_unknown_values_are_not_demo():
    assert settings.run_mode({}) == settings.RunMode.UNDECLARED
    assert settings.run_mode({settings.MODE_ENV_VAR: "local_demo"}) == settings.RunMode.LOCAL_DEMO
    assert settings.run_mode({settings.MODE_ENV_VAR: "production"}) == settings.RunMode.UNDECLARED
    assert "demo" not in settings.RUN_MODE_TEXT[settings.RunMode.UNDECLARED].lower()


def test_display_band_only_when_metadata_matches_the_service(tmp_path):
    policy = settings.load_display_policy()
    assert policy == settings.DisplayPolicy(threshold=0.12, uncertainty_band=0.05)
    assert settings.band_for(0.12, policy) == 0.05
    assert settings.band_for(0.3, policy) is None
    assert settings.load_display_policy(tmp_path / "missing.json") is None
    assert settings.band_for(0.12, None) is None


# -- basic-mode units -----------------------------------------------------------------------


@pytest.mark.parametrize("preset", features.load_presets(), ids=lambda p: p.key)
def test_display_conversions_round_trip_every_preset_value(preset):
    for name, value in preset.session.model_dump().items():
        back = features.to_canonical(name, features.to_display(name, value))
        assert back == pytest.approx(value, rel=1e-12, abs=1e-12)


def test_unit_conversions_and_readable_values():
    assert features.to_canonical("ProductRelated_Duration", 10.0) == 600.0
    assert features.to_display("ProductRelated_Duration", 600.0) == 10.0
    assert features.to_canonical("BounceRates", 5.0) == pytest.approx(0.05)
    assert features.to_display("ExitRates", 0.2) == pytest.approx(20.0)
    assert features.to_display("Month", "Nov") == "Nov"
    assert features.readable_duration(13453.25103) == "3 h 44 min"
    assert features.readable_duration(281) == "4 min 41 s"
    assert features.readable_duration(0) == "0 s"
    assert features.readable_value("TrafficType", 2) == "Category 2"
    assert features.FIELD_BY_NAME["Region"].label == "Region category ID"
    assert "without assigning a real-world name" in features.FIELD_BY_NAME["Browser"].help
    assert features.readable_value("BounceRates", 0.003240964) == "0.32%"
    assert features.basic_label(features.FIELD_BY_NAME["ExitRates"]) == "Average exit rate (%)"
