"""UI integration tests: Streamlit AppTest against the REAL FastAPI app.

The API runs in-process (FastAPI TestClient, real frozen artifacts, real
agent chain) with a deterministic demo provider injected explicitly, so no
test here ever opens a network connection or contacts an LLM provider.
Every test uses a temporary feedback store; the real local store is never touched.
"""

import json
import re
import tempfile
import uuid
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient
from streamlit.testing.v1 import AppTest

from conversion_router.agent.demo_providers import AlwaysInvalidProvider, RuleBasedFakeProvider
from conversion_router.api.app import create_app
from conversion_router.feedback.schemas import HumanResponse, OverrideReason
from conversion_router.feedback.store import FeedbackStore
from conversion_router.modeling.inference import DEFAULT_METADATA_PATH
from conversion_router.schemas import Decision
from conversion_router.ui.api_client import ApiClient
from conversion_router.ui.settings import RUN_MODE_TEXT, RunMode

pytestmark = pytest.mark.skipif(
    not DEFAULT_METADATA_PATH.exists(),
    reason="Frozen MLP artifacts not present; run the full P2-P4 pipeline scripts first.",
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
APP_PATH = PROJECT_ROOT / "src" / "conversion_router" / "ui" / "app.py"
FIXTURES = PROJECT_ROOT / "automation" / "fixtures"
_TMP_DIR = tempfile.TemporaryDirectory()

STRONG = "Strong purchase signal"
BORDERLINE = "Borderline signal"
WEAK = "Weak purchase signal"


def _tmp_db() -> Path:
    return Path(_TMP_DIR.name) / f"{uuid.uuid4().hex}.sqlite3"


class CountingClient(ApiClient):
    """Real ApiClient that records each endpoint and request body it sent."""

    def __init__(self, http_client):
        super().__init__(base_url="http://testserver", http_client=http_client)
        self.paths: list[str] = []
        self.bodies: list[dict | None] = []

    def _request(self, method, path, json_body=None):
        self.paths.append(path)
        self.bodies.append(json_body)
        return super()._request(method, path, json_body)

    def last_body(self, path: str) -> dict:
        return [b for p, b in zip(self.paths, self.bodies, strict=True) if p == path][-1]


@pytest.fixture
def api_factory():
    clients = []

    def make(provider):
        test_client = TestClient(create_app())
        test_client.__enter__()
        test_client.app.state.llm_provider = provider
        clients.append(test_client)
        return CountingClient(test_client)

    yield make
    for test_client in clients:
        test_client.__exit__(None, None, None)


def _app(client, store=None, run_mode=RunMode.LOCAL_DEMO) -> AppTest:
    at = AppTest.from_file(str(APP_PATH), default_timeout=60)
    at.session_state["api_client"] = client
    at.session_state["feedback_store"] = store or FeedbackStore(_tmp_db())
    at.session_state["run_mode"] = run_mode
    at.run()
    assert not at.exception, at.exception
    return at


def _evaluate(at: AppTest, preset: str, mode: str = "Full decision routing") -> AppTest:
    at.selectbox(key="preset_label").set_value(preset).run()
    at.radio(key="evaluation_mode").set_value(mode).run()
    at.button(key="evaluate_button").click().run()
    assert not at.exception, at.exception
    return at


def _metrics(at: AppTest) -> dict[str, str]:
    return {m.label: m.value for m in at.metric}


def _html(at: AppTest) -> str:
    return "\n".join(m.value for m in at.markdown)


def _sidebar_html(at: AppTest) -> str:
    return "\n".join(m.value for m in at.sidebar.markdown)


def _card(at: AppTest) -> str:
    cards = [m.value for m in at.markdown if 'class="cqr-card" data-route=' in m.value]
    assert len(cards) == 1, "exactly one decision card expected"
    return cards[0]


def _fixture_session(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))["input"]["session"]


# -- GA: service status and run mode --------------------------------------------------


def test_sidebar_shows_ready_service_and_demo_run_mode(api_factory):
    at = _app(api_factory(RuleBasedFakeProvider()))
    sidebar = _sidebar_html(at)
    assert "Decision service ready" in sidebar
    assert RUN_MODE_TEXT[RunMode.LOCAL_DEMO] in sidebar
    assert "No external LLM or messaging calls" in sidebar
    assert any("conversion_mlp" in c.value for c in at.sidebar.caption)
    routing_radio = at.radio(key="evaluation_mode")
    routing_help = " ".join(routing_radio.proto.captions)
    assert "deterministic provider" in routing_help
    assert "does not trigger the n8n workflow or send Slack notifications" in routing_help
    assert at.radio(key="evaluation_mode").options[0] == "Full decision routing"
    boundary = " ".join(m.value for m in at.sidebar.markdown)
    assert "Local audit feedback. It does not resume an n8n execution." in boundary
    assert "separate automation workflow" in boundary
    assert "does not produce predictions or decisions" in boundary


def test_undeclared_run_mode_never_claims_demo(api_factory):
    at = _app(api_factory(RuleBasedFakeProvider()), run_mode=RunMode.UNDECLARED)
    sidebar = _sidebar_html(at)
    assert "Local demo mode" not in sidebar
    assert "not declared" in sidebar
    assert "ANTHROPIC" not in sidebar and "api_key" not in sidebar.lower()


# -- GA: results -------------------------------------------------------------------------


def test_strong_preset_routes_to_priority_review_with_full_evidence(api_factory):
    client = api_factory(RuleBasedFakeProvider())
    at = _evaluate(_app(client), STRONG)

    card = _card(at)
    assert 'data-route="PRIORITY_REVIEW"' in card and "Priority review" in card
    assert "Add to priority review queue" in card  # full action text, not a truncated metric
    assert ">High<" in card and ">Not required<" in card
    page = _html(at)
    assert "Why this route?" in page and "Clearly above the decision threshold" in page
    assert 'class="cqr-path"' in page and "Uncertainty check: clear" in page
    assert 'class="cqr-chips"' in page and "Low bounce rate" in page
    metrics = _metrics(at)
    assert metrics["Purchase probability (calibrated)"] == "66.7%"
    assert metrics["Decision threshold"] == "12.0%"
    for removed in ("Priority", "Recommended action", "Human approval"):
        assert removed not in metrics
    # Full routing = /predict then /decide, once each, for the same request.
    assert client.paths[-2:] == ["/predict", "/decide"]
    assert client.last_body("/decide")["request_id"].startswith("req_")
    # Request ID: short in the main view, full only inside Technical details.
    request_id = client.last_body("/decide")["request_id"]
    assert any(f"…{request_id[-8:]}" in c.value for c in at.caption)
    assert not any(request_id in c.value for c in at.caption)
    [details] = [e for e in at.expander if e.label == "Technical details"]
    assert any(request_id in m.value for m in details.markdown)
    assert len(at.json) == 3 and len(details.json) == 3


def test_borderline_preset_is_uncertain_and_rail_marks_band(api_factory):
    at = _evaluate(_app(api_factory(RuleBasedFakeProvider())), BORDERLINE)
    card = _card(at)
    assert 'data-route="HUMAN_REVIEW"' in card and ">Required<" in card
    metrics = _metrics(at)
    assert metrics["Purchase probability (calibrated)"] == "11.1%"
    assert metrics["Uncertainty"] == "Uncertain"
    page = _html(at)
    assert "Inside the uncertainty band" in page
    assert 'data-role="band" style="left:7.00%;width:10.00%"' in page
    assert 'data-role="probability" style="left:11.11%' in page
    assert 'data-role="threshold" style="left:12.00%"' in page
    assert "Uncertainty check: uncertain" in page


def test_weak_preset_is_logged_automatically(api_factory):
    at = _evaluate(_app(api_factory(RuleBasedFakeProvider())), WEAK)
    card = _card(at)
    assert 'data-route="LOG_ONLY"' in card and "Log automatically" in card and ">Low<" in card
    assert "Clearly below the decision threshold" in _html(at)


def test_model_only_mode_never_requests_a_decision(api_factory):
    client = api_factory(RuleBasedFakeProvider())
    at = _evaluate(_app(client), STRONG, mode="Model only")
    assert "/decide" not in client.paths
    page = _html(at)
    assert "Model-only evaluation" in page and "Route not requested" in page
    assert 'data-route=' not in page.split("cqr-path")[0]  # no decision card
    assert 'data-role="probability"' in page


def test_invalid_agent_output_shows_safety_fallback(api_factory):
    at = _evaluate(_app(api_factory(AlwaysInvalidProvider())), STRONG)
    card = _card(at)
    assert 'data-route="SYSTEM_FALLBACK"' in card and "Safety fallback: human review" in card
    assert ">Required<" in card


def test_empty_state_before_any_evaluation(api_factory):
    at = _app(api_factory(RuleBasedFakeProvider()))
    assert "No session evaluated yet" in _html(at)


# -- GA: inputs ---------------------------------------------------------------------------


def test_basic_mode_uses_friendly_units_and_shows_preset_assumptions(api_factory):
    at = _app(api_factory(RuleBasedFakeProvider()))
    assert at.radio(key="input_mode").value == "Basic"
    labels = {w.label for w in at.number_input}
    assert "Time on product pages (minutes)" in labels
    assert "Average bounce rate (%)" in labels and "Average exit rate (%)" in labels
    business, encoded = (df.value for df in at.dataframe[:2])
    assert len(business) == 5 and "Account pages viewed" in set(business["Input"])
    assert not any("category ID" in label for label in business["Input"])
    basic_keys = {w.key for w in (*at.number_input, *at.selectbox, *at.toggle)}
    assert not {k for k in basic_keys if k and k.endswith(("OperatingSystems", "Browser"))}


def test_basic_mode_groups_encoded_categories_in_a_collapsed_expander(api_factory):
    at = _app(api_factory(RuleBasedFakeProvider()))
    [expander] = [e for e in at.expander if e.label.startswith("Technical preset assumptions")]
    assert expander.proto.expanded is False
    [table] = [df.value for df in expander.dataframe]
    assert list(table["Input"]) == [
        "Operating-system category ID", "Browser category ID",
        "Region category ID", "Traffic-source category ID",
    ]
    preset = _fixture_session("real_model_priority_review.json")
    assert list(table["Value used"]) == [
        f"Category {preset[name]}" for name in ("OperatingSystems", "Browser", "Region",
                                                "TrafficType")
    ]
    assert any("nothing is guessed" in c.value for c in expander.caption)
    assert any("without assigning a real-world name" in c.value for c in expander.caption)
    assert any("required by the model contract" in c.value for c in at.caption)


UNSUPPORTED_NAMES = re.compile(
    r"\b(windows|macos|mac os|linux|android|ios|ubuntu|chrome|firefox|safari|edge|opera|"
    r"internet explorer|europe|asia|america|africa|google|facebook|bing|email|newsletter)\b",
    re.IGNORECASE,
)


def _all_visible_text(at: AppTest) -> str:
    parts = [m.value for m in at.markdown] + [c.value for c in at.caption]
    for widget in (*at.number_input, *at.selectbox, *at.toggle, *at.radio):
        parts.append(str(widget.label))
        parts.append(str(getattr(widget.proto, "help", "")))
        parts.extend(str(o) for o in getattr(widget.proto, "options", []))
    for df in at.dataframe:
        parts.append(df.value.to_string())
    return "\n".join(parts)


def test_no_invented_names_for_encoded_categories(api_factory):
    at = _app(api_factory(RuleBasedFakeProvider()))
    assert not UNSUPPORTED_NAMES.search(_all_visible_text(at))
    at.radio(key="input_mode").set_value("Advanced").run()
    text = _all_visible_text(at)
    assert not UNSUPPORTED_NAMES.search(text)
    for name in ("OperatingSystems", "Browser", "Region", "TrafficType"):
        widget = at.selectbox(key=f"adv_{name}")
        assert "category ID" in widget.label
        assert "without assigning a real-world name" in widget.proto.help


def test_advanced_encoded_category_round_trip(api_factory):
    client = api_factory(RuleBasedFakeProvider())
    at = _app(client)
    at.radio(key="input_mode").set_value("Advanced").run()
    at.selectbox(key="adv_TrafficType").set_value(5).run()
    at.button(key="evaluate_button").click().run()
    body = client.last_body("/predict")
    expected = _fixture_session("real_model_priority_review.json") | {"TrafficType": 5}
    assert body == expected
    at.radio(key="input_mode").set_value("Basic").run()
    [expander] = [e for e in at.expander if e.label.startswith("Technical preset assumptions")]
    assert "Category 5" in list(expander.dataframe[0].value["Value used"])


def test_untouched_preset_values_reach_the_api_exactly(api_factory):
    client = api_factory(RuleBasedFakeProvider())
    _evaluate(_app(client), STRONG)
    assert client.last_body("/predict") == _fixture_session("real_model_priority_review.json")


def test_basic_edits_convert_back_to_canonical_units(api_factory):
    client = api_factory(RuleBasedFakeProvider())
    at = _app(client)
    at.number_input(key="basic_ProductRelated_Duration").set_value(10.0).run()
    at.number_input(key="basic_BounceRates").set_value(5.0).run()
    at.button(key="evaluate_button").click().run()
    body = client.last_body("/predict")
    assert body["ProductRelated_Duration"] == 600.0
    assert body["BounceRates"] == pytest.approx(0.05)
    assert body["ExitRates"] == _fixture_session("real_model_priority_review.json")["ExitRates"]


def test_advanced_mode_exposes_all_sixteen_exact_inputs(api_factory):
    at = _app(api_factory(RuleBasedFakeProvider()))
    at.radio(key="input_mode").set_value("Advanced").run()
    keys = {w.key for w in (*at.number_input, *at.selectbox, *at.toggle) if w.key}
    assert len({k for k in keys if k.startswith("adv_")}) == 16
    assert at.number_input(key="adv_ProductRelated_Duration").value == pytest.approx(13453.25103)
    at.number_input(key="adv_BounceRates").set_value(0.25).run()
    at.radio(key="input_mode").set_value("Basic").run()
    assert at.number_input(key="basic_BounceRates").value == pytest.approx(25.0)


# -- GA: review queue ---------------------------------------------------------------------


def test_queue_empty_state_and_session_scope_notice(api_factory):
    at = _app(api_factory(RuleBasedFakeProvider()))
    page = _html(at)
    assert "The review queue is empty" in page
    assert any("belongs to this browser session only" in c.value for c in at.caption)


def test_loading_demo_sessions_twice_keeps_one_per_route(api_factory):
    at = _app(api_factory(RuleBasedFakeProvider()))
    for _ in range(2):
        at.button(key="route_demo_button").click().run()
        metrics = _metrics(at)
        assert (metrics["Priority review"], metrics["Human review"],
                metrics["Logged automatically"]) == ("1", "1", "1")
    stack = re.findall(r'data-route="(\w+)" data-count="(\d+)"', _html(at))
    assert sorted(stack) == [("HUMAN_REVIEW", "1"), ("LOG_ONLY", "1"), ("PRIORITY_REVIEW", "1")]
    assert "Priority review: 1 (33%)" in _html(at)


def test_demo_reload_keeps_user_rows_and_clear_empties_the_queue(api_factory):
    store = FeedbackStore(_tmp_db())
    at = _evaluate(_app(api_factory(RuleBasedFakeProvider()), store), STRONG)
    at.button(key="route_demo_button").click().run()
    at.button(key="route_demo_button").click().run()
    assert _metrics(at)["Priority review"] == "2"  # one user row + one demo row
    tables = [df.value for df in at.dataframe if "Source" in df.value.columns]
    sources = sorted(s for t in tables for s in t["Source"])
    assert sources == ["Demo", "Demo", "Demo", "You"]
    assert all(r.startswith("…") for t in tables for r in t["Request"])
    at.button(key="clear_queue_button").click().run()
    assert "The review queue is empty" in _html(at)


def test_unreachable_service_shows_a_clear_message():
    def refuse(request):
        raise httpx.ConnectError("refused", request=request)

    client = ApiClient(
        http_client=httpx.Client(
            base_url="http://127.0.0.1:9", transport=httpx.MockTransport(refuse)
        )
    )
    at = _app(client)
    assert "not reachable" in _sidebar_html(at)
    at.button(key="evaluate_button").click().run()
    assert any("not reachable" in e.value for e in at.error)


def test_timeout_shows_a_clear_message():
    def slow(request):
        raise httpx.ReadTimeout("slow", request=request)

    client = ApiClient(
        http_client=httpx.Client(
            base_url="http://127.0.0.1:9", transport=httpx.MockTransport(slow)
        )
    )
    at = _app(client)
    at.button(key="evaluate_button").click().run()
    assert any("did not answer in time" in e.value for e in at.error)


# -- GB: reviewer feedback and audit loop ------------------------------------------------


def _routed_borderline(api_factory, store, provider=None):
    at = _app(api_factory(provider or RuleBasedFakeProvider()), store)
    return _evaluate(at, BORDERLINE)


def test_feedback_has_no_default_response_and_record_starts_disabled(api_factory):
    store = FeedbackStore(_tmp_db())
    at = _routed_borderline(api_factory, store)
    assert at.radio(key="fb_response").value is None
    assert at.button(key="fb_submit").disabled  # AppTest also refuses clicks on it
    assert store.count() == 0


def test_agree_feedback_is_recorded_with_full_lineage(api_factory):
    store = FeedbackStore(_tmp_db())
    at = _routed_borderline(api_factory, store)
    assert any("does not retrain the model" in c.value for c in at.caption)
    at.radio(key="fb_response").set_value(HumanResponse.AGREE).run()
    assert not at.button(key="fb_submit").disabled
    at.button(key="fb_submit").click().run()
    assert not at.exception, at.exception
    assert "Feedback recorded" in _html(at)

    [record] = store.records()
    assert record.human_response == HumanResponse.AGREE
    assert record.agent_decision == record.final_decision == Decision.HUMAN_REVIEW
    assert record.uncertain is True
    assert record.purchase_probability == 0.1111
    assert record.model_version and record.agent_version and record.decision_schema_version
    [details] = [e for e in at.expander if e.label == "Technical details"]
    assert any(record.request_id in m.value for m in details.markdown)


def test_override_requires_decision_and_reason_before_recording(api_factory):
    store = FeedbackStore(_tmp_db())
    at = _routed_borderline(api_factory, store)
    at.radio(key="fb_response").set_value(HumanResponse.OVERRIDE).run()
    assert at.selectbox(key="fb_decision").value is None
    assert at.selectbox(key="fb_reason").value is None
    assert at.button(key="fb_submit").disabled
    at.selectbox(key="fb_decision").set_value(Decision.PRIORITY_REVIEW).run()
    assert at.button(key="fb_submit").disabled  # reason still missing
    at.selectbox(key="fb_reason").set_value(OverrideReason.UNCERTAINTY_RESOLVED_BY_REVIEW).run()
    at.text_area(key="fb_note").input("Returning buyer, long product browsing.").run()
    at.button(key="fb_submit").click().run()
    assert not at.exception, at.exception

    [record] = store.records()
    assert record.human_response == HumanResponse.OVERRIDE
    assert record.final_decision == Decision.PRIORITY_REVIEW
    assert record.override_reason == OverrideReason.UNCERTAINTY_RESOLVED_BY_REVIEW
    assert record.reviewer_note == "Returning buyer, long product browsing."


def test_override_choices_exclude_the_recommended_route(api_factory):
    at = _routed_borderline(api_factory, FeedbackStore(_tmp_db()))
    at.radio(key="fb_response").set_value(HumanResponse.OVERRIDE).run()
    assert Decision.HUMAN_REVIEW not in at.selectbox(key="fb_decision").options
    assert len(at.selectbox(key="fb_decision").options) == 2


def test_other_reason_without_note_is_not_recorded(api_factory):
    store = FeedbackStore(_tmp_db())
    at = _routed_borderline(api_factory, store)
    at.radio(key="fb_response").set_value(HumanResponse.OVERRIDE).run()
    at.selectbox(key="fb_decision").set_value(Decision.LOG_ONLY).run()
    at.selectbox(key="fb_reason").set_value(OverrideReason.OTHER).run()
    assert at.text_area(key="fb_note").label == "Note (required for 'Other')"
    at.button(key="fb_submit").click().run()
    assert any("Feedback not recorded" in e.value for e in at.error)
    assert store.count() == 0


def test_fallback_offers_only_an_explicit_reviewer_decision(api_factory):
    store = FeedbackStore(_tmp_db())
    at = _routed_borderline(api_factory, store, provider=AlwaysInvalidProvider())
    assert at.radio(key="fb_response").options == ["Override recommendation"]
    at.radio(key="fb_response").set_value(HumanResponse.OVERRIDE).run()
    at.selectbox(key="fb_decision").set_value(Decision.HUMAN_REVIEW).run()
    at.selectbox(key="fb_reason").set_value(OverrideReason.FALLBACK_RESOLVED_BY_REVIEW).run()
    at.button(key="fb_submit").click().run()
    [record] = store.records()
    assert record.agent_decision == Decision.SYSTEM_FALLBACK
    assert record.final_decision == Decision.HUMAN_REVIEW


def test_a_recorded_case_cannot_be_reviewed_again_from_the_ui(api_factory):
    store = FeedbackStore(_tmp_db())
    at = _routed_borderline(api_factory, store)
    at.radio(key="fb_response").set_value(HumanResponse.AGREE).run()
    at.button(key="fb_submit").click().run()
    at.run()  # next interaction: the panel now shows the stored record only
    assert not [b for b in at.button if b.key == "fb_submit"]
    assert "Feedback recorded" in _html(at)
    assert store.count() == 1


def test_feedback_tab_disclaimer_and_empty_state(api_factory):
    at = _app(api_factory(RuleBasedFakeProvider()))
    page = _html(at)
    assert "entered manually" in page and "not a user study" in page
    assert "not a measure of model performance" in page
    assert "never changes the model" in page
    assert "No reviewer feedback recorded yet" in page


def test_summary_tab_reports_counts_rates_and_survives_restart(api_factory):
    path = _tmp_db()
    store = FeedbackStore(path)
    at = _routed_borderline(api_factory, store)
    at.radio(key="fb_response").set_value(HumanResponse.AGREE).run()
    at.button(key="fb_submit").click().run()

    at = _evaluate(at, STRONG)
    at.radio(key="fb_response").set_value(HumanResponse.OVERRIDE).run()
    at.selectbox(key="fb_decision").set_value(Decision.HUMAN_REVIEW).run()
    at.selectbox(key="fb_reason").set_value(OverrideReason.CONTEXT_NOT_IN_FEATURES).run()
    at.button(key="fb_submit").click().run()
    assert store.count() == 2

    # "Restart": a fresh app session with a fresh store object on the same file.
    restarted = _app(api_factory(RuleBasedFakeProvider()), FeedbackStore(path))
    metrics = _metrics(restarted)
    assert metrics["Total reviewed"] == "2"
    assert metrics["Agreed"] == "1 (50%)"
    assert metrics["Overridden"] == "1 (50%)"
    # Below the minimum record count the proportional bar is not drawn.
    assert "Illustrative local feedback" not in _html(restarted)
    tables = [df.value for df in restarted.dataframe]
    by_route = next(t for t in tables if "Override rate" in t.columns)
    assert set(by_route["Recommended route"]) == {"Human review", "Priority review"}
    reasons = next(t for t in tables if "Override reason" in t.columns and "Count" in t.columns)
    assert list(reasons["Override reason"]) == ["Reviewer has context the model does not see"]
    recent = next(t for t in tables if "Event ID" in t.columns)
    assert len(recent) == 2
    assert "Reason" not in recent.columns
    by_response = dict(zip(recent["Response"], recent["Override reason"], strict=True))
    assert by_response["Agree with recommendation"] == "Not applicable — reviewer agreed"
    assert by_response["Override recommendation"] == "Reviewer has context the model does not see"
    # Presentation only: the stored agree record still carries no reason.
    agree = next(r for r in FeedbackStore(path).records() if r.human_response == "AGREE")
    assert agree.override_reason is None


def test_feedback_bar_appears_with_enough_records_and_is_labelled_illustrative(api_factory):
    path = _tmp_db()
    store = FeedbackStore(path)
    at = _app(api_factory(RuleBasedFakeProvider()), store)
    for preset in (STRONG, BORDERLINE, WEAK):
        at = _evaluate(at, preset)
        at.radio(key="fb_response").set_value(HumanResponse.AGREE).run()
        at.button(key="fb_submit").click().run()
    assert store.count() == 3
    at.run()
    assert "Illustrative local feedback, n=3" in _html(at)
