"""Streamlit entry point for the local decision-support UI.

Run with `scripts/start_ui.sh`. The page talks to the FastAPI service over
HTTP only (`conversion_router.ui.api_client`); it never loads the model or
decides a route itself. Reviewer feedback goes to the local append-only store
in `conversion_router.feedback`. Tests inject `st.session_state["api_client"]`,
`st.session_state["feedback_store"]`, and optionally `run_mode` and
`display_policy`.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st
from pydantic import ValidationError

from conversion_router.feedback.schemas import (
    HUMAN_DECISIONS,
    FeedbackError,
    FeedbackSubmission,
    HumanResponse,
    OverrideReason,
    build_audit_record,
)
from conversion_router.feedback.store import (
    CorruptRecordError,
    DuplicateFeedbackError,
    FeedbackStore,
    IncompatibleStoreError,
)
from conversion_router.feedback.summary import summarize
from conversion_router.schemas import Decision, Priority, SessionFeaturesRequest
from conversion_router.ui import components, features, presentation, settings, theme
from conversion_router.ui.api_client import ApiClient, ApiError, ApiErrorKind, RoutedResult

MODE_ROUTING = "Full decision routing"
MODE_MODEL_ONLY = "Model only"
INPUT_BASIC = "Basic"
INPUT_ADVANCED = "Advanced"
ORIGIN_DEMO = "demo"
ORIGIN_USER = "user"
FEEDBACK_CHART_MIN_RECORDS = 3

_ERROR_MESSAGES = {
    ApiErrorKind.UNREACHABLE: "The decision service is not reachable. Start the local "
    "environment with `scripts/start_ui.sh` and try again.",
    ApiErrorKind.TIMEOUT: "The decision service did not answer in time. Nothing was "
    "recorded. Try again.",
    ApiErrorKind.INVALID_INPUT: "The service rejected this session input. Check the values "
    "against the allowed ranges.",
    ApiErrorKind.MODEL_UNAVAILABLE: "The model is not loaded on the service (artifacts "
    "missing or failed integrity checks). No prediction was made.",
    ApiErrorKind.CONTRACT_VIOLATION: "The service returned an unexpected response. It was "
    "not shown, to avoid displaying unverified results.",
    ApiErrorKind.HTTP_ERROR: "The decision service returned an error.",
}

_PRIORITY_RANK = {Priority.HIGH: 0, Priority.MEDIUM: 1, Priority.LOW: 2}


def html(markup: str, container=None) -> None:
    (container or st).markdown(markup, unsafe_allow_html=True)


# -- state helpers ---------------------------------------------------------


def get_client() -> ApiClient:
    if "api_client" not in st.session_state:
        st.session_state["api_client"] = ApiClient.from_env()
    return st.session_state["api_client"]


def get_store() -> FeedbackStore:
    if "feedback_store" not in st.session_state:
        st.session_state["feedback_store"] = FeedbackStore.from_env()
    return st.session_state["feedback_store"]


def get_run_mode() -> settings.RunMode:
    if "run_mode" not in st.session_state:
        st.session_state["run_mode"] = settings.run_mode()
    return st.session_state["run_mode"]


def get_display_policy() -> settings.DisplayPolicy | None:
    if "display_policy" not in st.session_state:
        st.session_state["display_policy"] = settings.load_display_policy()
    return st.session_state["display_policy"]


FEEDBACK_WIDGET_KEYS = ("fb_response", "fb_decision", "fb_reason", "fb_note")


def _reset_feedback_widgets() -> None:
    for key in FEEDBACK_WIDGET_KEYS:
        st.session_state.pop(key, None)


def _queue() -> list[dict]:
    return st.session_state.setdefault("queue", [])


@st.cache_data
def _cached_presets() -> list[features.Preset]:
    return features.load_presets()


def _presets_by_label() -> dict[str, features.Preset]:
    return {preset.label: preset for preset in _cached_presets()}


def _load_preset_into_inputs() -> None:
    preset = _presets_by_label()[st.session_state["preset_label"]]
    st.session_state["values"] = preset.session.model_dump()
    for key in [k for k in st.session_state if str(k).startswith(("basic_", "adv_"))]:
        del st.session_state[key]
    st.session_state["preset_loaded"] = preset.key


def show_api_error(error: ApiError) -> None:
    message = _ERROR_MESSAGES.get(error.kind, "The decision service returned an error.")
    detail = f" ({error.message})" if error.kind == ApiErrorKind.MODEL_UNAVAILABLE else ""
    reference = f" Reference: `{error.request_id}`." if error.request_id else ""
    st.error(f"{message}{detail}{reference}")


# -- sidebar ---------------------------------------------------------------


def render_sidebar(client: ApiClient, mode: settings.RunMode) -> None:
    st.sidebar.subheader("Service status")
    try:
        info = client.ready()
    except ApiError as error:
        html(
            components.status_line(_ERROR_MESSAGES.get(error.kind, error.message), theme.DANGER),
            st.sidebar,
        )
    else:
        html(components.status_line("Decision service ready", theme.SUCCESS), st.sidebar)
        st.sidebar.caption(
            f"Model: {info.model_name} {info.model_version} · "
            f"feature contract {info.feature_contract_version}"
        )
    st.sidebar.subheader("Run mode")
    tone = theme.NEUTRAL if mode == settings.RunMode.UNDECLARED else theme.PRIMARY_TONE
    html(components.status_line(settings.RUN_MODE_TEXT[mode], tone), st.sidebar)
    st.sidebar.subheader("Automation boundary")
    st.sidebar.caption(
        "This interface previews actions only; it never starts the n8n workflow or sends "
        "Slack messages."
    )
    st.sidebar.markdown(
        "\n".join(f"- **{who}:** {what}" for who, what in settings.AUTOMATION_BOUNDARY)
    )


# -- input form ------------------------------------------------------------


def _sync_from_widget(name: str, key: str, basic: bool) -> None:
    value = st.session_state[key]
    st.session_state["values"][name] = features.to_canonical(name, value) if basic else value
    st.session_state.pop(f"{'adv' if basic else 'basic'}_{name}", None)


def _render_field(spec: features.FieldSpec, container, basic: bool) -> None:
    name = spec.name
    key = f"{'basic' if basic else 'adv'}_{name}"
    canonical = st.session_state["values"][name]
    if key not in st.session_state:
        st.session_state[key] = features.to_display(name, canonical) if basic else canonical
    common = {"key": key, "on_change": _sync_from_widget, "args": (name, key, basic)}
    annotation = SessionFeaturesRequest.model_fields[name].annotation
    label = features.basic_label(spec) if basic else spec.label

    if name in features.CATEGORICAL_WHITELISTS:
        container.selectbox(
            label,
            features.categorical_options(name),
            format_func=lambda v, n=name: features.option_label(n, v),
            help=spec.help,
            **common,
        )
    elif annotation is bool:
        container.toggle(label, help=spec.help, **common)
    elif annotation is int:
        container.number_input(label, min_value=0, step=1, help=spec.help, **common)
    elif basic and name in features.MINUTE_FIELDS:
        container.number_input(
            label, min_value=0.0, step=1.0, format="%.1f",
            help=spec.help + " Shown in minutes; the service receives seconds.", **common,
        )
        container.caption(f"≈ {features.readable_duration(canonical)}")
    elif basic and name in features.PERCENT_FIELDS:
        container.number_input(
            label, min_value=0.0, max_value=100.0, step=0.1, format="%.2f",
            help=spec.help + " Shown in percent; the service receives a 0-1 rate.", **common,
        )
    else:
        unit_interval = name in {"BounceRates", "ExitRates", "SpecialDay"}
        container.number_input(
            label,
            min_value=0.0,
            max_value=1.0 if unit_interval else None,
            step=spec.step,
            format="%.6f" if unit_interval else "%.4f",
            help=spec.help,
            **common,
        )


def render_input_form(input_mode: str) -> None:
    if input_mode == INPUT_BASIC:
        cols = st.columns(2)
        for i, spec in enumerate(features.basic_fields()):
            _render_field(spec, cols[i % 2], basic=True)
        values = st.session_state["values"]

        def table(specs):
            rows = [
                {"Input": spec.label, "Value used": features.readable_value(spec.name,
                                                                            values[spec.name])}
                for spec in specs
            ]
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

        st.markdown("**Preset assumptions** (sent unchanged; switch to Advanced to edit)")
        table(features.preset_business_fields())
        with st.expander("Technical preset assumptions: encoded category IDs", expanded=False):
            st.caption(
                "These four values come from the selected preset; nothing is guessed. "
                + features.ENCODED_CATEGORY_HELP
            )
            table(features.encoded_category_fields())
        st.caption(
            "Operating-system, browser, region, and traffic-source category IDs are encoded "
            "fields required by the model contract. They have no real-world names here."
        )
    else:
        st.caption("Advanced mode shows the exact values sent to the service.")
        for group in features.GROUPS:
            st.markdown(f"**{group}**")
            specs = [s for s in features.FIELD_SPECS if s.group == group]
            cols = st.columns(2)
            for i, spec in enumerate(specs):
                _render_field(spec, cols[i % 2], basic=False)


def collect_session() -> SessionFeaturesRequest | None:
    try:
        return SessionFeaturesRequest.model_validate(st.session_state["values"])
    except ValidationError as exc:
        problems = sorted(
            {features.FIELD_BY_NAME[str(err["loc"][0])].label for err in exc.errors()}
        )
        st.error("Some inputs are outside the allowed range: " + ", ".join(problems) + ".")
        return None


# -- results ---------------------------------------------------------------


def render_model_evidence(prediction, tone: theme.Tone) -> None:
    detail = prediction.prediction
    cols = st.columns(3)
    cols[0].metric(
        "Purchase probability (calibrated)",
        presentation.format_probability(detail.purchase_probability),
    )
    cols[1].metric("Decision threshold", presentation.format_probability(detail.decision_threshold))
    cols[2].metric("Uncertainty", "Uncertain" if detail.uncertain else "Clear")
    band = settings.band_for(detail.decision_threshold, get_display_policy())
    html(
        components.probability_rail(
            detail.purchase_probability, detail.decision_threshold, detail.uncertain, band, tone
        )
    )
    st.markdown("**Supporting signals** (rule-based context returned by the service)")
    html(components.signal_chips(prediction))


def render_reference_line(prediction, decision=None) -> None:
    parts = [
        f"Request {presentation.short_request_id(prediction.request_id)}",
        f"model {prediction.model.name} {prediction.model.version}",
    ]
    if decision is not None:
        parts.append(f"policy {presentation.policy_version_text(decision)}")
    st.caption(" · ".join(parts))


def render_technical_details(prediction, session, decision=None) -> None:
    with st.expander("Technical details"):
        st.markdown(f"Full request ID: `{prediction.request_id}`")
        st.caption("Request sent to the service (exact contract values)")
        st.json(session.model_dump(mode="json"), expanded=False)
        st.caption("Prediction response")
        st.json(prediction.model_dump(mode="json"), expanded=False)
        if decision is not None:
            st.caption("Decision response")
            st.json(decision.model_dump(mode="json"), expanded=False)


def render_routed_result(result: RoutedResult, session) -> None:
    decision = result.decision
    tone = theme.ROUTE_TONES[decision.decision]
    html(components.decision_card(decision))
    html(components.why_this_route(decision))
    html(components.decision_path(decision, result.prediction.prediction.uncertain))
    st.markdown("##### Model evidence")
    render_model_evidence(result.prediction, tone)
    render_reference_line(result.prediction, decision)
    render_technical_details(result.prediction, session, decision)


def render_model_only(prediction, session) -> None:
    html(
        components.empty_state(
            "Model-only evaluation",
            f"No routing decision was requested. Switch to {MODE_ROUTING} to see the "
            "recommended route and action.",
        )
    )
    html(components.decision_path(None, prediction.prediction.uncertain))
    render_model_evidence(prediction, theme.PRIMARY_TONE)
    render_reference_line(prediction)
    render_technical_details(prediction, session)


# -- reviewer feedback -----------------------------------------------------


def render_feedback_panel(store: FeedbackStore, result: RoutedResult, session) -> None:
    st.markdown("---")
    st.markdown("##### Reviewer feedback")
    st.caption(presentation.FEEDBACK_NOTICE)
    try:
        existing = store.get_by_request(result.request_id)
    except (CorruptRecordError, IncompatibleStoreError) as error:
        st.error(f"The local feedback store cannot be read: {error}")
        return
    if existing is not None:
        html(
            components.status_line(
                f"Feedback recorded: {presentation.HUMAN_RESPONSE_LABELS[existing.human_response]}."
                f" Final decision: {presentation.ROUTE_DISPLAY[existing.final_decision].title}. "
                f"Audit event {existing.event_id}.",
                theme.SUCCESS,
            )
        )
        return

    decision = result.decision
    responses = list(HumanResponse)
    if presentation.is_fallback(decision):
        st.info(
            "This is a safety fallback, so there is no recommendation to agree with. "
            "Choose the decision you want to record."
        )
        responses = [HumanResponse.OVERRIDE]
    response = st.radio(
        "Your response",
        responses,
        index=None,
        format_func=lambda r: presentation.HUMAN_RESPONSE_LABELS[r],
        key="fb_response",
        horizontal=True,
    )
    override_decision = override_reason = None
    if response == HumanResponse.OVERRIDE:
        cols = st.columns(2)
        override_decision = cols[0].selectbox(
            "Your decision",
            [d for d in HUMAN_DECISIONS if d != decision.decision],
            index=None,
            placeholder="Choose a decision",
            format_func=lambda d: presentation.ROUTE_DISPLAY[d].title,
            key="fb_decision",
        )
        override_reason = cols[1].selectbox(
            "Reason for the override",
            list(OverrideReason),
            index=None,
            placeholder="Choose a reason",
            format_func=lambda r: presentation.OVERRIDE_REASON_LABELS[r],
            key="fb_reason",
        )
    note_label = (
        "Note (required for 'Other')"
        if override_reason == OverrideReason.OTHER
        else "Note (optional)"
    )
    note = st.text_area(note_label, max_chars=500, key="fb_note")

    ready = response == HumanResponse.AGREE or (
        response == HumanResponse.OVERRIDE
        and override_decision is not None
        and override_reason is not None
    )
    if not ready:
        st.caption(
            "Choose Agree or Override (with a decision and a reason) to enable recording."
        )
    recorded = st.session_state.setdefault("fb_recorded", set())
    clicked = st.button(
        "Record feedback",
        key="fb_submit",
        disabled=not ready or result.request_id in recorded,
    )
    if not clicked or result.request_id in recorded:
        return
    try:
        submission = FeedbackSubmission(
            response=response,
            override_decision=override_decision,
            override_reason=override_reason,
            note=note,
        )
        record = build_audit_record(session, result.prediction, decision, submission)
        store.append(record)
    except ValidationError as exc:
        st.error("Feedback not recorded: " + "; ".join(e["msg"] for e in exc.errors()))
        return
    except FeedbackError as exc:
        st.error(f"Feedback not recorded: {exc}")
        return
    except DuplicateFeedbackError:
        st.warning("This request already has recorded feedback. Records are append-only.")
        return
    recorded.add(result.request_id)
    html(
        components.status_line(
            f"Feedback recorded. Final decision: "
            f"{presentation.ROUTE_DISPLAY[record.final_decision].title}. "
            f"Audit event {record.event_id}.",
            theme.SUCCESS,
        )
    )


# -- tabs ------------------------------------------------------------------


def _remember(result: RoutedResult, origin: str) -> None:
    _queue().append({"origin": origin, "result": result})


def render_evaluate_tab(client: ApiClient, store: FeedbackStore, mode: settings.RunMode) -> None:
    labels = list(_presets_by_label())
    preset_col, mode_col = st.columns([2, 1])
    preset_col.selectbox(
        "Start from a session preset",
        labels,
        key="preset_label",
        on_change=_load_preset_into_inputs,
    )
    if "preset_loaded" not in st.session_state:
        _load_preset_into_inputs()
    preset_col.caption(_presets_by_label()[st.session_state["preset_label"]].description)
    input_mode = mode_col.radio(
        "Input mode", [INPUT_BASIC, INPUT_ADVANCED], horizontal=True, key="input_mode"
    )
    evaluation_mode = mode_col.radio(
        "Evaluation",
        [MODE_ROUTING, MODE_MODEL_ONLY],
        key="evaluation_mode",
        captions=[settings.ROUTING_HELP[mode], "Probability only, no route or action."],
    )

    render_input_form(input_mode)

    if st.button("Evaluate session", type="primary", key="evaluate_button"):
        session = collect_session()
        if session is None:
            return
        _reset_feedback_widgets()
        try:
            with st.status("Validating session…", expanded=False) as status:
                status.update(label="Running prediction…")
                prediction = client.predict(session)
                if evaluation_mode == MODE_ROUTING:
                    status.update(label="Applying decision policy…")
                    result = client.pair(prediction, client.decide(prediction))
                    status.update(label="Preparing result…")
                    _remember(result, ORIGIN_USER)
                    st.session_state["last_output"] = ("routed", result, session)
                else:
                    status.update(label="Preparing result…")
                    st.session_state["last_output"] = ("model_only", prediction, session)
                status.update(label="Result ready", state="complete")
        except ApiError as error:
            st.session_state.pop("last_output", None)
            show_api_error(error)
            return

    output = st.session_state.get("last_output")
    if output is None:
        html(
            components.empty_state(
                "No session evaluated yet",
                "Choose a preset, adjust the inputs if needed, and select Evaluate session.",
            )
        )
        return
    st.subheader("Result")
    kind, payload, session = output
    if kind == "routed":
        render_routed_result(payload, session)
        render_feedback_panel(store, payload, session)
    else:
        render_model_only(payload, session)


def queue_rows(entries: list[dict]) -> list[dict]:
    ordered = sorted(
        entries,
        key=lambda e: (
            _PRIORITY_RANK[e["result"].decision.priority],
            -e["result"].prediction.prediction.purchase_probability,
        ),
    )
    return [
        {
            "Request": presentation.short_request_id(e["result"].request_id),
            "Source": "Demo" if e["origin"] == ORIGIN_DEMO else "You",
            "Route": presentation.route_display(e["result"].decision).title,
            "Priority": presentation.PRIORITY_LABELS[e["result"].decision.priority],
            "Purchase probability": presentation.format_probability(
                e["result"].prediction.prediction.purchase_probability
            ),
            "Uncertain": "Yes" if e["result"].prediction.prediction.uncertain else "No",
            "Action": presentation.ACTION_LABELS[e["result"].decision.allowed_action],
        }
        for e in ordered
    ]


def render_queue_tab(client: ApiClient) -> None:
    st.caption(
        "This queue belongs to this browser session only. It is not saved, and clearing it "
        "does not touch the feedback audit records."
    )
    load_col, clear_col = st.columns(2)
    if load_col.button("Load one demo session for each route", key="route_demo_button",
                       help="Replaces earlier demo rows; sessions you evaluated stay."):
        try:
            demo_rows = [
                {"origin": ORIGIN_DEMO, "result": client.route(preset.session)}
                for preset in _cached_presets()
            ]
        except ApiError as error:
            show_api_error(error)
        else:
            st.session_state["queue"] = [
                e for e in _queue() if e["origin"] != ORIGIN_DEMO
            ] + demo_rows
    if clear_col.button("Clear current queue", key="clear_queue_button"):
        st.session_state["queue"] = []

    entries = _queue()
    if not entries:
        html(
            components.empty_state(
                "The review queue is empty",
                "Evaluate a session with full decision routing, or load one demo session for "
                "each route.",
            )
        )
        return
    counts = {route: 0 for route in Decision}
    for entry in entries:
        counts[entry["result"].decision.decision] += 1
    shown = [
        (Decision.PRIORITY_REVIEW, "Priority review"),
        (Decision.HUMAN_REVIEW, "Human review"),
        (Decision.LOG_ONLY, "Logged automatically"),
    ]
    if counts[Decision.SYSTEM_FALLBACK]:
        shown.insert(2, (Decision.SYSTEM_FALLBACK, "Safety fallback"))
    cols = st.columns(len(shown))
    for col, (route, title) in zip(cols, shown, strict=True):
        col.metric(title, counts[route])
    html(components.queue_composition(counts))
    for route, title in shown:
        group = [e for e in entries if e["result"].decision.decision == route]
        if group:
            st.markdown(f"**{title}**")
            st.dataframe(pd.DataFrame(queue_rows(group)), hide_index=True, width="stretch")
    with st.expander("Full request IDs"):
        for entry in entries:
            st.markdown(f"`{entry['result'].request_id}`")


def _rate(value: float) -> str:
    return f"{value * 100:.0f}%"


def render_feedback_tab(store: FeedbackStore) -> None:
    html(components.empty_state("About these numbers", presentation.FEEDBACK_DISCLAIMER))
    try:
        records = store.records()
    except (CorruptRecordError, IncompatibleStoreError) as error:
        st.error(f"The local feedback store cannot be read: {error}")
        return
    if not records:
        html(
            components.empty_state(
                "No reviewer feedback recorded yet",
                "Evaluate a session with full decision routing and record Agree or Override.",
            )
        )
        return
    summary = summarize(records)
    cols = st.columns(3)
    cols[0].metric("Total reviewed", summary.total_reviewed)
    cols[1].metric("Agreed", f"{summary.agreed} ({_rate(summary.agree_rate)})")
    cols[2].metric("Overridden", f"{summary.overridden} ({_rate(summary.override_rate)})")
    if summary.total_reviewed >= FEEDBACK_CHART_MIN_RECORDS:
        html(components.feedback_balance(summary.agreed, summary.overridden))

    st.markdown("**Overrides by recommended route**")
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Recommended route": presentation.ROUTE_DISPLAY[r.route].title,
                    "Reviewed": r.reviewed,
                    "Agreed": r.agreed,
                    "Overridden": r.overridden,
                    "Override rate": _rate(r.override_rate),
                }
                for r in summary.by_route
            ]
        ),
        hide_index=True,
        width="stretch",
    )
    if summary.override_reasons:
        st.markdown("**Most frequent override reasons**")
        st.dataframe(
            pd.DataFrame(
                [
                    {"Override reason": presentation.OVERRIDE_REASON_LABELS[reason], "Count": n}
                    for reason, n in summary.override_reasons
                ]
            ),
            hide_index=True,
            width="stretch",
        )
        st.markdown("**Override direction**")
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Recommended": presentation.ROUTE_DISPLAY[src].title,
                        "Reviewer chose": presentation.ROUTE_DISPLAY[dst].title,
                        "Count": n,
                    }
                    for src, dst, n in summary.transitions
                ]
            ),
            hide_index=True,
            width="stretch",
        )

    st.markdown("**Recent audit records** (newest first)")
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Recorded at (UTC)": r.created_at,
                    "Request ID": r.request_id,
                    "Probability": presentation.format_probability(r.purchase_probability),
                    "Uncertain": "Yes" if r.uncertain else "No",
                    "Recommended": presentation.ROUTE_DISPLAY[r.agent_decision].title,
                    "Response": presentation.HUMAN_RESPONSE_LABELS[r.human_response],
                    "Final decision": presentation.ROUTE_DISPLAY[r.final_decision].title,
                    "Override reason": presentation.override_reason_text(r.override_reason),
                    "Note": r.reviewer_note or "",
                    "Model / agent": f"{r.model_version} / {r.agent_version}",
                    "Event ID": r.event_id,
                }
                for r in reversed(records[-20:])
            ]
        ),
        hide_index=True,
        width="stretch",
    )


def main() -> None:
    st.set_page_config(page_title="Conversion Quality Router", layout="wide")
    html(theme.stylesheet())
    st.title("Conversion Quality Router")
    st.caption(
        "Decision-support prototype: routes e-commerce sessions to automatic logging, "
        "priority review, or human review using a calibrated probability, an uncertainty "
        "band, and a constrained decision policy. Runs on a public dataset; not connected "
        "to a live event stream."
    )
    client = get_client()
    store = get_store()
    mode = get_run_mode()
    render_sidebar(client, mode)
    evaluate_tab, queue_tab, feedback_tab = st.tabs(
        ["Evaluate a session", "Review queue", "Feedback and audit"]
    )
    with evaluate_tab:
        render_evaluate_tab(client, store, mode)
    with queue_tab:
        render_queue_tab(client)
    with feedback_tab:
        render_feedback_tab(store)


main()
