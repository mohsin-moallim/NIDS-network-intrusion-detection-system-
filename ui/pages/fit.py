"""02 Fit station: choose the detection mode, the features and the channels, then fit them on the sample.

Fitting starts only when the form's Fit button is pressed: its ``on_click`` callback reads the submitted options and
starts the job before the page is drawn, so the same rerun already shows the progress panel and a disabled Fit
button. Every other interaction (changing an option without submitting, visiting another station, coming back) just
redraws the stored run: nothing is refitted and the sample is not prepared again. The form starts from the options
of the last fit (or the Bench defaults before the first).
"""

from __future__ import annotations

import html

import streamlit as st

from nids.data.prepare import PreparedDataset
from nids.data.sampling import SingleClassError, target_for_mode
from nids.features import DEFAULT_K, FEATURE_MODES, select_features
from nids.models.train import TrainingRun, TrainRequest
from nids.schema import is_normal_traffic
from nids.settings import SVM_CAP_RANGE, AppSettings
from ui import components, state, training_ui

MODES: tuple[str, ...] = ("binary", "multiclass")
MODE_LABELS = {"binary": "Binary — normal vs attack", "multiclass": "Multi-class — name the attack"}
K_RANGE = (5, 40)
PORT_HELP = ("Adds the server port number as a feature. A port names a service rather than measuring a flow, and in "
             "a lab capture it can single out the attacked machines, so a channel may learn the port instead of "
             "the behaviour and score better than it would on other traffic. Off by default.")
WEIGHT_HELP = ("Weights every training row by n / (classes x rows in its class), capped (100, or 50 for the MLP) "
               "and rescaled to sum to n, so rare attacks are not ignored. Off gives every row the same weight.")


def _feature_label(mode: str, counts: dict[str, int]) -> str:
    """Radio text for a feature-set mode, with the number of columns where it is known before fitting."""
    if mode == "curated":
        return f"Curated ({counts.get('curated', 28)} columns)"
    if mode == "all":
        return f"All numeric ({counts['all']} columns)" if "all" in counts else "All numeric"
    return "Top-K by importance"


def _feature_counts(prepared: PreparedDataset) -> dict[str, int]:
    """Columns in the curated and all-numeric sets for this sample (degenerate columns left out)."""
    counts: dict[str, int] = {}
    for mode in ("curated", "all"):
        try:
            counts[mode] = select_features(mode, degenerate=prepared.degenerate).n_features  # type: ignore[arg-type]
        except ValueError:
            continue
    return counts


def _sample_line(prepared: PreparedDataset) -> None:
    """One line on the loaded sample: source, rows and classes."""
    request = prepared.request
    if request.source == "synthetic":
        source = "synthetic flows"
    else:
        source = f"{len(request.files)} CIC-IDS2017 file{'s' if len(request.files) != 1 else ''}"
    counts = prepared.class_counts
    normal = sum(n for name, n in counts.items() if is_normal_traffic(name))
    attack = sum(counts.values()) - normal
    merged = ", Web Attack types merged" if request.merge_web_attacks else ""
    st.markdown(
        f'On the bench: <span class="g-mono">{prepared.rows_sampled:,}</span> sampled rows from {html.escape(source)}, '
        f'{len(counts)} classes{merged} (<span class="g-mono">{normal:,}</span> normal, '
        f'<span class="g-mono">{attack:,}</span> attack), fingerprint '
        f'<span class="g-mono">{prepared.fingerprint[:12]}</span>.',
        unsafe_allow_html=True,
    )


def _fit_form(prepared: PreparedDataset, current: AppSettings, last: TrainRequest | None, running: bool) -> None:
    """The Fit form; pressing Fit calls :func:`_on_fit` (disabled while a fit of this session runs)."""
    counts = _feature_counts(prepared)
    channel_keys = list(training_ui.channel_options())
    mode_default = last.mode if last else "binary"
    features_default = last.feature_mode if last else "curated"
    k_default = int(min(max(last.top_k if last else DEFAULT_K, K_RANGE[0]), K_RANGE[1]))
    channels_default = [k for k in channel_keys if k in last.channels] if last else channel_keys
    min_default = int(last.min_class_count if last else current.min_class_count)
    cap_default = int(min(max(last.svm_cap if last else current.svm_cap, SVM_CAP_RANGE[0]), SVM_CAP_RANGE[1]))
    share_default = float(min(max(round(last.test_share if last else current.test_share, 2), 0.10), 0.50))
    with st.form("fit_form", border=True):
        left, right = st.columns(2, gap="large")
        with left:
            st.radio("Detection mode", MODES, index=MODES.index(mode_default),
                     format_func=lambda m: MODE_LABELS[m], key="fit_mode",
                     help="Binary asks only whether a flow is an attack. Multi-class asks which attack.")
            st.radio("Feature set", FEATURE_MODES, index=FEATURE_MODES.index(features_default),
                     format_func=lambda m: _feature_label(m, counts), key="fit_features",
                     help="Curated: 28 columns chosen for what they measure. All numeric: every "
                     "column except the port and the degenerate ones. Top-K: the K columns a small "
                     "XGBoost model finds most useful on the training split.")
            st.number_input("K, columns kept by Top-K", min_value=K_RANGE[0], max_value=K_RANGE[1],
                            value=k_default, step=1, key="fit_k",
                            help="Applies to Top-K only; the other feature sets ignore it.")
            st.caption("K applies only to Top-K. The ranking runs inside the fit, on training rows only.")
            st.checkbox("Include Destination Port", value=bool(last.include_port) if last else False,
                        key="fit_port", help=PORT_HELP)
        with right:
            # Pills wrap onto a second line in a narrow column, so all five channels stay readable.
            st.pills("Channels", channel_keys, selection_mode="multi", default=channels_default,
                     format_func=training_ui.channel_label, key="fit_channels",
                     help="Each channel is one model family; all of them see the same split. Click a channel to "
                          "add or remove it.")
            st.toggle("Balanced class weights", value=bool(last.balanced) if last else True,
                      key="fit_balanced", help=WEIGHT_HELP)
            st.number_input("Minimum rows per class (multi-class only)", min_value=10, max_value=100_000,
                            value=min_default, step=10, key="fit_min_rows",
                            help="In multi-class mode, classes with fewer rows are left out and "
                            "reported. Every mode leaves out classes under 10 rows.")
            svm_cap = st.number_input("SVM cap (training rows for CH3)", min_value=SVM_CAP_RANGE[0],
                                      max_value=SVM_CAP_RANGE[1], value=cap_default, step=1_000, key="fit_svm_cap",
                                      help="CH3 draws at most this many training rows, keeping rare classes.")
            st.caption(f"CH3 trains on at most {int(svm_cap):,} rows; kernel SVM fit time grows at least "
                       "quadratically with rows.")
            st.slider("Test share", min_value=0.10, max_value=0.50, value=share_default, step=0.05,
                      key="fit_test_share", help="Share of every class held out to measure the channels.")
        origin = ("Options repeat your last fit; the Bench holds the defaults." if last is not None
                  else "Options start from the Bench defaults.")
        st.caption(f"Seed {current.seed}, set on the Bench. {origin}")
        st.form_submit_button("Fit", type="primary", key="fit_submit", disabled=running, on_click=_on_fit)


def _form_request(prepared: PreparedDataset, current: AppSettings) -> TrainRequest | None:
    """The request described by the submitted Fit form (read from its widget keys), or None with the reason left as
    the fit message."""
    values = st.session_state
    channel_keys = list(training_ui.channel_options())
    chosen = values.get("fit_channels") or []
    channels = tuple(k for k in channel_keys if k in chosen)
    if not channels:
        state.set_fit_notice("warning", "Choose at least one channel to fit.")
        return None
    try:
        return TrainRequest(
            mode=values.get("fit_mode", "binary"), feature_mode=values.get("fit_features", "curated"),
            top_k=int(values.get("fit_k", DEFAULT_K)), include_port=bool(values.get("fit_port", False)),
            channels=channels, balanced=bool(values.get("fit_balanced", True)),
            min_class_count=int(values.get("fit_min_rows", current.min_class_count)),
            svm_cap=int(values.get("fit_svm_cap", current.svm_cap)),
            test_share=float(values.get("fit_test_share", current.test_share)), seed=int(current.seed),
            conflict_policy=prepared.request.conflict_policy,  # type: ignore[arg-type]
            profile=training_ui.training_profile(),
        )
    except ValueError as exc:
        state.set_fit_notice("warning", f"Nothing was fitted. {exc}")
        return None


def _precheck(prepared: PreparedDataset, request: TrainRequest) -> bool:
    """True when the sample gives at least two classes for the chosen mode; otherwise leave the reason and say no.

    This runs the same target rule the fit itself starts with, so a sample that holds one class (for example
    Monday's normal traffic in binary mode) is turned away before any job starts.
    """
    try:
        target_for_mode(prepared.labels(), request.mode, min_class_count=request.min_class_count)
    except SingleClassError as exc:
        state.set_fit_notice("warning", f"Nothing was fitted. {exc}")
        return False
    return True


def _on_fit() -> None:
    """Fit button callback: build the request from the submitted form and start the fit (before the page redraws).

    Messages (no channel chosen, one class only, a fit already running, the outcome of an inline fit) are left as
    the fit message, which the page shows right below the stepper.
    """
    prepared = state.get_prepared()
    if prepared is None:
        return
    if state.running_job() is not None:
        state.set_fit_notice("warning", "A fit is already running in this session. Wait for it to finish, or cancel "
                                        "it below.")
        return
    request = _form_request(prepared, state.settings())
    if request is None or not _precheck(prepared, request):
        return
    training_ui.start_fit(prepared, request)


def _show_notice() -> None:
    """Show the message left by the last finished, cancelled or failed fit (once)."""
    notice = state.pop_fit_notice()
    if notice is None:
        return
    kind, text = notice
    if kind == "success":
        st.toast(text)
    elif kind == "info":
        st.info(text)
    elif kind == "warning":
        st.warning(text)
    else:
        st.error(text)


def render() -> None:
    """Draw the 02 Fit station."""
    components.station_header("fit")
    _show_notice()
    prepared = state.get_prepared()
    run: TrainingRun | None = state.current_run()
    if prepared is None:
        message = ("Needs a sample: draw one at 01 Sample first." if run is None
                   else "To fit again, draw a sample at 01 Sample first. The readings below are kept with their run.")
        components.needs(message, "sample")
    else:
        _sample_line(prepared)
        last = run.request if run is not None else None
        _fit_form(prepared, state.settings(), last, running=state.running_job() is not None)
    pending = st.session_state.get(state.JOB_ID)
    if pending is not None:
        # Drawn even if the job has just ended: the panel then adopts the result and reruns the app once.
        training_ui.progress_panel(pending)
    elif state.other_fit_running():
        st.caption("A fit started earlier (in another tab, or before this page was refreshed) is still running. "
                   "When it ends, this station offers to restore it.")
    if run is None and pending is None and st.session_state.get(state.LAST_RUN_ID) is None:
        latest = state.run_registry().latest_id()
        if latest is not None:
            training_ui.restore_offer(latest)
    run = state.current_run()
    if run is not None:
        training_ui.readings_panel(run, prepared)
    elif prepared is not None and pending is None:
        st.caption("No fit in this session yet. Choose the options and press Fit.")
