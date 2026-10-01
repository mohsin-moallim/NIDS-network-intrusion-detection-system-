"""04 Probe station: one flow, every channel's verdict on it, and the features behind one channel's reading.

The flow comes from one of three sources: a held-out row of the current run (drawn at random within a true class,
or picked by its number), a typical flow (the per-class median of the run's reference rows, or the overall training
median for a channel set loaded without its rows), or values typed into an editor. Editing never refits anything:
it only scores the edited flow again.

Every reading is computed only when the flow or the choice behind it changes. Results are kept in this session under
a fingerprint of the flow's values plus the channel, method and class (:data:`CACHE`), so redrawing the page, or
coming back to a flow seen before, re-reads them instead of scoring again. Nothing on this page calls ``fit``.
Session values all use the ``g_probe_`` prefix.
"""

from __future__ import annotations

import html
import zlib
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from graticule import explain, theme, viz
from graticule.data.prepare import FILE_COL, ROW_COL, SYNTHETIC_FILE, PreparedDataset
from graticule.explain import Explanation, FlowVerdict
from graticule.models.train import TrainingRun
from graticule.models.verdict import ALERT_RULE
from graticule.schema import is_benign
from graticule.theme import Mode
from ui import components, state
from ui.stations import BY_KEY, PAGE_OBJECTS
from ui.training_ui import channel_label

#: Session keys (all prefixed ``g_probe_``).
SOURCE = "g_probe_source"
CACHE = "g_probe_cache"
LAST_FLOW = "g_probe_last_flow"
EDIT_BASE = "g_probe_edit_base"
EDITED = "g_probe_edited"
DRAWS = "g_probe_draws"
SHADOW = "g_probe_kept_widgets"
#: Flow sources, in the order the radio offers them.
SOURCES: tuple[str, ...] = ("held_out", "typical", "edit")
SOURCE_LABELS: dict[str, str] = {"held_out": "A held-out flow", "typical": "A typical flow", "edit": "Edit values"}
#: Readings, explanations and chart specs kept per session (oldest dropped first).
CACHE_SIZE = 32
ANY_CLASS = "__any__"
ALL_CHANNELS = "__all__"
VERDICT_CLASS = -1
METHOD_LABELS: dict[str, str] = {
    "xgboost_exact": "Exact contributions (XGBoost, log-odds)",
    "reference_swap": "Reference swap (approximate, probability)",
}
PROBABILITY_FORMAT = "%.4f"
EDITOR_HEIGHT = 420


@dataclass(frozen=True)
class _Flow:
    """The flow on the probe: its values, where it came from, and its held-out position (None otherwise)."""

    values: np.ndarray
    source: str
    description: str
    test_index: int | None = None


# --------------------------------------------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------------------------------------------
def _token(run: TrainingRun) -> str:
    """What tells runs apart here: the run id, plus where the run came from (a fit and its copy loaded from disk
    share an id, but the copy may lack CH3 or its rows)."""
    return f"{run.run_id}-{getattr(run, 'origin', 'fitted')}"


def _key(name: str, run: TrainingRun) -> str:
    """A session or widget key tied to one run, so a new run starts its widgets afresh."""
    return f"{name}-{_token(run)}"


def _run_tag(run: TrainingRun) -> tuple[Any, ...]:
    """The part of every cache key that names the run: its token, fitted channels and whether rows are in memory."""
    return (_token(run), tuple(run.ok_channels()), len(run.data.y_train) > 0)


def _cached(key: tuple[Any, ...], compute: Callable[[], Any]) -> Any:
    """``compute()`` once per ``key`` in this session; the oldest entries are dropped beyond :data:`CACHE_SIZE`."""
    cache: dict[tuple[Any, ...], Any] = st.session_state.setdefault(CACHE, {})
    if key in cache:
        return cache[key]
    value = compute()
    cache[key] = value
    while len(cache) > CACHE_SIZE:
        cache.pop(next(iter(cache)))
    return value


def _class_text(name: str) -> str:
    """A class name with its shape cue: ``"○ Normal"`` (or ``"○ BENIGN"``) and ``"◆ DoS Hulk"``."""
    glyph = theme.GLYPH_NORMAL if is_benign(name) or name == "Normal" else theme.GLYPH_ATTACK
    return f"{glyph} {name}"


def _plain_float(value: Any) -> float:
    """A float32 value as the shortest float64 that reads back as the same float32 (0.1, not 0.10000000149)."""
    return float(str(np.float32(value)))


def _is_binary(run: TrainingRun) -> bool:
    """True for a normal-vs-attack run (a multi-class run keeps its class names even with two classes)."""
    return run.request.mode == "binary"


def _quantile_row(run: TrainingRun, percent: int) -> np.ndarray:
    """Training quantile ``percent`` (0..100) of every feature."""
    table = np.asarray(run.feature_quantiles, dtype=np.float32)
    return table[min(max(int(percent), 0), table.shape[0] - 1)]


def _restore(widget_key: str, default: Any) -> None:
    """Before drawing a widget shown for one flow source only: give it back its last value.

    Streamlit forgets the state of a widget that is not drawn in a run, so the held-out row, its class filter and
    the typical flow's class would reset whenever the viewer looked at another source. :func:`_keep` saves a copy.
    """
    if widget_key not in st.session_state:
        st.session_state[widget_key] = st.session_state.get(SHADOW, {}).get(widget_key, default)


def _keep(widget_key: str) -> None:
    """After drawing such a widget: save a copy of its value for :func:`_restore`."""
    st.session_state.setdefault(SHADOW, {})[widget_key] = st.session_state.get(widget_key)


def _remember_flow(run: TrainingRun, flow: _Flow) -> None:
    """Keep the last held-out or typical flow, which the editor can start from."""
    store: dict[str, tuple[np.ndarray, str]] = st.session_state.setdefault(LAST_FLOW, {})
    store[_token(run)] = (np.array(flow.values, dtype=np.float32, copy=True), flow.description)


# --------------------------------------------------------------------------------------------------------------
# Flow source (a): a held-out flow
# --------------------------------------------------------------------------------------------------------------
def _draw(run: TrainingRun, label: str, counter: int) -> int:
    """Step ``counter`` of a seeded walk through the held-out rows of detailed class ``label`` (a row number).

    The walk is a seeded shuffle of the matching rows, so successive draws differ until every row has come up.
    """
    labels = np.asarray(run.data.detailed_test_labels).astype(str)
    candidates = np.arange(len(labels)) if label == ANY_CLASS else np.flatnonzero(labels == label)
    if len(candidates) == 0:
        candidates = np.arange(len(labels))
    salt = zlib.crc32(label.encode("utf-8"))
    order = np.random.default_rng([int(run.request.seed), salt]).permutation(candidates)
    return int(order[int(counter) % len(order)])


def _on_draw(token: str) -> None:
    """"Draw another" callback: the next row of the seeded walk within the chosen class."""
    run = state.current_run()
    if run is None or _token(run) != token or not run.has_test_rows:
        return
    counter = int(st.session_state.get(_key(DRAWS, run), 0)) + 1
    st.session_state[_key(DRAWS, run)] = counter
    label = str(st.session_state.get(_key("g_probe_filter", run), ANY_CLASS))
    st.session_state[_key("g_probe_row", run)] = _draw(run, label, counter)


def _on_filter(token: str) -> None:
    """Class filter callback: start the walk afresh within the newly chosen class."""
    run = state.current_run()
    if run is None or _token(run) != token or not run.has_test_rows:
        return
    st.session_state[_key(DRAWS, run)] = 0
    label = str(st.session_state.get(_key("g_probe_filter", run), ANY_CLASS))
    st.session_state[_key("g_probe_row", run)] = _draw(run, label, 0)


def _provenance(run: TrainingRun, prepared: PreparedDataset | None, index: int) -> str:
    """Where held-out row ``index`` came from: file and data row (real data), or the generator's flow number."""
    if prepared is None or prepared.fingerprint != run.dataset_fingerprint:
        return ("Source file and row: shown while 01 Sample holds the sample this run was fitted on (another sample, "
                "or none, is loaded now).")
    position = int(run.data.test_rows[index])
    if position >= len(prepared.frame):
        return "Source file and row: not available for this row."
    file_name = str(prepared.frame[FILE_COL].iloc[position])
    row = int(prepared.frame[ROW_COL].iloc[position])
    # Row numbers count from 0 here, as in every export (the predictions CSV's source_row, the Assay readings).
    if file_name == SYNTHETIC_FILE:
        return (f"Source: generated flow {row:,} of the synthetic sample (counting from 0; generator seed "
                f"{prepared.request.seed}); no recorded traffic.")
    return (f"Source: {file_name}, data row {row:,} (data rows count from 0, as in the exports; it is line "
            f"{row + 2:,} of the file, counting its header as line 1).")


def _no_held_out(run: TrainingRun) -> None:
    """A channel set loaded without its held-out rows: say so, and where they can be rebuilt."""
    if run.data_request.source == "synthetic":
        components.needs(
            f"Run {run.run_id} was loaded from disk without its held-out rows: generating its synthetic sample again "
            "did not give the very same rows. Typical flows and edited values still work.", "logbook")
        return
    components.needs(
        f"Run {run.run_id} was loaded from disk without its held-out rows. Point the Bench at the data folder it was "
        "fitted on, then load it again at the Logbook to rebuild them. Typical flows and edited values still work.",
        "logbook")
    bench = PAGE_OBJECTS.get("bench")
    if bench is not None:
        st.page_link(bench, label=f"Go to {BY_KEY['bench'].label}", icon=":material/arrow_forward:")


def _held_out_flow(run: TrainingRun, prepared: PreparedDataset | None) -> _Flow | None:
    """Pick a held-out row (class filter, row number, "Draw another") and describe it; None without held-out rows."""
    if not run.has_test_rows:
        _no_held_out(run)
        return None
    labels = np.asarray(run.data.detailed_test_labels).astype(str)
    total = len(labels)
    counts = pd.Series(labels, dtype="str").value_counts()
    options = [ANY_CLASS, *[str(name) for name in counts.index]]
    filter_key, row_key = _key("g_probe_filter", run), _key("g_probe_row", run)
    _restore(filter_key, ANY_CLASS)
    if st.session_state[filter_key] not in options:
        st.session_state[filter_key] = ANY_CLASS
    _restore(row_key, _draw(run, ANY_CLASS, 0))
    left, middle, right = st.columns([2, 1, 1], gap="small", vertical_alignment="bottom")
    with left:
        st.selectbox("Draw from true class", options, key=filter_key, on_change=_on_filter, args=(_token(run),),
                     format_func=lambda o: f"Any class ({total:,} rows)" if o == ANY_CLASS
                     else f"{_class_text(o)} ({int(counts[o]):,} rows)",
                     help="Detailed labels of the held-out rows; draws stay within the class chosen here.")
    with middle:
        st.number_input(f"Held-out row (0 to {total - 1:,})", min_value=0, max_value=total - 1, step=1,
                        key=row_key, help="Position among the held-out rows; type a number to pick a row.")
    with right:
        st.button("Draw another", key=_key("g_probe_draw", run), on_click=_on_draw, args=(_token(run),),
                  width="stretch", help="The next row of a seeded walk through the chosen class.")
    _keep(filter_key)
    _keep(row_key)
    index = int(min(max(int(st.session_state[row_key]), 0), total - 1))
    true_code = int(run.data.y_test[index])
    true_class = str(run.data.classes[true_code])
    detailed = str(labels[index])
    extra = (f' (detailed label <span class="g-mono">{html.escape(detailed)}</span>)'
             if detailed != true_class else "")
    st.markdown(f'Held-out row <span class="g-mono">{index:,}</span> of <span class="g-mono">{total:,}</span>. '
                f"True class: {components.verdict_chip(true_class)}{extra}", unsafe_allow_html=True)
    st.caption(_provenance(run, prepared, index))
    return _Flow(values=np.asarray(run.data.X_test[index], dtype=np.float32), source="held_out",
                 description=f"held-out row {index:,}", test_index=index)


# --------------------------------------------------------------------------------------------------------------
# Flow sources (b) a typical flow and (c) edited values
# --------------------------------------------------------------------------------------------------------------
def _has_training_rows(run: TrainingRun) -> bool:
    """True when the run holds its training rows (and so its reference sample of real rows)."""
    return len(run.data.y_train) > 0 and len(run.reference_sample) > 0


def _typical_flow(run: TrainingRun) -> _Flow:
    """A per-class median flow (or the overall training median for a set loaded without its rows)."""
    classes = [str(c) for c in run.data.classes]
    if _has_training_rows(run):
        class_key = _key("g_probe_typical", run)
        _restore(class_key, 0)
        picked = st.selectbox("Typical flow of class", list(range(len(classes))), key=class_key,
                              format_func=lambda i: _class_text(classes[int(i)]),
                              help="Per-feature median of this class's rows in the run's reference sample.")
        _keep(class_key)
        index: int | None = int(picked)
        name = classes[int(picked)]
    else:
        st.caption("This channel set was loaded without its training rows, so the typical flow is the overall "
                   "training median of every feature, read off the quantiles saved with it (no class choice).")
        index, name = None, "all classes"
    values, where = _cached(("typical", *_run_tag(run), index), lambda: explain.typical_flow(run, index))
    st.caption(f"The flow is {where}: a summary, not a recorded flow.")
    return _Flow(values=np.asarray(values, dtype=np.float32), source="typical", description=f"typical flow ({name})")


def _editor_start(run: TrainingRun) -> tuple[np.ndarray, str]:
    """Where the editor starts: the last held-out or typical flow shown, else the overall typical flow."""
    last = st.session_state.get(LAST_FLOW, {}).get(_token(run))
    if last is not None:
        return np.array(last[0], dtype=np.float32, copy=True), str(last[1])
    values, _ = _cached(("typical", *_run_tag(run), None), lambda: explain.typical_flow(run, None))
    return np.asarray(values, dtype=np.float32), "the overall typical flow"


def _on_restart_editor(token: str) -> None:
    """"Start again" callback: a fresh editor (new key) holding the last held-out or typical flow."""
    run = state.current_run()
    if run is None or _token(run) != token:
        return
    store: dict[str, tuple[int, np.ndarray, str]] = st.session_state.setdefault(EDIT_BASE, {})
    version = store[token][0] + 1 if token in store else 0
    values, origin = _editor_start(run)
    store[token] = (version, values, origin)


def _edited_flow(run: TrainingRun) -> _Flow:
    """The value editor: feature, value and the training p1/p99 as hints; returns the edited flow."""
    store: dict[str, tuple[int, np.ndarray, str]] = st.session_state.setdefault(EDIT_BASE, {})
    if _token(run) not in store:
        values, origin = _editor_start(run)
        store[_token(run)] = (0, values, origin)
    version, base, origin = store[_token(run)]
    names = [str(n) for n in run.data.feature_names]
    frame = pd.DataFrame({
        "Feature": pd.Series(names, dtype="str"),
        "Value": [_plain_float(v) for v in base],
        "p1": [_plain_float(v) for v in _quantile_row(run, 1)],
        "p99": [_plain_float(v) for v in _quantile_row(run, 99)],
    })
    edited = st.data_editor(
        frame, key=f"g_probe_editor-{_token(run)}-{version}", hide_index=True, num_rows="fixed", width="stretch",
        height=EDITOR_HEIGHT, disabled=["Feature", "p1", "p99"],
        column_config={
            "Feature": st.column_config.TextColumn("Feature"),
            "Value": st.column_config.NumberColumn("Value", help="Type a new value; a cleared cell counts as missing."),
            "p1": st.column_config.NumberColumn("p1", help="1st percentile of the training rows (a hint only)."),
            "p99": st.column_config.NumberColumn("p99", help="99th percentile of the training rows (a hint only)."),
        },
    )
    values = pd.to_numeric(edited["Value"], errors="coerce").to_numpy(dtype=np.float32, na_value=np.nan)
    same = (values == base) | (np.isnan(values) & np.isnan(base))
    changed = [names[i] for i in np.flatnonzero(~same)]
    edits: dict[str, tuple[int, np.ndarray, int]] = st.session_state.setdefault(EDITED, {})
    edits[_token(run)] = (version, values.copy(), len(changed))
    left, right = st.columns([3, 2], gap="small", vertical_alignment="center")
    with left:
        if changed:
            shown = ", ".join(changed[:4]) + (f" and {len(changed) - 4} more" if len(changed) > 4 else "")
            st.caption(f"Started from {origin}; {len(changed)} value{'s' if len(changed) != 1 else ''} edited "
                       f"({shown}). Editing only scores the flow again: nothing is refitted.")
        else:
            st.caption(f"Started from {origin}; no value edited yet. Editing only scores the flow again: nothing is "
                       "refitted.")
    with right:
        st.button("Start again from the last held-out or typical flow", key=_key("g_probe_restart", run),
                  on_click=_on_restart_editor, args=(_token(run),), width="stretch")
    return _Flow(values=values, source="edit", description=f"edited flow ({len(changed)} values changed)")


def _keep_edits(run: TrainingRun) -> None:
    """Carry edits over while the editor is hidden: Streamlit forgets the state of a widget that is not drawn, so
    the edited values become the start of a fresh editor (shown again when the viewer comes back)."""
    entry = st.session_state.get(EDITED, {}).pop(_token(run), None)
    store: dict[str, tuple[int, np.ndarray, str]] = st.session_state.setdefault(EDIT_BASE, {})
    if entry is None or _token(run) not in store:
        return
    version, values, changed = entry
    current, _, origin = store[_token(run)]
    if changed and version == current:
        earlier = origin if origin.startswith("your earlier edits") else f"your earlier edits of {origin}"
        store[_token(run)] = (current + 1, np.array(values, dtype=np.float32, copy=True), earlier)


def _values_expander(run: TrainingRun, flow: _Flow) -> None:
    """Every feature value of the flow, with the training p1/p99 for scale (folded away)."""
    with st.expander(f"All {len(run.data.feature_names)} feature values of this flow"):
        st.dataframe(pd.DataFrame({
            "Feature": pd.Series([str(n) for n in run.data.feature_names], dtype="str"),
            "Value": [_plain_float(v) for v in flow.values],
            "p1": [_plain_float(v) for v in _quantile_row(run, 1)],
            "p99": [_plain_float(v) for v in _quantile_row(run, 99)],
        }), hide_index=True, width="stretch")


def _flow_section(run: TrainingRun, prepared: PreparedDataset | None) -> _Flow | None:
    """(1) Choose the flow; returns it (None when the held-out source has no rows)."""
    st.subheader("Flow", anchor=False)
    default = SOURCES.index("held_out" if run.has_test_rows else "typical")
    if SOURCE not in st.session_state:
        st.session_state[SOURCE] = SOURCES[default]
    source = st.radio("Flow source", SOURCES, key=SOURCE, horizontal=True, format_func=SOURCE_LABELS.__getitem__,
                      help="A held-out row of this run, a typical flow of a class, or values you type in.")
    if source != "edit":
        _keep_edits(run)
    if source == "held_out":
        flow = _held_out_flow(run, prepared)
    elif source == "typical":
        flow = _typical_flow(run)
    else:
        flow = _edited_flow(run)
    if flow is not None and flow.source != "edit":
        _remember_flow(run, flow)
        _values_expander(run, flow)
    return flow


# --------------------------------------------------------------------------------------------------------------
# Verdict
# --------------------------------------------------------------------------------------------------------------
def _verdict_for(run: TrainingRun, flow: _Flow) -> FlowVerdict:
    """Every fitted channel's reading of the flow (scored once per flow and run, then re-read)."""
    digest = explain.row_digest(flow.values)
    return _cached(("verdict", *_run_tag(run), digest),
                   lambda: explain.score_flow(run, flow.values, run.ok_channels()))


def _top_text(verdict: FlowVerdict, key: str | None) -> str:
    """The three most probable classes, e.g. ``"DoS Hulk 0.9120 · BENIGN 0.0510 · PortScan 0.0200"``."""
    return " · ".join(f"{name} {p:.4f}" for name, p in verdict.top_classes(key, 3))


def _verdict_rows(run: TrainingRun, flow: _Flow, verdict: FlowVerdict, keys: list[str], with_consensus: bool,
                  threshold: float) -> pd.DataFrame:
    """The verdict table: one row per channel (plus the consensus)."""
    binary = _is_binary(run)
    truth = int(run.data.y_test[flow.test_index]) if flow.test_index is not None else None
    rows: list[dict[str, Any]] = []
    entries: list[tuple[str, str | None]] = [(channel_label(k), k) for k in keys]
    if with_consensus:
        entries.append((f"{theme.CONSENSUS_LABEL} ({verdict.consensus.voters} channels)", None))
    for name, key in entries:
        index = verdict.consensus_index if key is None else verdict.label_index(key)
        label = verdict.classes[index]
        attack = verdict.attack_probability(key)
        row: dict[str, Any] = {
            "Channel": name,
            "Verdict": theme.verdict_text(label),
            "P(verdict)": verdict.consensus_probability if key is None else verdict.probability(key),
        }
        if binary:
            row["P(attack)"] = attack
        else:
            row["Top classes"] = _top_text(verdict, key)
        row["Alert"] = f"{theme.GLYPH_ALERT} Alert" if verdict.raises_alert(key, threshold) else ""
        if truth is not None:
            row["Reading"] = "Correct" if index == truth else "Incorrect"
        rows.append(row)
    return pd.DataFrame(rows)


def _verdict_section(run: TrainingRun, flow: _Flow, verdict: FlowVerdict) -> None:
    """(2) The channels' verdicts, the consensus line and, for a held-out flow, whether each one was right."""
    st.subheader("Verdict", anchor=False)
    keys = run.ok_channels()
    view = st.selectbox("Channels", [ALL_CHANNELS, *keys], key=_key("g_probe_view", run),
                        format_func=lambda k: "All channels + consensus" if k == ALL_CHANNELS else channel_label(k))
    shown = keys if view == ALL_CHANNELS else [str(view)]
    threshold = float(state.settings().alert_threshold)
    table = _verdict_rows(run, flow, verdict, shown, view == ALL_CHANNELS, threshold)
    config = {name: st.column_config.NumberColumn(name, format=PROBABILITY_FORMAT)
              for name in ("P(verdict)", "P(attack)") if name in table.columns}
    config["Alert"] = st.column_config.TextColumn(
        "Alert", help=f"Raised by {ALERT_RULE} ({threshold:.3f}, set on the Bench), as at 05 Assay and 06 Sweep.")
    st.dataframe(table, hide_index=True, width="stretch", column_config=config)
    if view == ALL_CHANNELS:
        label = verdict.consensus_label
        noun = "channel agrees" if verdict.agreement == 1 else "channels agree"
        line = (f"**{theme.CONSENSUS_LABEL}:** {components.verdict_chip(label)} at "
                f'<span class="g-mono">{verdict.consensus_probability:.2f}</span> — '
                f"{verdict.agreement} of {verdict.consensus.voters} {noun}.")
        if flow.test_index is not None:
            right = verdict.consensus_index == int(run.data.y_test[flow.test_index])
            line += f' <span class="g-chip g-tag">{"Correct" if right else "Incorrect"}</span>'
        st.markdown(line, unsafe_allow_html=True)
        st.caption("Each channel counts equally: the consensus is the mean of their class probabilities.")
    elif not _is_binary(run):
        key = str(view)
        probabilities = pd.DataFrame({"Class": [_class_text(c) for c in verdict.classes],
                                      "Probability": np.asarray(verdict.proba[key], dtype=np.float64)})
        probabilities = probabilities.sort_values("Probability", ascending=False, kind="stable")
        st.dataframe(probabilities, hide_index=True, width="stretch",
                     column_config={"Probability": st.column_config.NumberColumn("Probability",
                                                                                 format=PROBABILITY_FORMAT)})


# --------------------------------------------------------------------------------------------------------------
# Explanation
# --------------------------------------------------------------------------------------------------------------
def _explain(run: TrainingRun, flow: _Flow, key: str, method: str, class_index: int) -> Explanation:
    """The explanation of channel ``key`` for the flow (computed once per flow, channel, method and class)."""
    digest = explain.row_digest(flow.values)
    name = str(run.data.classes[class_index])
    estimator = run.channels[key].estimator

    def compute() -> Explanation:
        if method == "xgboost_exact":
            return explain.xgboost_contributions(estimator, flow.values, run.data.feature_names, class_index,
                                                 channel=key, class_name=name)
        background, _ = _cached(("background", *_run_tag(run)), lambda: explain.background_for(run))
        with st.spinner(f"Swapping each feature with up to {explain.DEFAULT_BACKGROUND} background values..."):
            return explain.reference_swap(estimator, flow.values, background, run.data.feature_names, class_index,
                                          max_background=explain.DEFAULT_BACKGROUND, seed=int(run.request.seed),
                                          channel=key, class_name=name)

    return _cached(("explain", *_run_tag(run), digest, key, method, class_index), compute)


def _directions(run: TrainingRun, class_name: str) -> tuple[str, str, bool]:
    """Legend labels (towards, away) and whether "towards" means normal traffic, for the explained class."""
    if _is_binary(run):
        return "Towards Attack", "Towards Normal", False
    if is_benign(class_name) or class_name == "Normal":
        return f"Towards {class_name} (normal)", f"Away from {class_name}", True
    return f"Towards {class_name}", f"Away from {class_name}", False


def _explanation_caption(run: TrainingRun, explanation: Explanation, shown: int) -> str:
    """How to read the bars: units, what they add up to, or how the estimate was made."""
    n_features = len(explanation.table)
    rest = n_features - shown
    if not explanation.approximate:
        others = f", the other {rest} features ({explanation.rest(shown):+.3f})" if rest > 0 else ""
        softmax = "" if _is_binary(run) else " (a softmax over every class's raw score gives the probabilities)"
        return (f"Exact, in log-odds of {explanation.class_name}: the {shown} bars{others} and the bias "
                f"({float(explanation.bias or 0.0):+.3f}) add up to this flow's raw score of "
                f"{explanation.output:+.3f}{softmax}.")
    _, background = _cached(("background", *_run_tag(run)), lambda: explain.background_for(run))
    return (f"Approximate, in probability points of {explanation.class_name} (0.10 = ten points): each bar is the "
            f"channel's probability for this flow ({explanation.output:.3f}) minus its mean probability when that "
            f"one feature takes the value of each of {explanation.background_rows} background flows ({background}). "
            "Features are swapped one at a time, so effects that need two features to change together are not "
            "seen.")


def _explanation_section(run: TrainingRun, flow: _Flow, verdict: FlowVerdict, mode: Mode) -> None:
    """(3) Which features moved one channel's reading: channel, method and (multi-class) class pickers, the chart,
    one sentence and how to read it."""
    st.subheader("Explanation", anchor=False)
    keys = run.ok_channels()
    classes = [str(c) for c in run.data.classes]
    binary = _is_binary(run)
    columns = st.columns([1, 1, 1] if not binary else [1, 2], gap="medium", vertical_alignment="bottom")
    with columns[0]:
        key = str(st.selectbox("Channel to explain", keys, index=keys.index("xgboost") if "xgboost" in keys else 0,
                               format_func=channel_label, key=_key("g_probe_xchannel", run)))
    methods = ["xgboost_exact", "reference_swap"] if key == "xgboost" else ["reference_swap"]
    with columns[1]:
        method = str(st.radio("Method", methods, format_func=METHOD_LABELS.__getitem__, horizontal=True,
                              key=_key(f"g_probe_method_{key}", run),
                              help="Exact contributions exist for CH2 only; the reference swap works for every "
                                   "channel and is an estimate."))
    if binary:
        class_index = 1
    else:
        own = verdict.label_index(key)
        with columns[2]:
            choice = int(st.selectbox(
                "Explain the reading of", [VERDICT_CLASS, *range(len(classes))], key=_key("g_probe_xclass", run),
                format_func=lambda i: (f"Its verdict ({_class_text(classes[own])})" if int(i) == VERDICT_CLASS
                                       else _class_text(classes[int(i)]))))
        class_index = own if choice == VERDICT_CLASS else choice
    explanation = _explain(run, flow, key, method, class_index)
    toward, away, toward_is_normal = _directions(run, explanation.class_name)
    shown = min(explain.TOP_FEATURES, len(explanation.table))
    exact = not explanation.approximate
    what = "log-odds" if exact else "probability points"
    spec_key = ("chart", *_run_tag(run), explain.row_digest(flow.values), key, method, class_index, mode)
    spec = _cached(spec_key, lambda: viz.chart_spec(lambda: viz.contribution_chart(
        explanation.table, mode, top=explain.TOP_FEATURES, toward_label=toward, away_label=away,
        toward_is_normal=toward_is_normal,
        title=f"{channel_label(key)}: what moved its reading" + ("" if exact else " (approximate)"),
        subtitle=(f"{'Exact contributions' if exact else 'Reference swap'} for {explanation.class_name}; the "
                  f"{shown} largest of {len(explanation.table)} features, with this flow's values."),
        x_title=f"Contribution ({what} of {explanation.class_name})")))
    st.vega_lite_chart(spec=spec, width="stretch", theme=None)
    if binary:
        attack_side = verdict.label(key) == explanation.class_name
        sentence = explain.reading_sentence(explanation, towards="Attack" if attack_side else "Normal",
                                            sign=1 if attack_side else -1)
    else:
        sentence = explain.reading_sentence(explanation, towards=explanation.class_name, sign=1)
    st.markdown(f"**{html.escape(sentence)}**")
    st.caption(_explanation_caption(run, explanation, shown))


# --------------------------------------------------------------------------------------------------------------
# The page
# --------------------------------------------------------------------------------------------------------------
def _run_chips(run: TrainingRun) -> list[str]:
    """Short facts about the run on the probe."""
    chips = [
        f"run {run.run_id}",
        "binary: normal vs attack" if _is_binary(run) else f"multi-class: {len(run.data.classes)} classes",
        f"{len(run.data.feature_names)} features",
        f"{len(run.ok_channels())} channel{'s' if len(run.ok_channels()) != 1 else ''}",
    ]
    if getattr(run, "origin", "fitted") == "loaded":
        chips.append("loaded from disk")
    if not run.has_test_rows:
        chips.append("no held-out rows")
    return chips


def render() -> None:
    """Draw the 04 Probe station."""
    components.station_header("probe")
    run = state.current_run()
    if run is None or not run.ok_channels():
        components.needs("Needs a fitted channel: fit at least one at 02 Fit.", "fit")
        return
    components.chips(_run_chips(run))
    prepared = state.get_prepared()
    if run.has_test_rows and prepared is not None and prepared.fingerprint != run.dataset_fingerprint:
        st.caption(f"Run {run.run_id} was fitted on an earlier sample; its held-out flows are its own rows, but "
                   "their source file and row cannot be shown while 01 Sample holds another sample.")
    flow = _flow_section(run, prepared)
    if flow is None:
        return
    verdict = _verdict_for(run, flow)
    state.mark_done("probe")  # a flow has been read: the stepper ticks 04 Probe (redrawn by the shell this run)
    _verdict_section(run, flow, verdict)
    _explanation_section(run, flow, verdict, components.current_mode())
