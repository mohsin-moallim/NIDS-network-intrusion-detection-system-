"""03 Measure station: the fitted channels side by side, then each one in detail.

Everything here is read from the current run (:func:`ui.state.current_run`). The readings of every channel are
computed once per run by :func:`graticule.evaluate.evaluate_run` and kept on the run object, so changing a widget on
this page only redraws: it never recomputes the readings and never refits anything. Two measurements do extra work,
and only when their buttons are pressed:

* permutation importance (held-out rows only; nothing is fitted), and
* cross-validation (fresh channel copies fitted in folds of the TRAINING rows; the held-out rows play no part).

Both run as an exclusive :class:`graticule.evaluate.EvaluationTask` on a background thread (inline when
``GRATICULE_SYNC_TRAINING=1``, as in the headless tests), with a progress panel that refreshes itself once a second
and a Cancel button. Exclusive means they take the app's single work slot: neither starts while a fit or the other
measurement runs anywhere in the app (their buttons are disabled meanwhile, with a note), and 02 Fit refuses to
start a fit while one of them runs. Their results are kept on the run object as well
(:func:`graticule.evaluate.stored_cross_validation`, :func:`graticule.evaluate.stored_permutations`), where other
stations (07 Record) can read them.
"""

from __future__ import annotations

import html
import math
import os
from collections.abc import Callable, Sequence
from typing import Literal, Any

import numpy as np
import pandas as pd
import streamlit as st

from graticule import evaluate, theme, viz
from graticule.evaluate import ChannelEvaluation, EvaluationTask
from graticule.models.jobs import JobBusyError, slot_holder, sync_training_requested
from graticule.models.train import TrainingRun
from graticule.schema import is_benign
from graticule.theme import Mode
from ui import components, state
from ui.stations import BY_KEY, PAGE_OBJECTS
from ui.training_ui import channel_label, format_elapsed, svm_rows_note

#: Session keys: ids of the running measurement tasks, the task a full rerun was asked for, and one-off notices.
CV_TASK = "ms_cv_task"
PERM_TASK = "ms_perm_task"
RERUN_FOR = "ms_rerun_for"
NOTICE = "ms_notice"
#: Cross-validation choices.
CV_FOLDS = (3, 10)
CV_DEFAULT_FOLDS = 5
CV_MAX_ROWS = 50_000
#: Permutation importance choices (the core holds the SVM to 2,000 rows and 3 repeats).
PERM_ROWS = 5_000
PERM_REPEATS = 5
#: False-positive range of the zoomed ROC view.
ROC_ZOOM = 0.05
SHOW_OPTIONS = ("Row %", "Counts")
ROC_OPTIONS = ("Whole curve", f"Low false-positive corner (up to {ROC_ZOOM:.0%})")
SCORE_FORMAT = "%.4f"
#: Attribute of a run holding the chart specs drawn here (see :func:`_kept_chart`).
CHART_SPECS_ATTR = "measure_chart_specs"


# --------------------------------------------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------------------------------------------
def _rough(value: float) -> str:
    """A duration rounded for reading: ``"40 s"``, ``"2 min"``, ``"1 h 5 min"`` (5 s steps below a minute)."""
    if value < 60:
        return f"{max(int(round(value / 5.0) * 5), 5)} s"
    if value < 3600:
        return f"{int(round(value / 60.0))} min"
    hours, minutes = divmod(int(round(value / 60.0)), 60)
    return f"{hours} h {minutes} min"


def duration_range_text(low: float, high: float) -> str:
    """A rough range for an estimate whose speed-up is uncertain, e.g. ``"about 10 to 40 s"``.

    ``"under 5 s"`` when even the upper end is that short, ``"up to about 20 s"`` when the lower end is under 5 s,
    and a single value when both ends round alike.
    """
    low, high = sorted((max(float(low), 0.0), max(float(high), 0.0)))
    if high < 5:
        return "under 5 s"
    if low < 5:
        return f"up to about {_rough(high)}"
    first, last = _rough(low), _rough(high)
    if first == last:
        return f"about {first}"
    if first.endswith(" s") and last.endswith(" s"):
        first = first[:-2]
    elif first.endswith(" min") and last.endswith(" min"):
        first = first[:-4]
    return f"about {first} to {last}"


def duration_text(seconds: float) -> str:
    """A rough duration for an estimate, e.g. ``"under 5 s"``, ``"about 40 s"``, ``"about 2 min"``."""
    value = max(float(seconds), 0.0)
    if value < 5:
        return "under 5 s"
    if value < 60:
        return f"about {int(round(value / 5.0) * 5)} s"
    if value < 3600:
        return f"about {int(round(value / 60.0))} min"
    hours, minutes = divmod(int(round(value / 60.0)), 60)
    return f"about {hours} h {minutes} min"


def class_text(name: str) -> str:
    """A class name with its shape cue: ``"○ BENIGN"``, ``"○ Normal"`` or ``"◆ DoS Hulk"``."""
    glyph = theme.GLYPH_NORMAL if is_benign(name) or name == "Normal" else theme.GLYPH_ATTACK
    return f"{glyph} {name}"


def held_out_line(run: TrainingRun) -> str:
    """The held-out class distribution in one line, with shape cues (every reading is measured on these rows)."""
    counts = evaluate.held_out_counts(run)
    parts = [f"{class_text(name)} {count:,}" for name, count in counts.items()]
    return f"Measured on {sum(counts.values()):,} held-out rows: " + " · ".join(parts) + "."


def run_key(name: str, run: TrainingRun) -> str:
    """Widget key for a picker whose options depend on the run (a new run starts it afresh)."""
    return f"{name}-{run.run_id}"


def _mode() -> Mode:
    """The viewer's theme mode for charts."""
    return components.current_mode()


def _chart(build: Callable[[], Any]) -> None:
    """Build an Altair chart with ``build`` and draw it with its own styling, stretched to the column width."""
    st.vega_lite_chart(spec=viz.chart_spec(build), width="stretch", theme=None)


def _kept_chart(run: TrainingRun, name: tuple[Any, ...], build: Callable[[], Any], *,
                width: Literal["stretch", "content"] = "stretch") -> None:
    """Draw a chart whose Vega-Lite spec is built once per run, theme mode and options, then kept with the run.

    Building a chart (Altair copies and validates every layer) costs far more than drawing it, and the readings
    behind these charts never change, so a rerun of this page only re-sends the stored spec.
    """
    specs = run.__dict__.setdefault(CHART_SPECS_ATTR, {})
    spec = specs.get(name)
    if spec is None:
        spec = viz.chart_spec(build)
        specs[name] = spec
    st.vega_lite_chart(spec=spec, width=width, theme=None)


def _set_notice(kind: str, text: str) -> None:
    """Leave a message for the next drawing of this station (shown once)."""
    st.session_state[NOTICE] = (kind, text)


def _show_notice() -> None:
    """Show the pending message, if any."""
    notice = st.session_state.pop(NOTICE, None)
    if notice is None:
        return
    kind, text = notice
    {"success": st.success, "info": st.info, "warning": st.warning}.get(kind, st.error)(text)


def _rerun_once(task_id: str) -> None:
    """Rerun the whole app once for ``task_id`` (a second request for the same task does nothing)."""
    if st.session_state.get(RERUN_FOR) != task_id:
        st.session_state[RERUN_FOR] = task_id
        st.rerun(scope="app")


# --------------------------------------------------------------------------------------------------------------
# Measurement tasks (permutation importance, cross-validation)
# --------------------------------------------------------------------------------------------------------------
def _cv_work(run: TrainingRun, k: int, channels: tuple[str, ...]) -> Any:
    """The cross-validation work of a task: run it and keep any finished channels with the run."""
    def work(progress: Any, cancel: Any) -> pd.DataFrame:
        frame = evaluate.cross_validate_run(run, k=k, max_rows=CV_MAX_ROWS, channels=channels, progress=progress,
                                            cancel=cancel)
        if not frame.empty:
            evaluate.remember_cross_validation(run, frame)
        return frame

    return work


def _perm_work(run: TrainingRun, key: str) -> Any:
    """The permutation-importance work of a task: measure it and keep it with the run."""
    def work(progress: Any, cancel: Any) -> pd.DataFrame:
        frame = evaluate.permutation_importance_for(run, key, max_rows=PERM_ROWS, repeats=PERM_REPEATS,
                                                    progress=progress, cancel=cancel)
        evaluate.remember_permutation(run, key, frame)
        return frame

    return work


def _launch(task: EvaluationTask, session_key: str) -> None:
    """Start ``task`` in the background (or inline for headless tests) and remember it under ``session_key``.

    While a fit or another measurement holds the app's work slot the task is turned away with a note instead.
    """
    try:
        if sync_training_requested():
            task.run_inline()
            _report(task)
            return
        task.start()
    except JobBusyError as exc:
        _set_notice("warning", str(exc))
        return
    st.session_state[session_key] = task.task_id
    st.session_state.pop(RERUN_FOR, None)


def _report(task: EvaluationTask) -> None:
    """Turn a finished task into a one-off message (its result is already kept with the run)."""
    snap = task.snapshot()
    what = "Cross-validation" if task.kind == "cv" else f"Permutation importance of {task.label}"
    if snap.state == "done":
        _set_notice("success", f"{what} finished in {snap.elapsed:,.1f} s.")
    elif snap.state == "cancelled":
        kept = ""
        if task.kind == "cv" and isinstance(task.result, pd.DataFrame) and not task.result.empty:
            kept = f" The {len(task.result)} channel(s) that finished every fold are kept."
        _set_notice("info", f"{what} cancelled after {snap.elapsed:,.1f} s.{kept}")
    else:
        _set_notice("error", f"{what} failed: {state.first_line(snap.error) or 'no reason recorded'}")


def _collect(session_key: str) -> EvaluationTask | None:
    """The session's task under ``session_key`` while it runs; a finished one is reported and forgotten."""
    task = evaluate.get_task(st.session_state.get(session_key))
    if task is None:
        st.session_state.pop(session_key, None)
        return None
    if task.finished:
        st.session_state.pop(session_key, None)
        _report(task)
        evaluate.forget_task(task.task_id)
        return None
    return task


def _start_cv() -> None:
    """Run cross-validation button callback (reads the options from the widgets)."""
    run = state.current_run()
    if run is None or evaluate.get_task(st.session_state.get(CV_TASK)) is not None:
        return
    k = int(st.session_state.get("ms_cv_k", CV_DEFAULT_FOLDS))
    chosen = tuple(st.session_state.get(run_key("ms_cv_channels", run)) or ())
    channels = tuple(key for key in run.ok_channels() if key in chosen)
    if not channels:
        _set_notice("warning", "Choose at least one channel to cross-validate.")
        return
    task = EvaluationTask("cv", run.run_id, _cv_work(run, k, channels), label="cross-validation", exclusive=True)
    _launch(task, CV_TASK)


def _start_permutation(key: str) -> None:
    """Measure permutation importance button callback."""
    run = state.current_run()
    if run is None or evaluate.get_task(st.session_state.get(PERM_TASK)) is not None:
        return
    task = EvaluationTask("perm", run.run_id, _perm_work(run, key), label=channel_label(key), exclusive=True,
                          holder=f"a permutation-importance measurement of {channel_label(key)}")
    task.extra["key"] = key
    _launch(task, PERM_TASK)


def _busy_elsewhere() -> str | None:
    """What holds the app's work slot when it is not one of this session's measurements (a fit, say), or None."""
    holder = slot_holder()
    if holder is None:
        return None
    own = [evaluate.get_task(st.session_state.get(key)) for key in (CV_TASK, PERM_TASK)]
    if any(task is not None and not task.finished for task in own):
        return None
    return holder


def _busy_note(holder: str) -> None:
    """Say why a measurement button is disabled."""
    st.caption(f"{holder[:1].upper()}{holder[1:]} is running in this app; only one fit or measurement runs at a "
               "time, so this button waits until it ends.")


@st.fragment(run_every=1.0)
def task_panel(task_id: str, session_key: str) -> None:
    """Live view of a running measurement, refreshed every second: bar, elapsed time and Cancel.

    When the task ends, the whole app reruns once so the new results appear in place.
    """
    task = evaluate.get_task(task_id)
    if task is None or st.session_state.get(session_key) != task_id:
        _rerun_once(task_id)
        return
    snap = task.snapshot()
    title = "Cross-validating" if task.kind == "cv" else f"Measuring permutation importance of {task.label}"
    with st.container(border=True, key=f"{session_key}_box"):
        st.markdown(f"**{html.escape(title)}**")
        st.progress(float(min(max(snap.fraction, 0.0), 1.0)),
                    text=f"{snap.message} · elapsed {format_elapsed(snap.elapsed)}")
        st.button("Cancel", key=f"{session_key}_cancel", disabled=task.finished, on_click=task.cancel)
    if task.finished:
        _rerun_once(task_id)


# --------------------------------------------------------------------------------------------------------------
# Sections
# --------------------------------------------------------------------------------------------------------------
def _score_config(columns: Sequence[str]) -> dict[str, Any]:
    """Column formats for a table of scores and timings."""
    config: dict[str, Any] = {}
    for name in columns:
        if name in viz.SCORE_COLUMNS or name == "Gap to best":
            config[name] = st.column_config.NumberColumn(name, format=SCORE_FORMAT)
        elif name in ("Fit s", "Single-flow ms"):
            config[name] = st.column_config.NumberColumn(name, format="%.2f")
        elif name in ("Flows/s", "Rows used", "Training rows", "support", "Rows"):
            config[name] = st.column_config.NumberColumn(name, format="localized")
    config["Balanced accuracy"] = st.column_config.NumberColumn(
        "Balanced accuracy", format=SCORE_FORMAT, help="Mean recall over the classes: a rare class counts as much "
        "as a common one.")
    config["Gap to best"] = st.column_config.NumberColumn(
        "Gap to best", format=SCORE_FORMAT, help="Best balanced accuracy minus this channel's.")
    config["Rows used"] = st.column_config.NumberColumn(
        "Rows used", format="localized", help="Training rows the channel was fitted on.")
    config["Flows/s"] = st.column_config.NumberColumn(
        "Flows/s", format="localized", help="Held-out flows scored per second, in blocks of 5,000.")
    if "Recall weighted" in columns:
        config["Recall weighted"] = st.column_config.NumberColumn(
            "Recall weighted", format=SCORE_FORMAT, help="Recall averaged with each class weighted by its rows; "
            "this always equals accuracy.")
    if "Precision weighted" in columns:
        config["Precision weighted"] = st.column_config.NumberColumn(
            "Precision weighted", format=SCORE_FORMAT, help="Precision averaged with each class weighted by its "
            "held-out rows.")
    config["Single-flow ms"] = st.column_config.NumberColumn(
        "Single-flow ms", format="%.2f", help="Median time to score one flow on its own (up to 30 calls, at least 5).")
    return config


def _overview(run: TrainingRun, evals: dict[str, ChannelEvaluation], board: pd.DataFrame, mode: Mode) -> None:
    """(1) The leaderboard, its dot plot and the held-out class distribution."""
    st.subheader("Readings overview", anchor=False)
    shown = board.drop(columns=["key", "Training rows"])
    st.dataframe(shown, hide_index=True, width="stretch", column_config=_score_config(list(shown.columns)))
    note = svm_rows_note(run)
    if note:
        components.chips([note])
    st.caption(held_out_line(run))
    if not board.empty:
        best = board.iloc[0]
        line = (f"Best balanced accuracy: **{html.escape(str(best['Channel']))}**, "
                f'<span class="g-mono">{float(best["Balanced accuracy"]):.4f}</span>')
        if len(board) > 1:
            runner = board.iloc[1]
            line += (f"; next is {html.escape(str(runner['Channel']))}, "
                     f'<span class="g-mono">{float(runner["Gap to best"]):.4f}</span> behind')
        st.markdown(line + ".", unsafe_allow_html=True)
    left, right = st.columns([3, 2], gap="medium")
    with left:
        _kept_chart(run, ("board", mode), lambda: viz.leaderboard_chart(
            board, mode, subtitle=f"Each mark is one channel's reading on {len(run.data.y_test):,} held-out rows."))
    with right:
        _kept_chart(run, ("held_out", mode), lambda: viz.held_out_classes_chart(evaluate.held_out_counts(run), mode))


def _confusions(run: TrainingRun, evals: dict[str, ChannelEvaluation], mode: Mode) -> None:
    """(2) Every channel's confusion matrix, side by side."""
    st.subheader("Confusion matrices", anchor=False)
    shown = st.radio("Shade and lead with", SHOW_OPTIONS, horizontal=True, key="ms_cm_show",
                     help="Row %: share of each true class's rows that went to each predicted class. Counts: "
                          "rows, shaded on a log scale.")
    show = "count" if shown == "Counts" else "share"
    classes = list(run.data.classes)
    per_row = 3 if len(classes) <= 3 else 2 if len(classes) <= 8 else 1
    keys = list(evals)
    for start in range(0, len(keys), per_row):
        columns = st.columns(per_row, gap="medium")
        for column, key in zip(columns, keys[start:start + per_row]):
            # Each matrix keeps its own size and scrolls sideways in a narrow column instead of overlapping.
            with column, st.container(key=f"g_cm_{key}"):
                _kept_chart(run, ("confusion", key, show, mode), lambda key=key: viz.confusion_chart(
                    evals[key].confusion, classes, mode, show=show, title=channel_label(key)), width="content")
    st.caption("Rows: true class. Columns: the channel's verdict. The diagonal holds the flows read correctly.")


def _curves(run: TrainingRun, evals: dict[str, ChannelEvaluation], mode: Mode) -> None:
    """(3) ROC and precision-recall curves: all channels overlaid (binary), or per class for one channel."""
    st.subheader("Curves", anchor=False)
    classes = list(run.data.classes)
    view = st.radio("ROC view", ROC_OPTIONS, horizontal=True, key="ms_roc_zoom",
                    help="Near-perfect curves crowd the top-left corner; the zoomed view spreads them out.")
    max_fpr = ROC_ZOOM if view == ROC_OPTIONS[1] else 1.0
    if len(classes) == 2:
        name = classes[1]
        roc = {k: ev.roc[name] for k, ev in evals.items() if name in ev.roc}
        pr = {k: ev.pr[name] for k, ev in evals.items() if name in ev.pr}
        chance = float(np.mean(np.asarray(run.data.y_test) == 1)) if len(run.data.y_test) else None
        left, right = st.columns(2, gap="medium")
        with left:
            _kept_chart(run, ("roc", max_fpr, mode), lambda: viz.roc_chart(
                roc, mode, scores={k: ev.metrics.get("roc_auc", math.nan) for k, ev in evals.items()},
                max_fpr=max_fpr))
        with right:
            _kept_chart(run, ("pr", mode), lambda: viz.pr_chart(
                pr, mode, chance=chance,
                scores={k: ev.metrics.get("average_precision", math.nan) for k, ev in evals.items()}))
        st.caption(f"Curves score the {html.escape(name)} class's probability over every threshold; the readings "
                   "above use each channel's own verdict (the most probable class).")
        return
    keys = list(evals)
    key = st.selectbox("Channel", keys, format_func=channel_label, key=run_key("ms_curve_channel", run),
                       help="One-vs-rest curves: each class against all the others, for one channel.")
    ev = evals[key]
    table = ev.per_class.set_index("class")
    support = {str(k): int(v) for k, v in table["support"].items()}
    left, right = st.columns(2, gap="medium")
    with left:
        _kept_chart(run, ("class_roc", key, max_fpr, mode), lambda: viz.class_curves_chart(
            ev.roc, mode, kind="roc", support=support, max_fpr=max_fpr,
            scores={str(k): float(v) for k, v in table["roc_auc"].items()},
            title=f"{channel_label(key)}: one-vs-rest ROC"))
    with right:
        _kept_chart(run, ("class_pr", key, mode), lambda: viz.class_curves_chart(
            ev.pr, mode, kind="pr", support=support,
            scores={str(k): float(v) for k, v in table["average_precision"].items()},
            title=f"{channel_label(key)}: one-vs-rest precision-recall"))


def _per_class_table(ev: ChannelEvaluation) -> None:
    """One channel's per-class readings as a table (class names carry their shape cue)."""
    table = ev.per_class.copy()
    table["class"] = [class_text(str(c)) for c in table["class"]]
    table = table.rename(columns={"class": "Class", "support": "Held-out rows", "precision": "Precision",
                                  "recall": "Recall", "f1": "F1", "roc_auc": "ROC-AUC",
                                  "average_precision": "Average precision"})
    config = {name: st.column_config.NumberColumn(name, format=SCORE_FORMAT)
              for name in ("Precision", "Recall", "F1", "ROC-AUC", "Average precision")}
    config["Held-out rows"] = st.column_config.NumberColumn("Held-out rows", format="localized")
    st.dataframe(table, hide_index=True, width="stretch", column_config=config)


def permutation_estimate(run: TrainingRun, key: str, single_flow_ms: float, rows: int,
                         repeats: int) -> tuple[float, float]:
    """(lower, upper) seconds a permutation measurement of ``key`` should take on this machine.

    Each of the ``1 + features x repeats`` scorings costs the channel's per-row scoring time (measured at the fit)
    for ``rows`` rows, plus the fixed cost of one call, taken as the channel's single-flow latency. The upper end
    assumes no gain from scoring features on several threads; the lower end assumes the full gain (up to four
    threads; the tree channels already use their own threads, so they get none).
    """
    result = run.channels[key]
    per_row = float(result.predict_seconds or 0.0) / max(len(run.data.y_test), 1)
    per_call = float(single_flow_ms) / 1000.0 if math.isfinite(float(single_flow_ms)) else 0.0
    scorings = 1 + len(run.data.feature_names) * int(repeats)
    upper = (per_row * int(rows) + max(per_call, 0.0)) * scorings
    threads = 1 if key in evaluate.THREADED_CHANNELS else max(min(os.cpu_count() or 1, 4), 1)
    return upper / threads, upper


def _permutation_block(run: TrainingRun, ev: ChannelEvaluation, mode: Mode, running: EvaluationTask | None) -> None:
    """Permutation importance of one channel: the button, the progress panel, or the stored result."""
    key = ev.key
    rows, repeats = evaluate.permutation_plan(run, key, max_rows=PERM_ROWS, repeats=PERM_REPEATS)
    n_features = len(run.data.feature_names)
    low, high = permutation_estimate(run, key, ev.single_flow_ms, rows, repeats)
    stored = evaluate.stored_permutations(run).get(key)
    if running is not None:
        task_panel(running.task_id, PERM_TASK)
    else:
        busy = _busy_elsewhere()
        label = "Measure again" if stored is not None else "Measure permutation importance"
        st.button(label, key="ms_perm_run", on_click=_start_permutation, args=(key,), disabled=busy is not None,
                  help="Shuffles one feature at a time in held-out rows and measures the drop in balanced "
                       "accuracy. Nothing is refitted.")
        st.caption(f"Shuffles each of the {n_features} features {repeats} times in {rows:,} held-out rows "
                   f"(rare classes kept): {duration_range_text(low, high)} on this machine.")
        if busy is not None:
            _busy_note(busy)
    if stored is not None:
        seconds = float(stored.attrs.get("seconds", 0.0))
        _chart(lambda: viz.importance_chart(
            stored, mode, error="std", x_title="Drop in balanced accuracy when shuffled", number_format=".4f",
            title=f"{channel_label(key)}: permutation importance",
            subtitle=f"{int(stored.attrs.get('rows', rows)):,} held-out rows, {int(stored.attrs.get('repeats', 0))} "
                     f"repeats, measured in {seconds:,.1f} s. Bar: mean drop; line: one standard deviation."))


def _detail(run: TrainingRun, evals: dict[str, ChannelEvaluation], mode: Mode,
            running: EvaluationTask | None) -> None:
    """(4) One channel in detail: per-class readings, native and permutation importance."""
    st.subheader("Channel detail", anchor=False)
    keys = list(evals)
    key = st.selectbox("Channel", keys, format_func=channel_label, key=run_key("ms_detail_channel", run))
    ev = evals[key]
    result = run.channels[key]
    _per_class_table(ev)
    notes = " ".join(str(n).strip() for n in (result.notes or []) if str(n).strip())
    if notes:
        st.caption(f"Fit notes: {notes}")
    left, right = st.columns(2, gap="medium")
    with left:
        if ev.native_importance is not None:
            what = "mean decrease in impurity" if key == "forest" else "total gain over all splits"
            native = ev.native_importance
            _kept_chart(run, ("native", key, mode), lambda: viz.importance_chart(
                native, mode, title=f"{channel_label(key)}: built-in importance",
                subtitle=f"Share of the model's {what}; top 20 features.",
                colour=theme.CHANNEL_BY_KEY[key].colour(mode)))
        else:
            st.caption(f"{channel_label(key)} has no built-in importance. Permutation importance (right) works for "
                       "every channel.")
    with right:
        _permutation_block(run, ev, mode, running)


def _timing(run: TrainingRun, board: pd.DataFrame, mode: Mode) -> None:
    """(5) Fit time, scoring speed and single-flow latency per channel."""
    st.subheader("Timing", anchor=False)
    left, right = st.columns(2, gap="medium")
    with left:
        _kept_chart(run, ("fit_s", mode), lambda: viz.timing_chart(
            board, mode, measure="Fit s", subtitle="Fitting on the training rows (the SVM on its capped subset)."))
    with right:
        _kept_chart(run, ("flows", mode), lambda: viz.timing_chart(
            board, mode, measure="Flows/s", subtitle="Held-out flows scored in batches of at least 5,000."))
    left, right = st.columns(2, gap="medium")
    with left:
        _kept_chart(run, ("latency", mode), lambda: viz.timing_chart(
            board, mode, measure="Single-flow ms", subtitle="Median of up to 30 calls scoring one flow on its own."))
    with right:
        st.caption("Times are wall-clock seconds on this machine while the app was running, so they move a little "
                   "from fit to fit. Scoring one flow at a time costs far more per flow than scoring a batch.")


def _cv_results(run: TrainingRun, mode: Mode) -> None:
    """The stored cross-validation result: table, spread chart and how it was measured."""
    cv = evaluate.stored_cross_validation(run)
    if cv is None or cv.empty:
        return
    attrs = cv.attrs
    table = cv.drop(columns=["key"])
    if table["Error"].isna().all():
        table = table.drop(columns=["Error"])
    config = {name: st.column_config.NumberColumn(name, format=SCORE_FORMAT) for name in table.columns
              if name.endswith(" mean") or name.endswith(" std")}
    config["Fit s mean"] = st.column_config.NumberColumn("Fit s mean", format="%.2f")
    config["Rows"] = st.column_config.NumberColumn("Rows", format="localized")
    st.dataframe(table, hide_index=True, width="stretch", column_config=config)
    done = (f"{int(attrs.get('k', 0))} stratified folds of {int(attrs.get('rows', 0)):,} training rows (of "
            f"{int(attrs.get('rows_available', 0)):,}), measured in {float(attrs.get('seconds', 0.0)):,.1f} s. "
            "Standard deviations are over folds.")
    if attrs.get("note"):
        done += f" {attrs['note']}"
    if attrs.get("cancelled"):
        done += " Cancelled before every channel finished: only complete channels are shown."
    st.caption(done)
    metric = st.radio("Spread of", tuple(viz.CV_METRIC_FIELDS), index=1, horizontal=True, key="ms_cv_metric")
    _chart(lambda: viz.cv_spread_chart(cv, mode, metric=str(metric), folds=evaluate.cv_fold_frame(cv)))


def _cross_validation(run: TrainingRun, mode: Mode, running: EvaluationTask | None) -> None:
    """(6) Cross-validation on demand: folds, channels, an estimate first, then progress and the result."""
    st.subheader("Cross-validation", anchor=False)
    st.caption(f"Refits fresh copies of the channels in stratified folds of up to {CV_MAX_ROWS:,} training rows "
               "(rare classes kept). The held-out rows play no part, so these spreads show how much a reading "
               "moves with the training data; notes below say when they are not comparable with the readings "
               "above.")
    left, right = st.columns([1, 3], gap="medium")
    with left:
        k = st.number_input("Folds (k)", min_value=CV_FOLDS[0], max_value=CV_FOLDS[1], value=CV_DEFAULT_FOLDS,
                            step=1, key="ms_cv_k")
    with right:
        chosen = st.multiselect("Channels", run.ok_channels(), default=run.ok_channels(), format_func=channel_label,
                                key=run_key("ms_cv_channels", run))
    channels = [key for key in run.ok_channels() if key in (chosen or [])]
    try:
        plan = evaluate.plan_cross_validation(run, int(k), CV_MAX_ROWS, channels)
    except ValueError as exc:
        st.caption(str(exc))
        _cv_results(run, mode)
        return
    seconds = evaluate.estimate_cv_seconds(run, int(k), CV_MAX_ROWS, channels=channels)
    if running is not None:
        task_panel(running.task_id, CV_TASK)
    else:
        busy = _busy_elsewhere()
        st.button("Run cross-validation", key="ms_cv_run", type="primary", on_click=_start_cv,
                  disabled=not channels or busy is not None)
        if channels:
            st.caption(f"{duration_text(seconds).capitalize()} on this machine: {plan.k} folds for each of "
                       f"{len(channels)} channel{'s' if len(channels) != 1 else ''}, on {plan.rows:,} training rows."
                       + (f" {plan.note}" if plan.note else ""))
        else:
            st.caption("Choose at least one channel.")
        if busy is not None:
            _busy_note(busy)
    _cv_results(run, mode)


def _downloads(run: TrainingRun, evals: dict[str, ChannelEvaluation], board: pd.DataFrame) -> None:
    """(7) Quick CSV downloads (UTF-8 with a byte-order mark, so spreadsheet programs read them correctly)."""
    st.subheader("Downloads", anchor=False)
    columns = st.columns(3, gap="small")
    with columns[0]:
        st.download_button("Leaderboard (CSV)", board.to_csv(index=False).encode("utf-8-sig"),
                           file_name=f"graticule-leaderboard-{run.run_id}.csv", mime="text/csv",
                           key="ms_dl_board", on_click="ignore", width="stretch")
    with columns[1]:
        per_class = evaluate.per_class_frame(evals)
        st.download_button("Per-class readings (CSV)", per_class.to_csv(index=False).encode("utf-8-sig"),
                           file_name=f"graticule-per-class-{run.run_id}.csv", mime="text/csv",
                           key="ms_dl_per_class", on_click="ignore", width="stretch")
    with columns[2]:
        cv = evaluate.stored_cross_validation(run)
        if cv is not None and not cv.empty:
            st.download_button("Cross-validation (CSV)", cv.to_csv(index=False).encode("utf-8-sig"),
                               file_name=f"graticule-cross-validation-{run.run_id}.csv", mime="text/csv",
                               key="ms_dl_cv", on_click="ignore", width="stretch")
        else:
            st.caption("Run cross-validation to add its table here.")
    record = PAGE_OBJECTS.get("record")
    if record is not None:
        st.page_link(record, label=f"{BY_KEY['record'].label}: the full PDF record and every CSV",
                     icon=":material/arrow_forward:")


def _no_held_out_rows(run: TrainingRun) -> None:
    """A loaded channel set without its held-out rows: say why nothing can be measured and where to go.

    A CIC-IDS2017 run needs its data folder (the Bench); a synthetic run needs no folder, only a generator that
    still draws the very same sample, so it gets no Bench advice.
    """
    if run.data_request.source == "synthetic":
        components.needs(
            f"Run {run.run_id} was loaded from disk without its held-out rows, so there is nothing to measure here. "
            "Its sample was generated from a seed, but generating it again did not give the very same rows (the "
            "generator or the program has changed since the run was saved), so they cannot be rebuilt. Fit again "
            "at 02 Fit to measure these channel types on a fresh sample.", "logbook")
        info = state.loaded_info(run.run_id)
        if info is not None and info.rows_note:
            st.caption(info.rows_note)
    else:
        components.needs(
            f"Run {run.run_id} was loaded from disk without its held-out rows, so there is nothing to measure here. "
            "Point the Bench at the data folder it was fitted on, then load it again at the Logbook: the same "
            "held-out rows are rebuilt and checked against the saved fingerprints.", "logbook")
        bench = PAGE_OBJECTS.get("bench")
        if bench is not None:
            st.page_link(bench, label=f"Go to {BY_KEY['bench'].label}", icon=":material/arrow_forward:")
    rows = []
    for key in run.ok_channels():
        metrics = (run.channels[key].extra or {}).get("metrics") or {}
        if metrics:
            rows.append({"Channel": channel_label(key), "Balanced accuracy": metrics.get("balanced_accuracy"),
                         "Accuracy": metrics.get("accuracy"), "F1 macro": metrics.get("f1_macro"),
                         "F1 weighted": metrics.get("f1_weighted")})
    if rows:
        st.markdown("Readings recorded when these channels were fitted (from the saved set; not measured again):")
        frame = pd.DataFrame(rows)
        st.dataframe(frame, hide_index=True, width="stretch",
                     column_config={c: st.column_config.NumberColumn(c, format=SCORE_FORMAT)
                                    for c in frame.columns if c != "Channel"})


def _run_chips(run: TrainingRun) -> list[str]:
    """Short facts about the run being measured."""
    request = run.request
    chips = [
        f"run {run.run_id}",
        "binary: normal vs attack" if request.mode == "binary" else f"multi-class: {len(run.data.classes)} classes",
        f"{len(run.data.feature_names)} columns",
        f"test {len(run.data.y_test):,} rows",
        f"{len(run.ok_channels())} channel{'s' if len(run.ok_channels()) != 1 else ''}",
    ]
    if getattr(run, "origin", "fitted") == "loaded":
        chips.append("loaded from disk")
    if request.profile != "full":
        chips.append(f"profile {request.profile}")
    return chips


def _evaluations(run: TrainingRun) -> dict[str, ChannelEvaluation]:
    """The run's readings: computed on the first visit (with a spinner), read from the run afterwards."""
    cached = evaluate.cached_evaluations(run)
    if cached is not None and set(cached) >= set(run.ok_channels()):
        return evaluate.evaluate_run(run)
    with st.spinner("Taking readings on the held-out rows..."):
        return evaluate.evaluate_run(run)


def render() -> None:
    """Draw the 03 Measure station."""
    components.station_header("measure")
    run = state.current_run()
    if run is None or not run.ok_channels():
        components.needs("Needs a fitted channel: fit at least one at 02 Fit.", "fit")
        return
    components.chips(_run_chips(run))
    if not evaluate.has_test_rows(run):
        _no_held_out_rows(run)
        return
    prepared = state.get_prepared()
    if prepared is not None and prepared.fingerprint != run.dataset_fingerprint:
        st.caption(f"These readings belong to run {run.run_id}, fitted on an earlier sample; the sample loaded now "
                   "has not been fitted.")
    cv_task, perm_task = _collect(CV_TASK), _collect(PERM_TASK)  # a finished task leaves its message first
    _show_notice()
    evals = _evaluations(run)
    mode = _mode()
    board = evaluate.leaderboard(evals, run)
    _overview(run, evals, board, mode)
    _confusions(run, evals, mode)
    _curves(run, evals, mode)
    _detail(run, evals, mode, perm_task)
    _timing(run, board, mode)
    _cross_validation(run, mode, cv_task)
    _downloads(run, evals, board)
