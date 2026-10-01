"""05 Assay station: score a whole uploaded CSV of flows with one channel, or with the consensus of every channel.

The work happens only when Score file is pressed. It runs as an exclusive :class:`graticule.evaluate.EvaluationTask`,
so it takes the app's single work slot and never runs alongside a fit or a measurement (the button waits, with a
note, while one of those runs anywhere in the app). The task runs on a background thread with a progress panel that
refreshes itself every second and a Cancel button, or inline when ``GRATICULE_SYNC_TRAINING=1`` (headless tests).

The finished :class:`graticule.scoring.ScoredBatch` is kept in session state (:func:`ui.state.set_last_assay`), so
changing a widget afterwards only redraws the readings; 07 Record exports the same batch. Nothing is ever fitted
here: the channels only predict.
"""

from __future__ import annotations

import html
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from graticule import evaluate, scoring, viz
from graticule.data.reader import DataFileError
from graticule.report import exports
from graticule.evaluate import EvaluationTask
from graticule.models.jobs import CancelToken, JobBusyError, slot_holder, sync_training_requested
from graticule.models.train import TrainingRun
from graticule.scoring import ScoredBatch
from graticule.theme import GLYPH_ALERT, GLYPH_ATTACK, score_text, verdict_text
from ui import components, state
from ui.stations import BY_KEY, PAGE_OBJECTS
from ui.training_ui import format_elapsed

#: Session and widget keys.
UPLOAD = "as_file"
THRESHOLD = "as_threshold"
SCORE = "as_score"
TASK = "as_task"
NOTICE = "as_notice"
RERUN_FOR = "as_rerun_for"
DOWNLOAD = "as_download"
CANCEL = "as_cancel"
#: Copies of the choice widgets' values, kept while the station is not drawn (see :func:`_restore_choices`).
SHADOW = "as_kept_widgets"
#: Attribute of a scored batch holding the chart specs drawn here (built once per batch and theme mode).
CHART_SPECS_ATTR = "assay_chart_specs"
#: The threshold slider's range and step (the same as the Bench's alert threshold).
THRESHOLD_RANGE = (0.50, 0.999)
THRESHOLD_STEP = 0.005
PROBABILITY_FORMAT = "%.4f"


def channel_key(run: TrainingRun) -> str:
    """Widget key of the channel picker; its options depend on the run, so a new run starts it afresh."""
    return f"as_channel-{run.run_id}"


def _batch_of(run: TrainingRun) -> ScoredBatch | None:
    """The session's last scored batch when this very run scored it (a fit and its loaded copy differ)."""
    batch = state.get_last_assay()
    return batch if batch is not None and exports.belongs_to_run(batch, run) else None


def _restore_choices(run: TrainingRun, choices: list[str]) -> None:
    """Before drawing the channel and threshold widgets: give them back their values.

    Streamlit forgets the state of widgets that are not drawn in a run, so after a visit to another station they
    would come back at their defaults and the page would claim the choices differ from the readings shown. The
    viewer's last values are kept in :data:`SHADOW` (see :func:`_keep`); without them the readings' own channel and
    threshold are used, then the defaults.
    """
    shadow = st.session_state.get(SHADOW, {})
    batch = _batch_of(run)
    key = channel_key(run)
    if key not in st.session_state:
        value = shadow.get(key, batch.channel if batch is not None else choices[0])
        st.session_state[key] = value if value in choices else choices[0]
    elif st.session_state[key] not in choices:
        st.session_state[key] = choices[0]
    if THRESHOLD not in st.session_state:
        low, high = THRESHOLD_RANGE
        value = shadow.get(THRESHOLD, batch.alert_threshold if batch is not None else state.settings().alert_threshold)
        st.session_state[THRESHOLD] = float(min(max(float(value), low), high))


def _keep(widget_key: str) -> None:
    """After drawing a widget: save a copy of its value for :func:`_restore_choices`."""
    st.session_state.setdefault(SHADOW, {})[widget_key] = st.session_state.get(widget_key)


# --------------------------------------------------------------------------------------------------------------
# Messages and the scoring task
# --------------------------------------------------------------------------------------------------------------
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


def _work(run: TrainingRun, data: bytes, name: str, channel: str, threshold: float
          ) -> Callable[[Callable[[str, float], None], CancelToken], Any]:
    """The task's work: score the uploaded bytes. Problems with the file itself come back as the result (they are
    the user's to fix, not failures of the app). The task lets go of the bytes when the work ends; a scored batch
    keeps them (``getvalue`` shares the upload's own buffer, no copy) to write its download from the lines as
    written."""
    box = {"data": data}

    def work(progress: Callable[[str, float], None], cancel: CancelToken) -> Any:
        payload = box.pop("data", b"")
        try:
            return scoring.score_upload(run, payload, channel=channel, alert_threshold=threshold, name=name,
                                        progress=progress, cancel=cancel)
        except (scoring.MissingColumnsError, DataFileError) as exc:
            return exc
        finally:
            del payload

    return work


def _finish(task: EvaluationTask) -> None:
    """Turn a finished task into the session's last assay and a one-off message, then forget the task."""
    snap = task.snapshot()
    result = task.result
    name = str(task.extra.get("name") or "the file")
    if snap.state == "done" and isinstance(result, ScoredBatch):
        state.set_last_assay(result)
        state.mark_done("assay")
        _set_notice("success", f"Scored {result.rows:,} flows from {result.source_name} with {result.channel_name} "
                               f"in {result.seconds:,.1f} s.")
    elif isinstance(result, scoring.MissingColumnsError):
        _set_notice("error", str(result))
    elif isinstance(result, DataFileError):
        _set_notice("error", f"{name} could not be scored: {result}")
    elif snap.state == "cancelled":
        _set_notice("info", f"Scoring of {name} cancelled after {snap.elapsed:,.1f} s. The readings shown before, "
                            "if any, are unchanged.")
    else:
        _set_notice("error", f"Scoring of {name} failed: {state.first_line(snap.error) or 'no reason recorded'}")
    evaluate.forget_task(task.task_id)


def _collect() -> EvaluationTask | None:
    """The session's scoring task while it runs; a finished one is adopted, reported and forgotten."""
    task = evaluate.get_task(st.session_state.get(TASK))
    if task is None:
        st.session_state.pop(TASK, None)
        return None
    if task.finished:
        st.session_state.pop(TASK, None)
        _finish(task)
        return None
    return task


def _start() -> None:
    """Score file button callback: read the choices and the upload, then score (in the background, or inline)."""
    run = state.current_run()
    upload = st.session_state.get(UPLOAD)
    if run is None or not run.ok_channels():
        _set_notice("warning", "There is no fitted channel to score with. Fit channels at 02 Fit first.")
        return
    if upload is None:
        _set_notice("warning", "Choose a CSV file to score first.")
        return
    if evaluate.get_task(st.session_state.get(TASK)) is not None:
        return
    choices = scoring.channel_choices(run)
    channel = st.session_state.get(channel_key(run))
    if channel not in choices:
        channel = choices[0]
    threshold = float(st.session_state.get(THRESHOLD, state.settings().alert_threshold))
    name = str(getattr(upload, "name", "") or "uploaded file")
    task = EvaluationTask("assay", run.run_id, _work(run, upload.getvalue(), name, channel, threshold),
                          label=f"batch scoring of {name}", exclusive=True, holder="a batch scoring at 05 Assay")
    task.extra.update(name=name, channel=channel)
    try:
        if sync_training_requested():
            task.run_inline()
            _finish(task)
            return
        task.start()
    except JobBusyError as exc:
        _set_notice("warning", str(exc))
        return
    st.session_state[TASK] = task.task_id
    st.session_state.pop(RERUN_FOR, None)


@st.fragment(run_every=1.0)
def task_panel(task_id: str) -> None:
    """Live view of a running scoring task, refreshed every second: bar, elapsed time and Cancel.

    When the task ends, the whole app reruns once so the readings appear in place.
    """
    task = evaluate.get_task(task_id)
    if task is None or st.session_state.get(TASK) != task_id:
        _rerun_once(task_id)
        return
    snap = task.snapshot()
    with st.container(border=True, key="as_task_box"):
        st.markdown(f"**Scoring {html.escape(str(task.extra.get('name') or 'the file'))}**")
        st.progress(float(min(max(snap.fraction, 0.0), 1.0)),
                    text=f"{snap.message} · elapsed {format_elapsed(snap.elapsed)}")
        st.button("Cancel", key=CANCEL, disabled=task.finished, on_click=task.cancel)
    if task.finished:
        _rerun_once(task_id)


# --------------------------------------------------------------------------------------------------------------
# Sections
# --------------------------------------------------------------------------------------------------------------
def _run_chips(run: TrainingRun) -> list[str]:
    """Short facts about the run that scores the file."""
    fitted = run.ok_channels()
    chips = [
        f"run {run.run_id}",
        "binary: normal vs attack" if run.request.mode == "binary" else f"multi-class: {len(run.data.classes)} classes",
        f"{len(run.data.feature_names)} columns",
        f"{len(fitted)} channel{'s' if len(fitted) != 1 else ''}",
        f"bad values: {run.data_request.nonfinite_strategy}",
    ]
    if getattr(run, "origin", "fitted") == "loaded":
        chips.append("loaded from disk")
    return chips


def _controls(run: TrainingRun, task: EvaluationTask | None) -> None:
    """The upload, the channel and threshold choices, and the Score file button."""
    features = list(run.data.feature_names)
    st.markdown(
        f"Upload a flow CSV laid out like the CIC-IDS2017 files. It must hold the {len(features)} columns the "
        "channels read; other columns are kept in the output as they are. A `Label` column is optional: with one, "
        "the readings include accuracy against the labels.")
    with st.expander(f"The {len(features)} columns the channels read"):
        st.markdown(", ".join(f"`{name}`" for name in features))
    left, right = st.columns([3, 2], gap="medium")
    with left:
        upload = st.file_uploader(
            "Flow CSV (up to 1 GB)", type=["csv"], key=UPLOAD,
            help="Header names are matched after trimming spaces; a byte-order mark, the repeated Fwd Header Length "
                 "column and 'Infinity' cells are handled as in the published files.")
    with right:
        choices = scoring.channel_choices(run)
        _restore_choices(run, choices)
        st.selectbox("Channel", choices, format_func=scoring.channel_name, key=channel_key(run),
                     help="One fitted channel, or the consensus: the mean of every fitted channel's probabilities.")
        st.slider("Alert threshold (attack probability)", THRESHOLD_RANGE[0], THRESHOLD_RANGE[1],
                  step=THRESHOLD_STEP, format="%.3f", key=THRESHOLD,
                  help="A flow read as an attack raises an alert when its attack probability (1 - P(BENIGN) in "
                       "multi-class mode) reaches this value, as at 04 Probe and 06 Sweep. The default comes from "
                       "the Bench.")
        _keep(channel_key(run))
        _keep(THRESHOLD)
    holder = slot_holder()
    busy_elsewhere = holder is not None and task is None
    st.button("Score file", key=SCORE, type="primary", on_click=_start,
              disabled=upload is None or task is not None or busy_elsewhere)
    if busy_elsewhere:
        st.caption(f"{holder[:1].upper()}{holder[1:]} is running in this app; only one fit, measurement or scoring "
                   "runs at a time, so this button waits until it ends.")
    elif upload is None:
        st.caption("Choose a file to score.")


def _kept_chart(batch: ScoredBatch, name: tuple[Any, ...], build: Callable[[], Any]) -> None:
    """Draw a chart whose Vega-Lite spec is built once per batch and option set, then kept with the batch."""
    specs = batch.__dict__.setdefault(CHART_SPECS_ATTR, {})
    spec = specs.get(name)
    if spec is None:
        spec = viz.chart_spec(build)
        specs[name] = spec
    st.vega_lite_chart(spec=spec, width="content", theme=None)


def _cards(batch: ScoredBatch) -> None:
    """Headline readings: rows, bad values, attacks, alerts, time; accuracy and balanced accuracy when labelled."""
    rows = max(batch.rows, 1)
    speed = batch.flows_per_second
    cards: list[dict[str, object]] = [
        {"Reading": "Rows scored", "Value": batch.rows, "Note": "every uploaded row"},
        {"Reading": "Rows with bad values", "Value": batch.rows_with_bad_values,
         "Note": f"scored anyway (strategy: {batch.strategy})"},
        {"Reading": "Attacks found", "Value": batch.attacks,
         "Note": f"{GLYPH_ATTACK} {viz.share_text(batch.attacks / rows)} of rows"},
        {"Reading": "Alerts", "Value": batch.alerts,
         "Note": f"{GLYPH_ALERT} attack verdicts at attack probability ≥ {batch.alert_threshold:.3f}"},
        {"Reading": "Seconds", "Value": f"{batch.seconds:,.2f}",
         "Note": f"{speed:,.0f} flows/s predicting" if speed else "reading and scoring"},
    ]
    if batch.accuracy is not None and batch.balanced_accuracy is not None:
        cards.append({"Reading": "Accuracy", "Value": score_text(batch.accuracy),
                      "Note": f"on {batch.rows_measured:,} labelled rows"})
        cards.append({"Reading": "Balanced accuracy", "Value": score_text(batch.balanced_accuracy),
                      "Note": "mean recall over the classes"})
    if batch.rows_seen_in_training:
        cards.append({"Reading": "Rows the run trained on", "Value": batch.rows_seen_in_training,
                      "Note": "repeat a training row over the channels' columns"})
        if batch.unseen_accuracy is not None and batch.unseen_balanced_accuracy is not None:
            cards.append({"Reading": "Accuracy, other rows", "Value": score_text(batch.unseen_accuracy),
                          "Note": f"on {batch.rows_unseen_measured:,} labelled rows it never trained on"})
            cards.append({"Reading": "Balanced accuracy, other rows",
                          "Value": score_text(batch.unseen_balanced_accuracy),
                          "Note": "the same rows, mean recall over the classes"})
    components.reading_cards(cards)


def _verdict_table(batch: ScoredBatch) -> pd.DataFrame:
    """Rows per verdict, with shape cues and shares."""
    counts = batch.verdict_counts()
    total = max(batch.rows, 1)
    return pd.DataFrame({
        "Verdict": pd.Series([verdict_text(name) for name in counts], dtype="str"),
        "Rows": pd.Series(list(counts.values()), dtype="int64"),
        "Share": pd.Series([viz.share_text(v / total) for v in counts.values()], dtype="str"),
    })


def _against_labels(batch: ScoredBatch) -> None:
    """Verdict counts, and the confusion matrix when the file's labels could be measured against."""
    mode = components.current_mode()
    left, right = st.columns([2, 3], gap="medium")
    with left:
        st.markdown("**Verdicts**")
        st.dataframe(_verdict_table(batch), hide_index=True, width="stretch",
                     column_config={"Rows": st.column_config.NumberColumn("Rows", format="localized")})
    with right:
        if batch.confusion is not None:
            counts = batch.confusion.to_numpy(dtype=np.int64)
            classes = list(batch.classes)
            with st.container(key="g_cm_assay"):
                _kept_chart(batch, ("confusion", mode), lambda: viz.confusion_chart(
                    counts, classes, mode, show="share", title=f"{batch.channel_name} against the file's labels",
                    subtitle=f"{batch.rows_measured:,} labelled rows. Shade: row %."))
            st.caption("Rows: the file's label. Columns: the verdict. The diagonal holds the flows read correctly.")
        elif batch.labelled:
            st.caption("The file's labels name no class this run was fitted on, so there is nothing to measure them "
                       "against; the verdicts stand on their own.")
        else:
            st.caption("The file has no labels, so the readings stop at the verdicts. Add a Label column to measure "
                       "accuracy as well.")


def _preview(batch: ScoredBatch) -> None:
    """The first scored rows, led by the verdict with its shape cue."""
    shown = batch.preview(scoring.PREVIEW_ROWS)
    st.subheader(f"First {len(shown):,} scored rows", anchor=False)
    config: dict[str, Any] = {
        name: st.column_config.NumberColumn(name, format=PROBABILITY_FORMAT)
        for name in [*batch.probability_columns, scoring.ATTACK_PROBABILITY]
    }
    st.dataframe(shown, width="stretch", column_config=config)
    renamed = [c for c in shown.columns if str(c).endswith(scoring.UPLOADED_SUFFIX)]
    st.caption("The index is each row's position in the file (0 is the first data row, as in every export). "
               "○ normal traffic, ◆ an attack class, ▲ an alert (an attack verdict at attack probability ≥ "
               f"{batch.alert_threshold:.3f})."
               + (f" Uploaded columns named like a result column are shown as {', '.join(map(str, renamed))}."
                  if renamed else ""))


def _download(batch: ScoredBatch) -> None:
    """The full scored CSV (built only when the button is pressed) and a link to 07 Record."""
    st.download_button("Download the scored CSV", data=batch.to_csv_bytes, file_name=batch.file_name,
                       mime="text/csv", key=DOWNLOAD, on_click="ignore", type="primary")
    if batch.values_as_written:
        uploaded = "every uploaded line exactly as written"
    else:
        uploaded = ("the uploaded columns as read (feature columns as 32-bit numbers, which round integers beyond "
                    "16,777,216; see the note above)")
    st.caption(f"All {batch.rows:,} rows: {uploaded}, then the verdict, every class probability, the attack "
               "probability and the alert flag (UTF-8 CSV with a byte-order mark).")
    record = PAGE_OBJECTS.get("record")
    if record is not None:
        st.page_link(record, label=f"{BY_KEY['record'].label}: the PDF report and every CSV, this assay included",
                     icon=":material/arrow_forward:")


def _results(batch: ScoredBatch, run: TrainingRun) -> None:
    """Everything about the last scored file."""
    st.subheader("Readings", anchor=False)
    st.caption(f"{batch.source_name} · scored by {batch.channel_name} of run {batch.run_id} · alert: attack "
               f"verdict at attack probability ≥ {batch.alert_threshold:.3f}")
    if batch.run_id != run.run_id:
        st.caption(f"These readings come from run {batch.run_id}, not the current run {run.run_id}. Press Score "
                   "file to score the file with the current run.")
    elif not exports.belongs_to_run(batch, run):
        before = "as fitted" if getattr(batch, "run_origin", "fitted") == "fitted" else "as loaded from disk earlier"
        st.caption(f"These readings were taken with another copy of run {run.run_id} ({before}), not with the one "
                   "in use now (a saved set never holds CH3, and may lack its held-out rows), so 07 Record leaves "
                   "them out. Press Score file to score the file with the run in use.")
    else:
        chosen = st.session_state.get(channel_key(run), batch.channel)
        threshold = st.session_state.get(THRESHOLD, batch.alert_threshold)
        if chosen != batch.channel or not np.isclose(float(threshold), batch.alert_threshold):
            st.caption("The channel or threshold chosen above differs from these readings; press Score file to "
                       "score again with them.")
    _cards(batch)
    _against_labels(batch)
    for note in batch.notes:
        st.caption(note)
    _preview(batch)
    _download(batch)


def render() -> None:
    """Draw the 05 Assay station."""
    task = _collect()  # a finished task is adopted first (the shell redraws the stepper with its tick afterwards)
    components.station_header("assay")
    run = state.current_run()
    if run is None or not run.ok_channels():
        components.needs("Needs a fitted channel: fit at least one at 02 Fit, or load a saved channel set at the "
                         "Logbook.", "fit")
        return
    components.chips(_run_chips(run))
    if not run.has_test_rows:
        st.caption(f"Run {run.run_id} was loaded from disk without its held-out rows. Scoring an uploaded file "
                   "needs only the fitted channels, so it works as usual.")
    _show_notice()
    _controls(run, task)
    if task is not None:
        task_panel(task.task_id)
    batch = state.get_last_assay()
    if batch is not None:
        _results(batch, run)
