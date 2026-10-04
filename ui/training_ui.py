"""Streamlit side of a fit: starting the job, the live progress panel and the readings of a finished run.

The training itself lives in :mod:`graticule.models` and never touches Streamlit. This module only

* starts a :class:`~graticule.models.jobs.TrainingJob` from the Fit button's callback (in a background thread, or
  on the calling thread when ``GRATICULE_SYNC_TRAINING=1``, which the headless UI tests set), so the page that
  follows already shows the running fit;
* polls the job once a second from a fragment, so only the progress panel redraws while channels are fitted;
* draws the readings of a stored run. Readings are derived from the predictions stored in the run (never by
  refitting or predicting again) and are kept on the run object itself, so reruns stay cheap and two runs sharing
  an id (a fit and its copy loaded from disk) never share a table.

Every job also writes its run to the run history from its own thread as soon as the run exists
(:func:`record_finished_run`), so a fit that no page ever adopts (its tab was closed) is still recorded.

Test hook: when the environment variable ``GRATICULE_TEST_PROFILE`` is ``"1"`` every fit uses the shrunken
``profile="test"`` models (few trees, rounds and iterations), which keeps the UI tests fast. It is never set in
normal use.
"""

from __future__ import annotations

import functools
import html
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from graticule import theme
from graticule.data.prepare import PreparedDataset
from graticule.data.sampling import SingleClassError
from graticule.evaluate import held_out_repeats, quick_metrics, repeats_sentence
from graticule.history import RunHistory
from graticule.models.jobs import JobBusyError, TrainingCancelled, TrainingJob, get_job, sync_training_requested
from graticule.models.train import TrainingRun, TrainRequest
from graticule.models.zoo import MODEL_KEYS, Profile
from graticule.schema import is_normal_traffic
from ui import components, state
from ui.stations import BY_KEY, PAGE_OBJECTS

TEST_PROFILE_ENV = "GRATICULE_TEST_PROFILE"
# Session flag: the job id for which the progress panel has already asked for a full rerun (never twice).
RERUN_DONE = "fit_rerun_for_job"
STATUS_TEXT = {"ok": "fitted", "failed": "failed", "cancelled": "cancelled", "skipped": "skipped",
               "waiting": "waiting", "queued": "waiting", "running": "fitting", "done": "fitted",
               "not_saved": "not saved"}
READING_COLUMNS: tuple[str, ...] = ("Channel", "Status", "Rows used", "Fit s", "Flows/s", "Accuracy",
                                    "Balanced accuracy", "Macro F1", "Notes")
#: Attribute of a run object holding its readings table (see :func:`kept_readings`).
READINGS_ATTR = "fit_readings_table"
#: Characters with a meaning in Markdown, escaped by :func:`plain_markdown`.
MARKDOWN_SPECIALS = frozenset("\\`*_{}[]<>()#+-.!|$~")


def training_profile() -> Profile:
    """``"test"`` when the hidden test hook ``GRATICULE_TEST_PROFILE=1`` is set, else ``"full"``."""
    return "test" if os.environ.get(TEST_PROFILE_ENV, "").strip() == "1" else "full"


def channel_label(key: str) -> str:
    """Badge and name of a channel, e.g. ``"CH3 RBF SVM"``."""
    style = theme.CHANNEL_BY_KEY.get(key)
    return style.label if style is not None else key


def format_elapsed(seconds: float) -> str:
    """Elapsed time as ``mm:ss`` (minutes keep counting past an hour)."""
    total = max(int(seconds), 0)
    return f"{total // 60:02d}:{total % 60:02d}"


# --------------------------------------------------------------------------------------------------------------
# Starting a fit
# --------------------------------------------------------------------------------------------------------------
def record_finished_run(run: TrainingRun, *, db_path: Path | None = None) -> None:
    """Write a finished fit to the run history file ``db_path`` (default: the usual one).

    Called by the job on its own thread, so it touches no session state. The file is fixed when the fit starts,
    so a job that ends later still writes where the app was writing then. Recording is idempotent per run id: the
    session that adopts the run later (which records it again, see :func:`ui.state.record_history`) adds no second
    line.
    """
    RunHistory(db_path).record(run)


def start_fit(prepared: PreparedDataset, request: TrainRequest) -> bool:
    """Start fitting ``request`` on ``prepared``: in the background, or inline when synchronous fits are requested.

    Meant for the Fit button's callback, which runs before the page is drawn: it draws nothing itself. A background
    fit stores its job id in the session, and the page then shows the progress panel (and a disabled Fit button)
    in the same rerun. An inline fit stores its run, whose readings the page then shows. Every outcome worth a word
    (finished, nothing fitted, another fit running, failure) is left as the fit message that 02 Fit shows once.
    Returns True when a job started or a run was stored.
    """
    job = TrainingJob(prepared, request,
                      on_finished=functools.partial(record_finished_run, db_path=RunHistory().path))
    if not sync_training_requested():
        try:
            job.start()
        except JobBusyError as exc:
            state.set_fit_notice("warning", str(exc))
            return False
        st.session_state[state.JOB_ID] = job.job_id
        st.session_state.pop(RERUN_DONE, None)
        state.run_registry().watch(job.job_id)  # its run survives even if this tab is refreshed before it ends
        return True
    try:
        run = job.run_inline()
    except SingleClassError as exc:
        state.set_fit_notice("warning", f"Nothing was fitted. {exc}")
        return False
    except JobBusyError as exc:
        state.set_fit_notice("warning", str(exc))
        return False
    except TrainingCancelled:
        state.set_fit_notice("info", "Fit cancelled before any channel was fitted. Nothing was stored; the previous "
                                     "readings, if any, are unchanged.")
        return False
    except Exception as exc:  # noqa: BLE001 - any failure is reported on the page instead of crashing it
        state.set_fit_notice("error", f"The fit failed: {state.first_line(str(exc)) or type(exc).__name__}")
        return False
    outcome = state.adopt_run(run, cancelled=bool(getattr(run, "cancelled", False)))
    job.release_result()  # the session and the registry hold the run now
    return outcome == "stored"


# --------------------------------------------------------------------------------------------------------------
# Progress panel
# --------------------------------------------------------------------------------------------------------------
def _progress_table(snapshot: Any) -> pd.DataFrame:
    """One row per channel of a job snapshot: badge, status, share done, seconds and the latest message."""
    rows = [
        {"Channel": channel_label(cp.key), "Status": STATUS_TEXT.get(cp.status, cp.status),
         "Done": float(min(max(cp.fraction, 0.0), 1.0)), "Seconds": float(cp.seconds), "Detail": cp.message or ""}
        for cp in snapshot.channels
    ]
    return pd.DataFrame(rows, columns=["Channel", "Status", "Done", "Seconds", "Detail"])


def stage_text(stage: str) -> str:
    """The job's stage line with channel keys written as badges ("Fitting xgboost" -> "Fitting CH2 XGBoost")."""
    words = stage.split(" ")
    return " ".join(channel_label(w) if w in theme.CHANNEL_BY_KEY else w for w in words) if stage else "Starting"


def _ask_for_full_rerun(job_id: str) -> None:
    """Rerun the whole app once for ``job_id`` (the session flag makes a second request a no-op)."""
    if st.session_state.get(RERUN_DONE) != job_id:
        st.session_state[RERUN_DONE] = job_id
        st.rerun(scope="app")


@st.fragment(run_every=1.0)
def progress_panel(job_id: str) -> None:
    """Live view of a background fit, refreshed every second: bar, elapsed time, stage, channels and Cancel.

    When the job ends, its result is adopted (or its failure explained) and the whole app reruns exactly once, so
    the stepper tick and the readings appear together.
    """
    job = get_job(job_id)
    if job is None or st.session_state.get(state.JOB_ID) != job_id:
        _ask_for_full_rerun(job_id)  # adopted elsewhere (for example by the stepper) or no longer known
        return
    snapshot = job.snapshot()
    with st.container(border=True, key="fit_progress"):
        st.markdown("**Fitting**")
        st.progress(float(min(max(snapshot.fraction, 0.0), 1.0)),
                    text=f"{stage_text(snapshot.stage)} · elapsed {format_elapsed(snapshot.elapsed)}")
        if snapshot.channels:
            st.dataframe(
                _progress_table(snapshot), hide_index=True, width="stretch",
                column_config={
                    "Done": st.column_config.ProgressColumn("Done", min_value=0.0, max_value=1.0, format="percent"),
                    "Seconds": st.column_config.NumberColumn("Seconds", format="%.1f"),
                },
            )
        left, right = st.columns([3, 1], vertical_alignment="center")
        with left:
            st.caption("Other stations stay usable while this runs; the readings appear here when it ends.")
        with right:
            if st.button("Cancel", key="fit_cancel", width="stretch", disabled=job.finished):
                job.cancel()
                st.caption("Cancelling after the current step (a tree chunk, boosting round, epoch, iteration or "
                           "batch of test rows; the kernel SVM's own fit cannot be interrupted).")
    if job.finished:
        state.adopt_job(job)
        _ask_for_full_rerun(job_id)


# --------------------------------------------------------------------------------------------------------------
# Readings of a stored run
# --------------------------------------------------------------------------------------------------------------
def ordered_channels(run: TrainingRun) -> list[str]:
    """Channel keys of ``run`` in the fixed CH1..CH5 order."""
    return [key for key in MODEL_KEYS if key in run.channels] + [k for k in run.channels if k not in MODEL_KEYS]


def _channel_notes(result: Any) -> str:
    """The notes of one channel result as one line (its error first when it failed)."""
    parts: list[str] = []
    if result.error:
        parts.append(state.first_line(str(result.error)))
    parts.extend(str(note).strip() for note in (result.notes or []))
    return " ".join(part if part.endswith((".", "!", "?")) else f"{part}." for part in parts if part)


def build_readings(run: TrainingRun) -> pd.DataFrame:
    """The readings table of ``run``: one row per channel, metrics from the stored test-set predictions."""
    y_test = np.asarray(run.data.y_test)
    n_classes = len(run.data.classes)
    rows: list[dict[str, object]] = []
    for key in ordered_channels(run):
        result = run.channels[key]
        extra = result.extra or {}
        metrics: Mapping[str, float] = {}
        speed: float | None = None
        if result.status == "ok" and result.y_pred is not None:
            # The trainer stores these with the run; derive them from the stored predictions only if absent.
            metrics = extra.get("metrics") or quick_metrics(y_test, np.asarray(result.y_pred), n_classes)
            speed = extra.get("flows_per_second")
            if speed is None and result.predict_seconds and result.predict_seconds > 0:
                speed = len(y_test) / result.predict_seconds
        rows.append({
            "Channel": channel_label(key),
            "Status": STATUS_TEXT.get(result.status, result.status),
            "Rows used": int(result.rows_used),
            "Fit s": float(result.fit_seconds),
            "Flows/s": int(round(float(speed))) if speed is not None else None,
            "Accuracy": metrics.get("accuracy"),
            "Balanced accuracy": metrics.get("balanced_accuracy"),
            "Macro F1": metrics.get("f1_macro"),
            "Notes": _channel_notes(result),
        })
    return pd.DataFrame(rows, columns=list(READING_COLUMNS))


def kept_readings(run: TrainingRun) -> pd.DataFrame:
    """:func:`build_readings` once per run object, kept on the object (never modify the returned frame).

    Keeping the table on the run itself, rather than in a cache keyed by run id, means a fit and every copy of it
    loaded from disk (which share the id but may carry readings and notes of their own) each show their own table.
    """
    table = run.__dict__.get(READINGS_ATTR)
    if not isinstance(table, pd.DataFrame):
        table = build_readings(run)
        run.__dict__[READINGS_ATTR] = table
    return table


def split_sizes(run: TrainingRun) -> tuple[int, int, bool]:
    """(training rows, test rows, rows in memory) of a run.

    A run loaded from disk without its held-out rows holds no matrices; its sizes then come from the reports saved
    with it, and the third value is False.
    """
    n_train, n_test = len(run.data.y_train), len(run.data.y_test)
    if n_test:
        return n_train, n_test, True
    rows = (run.data.reports or {}).get("rows") or {}
    return _int(rows.get("train")) or n_train, _int(rows.get("test")), False


def best_channel(readings: pd.DataFrame) -> tuple[str, float] | None:
    """Channel label and balanced accuracy of the best fitted channel (ties go to the lower channel number)."""
    scored = readings.dropna(subset=["Balanced accuracy"])
    if scored.empty:
        return None
    position = int(np.argmax(scored["Balanced accuracy"].to_numpy(dtype=float)))
    row = scored.iloc[position]
    return str(row["Channel"]), float(row["Balanced accuracy"])


def svm_rows_note(run: TrainingRun) -> str | None:
    """How many training rows CH3 saw, e.g. ``"CH3 trained on 20,000 of 150,321 rows (SVM cap)"``; None without CH3.

    The kernel SVM never sees the whole training split: it is capped, and the rows it does not see calibrate its
    probabilities. The note says which of the two limited it.
    """
    result = run.channels.get("svm")
    if result is None or result.status != "ok" or not 0 < result.rows_used < result.rows_available:
        return None
    cap = _int((result.extra or {}).get("svm_cap")) or int(run.request.svm_cap)
    badge = theme.CHANNEL_BY_KEY["svm"].badge
    reason = (f"SVM cap {cap:,}" if result.rows_used >= cap
              else f"under the cap of {cap:,}; the other rows calibrate its probabilities")
    return f"{badge} trained on {result.rows_used:,} of {result.rows_available:,} rows ({reason})"


def _int(value: Any) -> int:
    """An integer report value (0 when missing or not a number)."""
    return int(value) if isinstance(value, (int, np.integer)) and not isinstance(value, bool) else 0


def _counts_text(counts: Mapping[str, Any] | None, limit: int = 5) -> str:
    """``BENIGN 1,234; DoS Hulk 56`` (and how many more), skipping zero counts."""
    items = [(str(k), _int(v)) for k, v in (counts or {}).items() if _int(v) > 0]
    text = "; ".join(f"{name} {count:,}" for name, count in items[:limit])
    if len(items) > limit:
        text += f"; {len(items) - limit} more"
    return text


def _left_out(dropped: Mapping[str, Any] | None, reasons: Mapping[str, Any] | None) -> list[str]:
    """``Name (12 rows, below the minimum ...)`` for every class left out."""
    return [f"{name} ({_int(count):,} rows, {(reasons or {}).get(name, 'too few rows')})"
            for name, count in (dropped or {}).items()]


def run_notes(run: TrainingRun) -> list[str]:
    """Plain-language notes on what happened between the sample and the fitted channels (from the run's reports)."""
    reports: Mapping[str, Any] = run.data.reports or {}
    notes: list[str] = []
    target = reports.get("target") or {}
    left_out = _left_out(target.get("dropped"), target.get("dropped_reasons"))
    late = _left_out(target.get("dropped_after_dedupe"), target.get("dropped_after_dedupe_reasons"))
    if left_out or late:
        text = "**Classes left out.** "
        if left_out:
            text += "; ".join(left_out) + ". "
        if late:
            text += "After de-duplication on the chosen columns: " + "; ".join(late) + ". "
        notes.append(text.strip())
    duplicates = reports.get("model_space_duplicates") or {}
    removed = _int(duplicates.get("rows_removed"))
    columns = _int(duplicates.get("columns")) or len(run.data.feature_names)
    if removed:
        detail = _counts_text(duplicates.get("by_class"))
        notes.append(f"**Repeated rows.** Over the {columns} columns in play, {removed:,} of "
                     f"{_int(duplicates.get('rows_before')):,} rows{f' ({detail})' if detail else ''} repeat another "
                     "row of the same class. The copies were removed before the split, so no row of a class appears "
                     f"in both the training and the test rows over these {columns} columns"
                     + (" (the Top-K note below counts test rows that match a training row on the columns kept)."
                        if run.request.feature_mode == "topk" else "."))
    else:
        notes.append(f"**Repeated rows.** None: every row is distinct over the {columns} columns in play.")
    conflicts = reports.get("conflicts") or {}
    if _int(conflicts.get("groups")):
        policy = conflicts.get("policy", run.request.conflict_policy)
        fate = ("kept, so the channels are measured on them too" if policy == "keep"
                else f"handled by the {policy} policy ({_int(conflicts.get('rows_removed')):,} rows removed)")
        notes.append(f"**Conflicting labels.** {_int(conflicts.get('groups')):,} groups of "
                     f"identical rows carry different classes ({_int(conflicts.get('rows')):,} rows: "
                     f"{_counts_text(conflicts.get('by_class'), 4)}); {fate}.")
    overlap = reports.get("topk_overlap")
    if run.request.feature_mode == "topk" and overlap:
        seen, tested = _int(overlap.get("test_rows_seen_in_train")), _int(overlap.get("test_rows"))
        share = seen / tested if tested else 0.0
        columns = _int(overlap.get("columns")) or len(run.data.feature_names)
        port = bool(overlap.get("port_added", run.data.feature_choice.include_port))
        kept = _int(overlap.get("kept")) or columns - int(port)
        candidates = _int(overlap.get("candidates"))
        ranked = f"{candidates} candidate columns were" if candidates else "The candidate columns were"
        port_text = "; Destination Port was added by choice and was not ranked" if port else ""
        notes.append(f"**Top-K.** {ranked} ranked on {_int(overlap.get('ranked_on_rows')):,} training rows only "
                     f"({float(overlap.get('ranking_seconds') or 0.0):.1f} s) and the top {kept} kept{port_text}. "
                     f"{seen:,} of {tested:,} test rows ({share:.1%}) share their values on the {columns} columns in "
                     "use with some training row, which makes them easier to read correctly.")
    for key in ordered_channels(run):
        result = run.channels[key]
        if result.status != "ok":
            reason = _channel_notes(result) or "no reason was recorded"
            notes.append(f"**{channel_label(key)} {STATUS_TEXT.get(result.status, result.status)}.** {reason}")
    return notes


def _run_chips(run: TrainingRun) -> list[str]:
    """Short facts about a run for the badge row."""
    request = run.request
    choice = run.data.feature_choice
    feature_text = {"curated": "curated", "all": "all numeric", "topk": f"top {request.top_k}"}.get(
        request.feature_mode, request.feature_mode)
    n_train, n_test, _ = split_sizes(run)
    chips = [
        f"run {run.run_id}",
        "binary: normal vs attack" if request.mode == "binary" else f"multi-class: {len(run.data.classes)} classes",
        f"{len(run.data.feature_names)} columns ({feature_text})",
        "weights: balanced" if request.balanced else "weights: none",
        f"train {n_train:,} · test {n_test:,} rows",
        f"seed {request.seed}",
        f"fitted in {state.fit_duration_text(run)}",
    ]
    if getattr(choice, "include_port", request.include_port):
        chips.insert(3, "Destination Port included")
    if getattr(run, "origin", "fitted") == "loaded":
        chips.append("loaded from disk")
    if request.profile != "full":
        chips.append(f"profile {request.profile}")
    return chips


def _test_distribution(run: TrainingRun) -> str:
    """The held-out classes and their row counts, with the shape cues (from the saved reports when the rows are
    not in memory)."""
    if len(run.data.y_test):
        codes, counts = np.unique(np.asarray(run.data.y_test), return_counts=True)
        pairs = [(run.data.classes[int(code)], int(count)) for code, count in zip(codes, counts)]
    else:
        recorded = (run.data.reports or {}).get("class_counts_test") or {}
        pairs = [(str(name), _int(count)) for name, count in recorded.items() if _int(count) > 0]
    parts = []
    for name, count in pairs:
        glyph = theme.GLYPH_NORMAL if is_normal_traffic(name) else theme.GLYPH_ATTACK
        parts.append(f"{glyph} {name} {count:,}")
    return " · ".join(parts)


def _sample_check(run: TrainingRun, prepared: PreparedDataset | None) -> None:
    """Say plainly when the run was fitted on a sample other than the one loaded now."""
    if prepared is None:
        st.caption(f"The sample behind run {run.run_id} (fingerprint {run.dataset_fingerprint[:12]}) is not loaded "
                   "in this session; its readings are kept with the run.")
    elif prepared.fingerprint != run.dataset_fingerprint:
        st.warning(f"These readings come from an earlier sample (fingerprint {run.dataset_fingerprint[:12]}). The "
                   f"sample loaded now (fingerprint {prepared.fingerprint[:12]}) has not been fitted: press Fit to "
                   f"fit on it. Until then every later station keeps reading run {run.run_id}.")


def plain_markdown(text: str) -> str:
    """``text`` with every Markdown control character escaped, so it shows exactly as written (feature names such
    as ``Init_Win_bytes_forward`` keep their underscores)."""
    return "".join("\\" + ch if ch in MARKDOWN_SPECIALS else ch for ch in str(text))


def channel_notes(readings: pd.DataFrame, run: TrainingRun) -> list[str]:
    """``"CH2 XGBoost: Early stopping ..."`` for every FITTED channel with notes (a channel that failed is
    explained by :func:`run_notes`), in the table's order."""
    out = []
    for key in ordered_channels(run):
        result = run.channels[key]
        note = _channel_notes(result)
        if result.status == "ok" and note:
            out.append(f"{channel_label(key)}: {note}")
    return out


def readings_panel(run: TrainingRun, prepared: PreparedDataset | None) -> None:
    """Everything known about a stored run, read from the run alone (nothing is refitted or re-predicted).

    The readings table leaves its Notes column out (long sentences would be cut at the page edge); the notes of the
    fitted channels follow the table in full, one line each.
    """
    st.subheader("Readings", anchor=False)
    components.chips(_run_chips(run))
    _sample_check(run, prepared)
    readings = kept_readings(run)
    st.dataframe(
        components.shown_scores(readings.drop(columns=["Notes"]), ("Accuracy", "Balanced accuracy", "Macro F1")),
        hide_index=True, width="stretch",
        column_config={
            "Rows used": st.column_config.NumberColumn("Rows used", format="localized",
                                                       help="Training rows this channel was fitted on."),
            "Fit s": st.column_config.NumberColumn("Fit s", format="%.2f", help="Seconds spent fitting."),
            "Flows/s": st.column_config.NumberColumn("Flows/s", format="localized",
                                                     help="Test flows scored per second (in batches of at least "
                                                     "5,000 rows)."),
            "Accuracy": st.column_config.NumberColumn("Accuracy", format="%.4f"),
            "Balanced accuracy": st.column_config.NumberColumn(
                "Balanced accuracy", format="%.4f", help="Mean recall over classes; rare classes count as much "
                "as common ones."),
            "Macro F1": st.column_config.NumberColumn("Macro F1", format="%.4f"),
        },
    )
    for line in channel_notes(readings, run):
        st.caption(plain_markdown(line))
    svm_note = svm_rows_note(run)
    if svm_note:
        components.chips([svm_note])
    _, n_test, in_memory = split_sizes(run)
    if in_memory:
        st.caption(f"Measured on {n_test:,} held-out rows: {_test_distribution(run)}.")
        repeats = repeats_sentence(held_out_repeats(run))
        if repeats:
            st.caption(repeats)
    else:
        st.caption(f"Measured on {n_test:,} held-out rows when fitted: {_test_distribution(run)}. This run was "
                   "loaded from disk without them; the readings are those saved with it.")
    best = best_channel(readings)
    if best is not None:
        label, value = best
        st.markdown(f"Best balanced accuracy: **{html.escape(label)}**, "
                    f'<span class="g-mono">{theme.score_text(value)}</span>. These readings stay in memory, and '
                    "nothing refits until you press Fit again.", unsafe_allow_html=True)
    for note in run_notes(run):
        st.markdown(note)
    page = PAGE_OBJECTS.get("measure")
    if page is not None:
        st.page_link(page, label=f"Go to {BY_KEY['measure'].label} for the full comparison",
                     icon=":material/arrow_forward:")


def restore_offer(latest_id: str) -> None:
    """Offer (never force) to make the newest run held by this app process the session's current run."""
    def _restore() -> None:
        run = state.run_registry().get(latest_id)
        if run is not None:
            state.store_run(run)
            state.record_history(run)  # a fit adopted by no session yet (its tab was closed) is recorded now
            state.set_fit_notice("info", f"Restored run {latest_id}. Nothing was refitted.")

    with st.container(border=True, key="fit_restore_box"):
        st.markdown("This app still holds a fit made earlier (perhaps before a browser refresh).")
        st.button(f"Restore the last fit (run {latest_id})", key="fit_restore", on_click=_restore)


def channel_options() -> Sequence[str]:
    """Channel keys offered by the Fit form, in CH1..CH5 order."""
    return tuple(MODEL_KEYS)
