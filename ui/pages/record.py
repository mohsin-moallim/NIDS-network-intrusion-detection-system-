"""07 Record station: the PDF measurement record of the current run, and CSV exports of every result.

Nothing is computed while the page draws. The PDF is built only when "Build PDF record" is pressed, as an exclusive
:class:`nids.evaluate.EvaluationTask`: it takes the app's single work slot (so it never runs alongside a fit or
a measurement), runs on a background thread with a progress panel that refreshes itself every second (inline when
``NIDS_SYNC_TRAINING=1``, as in the headless tests), and can be cancelled. The finished PDF is kept in the
session for its download button. Each CSV download builds its bytes only when it is clicked. Nothing here fits a
model.

The extras of the record come from the other stations: cross-validation and permutation importance from the run
(03 Measure keeps them on it), the last scored file from :func:`ui.state.get_last_assay` and the simulation from
:func:`ui.pages.sweep.current_session`, each only when it was made with the current run.
"""

from __future__ import annotations

import html
import weakref
from dataclasses import asdict
from typing import Any

import streamlit as st

from nids import evaluate
from nids.evaluate import EvaluationTask
from nids.models.jobs import JobBusyError, slot_holder, sync_training_requested
from nids.models.train import TrainingRun
from nids.report import exports
from nids.report import pdf as record_pdf
from ui import components, state
from ui.stations import BY_KEY, PAGE_OBJECTS
from ui.training_ui import format_elapsed

#: Session keys: the running build task, the built PDF, a one-off notice and the task a full rerun was asked for.
PDF_TASK = "rec_pdf_task"
PDF_RESULT = "rec_pdf"
NOTICE = "rec_notice"
RERUN_FOR = "rec_rerun_for"
#: Who holds the work slot while a record is being built (shown to other pages).
HOLDER = "a PDF record build"


# --------------------------------------------------------------------------------------------------------------
# What the other stations produced
# --------------------------------------------------------------------------------------------------------------
def last_assay() -> Any:
    """The last file scored at 05 Assay in this session, or None (also before that station exists)."""
    getter = getattr(state, "get_last_assay", None)
    try:
        return getter() if callable(getter) else None
    except Exception:  # noqa: BLE001 - a missing or changed station must not break this one
        return None


def sweep_session() -> Any:
    """The simulation session of 06 Sweep in this session, or None (also before that station exists)."""
    try:
        from ui.pages import sweep as sweep_page
    except ImportError:
        return None
    getter = getattr(sweep_page, "current_session", None)
    try:
        return getter() if callable(getter) else None
    except Exception:  # noqa: BLE001 - a missing or changed station must not break this one
        return None


def prepared_for(run: TrainingRun) -> Any:
    """The session's 01 Sample when it is the very sample ``run`` was fitted on, else None."""
    prepared = state.get_prepared()
    if prepared is None or getattr(prepared, "fingerprint", None) != run.dataset_fingerprint:
        return None
    return prepared


def extras_signature(run: TrainingRun, extras: record_pdf.ReportExtras) -> tuple[Any, ...]:
    """What the optional sections of a record hold, to tell when a built PDF no longer covers everything."""
    cv = extras.cross_validation
    assay = extras.assay
    sweep = extras.sweep
    return (
        run.run_id,
        None if cv is None else (len(cv), float(cv.attrs.get("seconds", 0.0) or 0.0)),
        tuple(sorted(extras.permutations)),
        None if assay is None else (assay.source_name, assay.rows, assay.channel),
        None if sweep is None else (sweep.flows, sweep.channel),
    )


def current_extras(run: TrainingRun) -> record_pdf.ReportExtras:
    """The optional sections the record of ``run`` would include now."""
    return record_pdf.ReportExtras.from_run(run, assay=last_assay(), sweep=sweep_session())


# --------------------------------------------------------------------------------------------------------------
# Notices and the build task
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


def _build_work(run: TrainingRun, summary: dict[str, Any] | None, settings: dict[str, Any],
                extras: record_pdf.ReportExtras) -> Any:
    """The work of a build task: take the run's readings (kept on the run) and lay out the record."""
    def work(progress: Any, cancel: Any) -> record_pdf.RenderedReport:
        progress("Taking the readings on the held-out rows", 0.0)
        evals = evaluate.evaluate_run(run)
        return record_pdf.render_report(run, evals, prepared_summary=summary, settings=settings, extras=extras,
                                        progress=progress, cancel=cancel)

    return work


def _run_reference(run: TrainingRun) -> Any:
    """A weak reference to ``run`` (None for an object that cannot be referenced weakly)."""
    try:
        return weakref.ref(run)
    except TypeError:
        return None


def built_record(run: TrainingRun) -> dict[str, Any] | None:
    """The PDF built in this session for ``run`` itself, or None.

    The run OBJECT is compared, not only its id: a fit and its copy loaded from disk share an id, but the copy may
    lack CH3 or its held-out rows, so a record of one is never offered as the record of the other (as 06 Sweep
    treats its stream).
    """
    built = st.session_state.get(PDF_RESULT)
    if not isinstance(built, dict) or built.get("run_id") != run.run_id:
        return None
    reference = built.get("run_ref")
    if reference is not None:
        return built if reference() is run else None
    return built if built.get("origin") == getattr(run, "origin", "fitted") else None


def _store(task: EvaluationTask) -> None:
    """Keep a finished build's PDF in the session and leave a one-off message about it."""
    snap = task.snapshot()
    if snap.state == "done" and isinstance(task.result, record_pdf.RenderedReport):
        report = task.result
        st.session_state[PDF_RESULT] = {
            "run_id": task.run_id, "data": report.data, "pages": report.pages, "images": report.images,
            "seconds": snap.elapsed, "sections": report.sections, "problems": report.problems,
            "signature": task.extra.get("signature"), "run_ref": task.extra.get("run_ref"),
            "origin": task.extra.get("origin"),
        }
        state.mark_done("record")
        _set_notice("success", f"PDF record built in {snap.elapsed:,.1f} s: {report.pages} pages, "
                               f"{report.size_bytes / 2**20:,.2f} MB.")
    elif snap.state == "cancelled":
        _set_notice("info", f"The PDF record build was cancelled after {snap.elapsed:,.1f} s.")
    else:
        _set_notice("error", f"The PDF record could not be built: {state.first_line(snap.error) or 'no reason'}")


def _start_build() -> None:
    """Build PDF record button callback: gather the inputs here, then build in the background (or inline)."""
    run = state.current_run()
    if run is None or not run.ok_channels() or evaluate.get_task(st.session_state.get(PDF_TASK)) is not None:
        return
    prepared = prepared_for(run)
    summary = record_pdf.summarise_prepared(prepared) if prepared is not None else None
    extras = current_extras(run)
    task = EvaluationTask("record", run.run_id, _build_work(run, summary, asdict(state.settings()), extras),
                          label="PDF record", exclusive=True, holder=HOLDER)
    task.extra["signature"] = extras_signature(run, extras)
    task.extra["run_ref"] = _run_reference(run)
    task.extra["origin"] = getattr(run, "origin", "fitted")
    try:
        if sync_training_requested():
            task.run_inline()
            _store(task)
            return
        task.start()
    except JobBusyError as exc:
        _set_notice("warning", str(exc))
        return
    st.session_state[PDF_TASK] = task.task_id
    st.session_state.pop(RERUN_FOR, None)


def _collect() -> EvaluationTask | None:
    """The session's build while it runs; a finished one is stored, reported and forgotten."""
    task = evaluate.get_task(st.session_state.get(PDF_TASK))
    if task is None:
        st.session_state.pop(PDF_TASK, None)
        return None
    if task.finished:
        st.session_state.pop(PDF_TASK, None)
        _store(task)
        evaluate.forget_task(task.task_id)
        return None
    return task


def _rerun_once(task_id: str) -> None:
    """Rerun the whole app once for ``task_id`` (a second request for the same task does nothing)."""
    if st.session_state.get(RERUN_FOR) != task_id:
        st.session_state[RERUN_FOR] = task_id
        st.rerun(scope="app")


@st.fragment(run_every=1.0)
def build_panel(task_id: str) -> None:
    """Live view of a running build, refreshed every second: bar, elapsed time and Cancel."""
    task = evaluate.get_task(task_id)
    if task is None or st.session_state.get(PDF_TASK) != task_id:
        _rerun_once(task_id)
        return
    snap = task.snapshot()
    with st.container(border=True, key="rec_build_box"):
        st.markdown("**Building the PDF record**")
        st.progress(float(min(max(snap.fraction, 0.0), 1.0)),
                    text=f"{snap.message} · elapsed {format_elapsed(snap.elapsed)}")
        st.button("Cancel", key="rec_cancel", disabled=task.finished, on_click=task.cancel)
    if task.finished:
        _rerun_once(task_id)


# --------------------------------------------------------------------------------------------------------------
# Sections
# --------------------------------------------------------------------------------------------------------------
def _run_chips(run: TrainingRun) -> list[str]:
    """Short facts about the run the record describes."""
    chips = [f"run {run.run_id}",
             "binary: normal vs attack" if run.request.mode == "binary"
             else f"multi-class: {len(run.data.classes)} classes",
             f"{len(run.ok_channels())} channel{'s' if len(run.ok_channels()) != 1 else ''}",
             f"test {len(run.data.y_test):,} rows"]
    if getattr(run, "origin", "fitted") == "loaded":
        chips.append("loaded from disk")
    return chips


def _no_held_out_rows(run: TrainingRun) -> None:
    """A run loaded without its held-out rows: what the record can still hold, and the way to the rest."""
    components.needs(
        f"Run {run.run_id} was loaded from disk without its held-out rows. Its PDF record can only hold the readings "
        "recorded when it was fitted (no confusion matrices, curves or timing charts), and the readings exports "
        "wait until the rows are back. Load it again at the Logbook with its data available to rebuild them.",
        "logbook")
    if run.data_request.source != "synthetic":
        bench = PAGE_OBJECTS.get("bench")
        if bench is not None:
            st.page_link(bench, label=f"Go to {BY_KEY['bench'].label} to set the data folder",
                         icon=":material/arrow_forward:")


def _contents_note(run: TrainingRun, extras: record_pdf.ReportExtras) -> None:
    """What the record will hold, and how to add the optional sections that are missing."""
    st.caption("The record holds: a cover with the run's identity and contents, the sample sheet, the fit "
               "settings, the readings (leaderboard, a recorded-traffic estimate table when the run allows it, and "
               "the per-class table), every channel's confusion matrix, ROC "
               "and precision-recall curves, feature importance, timing, notes and limitations, and the dataset "
               "citation. Charts are drawn in the light palette whatever the app's theme.")
    optional = [
        ("Cross-validation", extras.cross_validation is not None, "run it at 03 Measure"),
        ("Permutation importance", bool(extras.permutations), "measure it at 03 Measure"),
        ("Assay", extras.assay is not None, "score a file with this run at 05 Assay"),
        ("Sweep", extras.sweep is not None, "stream flows with this run at 06 Sweep"),
    ]
    parts = [f"{name}: {'included' if present else 'not yet; ' + how}" for name, present, how in optional]
    st.caption("Optional sections. " + " · ".join(parts) + ".")
    if prepared_for(run) is None:
        st.caption("01 Sample does not hold the sample this run was fitted on, so the sample sheet will use the "
                   "figures the run itself recorded (no per-file table).")


def _pdf_section(run: TrainingRun, running: EvaluationTask | None) -> None:
    """Build button, progress while building, then the download and facts about the file."""
    st.subheader("PDF record", anchor=False)
    extras = current_extras(run)
    _contents_note(run, extras)
    if running is not None:
        build_panel(running.task_id)
    else:
        holder = slot_holder()
        again = built_record(run) is not None
        st.button("Build PDF record again" if again else "Build PDF record", key="rec_build", type="primary",
                  on_click=_start_build, disabled=holder is not None,
                  help="Lays out the record of this run. Nothing is fitted; it takes a few seconds per chart.")
        if holder is not None:
            st.caption(f"{holder[:1].upper()}{holder[1:]} is running in this app; only one fit, measurement or "
                       "record build runs at a time, so this button waits until it ends.")
    built = built_record(run)
    if built is None:
        other = st.session_state.get(PDF_RESULT)
        if isinstance(other, dict) and other.get("run_id") == run.run_id:
            st.caption("The PDF built earlier in this session describes another copy of this run (the fit, or a "
                       "copy loaded from disk), so it is not offered here; build the record of this copy.")
        return
    data: bytes = built["data"]
    components.reading_cards([
        {"Reading": "Pages", "Value": int(built["pages"]), "Note": "A4, light palette"},
        {"Reading": "Size", "Value": f"{len(data) / 2**20:,.2f} MB", "Note": f"{int(built['images'])} charts"},
        {"Reading": "Built in", "Value": f"{float(built['seconds']):,.1f} s", "Note": "on this machine"},
    ])
    if built.get("signature") != extras_signature(run, extras):
        st.caption("New results have arrived since this PDF was built (cross-validation, permutation importance, "
                   "an Assay or a Sweep); build it again to include them.")
    for problem in built.get("problems") or ():
        st.warning(problem)
    st.download_button("Download PDF record", data=data, file_name=record_pdf.report_file_name(run.run_id),
                       mime="application/pdf", key="rec_dl_pdf", on_click="ignore", type="primary",
                       icon=":material/download:")


def _csv_section(run: TrainingRun | None) -> None:
    """One download per available table (built when clicked), the reason for each missing one, and the ZIP."""
    st.subheader("CSV exports", anchor=False)
    st.caption("UTF-8 with a byte-order mark, so spreadsheet programs read them correctly. No file holds flow "
               "feature values. Each file is built when its button is pressed.")
    # Readings are taken only in the app's work slot (03 Measure, or the PDF build above); a download never starts
    # that work itself, so the tables that need them wait until they exist.
    evals = evaluate.cached_evaluations(run) if run is not None else None
    prepared = prepared_for(run) if run is not None else None
    items = exports.export_items(run, evaluations=evals, prepared=prepared, assay=last_assay(),
                                 sweep=sweep_session(), measure_if_needed=False)
    for item in items:
        left, right = st.columns([3, 2], gap="medium", vertical_alignment="center")
        with left:
            st.markdown(f"**{html.escape(item.title)}**")
            st.caption(item.description)
        with right:
            if item.build is not None:
                st.download_button(f"{item.title} (CSV)", data=item.build, file_name=item.file_name,
                                   mime="text/csv", key=f"rec_dl_{item.key}", on_click="ignore", width="stretch")
            else:
                st.caption(item.missing)
    available = [item for item in items if item.available]
    if not available:
        return
    built = built_record(run) if run is not None else None
    extra: dict[str, tuple[bytes, str]] = {}
    if run is not None and built is not None:
        extra[record_pdf.report_file_name(run.run_id)] = (built["data"], "The PDF measurement record of the run.")
    stem = f"nids-record-{run.run_id}" if run is not None else "nids-exports"
    names = ", ".join(item.title.lower() for item in available)
    st.download_button("Download all as ZIP", data=lambda: exports.bundle_zip(run, items=available, extra_files=extra),
                       file_name=f"{stem}.zip", mime="application/zip", key="rec_dl_zip", on_click="ignore",
                       icon=":material/folder_zip:",
                       help=f"Packs {names}{' and the PDF record' if extra else ''}, with a README describing "
                            "each file.")


def render() -> None:
    """Draw the 07 Record station."""
    running = _collect()  # a finished build is stored first (the shell redraws the stepper with its tick afterwards)
    components.station_header("record")
    _show_notice()
    run = state.current_run()
    if run is None or not run.ok_channels():
        components.needs("Needs a fitted channel: fit at least one at 02 Fit to build the PDF record and export its "
                         "readings. The run history below can be exported already.", "fit")
        _csv_section(None)
        return
    components.chips(_run_chips(run))
    if not evaluate.has_test_rows(run):
        _no_held_out_rows(run)
    _pdf_section(run, running)
    _csv_section(run)


__all__ = ["build_panel", "built_record", "current_extras", "extras_signature", "last_assay", "prepared_for",
           "render", "sweep_session"]
