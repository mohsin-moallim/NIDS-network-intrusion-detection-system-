"""Session and process state for the UI.

Pages never compute expensive results themselves: they read what the explicit actions (Prepare, Fit, Score…) stored
here. Finished fit runs are also kept in a process-wide registry so a browser refresh does not lose them.

A fit started at 02 Fit runs as a background job; the session only remembers its id (:data:`JOB_ID`). Whichever
page renders first after the job ends adopts the result (:func:`collect_finished_job`, called by the stepper), so a
fit that finishes while the viewer is on another station is not lost.

Once a finished job's run is stored here (in the registry, the session, or both), the job lets go of it
(:meth:`~graticule.models.jobs.TrainingJob.release_result`): the few finished jobs the process remembers then hold
no models or matrices, and the registry's capacity really bounds how many runs stay in memory.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import streamlit as st

from graticule.settings import AppSettings, load_settings, resolve_data_dir, save_settings

if TYPE_CHECKING:
    from graticule.data.prepare import PreparedDataset
    from graticule.models.jobs import TrainingJob
    from graticule.models.train import TrainingRun
    from graticule.scoring import ScoredBatch

SETTINGS = "g_settings"
DONE = "g_done"
LAST_RUN_ID = "g_last_run_id"
PREPARED = "g_prepared"
JOB_ID = "g_job_id"
# The session's own reference to its current run: keeps it readable even if the bounded registry evicts it, and
# wins over the registry when another session stores a run under the same id (a loaded copy of a fit).
RUN = "g_run"
# One message about the end of a fit job (kind, text), shown once by 02 Fit.
FIT_NOTICE = "g_fit_notice"
# Run ids this session has written to the run history (each finished fit is recorded once), and the last failure.
HISTORY_RECORDED = "g_history_recorded"
HISTORY_ERROR = "g_history_error"
# Run id -> BundleLoadResult for the runs this session loaded from disk at the Logbook.
LOADED = "g_loaded_bundles"
# The last file scored at 05 Assay in this session (a graticule.scoring.ScoredBatch), for 07 Record's exports.
LAST_ASSAY = "g_last_assay"

NoticeKind = Literal["success", "info", "warning", "error"]
JobOutcome = Literal["stored", "cancelled", "failed", "lost"]


def settings() -> AppSettings:
    """Return this session's settings, loading them from disk the first time."""
    if SETTINGS not in st.session_state:
        st.session_state[SETTINGS] = load_settings()
    return st.session_state[SETTINGS]


def update_settings(new: AppSettings) -> AppSettings:
    """Validate, store and persist new settings; returns the validated copy."""
    clean = new.validated()
    save_settings(clean)
    st.session_state[SETTINGS] = clean
    return clean


def get_prepared() -> "PreparedDataset | None":
    """The dataset drawn at 01 Sample in this session, if any.

    Later stations read it from here and should compare its ``fingerprint`` with the one their own results were
    made from, to notice that a new sample has been drawn since.
    """
    value = st.session_state.get(PREPARED)
    # Duck-typed on purpose: after a code reload the stored object's class is an older copy of PreparedDataset.
    return value if value is not None and hasattr(value, "frame") and hasattr(value, "sampling") else None


def set_prepared(dataset: "PreparedDataset") -> None:
    """Store the dataset drawn at 01 Sample for the other stations of this session."""
    st.session_state[PREPARED] = dataset


def set_last_assay(batch: "ScoredBatch") -> None:
    """Keep the file just scored at 05 Assay as this session's last assay (it replaces the previous one)."""
    st.session_state[LAST_ASSAY] = batch


def get_last_assay() -> "ScoredBatch | None":
    """The last file scored at 05 Assay in this session (with the run id it was scored by), or ``None``.

    07 Record reads it for its exports: :meth:`graticule.scoring.ScoredBatch.to_csv_bytes` and ``file_name`` give
    the full scored CSV; ``run_id`` says which run scored it (it may be an earlier run than the current one), and
    :func:`graticule.report.exports.belongs_to_run` tells whether the current run object scored it (a fit and its
    copy loaded from disk share an id but not their results).
    """
    value = st.session_state.get(LAST_ASSAY)
    # Duck-typed on purpose: after a code reload the stored object's class is an older copy of ScoredBatch.
    return value if value is not None and hasattr(value, "frame") and hasattr(value, "unseen_labels") else None


def mark_done(station_key: str) -> None:
    """Record that a station produced a result in this session (drives the ticks on the stepper)."""
    done: set[str] = st.session_state.setdefault(DONE, set())
    done.add(station_key)


def is_done(station_key: str) -> bool:
    """True when ``station_key`` has produced a result in this session."""
    return station_key in st.session_state.get(DONE, set())


def done_stations() -> frozenset[str]:
    """The stations that have produced a result in this session (a copy, to compare before and after a draw)."""
    return frozenset(st.session_state.get(DONE, set()))


class RunRegistry:
    """Process-wide store of finished fit runs, keyed by run id, shared by every browser tab of this app.

    It also remembers the ids of background fit jobs still running (:meth:`watch`). Their results are taken in by
    :meth:`harvest` whichever session renders next, so a fit is not lost when the tab that started it was refreshed
    or closed: the next 02 Fit visit offers it for restoring.
    """

    def __init__(self, capacity: int = 3) -> None:
        self._lock = threading.Lock()
        self._runs: dict[str, Any] = {}
        self._capacity = capacity
        self._watched: list[str] = []

    def watch(self, job_id: str) -> None:
        """Remember a started background job until it finishes."""
        with self._lock:
            if job_id not in self._watched:
                self._watched.append(job_id)

    def watched(self) -> list[str]:
        """Ids of the watched jobs not yet harvested (finished or not)."""
        with self._lock:
            return list(self._watched)

    def harvest(self) -> int:
        """Store the runs of watched jobs that have finished with at least one fitted channel; returns how many.

        A stored run is released by its job (the job keeps its ``run_id``, so the session that started it still
        finds the run here). A run with no fitted channel stays with its job, whose session explains the failure.
        Jobs the process no longer knows are forgotten. Cheap when nothing is watched.
        """
        if not self._watched:
            return 0
        from graticule.models.jobs import get_job

        stored = 0
        for job_id in self.watched():
            job = get_job(job_id)
            if job is not None and not job.finished:
                continue
            with self._lock:
                if job_id in self._watched:
                    self._watched.remove(job_id)
            run = job.result if job is not None else None
            if run is not None and run.ok_channels():
                if self.get(run.run_id) is None:
                    self.put(run.run_id, run)
                    stored += 1
                job.release_result()  # type: ignore[union-attr]
        return stored

    def put(self, run_id: str, run: Any) -> None:
        """Keep ``run``; the oldest runs are dropped beyond the capacity to bound memory."""
        with self._lock:
            self._runs[run_id] = run
            while len(self._runs) > self._capacity:
                self._runs.pop(next(iter(self._runs)))

    def get(self, run_id: str | None) -> Any | None:
        """Return the run stored under ``run_id``, or ``None``."""
        if run_id is None:
            return None
        with self._lock:
            return self._runs.get(run_id)

    def latest_id(self) -> str | None:
        """Id of the most recently stored run, or ``None`` when empty."""
        with self._lock:
            return next(reversed(self._runs), None) if self._runs else None


REGISTRY_VERSION = 3


@st.cache_resource(show_spinner=False)
def run_registry(version: int = REGISTRY_VERSION) -> RunRegistry:
    """The single process-wide :class:`RunRegistry`.

    ``version`` is part of the cache key: bump :data:`REGISTRY_VERSION` whenever :class:`RunRegistry` changes, so a
    running server that reloads this module never hands out an instance of the old class.
    """
    return RunRegistry()


# --------------------------------------------------------------------------------------------------------------
# Fit runs and the background job
# --------------------------------------------------------------------------------------------------------------
def store_run(run: "TrainingRun") -> None:
    """Make ``run`` this session's current run: keep it in the process registry and the session, tick 02 Fit."""
    run_registry().put(run.run_id, run)
    st.session_state[LAST_RUN_ID] = run.run_id
    st.session_state[RUN] = run
    mark_done("fit")


def current_run() -> "TrainingRun | None":
    """This session's current fit run (the one 03 Measure onwards read), or ``None`` before the first fit.

    A background fit that has finished since the last rerun is adopted first. Nothing is ever refitted here.

    The session's own reference wins over the process registry: a fitted run and the same run loaded from disk
    share one run id, so when another session loads the saved copy (which replaces the registry entry), this
    session keeps reading the run it chose. The registry is the fallback (for example after a restore offer).
    """
    collect_finished_job()
    run_id = st.session_state.get(LAST_RUN_ID)
    if run_id is None:
        return None
    held = st.session_state.get(RUN)
    if held is not None and getattr(held, "run_id", None) == run_id:
        return held
    return run_registry().get(run_id)


def set_fit_notice(kind: NoticeKind, text: str) -> None:
    """Leave one message about a finished, cancelled or failed fit for 02 Fit to show."""
    st.session_state[FIT_NOTICE] = (kind, text)


def pop_fit_notice() -> tuple[NoticeKind, str] | None:
    """Take the pending fit message, if any (it is shown once)."""
    return st.session_state.pop(FIT_NOTICE, None)


def running_job() -> "TrainingJob | None":
    """This session's fit job while it is still running, else ``None``."""
    job_id = st.session_state.get(JOB_ID)
    if job_id is None:
        return None
    from graticule.models.jobs import get_job

    job = get_job(job_id)
    return job if job is not None and not job.finished else None


def other_fit_running() -> bool:
    """True while a background fit started by another session (or by this tab before a refresh) is running."""
    watched = run_registry().watched()
    if not watched:
        return False
    from graticule.models.jobs import get_job

    own = st.session_state.get(JOB_ID)
    for job_id in watched:
        job = get_job(job_id) if job_id != own else None
        if job is not None and not job.finished:
            return True
    return False


def fit_duration_text(run: "TrainingRun") -> str:
    """How long the fit took as the user waited for it, e.g. ``"42.0 s (matrices 20.3 s, channels 21.7 s)"``.

    The matrices part (target, de-duplication, split and any Top-K ranking) is spelled out when it took a second or
    more; the channels part covers fitting and scoring.
    """
    total = float(getattr(run, "total_seconds", run.seconds))
    prep = float(getattr(run, "prep_seconds", 0.0))
    if prep >= 1.0:
        return f"{total:,.1f} s (matrices {prep:,.1f} s, channels {float(run.seconds):,.1f} s)"
    return f"{total:,.1f} s"


def adopt_run(run: "TrainingRun", *, cancelled: bool = False) -> JobOutcome:
    """Store a finished run as this session's current run when at least one channel was fitted.

    A cancelled run keeps the channels that finished before the cancel. A cancel that came after every channel had
    finished stopped nothing, so such a run is reported as finished. When no channel was fitted nothing is stored
    (the previous run, if any, stays current) and the reason is left as the fit message.
    """
    fitted = list(run.ok_channels())
    missing = [k for k in run.channels if k not in fitted]
    if fitted:
        store_run(run)
        record_history(run)
        if cancelled and missing and bool(getattr(run, "cancelled", True)):
            names = ", ".join(_channel_label(k) for k in missing)
            set_fit_notice("info", f"Fit cancelled after {len(fitted)} of {len(run.channels)} channels. The fitted "
                                   f"ones are kept as run {run.run_id}; not fitted: {names}.")
        else:
            set_fit_notice("success", f"Fit finished in {fit_duration_text(run)}: {len(fitted)} of "
                                      f"{len(run.channels)} channels fitted (run {run.run_id}).")
        return "stored"
    if cancelled:
        set_fit_notice("info", "Fit cancelled before any channel was fitted. Nothing was stored; the previous "
                               "readings, if any, are unchanged.")
        return "cancelled"
    problems = [f"{_channel_label(key)}: {first_line(result.error) or result.status}"
                for key, result in run.channels.items()]
    set_fit_notice("error", "No channel could be fitted, so nothing was stored. " + "; ".join(problems) + ".")
    return "failed"


def adopt_job(job: "TrainingJob") -> JobOutcome:
    """Turn a finished job into this session's current run, or into a message explaining why there is none.

    The job's run is taken from the job, or from the process registry when :meth:`RunRegistry.harvest` already
    moved it there; either way the job lets go of it afterwards.
    """
    from graticule.data.sampling import SingleClassError

    st.session_state.pop(JOB_ID, None)
    snapshot = job.snapshot()
    cancelled = snapshot.state == "cancelled"
    run = job.result
    run_id = getattr(job, "run_id", None)
    if run is None and run_id is not None:
        run = run_registry().get(run_id)
        if run is None:
            set_fit_notice("warning", f"The fit finished as run {run_id}, but the app no longer holds that run (it "
                                      "keeps only the latest few). Press Fit to fit again.")
            return "lost"
    if run is not None:
        outcome = adopt_run(run, cancelled=cancelled or bool(getattr(run, "cancelled", False)))
        release = getattr(job, "release_result", None)
        if callable(release):
            release()
        return outcome
    if cancelled:
        set_fit_notice("info", "Fit cancelled before any channel was fitted. Nothing was stored; the previous "
                               "readings, if any, are unchanged.")
        return "cancelled"
    if isinstance(getattr(job, "exception", None), SingleClassError):
        # The friendly single-class message, raised inside the job (for example after de-duplication).
        set_fit_notice("warning", f"Nothing was fitted. {job.exception}")
    else:
        set_fit_notice("error", f"The fit failed: {first_line(job.error or snapshot.error) or 'no reason recorded'}")
    return "failed"


def _channel_label(key: str) -> str:
    """Badge and name of a channel key, e.g. ``"CH2 XGBoost"`` (the key itself when unknown)."""
    from graticule.theme import CHANNEL_BY_KEY

    style = CHANNEL_BY_KEY.get(key)
    return style.label if style is not None else key


def first_line(text: str | None) -> str:
    """The first non-empty line of an error text (the exception type and message, without the traceback)."""
    lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
    return lines[0] if lines else ""


def collect_finished_job() -> JobOutcome | None:
    """Adopt this session's background fit once it has finished; ``None`` while nothing has finished.

    Cheap when no job is pending (one dictionary lookup), so every page can call it before drawing. Finished jobs
    started by other sessions (or by this tab before a refresh) are moved into the process registry as well.
    """
    run_registry().harvest()
    job_id = st.session_state.get(JOB_ID)
    if job_id is None:
        return None
    from graticule.models.jobs import get_job

    job = get_job(job_id)
    if job is None:
        st.session_state.pop(JOB_ID, None)
        set_fit_notice("warning", "The fit started in this session is no longer known to the app (the server may "
                                  "have restarted). Press Fit to start it again.")
        return "lost"
    if not job.finished:
        return None
    return adopt_job(job)


# --------------------------------------------------------------------------------------------------------------
# Run history and saved channel sets (Logbook)
# --------------------------------------------------------------------------------------------------------------
def record_history(run: "TrainingRun") -> None:
    """Write a finished fit to the run history, once per run id; a failure is kept for the Logbook, never raised.

    Only fits are recorded (a run loaded from disk is not a new fit). The history call is idempotent per run id as
    well, so a second session adopting the same run cannot add a second line.
    """
    recorded: set[str] = st.session_state.setdefault(HISTORY_RECORDED, set())
    if run.run_id in recorded or getattr(run, "origin", "fitted") != "fitted":
        return
    from graticule.history import RunHistory

    saved = getattr(run, "bundle_path", None)
    try:
        RunHistory().record(run, saved_path=Path(saved) if saved else None)
    except Exception as exc:  # noqa: BLE001 - the history must never break a fit
        st.session_state[HISTORY_ERROR] = f"The run history could not be written ({type(exc).__name__}: {exc})."
        return
    recorded.add(run.run_id)
    st.session_state.pop(HISTORY_ERROR, None)


def history_error() -> str | None:
    """The last failure to write the run history in this session, if any."""
    return st.session_state.get(HISTORY_ERROR)


def bundle_on_disk(run: "TrainingRun") -> Path | None:
    """The folder ``run`` was saved to or loaded from, when it still exists on disk."""
    saved = getattr(run, "bundle_path", None)
    if not saved:
        return None
    folder = Path(saved)
    return folder if folder.is_dir() else None


def unsaved_channel_note(run: "TrainingRun") -> str:
    """Why some fitted channels of ``run`` are left out of a saved set (CH3), or "" when none are."""
    from graticule.persist import UNSAVED_CHANNELS

    left_out = [key for key in run.ok_channels() if key in UNSAVED_CHANNELS]
    return " ".join(UNSAVED_CHANNELS[key] for key in left_out)


def save_current_run() -> tuple[NoticeKind, str]:
    """Save this session's current run as a bundle and note the folder in the run history; returns a message.

    Any failure (no channel to keep, a folder of that name already there and damaged, a full disk, a model that
    cannot be written) comes back as an error message; nothing is raised into the page.
    """
    run = current_run()
    if run is None:
        return "warning", "There is no fitted run to save. Fit channels at 02 Fit first."
    from graticule import persist
    from graticule.history import RunHistory

    try:
        path = persist.save_run(run)
    except Exception as exc:  # noqa: BLE001 - save_run cleans up its temporary folder; the page only reports
        return "error", f"Run {run.run_id} could not be saved: {first_line(str(exc)) or type(exc).__name__}"
    try:
        if getattr(run, "origin", "fitted") == "fitted":
            RunHistory().record(run, saved_path=path)
        else:
            RunHistory().mark_saved(run.run_id, path)
    except Exception as exc:  # noqa: BLE001 - the bundle is saved; the history line is secondary
        st.session_state[HISTORY_ERROR] = f"The run history could not be updated ({type(exc).__name__}: {exc})."
    note = unsaved_channel_note(run)
    return "success", f"Saved run {run.run_id} to {path}." + (f" {note}" if note else "")


@dataclass(frozen=True)
class BundleLoadResult:
    """What happened when a saved channel set was loaded at the Logbook.

    ``kind`` is the tone of ``message`` (success when verified, warning when not verified because the libraries
    changed, error when refused or when verification failed). ``verification`` is the verification sentence and
    ``rows_note`` says whether the held-out rows were rebuilt (and why not, if they were not).
    """

    kind: NoticeKind
    message: str
    run_id: str | None = None
    path: str | None = None
    verified: bool = False
    rebuilt: bool = False
    verification: str = ""
    rows_note: str = ""


def loaded_info(run_id: str | None) -> BundleLoadResult | None:
    """How run ``run_id`` was loaded from disk in this session, or None when it was not loaded here."""
    if run_id is None:
        return None
    return st.session_state.get(LOADED, {}).get(run_id)


ProgressFn = Callable[[str, float], None]


def _sample_note(previous: "PreparedDataset | None", rebuilt: "PreparedDataset") -> str:
    """What happened to the session's 01 Sample when a loaded run's sample was rebuilt."""
    now = rebuilt.fingerprint[:12]
    if previous is None:
        return f" 01 Sample now holds this run's sample (fingerprint {now})."
    if previous.fingerprint == rebuilt.fingerprint:
        return f" 01 Sample already held this sample (fingerprint {now}); it was used as it is."
    return (f" 01 Sample now holds this run's sample (fingerprint {now}); the sample drawn there before "
            f"(fingerprint {previous.fingerprint[:12]}) was replaced.")


def load_bundle_into_session(path: Path | str, progress: ProgressFn | None = None) -> BundleLoadResult:
    """Load a saved channel set and make it this session's current run; returns what happened.

    The bundle is checked and verified (:func:`graticule.persist.load_bundle`); a refused bundle changes nothing.
    Its held-out rows are then rebuilt when possible: from the session's 01 Sample when it is the very sample the
    run was fitted on, else a synthetic run is regenerated from its seed and a CIC-IDS2017 run is read from the
    Bench's data folder (or the folder recorded with the run) when the files are there. When they are rebuilt, the
    rebuilt sample also becomes the session's 01 Sample, so every station agrees on it, and the message says so
    (and names the sample it replaced). The restored run (``origin="loaded"``) is stored like a fit; nothing is
    refitted. ``progress`` receives (message, fraction done) at every stage: checking the files, reading and
    cleaning the data files, building the matrices, and each channel reading the held-out rows.
    """
    from graticule import persist

    def tell(message: str, fraction: float) -> None:
        if progress is not None:
            progress(message, min(max(float(fraction), 0.0), 1.0))

    def within(start: float, span: float) -> ProgressFn:
        return lambda message, fraction: tell(message, start + span * float(fraction))

    folder = Path(path)
    tell("Checking the files and re-reading the probe flows", 0.0)
    try:
        bundle = persist.load_bundle(folder)
    except persist.BundleIntegrityError as exc:
        return BundleLoadResult("error", f"Refused: {exc}", path=str(folder))
    except Exception as exc:  # noqa: BLE001 - any unreadable bundle is reported, never raised into the page
        return BundleLoadResult("error", f"The saved channel set {folder.name} could not be loaded: "
                                         f"{first_line(str(exc)) or type(exc).__name__}", path=str(folder))
    previous = get_prepared()
    prepared = None
    data = None
    cache_note = ""
    try:
        if previous is not None and previous.fingerprint == bundle.manifest.get("dataset_fingerprint"):
            prepared = previous
        elif bundle.source == "synthetic":
            prepared = persist.rebuild_prepared(bundle, data_dir=None, progress=within(0.08, 0.67))
        else:
            from ui import data_cache

            resolution = resolve_data_dir(settings())
            prepared = persist.rebuild_prepared(
                bundle, data_dir=str(resolution.path) if resolution.path is not None else None,
                read_file=data_cache.file_reader(), stage_file=data_cache.file_stager(),
                progress=within(0.08, 0.67))
            cache_note = (" The data files read stay in the app's memory for later draws; Release cached files at "
                          "01 Sample frees them.")
        tell("Building the training and test matrices", 0.78)
        data = persist.rebuild_training_data(bundle, data_dir=None, prepared=prepared)
        rows_note = (f"Held-out rows rebuilt: {len(data.y_test):,} test rows, identical to those the run was "
                     "measured on (same rows, labels and feature values)." + _sample_note(previous, prepared)
                     + cache_note)
    except Exception as exc:  # noqa: BLE001 - RebuildError above all; the run loads without its rows either way
        prepared = None
        data = None
        if bundle.source == "synthetic":
            hint = "The synthetic sample could not be regenerated identically."
        elif "differ" in str(exc):
            hint = "The data files, or the way the program reads them, have changed since the run was saved."
        else:
            hint = "Set the data folder on the Bench, then load the set again."
        rows_note = (f"Held-out rows not rebuilt: {exc} {hint} Scoring single flows and uploaded files still "
                     "works; readings that need the held-out rows wait until then. 01 Sample is unchanged.")
    run = persist.restore_run(bundle, data, progress=within(0.85, 0.15))
    tell("Done", 1.0)
    store_run(run)
    if prepared is not None:
        set_prepared(prepared)
        mark_done("sample")
    left_out = " ".join(persist.UNSAVED_CHANNELS.get(key, "") for key, result in run.channels.items()
                        if result.status == persist.NOT_SAVED).strip()
    if left_out:
        rows_note += f" {left_out}"
    report = bundle.verification
    if report.verified:
        kind: NoticeKind = "success"
    elif report.version_mismatches:
        kind = "warning"
    else:
        kind = "error"
    message = f"Loaded run {run.run_id} from disk. {report.message} {rows_note}"
    result = BundleLoadResult(kind, message, run_id=run.run_id, path=str(folder), verified=report.verified,
                              rebuilt=data is not None, verification=report.message, rows_note=rows_note)
    st.session_state.setdefault(LOADED, {})[run.run_id] = result
    return result