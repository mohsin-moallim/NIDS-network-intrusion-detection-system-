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
from typing import TYPE_CHECKING, Any, Literal

import streamlit as st

from graticule.settings import AppSettings, load_settings, save_settings

if TYPE_CHECKING:
    from graticule.data.prepare import PreparedDataset
    from graticule.models.jobs import TrainingJob
    from graticule.models.train import TrainingRun

SETTINGS = "g_settings"
DONE = "g_done"
LAST_RUN_ID = "g_last_run_id"
PREPARED = "g_prepared"
JOB_ID = "g_job_id"
# The session's own reference to its current run: keeps it readable even if the bounded registry evicts it.
RUN = "g_run"
# One message about the end of a fit job (kind, text), shown once by 02 Fit.
FIT_NOTICE = "g_fit_notice"

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


def mark_done(station_key: str) -> None:
    """Record that a station produced a result in this session (drives the ticks on the stepper)."""
    done: set[str] = st.session_state.setdefault(DONE, set())
    done.add(station_key)


def is_done(station_key: str) -> bool:
    """True when ``station_key`` has produced a result in this session."""
    return station_key in st.session_state.get(DONE, set())


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
    """
    collect_finished_job()
    run_id = st.session_state.get(LAST_RUN_ID)
    if run_id is None:
        return None
    run = run_registry().get(run_id)
    if run is None:
        held = st.session_state.get(RUN)
        run = held if held is not None and getattr(held, "run_id", None) == run_id else None
    return run


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
