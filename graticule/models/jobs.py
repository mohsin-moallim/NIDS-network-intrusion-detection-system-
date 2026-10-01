"""Background training jobs: one fit at a time per process, with progress, elapsed time and cancellation.

A :class:`TrainingJob` runs the whole 02 Fit procedure (building the training matrices, then fitting each requested
channel) either in a daemon thread (:meth:`TrainingJob.start`) or on the calling thread
(:meth:`TrainingJob.run_inline`, used when ``GRATICULE_SYNC_TRAINING=1``). The worker never imports or calls the web
framework: it writes progress into a lock-protected object, and the UI polls :meth:`TrainingJob.snapshot`, which
returns an immutable copy.

Progress reporting goes through a :class:`ProgressSink`. Channel keys report per-channel progress; the special key
:data:`STAGE_KEY` reports the overall stage (``status`` names it: ``"preparing"``, ``"fitting"`` or
``"finishing"``; ``fraction`` is the share of that stage done; ``message`` is the line to show).

Hooks registry. Some libraries keep the callbacks they are given inside the fitted model (XGBoost does), so a
callback must not hold a lock, a sink or a thread object. Instead the trainer registers its sink and cancel token
here under a plain string id (:func:`register_hooks`) and the callback holds only that id, looking the hooks up on
each call (:func:`find_hooks`).

Warnings per thread. On this Python the warnings machinery (filters, ``showwarning``) is one process-wide state, and
``warnings.catch_warnings`` swaps it for every thread at once: a fit thread recording its warnings that way would
also swallow the warnings of the app's other threads, and two such blocks overlapping on different threads can
leave the process with a dead recorder. :func:`thread_warnings` never touches that state. One router, installed
once, sits in front of the hook that delivers every warning that passed the filters (``warnings._showwarnmsg``, which
``catch_warnings`` never replaces); it hands a warning to the innermost recorder of the thread that raised it, and
passes every other warning on unchanged. Because the filters stay process-wide, a warning another thread silences
at that moment (or a repeat that the default filter shows only once) is not recorded, so facts that matter, such as
a model stopping at its iteration limit, are read from the fitted model rather than from warnings. (scikit-learn's
own parallel helper still opens ``catch_warnings`` blocks on its worker threads, for example while a forest grows;
that is outside this module's reach and goes away with Python's context-aware warnings.)
"""

from __future__ import annotations

import os
import threading
import time
import traceback
import uuid
import warnings
from collections import OrderedDict
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Protocol

if TYPE_CHECKING:
    from graticule.data.prepare import PreparedDataset
    from graticule.models.train import TrainingRun, TrainRequest

STAGE_KEY = "stage"
SYNC_ENV = "GRATICULE_SYNC_TRAINING"
JobState = Literal["queued", "running", "done", "failed", "cancelled"]
#: Rough relative cost of each channel, used only to weight the overall progress bar.
CHANNEL_COST: dict[str, float] = {"forest": 3.0, "xgboost": 3.0, "svm": 3.0, "mlp": 2.0, "logreg": 1.0}
_KEEP_FINISHED_JOBS = 6


class JobBusyError(RuntimeError):
    """Raised when a training job is started while another one is still running in this process."""


class TrainingCancelled(RuntimeError):
    """Raised inside the trainer when the cancel token fires; the job then ends in state ``"cancelled"``."""


class CancelToken:
    """A thread-safe flag a running fit checks between channels, forest chunks and boosting rounds."""

    def __init__(self) -> None:
        self._event = threading.Event()

    def cancel(self) -> None:
        """Ask the fit to stop at its next check."""
        self._event.set()

    @property
    def cancelled(self) -> bool:
        """True once :meth:`cancel` has been called."""
        return self._event.is_set()


class ProgressSink(Protocol):
    """Anything that accepts progress updates for a channel key (or :data:`STAGE_KEY`)."""

    def update(self, key: str, *, status: str | None = None, fraction: float | None = None,
               message: str | None = None) -> None:
        """Record new progress for ``key``; arguments left as None keep their previous value."""
        ...


# --------------------------------------------------------------------------------------------------------------
# Hooks registry (for callbacks that must hold only plain values)
# --------------------------------------------------------------------------------------------------------------
_HOOKS: dict[str, tuple[ProgressSink | None, CancelToken | None]] = {}
_HOOKS_LOCK = threading.Lock()


def register_hooks(progress: ProgressSink | None, cancel: CancelToken | None, *, job_id: str | None = None) -> str:
    """Store a progress sink and cancel token under a fresh string id and return the id."""
    hook_id = f"{job_id or 'direct'}:{uuid.uuid4().hex[:12]}"
    with _HOOKS_LOCK:
        _HOOKS[hook_id] = (progress, cancel)
    return hook_id


def release_hooks(hook_id: str) -> None:
    """Forget the hooks stored under ``hook_id`` (unknown ids are ignored)."""
    with _HOOKS_LOCK:
        _HOOKS.pop(hook_id, None)


def find_hooks(hook_id: str) -> tuple[ProgressSink | None, CancelToken | None]:
    """The (progress sink, cancel token) stored under ``hook_id``; (None, None) once released."""
    with _HOOKS_LOCK:
        return _HOOKS.get(hook_id, (None, None))


# --------------------------------------------------------------------------------------------------------------
# Warnings per thread (see the module notes)
# --------------------------------------------------------------------------------------------------------------
_ROUTER_LOCK = threading.Lock()
# The router keeps its per-thread recorder stacks as an attribute of itself, so a reloaded copy of this module finds
# and reuses the router already installed instead of stacking a second one in front of it.
_ROUTER_ATTR = "graticule_thread_recorders"


def _thread_recorders() -> dict[int, list[list[Any]]] | None:
    """The router's recorder stacks per thread id, installing the router on first use (None if unsupported)."""
    with _ROUTER_LOCK:
        current = getattr(warnings, "_showwarnmsg", None)
        if current is None or not callable(current):
            return None
        recorders: dict[int, list[list[Any]]] | None = getattr(current, _ROUTER_ATTR, None)
        if recorders is not None:
            return recorders
        forward = current
        stacks: dict[int, list[list[Any]]] = {}

        def route(message: Any) -> None:
            """Give ``message`` to the raising thread's innermost recorder, or pass it on unchanged."""
            stack = stacks.get(threading.get_ident())
            if stack:
                stack[-1].append(message)
            else:
                forward(message)

        setattr(route, _ROUTER_ATTR, stacks)
        warnings._showwarnmsg = route  # type: ignore[attr-defined]
        return stacks


@contextmanager
def thread_warnings() -> Iterator[list[warnings.WarningMessage]]:
    """Record the warnings raised on the calling thread while the block runs, and only those.

    Yields the list that receives them (``warnings.WarningMessage`` objects). Blocks may nest on one thread; the
    innermost one records. Other threads' warnings are passed on as usual, and no filter or hook of the
    ``warnings`` module is changed, so this is safe on a background thread. Warnings follow the process's filters:
    one that the filters ignore, or a repeat that they show only once, is not recorded. Where the running Python
    offers no hook to route, the list stays empty and warnings are shown as usual.
    """
    caught: list[warnings.WarningMessage] = []
    recorders = _thread_recorders()
    if recorders is None:
        yield caught
        return
    ident = threading.get_ident()
    with _ROUTER_LOCK:
        recorders.setdefault(ident, []).append(caught)
    try:
        yield caught
    finally:
        with _ROUTER_LOCK:
            stack = recorders.get(ident, [])
            for position in range(len(stack) - 1, -1, -1):
                if stack[position] is caught:
                    del stack[position]
                    break
            if not stack:
                recorders.pop(ident, None)


# --------------------------------------------------------------------------------------------------------------
# Snapshots
# --------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class ChannelProgress:
    """Progress of one channel: status (waiting, running, ok, failed, cancelled, skipped), share done, a message
    and the seconds spent on it so far (final fit time once finished)."""

    key: str
    status: str
    fraction: float
    message: str
    seconds: float


@dataclass(frozen=True)
class JobSnapshot:
    """An immutable copy of a job's progress at one moment."""

    job_id: str
    state: JobState
    elapsed: float
    fraction: float
    channels: tuple[ChannelProgress, ...]
    stage: str
    error: str | None

    @property
    def finished(self) -> bool:
        """True in the final states (done, failed, cancelled)."""
        return self.state in ("done", "failed", "cancelled")


_FINAL_CHANNEL = ("ok", "failed", "cancelled", "skipped")


class _Tracker:
    """Lock-protected progress of one job; also the :class:`ProgressSink` the trainer writes to."""

    def __init__(self, channels: tuple[str, ...], topk: bool) -> None:
        self._lock = threading.Lock()
        self.channels = channels
        self.status = {k: "waiting" for k in channels}
        self.fraction = {k: 0.0 for k in channels}
        self.message = {k: "Waiting" for k in channels}
        self.started_at: dict[str, float] = {}
        self.seconds = {k: 0.0 for k in channels}
        self.stage = "Queued"
        self.stage_name = "queued"
        self.prep_share = 0.25 if topk else 0.08
        self.finish_share = 0.02
        self.stage_fraction = {"preparing": 0.0, "fitting": 0.0, "finishing": 0.0}
        self.best = 0.0

    def update(self, key: str, *, status: str | None = None, fraction: float | None = None,
               message: str | None = None) -> None:
        """Record progress for a channel or for the overall stage (see the module notes)."""
        now = time.perf_counter()
        with self._lock:
            if key == STAGE_KEY:
                if status is not None:
                    self.stage_name = status
                if fraction is not None and self.stage_name in self.stage_fraction:
                    old = self.stage_fraction[self.stage_name]
                    self.stage_fraction[self.stage_name] = max(old, min(max(float(fraction), 0.0), 1.0))
                if message is not None:
                    self.stage = message
                return
            if key not in self.status:
                return
            if status is not None:
                if status == "running" and key not in self.started_at:
                    self.started_at[key] = now
                if status in _FINAL_CHANNEL:
                    begun = self.started_at.get(key, now)
                    self.seconds[key] = now - begun
                    self.fraction[key] = 1.0
                self.status[key] = status
            if fraction is not None:
                self.fraction[key] = max(self.fraction[key], min(max(float(fraction), 0.0), 1.0))
            if message is not None:
                self.message[key] = message

    def skip_unfinished(self) -> None:
        """Mark every channel that has not reached a final status as skipped (used when a cancel stops the job
        before the channel loop got to them, for example during the Top-K ranking)."""
        for key in self.channels:
            with self._lock:
                pending = self.status[key] not in _FINAL_CHANNEL
            if pending:
                self.update(key, status="skipped", message="Not started: the fit was cancelled")

    def overall(self, state: str) -> float:
        """Overall share done (never decreases between calls)."""
        if state == "done":
            self.best = 1.0
            return 1.0
        total_cost = sum(CHANNEL_COST.get(k, 1.0) for k in self.channels) or 1.0
        fitted = sum(CHANNEL_COST.get(k, 1.0) * self.fraction[k] for k in self.channels) / total_cost
        fit_share = 1.0 - self.prep_share - self.finish_share
        value = (self.prep_share * self.stage_fraction["preparing"] + fit_share * fitted
                 + self.finish_share * self.stage_fraction["finishing"])
        self.best = max(self.best, min(value, 1.0))
        return self.best

    def snapshot(self, job_id: str, state: JobState, elapsed: float, error: str | None) -> JobSnapshot:
        """Copy the current progress into a :class:`JobSnapshot`."""
        now = time.perf_counter()
        with self._lock:
            rows = []
            for key in self.channels:
                status = self.status[key]
                if status == "running":
                    seconds = now - self.started_at.get(key, now)
                else:
                    seconds = self.seconds[key]
                rows.append(ChannelProgress(key=key, status=status, fraction=float(self.fraction[key]),
                                            message=self.message[key], seconds=float(seconds)))
            fraction = self.overall(state)
            stage = self.stage
        return JobSnapshot(job_id=job_id, state=state, elapsed=float(elapsed), fraction=float(fraction),
                           channels=tuple(rows), stage=stage, error=error)


# --------------------------------------------------------------------------------------------------------------
# Jobs
# --------------------------------------------------------------------------------------------------------------
_RUN_LOCK = threading.Lock()
_JOBS: "OrderedDict[str, TrainingJob]" = OrderedDict()
_JOBS_LOCK = threading.Lock()


def _remember(job: "TrainingJob") -> None:
    """Add ``job`` to the registry, forgetting the oldest idle jobs (finished or never started) beyond a few."""
    with _JOBS_LOCK:
        _JOBS[job.job_id] = job
        idle = [k for k, j in _JOBS.items() if k != job.job_id and (j.finished or not j._launched)]
        for key in idle[: max(len(idle) - _KEEP_FINISHED_JOBS, 0)]:
            _JOBS.pop(key, None)


def _forget(job: "TrainingJob") -> None:
    """Remove ``job`` from the registry (if it is the job registered under its id)."""
    with _JOBS_LOCK:
        if _JOBS.get(job.job_id) is job:
            _JOBS.pop(job.job_id, None)


def get_job(job_id: str) -> "TrainingJob | None":
    """The job with this id, if this process still remembers it."""
    with _JOBS_LOCK:
        return _JOBS.get(job_id)


def sync_training_requested() -> bool:
    """True when ``GRATICULE_SYNC_TRAINING`` is ``"1"`` (the UI then trains on its own thread, e.g. under AppTest)."""
    return os.environ.get(SYNC_ENV, "").strip() == "1"


def _friendly_error(exc: BaseException) -> str:
    """Error text for the UI: the plain message for expected data problems, with the traceback otherwise."""
    from graticule.data.sampling import SingleClassError

    if isinstance(exc, SingleClassError):
        return str(exc)
    trace = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    return f"{type(exc).__name__}: {exc}\n\n{trace}"


class TrainingJob:
    """One 02 Fit run: training matrices plus every requested channel, in a thread or inline.

    Attributes:
        job_id: unique string id (also the key in the process registry, see :func:`get_job`).
        prepared: the prepared sample being fitted on; released (None) once the job has finished, so the few
            finished jobs the registry remembers never keep an old sample alive (the run does not need it). A job
            turned away because another fit is running is dropped from the registry at once, so it keeps its
            sample only for as long as its caller holds the job (it can still be started later).
        request: the fit options.
        result: the finished :class:`~graticule.models.train.TrainingRun`; after a cancel it holds the channels
            finished before the cancel (the others are marked cancelled or skipped), or None when the cancel came
            before any channel ran. It is also kept when every channel failed. Whoever stores the run elsewhere
            calls :meth:`release_result`, so the finished jobs the registry remembers never keep old runs (models
            and matrices) alive; ``run_id`` still names the run afterwards.
        run_id: id of the run this job produced (kept after :meth:`release_result`), or None.
        error: user-facing error text (with traceback for unexpected errors), or None.
        exception: the exception that ended the job, or None.
    """

    def __init__(self, prepared: "PreparedDataset", request: "TrainRequest") -> None:
        self.job_id = f"job-{uuid.uuid4().hex[:12]}"
        self.prepared: PreparedDataset | None = prepared
        self.request = request
        self.result: TrainingRun | None = None
        self.run_id: str | None = None
        self.error: str | None = None
        self.exception: BaseException | None = None
        self._token = CancelToken()
        self._tracker = _Tracker(tuple(request.channels), request.feature_mode == "topk")
        self._state: JobState = "queued"
        self._state_lock = threading.Lock()
        self._started: float | None = None
        self._ended: float | None = None
        self._thread: threading.Thread | None = None
        self._done = threading.Event()
        self._launched = False
        _remember(self)

    # -- control ---------------------------------------------------------------------------------------------
    def _claim(self) -> None:
        """Take the process-wide training slot or raise :class:`JobBusyError` (the job then leaves the registry)."""
        with self._state_lock:
            if self._launched:
                raise RuntimeError("This job has already been started.")
            if not _RUN_LOCK.acquire(blocking=False):
                busy = True
            else:
                busy = False
                self._launched = True
                self._state = "running"
                self._started = time.perf_counter()
        if busy:
            _forget(self)  # nobody watches a job that never started; the caller decides whether to keep it
            raise JobBusyError("Another fit is still running in this app. Wait for it to finish or cancel it.")
        _remember(self)  # (again, if it was turned away earlier)

    def start(self) -> None:
        """Run the job in a daemon thread; raises :class:`JobBusyError` while another job is running."""
        self._claim()
        self._thread = threading.Thread(target=self._execute, name=f"graticule-{self.job_id}", daemon=True)
        self._thread.start()

    def run_inline(self) -> "TrainingRun":
        """Do the same work on the calling thread and return the run.

        Raises :class:`JobBusyError` while another job is running, and re-raises whatever ended the job (for
        example :class:`~graticule.data.sampling.SingleClassError`) after recording it in ``error``. A run that
        was cancelled from another thread is returned as it stands; a cancel before any channel ran raises
        :class:`TrainingCancelled`.
        """
        self._claim()
        self._execute()
        if self.exception is not None:
            raise self.exception
        if self.result is None:
            raise TrainingCancelled("The fit was cancelled before any channel was trained.")
        return self.result

    def cancel(self) -> None:
        """Ask the job to stop at its next check.

        Checks come between channels, forest chunks, boosting rounds (also those of the Top-K ranking), MLP epochs,
        logistic-regression iterations and batches of test rows being scored. A cancel that arrives after the last
        channel has finished changes nothing: the run then ends as done.
        """
        self._token.cancel()
        if not self.finished:
            self._tracker.update(STAGE_KEY, message="Cancelling...")

    def release_result(self) -> None:
        """Let go of the finished run once it is stored elsewhere (``run_id`` keeps naming it)."""
        with self._state_lock:
            self.result = None

    def wait(self, timeout: float | None = None) -> bool:
        """Block until the job has finished (or ``timeout`` seconds passed); True when it finished."""
        return self._done.wait(timeout)

    # -- state ------------------------------------------------------------------------------------------------
    @property
    def finished(self) -> bool:
        """True once the job has reached a final state."""
        return self._done.is_set()

    @property
    def state(self) -> JobState:
        """Current state: queued, running, done, failed or cancelled."""
        with self._state_lock:
            return self._state

    def snapshot(self) -> JobSnapshot:
        """An immutable copy of the current progress."""
        with self._state_lock:
            state = self._state
            started, ended = self._started, self._ended
            error = self.error
        if started is None:
            elapsed = 0.0
        else:
            elapsed = (ended if ended is not None else time.perf_counter()) - started
        return self._tracker.snapshot(self.job_id, state, elapsed, error)

    # -- work -------------------------------------------------------------------------------------------------
    def _execute(self) -> None:
        """The job body: build the matrices, fit the channels and record the outcome (holds the run slot)."""
        import joblib

        from graticule.models.train import build_training_data, train_all

        state: JobState = "failed"
        prepared = self.prepared
        try:
            if prepared is None:
                raise RuntimeError("This job has no sample to fit on.")
            with joblib.parallel_config(backend="threading"):
                self._tracker.update(STAGE_KEY, status="preparing", fraction=0.0,
                                     message="Building the training and test matrices")
                data = build_training_data(prepared, self.request, progress=self._tracker, cancel=self._token)
                if self._token.cancelled:
                    raise TrainingCancelled("The fit was cancelled while the matrices were being built.")
                run = train_all(data, self.request, data_request=prepared.request,
                                dataset_fingerprint=prepared.fingerprint, progress=self._tracker,
                                cancel=self._token, job_id=self.job_id)
            self.run_id = run.run_id
            self.result = run
            # Only the run says whether the cancel stopped something: a cancel pressed after the last channel had
            # finished (while the reference rows were kept, say) leaves a complete run, which ends as done.
            if run.cancelled:
                state = "cancelled"
                self._tracker.update(STAGE_KEY, message="Cancelled")
            elif not run.ok_channels():
                state = "failed"
                problems = "; ".join(f"{k}: {(r.error or r.status).splitlines()[0]}" for k, r in run.channels.items())
                self.error = f"No channel could be fitted. {problems}"
            else:
                state = "done"
                self._tracker.update(STAGE_KEY, status="finishing", fraction=1.0, message="Done")
        except TrainingCancelled:
            state = "cancelled"
            self._tracker.skip_unfinished()
            self._tracker.update(STAGE_KEY, message="Cancelled")
        except BaseException as exc:  # noqa: BLE001 - every failure must reach the UI instead of killing the thread
            state = "failed"
            self.exception = exc
            self.error = _friendly_error(exc)
            self._tracker.update(STAGE_KEY, message="Failed")
        finally:
            del prepared
            with self._state_lock:
                self._state = state
                self._ended = time.perf_counter()
                self.prepared = None
            self._done.set()
            _RUN_LOCK.release()
