"""Background training jobs: threads, progress snapshots, cancel, one job per process, error capture, memory held by
finished jobs, and warnings kept per thread."""

from __future__ import annotations

import dataclasses
import gc
import threading
import time
import warnings
import weakref
from collections.abc import Iterator
from dataclasses import replace

import numpy as np
import pytest

from graticule.data.prepare import DataRequest, PreparedDataset, prepare_dataset
from graticule.data.sampling import SingleClassError
from graticule.models import train as train_mod
from graticule.models import zoo
from graticule.models.jobs import (
    JobBusyError,
    JobSnapshot,
    TrainingCancelled,
    TrainingJob,
    get_job,
    sync_training_requested,
    thread_warnings,
)
from graticule.models.train import TrainingRun, TrainRequest
from graticule.schema import BENIGN, LABEL
from xgboost import callback as xgb_callback

pytestmark = pytest.mark.integration
REQUEST = TrainRequest(mode="binary", channels=("forest", "xgboost", "logreg"), profile="test", seed=5)
WAIT = 120.0


@pytest.fixture(scope="module")
def prepared() -> PreparedDataset:
    """About 2,500 synthetic flows."""
    return prepare_dataset(DataRequest(source="synthetic", synthetic_flows=2_500, seed=5))


@pytest.fixture
def cleanup() -> Iterator[list[TrainingJob]]:
    """Jobs added here are cancelled and awaited after the test, so a failure never leaves the slot taken."""
    jobs: list[TrainingJob] = []
    yield jobs
    for job in jobs:
        job.cancel()
        job.wait(WAIT)


def _watch(job: TrainingJob) -> list[JobSnapshot]:
    """Poll snapshots until the job finishes."""
    seen = [job.snapshot()]
    deadline = time.monotonic() + WAIT
    while not job.finished and time.monotonic() < deadline:
        seen.append(job.snapshot())
        job.wait(0.005)
    seen.append(job.snapshot())
    return seen


def test_threaded_job_runs_to_completion(prepared: PreparedDataset, cleanup: list[TrainingJob]) -> None:
    job = TrainingJob(prepared, REQUEST)
    cleanup.append(job)
    assert job.snapshot().state == "queued" and not job.finished
    assert get_job(job.job_id) is job
    job.start()
    snapshots = _watch(job)
    assert job.finished and job.state == "done", job.error
    assert job.error is None and isinstance(job.result, TrainingRun)
    assert job.result.ok_channels() == ["forest", "xgboost", "logreg"]
    assert job._thread is not None and job._thread.daemon
    assert job.prepared is None, "a finished job lets go of its sample"

    fractions = [s.fraction for s in snapshots]
    assert fractions == sorted(fractions), "overall progress never goes back"
    assert fractions[-1] == 1.0
    for key in REQUEST.channels:
        per_channel = [next(c for c in s.channels if c.key == key).fraction for s in snapshots]
        assert per_channel == sorted(per_channel), key
    final = snapshots[-1]
    assert final.state == "done" and final.finished
    assert [c.key for c in final.channels] == list(REQUEST.channels)
    assert all(c.status == "ok" and c.fraction == 1.0 and c.seconds > 0 for c in final.channels)
    assert final.elapsed > 0 and job.snapshot().elapsed == final.elapsed, "elapsed stops at the end"


def test_snapshots_are_immutable_copies(prepared: PreparedDataset) -> None:
    job = TrainingJob(prepared, REQUEST)
    snap = job.snapshot()
    assert isinstance(snap.channels, tuple)
    with pytest.raises(dataclasses.FrozenInstanceError):
        snap.fraction = 0.5  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        snap.channels[0].status = "ok"  # type: ignore[misc]
    assert job.snapshot() is not snap


def test_cancel_before_start_stops_early(prepared: PreparedDataset, cleanup: list[TrainingJob]) -> None:
    job = TrainingJob(prepared, REQUEST)
    cleanup.append(job)
    job.cancel()
    job.start()
    assert job.wait(WAIT)
    assert job.state == "cancelled" and job.snapshot().state == "cancelled"
    assert job.result is None and job.error is None


def test_cancel_during_the_fit_keeps_finished_channels(prepared: PreparedDataset, cleanup: list[TrainingJob],
                                                       monkeypatch: pytest.MonkeyPatch) -> None:
    job = TrainingJob(prepared, REQUEST)
    cleanup.append(job)
    real_builder = zoo.BUILDERS["xgboost"]

    def cancel_then_build(ctx: zoo.BuildContext) -> object:
        job.cancel()
        return real_builder(ctx)

    monkeypatch.setitem(zoo.BUILDERS, "xgboost", cancel_then_build)
    job.start()
    assert job.wait(WAIT)
    assert job.state == "cancelled"
    run = job.result
    assert run is not None and run.cancelled
    assert run.channels["forest"].status == "ok"
    assert run.channels["xgboost"].status == "cancelled"
    assert run.channels["logreg"].status == "skipped"
    statuses = {c.key: c.status for c in job.snapshot().channels}
    assert statuses == {"forest": "ok", "xgboost": "cancelled", "logreg": "skipped"}


def test_only_one_job_runs_at_a_time(prepared: PreparedDataset, cleanup: list[TrainingJob],
                                     monkeypatch: pytest.MonkeyPatch) -> None:
    gate = threading.Event()
    entered = threading.Event()
    real_builder = zoo.BUILDERS["forest"]

    def held_builder(ctx: zoo.BuildContext) -> object:
        entered.set()
        gate.wait(WAIT)
        return real_builder(ctx)

    monkeypatch.setitem(zoo.BUILDERS, "forest", held_builder)
    first = TrainingJob(prepared, REQUEST)
    second = TrainingJob(prepared, REQUEST)
    cleanup.extend([first, second])
    try:
        first.start()
        assert entered.wait(WAIT)
        assert first.snapshot().state == "running"
        with pytest.raises(JobBusyError):
            second.start()
        with pytest.raises(JobBusyError):
            second.run_inline()
        assert second.snapshot().state == "queued"
        assert get_job(second.job_id) is None, "a job turned away is not kept by the registry"
        # A turned-away job that its caller drops takes its sample with it.
        sample = replace(prepared)
        sample_ref = weakref.ref(sample)
        turned_away = TrainingJob(sample, REQUEST)
        with pytest.raises(JobBusyError):
            turned_away.start()
        del sample, turned_away
        gc.collect()
        assert sample_ref() is None
    finally:
        gate.set()
    assert first.wait(WAIT) and first.state == "done"
    monkeypatch.setitem(zoo.BUILDERS, "forest", real_builder)
    second.start()
    assert get_job(second.job_id) is second, "a job is registered again once it really starts"
    assert second.wait(WAIT) and second.state == "done"
    with pytest.raises(RuntimeError):
        second.start()


def test_inline_run_matches_the_threaded_run(prepared: PreparedDataset, cleanup: list[TrainingJob]) -> None:
    inline_job = TrainingJob(prepared, REQUEST)
    inline = inline_job.run_inline()
    assert inline_job.state == "done" and inline_job.result is inline
    threaded_job = TrainingJob(prepared, REQUEST)
    cleanup.append(threaded_job)
    threaded_job.start()
    assert threaded_job.wait(WAIT)
    threaded = threaded_job.result
    assert threaded is not None
    assert threaded.data.classes == inline.data.classes
    assert np.array_equal(threaded.data.train_rows, inline.data.train_rows)
    assert threaded.ok_channels() == inline.ok_channels()
    for key, result in inline.channels.items():
        other = threaded.channels[key]
        assert result.proba is not None and other.proba is not None
        assert other.proba.shape == result.proba.shape
        assert other.rows_used == result.rows_used
        assert np.array_equal(other.y_pred, result.y_pred), key  # type: ignore[arg-type]
    assert threaded.reference_sample.shape == inline.reference_sample.shape
    assert threaded.feature_quantiles.shape == inline.feature_quantiles.shape


def test_single_class_data_fails_with_the_friendly_message(prepared: PreparedDataset,
                                                            cleanup: list[TrainingJob]) -> None:
    frame = prepared.frame
    benign = replace(prepared, frame=frame[frame[LABEL] == BENIGN].reset_index(drop=True))
    job = TrainingJob(benign, REQUEST)
    cleanup.append(job)
    job.start()
    assert job.wait(WAIT)
    assert job.state == "failed" and isinstance(job.exception, SingleClassError)
    assert job.error == str(job.exception) and "Traceback" not in job.error
    assert job.snapshot().error == job.error
    with pytest.raises(SingleClassError):
        TrainingJob(benign, REQUEST).run_inline()


def test_unexpected_errors_are_captured_with_a_traceback(prepared: PreparedDataset, cleanup: list[TrainingJob],
                                                         monkeypatch: pytest.MonkeyPatch) -> None:
    def explode(*args: object, **kwargs: object) -> object:
        raise RuntimeError("matrix builder exploded")

    monkeypatch.setattr(train_mod, "build_training_data", explode)
    job = TrainingJob(prepared, REQUEST)
    cleanup.append(job)
    job.start()
    assert job.wait(WAIT)
    assert job.state == "failed" and job.result is None
    assert job.error is not None and "RuntimeError: matrix builder exploded" in job.error
    assert "Traceback" in job.error
    # The slot was released: another job can run.
    monkeypatch.undo()
    again = TrainingJob(prepared, REQUEST)
    assert isinstance(again.run_inline(), TrainingRun)


def test_every_channel_failing_ends_in_failed_state(prepared: PreparedDataset,
                                                     monkeypatch: pytest.MonkeyPatch) -> None:
    def broken(ctx: zoo.BuildContext) -> object:
        raise ValueError("no model today")

    for key in ("forest", "logreg"):
        monkeypatch.setitem(zoo.BUILDERS, key, broken)
    job = TrainingJob(prepared, replace(REQUEST, channels=("forest", "logreg")))
    run = job.run_inline()
    assert job.state == "failed" and run.ok_channels() == []
    assert job.error is not None and "no model today" in job.error


def test_inline_cancel_before_any_channel_raises(prepared: PreparedDataset) -> None:
    job = TrainingJob(prepared, REQUEST)
    job.cancel()
    with pytest.raises(TrainingCancelled):
        job.run_inline()
    assert job.state == "cancelled"


def test_sync_training_flag_reads_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GRATICULE_SYNC_TRAINING", "1")
    assert sync_training_requested()
    monkeypatch.setenv("GRATICULE_SYNC_TRAINING", "0")
    assert not sync_training_requested()
    monkeypatch.delenv("GRATICULE_SYNC_TRAINING")
    assert not sync_training_requested()
    assert get_job("job-unknown") is None


def test_a_released_result_is_no_longer_kept_alive_by_the_job(prepared: PreparedDataset,
                                                               cleanup: list[TrainingJob]) -> None:
    job = TrainingJob(prepared, replace(REQUEST, channels=("logreg",)))
    cleanup.append(job)
    job.start()
    assert job.wait(WAIT) and job.state == "done"
    run = job.result
    assert run is not None and job.run_id == run.run_id
    run_ref = weakref.ref(run)
    job.release_result()
    del run
    gc.collect()
    assert run_ref() is None, "nothing but the caller held the run"
    assert job.result is None and job.run_id is not None and get_job(job.job_id) is job


def test_a_cancel_after_the_last_channel_leaves_the_run_done(prepared: PreparedDataset, cleanup: list[TrainingJob],
                                                             monkeypatch: pytest.MonkeyPatch) -> None:
    job = TrainingJob(prepared, replace(REQUEST, channels=("forest", "logreg")))
    cleanup.append(job)
    real_quantiles = train_mod.feature_quantiles

    def cancel_while_finishing(X: np.ndarray) -> np.ndarray:
        job.cancel()  # every channel has finished; only the reference rows and quantiles remain
        return real_quantiles(X)

    monkeypatch.setattr(train_mod, "feature_quantiles", cancel_while_finishing)
    job.start()
    assert job.wait(WAIT)
    assert job.state == "done" and job.snapshot().state == "done"
    run = job.result
    assert run is not None and not run.cancelled and run.ok_channels() == ["forest", "logreg"]
    assert job.snapshot().stage == "Done"


def test_a_cancel_during_the_topk_ranking_stops_it_and_skips_every_channel(
        prepared: PreparedDataset, cleanup: list[TrainingJob], monkeypatch: pytest.MonkeyPatch) -> None:
    request = replace(REQUEST, feature_mode="topk", top_k=6)
    job = TrainingJob(prepared, request)
    cleanup.append(job)
    rounds: list[int] = []
    real_rank = train_mod.rank_features

    class Counting(xgb_callback.TrainingCallback):
        def after_iteration(self, model: object, epoch: int, evals_log: object) -> bool:
            rounds.append(epoch)
            return False

    def cancel_then_rank(*args: object, callbacks: list[object] | None = None, **kwargs: object) -> object:
        job.cancel()
        return real_rank(*args, callbacks=[Counting(), *(callbacks or [])], **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(train_mod, "rank_features", cancel_then_rank)
    before = dict(train_mod.FIT_CALLS)
    job.start()
    assert job.wait(WAIT)
    assert job.state == "cancelled" and job.result is None and job.error is None
    assert len(rounds) == 1, "the ranking stopped after its first boosting round"
    statuses = {c.key: c.status for c in job.snapshot().channels}
    assert statuses == {key: "skipped" for key in request.channels}
    assert dict(train_mod.FIT_CALLS) == before


def test_warnings_from_other_threads_stay_out_of_the_notes(prepared: PreparedDataset, cleanup: list[TrainingJob],
                                                           monkeypatch: pytest.MonkeyPatch) -> None:
    """The fit thread records only its own warnings and never changes the process-wide warning state, even when
    another thread opens and closes its own ``catch_warnings`` block while the fit runs."""
    entered, gate = threading.Event(), threading.Event()
    real_builder = zoo.BUILDERS["forest"]

    def held_builder(ctx: zoo.BuildContext) -> object:
        entered.set()
        gate.wait(WAIT)
        warnings.warn("raised inside CH1 by its builder", UserWarning)
        return real_builder(ctx)

    monkeypatch.setitem(zoo.BUILDERS, "forest", held_builder)
    filters_before = list(warnings.filters)
    showwarning_before = warnings.showwarning
    impl_before = warnings._showwarnmsg_impl  # type: ignore[attr-defined]
    job = TrainingJob(prepared, replace(REQUEST, channels=("forest", "logreg")))
    cleanup.append(job)
    job.start()
    assert entered.wait(WAIT)
    # Entered while the channel runs and left only after the job has ended: the order that used to leave the
    # process with the fit thread's dead recorder.
    with warnings.catch_warnings(record=True) as mine:
        warnings.simplefilter("always")
        for i in range(5):
            warnings.warn(f"main thread {i}", UserWarning)
        gate.set()
        assert job.wait(WAIT) and job.state == "done", job.error
        warnings.warn("main thread after the fit", UserWarning)
    texts = [str(w.message) for w in mine]
    assert texts == [f"main thread {i}" for i in range(5)] + ["main thread after the fit"]
    run = job.result
    assert run is not None
    every_note = " ".join(note for result in run.channels.values() for note in result.notes)
    assert "main thread" not in every_note
    assert any("raised inside CH1 by its builder" in note for note in run.channels["forest"].notes)
    assert not any("raised inside CH1" in note for note in run.channels["logreg"].notes)
    assert warnings.showwarning is showwarning_before
    assert warnings._showwarnmsg_impl is impl_before  # type: ignore[attr-defined]
    assert warnings.filters == filters_before
    with warnings.catch_warnings(record=True) as later:
        warnings.simplefilter("always")
        warnings.warn("still delivered", UserWarning)
    assert [str(w.message) for w in later] == ["still delivered"]


def test_thread_warnings_nest_and_record_only_their_own_thread() -> None:
    other: list[str] = []

    def elsewhere() -> None:
        with thread_warnings() as theirs:
            warnings.warn("from the other thread", UserWarning)
        other.extend(str(w.message) for w in theirs)

    with warnings.catch_warnings():
        warnings.simplefilter("always")
        with thread_warnings() as outer:
            warnings.warn("outer", UserWarning)
            with thread_warnings() as inner:
                warnings.warn("inner", UserWarning)
                thread = threading.Thread(target=elsewhere)
                thread.start()
                thread.join()
            warnings.warn("outer again", UserWarning)
    assert [str(w.message) for w in outer] == ["outer", "outer again"]
    assert [str(w.message) for w in inner] == ["inner"]
    assert other == ["from the other thread"]
