"""Headless checks of the 02 Fit station: prerequisites, a fit through the form, and the single-class guard."""

from __future__ import annotations

import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from streamlit.testing.v1 import AppTest

from graticule.models import jobs, train, zoo
from graticule.settings import AppSettings, save_settings
from tests.helpers import make_rows, write_cic_csv
from tests.ui.harness import draw_synthetic_sample, errors, fresh_caches, goto, new_app  # noqa: F401
from ui import state, training_ui

pytestmark = pytest.mark.ui
MON = "Monday-WorkingHours.pcap_ISCX.csv"


def _markdown(at: AppTest) -> str:
    return " ".join(m.value for m in at.markdown)


def _fit_calls() -> int:
    return sum(train.FIT_CALLS.values())


def _readings(at: AppTest):  # noqa: ANN202 - a pandas frame from the element tree
    frames = [d.value for d in at.dataframe if "Balanced accuracy" in d.value.columns]
    assert frames, "no readings table rendered"
    return frames[0]


@pytest.fixture
def job_spy(monkeypatch: pytest.MonkeyPatch) -> list[object]:
    """Record every TrainingJob the Fit page creates (the real class still does the work)."""
    created: list[object] = []
    real = training_ui.TrainingJob

    def spy(*args: object, **kwargs: object) -> object:
        job = real(*args, **kwargs)  # type: ignore[arg-type]
        created.append(job)
        return job

    monkeypatch.setattr(training_ui, "TrainingJob", spy)
    return created


def test_without_a_sample_the_station_points_to_01_sample(fresh_caches: None) -> None:
    at = new_app("fit").run()
    assert not errors(at), errors(at)
    assert "Needs a sample: draw one at 01 Sample first." in _markdown(at)
    assert "fit_submit" not in [b.key for b in at.button]
    assert not [d for d in at.dataframe if "Balanced accuracy" in d.value.columns]


def test_a_fit_through_the_form_stores_one_run(fresh_caches: None, job_spy: list[object]) -> None:
    at = new_app().run()
    draw_synthetic_sample(at)
    assert not errors(at), errors(at)
    goto(at, "fit")
    assert not errors(at), errors(at)
    notes = _markdown(at)
    assert "On the bench:" in notes and "synthetic flows" in notes
    assert at.radio(key="fit_mode").value == "binary"
    assert at.checkbox(key="fit_port").value is False
    assert at.multiselect(key="fit_channels").value == list(training_ui.channel_options())
    assert at.toggle(key="fit_balanced").value is True
    assert any("kernel SVM fit time grows at least quadratically" in c.value for c in at.caption)
    before = _fit_calls()

    at.button(key="fit_submit").click().run()
    assert not errors(at), errors(at)
    assert len(job_spy) == 1
    assert _fit_calls() - before == 5  # one fit per channel, nothing else
    run_id = at.session_state[state.LAST_RUN_ID]
    run = at.session_state[state.RUN]
    assert run.run_id == run_id and run.request.profile == "test" and run.request.mode == "binary"
    assert not run.request.include_port
    assert run.dataset_fingerprint == at.session_state[state.PREPARED].fingerprint
    assert "fit" in at.session_state[state.DONE]
    assert "02 Fit ✓" in [link.proto.label for link in at.get("page_link")]  # the stepper ticks the station
    readings = _readings(at)
    assert readings["Channel"].tolist() == ["CH1 Random forest", "CH2 XGBoost", "CH3 RBF SVM", "CH4 Neural net (MLP)",
                                            "CH5 Logistic regression"]
    assert (readings["Status"] == "fitted").all()
    assert readings["Balanced accuracy"].between(0, 1).all() and readings["Accuracy"].between(0, 1).all()
    assert "Best balanced accuracy: **CH" in _markdown(at)
    assert "nothing refits until you press Fit again" in _markdown(at)
    assert any(t.value.startswith("Fit finished in") for t in at.toast)
    # The session and the registry hold the run; the finished job only remembers its id.
    assert job_spy[0].result is None and job_spy[0].run_id == run_id  # type: ignore[attr-defined]
    assert state.run_registry().get(run_id) is run

    # A plain rerun shows the same stored run without fitting again.
    at.run()
    assert _fit_calls() - before == 5 and at.session_state[state.LAST_RUN_ID] == run_id
    assert _readings(at)["Channel"].tolist() == readings["Channel"].tolist()


def test_a_benign_only_sample_warns_and_starts_no_job(tmp_path: Path, fresh_caches: None,
                                                       job_spy: list[object]) -> None:
    folder = tmp_path / "data"
    write_cic_csv(folder / MON, make_rows({"BENIGN": 40}))
    save_settings(AppSettings(data_dir=str(folder)))
    at = new_app().run()
    at.button(key="smp_draw").click().run()
    assert not errors(at), errors(at)
    goto(at, "fit")
    before = _fit_calls()
    at.radio(key="fit_mode").set_value("binary")
    at.button(key="fit_submit").click().run()
    assert not errors(at), errors(at)
    warnings = " ".join(w.value for w in at.warning)
    assert "Nothing was fitted." in warnings and "Only one class in this sample: every row is BENIGN" in warnings
    assert job_spy == [] and _fit_calls() == before
    assert state.JOB_ID not in at.session_state and state.LAST_RUN_ID not in at.session_state


def test_no_channel_chosen_is_a_warning(fresh_caches: None, job_spy: list[object]) -> None:
    at = new_app().run()
    draw_synthetic_sample(at)
    goto(at, "fit")
    at.multiselect(key="fit_channels").set_value([])
    at.button(key="fit_submit").click().run()
    assert not errors(at), errors(at)
    assert any("Choose at least one channel" in w.value for w in at.warning)
    assert job_spy == []


def test_a_multiclass_fit_respects_the_form(fresh_caches: None) -> None:
    at = new_app().run()
    draw_synthetic_sample(at)
    goto(at, "fit")
    at.radio(key="fit_mode").set_value("multiclass")
    at.multiselect(key="fit_channels").set_value(["forest", "svm"])
    at.toggle(key="fit_balanced").set_value(False)
    at.number_input(key="fit_svm_cap").set_value(2_000)
    at.slider(key="fit_test_share").set_value(0.3)
    at.button(key="fit_submit").click().run()
    assert not errors(at), errors(at)
    run = at.session_state[state.RUN]
    request = run.request
    assert request.mode == "multiclass" and request.channels == ("forest", "svm")
    assert request.balanced is False and request.svm_cap == 2_000 and request.test_share == pytest.approx(0.3)
    assert len(run.data.classes) > 2 and run.data.classes[0] == "BENIGN"
    assert _readings(at)["Channel"].tolist() == ["CH1 Random forest", "CH3 RBF SVM"]
    # The form now repeats the options of that fit.
    at.run()
    assert at.radio(key="fit_mode").value == "multiclass"
    assert at.multiselect(key="fit_channels").value == ["forest", "svm"]
    assert any("Options repeat your last fit" in c.value for c in at.caption)


def test_a_new_sample_after_a_fit_is_flagged(fresh_caches: None) -> None:
    at = new_app().run()
    draw_synthetic_sample(at)
    goto(at, "fit")
    at.multiselect(key="fit_channels").set_value(["logreg"])
    at.button(key="fit_submit").click().run()
    run_id = at.session_state[state.LAST_RUN_ID]
    draw_synthetic_sample(at, flows=2_400)
    goto(at, "fit")
    assert not errors(at), errors(at)
    warnings = " ".join(w.value for w in at.warning)
    assert "These readings come from an earlier sample" in warnings and run_id in warnings
    assert at.session_state[state.LAST_RUN_ID] == run_id


def test_the_last_fit_can_be_restored_in_a_new_session(fresh_caches: None) -> None:
    first = new_app().run()
    draw_synthetic_sample(first)
    goto(first, "fit")
    first.multiselect(key="fit_channels").set_value(["forest"])
    first.button(key="fit_submit").click().run()
    run_id = first.session_state[state.LAST_RUN_ID]
    calls = _fit_calls()

    # A browser refresh: a new session in the same app process. The run is offered, not restored silently.
    second = new_app("fit").run()
    assert not errors(second), errors(second)
    assert state.LAST_RUN_ID not in second.session_state
    restore = second.button(key="fit_restore")
    assert restore.label == f"Restore the last fit (run {run_id})"
    restore.click().run()
    assert not errors(second), errors(second)
    assert second.session_state[state.LAST_RUN_ID] == run_id
    assert _readings(second)["Channel"].tolist() == ["CH1 Random forest"]
    assert any(f"Restored run {run_id}" in i.value for i in second.info)
    assert _fit_calls() == calls
    # Without the sample in this session, the run still reads on its own; fitting again needs a sample.
    assert "To fit again, draw a sample at 01 Sample first." in _markdown(second)
    assert "fit_submit" not in [b.key for b in second.button]


def test_a_background_fit_shows_progress_then_its_readings(fresh_caches: None,
                                                          monkeypatch: pytest.MonkeyPatch) -> None:
    at = new_app().run()
    draw_synthetic_sample(at)
    goto(at, "fit")
    monkeypatch.setenv("GRATICULE_SYNC_TRAINING", "0")
    at.multiselect(key="fit_channels").set_value(["forest", "logreg"])
    at.button(key="fit_submit").click().run()
    assert not errors(at), errors(at)
    deadline = time.monotonic() + 60
    saw_progress = state.JOB_ID in at.session_state
    while state.LAST_RUN_ID not in at.session_state and time.monotonic() < deadline:
        time.sleep(0.2)
        at.run()
        assert not errors(at), errors(at)
    assert saw_progress
    assert state.LAST_RUN_ID in at.session_state and state.JOB_ID not in at.session_state
    assert _readings(at)["Channel"].tolist() == ["CH1 Random forest", "CH5 Logistic regression"]
    assert "fit" in at.session_state[state.DONE]


def test_a_background_fit_is_not_lost_when_its_tab_is_refreshed(fresh_caches: None, job_spy: list[object],
                                                                monkeypatch: pytest.MonkeyPatch) -> None:
    at = new_app().run()
    draw_synthetic_sample(at)
    goto(at, "fit")
    monkeypatch.setenv("GRATICULE_SYNC_TRAINING", "0")
    at.multiselect(key="fit_channels").set_value(["logreg"])
    at.button(key="fit_submit").click().run()
    assert not errors(at), errors(at)
    assert len(job_spy) == 1
    job = job_spy[0]
    assert job.wait(60)  # type: ignore[attr-defined]

    # The first tab is never rerun (it was refreshed); a new session still finds the run, as an offer.
    other = new_app("fit").run()
    assert not errors(other), errors(other)
    run_id = job.run_id  # type: ignore[attr-defined]
    assert run_id is not None
    assert other.button(key="fit_restore").label == f"Restore the last fit (run {run_id})"
    assert state.LAST_RUN_ID not in other.session_state
    # The registry took the run in, so the finished job no longer holds it (nor its sample).
    assert job.result is None and job.prepared is None  # type: ignore[attr-defined]
    assert state.run_registry().get(run_id) is not None


def _failing_builder(failing: set[str], monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the model builder raise for the channel keys in ``failing`` (the others build normally)."""
    real = zoo.build_estimator

    def build(key: str, ctx: zoo.BuildContext) -> object:
        if key in failing:
            raise RuntimeError(f"broken builder for {key}")
        return real(key, ctx)

    monkeypatch.setattr(zoo, "build_estimator", build)


def test_a_failing_channel_is_reported_and_the_others_kept(fresh_caches: None,
                                                          monkeypatch: pytest.MonkeyPatch) -> None:
    _failing_builder({"svm"}, monkeypatch)
    at = new_app().run()
    draw_synthetic_sample(at)
    goto(at, "fit")
    at.multiselect(key="fit_channels").set_value(["forest", "svm", "logreg"])
    at.button(key="fit_submit").click().run()
    assert not errors(at), errors(at)
    readings = _readings(at)
    assert readings["Status"].tolist() == ["fitted", "failed", "fitted"]
    assert "broken builder for svm" in readings["Notes"].iloc[1]
    assert readings["Balanced accuracy"].isna().tolist() == [False, True, False]
    assert "**CH3 RBF SVM failed.** RuntimeError: broken builder for svm" in _markdown(at)
    assert at.session_state[state.RUN].ok_channels() == ["forest", "logreg"]


def test_when_every_channel_fails_nothing_is_stored(fresh_caches: None, monkeypatch: pytest.MonkeyPatch) -> None:
    _failing_builder({"forest", "logreg"}, monkeypatch)
    at = new_app().run()
    draw_synthetic_sample(at)
    goto(at, "fit")
    at.multiselect(key="fit_channels").set_value(["forest", "logreg"])
    at.button(key="fit_submit").click().run()
    assert not errors(at), errors(at)
    problems = " ".join(e.value for e in at.error)
    assert "No channel could be fitted, so nothing was stored." in problems
    assert "CH1 Random forest: RuntimeError: broken builder for forest" in problems
    assert state.LAST_RUN_ID not in at.session_state
    assert not [d for d in at.dataframe if "Balanced accuracy" in d.value.columns]


class _FakeJob:
    """A job that stays "running" until the test says otherwise (real fits on tiny data end too fast to watch)."""

    job_id = "job-watched"

    def __init__(self) -> None:
        self.finished = False
        self.cancel_requests = 0
        self.result = None
        self.error: str | None = None
        self.exception: BaseException | None = None
        self.state = "running"

    def snapshot(self) -> jobs.JobSnapshot:
        channels = (
            jobs.ChannelProgress("forest", "ok", 1.0, "Fitted in 3.0 s", 3.0),
            jobs.ChannelProgress("xgboost", "running", 0.5, "Round 40", 1.5),
            jobs.ChannelProgress("svm", "waiting", 0.0, "Waiting", 0.0),
        )
        return jobs.JobSnapshot(self.job_id, self.state, 75.4, 0.4, channels,  # type: ignore[arg-type]
                                "Fitting xgboost", None)

    def cancel(self) -> None:
        self.cancel_requests += 1


def test_the_progress_panel_shows_a_running_job_and_cancels_it(fresh_caches: None,
                                                               monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeJob()
    lookup = {fake.job_id: fake}
    monkeypatch.setattr(jobs, "get_job", lookup.get)
    monkeypatch.setattr(training_ui, "get_job", lookup.get)
    at = new_app().run()
    draw_synthetic_sample(at)
    at.session_state[state.JOB_ID] = fake.job_id
    goto(at, "fit")
    assert not errors(at), errors(at)
    bars = at.get("progress")
    assert len(bars) == 1 and bars[0].proto.text == "Fitting CH2 XGBoost · elapsed 01:15"
    table = next(d.value for d in at.dataframe if "Done" in d.value.columns)
    assert table["Channel"].tolist() == ["CH1 Random forest", "CH2 XGBoost", "CH3 RBF SVM"]
    assert table["Status"].tolist() == ["fitted", "fitting", "waiting"]
    assert at.button(key="fit_submit").disabled  # no second fit while one runs
    at.button(key="fit_cancel").click().run()
    assert fake.cancel_requests == 1

    # The job ends as cancelled before any channel was kept: the session lets go of it and says so.
    fake.finished, fake.state = True, "cancelled"
    at.run()
    assert not errors(at), errors(at)
    assert state.JOB_ID not in at.session_state and state.LAST_RUN_ID not in at.session_state
    assert any("Fit cancelled before any channel was fitted" in i.value for i in at.info)
    assert not at.get("progress")


def test_elapsed_time_format() -> None:
    assert training_ui.format_elapsed(0) == "00:00"
    assert training_ui.format_elapsed(75.9) == "01:15"
    assert training_ui.format_elapsed(3_725) == "62:05"


def test_a_cancel_after_the_last_channel_reads_as_a_finished_fit(fresh_caches: None,
                                                                  monkeypatch: pytest.MonkeyPatch) -> None:
    real_quantiles = train.feature_quantiles

    def cancel_while_finishing(X: object) -> object:
        for job in list(jobs._JOBS.values()):
            if not job.finished:
                job.cancel()
        return real_quantiles(X)  # type: ignore[arg-type]

    monkeypatch.setattr(train, "feature_quantiles", cancel_while_finishing)
    at = new_app().run()
    draw_synthetic_sample(at)
    goto(at, "fit")
    at.multiselect(key="fit_channels").set_value(["forest", "logreg"])
    at.button(key="fit_submit").click().run()
    assert not errors(at), errors(at)
    run = at.session_state[state.RUN]
    assert not run.cancelled and run.ok_channels() == ["forest", "logreg"]
    assert any(t.value.startswith("Fit finished in") for t in at.toast)
    assert not any("cancelled" in i.value for i in at.info)


def test_fit_duration_counts_the_matrices_too() -> None:
    quick = SimpleNamespace(seconds=4.0, prep_seconds=0.3, total_seconds=4.3)
    slow = SimpleNamespace(seconds=21.7, prep_seconds=20.3, total_seconds=42.0)
    assert state.fit_duration_text(quick) == "4.3 s"  # type: ignore[arg-type]
    assert state.fit_duration_text(slow) == "42.0 s (matrices 20.3 s, channels 21.7 s)"  # type: ignore[arg-type]
