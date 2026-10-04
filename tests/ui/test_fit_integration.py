"""02 Fit end to end across the station and the training runtime: real jobs, real cancel, Top-K with the port.

The page tests in ``test_fit_page.py`` drive each feature of the station; these tests check that the station and
:mod:`graticule.models` agree on what flows between them: a real background job cancelled from the progress panel
(the partial run is kept and read back), and a Top-K fit whose ranking report and port opt-in reach the readings.
Each starts from the session-wide synthetic sample, already drawn (``drawn_sample`` in tests/ui/harness.py).
"""

from __future__ import annotations

import threading
import time

import pytest
from streamlit.testing.v1 import AppTest

from graticule.models import jobs, train, zoo
from graticule.schema import DESTINATION_PORT
from tests.ui.harness import app_with_sample, drawn_sample, errors, fresh_caches, goto  # noqa: F401
from ui import state

pytestmark = pytest.mark.ui
WAIT = 60.0


def _markdown(at: AppTest) -> str:
    return " ".join(m.value for m in at.markdown)


def _readings(at: AppTest):  # noqa: ANN202 - a pandas frame from the element tree
    frames = [d.value for d in at.dataframe if "Balanced accuracy" in d.value.columns]
    assert frames, "no readings table rendered"
    return frames[0]


def test_cancel_in_the_progress_panel_stops_a_real_background_fit(fresh_caches: None,
                                                                  monkeypatch: pytest.MonkeyPatch) -> None:
    entered, gate = threading.Event(), threading.Event()
    real_builder = zoo.BUILDERS["xgboost"]

    def held_builder(ctx: zoo.BuildContext) -> object:
        """Hold CH2 at its start until the test lets it go, so the job is surely running when Cancel is pressed."""
        entered.set()
        gate.wait(WAIT)
        return real_builder(ctx)

    monkeypatch.setitem(zoo.BUILDERS, "xgboost", held_builder)
    at = app_with_sample(drawn_sample(), "fit").run()
    monkeypatch.setenv("GRATICULE_SYNC_TRAINING", "0")
    at.pills(key="fit_channels").set_value(["forest", "xgboost", "logreg"])
    before = train.FIT_CALLS.copy()
    at.button(key="fit_submit").click().run()
    assert not errors(at), errors(at)
    job = jobs.get_job(at.session_state[state.JOB_ID])
    assert job is not None
    try:
        # The Fit button's callback started the job before the page was drawn, so this very run already shows
        # the progress panel and a disabled Fit button.
        assert at.button(key="fit_submit").disabled
        assert len(at.get("progress")) == 1
        assert entered.wait(WAIT), "CH2 never started"
        at.run()
        assert not errors(at), errors(at)
        progress = next(d.value for d in at.dataframe if "Done" in d.value.columns)
        assert progress["Status"].tolist() == ["fitted", "fitting", "waiting"]
        assert at.button(key="fit_submit").disabled
        at.button(key="fit_cancel").click().run()
        assert not errors(at), errors(at)
    finally:
        gate.set()
        assert job.wait(WAIT)
    assert job.state == "cancelled"

    # The next rerun adopts the partial run: CH1 kept, CH2 cancelled at its start, CH5 never started.
    deadline = time.monotonic() + WAIT
    while state.JOB_ID in at.session_state and time.monotonic() < deadline:
        at.run()
    assert not errors(at), errors(at)
    assert state.JOB_ID not in at.session_state
    run = at.session_state[state.RUN]
    assert run.run_id == job.run_id and run.cancelled and run.ok_channels() == ["forest"]
    assert job.result is None, "the adopted run is held by the session and the registry, not by the job"
    assert _readings(at)["Status"].tolist() == ["fitted", "cancelled", "skipped"]
    assert any("Fit cancelled after 1 of 3 channels" in i.value for i in at.info)
    assert "**CH5 Logistic regression skipped.**" in _markdown(at)
    assert not at.get("progress")
    added = train.FIT_CALLS - before
    assert added["forest"] == 1 and added["logreg"] == 0  # CH2 entered fit_model once and stopped at its check
    assert "fit" in at.session_state[state.DONE]


def test_a_background_fit_that_ends_on_another_station_is_adopted_there(fresh_caches: None,
                                                                       monkeypatch: pytest.MonkeyPatch) -> None:
    entered, gate = threading.Event(), threading.Event()
    real_builder = zoo.BUILDERS["logreg"]

    def held_builder(ctx: zoo.BuildContext) -> object:
        """Keep the job running until the viewer has moved on to 03 Measure."""
        entered.set()
        gate.wait(WAIT)
        return real_builder(ctx)

    monkeypatch.setitem(zoo.BUILDERS, "logreg", held_builder)
    at = app_with_sample(drawn_sample(), "fit").run()
    monkeypatch.setenv("GRATICULE_SYNC_TRAINING", "0")
    at.pills(key="fit_channels").set_value(["logreg"])
    at.button(key="fit_submit").click().run()
    job = jobs.get_job(at.session_state[state.JOB_ID])
    assert job is not None
    try:
        assert entered.wait(WAIT)
        goto(at, "measure")
        assert not errors(at), errors(at)
        assert state.JOB_ID in at.session_state and state.LAST_RUN_ID not in at.session_state
    finally:
        gate.set()
        assert job.wait(WAIT)

    # The stepper of whatever page renders next takes the run in and ticks 02 Fit.
    at.run()
    assert not errors(at), errors(at)
    assert state.JOB_ID not in at.session_state
    assert at.session_state[state.LAST_RUN_ID] == job.run_id
    assert job.result is None
    assert "02 Fit ✓" in [link.proto.label for link in at.get("page_link")]
    goto(at, "fit")
    assert not errors(at), errors(at)
    assert any(t.value.startswith("Fit finished in") for t in at.toast)
    assert _readings(at)["Channel"].tolist() == ["CH5 Logistic regression"]


@pytest.mark.usefixtures("quick_ranking")  # a 20-round ranking model (tests/conftest.py)
def test_a_topk_fit_with_the_port_reports_its_ranking(fresh_caches: None) -> None:
    at = app_with_sample(drawn_sample(), "fit").run()
    at.radio(key="fit_features").set_value("topk")
    at.number_input(key="fit_k").set_value(10)
    at.checkbox(key="fit_port").set_value(True)
    at.pills(key="fit_channels").set_value(["xgboost"])
    at.button(key="fit_submit").click().run()
    assert not errors(at), errors(at)
    run = at.session_state[state.RUN]
    assert run.request.feature_mode == "topk" and run.request.top_k == 10 and run.request.include_port
    names = run.data.feature_names
    assert len(names) == 11 and names[-1] == DESTINATION_PORT and run.data.feature_choice.include_port
    overlap = run.data.reports["topk_overlap"]
    assert overlap["k"] == 10 and overlap["ranked_on_rows"] == len(run.data.y_train)
    text = _markdown(at)
    assert "11 columns (top 10)" in text and "Destination Port included" in text
    candidates = overlap["candidates"]
    assert candidates == len(run.data.feature_choice.ranking or ()) and DESTINATION_PORT not in dict(
        run.data.feature_choice.ranking or ())
    assert (f"**Top-K.** {candidates} candidate columns were ranked on {len(run.data.y_train):,} training rows only"
            in text)
    assert "and the top 10 kept; Destination Port was added by choice and was not ranked." in text
    assert "share their values on the 11 columns in use" in text
    assert _readings(at)["Channel"].tolist() == ["CH2 XGBoost"]
