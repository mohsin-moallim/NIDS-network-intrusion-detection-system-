"""Headless checks of the 03 Measure station: prerequisites, every section after a tiny fit, no refits on widget
changes, the on-demand measurements (cross-validation, permutation importance) and a run without held-out rows."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from nids import evaluate
from nids.models import train
from nids.models.jobs import claim_slot, release_slot
from tests.ui.harness import app_with_run, errors, fit_synthetic, fresh_caches, goto, new_app  # noqa: F401
from ui import state
from ui.pages import measure

pytestmark = pytest.mark.ui
SECTIONS = ("Readings overview", "Confusion matrices", "Curves", "Channel detail", "Timing", "Cross-validation",
            "Downloads")


def _fits() -> int:
    return sum(train.FIT_CALLS.values())


def _subheaders(at: AppTest) -> list[str]:
    return [h.value for h in at.subheader]


def _charts(at: AppTest) -> list[dict]:
    """The Vega-Lite specs drawn on the page."""
    import json

    return [json.loads(c.proto.spec) for c in at.get("vega_lite_chart")]


def _titles(at: AppTest) -> list[str]:
    titles = []
    for spec in _charts(at):
        title = spec.get("title")
        titles.append(title.get("text") if isinstance(title, dict) else str(title or ""))
    return titles


def _frames(at: AppTest, column: str) -> list:
    return [d.value for d in at.dataframe if column in d.value.columns]


def _fitted_app(mode: str = "binary", channels: list[str] | None = None) -> AppTest:
    """A session holding a small synthetic sample and a fit of it, at 02 Fit: fitted directly, as 02 Fit fits
    (the 02 Fit form has its own tests; ``test_no_retrain.py`` goes from the form to this station)."""
    options: dict[str, object] = {"mode": mode}
    if channels is not None:
        options["channels"] = tuple(channels)
    prepared, run = fit_synthetic(flows=2_000, budget=1_200, **options)
    at = app_with_run(run, prepared, key="fit").run()
    assert not errors(at), errors(at)
    assert at.session_state[state.LAST_RUN_ID] == run.run_id
    return at


def test_without_a_fit_the_station_points_to_02_fit(fresh_caches: None) -> None:
    at = new_app("measure").run()
    assert not errors(at), errors(at)
    assert "Needs a fitted channel" in " ".join(m.value for m in at.markdown)
    assert "Go to 02 Fit" in [link.proto.label for link in at.get("page_link")]
    assert not at.get("vega_lite_chart")


def test_binary_run_renders_every_section_and_widgets_never_refit(fresh_caches: None) -> None:
    at = _fitted_app("binary")
    run = at.session_state[state.RUN]
    fits = _fits()
    goto(at, "measure")
    assert not errors(at), errors(at)
    assert _subheaders(at) == list(SECTIONS)
    # Readings computed once and kept on the run.
    evals = evaluate.cached_evaluations(run)
    assert evals is not None and list(evals) == run.ok_channels()
    board = _frames(at, "Gap to best")[0]
    assert list(board.columns[:3]) == ["Channel", "Balanced accuracy", "Accuracy"]
    assert {"Precision (attack)", "Recall (attack)", "F1 (attack)", "ROC-AUC", "Average precision"} <= set(board.columns)
    assert board["Balanced accuracy"].is_monotonic_decreasing
    # Timing and rows sit in their own table under Timing (so the leaderboard fits the page), in the same order,
    # with scoring speed in whole flows per second.
    timing = _frames(at, "Single-flow ms")[0]
    assert list(timing.columns) == ["Channel", "Fit s", "Flows/s", "Single-flow ms", "Rows used", "Training rows"]
    assert list(timing["Channel"]) == list(board["Channel"]) and "Fit s" not in board.columns
    assert all(float(v).is_integer() for v in timing["Flows/s"].dropna())
    assert "measure" in at.session_state[state.DONE]  # the readings are shown: 03 Measure is ticked
    assert any("held-out rows: ○ Normal" in c.value and "◆ Attack" in c.value for c in at.caption)
    assert "CH3 trained on" in " ".join(m.value for m in at.markdown)  # the SVM row-cap badge
    titles = _titles(at)
    assert "Readings by channel" in titles and "Held-out rows per class" in titles
    assert titles.count("CH1 Random forest") == 1 and "CH5 Logistic regression" in titles  # confusion matrices
    assert "ROC curves" in titles and "Precision-recall curves" in titles
    assert any(t.endswith("built-in importance") for t in titles)
    assert {"Fit time", "Scoring speed", "Single-flow latency"} <= set(titles)
    assert [b.key for b in at.get("download_button")][:2] == ["ms_dl_board", "ms_dl_per_class"]

    # Changing every widget redraws without refitting, and never recomputes the readings. (Here all of them in one
    # redraw; tests/ui/test_no_retrain.py changes each widget of this station on its own, with a redraw after each.)
    first = evaluate.cached_evaluations(run)
    at.radio(key="ms_cm_show").set_value("Counts")
    at.radio(key="ms_roc_zoom").set_value(measure.ROC_OPTIONS[1])
    at.selectbox(key=measure.run_key("ms_detail_channel", run)).set_value("svm")
    at.number_input(key="ms_cv_k").set_value(3)
    at.multiselect(key=measure.run_key("ms_cv_channels", run)).set_value(["forest", "logreg"])
    at.run()
    assert not errors(at), errors(at)
    assert _fits() == fits
    assert all(len(chart.proto.datasets) > 0 for chart in at.get("vega_lite_chart"))  # kept specs keep their data
    later = evaluate.cached_evaluations(run)
    assert later is not None and all(later[k] is first[k] for k in first)  # type: ignore[index]
    assert "CH3 RBF SVM has no built-in importance" in " ".join(c.value for c in at.caption)
    estimate = [c.value for c in at.caption if "3 folds for each of 2 channels" in c.value]
    assert estimate and "on this machine" in estimate[0]

    # Cross-validation runs only on its button, on the two chosen channels, and adds its results.
    at.button(key="ms_cv_run").click().run()
    assert not errors(at), errors(at)
    assert _fits() - fits == 3 * 2
    cv = evaluate.stored_cross_validation(run)
    assert cv is not None and list(cv["key"]) == ["forest", "logreg"] and cv.attrs["k"] == 3
    assert _frames(at, "Balanced accuracy mean")
    assert "Cross-validation: balanced accuracy" in _titles(at)
    assert any(s.value.startswith("Cross-validation finished") for s in at.success)
    assert "ms_dl_cv" in [b.key for b in at.get("download_button")]
    at.radio(key="ms_cv_metric").set_value("F1 macro").run()
    assert not errors(at) and _fits() - fits == 6
    assert "Cross-validation: f1 macro" in _titles(at)

    # Permutation importance fits nothing and adds its chart.
    at.button(key="ms_perm_run").click().run()
    assert not errors(at), errors(at)
    assert _fits() - fits == 6
    assert "svm" in evaluate.stored_permutations(run)
    assert "CH3 RBF SVM: permutation importance" in _titles(at)

    # The same run loaded without its held-out rows: explained, with the way back; nothing is measured.
    _show_hollow_run(at, run)


def _show_hollow_run(at: AppTest, run: object) -> None:
    """Make a copy of ``run`` without held-out rows the session's current run, and check how 03 Measure reads it.

    The run is synthetic, so the station explains that its generated sample could not be reproduced, and sends
    the viewer to the Logbook, not to the Bench's data folder (which plays no part for generated flows).
    """
    data = run.data  # type: ignore[attr-defined]
    empty = replace(data, X_test=np.empty((0, data.n_features), dtype=np.float32),
                    y_test=np.empty(0, dtype=np.int64), test_rows=np.empty(0, dtype=np.int64),
                    detailed_test_labels=np.empty(0, dtype=str))
    hollow = replace(run, run_id=run.run_id + "-h", data=empty)  # type: ignore[type-var, attr-defined]
    state.run_registry().put(hollow.run_id, hollow)
    at.session_state[state.LAST_RUN_ID] = hollow.run_id
    at.session_state[state.RUN] = hollow
    fits = _fits()
    goto(at, "measure")
    assert not errors(at), errors(at)
    text = " ".join(m.value for m in at.markdown)
    assert "without its held-out rows" in text and "generated from a seed" in text and "data folder" not in text
    links = [link.proto.label for link in at.get("page_link")]
    assert "Go to Logbook" in links and "Go to Bench" not in links
    assert not at.get("vega_lite_chart") and "ms_cv_run" not in [b.key for b in at.button]
    assert _frames(at, "Balanced accuracy")  # the readings recorded at fit time are still listed
    assert evaluate.cached_evaluations(hollow) is None and _fits() == fits


@pytest.mark.slow  # the binary test above covers every section; this adds the per-class views (about 4 s)
def test_multiclass_run_shows_per_class_curves(fresh_caches: None, monkeypatch: pytest.MonkeyPatch) -> None:
    at = _fitted_app("multiclass", channels=["forest", "logreg"])
    run = at.session_state[state.RUN]
    fits = _fits()
    goto(at, "measure")
    assert not errors(at), errors(at)
    assert _subheaders(at) == list(SECTIONS)
    board = _frames(at, "Gap to best")[0]
    assert {"F1 macro", "F1 weighted", "Precision macro", "Precision weighted", "Recall macro",
            "Recall weighted"} <= set(board.columns) and "F1 (attack)" not in board.columns
    titles = _titles(at)
    assert "CH1 Random forest: one-vs-rest ROC" in titles
    at.selectbox(key=measure.run_key("ms_curve_channel", run)).set_value("logreg").run()
    assert not errors(at), errors(at)
    assert "CH5 Logistic regression: one-vs-rest precision-recall" in _titles(at)
    at.radio(key="ms_roc_zoom").set_value(measure.ROC_OPTIONS[1]).run()
    assert not errors(at), errors(at)
    zoomed = next(spec for spec in _charts(at) if spec["title"]["text"].endswith("one-vs-rest ROC"))
    assert zoomed["title"]["subtitle"].startswith("Zoomed: false-positive rate up to 5%")
    per_class = _frames(at, "Held-out rows")[0]
    assert per_class["Class"].iloc[0].startswith("○ ") and per_class["Class"].iloc[1].startswith("◆ ")
    assert len(per_class) == len(run.data.classes)
    assert _fits() == fits

    # A measurement started in the background: the progress panel shows, and the result is read once it ends.
    at.selectbox(key=measure.run_key("ms_detail_channel", run)).set_value("logreg").run()
    monkeypatch.setenv("NIDS_SYNC_TRAINING", "0")
    at.button(key="ms_perm_run").click().run()
    assert not errors(at), errors(at)
    task = evaluate.get_task(at.session_state[measure.PERM_TASK])
    assert task is not None and task.kind == "perm" and task.extra["key"] == "logreg"
    assert task.wait(60)
    at.run()
    assert not errors(at), errors(at)
    assert measure.PERM_TASK not in at.session_state and evaluate.get_task(task.task_id) is None
    assert any(s.value.startswith("Permutation importance of CH5 Logistic regression finished") for s in at.success)
    assert "CH5 Logistic regression: permutation importance" in _titles(at)
    assert "logreg" in evaluate.stored_permutations(run) and _fits() == fits

    # While a fit runs anywhere in the app, neither measurement can start: one fit or measurement at a time.
    monkeypatch.setenv("NIDS_SYNC_TRAINING", "1")
    assert claim_slot("a fit")
    try:
        at.run()
        assert not errors(at), errors(at)
        assert at.button(key="ms_cv_run").disabled and at.button(key="ms_perm_run").disabled
        assert sum("A fit is running in this app" in c.value for c in at.caption) == 2
    finally:
        release_slot()
    at.run()
    assert not at.button(key="ms_cv_run").disabled and not at.button(key="ms_perm_run").disabled


def test_estimates_read_as_rough_durations() -> None:
    assert measure.duration_text(2) == "under 5 s"
    assert measure.duration_text(38) == "about 40 s"
    assert measure.duration_text(130) == "about 2 min"
    assert measure.duration_text(7_500) == "about 2 h 5 min"
    assert measure.duration_range_text(0.4, 3.9) == "under 5 s"
    assert measure.duration_range_text(1.2, 18) == "up to about 20 s"
    assert measure.duration_range_text(9, 41) == "about 10 to 40 s"
    assert measure.duration_range_text(31, 29) == "about 30 s"
    assert measure.duration_range_text(70, 400) == "about 1 to 7 min"
    assert measure.class_text("BENIGN") == "○ BENIGN" and measure.class_text("Bot") == "◆ Bot"


def test_permutation_estimate_counts_the_cost_of_every_call() -> None:
    """Each scoring pays the channel's per-row time on the drawn rows plus the fixed cost of one call (its
    single-flow latency); only the channels that do not thread themselves may gain from several threads."""
    from types import SimpleNamespace

    result = SimpleNamespace(predict_seconds=0.5)
    run = SimpleNamespace(channels={"forest": result, "mlp": result},
                          data=SimpleNamespace(y_test=np.zeros(50_000), feature_names=tuple("abcd")))
    low, high = measure.permutation_estimate(run, "forest", 30.0, 5_000, 5)  # type: ignore[arg-type]
    assert low == high == pytest.approx(21 * (0.05 + 0.03))  # 1 + 4 features x 5 repeats scorings
    low, high = measure.permutation_estimate(run, "mlp", 1.0, 5_000, 5)  # type: ignore[arg-type]
    assert high == pytest.approx(21 * 0.051) and low <= high
