"""03 Measure's "Count each reading over" control: distinct flows by default, the recorded-traffic estimate on demand
(with a caution and a range table when heavily repeated flows missed the held-out rows), and a plain note instead of
the control when a run cannot be weighted. Nothing is refitted and the readings of either kind are computed once per
run, the estimate only when it is first asked for."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from nids import evaluate, theme
from nids.models import train
from tests.ui.harness import app_with_run, errors, fit_synthetic, fresh_caches, goto  # noqa: F401
from ui.pages import measure

pytestmark = pytest.mark.ui
SECTIONS = ["Readings overview", "Confusion matrices", "Curves", "Channel detail", "Timing", "Cross-validation",
            "Downloads"]


def _fits() -> int:
    return sum(train.FIT_CALLS.values())


def _titles(at: AppTest) -> list[str]:
    titles = []
    for chart in at.get("vega_lite_chart"):
        title = json.loads(chart.proto.spec).get("title")
        titles.append(title.get("text") if isinstance(title, dict) else str(title or ""))
    return titles


def _board(at: AppTest):  # noqa: ANN202 - a pandas frame from the element tree
    return next(d.value for d in at.dataframe if "Gap to best" in d.value.columns)


def _captions(at: AppTest) -> str:
    return " ".join(c.value for c in at.caption)


@pytest.fixture(scope="module")
def fitted() -> tuple[object, object]:
    """A small synthetic sample thinned by its budget (so the weights scale classes up) and a CH1 + CH5 fit."""
    return fit_synthetic(flows=2_500, budget=1_200, seed=5, channels=("forest", "logreg"))


def test_the_view_switches_between_distinct_flows_and_recorded_traffic_without_refitting(
        fresh_caches: None, fitted: tuple[object, object]) -> None:
    prepared, run = fitted
    at = app_with_run(run, prepared, key="measure").run()
    assert not errors(at), errors(at)
    control = at.radio(key=measure.COUNT_KEY)
    assert list(control.options) == list(measure.COUNT_OPTIONS) and control.value == "Distinct flows"
    assert [h.value for h in at.subheader] == SECTIONS
    evals = evaluate.cached_evaluations(run)  # type: ignore[arg-type]
    assert evals is not None
    distinct = evaluate.leaderboard(evals, run)  # type: ignore[arg-type]
    shown = _board(at)
    assert list(shown["Channel"]) == list(distinct["Channel"])
    assert "Readings by channel" in _titles(at) and "Held-out rows per class" in _titles(at)
    # The default view never computes the estimate (its downloads make their files only when pressed).
    assert evaluate.cached_traffic_readings(run) is None  # type: ignore[arg-type]
    fits = _fits()

    control.set_value("Recorded traffic (estimate)").run()
    assert not errors(at), errors(at)
    assert [h.value for h in at.subheader] == SECTIONS
    traffic = evaluate.cached_traffic_readings(run)  # type: ignore[arg-type]
    assert traffic is not None and set(traffic) == {"forest", "logreg"}  # computed now, once
    summary = evaluate.traffic_summary(run)  # type: ignore[arg-type]
    assert summary is not None
    text = _captions(at)
    assert (f"The {summary.rows:,} held-out rows stand for {summary.repeats:,} rows of the cleaned files and for "
            f"{evaluate.flows_text(summary.flows)} recorded flows") in text
    assert "Weighted to" in text and "Curves, importance, timing and cross-validation still count" in text
    assert "The curves count each distinct held-out flow once" in text
    weighted = evaluate.traffic_leaderboard(traffic, run, evals)  # type: ignore[arg-type]
    board = _board(at)
    assert list(board["Channel"]) == list(weighted["Channel"])
    for column in ("Balanced accuracy", "Accuracy"):  # the weighted scores, as a table prints them
        np.testing.assert_array_equal(board[column].to_numpy(dtype=float), theme.shown_scores(weighted[column]))
    titles = _titles(at)
    assert "Readings by channel: recorded traffic" in titles and "Recorded flows per class" in titles
    assert any("Estimated flows" in d.value.columns for d in at.dataframe)  # the per-class table follows
    at.radio(key="ms_cm_show").set_value("Counts").run()
    at.selectbox(key=measure.run_key("ms_detail_channel", run)).set_value("logreg").run()
    assert not errors(at), errors(at)
    assert _fits() == fits  # switching views and widgets never fits
    again = evaluate.cached_traffic_readings(run)  # type: ignore[arg-type]
    assert again is not None and all(again[k] is traffic[k] for k in traffic)  # never recomputed
    later = evaluate.cached_evaluations(run)  # type: ignore[arg-type]
    assert later is not None and all(later[k] is evals[k] for k in evals)

    at.radio(key=measure.COUNT_KEY).set_value("Distinct flows").run()
    assert not errors(at), errors(at)
    assert "Readings by channel" in _titles(at) and "Weighted to" not in _captions(at)


def test_a_run_without_repeat_counts_explains_why_there_is_no_estimate(
        fresh_caches: None, fitted: tuple[object, object]) -> None:
    prepared, run = fitted
    bare = replace(run, run_id=run.run_id + "-bare", data=replace(run.data, test_copies=None))  # type: ignore[type-var]
    at = app_with_run(bare, prepared, key="measure").run()
    assert not errors(at), errors(at)
    assert measure.COUNT_KEY not in [r.key for r in at.radio]
    text = _captions(at)
    assert "A recorded-traffic estimate is not available for this run" in text
    assert "distinct-flow readings" in text
    assert [h.value for h in at.subheader] == SECTIONS
    assert evaluate.cached_traffic_readings(bare) is None  # type: ignore[arg-type]


def test_heavily_repeated_flows_missing_from_the_held_out_rows_bring_a_caution_and_the_range_table(
        fresh_caches: None, tmp_path: Path) -> None:
    from nids.data.prepare import DataRequest, prepare_dataset
    from nids.models.train import TrainRequest, build_training_data, train_all
    from tests.unit.test_traffic import SEED, WEDNESDAY, _heavy_tail_file

    _heavy_tail_file(tmp_path)  # four attack flows recorded 150 times each; seed 0 holds none of them out
    prepared = prepare_dataset(DataRequest(source="cicids", data_dir=str(tmp_path), files=(WEDNESDAY,), seed=SEED))
    request = TrainRequest(profile="test", seed=0, channels=("forest", "logreg"))
    data = build_training_data(prepared, request)
    assert data.reports["heavy_flows"]["classes"]["Attack"]["held_out"] == 0
    run = train_all(data, request, data_request=prepared.request, dataset_fingerprint=prepared.fingerprint)
    at = app_with_run(run, prepared, key="measure").run()
    assert not errors(at), errors(at)
    assert not any("Bal. low" in d.value.columns for d in at.dataframe)  # the default view shows no estimate
    at.radio(key=measure.COUNT_KEY).set_value("Recorded traffic (estimate)").run()
    assert not errors(at), errors(at)
    text = _captions(at)
    assert "**Caution.** Heavily repeated flows decide much of this estimate." in text
    assert "none of them is among the held-out rows" in text and "standard errors cannot show" in text
    ranges = next(d.value for d in at.dataframe if "Bal. low" in d.value.columns)
    assert list(ranges.columns) == ["Channel", "Bal. accuracy", "Bal. low", "Bal. high", "Accuracy", "Acc. low",
                                    "Acc. high"]
    traffic = evaluate.cached_traffic_readings(run)  # type: ignore[arg-type]
    assert traffic is not None
    for _, row in ranges.iterrows():
        key = next(k for k in traffic if evaluate.channel_label(k) == row["Channel"])
        assert (row["Bal. low"], row["Bal. high"]) == pytest.approx(traffic[key].bounds["balanced_accuracy"], abs=1e-4)
