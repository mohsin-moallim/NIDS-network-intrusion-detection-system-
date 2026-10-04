"""Headless checks of the 04 Probe station.

* Without a fitted run the station points to 02 Fit.
* After a tiny synthetic fit: every flow source works (a held-out row drawn within a class, picked by number or by
  "Draw another"; a typical flow per class; values typed into the editor, which scores the edited flow again), the
  verdict table and the consensus line read as specified, the explanation chart renders for both methods and for a
  channel without exact contributions, and readings are scored only when the flow or the choice changes. Nothing is
  ever fitted on this page (``FIT_CALLS`` stays put).
* Multi-class: the class picker of the explanation, the top classes and a single channel's class probabilities.
* A channel set saved and loaded back without its held-out rows: the held-out source explains itself and links to
  the Logbook; typical flows (the quantile median) and the reference swap (a quantile background) still work.

The editor (``st.data_editor``) has no setter in Streamlit's AppTest, so :func:`_edit_cell` sends the edit as the
browser would: a widget state carrying the editor's JSON edit record.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from streamlit.proto.WidgetStates_pb2 import WidgetState
from streamlit.testing.v1 import AppTest

from graticule import explain, persist
from graticule.models import train
from tests.ui.harness import app_with_run, errors, fit_synthetic, fresh_caches, goto, new_app  # noqa: F401
from ui import state
from ui.pages import probe

pytestmark = pytest.mark.ui
SECTIONS = ["Flow", "Verdict", "Explanation"]


def _fits() -> int:
    return sum(train.FIT_CALLS.values())


def _text(at: AppTest) -> str:
    return " ".join([m.value for m in at.markdown] + [c.value for c in at.caption])


def _titles(at: AppTest) -> list[str]:
    titles = []
    for chart in at.get("vega_lite_chart"):
        title = json.loads(chart.proto.spec).get("title")
        titles.append(title.get("text") if isinstance(title, dict) else str(title or ""))
    return titles


def _chart_rows(at: AppTest) -> pd.DataFrame:
    """The bars of the (only) contribution chart on the page, read from the data Streamlit sends with it."""
    from streamlit import dataframe_util

    charts = at.get("vega_lite_chart")
    assert len(charts) == 1
    frames = [dataframe_util.convert_arrow_bytes_to_pandas_df(d.data.data) for d in charts[0].proto.datasets]
    return next(frame for frame in frames if "label" in frame.columns)


def _verdicts(at: AppTest) -> pd.DataFrame:
    return next(d.value for d in at.dataframe if "P(verdict)" in d.value.columns)


def _editors(at: AppTest) -> list[Any]:
    return [d for d in at.dataframe if d.proto.editing_mode != d.proto.EditingMode.READ_ONLY]


def _edit_cell(at: AppTest, row: int, value: float | None) -> AppTest:
    """Type ``value`` into the Value cell of editor row ``row`` (as the browser reports an edit) and rerun."""
    editors = _editors(at)
    assert len(editors) == 1
    states = at._tree.get_widget_states()
    record = {"edited_rows": {str(row): {"Value": value}}, "added_rows": [], "deleted_rows": []}
    states.widgets.append(WidgetState(id=editors[0].proto.id, string_value=json.dumps(record)))
    return at._run(states)


def _fitted_app(mode: str = "binary", channels: list[str] | None = None) -> AppTest:
    """A session holding a small synthetic sample and a fit of it (fitted directly, as 02 Fit would), at 02 Fit."""
    options: dict[str, Any] = {"mode": mode}
    if channels is not None:
        options["channels"] = tuple(channels)
    prepared, run = fit_synthetic(**options)
    at = app_with_run(run, prepared, key="fit").run()
    assert not errors(at), errors(at)
    return at


def _verdict_calls(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Count the page's calls of the scoring and explaining functions of graticule.explain."""
    calls: list[int] = []
    for name in ("score_flow", "reference_swap", "xgboost_contributions"):
        original = getattr(explain, name)

        def counted(*args: Any, __original: Any = original, **kwargs: Any) -> Any:
            calls.append(1)
            return __original(*args, **kwargs)

        monkeypatch.setattr(explain, name, counted)
    return calls


def test_without_a_fit_the_station_points_to_02_fit(fresh_caches: None) -> None:
    at = new_app("probe").run()
    assert not errors(at), errors(at)
    assert "Needs a fitted channel" in _text(at)
    assert "Go to 02 Fit" in [link.proto.label for link in at.get("page_link")]
    assert not at.get("vega_lite_chart")


def test_binary_probe_sources_verdicts_and_explanations(fresh_caches: None, monkeypatch: pytest.MonkeyPatch,
                                                        tmp_path: Path) -> None:
    at = _fitted_app("binary")
    run = at.session_state[state.RUN]
    keys = run.ok_channels()
    fits = _fits()
    calls = _verdict_calls(monkeypatch)
    goto(at, "probe")
    assert not errors(at), errors(at)
    assert [h.value for h in at.subheader] == SECTIONS

    # (a) A held-out flow: verdicts of every channel plus the consensus, each checked against the true class.
    row_key = probe._key("g_probe_row", run)
    index = int(at.number_input(key=row_key).value)
    table = _verdicts(at)
    assert list(table["Channel"])[:-1] == [explain.channel_label(k) for k in keys]
    assert table["Channel"].iloc[-1] == f"Consensus ({len(keys)} channels)"
    assert {"Verdict", "P(verdict)", "P(attack)", "Alert", "Reading"} <= set(table.columns)
    assert set(table["Reading"]) <= {"Correct", "Incorrect"}
    assert set(table["Verdict"]) <= {"○ Normal", "◆ Attack"}
    for position, key in enumerate(keys):
        stored = run.channels[key].proba[index]
        assert table["P(attack)"].iloc[position] == pytest.approx(float(stored[1]), abs=1e-5)
    text = _text(at)
    assert "Consensus:" in text and "channels agree" in text
    assert "Held-out row <span" in " ".join(m.value for m in at.markdown)
    assert "generated flow" in text and "counting from 0" in text  # provenance of a synthetic row (0-based)
    # Reading a flow earns 04 Probe its tick, shown in the very run that read it.
    assert "04 Probe ✓" in [link.proto.label for link in at.get("page_link")]
    assert "CH2 XGBoost: what moved its reading" in _titles(at)
    assert "Exact, in log-odds of Attack" in text and "pushed CH2 towards" in text
    scored = len(calls)

    # A rerun without a change scores nothing again.
    at.run()
    assert not errors(at) and len(calls) == scored

    # "Draw another" moves to another row; a class filter keeps the draws within that class.
    at.button(key=probe._key("g_probe_draw", run)).click().run()
    assert not errors(at), errors(at)
    assert int(at.number_input(key=row_key).value) != index
    labels = list(map(str, run.data.detailed_test_labels))
    attack_class = next(name for name in labels if name != "BENIGN")
    at.selectbox(key=probe._key("g_probe_filter", run)).set_value(attack_class).run()
    for _ in range(2):
        assert labels[int(at.number_input(key=row_key).value)] == attack_class
        at.button(key=probe._key("g_probe_draw", run)).click().run()
    at.number_input(key=row_key).set_value(7).run()
    assert not errors(at), errors(at)
    assert int(at.number_input(key=row_key).value) == 7
    assert '<span class="g-mono">7</span>' in " ".join(m.value for m in at.markdown)
    # Looking at another source keeps the held-out choices.
    at.radio(key=probe.SOURCE).set_value("typical").run()
    at.radio(key=probe.SOURCE).set_value("held_out").run()
    assert not errors(at), errors(at)
    assert int(at.number_input(key=row_key).value) == 7
    assert at.selectbox(key=probe._key("g_probe_filter", run)).value == attack_class

    # The explanation for both methods, then a channel that only has the reference swap.
    method_key = probe._key("g_probe_method_xgboost", run)
    at.radio(key=method_key).set_value("reference_swap").run()
    assert not errors(at), errors(at)
    assert "CH2 XGBoost: what moved its reading (approximate)" in _titles(at)
    assert "Approximate, in probability points of Attack" in _text(at)
    at.selectbox(key=probe._key("g_probe_xchannel", run)).set_value("logreg").run()
    assert not errors(at), errors(at)
    assert list(at.radio(key=probe._key("g_probe_method_logreg", run)).options) == [
        probe.METHOD_LABELS["reference_swap"]]
    assert "CH5 Logistic regression: what moved its reading (approximate)" in _titles(at)
    bars = _chart_rows(at)
    assert 0 < len(bars) <= explain.TOP_FEATURES and all(" = " in label for label in bars["label"])

    # One channel only: its row, no consensus line.
    at.selectbox(key=probe._key("g_probe_view", run)).set_value("forest").run()
    assert not errors(at), errors(at)
    assert list(_verdicts(at)["Channel"]) == ["CH1 Random forest"] and "Consensus:" not in _text(at)
    at.selectbox(key=probe._key("g_probe_view", run)).set_value(probe.ALL_CHANNELS).run()
    assert len(_verdicts(at)) == len(keys) + 1

    # (b) A typical flow of each class.
    at.radio(key=probe.SOURCE).set_value("typical").run()
    assert not errors(at), errors(at)
    for class_index in (1, 0):
        at.selectbox(key=probe._key("g_probe_typical", run)).set_value(class_index).run()
        assert not errors(at), errors(at)
        assert "a summary, not a recorded flow" in _text(at)
        assert "Reading" not in _verdicts(at).columns  # no true class to compare with

    # (c) Edited values: the editor starts from the last typical flow; an edit scores the new flow.
    at.radio(key=probe.SOURCE).set_value("edit").run()
    assert not errors(at), errors(at)
    assert len(_editors(at)) == 1 and "no value edited yet" in _text(at)
    editor = _editors(at)[0].value
    assert list(editor.columns) == ["Feature", "Value", "p1", "p99"] and len(editor) == len(run.data.feature_names)
    base, _ = explain.typical_flow(run, 0)
    assert editor["Value"].to_numpy(dtype="float32").tolist() == base.tolist()
    before = len(calls)
    _edit_cell(at, 2, 100000000.0)
    assert not errors(at), errors(at)
    assert "1 value edited" in _text(at) and run.data.feature_names[2] in _text(at)
    assert len(calls) > before
    changed = base.copy()
    changed[2] = 100000000.0
    assert ("verdict", *probe._run_tag(run), explain.row_digest(changed), None) in at.session_state[probe.CACHE]
    expected = explain.score_flow(run, changed, keys)
    shown = _verdicts(at)
    for position, key in enumerate(keys):
        assert shown["P(attack)"].iloc[position] == pytest.approx(expected.attack_probability(key), abs=1e-6)
    # Looking at another source and coming back keeps the edits (they start the editor again).
    at.radio(key=probe.SOURCE).set_value("typical").run()
    at.radio(key=probe.SOURCE).set_value("edit").run()
    assert not errors(at), errors(at)
    assert "Started from your earlier edits of typical flow" in _text(at)
    assert float(_editors(at)[0].value["Value"].iloc[2]) == 100000000.0
    assert ("verdict", *probe._run_tag(run), explain.row_digest(changed), None) in at.session_state[probe.CACHE]
    # A cleared cell is a missing value, which every channel reads like any other gap.
    _edit_cell(at, 0, None)
    assert not errors(at), errors(at)
    assert "1 value edited" in _text(at) and len(_verdicts(at)) == len(keys) + 1
    assert _verdicts(at)["P(verdict)"].notna().all()
    at.button(key=probe._key("g_probe_restart", run)).click().run()
    assert not errors(at), errors(at)
    assert "no value edited yet" in _text(at)
    assert _fits() == fits

    # The same channels saved and loaded back without their held-out rows.
    _check_loaded_without_rows(at, run, tmp_path)
    assert _fits() == fits


def _check_loaded_without_rows(at: AppTest, run: Any, folder: Path) -> None:
    """Save ``run``, load it back without any rows, make it current and check every source of 04 Probe."""
    path = persist.save_run(run, folder / "saved")
    loaded = persist.restore_run(persist.load_bundle(path))
    assert loaded.origin == "loaded" and not loaded.has_test_rows and "svm" not in loaded.ok_channels()
    assert loaded.run_id == run.run_id  # the copy shares the fit's id; nothing read for the fit may leak into it
    state.run_registry().put(loaded.run_id, loaded)
    at.session_state[state.LAST_RUN_ID] = loaded.run_id
    at.session_state[state.RUN] = loaded
    at.session_state[probe.SOURCE] = "held_out"
    goto(at, "probe")
    assert not errors(at), errors(at)
    assert "loaded from disk without its held-out rows" in _text(at)
    assert "Go to Logbook" in [link.proto.label for link in at.get("page_link")]
    assert not at.get("vega_lite_chart")
    at.radio(key=probe.SOURCE).set_value("typical").run()
    assert not errors(at), errors(at)
    assert "overall training median" in _text(at)
    assert probe._key("g_probe_typical", loaded) not in [s.key for s in at.selectbox]
    table = _verdicts(at)
    assert len(table) == len(loaded.ok_channels()) + 1
    assert table["Channel"].iloc[-1] == f"Consensus ({len(loaded.ok_channels())} channels)"
    at.selectbox(key=probe._key("g_probe_xchannel", loaded)).set_value("forest").run()
    assert not errors(at), errors(at)
    assert "synthetic flows read off the saved training quantiles" in _text(at)
    assert "CH1 Random forest: what moved its reading (approximate)" in _titles(at)
    at.radio(key=probe.SOURCE).set_value("edit").run()
    assert not errors(at) and len(_editors(at)) == 1


def test_multiclass_probe_explains_a_chosen_class(fresh_caches: None) -> None:
    at = _fitted_app("multiclass", channels=["forest", "xgboost"])
    run = at.session_state[state.RUN]
    fits = _fits()
    goto(at, "probe")
    assert not errors(at), errors(at)
    table = _verdicts(at)
    assert "Top classes" in table.columns and "P(attack)" not in table.columns
    assert all(text.count(" · ") == 2 for text in table["Top classes"])
    class_key = probe._key("g_probe_xclass", run)
    assert at.selectbox(key=class_key).value == probe.VERDICT_CLASS
    assert "Exact, in log-odds of" in _text(at)
    attack = next(i for i, name in enumerate(run.data.classes) if name != "BENIGN")
    at.selectbox(key=class_key).set_value(attack).run()
    assert not errors(at), errors(at)
    name = run.data.classes[attack]
    assert f"Exact, in log-odds of {name}" in _text(at)
    spec = json.loads(at.get("vega_lite_chart")[0].proto.spec)
    assert f"Towards {name}" in json.dumps(spec)
    at.selectbox(key=class_key).set_value(0).run()  # the normal class: "towards" is drawn as normal traffic
    assert not errors(at), errors(at)
    assert "Towards BENIGN (normal)" in json.dumps(json.loads(at.get("vega_lite_chart")[0].proto.spec))
    at.selectbox(key=probe._key("g_probe_view", run)).set_value("xgboost").run()
    assert not errors(at), errors(at)
    probabilities = next(d.value for d in at.dataframe if "Probability" in d.value.columns)
    assert len(probabilities) == len(run.data.classes)
    assert probabilities["Probability"].sum() == pytest.approx(1.0, abs=1e-4)
    assert _fits() == fits
