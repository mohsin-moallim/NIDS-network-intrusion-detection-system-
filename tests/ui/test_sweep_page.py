"""Headless checks of the 06 Sweep station: prerequisites, stepping, Start/Pause/Reset, Apply, the replay stream,
a real-data run loaded without its held-out rows, and that nothing here ever fits a model."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from graticule import persist
from graticule.data.prepare import DataRequest, prepare_dataset
from graticule.models import train
from graticule.models.train import TrainRequest, build_training_data, train_all
from graticule.settings import AppSettings
from tests.helpers import make_rows, write_cic_csv
from tests.ui.harness import app_with_run, errors, fit_synthetic, fresh_caches, goto, new_app  # noqa: F401
from ui import state
from ui.pages import sweep

pytestmark = pytest.mark.ui
MONDAY = "Monday-WorkingHours.pcap_ISCX.csv"
_FITS: dict[tuple[str, ...], tuple[object, object]] = {}


def _fits() -> int:
    return sum(train.FIT_CALLS.values())


def _text(at: AppTest) -> str:
    """Every text the page shows: markdown, captions and notices."""
    parts = [m.value for m in at.markdown] + [c.value for c in at.caption]
    for kind in ("info", "success", "warning", "error"):
        parts += [n.value for n in getattr(at, kind)]
    return " ".join(parts)


def _session(at: AppTest) -> object | None:
    return at.session_state[sweep.SESSION_KEY] if sweep.SESSION_KEY in at.session_state else None


def _running(at: AppTest) -> bool:
    return bool(at.session_state[sweep.RUNNING_KEY]) if sweep.RUNNING_KEY in at.session_state else False


def _cards(at: AppTest) -> str:
    return " ".join(m.value for m in at.markdown if "g-card" in m.value)


def _fitted_app(channels: list[str] | None = None) -> AppTest:
    """A session holding a small synthetic sample and a fit of ``channels`` (fitted once per module and channel set,
    as 01 Sample and 02 Fit would make them), at 02 Fit."""
    key = tuple(channels or ["xgboost", "logreg"])
    if key not in _FITS:
        _FITS[key] = fit_synthetic(channels=key)
    prepared, run = _FITS[key]
    at = app_with_run(run, prepared, key="fit").run()
    assert not errors(at), errors(at)
    return at


def test_without_a_fit_the_station_points_to_02_fit(fresh_caches: None) -> None:
    at = new_app("sweep").run()
    assert not errors(at), errors(at)
    assert "Needs a fitted channel" in _text(at)
    assert "Go to 02 Fit" in [link.proto.label for link in at.get("page_link")]
    assert "sw_start" not in [b.key for b in at.button]


def _spy_on_fragments(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, object]]:
    """Record (function name, run_every) for every fragment the Sweep page defines (the real fragment still runs)."""
    calls: list[tuple[str, object]] = []
    real = st.fragment

    def spy(func: object = None, *, run_every: object = None, **kwargs: object) -> object:
        if func is not None and getattr(func, "__module__", "") == sweep.__name__:
            calls.append((getattr(func, "__name__", "?"), run_every))
        return real(func, run_every=run_every, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(st, "fragment", spy)
    return calls


def test_step_start_pause_reset_and_apply_never_fit(fresh_caches: None, monkeypatch: pytest.MonkeyPatch) -> None:
    at = _fitted_app()
    run = at.session_state[state.RUN]
    fits = _fits()
    fragments = _spy_on_fragments(monkeypatch)
    goto(at, "sweep")
    assert not errors(at), errors(at)
    assert "Synthetic stream" in _text(at)
    assert _session(at) is None and "No stream yet" in _text(at)
    pace = int(at.number_input(key="sw_pace").value)

    # Step once builds the stream and scores one tick.
    at.button(key="sw_step").click().run()
    assert not errors(at), errors(at)
    session = _session(at)
    assert session is not None
    assert session.stats.emitted == pace and session.stats.ticks == 1  # type: ignore[attr-defined]
    assert "Flows seen" in _cards(at) and f">{pace}<" in _cards(at) and "Live balanced accuracy" in _cards(at)
    feed = [d.value for d in at.dataframe if "Verdict" in d.value.columns and "Tick" in d.value.columns]
    assert feed and len(feed[0]) == pace
    assert set(feed[0]["Reading"]) <= {"Correct", "Incorrect"}
    assert all(v.startswith(("○ ", "◆ ")) for v in feed[0]["Verdict"])
    titles = [json.loads(c.proto.spec).get("title", {}).get("text") for c in at.get("vega_lite_chart")]
    assert "Detections over time" in titles
    assert fragments[-1] == ("_live_panel", None)  # paused: the live panel does not refresh by itself

    # Start runs the stream: the redraw that follows the click takes its first tick.
    at.button(key="sw_start").click().run()
    assert not errors(at), errors(at)
    assert _running(at) and session.stats.ticks == 2  # type: ignore[attr-defined]
    # Running: only the live panel refreshes, once per tick interval.
    assert fragments[-1] == ("_live_panel", float(at.select_slider(key="sw_interval").value))
    assert at.button(key="sw_start").disabled and not at.button(key="sw_pause").disabled
    assert at.button(key="sw_step").disabled

    # Pause keeps every reading; nothing moves while paused.
    at.button(key="sw_pause").click().run()
    assert not errors(at) and not _running(at)
    assert fragments[-1] == ("_live_panel", None)
    before = session.stats.emitted  # type: ignore[attr-defined]
    at.run()
    assert session.stats.emitted == before and "Paused" in _text(at)  # type: ignore[attr-defined]

    # Pace changes wait for Apply, then apply in place (the stream carries on).
    at.number_input(key="sw_pace").set_value(pace + 5).run()
    assert not at.button(key="sw_apply").disabled and "wait for Apply" in _text(at)
    at.button(key="sw_step").click().run()
    assert session.stats.emitted == before + pace  # type: ignore[attr-defined]  # still the applied pace
    at.button(key="sw_apply").click().run()
    assert _session(at) is session and at.session_state[sweep.CONFIG_KEY].pace == pace + 5
    at.button(key="sw_step").click().run()
    assert session.stats.emitted == before + 2 * pace + 5  # type: ignore[attr-defined]

    # A new seed changes the stream itself: Apply starts a fresh one.
    at.number_input(key="sw_seed").set_value(int(at.number_input(key="sw_seed").value) + 1).run()
    at.button(key="sw_apply").click().run()
    fresh = _session(at)
    assert fresh is not session and fresh.stats.emitted == 0  # type: ignore[attr-defined]

    # Reset clears the stream.
    at.button(key="sw_step").click().run()
    at.button(key="sw_reset").click().run()
    assert not errors(at) and _session(at) is None and not _running(at)
    assert "cleared" in _text(at)

    # Every other widget redraws without fitting or stepping anything.
    at.button(key="sw_step").click().run()
    stepped = _session(at).stats.emitted  # type: ignore[union-attr]
    at.radio(key=f"sw_share_mode-{run.run_id}-synthetic").set_value(sweep.SHARE_OPTIONS[1]).run()
    at.slider(key=f"sw_share-{run.run_id}-synthetic").set_value(50).run()
    at.radio(key=f"sw_mix_mode-{run.run_id}-synthetic").set_value(sweep.MIX_OPTIONS[1]).run()
    assert not errors(at), errors(at)
    assert _session(at).stats.emitted == stepped  # type: ignore[union-attr]
    assert _fits() == fits


def test_settings_survive_a_visit_to_another_station(fresh_caches: None) -> None:
    """Streamlit forgets widgets that are not drawn; the page keeps a copy, so coming back shows the settings the
    stream runs with, Apply is not pending, and the stream carries on (pressing Apply would have restarted it)."""
    at = _fitted_app(["logreg"])
    run = at.session_state[state.RUN]
    goto(at, "sweep")
    at.number_input(key="sw_pace").set_value(60)
    at.number_input(key="sw_seed").set_value(9)
    at.number_input(key="sw_threshold").set_value(0.75)
    at.radio(key=f"sw_share_mode-{run.run_id}-synthetic").set_value(sweep.SHARE_OPTIONS[1]).run()
    at.slider(key=f"sw_share-{run.run_id}-synthetic").set_value(40)
    at.button(key="sw_step").click().run()
    assert not errors(at), errors(at)
    session = _session(at)
    applied = at.session_state[sweep.CONFIG_KEY]
    assert (applied.pace, applied.seed, applied.threshold, applied.share) == (60, 9, 0.75, 0.4)
    assert at.button(key="sw_apply").disabled and "06 Sweep ✓" in [l.proto.label for l in at.get("page_link")]
    goto(at, "bench")
    goto(at, "sweep")
    assert not errors(at), errors(at)
    assert int(at.number_input(key="sw_pace").value) == 60 and int(at.number_input(key="sw_seed").value) == 9
    assert at.number_input(key="sw_threshold").value == pytest.approx(0.75)
    assert at.radio(key=f"sw_share_mode-{run.run_id}-synthetic").value == sweep.SHARE_OPTIONS[1]
    assert at.slider(key=f"sw_share-{run.run_id}-synthetic").value == 40
    assert at.button(key="sw_apply").disabled and "wait for Apply" not in _text(at)
    at.button(key="sw_step").click().run()
    assert _session(at) is session and session.stats.emitted == 120  # type: ignore[attr-defined]


def test_replay_stream_of_a_synthetic_run_replays_held_out_rows(fresh_caches: None) -> None:
    at = _fitted_app()
    run = at.session_state[state.RUN]
    goto(at, "sweep")
    at.radio(key=f"sw_stream-{run.run_id}").set_value("replay").run()
    assert "Replaying held-out synthetic rows" in _text(at)
    at.radio(key=f"sw_share_mode-{run.run_id}-replay").set_value(sweep.SHARE_OPTIONS[1]).run()
    at.slider(key=f"sw_share-{run.run_id}-replay").set_value(80)
    at.number_input(key="sw_pace").set_value(500)
    at.button(key="sw_step").click().run()  # the pace typed just before the click is the one used
    at.button(key="sw_step").click().run()
    assert not errors(at), errors(at)
    session = _session(at)
    log = session.log_frame()  # type: ignore[union-attr]
    assert len(log) == 1_000 and set(log["row_id"].astype(int)) <= set(run.data.test_rows.tolist())
    assert abs(float((log["true_label"] == "Attack").mean()) - 0.8) <= 0.002  # exact in every 1,000 flows
    assert session.source.kind == "replay"  # type: ignore[union-attr]


def test_custom_mix_without_a_type_cannot_start(fresh_caches: None) -> None:
    at = _fitted_app(["logreg"])
    run = at.session_state[state.RUN]
    goto(at, "sweep")
    base = f"{run.run_id}-synthetic"
    at.radio(key=f"sw_mix_mode-{base}").set_value(sweep.MIX_OPTIONS[1]).run()
    at.multiselect(key=f"sw_mix_types-{base}").set_value([]).run()
    assert not errors(at), errors(at)
    assert "Choose at least one attack type" in _text(at)
    assert at.button(key="sw_start").disabled and at.button(key="sw_step").disabled


def test_real_data_run_without_held_out_rows_is_explained(fresh_caches: None) -> None:
    at = _fitted_app(["logreg"])
    run = at.session_state[state.RUN]
    data = run.data
    empty = replace(data, X_test=np.empty((0, data.n_features), dtype=np.float32), y_test=np.empty(0, dtype=np.int64),
                    test_rows=np.empty(0, dtype=np.int64), detailed_test_labels=np.empty(0, dtype=str))
    real = replace(run.data_request, source="cicids", files=("Monday-WorkingHours.pcap_ISCX.csv",))
    hollow = replace(run, run_id=run.run_id + "-h", data=empty, data_request=real, origin="loaded", bundle_path=None)
    state.run_registry().put(hollow.run_id, hollow)
    at.session_state[state.LAST_RUN_ID] = hollow.run_id
    at.session_state[state.RUN] = hollow
    fits = _fits()
    goto(at, "sweep")
    assert not errors(at), errors(at)
    text = _text(at)
    assert "without its held-out rows" in text and "never generated ones" in text
    links = [link.proto.label for link in at.get("page_link")]
    assert "Go to Logbook" in links and "Go to Bench" in links
    assert "sw_start" not in [b.key for b in at.button] and not at.get("vega_lite_chart")
    assert _fits() == fits


def test_a_new_run_clears_the_stream_of_the_old_one(fresh_caches: None) -> None:
    at = _fitted_app(["logreg"])
    run = at.session_state[state.RUN]
    goto(at, "sweep")
    at.button(key="sw_start").click().run()
    assert _session(at) is not None and _running(at)
    other = replace(run, run_id=run.run_id + "-b")
    state.run_registry().put(other.run_id, other)
    at.session_state[state.LAST_RUN_ID] = other.run_id
    at.session_state[state.RUN] = other
    at.run()
    assert not errors(at), errors(at)
    assert _session(at) is None and not _running(at)
    assert f"stream of run {run.run_id} was cleared" in _text(at)


def test_a_loaded_real_data_run_is_rebuilt_here_and_replays_real_rows(fresh_caches: None, tmp_path: Path) -> None:
    """A CIC-IDS2017-style run (tiny made-up files in ``tmp_path``) saved, then loaded while its data folder had
    moved: the station explains, the rebuild button reloads the set with the Bench's folder (nothing is fitted),
    and the stream then replays the rebuilt held-out rows."""
    folder = tmp_path / "data"
    rows = make_rows({"BENIGN": 160, "DoS Hulk": 90})
    write_cic_csv(folder / MONDAY, rows)
    prepared = prepare_dataset(DataRequest(source="cicids", data_dir=str(folder), files=(MONDAY,), row_budget=5_000,
                                           seed=3))
    request = TrainRequest(profile="test", seed=3, channels=("xgboost", "logreg"))
    run = train_all(build_training_data(prepared, request), request, data_request=prepared.request,
                    dataset_fingerprint=prepared.fingerprint)
    saved = persist.save_run(run)
    moved = folder.rename(tmp_path / "moved")
    bundle = persist.load_bundle(saved)
    loaded = persist.restore_run(bundle, None)
    assert loaded.origin == "loaded" and not loaded.has_test_rows
    fits = _fits()

    at = new_app("sweep")
    state.run_registry().put(loaded.run_id, loaded)
    at.session_state[state.LAST_RUN_ID] = loaded.run_id
    at.session_state[state.RUN] = loaded
    at.run()
    assert not errors(at), errors(at)
    assert "without its held-out rows" in _text(at) and "sw_start" not in [b.key for b in at.button]
    assert not at.button(key="sw_rebuild").disabled
    at.session_state[state.SETTINGS] = AppSettings(data_dir=str(moved))  # the Bench now points at the files
    at.button(key="sw_rebuild").click().run()
    assert not errors(at), errors(at)
    rebuilt = at.session_state[state.RUN]
    assert rebuilt.run_id == run.run_id and rebuilt.has_test_rows
    assert np.array_equal(rebuilt.data.test_rows, run.data.test_rows)
    assert "Held-out rows rebuilt" in _text(at)
    assert "Replaying held-out CIC-IDS2017 flows with their true labels" in _text(at)
    at.button(key="sw_step").click().run()
    assert not errors(at), errors(at)
    log = _session(at).log_frame()  # type: ignore[union-attr]
    assert len(log) == int(at.number_input(key="sw_pace").value)
    assert set(log["row_id"].astype(int)) <= set(run.data.test_rows.tolist())
    assert _fits() == fits
