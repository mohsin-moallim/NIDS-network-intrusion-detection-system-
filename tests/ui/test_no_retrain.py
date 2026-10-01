"""Definition of done for 02 Fit: after one fit, no interaction anywhere refits a channel or prepares data again.

The test draws a small synthetic sample through 01 Sample, fits all five channels once through the 02 Fit form
(``profile="test"`` models) and checks the readings, including the CH3 row-cap badge. It then changes every widget
on the Fit page without submitting, visits every other station and changes each of its widgets in turn (rerunning
after every change), and comes back to 02 Fit. Throughout, ``graticule.models.train.FIT_CALLS`` must not move,
neither the training matrices nor a training job may be built again, ``prepare_dataset`` must not run again, and
the stored run and every fitted estimator must stay the very same objects. Widgets are found generically, so
stations built in later phases are covered as soon as they exist; 03 Measure, 04 Probe, 05 Assay, 06 Sweep and the
Logbook are checked by name.

Buttons are explicit actions and are not pressed while widgets are changed. Afterwards the actions of 04 Probe to
07 Record are pressed, since none of them may fit anything either: a new held-out flow is drawn at 04 Probe, a file
is scored at 05 Assay, the simulation is stepped, started (one live tick) and paused at 06 Sweep, and 07 Record
then offers their exports (its PDF build is checked by the Record and loaded-run tests).
Save at the Logbook is pressed at the end, because it must not fit anything either (the run stays the very same
object); a second saved set then gives the Logbook's picker something to change to. Loading a saved set and working
03 Measure on the loaded run are covered by ``tests/ui/test_saved_runs_measure.py``; the stations 04 to 07 on loaded
runs by ``tests/ui/test_loaded_runs_stations.py``.
"""

from __future__ import annotations

import io
from collections import Counter
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image
from streamlit.testing.v1 import AppTest

import graticule.settings as settings_mod
from graticule import persist, viz
from graticule.data import prepare
from graticule.models import train
from graticule.models.zoo import MODEL_KEYS
from graticule.schema import FEATURES, LABEL
from tests.helpers import write_cic_csv
from tests.ui.harness import (  # noqa: F401
    different_value,
    draw_synthetic_sample,
    errors,
    fresh_caches,
    goto,
    new_app,
    touch_every_widget,
)
from ui import state, training_ui
from ui.pages import assay, probe, sweep
from ui.stations import ALL_STATIONS

pytestmark = pytest.mark.ui
# What one press of Fit builds: the matrices once, the fit loop once, one job.
ONE_FIT = Counter(build_training_data=1, train_all=1, TrainingJob=1)


@pytest.fixture
def prepare_calls(monkeypatch: pytest.MonkeyPatch) -> list[prepare.DataRequest]:
    """Record every call of prepare_dataset (the real function still runs)."""
    calls: list[prepare.DataRequest] = []
    real = prepare.prepare_dataset

    def spy(request: prepare.DataRequest, **kwargs: object) -> prepare.PreparedDataset:
        calls.append(request)
        return real(request, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(prepare, "prepare_dataset", spy)
    return calls


@pytest.fixture
def training_calls(monkeypatch: pytest.MonkeyPatch) -> Counter[str]:
    """Count the expensive training steps besides the channel fits: the matrices, the fit loop and job creation.

    The job looks ``build_training_data`` and ``train_all`` up on the module when it runs, so wrapping the module
    attributes sees every call.
    """
    calls: Counter[str] = Counter()

    def counted(name: str, real: Callable[..., object]) -> Callable[..., object]:
        def wrapper(*args: object, **kwargs: object) -> object:
            calls[name] += 1
            return real(*args, **kwargs)

        return wrapper

    monkeypatch.setattr(train, "build_training_data", counted("build_training_data", train.build_training_data))
    monkeypatch.setattr(train, "train_all", counted("train_all", train.train_all))
    monkeypatch.setattr(training_ui, "TrainingJob", counted("TrainingJob", training_ui.TrainingJob))
    return calls


@pytest.fixture
def quick_charts(monkeypatch: pytest.MonkeyPatch) -> None:
    """Charts of the PDF record are built (and their specs checked) but not rendered."""

    def blank_png(chart: Any, scale: float = 2, *, background: str | None = None) -> bytes:
        chart.to_dict(validate=False)  # the specs are checked against the schema by tests/unit/test_report_pdf.py
        buffer = io.BytesIO()
        Image.new("RGB", (600, 300), "#FFFFFF").save(buffer, format="PNG")
        return buffer.getvalue()

    monkeypatch.setattr(viz, "to_png", blank_png)


def _markdown(at: AppTest) -> str:
    return " ".join(m.value for m in at.markdown)


def _readings(at: AppTest):  # noqa: ANN202 - a pandas frame from the element tree
    return next(d.value for d in at.dataframe if "Balanced accuracy" in d.value.columns)


def _keys(touched: list[str]) -> set[str]:
    return {label.split(":", 1)[1] for label in touched}


def test_no_widget_anywhere_refits_or_reprepares(fresh_caches: None, prepare_calls: list[prepare.DataRequest],
                                                 training_calls: Counter[str], quick_charts: None,
                                                 tmp_path: Path) -> None:
    at = new_app().run()
    # 4,000 generated flows sampled down to 2,000 rows leave about 1,500 training rows: more than the test-profile
    # SVM cap of 1,000, so CH3 is visibly capped.
    draw_synthetic_sample(at, flows=4_000, budget=2_000)
    assert not errors(at), errors(at)
    assert len(prepare_calls) == 1
    goto(at, "fit")
    fits_before = sum(train.FIT_CALLS.values())
    at.button(key="fit_submit").click().run()
    assert not errors(at), errors(at)

    # One fit of each of the five channels, one set of matrices, one job, and every reading on the page.
    assert sum(train.FIT_CALLS.values()) - fits_before == len(MODEL_KEYS)
    assert training_calls == ONE_FIT
    run_id = at.session_state[state.LAST_RUN_ID]
    run = at.session_state[state.RUN]
    run_identity = id(run)
    estimators = {key: id(result.estimator) for key, result in run.channels.items()}
    fits = sum(train.FIT_CALLS.values())
    assert run.request.profile == "test" and run.ok_channels() == list(MODEL_KEYS)
    readings = _readings(at)
    assert len(readings) == len(MODEL_KEYS) and (readings["Status"] == "fitted").all()
    svm = run.channels["svm"]
    assert svm.rows_used == svm.extra["svm_cap"] == 1_000 < svm.rows_available
    badge = f"CH3 trained on 1,000 of {svm.rows_available:,} rows (SVM cap 1,000)"
    assert badge in _markdown(at)

    def unchanged(label: str) -> None:
        assert sum(train.FIT_CALLS.values()) == fits, f"{label} caused a refit"
        assert training_calls == ONE_FIT, f"{label} rebuilt the training data or started a job"
        assert len(prepare_calls) == 1, f"{label} prepared the data again"
        assert at.session_state[state.LAST_RUN_ID] == run_id, f"{label} replaced the run"
        current = at.session_state[state.RUN]
        assert id(current) == run_identity, f"{label} replaced the run object"
        assert {k: id(r.estimator) for k, r in current.channels.items()} == estimators, f"{label} swapped a model"
        assert state.JOB_ID not in at.session_state, f"{label} started a fit job"

    # 1. Every widget of the Fit form, changed without pressing Fit.
    fit_keys = _keys(touch_every_widget(at, unchanged))
    assert {"fit_mode", "fit_features", "fit_k", "fit_port", "fit_channels", "fit_balanced", "fit_min_rows",
            "fit_svm_cap", "fit_test_share"} <= fit_keys

    # 2. Every other station, every widget on it.
    touched_elsewhere: dict[str, list[str]] = {}
    for station in ALL_STATIONS:
        if station.key == "fit":
            continue
        goto(at, station.key)
        assert not errors(at), (station.key, errors(at))
        unchanged(f"visiting {station.key}")
        touched_elsewhere[station.key] = touch_every_widget(at, unchanged)
    assert touched_elsewhere["sample"] and touched_elsewhere["bench"]
    # 03 Measure: the confusion and ROC views, the detail channel, the folds and channels of cross-validation.
    assert {"ms_cm_show", "ms_roc_zoom", f"ms_detail_channel-{run_id}", "ms_cv_k",
            f"ms_cv_channels-{run_id}"} <= _keys(touched_elsewhere["measure"])
    # 04 Probe: the flow source (which hides the held-out pickers, changed below), the channels shown, the
    # explanation's channel and method.
    assert {probe.SOURCE, probe._key("g_probe_view", run), probe._key("g_probe_xchannel", run),
            probe._key("g_probe_method_xgboost", run)} <= _keys(touched_elsewhere["probe"])
    # 05 Assay: the channel (or the consensus) and the alert threshold.
    assert {assay.channel_key(run), assay.THRESHOLD} <= _keys(touched_elsewhere["assay"])
    # 06 Sweep: channel, stream, pace, interval, threshold, seed, attack share and mix (of the stream chosen).
    sweep_keys = _keys(touched_elsewhere["sweep"])
    assert {f"sw_channel-{run_id}", f"sw_stream-{run_id}", "sw_pace", "sw_interval", "sw_threshold",
            "sw_seed"} <= sweep_keys
    for stem in ("sw_share_mode", "sw_share", "sw_mix_mode"):
        assert any(key.startswith(f"{stem}-{run_id}-") for key in sweep_keys), stem

    # 2b. The actions of 04 Probe to 07 Record: they read, score, stream and lay out, and never fit.
    goto(at, "probe")
    at.radio(key=probe.SOURCE).set_value("held_out").run()
    row_key = probe._key("g_probe_row", run)
    filter_key = probe._key("g_probe_filter", run)
    at.selectbox(key=filter_key).set_value(different_value(at.selectbox(key=filter_key), "selectbox")).run()
    unchanged("choosing the class of held-out flows at probe")
    at.number_input(key=row_key).set_value(different_value(at.number_input(key=row_key), "number_input")).run()
    unchanged("choosing a held-out row at probe")
    before = int(at.number_input(key=row_key).value)
    at.button(key=probe._key("g_probe_draw", run)).click().run()
    assert not errors(at), errors(at)
    assert int(at.number_input(key=row_key).value) != before
    unchanged("drawing another flow at probe")

    held_out = at.session_state[state.PREPARED].frame.iloc[run.data.test_rows[:60]]
    values = held_out[list(FEATURES)].to_numpy(dtype=np.float64)
    rows = [{**dict(zip(FEATURES, (float(v) for v in row))), LABEL: label}
            for row, label in zip(values, held_out[LABEL].astype(str))]
    upload = write_cic_csv(tmp_path / "upload.csv", rows).read_bytes()  # type: ignore[arg-type]
    goto(at, "assay")
    at.file_uploader(key=assay.UPLOAD).set_value(("upload.csv", upload, "text/csv")).run()
    at.button(key=assay.SCORE).click().run()
    assert not errors(at), errors(at)
    batch = at.session_state[state.LAST_ASSAY]
    assert batch.rows == 60
    unchanged("scoring a file at assay")

    goto(at, "sweep")
    at.button(key="sw_step").click().run()
    assert not errors(at), errors(at)
    stream = at.session_state[sweep.SESSION_KEY]
    assert stream.stats.ticks == 1
    at.button(key="sw_start").click().run()  # running: the live panel takes a tick with this redraw
    assert not errors(at), errors(at)
    assert at.session_state[sweep.RUNNING_KEY] and stream.stats.ticks == 2
    at.button(key="sw_pause").click().run()
    assert not errors(at), errors(at)
    assert not at.session_state[sweep.RUNNING_KEY]
    assert stream.stats.emitted == 2 * int(at.number_input(key="sw_pace").value)
    unchanged("stepping the simulation at sweep")
    # (Reset and later steps are covered by tests/ui/test_sweep_page.py; the record below needs this stream.)

    goto(at, "record")
    assert not errors(at), errors(at)
    assert {"rec_dl_assay", "rec_dl_sweep"} <= {b.key for b in at.get("download_button")}
    unchanged("offering the exports at record")
    # (Building the PDF after real Assay and Sweep work, with FIT_CALLS unchanged, is covered by
    # tests/ui/test_loaded_runs_stations.py and tests/ui/test_record_page.py.)

    # 3. Back at 02 Fit: the same run, read from memory, with the same readings and the CH3 badge.
    goto(at, "fit")
    assert not errors(at), errors(at)
    unchanged("returning to fit")
    assert _readings(at).equals(readings)
    assert badge in _markdown(at)
    assert "nothing refits until you press Fit again" in _markdown(at)
    assert id(state.run_registry().get(run_id)) == run_identity

    # 4. The Logbook with saved sets. Save writes the run (the very same object stays current); a second saved set
    # (a copy of the run under another id) gives the picker something to change to.
    goto(at, "logbook")
    at.button(key="lb_save").click().run()
    assert not errors(at), errors(at)
    unchanged("saving at the Logbook")
    assert (settings_mod.MODELS_DIR / run_id / "manifest.json").is_file()
    persist.save_run(replace(run, run_id=f"{run_id}-copy", bundle_path=None))
    goto(at, "logbook")
    assert "selectbox:lb_pick" in touch_every_widget(at, unchanged)
    # Loading a saved set and every Measure control on the loaded run: tests/ui/test_saved_runs_measure.py.
