"""04 Probe, 05 Assay, 06 Sweep and 07 Record on one run in each of its three forms, in one headless session.

1. As fitted in memory (all five channels): a generated file is scored and the stream takes a tick, both with
   CH3 (04 Probe and 07 Record on a fit have their own tests, and ``test_no_retrain.py`` builds a record after
   real Assay and Sweep work).
2. Saved, then loaded again at the Logbook with its held-out rows rebuilt (the synthetic sample is regenerated):
   the loaded copy has no CH3, so every channel picker offers the four saved channels; the stream of the fit is
   cleared (the copy shares the fit's id but streams afresh); every station works on the copy.
3. The same set loaded without any rows (as when its data cannot be rebuilt): the Probe falls back to the typical
   flow and a quantile background, the Assay still scores files, the Sweep streams fresh generator flows (only a
   synthetic run may), and the Record builds the shorter record of the readings saved with the run (in the
   multi-class test, marked slow; each station's own quick tests cover a run without rows too).

Throughout, nothing is fitted (``FIT_CALLS`` stays put). Charts are built but not rendered (see
``tests/unit/test_report_pdf.py`` for real renders). The run is fitted directly (as 02 Fit fits it; 02 Fit has its
own tests) and handed to the session; saving and loading go through the Logbook's buttons. The multi-class form of
the same path is checked on the loaded copies only (``test_a_multiclass_run_loaded_with_and_without_rows``).

Both tests walk the whole path and are marked ``slow`` (``-m slow`` runs them). The default suite keeps the quick
per-station checks of the same cases: a copy loaded without rows at 04 Probe, 06 Sweep and 07 Record, the copy's
scoring and the fit/copy separation in ``tests/unit/test_scoring.py`` and ``tests/ui/test_assay_page.py``, and the
Logbook's own save and load tests.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from PIL import Image
from streamlit.testing.v1 import AppTest

import graticule.settings as settings_mod
from graticule import evaluate, explain, persist, scoring, viz
from graticule.models import train
from graticule.models.train import TrainingRun
from graticule.report import exports
from graticule.schema import FEATURES, LABEL
from graticule.scoring import CONSENSUS, ScoredBatch
from tests.helpers import write_cic_csv
from tests.ui.harness import app_with_run, errors, fit_synthetic, fresh_caches, goto  # noqa: F401
from ui import state
from ui.pages import assay, probe, record, sweep
from ui.training_ui import channel_label

pytestmark = pytest.mark.ui
#: Rows of the generated file scored at 05 Assay.
UPLOAD_ROWS = 120
#: Every channel a saved set keeps by default (CH3 only when saved by choice: its model is made of training rows).
SAVED_CHANNELS = ["forest", "xgboost", "mlp", "logreg"]


def _fits() -> int:
    return sum(train.FIT_CALLS.values())


def _text(at: AppTest) -> str:
    parts = [m.value for m in at.markdown] + [c.value for c in at.caption]
    parts += [a.value for kind in ("success", "info", "warning", "error") for a in getattr(at, kind)]
    return " ".join(parts)


def _titles(at: AppTest) -> list[str]:
    titles = []
    for chart in at.get("vega_lite_chart"):
        title = json.loads(chart.proto.spec).get("title")
        titles.append(title.get("text") if isinstance(title, dict) else str(title or ""))
    return titles


def _downloads(at: AppTest) -> list[str]:
    return [b.key for b in at.get("download_button")]


def _blank_png(chart: Any, scale: float = 2, *, background: str | None = None) -> bytes:
    """Stand-in for :func:`graticule.viz.to_png`: the chart is serialised (its spec is checked against the schema by
    ``tests/unit/test_report_pdf.py``), a blank PNG comes back."""
    chart.to_dict(validate=False)
    buffer = io.BytesIO()
    Image.new("RGB", (600, 300), "#FFFFFF").save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture
def quick_charts(monkeypatch: pytest.MonkeyPatch) -> None:
    """Charts are built (and validated) but not rendered."""
    monkeypatch.setattr(viz, "to_png", _blank_png)


def _upload_bytes(at: AppTest, run: TrainingRun, folder: Path) -> bytes:
    """A labelled CIC-style file of held-out flows of the session's sample (written to the test's folder)."""
    prepared = at.session_state[state.PREPARED]
    frame = prepared.frame.iloc[run.data.test_rows[:UPLOAD_ROWS]]
    values = frame[list(FEATURES)].to_numpy(dtype=np.float64)
    rows = [{**dict(zip(FEATURES, (float(v) for v in row))), LABEL: label}
            for row, label in zip(values, frame[LABEL].astype(str))]
    return write_cic_csv(folder / "upload.csv", rows, bom=True).read_bytes()  # type: ignore[arg-type]


def _make_current(at: AppTest, run: TrainingRun) -> None:
    """Make ``run`` the session's current run, as the Logbook does after a load."""
    state.run_registry().put(run.run_id, run)
    at.session_state[state.LAST_RUN_ID] = run.run_id
    at.session_state[state.RUN] = run


# --------------------------------------------------------------------------------------------------------------
# One visit per station
# --------------------------------------------------------------------------------------------------------------
def _probe(at: AppTest, run: TrainingRun) -> None:
    """04 Probe: the verdicts of every fitted channel plus the consensus, and an explanation of the last channel."""
    goto(at, "probe")
    assert not errors(at), errors(at)
    keys = run.ok_channels()
    if not run.has_test_rows:
        if at.radio(key=probe.SOURCE).value == "held_out":
            assert "without its held-out rows" in _text(at)
        at.radio(key=probe.SOURCE).set_value("typical").run()
        assert not errors(at), errors(at)
    else:
        assert at.radio(key=probe.SOURCE).value == "held_out"
    table = next(d.value for d in at.dataframe if "P(verdict)" in d.value.columns)
    assert list(table["Channel"])[:-1] == [explain.channel_label(k) for k in keys]
    assert table["Channel"].iloc[-1] == f"Consensus ({len(keys)} channels)"
    assert ("Reading" in table.columns) == run.has_test_rows  # correct/incorrect only for a held-out flow
    picker = at.selectbox(key=probe._key("g_probe_xchannel", run))
    assert list(picker.options) == [explain.channel_label(k) for k in keys]
    picker.set_value(keys[-1]).run()
    assert not errors(at), errors(at)
    assert f"{explain.channel_label(keys[-1])}: what moved its reading (approximate)" in _titles(at)


def _assay(at: AppTest, run: TrainingRun, upload: bytes, channel: str) -> ScoredBatch:
    """05 Assay: score the generated file with ``channel``; returns the batch kept for 07 Record."""
    goto(at, "assay")
    assert not errors(at), errors(at)
    picker = at.selectbox(key=assay.channel_key(run))
    assert list(picker.options) == [scoring.channel_name(k) for k in scoring.channel_choices(run)]
    at.file_uploader(key=assay.UPLOAD).set_value(("upload.csv", upload, "text/csv")).run()
    at.selectbox(key=assay.channel_key(run)).set_value(channel)
    at.button(key=assay.SCORE).click().run()
    assert not errors(at), errors(at)
    batch = at.session_state[state.LAST_ASSAY]
    assert isinstance(batch, ScoredBatch)
    assert batch.run_id == run.run_id and batch.channel == channel and batch.rows == UPLOAD_ROWS and batch.labelled
    voters = run.ok_channels() if channel == CONSENSUS else [channel]
    assert list(batch.voters) == voters
    assert batch.accuracy is not None and 0.0 <= batch.accuracy <= 1.0
    assert "as_download" in _downloads(at)
    return batch


def _sweep(at: AppTest, run: TrainingRun, channel: str, stream: str) -> Any:
    """06 Sweep: one tick of ``stream`` through ``channel``; returns the simulation session."""
    goto(at, "sweep")
    assert not errors(at), errors(at)
    picker = at.selectbox(key=f"sw_channel-{run.run_id}")
    assert list(picker.options) == [channel_label(k) for k in run.ok_channels()]
    picker.set_value(channel).run()
    kinds = ["synthetic", "replay"] if run.has_test_rows else ["synthetic"]
    if len(kinds) > 1:
        at.radio(key=f"sw_stream-{run.run_id}").set_value(stream).run()
    else:
        assert stream == kinds[0] and not [r for r in at.radio if r.key == f"sw_stream-{run.run_id}"]
    assert not errors(at), errors(at)
    at.button(key="sw_step").click().run()
    assert not errors(at), errors(at)
    session = at.session_state[sweep.SESSION_KEY]
    assert session.run is run and session.channel == channel
    pace = int(at.number_input(key="sw_pace").value)
    assert session.stats.emitted == pace and session.stats.ticks == 1
    log = session.log_frame()
    assert len(log) == pace
    if stream == "replay":
        assert set(log["row_id"].astype(int)) <= set(run.data.test_rows.tolist())
    else:
        assert log["row_id"].isna().all()
    assert "Detections over time" in _titles(at)
    return session


def _record(at: AppTest, run: TrainingRun, batch: ScoredBatch, session: Any) -> dict[str, Any]:
    """07 Record: every export offered (the Assay and Sweep ones included), then the PDF built inline."""
    goto(at, "record")
    assert not errors(at), errors(at)
    downloads = set(_downloads(at))
    assert {"rec_dl_assay", "rec_dl_sweep", "rec_dl_run_history"} <= downloads
    # The leaderboard needs the readings, which only 03 Measure or the PDF build take (never a download).
    measured = run.has_test_rows and evaluate.cached_evaluations(run) is not None
    assert ("rec_dl_leaderboard" in downloads) == measured
    items = {item.key: item for item in exports.export_items(run, assay=batch, sweep=session)}
    assay_csv = pd.read_csv(io.BytesIO(items["assay"].build()), encoding="utf-8-sig")  # type: ignore[misc]
    assert len(assay_csv) == UPLOAD_ROWS and "predicted_label" in assay_csv.columns
    assert not set(FEATURES) & set(assay_csv.columns)
    sweep_csv = pd.read_csv(io.BytesIO(items["sweep"].build()), encoding="utf-8-sig")  # type: ignore[misc]
    assert len(sweep_csv) == session.stats.emitted and not set(FEATURES) & set(sweep_csv.columns)
    at.button(key="rec_build").click().run()
    assert not errors(at), errors(at)
    built = at.session_state[record.PDF_RESULT]
    assert built["run_id"] == run.run_id and built["data"].startswith(b"%PDF-")
    sections = list(built["sections"])
    assert {"Assay", "Sweep", "Notes and limitations"} <= set(sections)
    assert ("Confusion matrices" in sections) == run.has_test_rows
    assert not built["problems"], built["problems"]
    assert "rec_dl_pdf" in _downloads(at)
    assert ("rec_dl_leaderboard" in _downloads(at)) == run.has_test_rows  # taken by the build
    links = [link.proto.label for link in at.get("page_link")]
    assert {"05 Assay ✓", "06 Sweep ✓", "07 Record ✓"} <= set(links)
    return built


def _visit_every_station(at: AppTest, run: TrainingRun, upload: bytes, *, channel: str, stream: str,
                         score_with: str) -> dict[str, Any]:
    """Probe, Assay, Sweep and Record on ``run``; returns the built record."""
    _probe(at, run)
    batch = _assay(at, run, upload, score_with)
    session = _sweep(at, run, channel, stream)
    return _record(at, run, batch, session)


# --------------------------------------------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------------------------------------------
@pytest.mark.slow
def test_every_station_works_on_a_fit_and_on_its_loaded_copies(fresh_caches: None, quick_charts: None,
                                                                tmp_path: Path) -> None:
    """The whole path through the Logbook (slow; the default suite checks each station on a copy loaded without
    rows, and the fit/copy separation, in each station's own quick tests)."""
    prepared, run = fit_synthetic()  # every channel, as 02 Fit fits them by default
    at = app_with_run(run, prepared).run()
    assert not errors(at), errors(at)
    fitted = at.session_state[state.RUN]
    assert fitted.ok_channels() == ["forest", "xgboost", "svm", "mlp", "logreg"]
    fits = _fits()
    upload = _upload_bytes(at, fitted, tmp_path)

    # 1. The fit in memory, with CH3 chosen at 05 Assay and 06 Sweep.
    _assay(at, fitted, upload, "svm")
    _sweep(at, fitted, "svm", "replay")
    assert _fits() == fits

    # 2. Saved and loaded again at the Logbook: the held-out rows are rebuilt (the same sample), CH3 is not kept.
    goto(at, "logbook")
    at.button(key="lb_save").click().run()
    assert not errors(at), errors(at)
    at.button(key="lb_load").click().run()
    assert not errors(at), errors(at)
    loaded = at.session_state[state.RUN]
    assert loaded is not fitted and loaded.run_id == fitted.run_id and loaded.origin == "loaded"
    assert loaded.has_test_rows and loaded.ok_channels() == SAVED_CHANNELS
    assert np.array_equal(loaded.data.test_rows, fitted.data.test_rows)
    assert _fits() == fits
    # The stream of the fit (CH3) does not carry over to the loaded copy, nor does its CH3 choice.
    goto(at, "sweep")
    assert not errors(at), errors(at)
    assert sweep.SESSION_KEY not in at.session_state
    assert "loaded again from disk" in _text(at)
    assert at.selectbox(key=f"sw_channel-{loaded.run_id}").value in SAVED_CHANNELS
    goto(at, "assay")
    assert at.selectbox(key=assay.channel_key(loaded)).value in [*SAVED_CHANNELS, CONSENSUS]
    # Nor does the fit's CH3 Assay batch, or its record: they belong to the fit, not to the loaded copy.
    assert "another copy of run" in _text(at)
    goto(at, "record")
    assert not errors(at), errors(at)
    assert "rec_dl_assay" not in _downloads(at) and "another copy of this run" in _text(at)
    assert "Assay: not yet" in _text(at)
    built = _visit_every_station(at, loaded, upload, channel="xgboost", stream="synthetic", score_with=CONSENSUS)
    assert built["pages"] >= 6  # the full record: confusion matrices, curves, importance, timing
    assert _fits() == fits
    # 3. The same set loaded without any rows: test_a_multiclass_run_loaded_with_and_without_rows (slow), and the
    # quick per-station tests (tests/ui/test_probe_page.py, test_sweep_page.py, test_record_page.py).


@pytest.mark.slow
def test_a_multiclass_run_loaded_with_and_without_rows(fresh_caches: None, quick_charts: None,
                                                       tmp_path: Path) -> None:
    """The multi-class form of the path above, ending with the copy loaded without rows (slow, like the binary
    form; the multi-class readings of every station have their own quick tests)."""
    prepared, run = fit_synthetic(mode="multiclass", channels=("forest", "xgboost", "logreg"))
    at = app_with_run(run, prepared).run()
    assert not errors(at), errors(at)
    fitted = at.session_state[state.RUN]
    assert fitted.request.mode == "multiclass" and len(fitted.data.classes) > 2
    fits = _fits()
    upload = _upload_bytes(at, fitted, tmp_path)
    goto(at, "logbook")
    at.button(key="lb_save").click().run()
    at.button(key="lb_load").click().run()
    assert not errors(at), errors(at)
    loaded = at.session_state[state.RUN]
    assert loaded.origin == "loaded" and loaded.has_test_rows and loaded.ok_channels() == ["forest", "xgboost",
                                                                                           "logreg"]
    built = _visit_every_station(at, loaded, upload, channel="forest", stream="replay", score_with=CONSENSUS)
    batch = at.session_state[state.LAST_ASSAY]
    assert list(batch.classes) == list(loaded.data.classes) and batch.confusion is not None
    hollow = persist.restore_run(persist.load_bundle(settings_mod.MODELS_DIR / fitted.run_id))
    assert not hollow.has_test_rows and hollow.ok_channels() == ["forest", "xgboost", "logreg"]
    _make_current(at, hollow)
    short = _visit_every_station(at, hollow, upload, channel="xgboost", stream="synthetic", score_with="logreg")
    assert short["pages"] < built["pages"]
    assert evaluate.cached_evaluations(hollow) in (None, {})
    assert _fits() == fits
