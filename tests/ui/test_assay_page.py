"""Headless checks of the 05 Assay station: its prerequisite, scoring an uploaded file only on the button, the
readings, the preview and the download, a file lacking columns, and the shared work slot. Nothing is ever fitted.

The run is fitted once for the module (tiny models) and handed to each session, as right after a fit at 02 Fit; the
uploads are generated flows written to the test's ``tmp_path``.
"""

from __future__ import annotations

import threading
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from graticule import evaluate, scoring
from graticule.data.prepare import DataRequest, PreparedDataset, prepare_dataset
from graticule.models import train
from graticule.models.jobs import claim_slot, release_slot, slot_holder
from graticule.models.train import TrainingRun, TrainRequest, build_training_data, train_all
from graticule.schema import FEATURES, LABEL
from graticule.scoring import CONSENSUS, ScoredBatch
from tests.helpers import write_cic_csv
from tests.ui.harness import errors, fresh_caches, goto, new_app  # noqa: F401
from ui import state
from ui.pages import assay

pytestmark = pytest.mark.ui
ROWS = 230


@pytest.fixture(scope="module")
def fitted() -> tuple[PreparedDataset, TrainingRun]:
    """A small synthetic sample and a binary fit of three quick channels."""
    prepared = prepare_dataset(DataRequest(source="synthetic", synthetic_flows=1_500, seed=8))
    request = TrainRequest(profile="test", seed=8, channels=("forest", "xgboost", "logreg"))
    run = train_all(build_training_data(prepared, request), request, data_request=prepared.request,
                    dataset_fingerprint=prepared.fingerprint)
    return prepared, run


def _fits() -> int:
    return sum(train.FIT_CALLS.values())


def _text(at: AppTest) -> str:
    return " ".join([m.value for m in at.markdown] + [c.value for c in at.caption])


def _upload(prepared: PreparedDataset, run: TrainingRun, path: Path, n: int, **options: object) -> bytes:
    """Write ``n`` generated flows (held-out rows of the synthetic sample) as a CIC-style CSV; return its bytes."""
    frame = prepared.frame.iloc[run.data.test_rows[:n]]
    values = frame[list(FEATURES)].to_numpy(dtype=np.float64)
    rows = [{**dict(zip(FEATURES, (float(v) for v in row))), LABEL: label}
            for row, label in zip(values, frame[LABEL].astype(str))]
    return write_cic_csv(path, rows, **options).read_bytes()  # type: ignore[arg-type]


def _session(run: TrainingRun) -> AppTest:
    """An Assay session whose current run is ``run``."""
    at = new_app("assay")
    state.run_registry().put(run.run_id, run)
    at.session_state[state.RUN] = run
    at.session_state[state.LAST_RUN_ID] = run.run_id
    at.run()
    assert not errors(at), errors(at)
    return at


def test_without_a_fit_the_station_points_to_02_fit(fresh_caches: None) -> None:
    at = new_app("assay").run()
    assert not errors(at), errors(at)
    assert "Needs a fitted channel" in _text(at)
    assert "Go to 02 Fit" in [link.proto.label for link in at.get("page_link")]
    assert not at.file_uploader


def test_upload_score_read_and_download(fresh_caches: None, fitted: tuple[PreparedDataset, TrainingRun],
                                        tmp_path: Path) -> None:
    prepared, run = fitted
    fits = _fits()
    at = _session(run)
    assert at.button(key=assay.SCORE).disabled  # nothing to score yet
    assert at.selectbox(key=assay.channel_key(run)).options == ["CH1 Random forest", "CH2 XGBoost",
                                                                  "CH5 Logistic regression", "Consensus of all channels"]
    assert at.slider(key=assay.THRESHOLD).value == pytest.approx(0.90)
    assert f"The {len(run.data.feature_names)} columns the channels read" in [e.label for e in at.expander]

    data = _upload(prepared, run, tmp_path / "upload.csv", ROWS, bom=True)
    at.file_uploader(key=assay.UPLOAD).set_value(("upload.csv", data, "text/csv")).run()
    assert not errors(at), errors(at)
    assert state.LAST_ASSAY not in at.session_state  # uploading alone scores nothing
    at.selectbox(key=assay.channel_key(run)).set_value("xgboost")
    at.slider(key=assay.THRESHOLD).set_value(0.8)
    at.button(key=assay.SCORE).click().run()
    assert not errors(at), errors(at)

    batch = at.session_state[state.LAST_ASSAY]
    assert isinstance(batch, ScoredBatch) and batch.rows == ROWS and len(batch.frame) == ROWS
    assert batch.channel == "xgboost" and batch.alert_threshold == pytest.approx(0.8) and batch.labelled
    assert [s.value for s in at.success] == [f"Scored {ROWS:,} flows from upload.csv with CH2 XGBoost in "
                                             f"{batch.seconds:,.1f} s."]
    text = _text(at)
    for reading in ("Rows scored", "Rows with bad values", "Attacks found", "Alerts", "Seconds", "Accuracy",
                    "Balanced accuracy"):
        assert reading in text, reading
    assert f"{batch.accuracy:.4f}" in text and f"{batch.balanced_accuracy:.4f}" in text
    assert [s for s in at.subheader if s.value == "Readings"]
    # Confusion matrix against the labels, verdict counts and the preview of the first 200 rows.
    charts = at.get("vega_lite_chart")
    assert len(charts) == 1 and "CH2 XGBoost against the file" in charts[0].proto.spec
    verdicts = next(d.value for d in at.dataframe if list(d.value.columns) == ["Verdict", "Rows", "Share"])
    assert verdicts["Verdict"].tolist() == ["○ Normal", "◆ Attack"] and verdicts["Rows"].sum() == ROWS
    preview = next(d.value for d in at.dataframe if "attack_probability" in d.value.columns)
    assert len(preview) == 200 and preview.columns[0] == "Verdict"
    assert any(v.startswith("▲ Alert") for v in preview["Verdict"]) or not batch.alerts
    assert "First 200 scored rows" in [s.value for s in at.subheader]
    # The download: the full scored CSV, named after the run and the channel.
    download = at.get("download_button")
    assert [d.proto.id for d in download] and download[0].proto.label == "Download the scored CSV"
    assert batch.file_name == f"graticule-assay-{run.run_id}-xgboost.csv"
    assert batch.to_csv_bytes().startswith(b"\xef\xbb\xbf")
    assert "05 Assay ✓" in [link.proto.label for link in at.get("page_link")]

    # Changing the choices afterwards redraws only: the batch stays, and the page says it is out of date.
    at.selectbox(key=assay.channel_key(run)).set_value(CONSENSUS).run()
    assert not errors(at), errors(at)
    assert at.session_state[state.LAST_ASSAY] is batch
    assert "differs from these readings" in _text(at)
    # Scoring again with the consensus replaces the batch.
    at.button(key=assay.SCORE).click().run()
    assert not errors(at), errors(at)
    again = at.session_state[state.LAST_ASSAY]
    assert again is not batch and again.channel == CONSENSUS and again.voters == ("forest", "xgboost", "logreg")
    assert "differs from these readings" not in _text(at)
    # A visit to another station keeps the choices the readings were taken with (Streamlit forgets the state of
    # widgets that are not drawn; the page keeps a copy), so the readings are not called out of date.
    goto(at, "bench")
    goto(at, "assay")
    assert not errors(at), errors(at)
    assert at.selectbox(key=assay.channel_key(run)).value == CONSENSUS
    assert at.slider(key=assay.THRESHOLD).value == pytest.approx(0.8)
    assert "differs from these readings" not in _text(at)
    assert _fits() == fits  # scoring never fits


def test_a_copy_of_the_run_does_not_take_over_the_fits_readings(fresh_caches: None,
                                                                fitted: tuple[PreparedDataset, TrainingRun],
                                                                tmp_path: Path) -> None:
    """A fit and its copy loaded from disk share an id; a file scored with the fit (here with CH1, which a copy
    could hold, so only the run object tells them apart) is not shown as the copy's readings, and 07 Record
    leaves it out of the copy's exports."""
    prepared, run = fitted
    at = _session(run)
    data = _upload(prepared, run, tmp_path / "upload.csv", 30)
    at.file_uploader(key=assay.UPLOAD).set_value(("upload.csv", data, "text/csv")).run()
    at.button(key=assay.SCORE).click().run()
    assert not errors(at), errors(at)
    assert "another copy of run" not in _text(at)
    copy = replace(run, origin="loaded", bundle_path=str(tmp_path / "saved"))
    state.run_registry().put(copy.run_id, copy)
    at.session_state[state.RUN] = copy
    at.run()
    assert not errors(at), errors(at)
    assert "another copy of run" in _text(at) and "as fitted" in _text(at)
    goto(at, "record")
    assert not errors(at), errors(at)
    assert "rec_dl_assay" not in [b.key for b in at.get("download_button")]
    assert "another copy of this run" in " ".join(c.value for c in at.caption)


def test_a_file_lacking_columns_is_refused_with_their_names(fresh_caches: None,
                                                            fitted: tuple[PreparedDataset, TrainingRun],
                                                            tmp_path: Path) -> None:
    prepared, run = fitted
    used = list(run.data.feature_names)
    at = _session(run)
    data = _upload(prepared, run, tmp_path / "short.csv", 20, drop_columns=[used[1], used[4]], include_label=False)
    at.file_uploader(key=assay.UPLOAD).set_value(("short.csv", data, "text/csv")).run()
    at.button(key=assay.SCORE).click().run()
    assert not errors(at), errors(at)
    shown = " ".join(e.value for e in at.error)
    assert "short.csv cannot be scored" in shown and used[1] in shown and used[4] in shown
    assert state.LAST_ASSAY not in at.session_state


def test_the_button_waits_while_other_work_holds_the_slot(fresh_caches: None,
                                                          fitted: tuple[PreparedDataset, TrainingRun],
                                                          tmp_path: Path) -> None:
    prepared, run = fitted
    at = _session(run)
    data = _upload(prepared, run, tmp_path / "upload.csv", 10)
    at.file_uploader(key=assay.UPLOAD).set_value(("upload.csv", data, "text/csv")).run()
    assert not at.button(key=assay.SCORE).disabled
    assert claim_slot("a fit")
    try:
        at.run()
        assert at.button(key=assay.SCORE).disabled
        assert "A fit is running in this app" in _text(at)
    finally:
        release_slot()
    at.run()
    assert not at.button(key=assay.SCORE).disabled


def test_background_scoring_shows_progress_can_be_cancelled_and_finishes(
        fresh_caches: None, fitted: tuple[PreparedDataset, TrainingRun], tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch) -> None:
    prepared, run = fitted
    monkeypatch.setenv("GRATICULE_SYNC_TRAINING", "0")
    gate = threading.Event()
    real = scoring.score_upload

    def held(*args: object, **kwargs: object) -> ScoredBatch:
        assert gate.wait(30), "the test never opened the gate"
        return real(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(scoring, "score_upload", held)
    at = _session(run)
    data = _upload(prepared, run, tmp_path / "upload.csv", 40)
    at.file_uploader(key=assay.UPLOAD).set_value(("upload.csv", data, "text/csv")).run()

    # Started in the background: the panel shows progress, the button waits, the slot is taken.
    at.button(key=assay.SCORE).click().run()
    assert not errors(at), errors(at)
    task = evaluate.get_task(at.session_state[assay.TASK])
    assert task is not None and not task.finished
    assert "Scoring upload.csv" in _text(at) and at.button(key=assay.SCORE).disabled
    assert slot_holder() == "a batch scoring at 05 Assay"
    at.button(key=assay.CANCEL).click().run()
    gate.set()
    assert task.wait(30)
    at.run()
    assert not errors(at), errors(at)
    assert "Scoring of upload.csv cancelled" in " ".join(i.value for i in at.info)
    assert state.LAST_ASSAY not in at.session_state and assay.TASK not in at.session_state
    assert slot_holder() is None

    # A second try runs to the end and its readings appear once the page is drawn again.
    at.button(key=assay.SCORE).click().run()
    task = evaluate.get_task(at.session_state[assay.TASK])
    assert task is not None and task.wait(30)
    deadline = time.monotonic() + 30
    while state.LAST_ASSAY not in at.session_state and time.monotonic() < deadline:
        at.run()
        assert not errors(at), errors(at)
    batch = at.session_state[state.LAST_ASSAY]
    assert batch.rows == 40 and batch.channel == "forest"
    assert any("Scored 40 flows from upload.csv" in s.value for s in at.success)
    assert evaluate.get_task(task.task_id) is None  # a finished task is forgotten
    # The run that adopted the finished task already shows the tick (no further click needed).
    assert "05 Assay ✓" in [link.proto.label for link in at.get("page_link")]
