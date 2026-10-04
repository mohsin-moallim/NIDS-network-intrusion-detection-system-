"""Headless checks of the 07 Record station: the prerequisite note (with the history export still offered), the PDF
build (inline and in the background, sharing the app's work slot), the CSV downloads, and a run without held-out
rows. Charts are built but not rendered here (the PDF unit tests render them), to keep these runs quick."""

from __future__ import annotations

import io
from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd
import pytest
from PIL import Image
from streamlit.testing.v1 import AppTest

from graticule import evaluate, viz
from graticule.history import RunHistory
from graticule.models import train
from graticule.models.jobs import claim_slot, release_slot
from tests.ui.harness import app_with_run, errors, fit_synthetic, fresh_caches, goto, new_app  # noqa: F401
from ui import state
from ui.pages import record

pytestmark = pytest.mark.ui


def _fits() -> int:
    return sum(train.FIT_CALLS.values())


def _downloads(at: AppTest) -> list[str]:
    return [b.key for b in at.get("download_button")]


def _captions(at: AppTest) -> str:
    return " ".join(c.value for c in at.caption)


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


def _fitted_app() -> AppTest:
    """A session holding a small synthetic sample and a fresh three-channel fit of it (fitted directly, as 02 Fit
    would; a fresh run each time, since these tests keep readings and results on it), at 02 Fit."""
    prepared, run = fit_synthetic(channels=("forest", "xgboost", "logreg"))
    RunHistory().record(run)  # as 02 Fit records every fit (the test's own history file)
    at = app_with_run(run, prepared, key="fit").run()
    assert not errors(at), errors(at)
    return at


def test_without_a_fit_the_station_points_to_02_fit(fresh_caches: None) -> None:
    at = new_app("record").run()
    assert not errors(at), errors(at)
    assert "Needs a fitted channel" in " ".join(m.value for m in at.markdown)
    assert "Go to 02 Fit" in [link.proto.label for link in at.get("page_link")]
    assert "rec_build" not in [b.key for b in at.button]
    assert [h.value for h in at.subheader] == ["CSV exports"]
    assert "No fit has been recorded yet." in _captions(at)  # the history export is offered, empty for now
    assert not _downloads(at)


def test_record_builds_the_pdf_and_offers_every_export(fresh_caches: None, quick_charts: None,
                                                       monkeypatch: pytest.MonkeyPatch) -> None:
    at = _fitted_app()
    run = at.session_state[state.RUN]
    fits = _fits()
    goto(at, "record")
    assert not errors(at), errors(at)
    assert [h.value for h in at.subheader] == ["PDF record", "CSV exports"]
    keys = _downloads(at)
    assert {"rec_dl_predictions", "rec_dl_run_history", "rec_dl_zip"} <= set(keys)
    assert "rec_dl_pdf" not in keys and "rec_dl_cross_validation" not in keys
    # The readings are not taken by a download (only in the work slot: 03 Measure or the PDF build).
    assert "rec_dl_leaderboard" not in keys and "rec_dl_per_class" not in keys
    assert evaluate.cached_evaluations(run) is None
    captions = _captions(at)
    assert "Run cross-validation at 03 Measure" in captions and "Score a file at 05 Assay" in captions
    assert "Cross-validation: not yet" in captions and "readings have not been taken yet" in captions

    # While a fit (or any other long job) holds the work slot, the build waits.
    assert claim_slot("a fit")
    try:
        at.run()
        assert at.button(key="rec_build").proto.disabled
        assert "A fit is running in this app" in _captions(at)
    finally:
        release_slot()

    # Built inline (as in headless runs): the PDF is kept, and its download appears.
    at.run()
    at.button(key="rec_build").click().run()
    assert not errors(at), errors(at)
    assert any(s.value.startswith("PDF record built in") for s in at.success)
    built = at.session_state[record.PDF_RESULT]
    assert built["run_id"] == run.run_id and built["data"].startswith(b"%PDF-") and built["pages"] >= 5
    assert "rec_dl_pdf" in _downloads(at)
    pdf_button = next(b for b in at.get("download_button") if b.key == "rec_dl_pdf")
    assert pdf_button.proto.label == "Download PDF record"
    assert at.button(key="rec_build").proto.label == "Build PDF record again"
    assert "Pages" in " ".join(m.value for m in at.markdown)
    assert evaluate.cached_evaluations(run) is not None
    assert {"rec_dl_leaderboard", "rec_dl_per_class"} <= set(_downloads(at))  # the build took the readings
    # The stepper ticks 07 Record in the very run that stored the PDF.
    assert "record" in at.session_state[state.DONE]
    assert "07 Record ✓" in [link.proto.label for link in at.get("page_link")]
    assert _fits() == fits  # a record never fits anything

    # A result arriving later marks the PDF as out of date and adds its export.
    frame = pd.DataFrame([{"key": "logreg", "Channel": "CH5 Logistic regression", "Folds": 2, "Rows": 100,
                           "Accuracy mean": 0.9, "Accuracy std": 0.01, "Balanced accuracy mean": 0.9,
                           "Balanced accuracy std": 0.01, "F1 macro mean": 0.9, "F1 macro std": 0.01,
                           "Fit s mean": 0.1, "Error": None}], columns=list(evaluate.CV_COLUMNS))
    frame.attrs.update({"k": 2, "rows": 100, "rows_available": 900, "note": "", "cancelled": False, "seconds": 1.0,
                        "folds": []})
    evaluate.remember_cross_validation(run, frame)
    at.run()
    assert not errors(at), errors(at)
    assert "New results have arrived since this PDF was built" in _captions(at)
    assert {"rec_dl_cross_validation", "rec_dl_cross_validation_folds"} <= set(_downloads(at))

    # Built again in the background: the progress panel takes over until the task ends.
    monkeypatch.setenv("GRATICULE_SYNC_TRAINING", "0")
    at.button(key="rec_build").click().run()
    assert not errors(at), errors(at)
    task = evaluate.get_task(at.session_state[record.PDF_TASK])
    assert task is not None and task.kind == "record" and task.run_id == run.run_id
    assert task.wait(120)
    at.run()
    assert not errors(at), errors(at)
    assert record.PDF_TASK not in at.session_state and evaluate.get_task(task.task_id) is None
    assert any(s.value.startswith("PDF record built in") for s in at.success)
    assert "New results have arrived" not in _captions(at)
    assert _fits() == fits

    # Another copy of the same run (as loaded from disk: same id, another object) is not offered this PDF.
    copy = replace(run, origin="loaded")
    at.session_state[state.RUN] = copy
    at.run()
    assert not errors(at), errors(at)
    assert "rec_dl_pdf" not in _downloads(at) and "describes another copy of this run" in _captions(at)
    assert at.button(key="rec_build").proto.label == "Build PDF record"


def test_a_run_without_held_out_rows_is_explained(fresh_caches: None, quick_charts: None) -> None:
    at = _fitted_app()
    run = at.session_state[state.RUN]
    data = run.data
    empty = replace(data, X_test=np.empty((0, data.n_features), dtype=np.float32), y_test=np.empty(0, dtype=np.int64),
                    test_rows=np.empty(0, dtype=np.int64), detailed_test_labels=np.empty(0, dtype=str))
    hollow = replace(run, run_id=run.run_id + "-h", data=empty, origin="loaded")
    state.run_registry().put(hollow.run_id, hollow)
    at.session_state[state.LAST_RUN_ID] = hollow.run_id
    at.session_state[state.RUN] = hollow
    goto(at, "record")
    assert not errors(at), errors(at)
    assert "without its held-out rows" in " ".join(m.value for m in at.markdown)
    assert "Go to Logbook" in [link.proto.label for link in at.get("page_link")]
    assert "rec_dl_leaderboard" not in _downloads(at) and "held-out rows" in _captions(at)
    at.button(key="rec_build").click().run()  # the short record of recorded readings still builds
    assert not errors(at), errors(at)
    assert at.session_state[record.PDF_RESULT]["run_id"] == hollow.run_id
