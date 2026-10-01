"""04 Probe to 07 Record on a real run (skipped when the CIC-IDS2017 folder is not available).

One 20,000-row Wednesday sample is fitted with every channel at full size, once per mode. On each run:

* 04 Probe: every channel's reading of real held-out flows reproduces its stored verdict, and CH2's exact
  contributions add up to its margin;
* 05 Assay: the Thursday-morning capture (a labelled file the run never saw) is scored with the consensus, every row
  comes back, and the accuracy equals scikit-learn's on the rows whose label the run knows;
* 06 Sweep: a real-data run only replays real held-out rows (never generated ones), and the live accuracy equals
  the accuracy over the emitted flows;
* 07 Record: the PDF record builds with the network blocked, within the size and time targets, and its pages carry
  every section, the run id and the dataset citation; the CSV exports hold no feature values.

Every check reads whole recorded files and fits full-size channels, so the module is also marked ``slow``:
``-m realdata`` runs it, the default run does not. Nothing is written to disk but the test's own folders.
"""

from __future__ import annotations

import io
import socket
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import accuracy_score

from graticule import evaluate, explain, scoring, simulate
from graticule.data.prepare import DataRequest, PreparedDataset, prepare_dataset
from graticule.models import train
from graticule.models.train import TrainingRun, TrainRequest, build_training_data, train_all
from graticule.report import exports
from graticule.report import pdf as rp
from graticule.schema import FEATURES
from graticule.settings import AppSettings
from tests.unit.test_report_pdf import inflated, outline, pdf_text

pytestmark = [pytest.mark.realdata, pytest.mark.slow]
WEDNESDAY = "Wednesday-workingHours.pcap_ISCX.csv"
THURSDAY_WEB = "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv"
ROWS = 20_000
_CACHE: dict[str, Any] = {}


def _path(folder: Path, name: str) -> Path:
    path = folder / name
    if not path.is_file():
        pytest.skip(f"{name} is not in the data folder")
    return path


def _prepared(folder: Path) -> PreparedDataset:
    """The 20,000-row Wednesday sample (drawn once per test session)."""
    _path(folder, WEDNESDAY)
    if "prepared" not in _CACHE:
        _CACHE["prepared"] = prepare_dataset(DataRequest(source="cicids", data_dir=str(folder), files=(WEDNESDAY,),
                                                         row_budget=ROWS, seed=42))
    return _CACHE["prepared"]


@pytest.fixture(params=["binary", "multiclass"])
def real_run(request: pytest.FixtureRequest, real_data_dir: Path) -> tuple[PreparedDataset, TrainingRun]:
    """Every channel, full size, fitted on the Wednesday sample in one mode (once per mode and test session)."""
    prepared = _prepared(real_data_dir)
    key = f"run-{request.param}"
    if key not in _CACHE:
        train_request = TrainRequest(mode=request.param, profile="full", seed=42)
        data = build_training_data(prepared, train_request)
        _CACHE[key] = train_all(data, train_request, data_request=prepared.request,
                                dataset_fingerprint=prepared.fingerprint)
    run = _CACHE[key]
    assert run.ok_channels() == ["forest", "xgboost", "svm", "mlp", "logreg"], {
        k: r.error for k, r in run.channels.items()}
    return prepared, run


def test_probe_reads_real_held_out_flows_as_the_fit_did(real_run: tuple[PreparedDataset, TrainingRun]) -> None:
    _, run = real_run
    keys = run.ok_channels()
    fits = sum(train.FIT_CALLS.values())
    rng = np.random.default_rng(0)
    for index in rng.choice(len(run.data.y_test), size=8, replace=False):
        verdict = explain.score_flow(run, run.data.X_test[index], keys)
        for key in keys:
            stored = run.channels[key].proba[index]
            np.testing.assert_allclose(verdict.proba[key], stored, atol=1e-5)
        exact = explain.xgboost_contributions(run.channels["xgboost"].estimator, run.data.X_test[index],
                                              run.data.feature_names, int(run.channels["xgboost"].y_pred[index]))
        assert exact.total() == pytest.approx(exact.output, abs=1e-4)
    assert sum(train.FIT_CALLS.values()) == fits


def test_assay_scores_an_unseen_capture(real_run: tuple[PreparedDataset, TrainingRun], real_data_dir: Path) -> None:
    _, run = real_run
    batch = scoring.score_upload(run, _path(real_data_dir, THURSDAY_WEB), channel=scoring.CONSENSUS,
                                 alert_threshold=0.9)
    assert batch.rows == len(batch.frame) == 170_366 and batch.labelled
    assert list(batch.voters) == run.ok_channels()
    known = batch.frame["true_label"].notna() & batch.frame["true_label"].astype("str").isin(run.data.classes)
    expected = accuracy_score(batch.frame.loc[known, "true_label"].astype("str"),
                              batch.frame.loc[known, "predicted_label"].astype("str"))
    assert batch.accuracy == pytest.approx(expected, abs=1e-12)
    if run.request.mode == "multiclass":  # the web attacks are not among the Wednesday classes
        assert set(batch.unseen_labels) == {"Web Attack - Brute Force", "Web Attack - XSS",
                                            "Web Attack - Sql Injection"}
    probabilities = batch.frame[batch.probability_columns].to_numpy(dtype=np.float64)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-5)
    _CACHE[f"assay-{run.request.mode}"] = batch


def test_sweep_replays_only_real_held_out_rows(real_run: tuple[PreparedDataset, TrainingRun]) -> None:
    _, run = real_run
    assert simulate.stream_kinds(run) == ["replay"]
    with pytest.raises(ValueError):
        simulate.make_source(run, "synthetic")
    session = simulate.SimulationSession(run, "xgboost", simulate.make_source(run, "replay", seed=42),
                                         alert_threshold=0.9)
    for _ in range(15):
        session.step(200)
    log = session.log_frame()
    assert len(log) == session.stats.emitted == 3_000
    assert set(log["row_id"].astype(int)) <= set(run.data.test_rows.tolist())
    assert session.stats.live_accuracy == pytest.approx(
        accuracy_score(log["true_label"].astype("str"), log["predicted"].astype("str")), abs=1e-12)
    _CACHE[f"sweep-{run.request.mode}"] = session


def test_record_of_a_real_run(real_run: tuple[PreparedDataset, TrainingRun]) -> None:
    prepared, run = real_run
    fits = sum(train.FIT_CALLS.values())
    evals = evaluate.evaluate_run(run)
    assay = _CACHE.get(f"assay-{run.request.mode}")
    sweep = _CACHE.get(f"sweep-{run.request.mode}")
    extras = rp.ReportExtras.from_run(run, assay=assay, sweep=sweep)

    def refuse(*args: Any, **kwargs: Any) -> Any:
        raise OSError("network access is blocked in this test")

    started = time.perf_counter()
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(socket, "socket", refuse)
        patch.setattr(socket, "create_connection", refuse)
        report = rp.render_report(run, evals, prepared_summary=rp.summarise_prepared(prepared),
                                  settings=asdict(AppSettings()), extras=extras)
    seconds = time.perf_counter() - started
    data = report.data
    assert data.startswith(b"%PDF-") and data.rstrip().endswith(b"%%EOF") and not report.problems
    assert 6 <= report.pages <= 24 and report.images >= 12
    assert report.size_bytes < 3 * 2**20
    assert seconds < 60, seconds  # about 7-9 s on the development machine; the target is 20 s
    sections = ["Sample sheet", "Fit settings", "Readings", "Confusion matrices", "Curves", "Feature importance",
                "Timing", *(["Assay"] if assay is not None else []), *(["Sweep"] if sweep is not None else []),
                "Notes and limitations"]
    plain = inflated(data)
    assert outline(plain) == sections == list(report.sections)
    text = pdf_text(plain)
    for phrase in ("Graticule", run.run_id, WEDNESDAY, "Sharafaldin", "ICISSP", "Balanced accuracy", "Consensus",
                   "CH3 RBF SVM", f"page {report.pages} of {report.pages}"):
        assert phrase in text, phrase
    assert not {"○", "◆", "▲", "µ"} & set(text)  # the fonts have no such glyphs
    predictions = pd.read_csv(io.BytesIO(exports.predictions_csv(run, prepared=prepared)), encoding="utf-8-sig")
    assert len(predictions) == len(run.data.y_test) and not set(FEATURES) & set(predictions.columns)
    assert set(predictions["source_file"].astype("str")) == {WEDNESDAY}
    assert sum(train.FIT_CALLS.values()) == fits
