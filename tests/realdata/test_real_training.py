"""02 Fit on the recorded files (skipped when the CIC-IDS2017 folder is not available).

Every check here reads a whole recorded file and fits full-size channels, so all are also marked ``slow``: the
default run stays short, and ``-m realdata`` runs them (the quick real-data checks of headers and labels in
``test_real_files.py`` stay in the default run). The two Wednesday checks share one 10,000-row sample, fitted in
both modes with the forest, XGBoost and logistic regression; the Thursday-morning check fits the two quick channels
in multi-class mode. The 200,000-row timings live in ``tests/slow/test_benchmark.py``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from nids.data.prepare import DataRequest, PreparedDataset, prepare_dataset
from nids.models.train import TrainingData, TrainingRun, TrainRequest, build_training_data, train_all
from nids.schema import BENIGN

pytestmark = pytest.mark.realdata
WEDNESDAY = "Wednesday-workingHours.pcap_ISCX.csv"
THURSDAY_WEB = "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv"
ROWS = 10_000
_PREPARED: dict[tuple[str, str], PreparedDataset] = {}


def _prepared(folder: Path, name: str) -> PreparedDataset:
    """Prepare (once per test session) a 10,000-row sample of one file."""
    if not (folder / name).is_file():
        pytest.skip(f"{name} is not in the data folder")
    key = (str(folder), name)
    if key not in _PREPARED:
        _PREPARED[key] = prepare_dataset(DataRequest(source="cicids", data_dir=str(folder), files=(name,),
                                                     row_budget=ROWS, seed=42))
    return _PREPARED[key]


def _fit(prepared: PreparedDataset, request: TrainRequest) -> tuple[TrainingData, TrainingRun]:
    data = build_training_data(prepared, request)
    run = train_all(data, request, data_request=prepared.request, dataset_fingerprint=prepared.fingerprint)
    return data, run


@pytest.mark.slow  # reads the whole Wednesday file (about 6 s with the fits)
def test_wednesday_binary_full_size_channels(real_data_dir: Path) -> None:
    prepared = _prepared(real_data_dir, WEDNESDAY)
    request = TrainRequest(mode="binary", channels=("forest", "xgboost", "logreg"), profile="full")
    data, run = _fit(prepared, request)
    assert data.classes == ("Normal", "Attack")
    assert run.ok_channels() == ["forest", "xgboost", "logreg"], {k: r.error for k, r in run.channels.items()}
    for key, result in run.channels.items():
        balanced = result.extra["metrics"]["balanced_accuracy"]
        assert balanced > 0.9, (key, balanced)
        assert result.proba is not None and result.proba.shape == (len(data.y_test), 2)
    assert run.channels["forest"].extra["n_trees"] == 150
    assert 0 < run.channels["xgboost"].extra["best_iteration"] < 300


@pytest.mark.slow  # shares the Wednesday sample of the test above
def test_wednesday_multiclass_full_size_channels(real_data_dir: Path) -> None:
    prepared = _prepared(real_data_dir, WEDNESDAY)
    request = TrainRequest(mode="multiclass", channels=("forest", "xgboost", "logreg"), profile="full")
    data, run = _fit(prepared, request)
    assert data.classes[0] == BENIGN and len(data.classes) >= 5
    assert "Heartbleed" not in data.classes, "11 Heartbleed flows are below the minimum class count of 50"
    assert data.reports["target"]["dropped"].get("Heartbleed", 0) > 0
    assert run.ok_channels() == ["forest", "xgboost", "logreg"], {k: r.error for k, r in run.channels.items()}
    for result in run.channels.values():
        assert result.proba is not None and result.proba.shape == (len(data.y_test), len(data.classes))
        np.testing.assert_allclose(result.proba.sum(axis=1), 1.0, atol=1e-5)
    assert run.channels["forest"].extra["metrics"]["balanced_accuracy"] > 0.9


@pytest.mark.slow  # reads the whole Thursday-morning file (about 2 s with the fits)
def test_thursday_web_attacks_drop_the_rare_class(real_data_dir: Path) -> None:
    prepared = _prepared(real_data_dir, THURSDAY_WEB)
    assert not prepared.request.merge_web_attacks
    request = TrainRequest(mode="multiclass", channels=("forest", "logreg"), profile="full")
    data, run = _fit(prepared, request)
    dropped = data.reports["target"]["dropped"]
    assert 0 < dropped["Web Attack - Sql Injection"] < 50
    assert "below the minimum" in data.reports["target"]["dropped_reasons"]["Web Attack - Sql Injection"]
    assert {"Web Attack - Brute Force", "Web Attack - XSS"} <= set(data.classes)
    assert data.classes[0] == BENIGN
    assert run.ok_channels() == ["forest", "logreg"], {k: r.error for k, r in run.channels.items()}
