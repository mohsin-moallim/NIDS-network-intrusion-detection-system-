"""CSV exports and the ZIP of 07 Record: every file parses back, is UTF-8 with a byte-order mark, has the right columns,
never carries feature values, and the ZIP holds exactly the available files plus a README."""

from __future__ import annotations

import io
import zipfile
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest

from graticule import evaluate
from graticule.data.prepare import FILE_COL, ROW_COL, DataRequest, PreparedDataset, prepare_dataset
from graticule.history import RunHistory
from graticule.models.train import TrainingRun, TrainRequest, build_training_data, train_all
from graticule.report import exports
from graticule.schema import FEATURE_SET

pytestmark = pytest.mark.unit
SEED = 11
BOM = b"\xef\xbb\xbf"


@pytest.fixture(scope="module")
def prepared() -> PreparedDataset:
    """About 1,200 synthetic flows."""
    return prepare_dataset(DataRequest(source="synthetic", synthetic_flows=1_200, seed=SEED))


def _fit(prepared: PreparedDataset, **changes: Any) -> TrainingRun:
    request = TrainRequest(**{"profile": "test", "seed": SEED, **changes})
    data = build_training_data(prepared, request)
    return train_all(data, request, data_request=prepared.request, dataset_fingerprint=prepared.fingerprint)


@pytest.fixture(scope="module")
def runs(prepared: PreparedDataset) -> dict[str, TrainingRun]:
    """A three-channel binary run and a two-channel multi-class run."""
    return {"binary": _fit(prepared, mode="binary", channels=("forest", "xgboost", "logreg")),
            "multiclass": _fit(prepared, mode="multiclass", channels=("forest", "logreg"))}


def read(data: bytes) -> pd.DataFrame:
    """Parse CSV bytes the way a user's tool would (and check the byte-order mark)."""
    assert data.startswith(BOM)
    return pd.read_csv(io.BytesIO(data), encoding="utf-8-sig")


def fake_assay(run: TrainingRun, run_id: str | None = None) -> SimpleNamespace:
    """An Assay result shaped like :class:`graticule.scoring.ScoredBatch`: uploaded columns, then scoring columns."""
    n = 30
    rng = np.random.default_rng(3)
    p = rng.random(n)
    frame = pd.DataFrame({
        "Flow Duration": rng.random(n), "Total Fwd Packets": rng.random(n), "Source IP": ["10.0.0.1"] * n,
        "Label": ["BENIGN"] * n, "predicted_label": np.where(p > 0.5, "Attack", "Normal"), "prob_Normal": 1 - p,
        "prob_Attack": p, "attack_probability": p, "alert": p >= 0.9, "true_label": ["Normal"] * n,
    }, index=pd.RangeIndex(n))
    return SimpleNamespace(frame=frame, channel="forest", rows=n, rows_with_bad_values=0, seconds=0.1,
                           labelled=True, accuracy=0.5, balanced_accuracy=0.5, confusion=None, unseen_labels={},
                           run_id=run.run_id if run_id is None else run_id, classes=run.data.classes)


def fake_sweep(run: TrainingRun, flows: int = 3) -> SimpleNamespace:
    """A simulation session shaped like :class:`graticule.simulate.SimulationSession`."""
    log = pd.DataFrame({"seq": np.arange(flows), "tick": np.ones(flows, dtype=int), "row_id": np.arange(flows),
                        "true_label": ["Normal"] * flows, "predicted": ["Attack"] * flows,
                        "attack_probability": np.full(flows, 0.95), "confidence": np.full(flows, 0.95),
                        "alert": [True] * flows, "correct": [False] * flows})
    return SimpleNamespace(run=run, run_id=run.run_id, classes=run.data.classes, log_frame=lambda: log,
                           stats=SimpleNamespace(emitted=flows))


def test_leaderboard_and_per_class_exports(runs: dict[str, TrainingRun]) -> None:
    for mode, run in runs.items():
        evals = evaluate.evaluate_run(run)
        board = read(exports.leaderboard_csv(run, evals))
        scores = [title for _, title in evaluate.score_columns(run.request.mode)]
        assert list(board.columns) == ["key", "Channel", *scores, *evaluate.LEADERBOARD_TAIL]
        assert list(board["key"]) == [*evaluate.leaderboard(evals, run)["key"], exports.CONSENSUS_KEY]
        assert board["Balanced accuracy"].iloc[:-1].is_monotonic_decreasing
        consensus = exports.consensus_metrics(run)
        assert consensus is not None
        assert board["Balanced accuracy"].iloc[-1] == pytest.approx(consensus["balanced_accuracy"])
        assert pd.isna(board["Fit s"].iloc[-1]) and board["Channel"].iloc[-1].startswith("Consensus")
        per_class = read(exports.per_class_csv(run, evals))
        assert list(per_class.columns) == ["key", "Channel", "class", "support", "precision", "recall", "f1",
                                           "roc_auc", "average_precision"]
        assert len(per_class) == len(evals) * len(run.data.classes), mode


def test_predictions_hold_labels_and_probabilities_but_no_feature_values(runs: dict[str, TrainingRun]) -> None:
    for run in runs.values():
        frame = read(exports.predictions_csv(run))
        classes = list(run.data.classes)
        assert not set(frame.columns) & FEATURE_SET
        assert len(frame) == len(run.data.y_test)
        assert list(frame.columns[:4]) == ["test_index", "sample_row", "true_class", "true_label"]
        assert list(frame["true_class"]) == [classes[c] for c in run.data.y_test]
        assert list(frame["sample_row"]) == list(run.data.test_rows)
        for key in run.ok_channels():
            probs = frame[[f"{key}_prob_{c}" for c in classes]].to_numpy()
            assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)
            assert list(frame[f"{key}_predicted"]) == [classes[c] for c in run.channels[key].y_pred]
        assert "consensus_predicted" in frame.columns and frame["consensus_agreement"].between(
            0, len(run.ok_channels())).all()
        assert "source_file" not in frame.columns  # synthetic flows: no provenance columns


def test_predictions_name_source_file_and_row_for_real_data(prepared: PreparedDataset,
                                                            runs: dict[str, TrainingRun]) -> None:
    run = runs["binary"]
    as_real = replace(run, data_request=replace(run.data_request, source="cicids", files=("Tuesday.csv",)))
    frame = read(exports.predictions_csv(as_real, prepared=prepared))
    rows = run.data.test_rows
    assert list(frame["source_file"]) == list(prepared.frame[FILE_COL].astype(str).to_numpy()[rows])
    assert list(frame["source_row"]) == list(prepared.frame[ROW_COL].to_numpy()[rows])
    other = replace(prepared, fingerprint="0" * 64)  # not the run's sample: no provenance
    assert "source_file" not in read(exports.predictions_csv(as_real, prepared=other)).columns


def test_cross_validation_and_history_exports(runs: dict[str, TrainingRun]) -> None:
    run = runs["binary"]
    cv = evaluate.cross_validate_run(run, k=2, max_rows=400, channels=("logreg",))
    summary = read(exports.cross_validation_csv(cv))
    assert list(summary.columns) == list(evaluate.CV_COLUMNS) and list(summary["key"]) == ["logreg"]
    folds = read(exports.cross_validation_folds_csv(cv))
    assert list(folds.columns) == ["key", "Channel", "fold", "accuracy", "balanced_accuracy", "f1_macro",
                                   "fit_seconds"]
    assert len(folds) == 2
    history = RunHistory()
    history.record(run)
    lines = read(exports.history_csv(history))
    assert run.run_id in set(lines["run_id"]) and "metrics_json" in lines.columns


def test_assay_export_keeps_scoring_columns_only(runs: dict[str, TrainingRun]) -> None:
    batch = fake_assay(runs["binary"])
    frame = read(exports.assay_csv(batch))
    assert list(frame.columns) == ["row", "predicted_label", "prob_Normal", "prob_Attack", "attack_probability",
                                   "alert", "true_label"]
    assert list(frame["row"]) == list(range(30))  # 0-based data rows, as every export and 05 Assay count them
    assert not set(frame.columns) & (FEATURE_SET | {"Source IP", "Label"})
    text = exports.assay_csv(batch).decode("utf-8-sig")
    assert frame["alert"].dtype == bool and "true" in text and "True" not in text  # flags as the Assay file has them
    # An uploaded column that merely looks like a probability column (a scored file scored again) is left out.
    again = SimpleNamespace(**{**vars(batch), "frame": batch.frame.assign(**{"prob_Attack (uploaded)": 0.5})})
    assert "prob_Attack (uploaded)" not in read(exports.assay_csv(again)).columns


def test_sweep_export_is_the_flow_log(runs: dict[str, TrainingRun]) -> None:
    frame = read(exports.sweep_csv(fake_sweep(runs["binary"])))
    assert list(frame.columns) == ["seq", "tick", "row_id", "true_label", "predicted", "attack_probability",
                                   "confidence", "alert", "correct"]
    assert len(frame) == 3


def test_export_items_say_what_is_missing(runs: dict[str, TrainingRun]) -> None:
    run = runs["multiclass"]
    items = {item.key: item for item in exports.export_items(run)}
    assert list(items) == list(exports.EXPORTS)
    assert all(items[k].available for k in ("leaderboard", "per_class", "predictions"))
    assert not items["cross_validation"].available and "03 Measure" in items["cross_validation"].missing
    assert not items["assay"].available and "05 Assay" in items["assay"].missing
    assert not items["sweep"].available and "06 Sweep" in items["sweep"].missing
    assert not items["run_history"].available and "No fit" in items["run_history"].missing
    assert items["leaderboard"].file_name == f"graticule-leaderboard-{run.run_id}.csv"
    foreign = {i.key: i for i in exports.export_items(run, assay=fake_assay(run, run_id="another"))}
    assert not foreign["assay"].available and "another run" in foreign["assay"].missing
    mine = {i.key: i for i in exports.export_items(run, assay=fake_assay(run), sweep=fake_sweep(run))}
    assert mine["assay"].available and mine["sweep"].available
    empty_sweep = {i.key: i for i in exports.export_items(run, sweep=fake_sweep(run, flows=0))}
    assert not empty_sweep["sweep"].available
    nothing = {i.key: i for i in exports.export_items(None)}
    assert not any(i.available for i in nothing.values())
    assert "02 Fit" in nothing["leaderboard"].missing
    # A page that must not take the readings itself (07 Record) gets them offered only once they exist.
    fresh = replace(run)  # a copy without readings kept on it
    setattr(fresh, evaluate.EVALUATIONS_ATTR, None)
    waiting = {i.key: i for i in exports.export_items(fresh, measure_if_needed=False)}
    assert not waiting["leaderboard"].available and "03 Measure" in waiting["leaderboard"].missing
    assert not waiting["per_class"].available and waiting["predictions"].available
    ready = {i.key: i for i in exports.export_items(fresh, evaluations=evaluate.evaluate_run(run),
                                                     measure_if_needed=False)}
    assert ready["leaderboard"].available and ready["per_class"].available


def test_a_run_without_held_out_rows_exports_no_readings(runs: dict[str, TrainingRun]) -> None:
    run = runs["binary"]
    data = run.data
    empty = replace(data, X_test=np.empty((0, data.n_features), dtype=np.float32), y_test=np.empty(0, dtype=np.int64),
                    test_rows=np.empty(0, dtype=np.int64), detailed_test_labels=np.empty(0, dtype=str))
    hollow = replace(run, run_id=run.run_id + "-h", data=empty)
    items = {i.key: i for i in exports.export_items(hollow)}
    assert not items["leaderboard"].available and "held-out rows" in items["leaderboard"].missing
    assert exports.consensus_metrics(hollow) is None
    assert len(read(exports.predictions_csv(hollow))) == 0


def test_zip_holds_every_available_file_and_a_readme(runs: dict[str, TrainingRun]) -> None:
    run = runs["binary"]
    RunHistory().record(run)
    evals = evaluate.evaluate_run(run)
    data = exports.bundle_zip(run, evaluations=evals, assay=fake_assay(run), sweep=fake_sweep(run),
                              extra_files={"graticule-record-x.pdf": (b"%PDF-1.4 test", "The PDF record.")})
    names = exports.zip_names(data)
    expected = {exports.export_file_name(k, run.run_id) for k in ("leaderboard", "per_class", "predictions",
                                                                  "assay", "sweep")}
    expected |= {exports.export_file_name("run_history", None), "graticule-record-x.pdf", exports.README_NAME}
    assert set(names) == expected
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        readme = archive.read(exports.README_NAME).decode("utf-8")
        for name in expected - {exports.README_NAME}:
            assert name in readme
            if name.endswith(".csv"):
                assert archive.read(name).startswith(BOM)
                pd.read_csv(io.BytesIO(archive.read(name)), encoding="utf-8-sig")
        assert run.run_id in readme and "No file here holds flow feature values" in readme


def test_zip_names_a_table_that_failed_to_build(runs: dict[str, TrainingRun]) -> None:
    def broken() -> bytes:
        raise RuntimeError("disk on fire")

    items = [exports.ExportItem("leaderboard", "Leaderboard", "board.csv", "x", broken),
             exports.ExportItem("per_class", "Per class", "per-class.csv", "y", lambda: b"\xef\xbb\xbfa\n1\n"),
             exports.ExportItem("cv", "CV", "cv.csv", "z", None, "not measured")]
    data = exports.bundle_zip(runs["binary"], items=items)
    assert set(exports.zip_names(data)) == {"per-class.csv", exports.README_NAME}
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        readme = archive.read(exports.README_NAME).decode("utf-8")
    assert "board.csv (RuntimeError: disk on fire)" in readme and "cv.csv" not in readme


def test_real_scoring_and_simulation_objects_export(runs: dict[str, TrainingRun], tmp_path: Any) -> None:
    """The Assay and Sweep objects of the app itself (when those stations are present) export as well."""
    scoring = pytest.importorskip("graticule.scoring")
    simulate = pytest.importorskip("graticule.simulate")
    from tests.helpers import make_rows, write_cic_csv

    run = runs["binary"]
    path = write_cic_csv(tmp_path / "upload.csv", make_rows({"BENIGN": 4, "DoS Hulk": 3}))
    batch = scoring.score_upload(run, path, channel="forest", alert_threshold=0.9)
    assert exports.belongs_to_run(batch, run)
    frame = read(exports.assay_csv(batch))
    assert len(frame) == 7 and not set(frame.columns) & FEATURE_SET and "predicted_label" in frame.columns
    session = simulate.SimulationSession(run, "forest", simulate.ReplaySource.from_run(run), alert_threshold=0.9)
    session.step(5)
    assert exports.belongs_to_run(session, run)
    log = read(exports.sweep_csv(session))
    assert len(log) == 5 and not set(log.columns) & FEATURE_SET
