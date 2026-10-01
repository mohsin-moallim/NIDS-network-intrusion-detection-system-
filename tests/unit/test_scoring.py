"""Batch scoring (05 Assay): every uploaded row comes back scored, readings match scikit-learn, and awkward files
are read as training files are.

The uploads are small CSV files written into each test's ``tmp_path``: flows from the synthetic generator (never
dataset rows), or the made-up rows of ``tests.helpers``. Channels use ``profile="test"``, so fitting the two runs
takes about a second.
"""

from __future__ import annotations

import io
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix

from graticule import persist, scoring
from graticule.data.clean import BYTES_PER_S, PACKETS_PER_S
from graticule.data.prepare import DataRequest, PreparedDataset, prepare_dataset
from graticule.data.reader import DataFileError, read_flow_csv
from graticule.models import train
from graticule.models.train import TrainingRun, TrainRequest, build_training_data, train_all
from graticule.models.verdict import alert_flags
from graticule.report import exports
from graticule.report.pdf import ReportExtras
from graticule.schema import FEATURES, LABEL
from graticule.scoring import (
    AGREEMENT,
    ALERT,
    ATTACK_PROBABILITY,
    CONSENSUS,
    PREDICTED,
    TRUE_LABEL,
    MissingColumnsError,
    ScoredBatch,
    score_upload,
)
from tests.helpers import make_rows, with_values, write_cic_csv

pytestmark = pytest.mark.unit
SEED = 21
THRESHOLD = 0.9


@pytest.fixture(scope="module")
def prepared() -> PreparedDataset:
    """About 1,500 synthetic flows (six classes)."""
    return prepare_dataset(DataRequest(source="synthetic", synthetic_flows=1_500, seed=SEED))


@pytest.fixture(scope="module")
def runs(prepared: PreparedDataset) -> dict[str, TrainingRun]:
    """A binary run with four channels and a multi-class run with two (tiny models)."""
    out = {}
    for mode, channels in (("binary", ("forest", "xgboost", "svm", "logreg")), ("multiclass", ("forest", "logreg"))):
        request = TrainRequest(profile="test", seed=SEED, mode=mode, channels=channels)
        out[mode] = train_all(build_training_data(prepared, request), request, data_request=prepared.request,
                              dataset_fingerprint=prepared.fingerprint)
    return out


def _generated_rows(prepared: PreparedDataset, run: TrainingRun, n: int) -> list[dict[str, object]]:
    """``n`` generated flows (held-out rows of the synthetic sample) as row dicts for :func:`write_cic_csv`."""
    frame = prepared.frame.iloc[run.data.test_rows[:n]]
    values = frame[list(FEATURES)].to_numpy(dtype=np.float64)
    labels = frame[LABEL].astype(str).tolist()
    return [{**dict(zip(FEATURES, (float(v) for v in row))), LABEL: label} for row, label in zip(values, labels)]


def _probabilities(batch: ScoredBatch) -> np.ndarray:
    return batch.frame[batch.probability_columns].to_numpy(dtype=np.float64)


def _check_shape(batch: ScoredBatch, uploaded_rows: int) -> None:
    """Invariants of every scored batch."""
    assert batch.rows == uploaded_rows == len(batch.frame)
    proba = _probabilities(batch)
    assert proba.shape == (uploaded_rows, len(batch.classes))
    assert np.all(proba >= 0) and np.abs(proba.sum(axis=1) - 1.0).max() < 1e-6
    assert set(batch.frame[PREDICTED].unique()) <= set(batch.classes)
    assert np.array_equal(batch.frame[PREDICTED].to_numpy(dtype=object),
                          np.asarray(batch.classes, dtype=object)[proba.argmax(axis=1)])
    # The alert rule every station shares: an attack verdict at or above the threshold.
    attack_verdict = ~batch.frame[PREDICTED].isin(["BENIGN", "Normal"]).to_numpy()
    assert np.array_equal(batch.frame[ALERT].to_numpy(dtype=bool),
                          attack_verdict & (batch.frame[ATTACK_PROBABILITY].to_numpy()
                                            >= np.float32(batch.alert_threshold)))
    assert batch.frame.index.tolist() == list(range(uploaded_rows))
    # The readings hold the result columns only; the uploaded columns come back in the download.
    assert list(batch.frame.columns) == batch.result_columns


def _download(batch: ScoredBatch, **options: Any) -> pd.DataFrame:
    """The scored CSV read back as a user's tool reads it (text cells kept as text when ``dtype=str``)."""
    data = batch.to_csv_bytes()
    assert data.startswith(b"\xef\xbb\xbf")
    return pd.read_csv(io.BytesIO(data), encoding="utf-8-sig", **options)


def test_unlabelled_upload_scores_every_row(prepared: PreparedDataset, runs: dict[str, TrainingRun],
                                            tmp_path: Path) -> None:
    run = runs["binary"]
    rows = _generated_rows(prepared, run, 60)
    path = write_cic_csv(tmp_path / "flows.csv", rows, include_label=False)
    fits = sum(train.FIT_CALLS.values())
    batch = score_upload(run, path, channel="xgboost", alert_threshold=THRESHOLD)
    assert sum(train.FIT_CALLS.values()) == fits  # scoring never fits
    _check_shape(batch, 60)
    assert not batch.labelled and batch.accuracy is None and batch.balanced_accuracy is None
    assert batch.confusion is None and batch.unseen_labels == {} and batch.rows_measured == 0
    assert LABEL not in batch.frame.columns and TRUE_LABEL not in batch.frame.columns
    assert np.array_equal(batch.frame[ATTACK_PROBABILITY].to_numpy(), batch.frame["prob_Attack"].to_numpy())
    assert batch.attacks == int((batch.frame[PREDICTED] == "Attack").sum())
    assert batch.alerts == int(batch.frame[ALERT].sum())
    assert batch.channel == "xgboost" and batch.voters == ("xgboost",) and batch.run_id == run.run_id
    assert batch.source_name == "flows.csv" and batch.seconds > 0
    # The download repeats the uploaded columns as written (the header's own spacing, both copies of the repeated
    # column), then the result columns.
    back = _download(batch)
    header = path.read_text(encoding="utf-8").splitlines()[0].split(",")
    assert list(back.columns[: len(header)]) == [h if h not in header[:i] else f"{h}.1"
                                                 for i, h in enumerate(header)]
    assert list(back.columns[len(header):]) == batch.result_columns
    assert batch.values_as_written and len(back) == 60
    # The same readings as the channel gives the same rows directly.
    frame, _ = read_flow_csv(path, require_label=False)
    direct = run.channels["xgboost"].estimator.predict_proba(
        frame[list(run.data.feature_names)].to_numpy(dtype=np.float32))
    assert np.allclose(_probabilities(batch), direct, atol=1e-6)


def test_labelled_binary_readings_equal_scikit_learn(prepared: PreparedDataset, runs: dict[str, TrainingRun],
                                                     tmp_path: Path) -> None:
    run = runs["binary"]
    rows = _generated_rows(prepared, run, 120)
    path = write_cic_csv(tmp_path / "labelled.csv", rows)
    batch = score_upload(run, path.read_bytes(), channel="logreg", alert_threshold=0.75, name="labelled.csv")
    _check_shape(batch, 120)
    truth = np.where(np.array([r[LABEL] for r in rows]) == "BENIGN", "Normal", "Attack")
    assert batch.frame[TRUE_LABEL].tolist() == truth.tolist()
    guess = batch.frame[PREDICTED].to_numpy(dtype=str)
    assert batch.labelled and batch.rows_measured == 120 and batch.unseen_labels == {}
    assert batch.accuracy == pytest.approx(accuracy_score(truth, guess), abs=1e-12)
    assert batch.balanced_accuracy == pytest.approx(balanced_accuracy_score(truth, guess), abs=1e-12)
    expected = confusion_matrix(truth, guess, labels=list(run.data.classes))
    assert batch.confusion is not None and np.array_equal(batch.confusion.to_numpy(), expected)
    assert list(batch.confusion.index) == list(batch.confusion.columns) == ["Normal", "Attack"]
    assert batch.alert_threshold == 0.75


def test_labelled_multiclass_readings_equal_scikit_learn(prepared: PreparedDataset,
                                                         runs: dict[str, TrainingRun], tmp_path: Path) -> None:
    run = runs["multiclass"]
    rows = _generated_rows(prepared, run, 150)
    path = write_cic_csv(tmp_path / "multi.csv", rows)
    batch = score_upload(run, path, channel="forest", alert_threshold=THRESHOLD)
    _check_shape(batch, 150)
    truth = np.array([r[LABEL] for r in rows])
    guess = batch.frame[PREDICTED].to_numpy(dtype=str)
    assert batch.frame[TRUE_LABEL].tolist() == truth.tolist()
    assert batch.accuracy == pytest.approx(accuracy_score(truth, guess), abs=1e-12)
    assert batch.balanced_accuracy == pytest.approx(balanced_accuracy_score(truth, guess), abs=1e-12)
    expected = confusion_matrix(truth, guess, labels=list(run.data.classes))
    assert batch.confusion is not None and np.array_equal(batch.confusion.to_numpy(), expected)
    # Multi-class attack probability is 1 - P(BENIGN).
    benign = batch.frame["prob_BENIGN"].to_numpy(dtype=np.float32)
    assert np.allclose(batch.frame[ATTACK_PROBABILITY].to_numpy(), 1.0 - benign, atol=1e-7)
    assert sum(batch.verdict_counts().values()) == 150
    assert batch.attacks == int((batch.frame[PREDICTED] != "BENIGN").sum())


def test_consensus_is_the_mean_of_the_channels(prepared: PreparedDataset, runs: dict[str, TrainingRun],
                                               tmp_path: Path) -> None:
    run = runs["binary"]
    path = write_cic_csv(tmp_path / "flows.csv", _generated_rows(prepared, run, 80))
    assert scoring.channel_choices(run) == ["forest", "xgboost", "svm", "logreg", CONSENSUS]
    batch = score_upload(run, path, channel=CONSENSUS, alert_threshold=THRESHOLD)
    _check_shape(batch, 80)
    assert batch.voters == ("forest", "xgboost", "svm", "logreg")
    assert batch.channel_name == "Consensus of all channels"
    singles = [_probabilities(score_upload(run, path, channel=key, alert_threshold=THRESHOLD))
               for key in batch.voters]
    assert np.allclose(_probabilities(batch), np.mean(singles, axis=0), atol=1e-6)
    own = np.stack([s.argmax(axis=1) for s in singles])
    picked = _probabilities(batch).argmax(axis=1)
    assert np.array_equal(batch.frame[AGREEMENT].to_numpy(), (own == picked).sum(axis=0))
    assert batch.frame[AGREEMENT].between(0, 4).all()
    assert batch.accuracy is not None


def test_missing_columns_are_named_and_nothing_is_scored(prepared: PreparedDataset,
                                                         runs: dict[str, TrainingRun], tmp_path: Path) -> None:
    run = runs["binary"]
    used = list(run.data.feature_names)
    gone = [used[3], used[0]]
    path = write_cic_csv(tmp_path / "short.csv", _generated_rows(prepared, run, 20), drop_columns=gone)
    with pytest.raises(MissingColumnsError) as caught:
        score_upload(run, path, channel="forest", alert_threshold=THRESHOLD)
    assert caught.value.missing == [used[0], used[3]]  # in the order the channels read them
    assert caught.value.needed == len(used)
    message = str(caught.value)
    assert used[0] in message and used[3] in message and f"lacks 2 of the {len(used)} columns" in message
    assert isinstance(caught.value, ValueError)
    # A column the channels do not read may be missing.
    unused = next(f for f in FEATURES if f not in used)
    path = write_cic_csv(tmp_path / "fine.csv", _generated_rows(prepared, run, 20), drop_columns=[unused])
    batch = score_upload(run, path, channel="forest", alert_threshold=THRESHOLD)
    assert batch.rows == 20 and unused not in batch.frame.columns


def test_awkward_files_are_read_as_training_files_are(runs: dict[str, TrainingRun], tmp_path: Path) -> None:
    run = runs["binary"]
    rows = make_rows({"BENIGN": 8, "DoS Hulk": 4, "Web Attack - XSS": 3})
    rows[2] = with_values(rows[2], {BYTES_PER_S: float("inf"), PACKETS_PER_S: float("inf")})
    rows[5] = with_values(rows[5], {"Flow IAT Mean": None})  # an empty cell
    rows[9] = with_values(rows[9], {BYTES_PER_S: float("-inf")})
    path = write_cic_csv(tmp_path / "awkward.csv", rows, bom=True, leading_spaces=True, with_duplicate_column=True,
                         label_style="fffd", extra_columns={"Source IP": "10.0.0.7"})
    batch = score_upload(run, path, channel="forest", alert_threshold=THRESHOLD)
    _check_shape(batch, 15)
    assert batch.rows_with_bad_values == 3
    # The download repeats every line as written: "Infinity" cells, the empty cell, the extra column, both copies of
    # the repeated column and the label with its stand-in character; the readings use the tidied labels.
    back = _download(batch, dtype=str, keep_default_na=False)
    rates = back[next(c for c in back.columns if c.strip() == BYTES_PER_S)]
    assert rates.iloc[2] == "Infinity" and rates.iloc[9] == "-Infinity"
    iat = next(c for c in back.columns if c.strip() == "Flow IAT Mean")
    assert back[iat].iloc[5] == ""
    assert sum(1 for c in back.columns if c.strip().startswith("Fwd Header Length")) == 2
    ip = next(c for c in back.columns if c.strip() == "Source IP")
    assert back[ip].tolist() == ["10.0.0.7"] * 15 and batch.extra_columns == ["Source IP"]
    label = next(c for c in back.columns if c.strip() == LABEL)
    assert back[label].tolist()[-3:] == ["Web Attack � XSS"] * 3
    assert batch.frame[TRUE_LABEL].tolist() == ["Normal"] * 8 + ["Attack"] * 7
    notes = " ".join(batch.notes)
    assert "Source IP" in notes and "3 rows hold infinite or missing values" in notes
    assert "repeated" not in notes  # the routine repeated column is not worth a note
    # The preview shows the uploaded values too (numbers exact, other columns as their text).
    preview = batch.preview(4)
    assert preview["Source IP"].tolist() == ["10.0.0.7"] * 4 and len(set(preview.columns)) == len(preview.columns)
    # A Windows-1252 copy (en dash byte in the Web Attack labels) reads the same, and comes back as UTF-8 text.
    legacy = write_cic_csv(tmp_path / "legacy.csv", rows, encoding="cp1252", label_style="cp1252")
    again = score_upload(run, legacy, channel="forest", alert_threshold=THRESHOLD)
    assert again.frame[TRUE_LABEL].tolist() == batch.frame[TRUE_LABEL].tolist()
    assert np.array_equal(_probabilities(again), _probabilities(batch))
    assert "Read as cp1252 text." in again.notes
    legacy_back = _download(again, dtype=str, keep_default_na=False)
    assert legacy_back[next(c for c in legacy_back.columns if c.strip() == LABEL)].tolist()[-3:] == \
        ["Web Attack – XSS"] * 3


def test_the_runs_bad_value_strategy_is_applied(runs: dict[str, TrainingRun], tmp_path: Path) -> None:
    """Under "recompute" an infinite rate is rebuilt from the totals and the duration before scoring."""
    base = runs["binary"]
    run = replace(base, data_request=replace(base.data_request, nonfinite_strategy="recompute"))
    clean = make_rows({"BENIGN": 3, "DoS Hulk": 3})
    broken = [with_values(row, {BYTES_PER_S: float("inf")}) for row in clean]
    a = score_upload(run, write_cic_csv(tmp_path / "clean.csv", clean), channel="logreg", alert_threshold=0.9)
    b = score_upload(run, write_cic_csv(tmp_path / "broken.csv", broken), channel="logreg", alert_threshold=0.9)
    assert a.rows_with_bad_values == 0 and b.rows_with_bad_values == 6 and b.strategy == "recompute"
    # make_rows writes the rate exactly as the totals and duration give it, so rebuilding it restores the readings.
    assert np.allclose(_probabilities(a), _probabilities(b), atol=1e-6)
    back = _download(b, dtype=str, keep_default_na=False)  # the download keeps the values as written
    assert (back[next(c for c in back.columns if c.strip() == BYTES_PER_S)] == "Infinity").all()
    # Under "drop" the same rows are still scored (the rate becomes a gap the pipeline fills).
    c = score_upload(base, write_cic_csv(tmp_path / "broken2.csv", broken), channel="logreg", alert_threshold=0.9)
    assert c.rows == 6 and c.rows_with_bad_values == 6 and np.isfinite(_probabilities(c)).all()


def test_unseen_and_empty_labels_are_left_out_of_the_readings(prepared: PreparedDataset,
                                                              runs: dict[str, TrainingRun], tmp_path: Path) -> None:
    run = runs["multiclass"]
    rows = _generated_rows(prepared, run, 40)
    for i in (0, 5, 6):
        rows[i][LABEL] = "Port Probe"
    rows[7][LABEL] = ""
    path = write_cic_csv(tmp_path / "unseen.csv", rows)
    batch = score_upload(run, path, channel="logreg", alert_threshold=THRESHOLD)
    _check_shape(batch, 40)
    assert batch.unseen_labels == {"Port Probe": 3}
    assert batch.rows_without_label == 1 and batch.rows_measured == 36
    known = np.array([i not in (0, 5, 6, 7) for i in range(40)])
    truth = np.array([r[LABEL] for r in rows])[known]
    guess = batch.frame[PREDICTED].to_numpy(dtype=str)[known]
    assert batch.accuracy == pytest.approx(accuracy_score(truth, guess), abs=1e-12)
    assert batch.balanced_accuracy == pytest.approx(balanced_accuracy_score(truth, guess), abs=1e-12)
    assert batch.confusion is not None and int(batch.confusion.to_numpy().sum()) == 36
    assert batch.frame[TRUE_LABEL].iloc[0] == "Port Probe" and batch.frame[TRUE_LABEL].iloc[7] == ""
    assert "Port Probe (3)" in " ".join(batch.notes)


def test_chunked_scoring_equals_one_chunk(prepared: PreparedDataset, runs: dict[str, TrainingRun],
                                          tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Predicting 7 rows at a time, and parsing the file in many small blocks of lines, change nothing."""
    run = runs["multiclass"]
    path = write_cic_csv(tmp_path / "flows.csv", _generated_rows(prepared, run, 101))
    for channel in (CONSENSUS,):  # every channel of the run, read chunk by chunk
        whole = score_upload(run, path, channel=channel, alert_threshold=THRESHOLD)
        assert whole.upload is not None and len(whole.upload.blocks) == 1
        monkeypatch.setattr(scoring, "READ_BLOCK_MIN", 2_000)
        pieces = score_upload(run, path, channel=channel, alert_threshold=THRESHOLD, chunk_rows=7)
        monkeypatch.undo()
        assert pieces.upload is not None and len(pieces.upload.blocks) > 5
        assert sum(block.rows for block in pieces.upload.blocks) == 101
        _check_shape(pieces, 101)
        assert np.allclose(_probabilities(whole), _probabilities(pieces), atol=1e-6)
        assert whole.frame[PREDICTED].tolist() == pieces.frame[PREDICTED].tolist()
        assert whole.accuracy == pieces.accuracy and whole.rows_with_bad_values == pieces.rows_with_bad_values
        # The same download, line for line (probabilities may differ in the last bits between chunkings).
        pd.testing.assert_frame_equal(_download(whole), _download(pieces), check_exact=False, atol=1e-6)


def test_the_download_repeats_uploaded_values_exactly(runs: dict[str, TrainingRun], tmp_path: Path) -> None:
    """Large integers and long decimals come back exactly as written (no 32-bit rounding), read or not."""
    run = runs["binary"]
    rows = make_rows({"BENIGN": 3, "DoS Hulk": 2})
    rows[0] = with_values(rows[0], {"Flow Duration": 119_999_993.0, "Fwd IAT Total": 98_765_432.1,
                                    BYTES_PER_S: 1234.5678901, "Flow IAT Max": 33_333_333.0})
    path = write_cic_csv(tmp_path / "exact.csv", rows, extra_columns={"Note": '"a, b"'})  # a quoted comma
    batch = score_upload(run, path, channel="logreg", alert_threshold=THRESHOLD)
    assert batch.values_as_written
    original = path.read_text(encoding="utf-8").splitlines()
    produced = batch.to_csv_bytes().decode("utf-8-sig").splitlines()
    assert len(produced) == len(original)
    for line, source in zip(produced[1:], original[1:]):
        assert line.startswith(source + ",")  # the uploaded line itself, verbatim, then the results
    back = _download(batch, dtype=str, keep_default_na=False)
    column = {c.strip(): c for c in back.columns}
    assert back[column["Flow Duration"]].iloc[0] == "119999993"
    assert back[column["Fwd IAT Total"]].iloc[0] == "98765432.1"
    assert back[column[BYTES_PER_S]].iloc[0] == "1234.5678901"
    assert back[column["Flow IAT Max"]].iloc[0] == "33333333"
    assert back[column["Note"]].iloc[0] == "a, b"


def test_uploaded_columns_named_like_result_columns(prepared: PreparedDataset, runs: dict[str, TrainingRun],
                                                    tmp_path: Path) -> None:
    """A file that already holds a Verdict column or the result columns (a scored file scored again) keeps them,
    renamed where they clash, so the preview and the download never repeat a column name."""
    run = runs["binary"]
    path = write_cic_csv(tmp_path / "again.csv", _generated_rows(prepared, run, 12),
                         extra_columns={"Verdict": "seen", PREDICTED: "Normal", "prob_Attack": "0.5"})
    batch = score_upload(run, path, channel="forest", alert_threshold=THRESHOLD)
    preview = batch.preview()
    assert len(set(preview.columns)) == len(preview.columns) and preview.columns[0] == scoring.VERDICT
    assert {"Verdict (uploaded)", "predicted_label (uploaded)", "prob_Attack (uploaded)"} <= set(preview.columns)
    import pyarrow as pa

    pa.Table.from_pandas(preview)  # what the app's table needs: unique names
    back = _download(batch)
    assert {"predicted_label (uploaded)", "prob_Attack (uploaded)"} <= set(back.columns)
    assert "Verdict" in {c.strip() for c in back.columns}  # not a result column, so kept as written
    assert list(back.columns[-len(batch.result_columns):]) == batch.result_columns
    # 07 Record exports the result columns only, never the uploaded copies; rows count from 0.
    frame = exports.assay_frame(batch)
    assert list(frame.columns) == ["row", *batch.result_columns] and frame["row"].tolist() == list(range(12))


def test_a_pandas_index_column_is_kept_without_a_false_note(runs: dict[str, TrainingRun], tmp_path: Path) -> None:
    """A file written by ``DataFrame.to_csv()`` leads with an unnamed index column."""
    frame, _ = read_flow_csv(write_cic_csv(tmp_path / "source.csv", make_rows({"BENIGN": 3, "DoS Hulk": 2})))
    data = frame.to_csv().encode("utf-8")
    batch = score_upload(runs["binary"], data, channel="forest", alert_threshold=THRESHOLD, name="indexed.csv")
    assert batch.rows == 5 and batch.extra_columns == [""]
    notes = " ".join(batch.notes)
    assert "second reading" not in notes and "(unnamed)" in notes
    assert _download(batch).iloc[:, 0].tolist() == [0, 1, 2, 3, 4]
    # A file read whole matches the unnamed column on its second reading too.
    extra, problem = scoring._extra_columns(b",a,note\n0,1,x\n1,2,y\n", "utf-8", ["", "note"], pd.RangeIndex(2))
    assert problem is None and extra is not None and list(extra.columns) == ["", "note"]
    assert extra["note"].tolist() == ["x", "y"]
    assert scoring._extra_columns(b"a,b\n1,2\n", "utf-8", ["zzz"], pd.RangeIndex(1)) == (None, None)


def test_a_file_whose_lines_are_not_its_rows_is_read_whole(runs: dict[str, TrainingRun], tmp_path: Path) -> None:
    """A quoted line break: the lines cannot be matched with the rows, so the file is read whole, still every row
    is scored, and the note says the download writes the columns as read."""
    run = runs["binary"]
    rows = make_rows({"BENIGN": 4, "DoS Hulk": 3})
    path = write_cic_csv(tmp_path / "quoted.csv", rows, extra_columns={"Comment": '"two\nlines"'})
    batch = score_upload(run, path, channel="logreg", alert_threshold=THRESHOLD)
    _check_shape(batch, 7)
    assert not batch.values_as_written and "read whole" in " ".join(batch.notes)
    reference = score_upload(run, write_cic_csv(tmp_path / "plain.csv", rows), channel="logreg",
                             alert_threshold=THRESHOLD)
    assert np.array_equal(_probabilities(batch), _probabilities(reference))
    back = _download(batch)
    assert len(back) == 7 and back["Comment"].tolist() == ["two\nlines"] * 7
    assert list(back.columns[-len(batch.result_columns):]) == batch.result_columns
    assert batch.preview(3)["Comment"].tolist() == ["two\nlines"] * 3


class _Undecided:
    """A stand-in channel that reads BENIGN at 0.40 with the two attack classes at 0.30 each (attack probability
    0.60), or DoS Hulk at 0.90 for long flows."""

    classes_ = np.array([0, 1, 2])

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        long = np.asarray(X)[:, 0] > 5_000
        return np.where(long[:, None], [0.05, 0.9, 0.05], [0.4, 0.3, 0.3])


def test_an_alert_needs_an_attack_verdict(tmp_path: Path) -> None:
    """A flow read as BENIGN never raises an alert, however low the threshold (the rule of 04 Probe and 06 Sweep)."""
    run: Any = SimpleNamespace(
        run_id="20261001-000000-abcd", ok_channels=lambda: ["forest"],
        channels={"forest": SimpleNamespace(estimator=_Undecided())},
        data=SimpleNamespace(feature_names=("Flow Duration",), classes=("BENIGN", "DoS Hulk", "Web Attack")),
        data_request=SimpleNamespace(nonfinite_strategy="drop", merge_web_attacks=False),
        request=SimpleNamespace(mode="multiclass"),
    )
    rows = [with_values(row, {"Flow Duration": d}) for row, d in zip(make_rows({"BENIGN": 4}), (100, 9_000, 100, 9e3))]
    batch = score_upload(run, write_cic_csv(tmp_path / "close.csv", rows), channel="forest", alert_threshold=0.5)
    assert batch.frame[PREDICTED].tolist() == ["BENIGN", "DoS Hulk", "BENIGN", "DoS Hulk"]
    assert batch.frame[ATTACK_PROBABILITY].to_numpy() == pytest.approx([0.6, 0.95, 0.6, 0.95])
    assert batch.frame[ALERT].tolist() == [False, True, False, True] and batch.alerts == batch.attacks == 2
    assert alert_flags(np.array([0.6, 0.6], dtype=np.float32), np.array([0, 1]), 0, 0.5).tolist() == [False, True]
    # A run object that cannot be referenced weakly is recognised by its id, origin and channels.
    assert batch.made_with(run) and not batch.made_with(SimpleNamespace(run_id="another", ok_channels=lambda: []))


def test_uploads_as_bytes_or_file_objects_and_csv_export(prepared: PreparedDataset,
                                                         runs: dict[str, TrainingRun], tmp_path: Path) -> None:
    run = runs["binary"]
    path = write_cic_csv(tmp_path / "flows.csv", _generated_rows(prepared, run, 30))
    from_path = score_upload(run, path, channel="svm", alert_threshold=THRESHOLD)
    handle = io.BytesIO(path.read_bytes())
    handle.name = "upload.csv"  # type: ignore[attr-defined]
    handle.read(10)  # a file object that was read before is rewound
    from_object = score_upload(run, handle, channel="svm", alert_threshold=THRESHOLD)
    assert from_object.source_name == "upload.csv"
    assert np.array_equal(_probabilities(from_path), _probabilities(from_object))
    back = _download(from_object)
    assert len(back) == 30 and list(back.columns[-len(from_object.result_columns):]) == from_object.result_columns
    assert back[PREDICTED].tolist() == from_object.frame[PREDICTED].tolist()
    assert back[ALERT].dtype == bool  # written true/false, read back as flags
    assert from_object.file_name == f"graticule-assay-{run.run_id}-svm.csv"
    preview = from_object.preview(12)
    assert len(preview) == 12 and preview.columns[0] == scoring.VERDICT
    assert all(v.endswith(("○ Normal", "◆ Attack")) for v in preview[scoring.VERDICT])
    assert all(v.startswith("▲ Alert") == bool(a) for v, a in zip(preview[scoring.VERDICT], preview[ALERT]))
    assert len(from_object.preview(10_000)) == min(30, scoring.PREVIEW_ROWS)


def test_bad_requests_are_refused(prepared: PreparedDataset, runs: dict[str, TrainingRun], tmp_path: Path) -> None:
    run = runs["multiclass"]
    path = write_cic_csv(tmp_path / "flows.csv", _generated_rows(prepared, run, 5))
    with pytest.raises(ValueError, match="not a fitted channel"):
        score_upload(run, path, channel="xgboost", alert_threshold=THRESHOLD)
    with pytest.raises(ValueError, match="between 0 and 1"):
        score_upload(run, path, channel="forest", alert_threshold=1.5)
    empty = write_cic_csv(tmp_path / "empty.csv", [])
    with pytest.raises(DataFileError, match="no flow rows"):
        score_upload(run, empty, channel="forest", alert_threshold=THRESHOLD)
    with pytest.raises(DataFileError):
        score_upload(run, b"", channel="forest", alert_threshold=THRESHOLD)


class _FixedModel:
    """A stand-in channel: class 2 when Flow Duration is large, class 1 when it is middling, else class 0."""

    classes_ = np.array([0, 1, 2])

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        duration = np.asarray(X)[:, 0]
        out = np.full((len(X), 3), 0.05)
        out[np.arange(len(X)), np.where(duration > 5_000, 2, np.where(duration > 2_000, 1, 0))] = 0.9
        return out


def test_labels_follow_the_runs_classes_and_web_attack_merge(tmp_path: Path) -> None:
    run: Any = SimpleNamespace(
        run_id="20261001-000000-abcd", ok_channels=lambda: ["forest"],
        channels={"forest": SimpleNamespace(estimator=_FixedModel())},
        data=SimpleNamespace(feature_names=("Flow Duration",), classes=("BENIGN", "DoS Hulk", "Web Attack")),
        data_request=SimpleNamespace(nonfinite_strategy="drop", merge_web_attacks=True),
        request=SimpleNamespace(mode="multiclass"),
    )
    labels = ["benign", "Normal", "DoS Hulk", "dos hulk", "Web Attack - Brute Force", "Web Attack - XSS", "PortScan"]
    durations = [100.0, 100.0, 3_000.0, 3_000.0, 9_000.0, 100.0, 9_000.0]
    rows = [with_values(row, {LABEL: label, "Flow Duration": d})
            for row, label, d in zip(make_rows({"BENIGN": len(labels)}), labels, durations)]
    path = write_cic_csv(tmp_path / "labels.csv", rows, label_style="fffd")
    batch = score_upload(run, path, channel="forest", alert_threshold=0.9)
    assert batch.frame[TRUE_LABEL].tolist() == ["BENIGN", "BENIGN", "DoS Hulk", "DoS Hulk", "Web Attack",
                                                "Web Attack", "PortScan"]
    assert batch.frame[PREDICTED].tolist() == ["BENIGN", "BENIGN", "DoS Hulk", "DoS Hulk", "Web Attack",
                                               "BENIGN", "Web Attack"]
    assert batch.unseen_labels == {"PortScan": 1} and batch.rows_measured == 6
    assert batch.accuracy == pytest.approx(5 / 6) and batch.balanced_accuracy == pytest.approx((1 + 1 + 0.5) / 3)
    assert batch.frame[ALERT].tolist() == [False, False, True, True, True, False, True]
    assert batch.frame[ATTACK_PROBABILITY].to_numpy() == pytest.approx([0.1, 0.1, 0.95, 0.95, 0.95, 0.1, 0.95])


def test_a_run_loaded_from_disk_without_its_rows_scores_the_same(prepared: PreparedDataset,
                                                                 runs: dict[str, TrainingRun],
                                                                 tmp_path: Path) -> None:
    run = runs["binary"]
    loaded = persist.restore_run(persist.load_bundle(persist.save_run(run, tmp_path / "saved")))
    assert loaded.origin == "loaded" and not loaded.has_test_rows
    # The kernel SVM is not saved by default, so it is not offered; the other channels score as the fitted ones do.
    assert scoring.channel_choices(loaded) == ["forest", "xgboost", "logreg", CONSENSUS]
    path = write_cic_csv(tmp_path / "flows.csv", _generated_rows(prepared, run, 50))
    fits = sum(train.FIT_CALLS.values())
    for channel in ("forest", "xgboost", "logreg"):
        fresh = score_upload(run, path, channel=channel, alert_threshold=THRESHOLD)
        again = score_upload(loaded, path, channel=channel, alert_threshold=THRESHOLD)
        assert np.allclose(_probabilities(fresh), _probabilities(again), atol=1e-6)
        assert fresh.frame[PREDICTED].tolist() == again.frame[PREDICTED].tolist()
        assert fresh.accuracy == again.accuracy
    assert sum(train.FIT_CALLS.values()) == fits
    # The fit and its loaded copy share an id, but a batch belongs only to the run object that scored it: a CH3
    # batch of the fit never shows up in the loaded copy's record or exports (and the other way round).
    by_svm = score_upload(run, path, channel="svm", alert_threshold=THRESHOLD)
    assert loaded.run_id == run.run_id and by_svm.made_with(run) and not by_svm.made_with(loaded)
    assert exports.belongs_to_run(by_svm, run) and not exports.belongs_to_run(by_svm, loaded)
    assert ReportExtras.from_run(loaded, assay=by_svm).assay is None
    assert ReportExtras.from_run(run, assay=by_svm).assay is not None
    items = {item.key: item for item in exports.export_items(loaded, assay=by_svm)}
    assert not items["assay"].available and "another run" in items["assay"].missing
    assert not again.made_with(run) and again.made_with(loaded) and again.run_origin == "loaded"
