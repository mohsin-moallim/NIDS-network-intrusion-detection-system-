"""Rules every station shares about readings and the files they come from.

* A score short of 1 never prints as a perfect 1.0000 (screen tables, cards, chart labels and the PDF).
* The alert rule gives one answer for one flow at 04 Probe, 05 Assay and 06 Sweep, even when the attack
  probability sits exactly on the threshold.
* A held-out flow at 04 Probe takes the readings stored for it, so its verdict is the one 03 Measure counted.
* "Normal" and "BENIGN" are both normal traffic; labels a binary run reads as Attack are named; 0/1 labels are not
  read as Attack.
* Awkward uploads: lines ending in a bare carriage return, UTF-16 text, a byte-order mark before Windows-1252 text.
* 05 Assay says which uploaded rows the run was trained on (or held out), and reads the rest on their own.
* A bundle file another program holds is reported as such.

Uploads are written into each test's ``tmp_path``; flows come from the synthetic generator or ``tests.helpers``.
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

from graticule import explain, persist, scoring, theme
from graticule.data.prepare import DataRequest, PreparedDataset, prepare_dataset
from graticule.data.reader import DataFileError, read_flow_csv
from graticule.data.sampling import target_for_mode
from graticule.evaluate import quick_metrics
from graticule.models.train import TrainingRun, TrainRequest, build_training_data, train_all
from graticule.models.verdict import alert_flags, attack_probability
from graticule.report.pdf import _score
from graticule.schema import FEATURES, LABEL, is_normal_traffic
from graticule.scoring import ALERT, ATTACK_PROBABILITY, PREDICTED, score_upload
from graticule.simulate import ReplaySource, SimulationSession
from tests.helpers import make_rows, with_values, write_cic_csv

pytestmark = pytest.mark.unit
SEED = 13


@pytest.fixture(scope="module")
def prepared() -> PreparedDataset:
    """About 1,200 synthetic flows."""
    return prepare_dataset(DataRequest(source="synthetic", synthetic_flows=1_200, seed=SEED))


@pytest.fixture(scope="module")
def run(prepared: PreparedDataset) -> TrainingRun:
    """A binary run with a forest, a neural net and a logistic regression (tiny models)."""
    request = TrainRequest(profile="test", seed=SEED, channels=("forest", "mlp", "logreg"))
    return train_all(build_training_data(prepared, request), request, data_request=prepared.request,
                     dataset_fingerprint=prepared.fingerprint)


def _rows(prepared: PreparedDataset, positions: np.ndarray) -> list[dict[str, object]]:
    """Rows of the prepared sample (generated flows) as dicts for :func:`write_cic_csv`."""
    frame = prepared.frame.iloc[positions]
    values = frame[list(FEATURES)].to_numpy(dtype=np.float64)
    return [{**dict(zip(FEATURES, (float(v) for v in row))), LABEL: label}
            for row, label in zip(values, frame[LABEL].astype(str))]


# --------------------------------------------------------------------------------------------------------------
# Printing scores
# --------------------------------------------------------------------------------------------------------------
def test_a_score_short_of_one_never_prints_as_perfect() -> None:
    assert theme.score_text(0.999978) == "0.9999" and theme.score_text(0.99995) == "0.9999"
    assert theme.score_text(1.0) == "1.0000" and theme.score_text(0.9995) == "0.9995"
    assert theme.score_text(0.12344) == "0.1234" and theme.score_text(None) == "n/a"
    assert theme.score_text(float("nan"), missing="–") == "–"
    shown = theme.shown_scores([0.99999, 1.0, np.nan, 0.5])
    assert shown[0] == theme.BELOW_PERFECT and shown[1] == 1.0 and np.isnan(shown[2]) and shown[3] == 0.5
    assert _score(0.999969) == "0.9999" and _score(1) == "1.0000"


def test_display_tables_hold_imperfect_scores_below_one() -> None:
    from ui import components

    table = pd.DataFrame({"Channel": ["CH1", "CH2"], "ROC-AUC": [0.999978, 1.0], "Rows": [10, 20]})
    shown = components.shown_scores(table, ("ROC-AUC", "Missing column"))
    assert [f"{v:.4f}" for v in shown["ROC-AUC"]] == ["0.9999", "1.0000"]
    assert table["ROC-AUC"].iloc[0] == 0.999978  # the readings themselves are untouched
    assert shown["Rows"].tolist() == [10, 20]


# --------------------------------------------------------------------------------------------------------------
# One alert rule, one answer
# --------------------------------------------------------------------------------------------------------------
class _OnTheLine:
    """A stand-in channel that reads every flow as an attack with probability exactly 0.7 (as float32)."""

    classes_ = np.array([0, 1])

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return np.tile([0.3, 0.7], (len(np.asarray(X)), 1))


def test_attack_probability_and_alerts_use_one_precision() -> None:
    proba = np.array([[0.3, 0.7], [0.9, 0.1]], dtype=np.float32)
    values = attack_probability(proba, 0)
    assert values.dtype == np.float32 and values.tolist() == proba[:, 1].tolist()
    three = np.array([[0.3, 0.4, 0.3]], dtype=np.float32)
    assert attack_probability(three, 0)[0] == np.float32(1.0 - np.float64(np.float32(0.3)))
    assert attack_probability(three, None).tolist() == [1.0]
    # The same flow in float32 or float64 gets the same flag on the threshold.
    on_line = np.float32(0.7)
    for given in (np.array([on_line]), np.array([float(on_line)])):
        assert alert_flags(given, [1], 0, 0.7).tolist() == [True]
    assert alert_flags(np.array([0.95]), [0], 0, 0.5).tolist() == [False]  # never without an attack verdict


def test_probe_assay_and_sweep_agree_on_a_flow_at_the_threshold(prepared: PreparedDataset, run: TrainingRun,
                                                                tmp_path: Path) -> None:
    stub = replace(run.channels["forest"], estimator=_OnTheLine())
    on_line = replace(run, channels={**run.channels, "forest": stub})
    threshold = 0.7

    batch = score_upload(on_line, write_cic_csv(tmp_path / "flows.csv", _rows(prepared, run.data.test_rows[:12])),
                         channel="forest", alert_threshold=threshold)
    assert batch.frame[PREDICTED].eq("Attack").all() and batch.frame[ALERT].all()

    session = SimulationSession(on_line, "forest", ReplaySource.from_run(on_line, seed=2), alert_threshold=threshold)
    events = session.step(12)
    assert all(e.alert for e in events)

    verdict = explain.score_flow(on_line, run.data.X_test[0], ["forest"])
    assert verdict.raises_alert("forest", threshold)
    assert verdict.attack_probability("forest") == pytest.approx(float(batch.frame[ATTACK_PROBABILITY].iloc[0]),
                                                                 abs=0.0)
    assert events[0].attack_probability == float(np.float32(0.7))


def test_a_held_out_flow_takes_the_readings_stored_for_it(run: TrainingRun) -> None:
    keys = run.ok_channels()
    for index in (0, 7, len(run.data.y_test) - 1):
        verdict = explain.score_flow(run, run.data.X_test[index], keys, test_index=index)
        for key in keys:
            assert np.array_equal(verdict.proba[key], run.channels[key].proba[index])
    # A test index whose row holds other values is ignored: the flow is scored as given.
    other = explain.score_flow(run, run.data.X_test[1], keys, test_index=0)
    for key in keys:
        assert np.allclose(other.proba[key], run.channels[key].proba[1], atol=1e-5)


# --------------------------------------------------------------------------------------------------------------
# Labels
# --------------------------------------------------------------------------------------------------------------
def test_normal_labels_count_as_normal_traffic_in_binary_mode() -> None:
    assert all(is_normal_traffic(x) for x in ("BENIGN", "benign", "Normal", " normal "))
    assert not any(is_normal_traffic(x) for x in ("Attack", "DoS", "0", ""))
    result = target_for_mode(pd.Series(["Normal"] * 50 + ["DoS"] * 50), "binary", min_class_count=50)
    assert result.classes == ["Normal", "Attack"] and result.counts == {"Normal": 50, "Attack": 50}


class _Halves:
    """A stand-in binary channel: Attack for long flows, Normal otherwise."""

    classes_ = np.array([0, 1])

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        long = np.asarray(X)[:, 0] > 5_000
        return np.where(long[:, None], [0.1, 0.9], [0.8, 0.2])


def _binary_stub_run(files: tuple[str, ...] = ()) -> Any:
    return SimpleNamespace(
        run_id="20261001-000000-abcd", ok_channels=lambda: ["forest"],
        channels={"forest": SimpleNamespace(estimator=_Halves())},
        data=SimpleNamespace(feature_names=("Flow Duration",), classes=("Normal", "Attack")),
        data_request=SimpleNamespace(nonfinite_strategy="drop", merge_web_attacks=False, source="cicids",
                                     files=files),
        request=SimpleNamespace(mode="binary"),
    )


def _labelled(labels: list[str], durations: list[float]) -> list[dict[str, object]]:
    return [with_values(row, {LABEL: label, "Flow Duration": d})
            for row, label, d in zip(make_rows({"BENIGN": len(labels)}), labels, durations)]


def test_a_binary_assay_names_the_labels_it_read_as_attack(tmp_path: Path) -> None:
    rows = _labelled(["BENIGN", "Normal", "DoS Hulk", "Odd thing", "DoS Hulk"], [100, 100, 9_000, 9_000, 100])
    batch = score_upload(_binary_stub_run(), write_cic_csv(tmp_path / "named.csv", rows), channel="forest",
                         alert_threshold=0.5)
    assert batch.rows_measured == 5 and batch.accuracy == pytest.approx(4 / 5)
    note = next(n for n in batch.notes if n.startswith("In binary mode"))
    assert "DoS Hulk (2), Odd thing (1)" in note


def test_numeric_labels_are_not_read_as_attack(tmp_path: Path) -> None:
    rows = _labelled(["0", "1", "0", "1"], [100, 9_000, 100, 9_000])
    batch = score_upload(_binary_stub_run(), write_cic_csv(tmp_path / "digits.csv", rows), channel="forest",
                         alert_threshold=0.5)
    assert batch.unseen_labels == {"0": 2, "1": 2} and batch.rows_measured == 0 and batch.accuracy is None
    assert any("only of digits" in n for n in batch.notes)
    assert not any(n.startswith("In binary mode") for n in batch.notes)


# --------------------------------------------------------------------------------------------------------------
# Awkward files
# --------------------------------------------------------------------------------------------------------------
def test_old_mac_line_ends_are_scored_like_any_file(tmp_path: Path) -> None:
    rows = _labelled(["BENIGN", "DoS Hulk", "BENIGN", "DoS Hulk"], [100, 9_000, 200, 8_000])
    plain = write_cic_csv(tmp_path / "plain.csv", rows)
    old_mac = plain.read_bytes().replace(b"\n", b"\r")
    reference = score_upload(_binary_stub_run(), plain, channel="forest", alert_threshold=0.5)
    batch = score_upload(_binary_stub_run(), old_mac, channel="forest", alert_threshold=0.5, name="mac.csv")
    assert batch.rows == 4 and batch.frame[PREDICTED].tolist() == reference.frame[PREDICTED].tolist()
    assert batch.accuracy == reference.accuracy == 1.0
    assert any("bare carriage return" in n for n in batch.notes)
    assert len(pd.read_csv(io.BytesIO(batch.to_csv_bytes()), encoding="utf-8-sig")) == 4


def test_wide_text_is_refused_with_its_encoding_named(tmp_path: Path) -> None:
    text = write_cic_csv(tmp_path / "flows.csv").read_bytes().decode("utf-8")
    for encoding, name in (("utf-16", "UTF-16"), ("utf-16-le", "UTF-16"), ("utf-32", "UTF-32")):
        with pytest.raises(DataFileError, match=name):
            read_flow_csv(text.encode(encoding), name="wide.csv")
        with pytest.raises(DataFileError, match=name):
            score_upload(_binary_stub_run(), text.encode(encoding), channel="forest", alert_threshold=0.5)


def test_a_byte_order_mark_before_windows_text_does_not_cost_a_column(tmp_path: Path) -> None:
    path = write_cic_csv(tmp_path / "mixed.csv", encoding="cp1252", label_style="cp1252", bom=True)
    frame, report = read_flow_csv(path)
    assert report.encoding == "cp1252" and report.missing_features == [] and report.extra_columns == []
    assert list(frame.columns)[0] == "Destination Port"
    batch = score_upload(_binary_stub_run(), path, channel="forest", alert_threshold=0.5)
    assert batch.extra_columns == []
    written = batch.to_csv_bytes().decode("utf-8-sig")
    assert "ï»¿" not in written and written.lstrip().startswith("Destination Port")


# --------------------------------------------------------------------------------------------------------------
# Rows the run has seen
# --------------------------------------------------------------------------------------------------------------
def test_the_assay_names_rows_the_run_was_trained_on(prepared: PreparedDataset, run: TrainingRun,
                                                     tmp_path: Path) -> None:
    trained, held_out = run.data.train_rows[:20], run.data.test_rows[:15]
    path = write_cic_csv(tmp_path / "mixed.csv", _rows(prepared, np.concatenate([trained, held_out])))
    batch = score_upload(run, path, channel="forest", alert_threshold=0.9)
    assert batch.rows_seen_in_training == 20 and batch.rows_seen_held_out == 15
    assert batch.rows_unseen_measured == 15
    truth = run.data.y_test[:15]
    predicted = run.channels["forest"].proba[:15].argmax(axis=1)
    expected = quick_metrics(truth, predicted, run.data.n_classes)
    assert batch.unseen_accuracy == pytest.approx(expected["accuracy"])
    assert batch.unseen_balanced_accuracy == pytest.approx(expected["balanced_accuracy"])
    assert any("repeat a row this run was trained on" in n for n in batch.notes)

    only_held_out = score_upload(run, write_cic_csv(tmp_path / "held.csv", _rows(prepared, held_out)),
                                 channel="forest", alert_threshold=0.9)
    assert only_held_out.rows_seen_in_training == 0 and only_held_out.unseen_accuracy is None

    # A file the run's sample was drawn from is named as such.
    named = score_upload(_binary_stub_run(files=("Wednesday-workingHours.pcap_ISCX.csv",)),
                         write_cic_csv(tmp_path / "Wednesday-workingHours.pcap_ISCX.csv",
                                       _labelled(["BENIGN"], [100])), channel="forest", alert_threshold=0.5)
    assert named.from_sample_file and any("sample was drawn from" in n for n in named.notes)


def test_a_bundle_file_another_program_holds_is_reported(run: TrainingRun, tmp_path: Path,
                                                         monkeypatch: pytest.MonkeyPatch) -> None:
    folder = persist.save_run(run, tmp_path / "saved")
    real = persist.sha256_file

    def held(path: Path) -> str:
        if Path(path).suffix == ".joblib":
            raise PermissionError(13, "Access is denied", str(path))
        return real(path)

    monkeypatch.setattr(persist, "sha256_file", held)
    with pytest.raises(persist.BundleReadError, match="another program may be holding it"):
        persist.load_bundle(folder)


# --------------------------------------------------------------------------------------------------------------
# Distinct flows and the traffic they stand for
# --------------------------------------------------------------------------------------------------------------
def test_each_row_knows_how_many_recorded_flows_it_stands_for(tmp_path: Path) -> None:
    from graticule.evaluate import held_out_repeats, repeats_sentence

    monday, tuesday = "Monday-WorkingHours.pcap_ISCX.csv", "Tuesday-WorkingHours.pcap_ISCX.csv"
    normal = make_rows({"BENIGN": 60})
    attacks = make_rows({"FTP-Patator": 40}, start=500)
    # Row 0 occurs 3 times on Monday and twice more on Tuesday; attack row 0 occurs 4 times on Tuesday.
    write_cic_csv(tmp_path / monday, normal + [normal[0], normal[0]])
    write_cic_csv(tmp_path / tuesday, attacks + [attacks[0]] * 3 + [normal[0], normal[0]])
    ds = prepare_dataset(DataRequest(source="cicids", data_dir=str(tmp_path), files=(monday, tuesday)))
    assert ds.copies is not None and len(ds.copies) == len(ds.frame) == 100
    assert int(ds.copies.sum()) == 60 + 2 + 40 + 3 + 2
    keys = list(zip(ds.frame["_file"].astype(str), ds.frame["_row"]))
    assert int(ds.copies[keys.index((monday, 0))]) == 5 and int(ds.copies[keys.index((tuesday, 0))]) == 4

    request = TrainRequest(profile="test", seed=SEED, channels=("logreg",))
    data = build_training_data(ds, request)
    assert data.test_copies is not None and len(data.test_copies) == len(data.y_test)
    run = train_all(data, request, data_request=ds.request, dataset_fingerprint=ds.fingerprint)
    repeats = held_out_repeats(run)
    assert repeats is not None and repeats["rows"] == len(data.y_test)
    assert repeats["flows"] == int(data.test_copies.sum()) >= repeats["rows"]
    sentence = repeats_sentence({"rows": 10, "flows": 25, "largest": 9, "repeated": 2})
    assert sentence is not None and "10 rows stand for 25 recorded flows" in sentence
    assert repeats_sentence({"rows": 10, "flows": 10, "largest": 1, "repeated": 0}) is None
