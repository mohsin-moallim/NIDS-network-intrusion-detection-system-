"""The recorded-traffic readings: each held-out row weighted by the recorded flows it stands for (an estimate).

Small CIC-style files written into ``tmp_path`` repeat some flows (within a file, across files, and over the chosen
columns only), so the weights, their sums and the weighted readings can be checked against counts worked out by
hand: with every class taken whole the weighted readings equal the ordinary ones on the explicitly repeated rows,
and a class thinned by the sampler scales its rows up by exactly the inverse of its sampling share. A heavy-tailed
file (a few attack flows recorded 150 times each, none of them held out) checks the caution and the range that
account for heavily repeated flows the held-out rows miss: the channel's reading over every recorded row lies
inside that range. Also covered: the weights computed once per run, the unavailable cases, the no-refit guarantee,
the exports, the manifest of a saved set and the PDF record.
"""

from __future__ import annotations

import io
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from graticule import evaluate, persist, viz
from graticule.data.prepare import DataRequest, PreparedDataset, prepare_dataset
from graticule.data.reader import read_flow_csv
from graticule.models import train
from graticule.models.train import TrainingRun, TrainRequest, build_training_data, train_all
from graticule.report import exports
from graticule.report import pdf as rp
from tests.helpers import make_rows, with_values, write_cic_csv

pytestmark = pytest.mark.unit
SEED = 11
MONDAY, TUESDAY = "Monday-WorkingHours.pcap_ISCX.csv", "Tuesday-WorkingHours.pcap_ISCX.csv"
WEDNESDAY = "Wednesday-workingHours.pcap_ISCX.csv"
THURSDAY = "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv"
CHANNELS = ("forest", "logreg")


def _fits() -> int:
    return sum(train.FIT_CALLS.values())


def _same(a: float, b: float) -> bool:
    """Exactly equal, NaN matching NaN."""
    return (np.isnan(a) and np.isnan(b)) or a == b


def _repeating_files(folder: Path) -> int:
    """Two files whose flows repeat within a file, across the files and over the curated columns only (two rows
    differ from another only in the port). Returns the rows the cleaned files hold (every repeat counted)."""
    normal = make_rows({"BENIGN": 60})
    attacks = make_rows({"FTP-Patator": 40}, start=500)
    monday = list(normal)
    for i in range(0, 60, 3):
        monday += [normal[i]] * 2  # recorded three times in all
    monday += [with_values(normal[5], {"Destination Port": 4444.0}), with_values(normal[7], {"Destination Port": 5555.0})]
    tuesday = list(attacks)
    for i in range(0, 40, 2):
        tuesday += [attacks[i]] * (1 + i % 4)  # one or three extra copies
    tuesday += [normal[1]] * 2  # repeats of a Monday flow
    write_cic_csv(folder / MONDAY, monday)
    write_cic_csv(folder / TUESDAY, tuesday)
    return len(monday) + len(tuesday)


def _fit(prepared: PreparedDataset, **changes: Any) -> TrainingRun:
    request = TrainRequest(**{"profile": "test", "seed": SEED, "channels": CHANNELS, **changes})
    data = build_training_data(prepared, request)
    return train_all(data, request, data_request=prepared.request, dataset_fingerprint=prepared.fingerprint)


@pytest.fixture(scope="module")
def whole(tmp_path_factory: pytest.TempPathFactory) -> tuple[PreparedDataset, TrainingRun, int]:
    """Every class taken whole (f_c = 1): the sample, a binary fit of CH1 and CH5, and the rows in the files."""
    folder = tmp_path_factory.mktemp("whole")
    recorded = _repeating_files(folder)
    prepared = prepare_dataset(DataRequest(source="cicids", data_dir=str(folder), files=(MONDAY, TUESDAY), seed=SEED))
    return prepared, _fit(prepared), recorded


@pytest.fixture(scope="module")
def thinned(tmp_path_factory: pytest.TempPathFactory) -> tuple[PreparedDataset, TrainingRun]:
    """300 distinct normal flows thinned to 120 by a 200-row budget; 80 attack flows (with repeats) kept whole."""
    folder = tmp_path_factory.mktemp("thinned")
    normal = make_rows({"BENIGN": 300})
    attacks = make_rows({"FTP-Patator": 80}, start=1_000)
    tuesday = list(attacks) + [attacks[i] for i in range(0, 80, 4)] * 2
    write_cic_csv(folder / MONDAY, normal)
    write_cic_csv(folder / TUESDAY, tuesday)
    prepared = prepare_dataset(DataRequest(source="cicids", data_dir=str(folder), files=(MONDAY, TUESDAY),
                                           row_budget=200, seed=SEED))
    return prepared, _fit(prepared)


# --------------------------------------------------------------------------------------------------------------
# The weighted metrics against repeated rows
# --------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("k", [2, 4])
def test_whole_number_weights_read_exactly_like_the_repeated_rows(k: int) -> None:
    rng = np.random.default_rng(k)
    n = 240
    y = rng.integers(0, k, n)
    proba = np.round(rng.random((n, k)), 2) + 1e-3  # rounded: many tied scores
    proba /= proba.sum(axis=1, keepdims=True)
    predicted = proba.argmax(axis=1)
    predicted[:30] = y[:30]
    copies = rng.integers(1, 25, n)
    rows = np.repeat(np.arange(n), copies)
    classes = [f"c{i}" for i in range(k)]
    weighted = evaluate.weighted_classification_metrics(y, predicted, proba, k, copies)
    repeated = evaluate.classification_metrics(y[rows], predicted[rows], proba[rows], k)
    assert set(weighted) == set(repeated)
    assert all(_same(weighted[m], repeated[m]) for m in repeated), (weighted, repeated)
    per_class = evaluate.weighted_per_class_metrics(y, predicted, proba, classes, copies)
    plain = evaluate.per_class_metrics(y[rows], predicted[rows], proba[rows], classes)
    assert list(per_class.columns) == list(evaluate.TRAFFIC_PER_CLASS_COLUMNS)
    np.testing.assert_array_equal(per_class["flows"].to_numpy(), plain["support"].to_numpy().astype(np.float64))
    for column in ("precision", "recall", "f1", "roc_auc", "average_precision"):
        np.testing.assert_array_equal(per_class[column].to_numpy(), plain[column].to_numpy())
    from sklearn.metrics import confusion_matrix

    np.testing.assert_array_equal(evaluate.weighted_confusion(y, predicted, k, copies),
                                  confusion_matrix(y[rows], predicted[rows], labels=list(range(k))))


def test_standard_errors_reduce_to_the_familiar_ones_and_grow_with_heavy_rows() -> None:
    y = np.array([0] * 60 + [1] * 40)
    guess = y.copy()
    guess[:6] = 1  # 54 of 60 normal rows right
    guess[60:64] = 0  # 36 of 40 attack rows right
    plain = evaluate.weighted_standard_errors(y, guess, np.ones(100))
    assert plain["accuracy"] == pytest.approx(np.sqrt(0.9 * 0.1 / 100))
    assert plain["balanced_accuracy"] == pytest.approx(np.sqrt(0.9 * 0.1 / 60 + 0.9 * 0.1 / 40) / 2)
    heavy = np.ones(100)
    heavy[60] = 200.0  # one misread attack row standing for 200 flows
    assert evaluate.weighted_standard_errors(y, guess, heavy)["accuracy"] > 5 * plain["accuracy"]
    assert np.isnan(evaluate.weighted_standard_errors(y, guess, np.zeros(100))["accuracy"])


def test_bad_weights_are_refused() -> None:
    y = np.array([0, 1, 1, 0])
    proba = np.array([[0.9, 0.1], [0.2, 0.8], [0.6, 0.4], [0.7, 0.3]])
    with pytest.raises(ValueError, match="finite"):
        evaluate.weighted_classification_metrics(y, proba.argmax(1), proba, 2, [1, -1, 1, 1])
    with pytest.raises(ValueError, match="Expected 4 weights"):
        evaluate.weighted_classification_metrics(y, proba.argmax(1), proba, 2, [1, 1])
    empty = evaluate.weighted_classification_metrics(y, proba.argmax(1), proba, 2, [0, 0, 0, 0])
    assert all(np.isnan(v) for v in empty.values())


# --------------------------------------------------------------------------------------------------------------
# Weights of a real fit
# --------------------------------------------------------------------------------------------------------------
def test_with_every_class_whole_the_weights_are_the_repeat_counts_and_the_readings_those_of_the_repeats(
        whole: tuple[PreparedDataset, TrainingRun, int]) -> None:
    prepared, run, recorded = whole
    assert prepared.sampling.before == prepared.sampling.after  # nothing thinned
    assert int(prepared.copies.sum()) == recorded  # every row of the files accounted for once
    copies = np.asarray(run.data.test_copies)
    assert (copies > 1).any()  # repeats did land among the held-out rows
    assert run.data.reports["model_space_duplicates"]["rows_removed"] == 2  # the port-only twins were merged
    assert evaluate.sampling_fractions(run) == {"BENIGN": 1.0, "FTP-Patator": 1.0}
    weights = evaluate.traffic_weights(run)
    assert weights is not None and weights.dtype == np.float64
    np.testing.assert_array_equal(weights, copies.astype(np.float64))

    fits = _fits()
    traffic = evaluate.traffic_readings(run)
    assert _fits() == fits  # nothing is fitted
    assert traffic is not None and list(traffic) == list(CHANNELS)
    rows = np.repeat(np.arange(len(copies)), copies)
    y = np.asarray(run.data.y_test)
    for key, reading in traffic.items():
        result = run.channels[key]
        repeated = evaluate.classification_metrics(y[rows], result.y_pred[rows], result.proba[rows], 2)
        assert all(_same(reading.metrics[m], repeated[m]) for m in repeated), key
        assert reading.flows == float(copies.sum()) and reading.n_test == len(y)
        assert reading.flows_by_class == {"Normal": float(copies[y == 0].sum()), "Attack": float(copies[y == 1].sum())}
        np.testing.assert_array_equal(reading.confusion.sum(axis=1), [copies[y == 0].sum(), copies[y == 1].sum()])
    again = evaluate.traffic_readings(run)
    assert again is not None and all(again[k] is traffic[k] for k in traffic)  # computed once, kept on the run
    assert evaluate.cached_traffic_readings(run) is not None

    summary = evaluate.traffic_summary(run)
    assert summary is not None and summary.rows == len(y) and summary.flows == float(copies.sum())
    assert summary.repeats == int(copies.sum()) and summary.largest == float(copies.max())
    assert summary.largest_share == float(copies.max()) / float(copies.sum())
    in_class = [copies[i] / copies[y == y[i]].sum() for i in range(len(y))]
    assert summary.balanced_swing == pytest.approx(max(in_class) / 2)
    assert "stand for" in evaluate.traffic_sentence(summary)
    # A few dozen held-out rows, some standing for several flows: one verdict moves the estimate a lot.
    assert summary.concentrated
    warning = evaluate.concentration_sentence(summary)
    assert warning is not None and "heaviest held-out row" in warning and "rough" in warning


def test_weight_spread_over_many_rows_raises_no_warning() -> None:
    spread = evaluate.TrafficSummary(rows=40_000, flows=160_000.0, repeats=50_000, by_class={"Normal": 1e5,
                                     "Attack": 6e4}, largest=320.0, largest_share=0.002, balanced_swing=0.004,
                                     rows_for_half=15_000, sampled_classes={})
    assert not spread.concentrated and evaluate.concentration_sentence(spread) is None
    heavy = replace(spread, largest=4_600.0, largest_share=0.028, balanced_swing=0.04, largest_copies=1_472,
                    largest_fraction=0.32)
    warning = evaluate.concentration_sentence(heavy)
    assert warning is not None and "weighs about 4,600 estimated recorded flows" in warning
    assert "its 1,472 copies in the cleaned files divided by its class's sampling share, 0.320" in warning
    assert "2.8% of the weight" in warning
    assert "by up to 0.028" in warning and "balanced accuracy by up to 0.040" in warning
    assert "rough" in warning  # without a heavy-flow account the run cannot see the flows that missed the rows
    # Estimated flows always read as approximate, rounded to three significant figures.
    assert evaluate.flows_text(166_399.1) == "about 166,000" and evaluate.flows_text(1_312_000) == "about 1.3 million"
    assert evaluate.flows_text(812.4) == "about 812" and evaluate.flows_text(4_627.2) == "about 4,630"


def test_a_thinned_class_is_scaled_up_by_its_sampling_share(thinned: tuple[PreparedDataset, TrainingRun]) -> None:
    prepared, run = thinned
    assert prepared.sampling.before == {"BENIGN": 300, "FTP-Patator": 80}
    assert prepared.sampling.after == {"BENIGN": 120, "FTP-Patator": 80}
    assert run.data.reports["sampling"] == {"merge_web_attacks": False, "before": {"BENIGN": 300, "FTP-Patator": 80},
                                            "after": {"BENIGN": 120, "FTP-Patator": 80}}
    assert evaluate.sampling_fractions(run) == {"BENIGN": 0.4, "FTP-Patator": 1.0}
    weights = evaluate.traffic_weights(run)
    assert weights is not None
    y = np.asarray(run.data.y_test)
    copies = np.asarray(run.data.test_copies, dtype=np.float64)
    normal, attack = y == 0, y == 1
    np.testing.assert_array_equal(copies[normal], 1.0)  # no normal flow repeats
    np.testing.assert_array_equal(weights[normal], 2.5)  # 1 / 0.4
    np.testing.assert_array_equal(weights[attack], copies[attack])  # taken whole: just the repeats
    expected = 2.5 * int(normal.sum()) + float(copies[attack].sum())
    summary = evaluate.traffic_summary(run)
    assert summary is not None and summary.flows == expected
    assert summary.by_class == {"Normal": 2.5 * int(normal.sum()), "Attack": float(copies[attack].sum())}
    traffic = evaluate.traffic_readings(run)
    assert traffic is not None and all(t.flows == expected for t in traffic.values())
    # Scaling every normal row by 2.5 is the same as repeating the readings with those weights.
    for key, reading in traffic.items():
        result = run.channels[key]
        direct = evaluate.weighted_classification_metrics(y, result.y_pred, result.proba, 2, weights)
        assert reading.metrics == direct


def test_merged_web_attack_types_take_their_merged_class_share(tmp_path: Path) -> None:
    rows = (make_rows({"BENIGN": 40}) + make_rows({"Web Attack - Brute Force": 30}, start=100)
            + make_rows({"Web Attack - XSS": 20}, start=200))
    write_cic_csv(tmp_path / THURSDAY, rows + rows[40:46])  # six brute-force flows recorded twice
    prepared = prepare_dataset(DataRequest(source="cicids", data_dir=str(tmp_path), files=(THURSDAY,),
                                           merge_web_attacks=True, seed=SEED))
    run = _fit(prepared, mode="multiclass", min_class_count=10, channels=("logreg",))
    assert run.data.reports["sampling"]["merge_web_attacks"] is True
    assert set(run.data.reports["sampling"]["before"]) == {"BENIGN", "Web Attack"}
    weights = evaluate.traffic_weights(run)
    assert weights is not None
    np.testing.assert_array_equal(weights, np.asarray(run.data.test_copies, dtype=np.float64))
    assert set(np.asarray(run.data.detailed_test_labels)) > {"BENIGN"}  # detailed labels mapped to "Web Attack"


def test_the_weights_and_their_summary_are_computed_once_per_run(thinned: tuple[PreparedDataset, TrainingRun],
                                                                    monkeypatch: pytest.MonkeyPatch) -> None:
    _, fitted = thinned
    run = replace(fitted, run_id=fitted.run_id + "-once")  # a fresh run object: nothing kept on it yet
    calls: list[int] = []
    real = evaluate._traffic_basis

    def counted(target: Any) -> Any:
        calls.append(1)
        return real(target)

    monkeypatch.setattr(evaluate, "_traffic_basis", counted)
    first = evaluate.traffic_summary(run)
    for _ in range(3):  # what 03 Measure asks for on every redraw
        assert evaluate.traffic_unavailable_reason(run) is None
        assert evaluate.traffic_summary(run) is first
        assert evaluate.traffic_readings(run) is not None
        weights = evaluate.traffic_weights(run)
        assert weights is not None and weights.flags.writeable  # a copy: changing it cannot touch the kept one
    assert len(calls) == 1
    other = replace(run, data=replace(run.data, test_copies=None))  # other matrices: a fresh basis
    assert evaluate.traffic_unavailable_reason(other) is not None and len(calls) == 2


def _heavy_tail_file(folder: Path) -> None:
    """A Wednesday-named file: 240 normal flows, 160 attack flows, and four attack flows that look like normal ones,
    each recorded 150 times (600 of the 760 attack rows)."""
    normal = make_rows({"BENIGN": 240})
    attacks = make_rows({"DoS Hulk": 160}, start=2_000)
    heavy = [with_values(normal[i * 50], {"Label": "DoS Hulk", "Flow Duration": 7_777.0 + i}) for i in range(4)]
    write_cic_csv(folder / WEDNESDAY, normal + attacks + [row for row in heavy for _ in range(150)])


def test_heavily_repeated_flows_missing_from_the_held_out_rows_raise_the_caution_and_a_range_that_holds(
        tmp_path: Path) -> None:
    _heavy_tail_file(tmp_path)
    prepared = prepare_dataset(DataRequest(source="cicids", data_dir=str(tmp_path), files=(WEDNESDAY,), seed=SEED))
    profile = prepared.repeat_profile
    assert profile is not None and profile["classes"]["DoS Hulk"]["heavy"] == [150] * 4
    # The first split seed that holds out none of the four heavy flows (fixed: the split is seeded).
    request = None
    for seed in range(40):
        candidate = TrainRequest(profile="test", seed=seed, channels=("logreg",))
        report = build_training_data(prepared, candidate).reports["heavy_flows"]["classes"]["Attack"]
        if report["held_out"] == 0:
            request = candidate
            break
    assert request is not None
    data = build_training_data(prepared, request)
    account = data.reports["heavy_flows"]["classes"]["Attack"]
    assert account["flows"] == 4 and account["copies"] == 600 and account["recorded"] == 760
    assert account["held_out"] == 0 and account["in_training"] == 4 and account["rows"] == []
    run = train_all(data, request, data_request=prepared.request, dataset_fingerprint=prepared.fingerprint)

    summary = evaluate.traffic_summary(run)
    assert summary is not None and summary.heavy_concentrated and summary.concentrated
    assert summary.recorded == 1_000 and summary.heavy_accuracy_swing == pytest.approx(600 / 1_000)
    warning = evaluate.concentration_sentence(summary)
    assert warning is not None and "none of them is among the held-out rows" in warning
    assert "05 Assay" in warning and "standard errors cannot show" in warning
    reading = evaluate.traffic_readings(run)["logreg"]  # type: ignore[index]
    # What the channel reads over every row the file recorded: the target of the estimate.
    frame, _ = read_flow_csv(tmp_path / WEDNESDAY)
    truth = (frame["Label"].astype(str) != "BENIGN").to_numpy().astype(np.int64)
    guess = run.channels["logreg"].estimator.predict_proba(
        frame[list(run.data.feature_names)].to_numpy(dtype=np.float32)).argmax(axis=1)
    accuracy = float((guess == truth).mean())
    balanced = float(np.mean([(guess[truth == c] == c).mean() for c in (0, 1)]))
    # The held-out rows alone miss it by far more than their standard error says (the channel misreads the heavy
    # flows, which only the files know about) ...
    assert abs(reading.metrics["accuracy"] - accuracy) > 2 * reading.errors["accuracy"] + 0.1
    # ... while the range that accounts for them holds the reading over every recorded row.
    low, high = reading.bounds["accuracy"]
    assert low - 1e-12 <= accuracy <= high + 1e-12, (low, accuracy, high)
    low, high = reading.bounds["balanced_accuracy"]
    assert low - 1e-12 <= balanced <= high + 1e-12, (low, balanced, high)
    # The range is as wide as the share of the class no held-out verdict covers.
    assert high - low == pytest.approx(summary.heavy_balanced_swing)
    combined = exports.leaderboard_frame(run)
    assert {"traffic_accuracy_low", "traffic_accuracy_high"} <= set(combined.columns)


# --------------------------------------------------------------------------------------------------------------
# Unavailable
# --------------------------------------------------------------------------------------------------------------
def test_without_repeat_counts_or_sampling_shares_there_is_no_estimate_and_the_reason_is_given(
        whole: tuple[PreparedDataset, TrainingRun, int]) -> None:
    _, run, _ = whole
    no_counts = replace(run, run_id=run.run_id + "-n", data=replace(run.data, test_copies=None))
    assert evaluate.traffic_weights(no_counts) is None and evaluate.traffic_readings(no_counts) is None
    assert evaluate.traffic_summary(no_counts) is None
    reason = evaluate.traffic_unavailable_reason(no_counts)
    assert reason is not None and "distinct-flow readings" in reason

    reports = {k: v for k, v in run.data.reports.items() if k != "sampling"}
    no_shares = replace(run, run_id=run.run_id + "-s", data=replace(run.data, reports=reports))
    assert evaluate.traffic_readings(no_shares) is None and evaluate.sampling_fractions(no_shares) is None
    assert "01 Sample" in (evaluate.traffic_unavailable_reason(no_shares) or "")

    data = run.data
    hollow = replace(run, run_id=run.run_id + "-h", data=replace(
        data, X_test=np.empty((0, data.n_features), dtype=np.float32), y_test=np.empty(0, dtype=np.int64),
        test_rows=np.empty(0, dtype=np.int64), detailed_test_labels=np.empty(0, dtype=str)))
    assert evaluate.traffic_readings(hollow) is None
    assert evaluate.traffic_unavailable_reason(hollow) == "This run holds no held-out rows."
    assert evaluate.traffic_unavailable_reason(run) is None

    # The exports keep their distinct-flow columns and simply leave the estimate out.
    board = exports.leaderboard_frame(no_counts)
    assert not [c for c in board.columns if str(c).startswith(evaluate.TRAFFIC_PREFIX)]
    per_class = pd.read_csv(io.BytesIO(exports.per_class_csv(no_counts)), encoding="utf-8-sig")
    assert "traffic_flows" not in per_class.columns and "support" in per_class.columns


# --------------------------------------------------------------------------------------------------------------
# Exports, manifest, PDF
# --------------------------------------------------------------------------------------------------------------
def test_the_exports_carry_traffic_columns(thinned: tuple[PreparedDataset, TrainingRun]) -> None:
    _, run = thinned
    traffic = evaluate.traffic_readings(run)
    assert traffic is not None
    board = exports.leaderboard_frame(run)
    names = [f"traffic_{metric}" for metric, _ in evaluate.score_columns("binary")]
    assert [c for c in board.columns if c.startswith("traffic_")] == [
        "traffic_flows_represented", *names, "traffic_balanced_accuracy_se", "traffic_accuracy_se",
        "traffic_balanced_accuracy_low", "traffic_balanced_accuracy_high", "traffic_accuracy_low",
        "traffic_accuracy_high"]
    assert board["traffic_accuracy_se"].notna().all()
    assert list(board["key"]) == [*list(evaluate.leaderboard(evaluate.evaluate_run(run), run)["key"]), "consensus"]
    for _, row in board.iterrows():
        if row["key"] == exports.CONSENSUS_KEY:
            combined = exports.consensus_traffic_metrics(run)
            assert combined is not None and row["traffic_balanced_accuracy"] == combined["balanced_accuracy"]
        else:
            assert row["traffic_balanced_accuracy"] == traffic[row["key"]].metrics["balanced_accuracy"]
        assert row["traffic_flows_represented"] == traffic["forest"].flows
    read_back = pd.read_csv(io.BytesIO(exports.leaderboard_csv(run)), encoding="utf-8-sig")
    assert "traffic_balanced_accuracy" in read_back.columns and len(read_back) == len(board)
    per_class = pd.read_csv(io.BytesIO(exports.per_class_csv(run)), encoding="utf-8-sig")
    assert {"support", "traffic_flows", "traffic_precision", "traffic_recall", "traffic_f1", "traffic_roc_auc",
            "traffic_average_precision"} <= set(per_class.columns)
    normal = per_class[(per_class["key"] == "forest") & (per_class["class"] == "Normal")].iloc[0]
    assert normal["traffic_flows"] == pytest.approx(2.5 * normal["support"])
    assert "traffic_" in exports.readme_text([], run)


def test_a_saved_set_records_the_estimate_and_an_older_one_regains_its_shares(
        thinned: tuple[PreparedDataset, TrainingRun], tmp_path: Path) -> None:
    prepared, run = thinned
    traffic = evaluate.traffic_readings(run)
    assert traffic is not None
    fits = _fits()
    folder = persist.save_run(run, tmp_path)
    bundle = persist.load_bundle(folder)
    for key in CHANNELS:
        saved = bundle.manifest["channels"][key]["evaluation_metrics"]
        assert saved["traffic_balanced_accuracy"] == pytest.approx(traffic[key].metrics["balanced_accuracy"])
        assert saved["traffic_flows_represented"] == pytest.approx(traffic[key].flows)
    assert bundle.manifest["reports"]["sampling"] == run.data.reports["sampling"]
    # A bundle from before the shares were kept: the rebuilt (fingerprint-checked) sample supplies them.
    older = replace(bundle, manifest={**bundle.manifest, "reports": {
        k: v for k, v in bundle.manifest["reports"].items() if k != "sampling"}})
    data = persist.rebuild_training_data(older, data_dir=None, prepared=prepared)
    restored = persist.restore_run(older, data)
    assert restored.data.reports["sampling"] == run.data.reports["sampling"]
    again = evaluate.traffic_readings(restored)
    assert again is not None
    for key in CHANNELS:
        assert again[key].metrics == pytest.approx(traffic[key].metrics, nan_ok=True)
    assert _fits() == fits  # saving, loading and weighting fit nothing


@pytest.fixture
def blank_charts(monkeypatch: pytest.MonkeyPatch) -> None:
    """Charts of the PDF record are built but rendered as blank pictures (the real rendering is tested elsewhere)."""

    def blank_png(chart: Any, scale: float = 2, *, background: str | None = None) -> bytes:
        chart.to_dict(validate=False)
        buffer = io.BytesIO()
        Image.new("RGB", (600, 300), "#FFFFFF").save(buffer, format="PNG")
        return buffer.getvalue()

    monkeypatch.setattr(viz, "to_png", blank_png)


def _record_parts(run: TrainingRun, monkeypatch: pytest.MonkeyPatch) -> tuple[list[Any], list[str]]:
    """Build the record of ``run`` and return the tables (headings, rows) and the subheadings and notes it wrote."""
    tables: list[Any] = []
    texts: list[str] = []
    real_table, real_note, real_sub = rp._Builder.table, rp._Builder.note, rp._Builder.subheading

    def table(self: Any, headers: Any, rows: Any, **options: Any) -> None:
        tables.append((list(headers), [list(r) for r in rows], list(options.get("bold_rows", ()))))
        real_table(self, headers, rows, **options)

    def note(self: Any, text: str) -> None:
        texts.append(text)
        real_note(self, text)

    def subheading(self: Any, title: str) -> None:
        texts.append(title)
        real_sub(self, title)

    monkeypatch.setattr(rp._Builder, "table", table)
    monkeypatch.setattr(rp._Builder, "note", note)
    monkeypatch.setattr(rp._Builder, "subheading", subheading)
    report = rp.render_report(run, evaluate.evaluate_run(run), prepared_summary=None, settings={}, compress=False)
    assert report.pages >= 3 and not report.problems
    return tables, texts


def test_the_pdf_record_holds_the_recorded_traffic_table(thinned: tuple[PreparedDataset, TrainingRun],
                                                         blank_charts: None, monkeypatch: pytest.MonkeyPatch) -> None:
    _, run = thinned
    fits = _fits()
    tables, texts = _record_parts(run, monkeypatch)
    assert _fits() == fits
    assert "Recorded traffic (estimate)" in texts
    position = texts.index("Recorded traffic (estimate)")
    assert "stand for" in texts[position + 1] and "01 Sample" in texts[position + 1]
    leaderboard, weighted = [t for t in tables if t[0][:2] == ["Channel", "Bal. acc."]][:2]
    assert weighted[0] == ["Channel", "Bal. acc.", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "Bal. s.e.",
                           "Acc. s.e."]
    assert [r[0] for r in weighted[1]] == [*[r[0] for r in leaderboard[1]]]  # same rows, consensus included
    assert weighted[1][-1][0].startswith("Consensus") and weighted[2] == [len(weighted[1]) - 1]
    assert leaderboard[2] == [len(leaderboard[1]) - 1]  # the consensus row is set in bold
    traffic = evaluate.traffic_readings(run)
    assert traffic is not None
    first = weighted[1][0]
    key = next(k for k in CHANNELS if evaluate.channel_label(k) == first[0])
    assert first[1] == rp._score(traffic[key].metrics["balanced_accuracy"])


def test_the_pdf_record_says_why_the_estimate_is_missing(whole: tuple[PreparedDataset, TrainingRun, int],
                                                         blank_charts: None, monkeypatch: pytest.MonkeyPatch) -> None:
    _, run, _ = whole
    no_counts = replace(run, run_id=run.run_id + "-p", data=replace(run.data, test_copies=None))
    _, texts = _record_parts(no_counts, monkeypatch)
    position = texts.index("Recorded traffic (estimate)")
    assert texts[position + 1].startswith("Not available for this run:")
