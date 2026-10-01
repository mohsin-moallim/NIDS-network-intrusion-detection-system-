"""02 Fit on small synthetic samples: matrices, all five channels, leakage guards, failures and determinism.

Every fit uses ``profile="test"`` (tiny models) on about 3,000 generated flows, so the whole module runs in
seconds. The destination port stays out of every feature set here: in the generator it nearly identifies some
attacks.
"""

from __future__ import annotations

import pickle
import warnings
from collections import Counter
from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd
import pytest

from graticule.data import sampling
from graticule.data.clean import row_hashes
from graticule.data.prepare import DataRequest, PreparedDataset, prepare_dataset
from graticule.data.sampling import SingleClassError
from graticule.evaluate import quick_metrics
from graticule.features import rank_features
from graticule.models import train as train_mod
from graticule.models import zoo
from graticule.models.jobs import STAGE_KEY, CancelToken
from graticule.models.train import (
    FIT_CALLS,
    ChannelResult,
    TrainingData,
    TrainingRun,
    TrainRequest,
    build_training_data,
    fit_model,
    new_run_id,
    train_all,
)
from graticule.models.zoo import MODEL_KEYS, BuildContext
from graticule.schema import BENIGN, DESTINATION_PORT, LABEL

pytestmark = pytest.mark.integration
SEED = 11


@pytest.fixture(scope="module")
def prepared() -> PreparedDataset:
    """About 3,000 synthetic flows (six classes)."""
    return prepare_dataset(DataRequest(source="synthetic", synthetic_flows=3_000, seed=SEED))


def _request(**changes: object) -> TrainRequest:
    base: dict[str, object] = dict(profile="test", seed=SEED)
    base.update(changes)
    return TrainRequest(**base)  # type: ignore[arg-type]


def _fit(prepared: PreparedDataset, request: TrainRequest) -> tuple[TrainingData, TrainingRun]:
    data = build_training_data(prepared, request)
    run = train_all(data, request, data_request=prepared.request, dataset_fingerprint=prepared.fingerprint)
    return data, run


@pytest.fixture(scope="module")
def runs(prepared: PreparedDataset) -> dict[str, tuple[TrainingData, TrainingRun]]:
    """All five channels fitted once per mode."""
    return {mode: _fit(prepared, _request(mode=mode)) for mode in ("binary", "multiclass")}


def _with_frame(prepared: PreparedDataset, frame: pd.DataFrame) -> PreparedDataset:
    """A copy of ``prepared`` holding different rows (the fingerprint is left as it was)."""
    return replace(prepared, frame=frame.reset_index(drop=True))


# --------------------------------------------------------------------------------------------------------------
# Requests
# --------------------------------------------------------------------------------------------------------------
def test_request_normalises_channels_and_rejects_bad_options() -> None:
    assert TrainRequest(channels=("logreg", "forest", "logreg")).channels == ("forest", "logreg")
    assert TrainRequest().channels == MODEL_KEYS
    for bad in (dict(channels=("knn",)), dict(channels=()), dict(mode="ternary"), dict(test_share=1.0),
                dict(feature_mode="best"), dict(top_k=0), dict(conflict_policy="vote"), dict(svm_cap=0)):
        with pytest.raises(ValueError):
            TrainRequest(**bad)  # type: ignore[arg-type]
    assert TrainRequest(seed=3).to_dict()["channels"] == list(MODEL_KEYS)


def test_run_ids_look_like_timestamps() -> None:
    run_id = new_run_id()
    date, clock, tail = run_id.split("-")
    assert len(date) == 8 and date.isdigit() and len(clock) == 6 and clock.isdigit()
    assert len(tail) == 4 and int(tail, 16) >= 0


# --------------------------------------------------------------------------------------------------------------
# All channels, both modes
# --------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("mode", ["binary", "multiclass"])
def test_all_five_channels_train_and_score(runs: dict[str, tuple[TrainingData, TrainingRun]], mode: str) -> None:
    data, run = runs[mode]
    assert run.ok_channels() == list(MODEL_KEYS), {k: r.error for k, r in run.channels.items()}
    assert not run.cancelled
    n_test, k = len(data.y_test), data.n_classes
    for key, result in run.channels.items():
        assert isinstance(result, ChannelResult) and result.ok
        assert result.proba is not None and result.y_pred is not None
        assert result.proba.shape == (n_test, k) and result.proba.dtype == np.float32
        np.testing.assert_allclose(result.proba.sum(axis=1), 1.0, atol=1e-5)
        assert (result.proba >= 0).all()
        assert np.array_equal(result.y_pred, result.proba.argmax(axis=1))
        assert result.fit_seconds > 0 and result.predict_seconds > 0
        assert 0 < result.rows_used <= result.rows_available == len(data.y_train)
        metrics = result.extra["metrics"]
        assert set(metrics) == {"accuracy", "balanced_accuracy", "f1_macro", "f1_weighted"}
        assert metrics["balanced_accuracy"] > 0.5, key
        assert result.extra["flows_per_second"] > 0
    # The bigger channels separate the generated classes well even in the test profile.
    for key in ("forest", "xgboost", "logreg"):
        assert run.channels[key].extra["metrics"]["balanced_accuracy"] > 0.85, key
    assert run.prep_seconds == pytest.approx(data.reports["seconds"])
    assert run.total_seconds == pytest.approx(run.seconds + data.reports["seconds"])
    # No channel carries notes about the calibration folds or the frozen SVM's weights.
    every_note = " ".join(n for r in run.channels.values() for n in r.notes)
    assert "n_splits" not in every_note and "least populated" not in every_note
    assert "does not appear to accept sample_weight" not in every_note


def test_class_order_and_codes(runs: dict[str, tuple[TrainingData, TrainingRun]]) -> None:
    binary, _ = runs["binary"]
    assert binary.classes == ("Normal", "Attack")
    multi, _ = runs["multiclass"]
    assert multi.classes[0] == BENIGN
    assert list(multi.classes[1:]) == sorted(multi.classes[1:], key=str.lower)
    for data in (binary, multi):
        assert data.y_train.dtype == np.int64 and data.y_test.dtype == np.int64
        assert set(np.unique(data.y_train)) == set(range(data.n_classes)) == set(np.unique(data.y_test))


@pytest.mark.parametrize("mode", ["binary", "multiclass"])
def test_matrices_rows_and_labels_line_up(prepared: PreparedDataset,
                                          runs: dict[str, tuple[TrainingData, TrainingRun]], mode: str) -> None:
    data, run = runs[mode]
    assert data.X_train.dtype == np.float32 and data.X_test.dtype == np.float32
    assert data.X_train.shape == (len(data.y_train), data.n_features) and data.X_train.flags.writeable
    assert DESTINATION_PORT not in data.feature_names
    assert data.feature_names == data.feature_choice.columns
    assert not set(data.train_rows) & set(data.test_rows)
    frame = prepared.frame
    expected = frame[list(data.feature_names)].to_numpy(dtype=np.float32)
    np.testing.assert_array_equal(data.X_test, expected[data.test_rows])
    np.testing.assert_array_equal(data.X_train, expected[data.train_rows])
    detailed = frame[LABEL].to_numpy(dtype=object)[data.test_rows].astype(str)
    assert np.array_equal(data.detailed_test_labels, detailed)
    names = np.asarray(data.classes, dtype=object)[data.y_test]
    if mode == "binary":
        assert np.array_equal(names == "Normal", detailed == BENIGN)
    else:
        assert np.array_equal(names.astype(str), detailed)
    counts = data.reports["class_counts_test"]
    assert list(counts) == list(data.classes) and sum(counts.values()) == len(data.y_test)
    assert run.dataset_fingerprint == prepared.fingerprint and run.data_request == prepared.request


def test_reference_sample_and_quantiles(runs: dict[str, tuple[TrainingData, TrainingRun]]) -> None:
    data, run = runs["multiclass"]
    ref = run.reference_sample
    assert ref.dtype == np.float32 and ref.shape == (min(2_000, len(data.y_train)), data.n_features)
    train_rows = {row.tobytes() for row in data.X_train}
    assert all(row.tobytes() in train_rows for row in ref), "reference rows come from the training split"
    q = run.feature_quantiles
    assert q.shape == (101, data.n_features) and q.dtype == np.float32
    assert np.all(np.diff(q, axis=0) >= 0)
    np.testing.assert_allclose(q[0], np.nanmin(data.X_train, axis=0))
    np.testing.assert_allclose(q[-1], np.nanmax(data.X_train, axis=0))


def test_fitted_channels_pickle_and_hold_no_callbacks(runs: dict[str, tuple[TrainingData, TrainingRun]]) -> None:
    data, run = runs["binary"]
    assert run.channels["xgboost"].estimator.named_steps["model"].get_params()["callbacks"] is None
    assert not hasattr(run.channels["logreg"].estimator.named_steps["model"], "_skl_callbacks")
    assert "epoch_observer" not in vars(run.channels["mlp"].estimator.named_steps["model"])
    assert run.channels["forest"].extra["n_trees"] == 20
    for key, result in run.channels.items():
        clone = pickle.loads(pickle.dumps(result.estimator))
        np.testing.assert_allclose(clone.predict_proba(data.X_test[:20]), result.estimator.predict_proba(
            data.X_test[:20]), atol=1e-6, err_msg=key)


# --------------------------------------------------------------------------------------------------------------
# SVM cap
# --------------------------------------------------------------------------------------------------------------
def test_svm_cap_is_honoured_and_noted(prepared: PreparedDataset) -> None:
    data, run = _fit(prepared, _request(mode="multiclass", channels=("svm",), svm_cap=400))
    svm = run.channels["svm"]
    assert svm.ok, svm.error
    assert svm.rows_used == svm.extra["svm_rows_used"] <= 400
    assert svm.rows_available == len(data.y_train) > 400
    assert any("SVM cap" in note and "400" in note for note in svm.notes), svm.notes
    assert 0 < svm.extra["calibration_rows"] <= 5_000
    assert svm.rows_used + svm.extra["calibration_rows"] <= len(data.y_train)
    assert svm.proba is not None and svm.proba.shape == (len(data.y_test), data.n_classes)


def test_stratified_take_is_exact_and_keeps_every_class() -> None:
    y = np.array([0] * 900 + [1] * 95 + [2] * 3 + [3] * 2)
    picked = train_mod._stratified_take(y, 250, seed=1, leave=1)
    assert len(picked) == 250 and len(np.unique(picked)) == 250
    taken = np.bincount(y[picked], minlength=4)
    assert (taken >= 1).all() and (np.bincount(y, minlength=4) - taken >= 1).all()


# --------------------------------------------------------------------------------------------------------------
# Model-space de-duplication, conflicts and class re-checks
# --------------------------------------------------------------------------------------------------------------
def test_dedupe_removes_rows_made_identical_by_leaving_out_the_port(prepared: PreparedDataset) -> None:
    base_data = build_training_data(prepared, _request(mode="multiclass"))
    frame = prepared.frame
    # Copy 40 kept rows, changing only the destination port: identical over the curated columns.
    copied = base_data.train_rows[:40]
    extra = frame.iloc[copied].copy()
    extra[DESTINATION_PORT] = (extra[DESTINATION_PORT] + 1_000).astype(np.float32)
    bigger = _with_frame(prepared, pd.concat([frame, extra], ignore_index=True))
    added = set(range(len(frame), len(frame) + 40))

    curated = build_training_data(bigger, _request(mode="multiclass"))
    before = base_data.reports["model_space_duplicates"]
    after = curated.reports["model_space_duplicates"]
    assert after["rows_removed"] == before["rows_removed"] + 40
    expected = Counter(before["by_class"]) + Counter(frame[LABEL].iloc[copied].astype(str).tolist())
    assert after["by_class"] == dict(expected)
    assert not added & (set(curated.train_rows) | set(curated.test_rows))

    with_port = build_training_data(bigger, _request(mode="multiclass", include_port=True))
    assert DESTINATION_PORT in with_port.feature_names and with_port.feature_choice.include_port
    port_only = build_training_data(prepared, _request(mode="multiclass", include_port=True))
    assert with_port.reports["model_space_duplicates"]["rows_removed"] == \
        port_only.reports["model_space_duplicates"]["rows_removed"]
    assert added <= set(with_port.train_rows) | set(with_port.test_rows)


def test_conflicting_labels_are_counted_and_follow_the_policy(prepared: PreparedDataset) -> None:
    frame = prepared.frame
    data = build_training_data(prepared, _request(mode="binary"))
    benign_rows = [int(r) for r in data.train_rows if frame[LABEL].iloc[r] == BENIGN][:25]
    twins = frame.iloc[benign_rows].copy()
    twins[LABEL] = pd.Series(["Flood"] * len(twins), index=twins.index, dtype="str")
    bigger = _with_frame(prepared, pd.concat([frame, twins], ignore_index=True))

    kept = build_training_data(bigger, _request(mode="binary"))
    assert kept.reports["conflicts"]["groups"] >= 25 and kept.reports["conflicts"]["rows_removed"] == 0
    assert kept.reports["conflicts"]["by_class"]["Attack"] >= 25
    dropped = build_training_data(bigger, _request(mode="binary", conflict_policy="drop"))
    assert dropped.reports["conflicts"]["rows_removed"] == kept.reports["conflicts"]["rows"]
    assert len(dropped.y_train) + len(dropped.y_test) == len(kept.y_train) + len(kept.y_test) - \
        kept.reports["conflicts"]["rows"]


def test_multiclass_drops_small_classes_and_reports_them(prepared: PreparedDataset) -> None:
    smallest = min(prepared.class_counts, key=prepared.class_counts.get)  # type: ignore[arg-type]
    count = prepared.class_counts[smallest]
    data = build_training_data(prepared, _request(mode="multiclass", min_class_count=count + 1))
    assert smallest not in data.classes
    assert data.reports["target"]["dropped"] == {smallest: count}
    assert "minimum" in data.reports["target"]["dropped_reasons"][smallest]


def test_benign_only_binary_data_raises_a_friendly_error(prepared: PreparedDataset) -> None:
    frame = prepared.frame
    benign = _with_frame(prepared, frame[frame[LABEL] == BENIGN])
    with pytest.raises(SingleClassError) as info:
        build_training_data(benign, _request(mode="binary"))
    assert BENIGN in str(info.value) and "attack" in str(info.value).lower()


# --------------------------------------------------------------------------------------------------------------
# Top-K never looks at test rows
# --------------------------------------------------------------------------------------------------------------
def test_topk_ranks_on_training_rows_only(prepared: PreparedDataset, monkeypatch: pytest.MonkeyPatch) -> None:
    request = _request(mode="binary", feature_mode="topk", top_k=8)
    splits: list[sampling.SplitIndices] = []
    real_split = sampling.stratified_split

    def remembered_split(y: object, test_share: float, seed: int, **kwargs: object) -> sampling.SplitIndices:
        if not splits:
            splits.append(real_split(y, test_share, seed, **kwargs))  # type: ignore[arg-type]
        return splits[0]

    monkeypatch.setattr(sampling, "stratified_split", remembered_split)
    first = build_training_data(prepared, request)
    overlap = first.reports["topk_overlap"]
    assert overlap["k"] == 8 and overlap["test_rows"] == len(first.y_test)
    assert overlap["kept"] == 8 and overlap["port_added"] is False
    assert overlap["candidates"] == len(first.feature_choice.ranking or ())
    assert 0 <= overlap["test_rows_seen_in_train"] <= len(first.y_test)
    assert overlap["ranked_on_rows"] == len(first.y_train)

    # Poison the labels of every test row whose candidate-feature vector occurs only once, so the
    # de-duplication keeps exactly the same rows and the remembered split still applies.
    frame = prepared.frame
    candidates = [name for name, _ in first.feature_choice.ranking or ()]
    hashes = row_hashes(frame, candidates)
    unique = ~pd.Index(hashes).duplicated(keep=False)
    poison = np.array([r for r in first.test_rows if unique[r]])
    assert len(poison) > 0.9 * len(first.test_rows)
    labels = frame[LABEL].to_numpy(dtype=object).copy()
    labels[poison] = np.where(labels[poison] == BENIGN, "Flood", BENIGN)
    poisoned_frame = frame.copy()
    poisoned_frame[LABEL] = pd.Series(labels, dtype="str")
    poisoned = _with_frame(prepared, poisoned_frame)

    second = build_training_data(poisoned, request)
    assert np.array_equal(second.train_rows, first.train_rows)
    assert np.array_equal(second.y_train, first.y_train)
    assert not np.array_equal(second.y_test, first.y_test), "the poison reached the test labels"
    assert second.feature_choice.ranking == first.feature_choice.ranking
    assert second.feature_names == first.feature_names

    # Control: a ranking that did see the poisoned test rows would differ.
    X_all = np.vstack([second.X_train, second.X_test])
    y_all = np.concatenate([second.y_train, second.y_test])
    leaky = rank_features(_candidate_matrix(poisoned, candidates, np.concatenate([second.train_rows,
                                                                                    second.test_rows])),
                          y_all, candidates, seed=SEED)
    assert X_all.shape[0] == len(y_all)
    assert leaky != list(first.feature_choice.ranking or ())


def _candidate_matrix(prepared: PreparedDataset, columns: list[str], rows: np.ndarray) -> np.ndarray:
    return prepared.frame[columns].to_numpy(dtype=np.float32)[rows]


# --------------------------------------------------------------------------------------------------------------
# Failures, cancels, fit counting, determinism
# --------------------------------------------------------------------------------------------------------------
def test_one_failing_channel_does_not_stop_the_run(prepared: PreparedDataset, monkeypatch: pytest.MonkeyPatch) -> None:
    def broken(ctx: zoo.BuildContext) -> object:
        raise RuntimeError("this channel cannot be built")

    monkeypatch.setitem(zoo.BUILDERS, "svm", broken)
    before = Counter(FIT_CALLS)
    data, run = _fit(prepared, _request(mode="binary", channels=("forest", "svm", "logreg")))
    svm = run.channels["svm"]
    assert svm.status == "failed" and "cannot be built" in (svm.error or "")
    assert svm.proba is None and svm.estimator is None and "traceback" in svm.extra
    assert run.ok_channels() == ["forest", "logreg"]
    assert Counter(FIT_CALLS) - before == Counter({"forest": 1, "logreg": 1})


def test_fit_calls_count_once_per_trained_channel(prepared: PreparedDataset) -> None:
    before = Counter(FIT_CALLS)
    _fit(prepared, _request(mode="multiclass", channels=("xgboost", "svm", "mlp")))
    assert Counter(FIT_CALLS) - before == Counter({"xgboost": 1, "svm": 1, "mlp": 1})


def test_cancel_marks_the_current_channel_and_skips_the_rest(prepared: PreparedDataset) -> None:
    request = _request(mode="binary", channels=("forest", "xgboost", "logreg"))
    data = build_training_data(prepared, request)
    token = CancelToken()

    class CancelOnFirstTrees:
        def __init__(self) -> None:
            self.seen: list[tuple[str, str | None]] = []

        def update(self, key: str, *, status: str | None = None, fraction: float | None = None,
                   message: str | None = None) -> None:
            self.seen.append((key, status))
            if key == "forest" and message and "trees" in message:
                token.cancel()

    sink = CancelOnFirstTrees()
    run = train_all(data, request, data_request=prepared.request, dataset_fingerprint=prepared.fingerprint,
                    progress=sink, cancel=token)
    assert run.channels["forest"].status == "cancelled"
    assert [run.channels[k].status for k in ("xgboost", "logreg")] == ["skipped", "skipped"]
    assert run.cancelled and run.ok_channels() == []
    assert (STAGE_KEY, "fitting") in sink.seen and ("forest", "running") in sink.seen


def test_unweighted_fit_is_noted(prepared: PreparedDataset) -> None:
    _, run = _fit(prepared, _request(mode="binary", channels=("logreg",), balanced=False))
    assert any("Unweighted" in note for note in run.channels["logreg"].notes)


def test_predictions_are_deterministic_for_a_seed(prepared: PreparedDataset,
                                                  runs: dict[str, tuple[TrainingData, TrainingRun]]) -> None:
    data_a, run_a = runs["multiclass"]
    data_b, run_b = _fit(prepared, _request(mode="multiclass"))
    assert np.array_equal(data_a.train_rows, data_b.train_rows)
    assert np.array_equal(data_a.X_train, data_b.X_train)
    for key in MODEL_KEYS:
        a, b = run_a.channels[key], run_b.channels[key]
        assert np.array_equal(a.y_pred, b.y_pred), key  # type: ignore[arg-type]
        np.testing.assert_allclose(a.proba, b.proba, atol=1e-6, err_msg=key)  # type: ignore[arg-type]
    np.testing.assert_array_equal(run_a.reference_sample, run_b.reference_sample)


def test_quick_metrics_match_hand_counts() -> None:
    truth = np.array([0, 0, 0, 0, 1, 1, 2, 2])
    guess = np.array([0, 0, 0, 1, 1, 0, 2, 2])
    metrics = quick_metrics(truth, guess, 3)
    assert metrics["accuracy"] == pytest.approx(6 / 8)
    assert metrics["balanced_accuracy"] == pytest.approx((3 / 4 + 1 / 2 + 1) / 3)
    assert 0 < metrics["f1_macro"] < 1 and 0 < metrics["f1_weighted"] < 1
    assert np.isnan(quick_metrics(np.array([]), np.array([]), 2)["accuracy"])
    # Classes that are never true or never predicted raise no warning (none is filtered away either).
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        quick_metrics(np.array([0, 0, 1]), np.array([0, 0, 0]), 4)
        quick_metrics(np.array([0, 0, 0]), np.array([2, 2, 2]), 3)
    assert caught == []
    with pytest.raises(ValueError):
        quick_metrics(truth, guess[:-1], 3)


# --------------------------------------------------------------------------------------------------------------
# Cancel inside a channel, progress inside a fit
# --------------------------------------------------------------------------------------------------------------
class _Sink:
    """Progress sink that records every message and fires ``token`` on the first message for ``key`` containing
    ``text`` (when given)."""

    def __init__(self, token: CancelToken | None = None, key: str = "", text: str = "") -> None:
        self.token, self.key, self.text = token, key, text
        self.messages: list[tuple[str, str | None]] = []

    def update(self, key: str, *, status: str | None = None, fraction: float | None = None,
               message: str | None = None) -> None:
        self.messages.append((key, message))
        if self.token is not None and key == self.key and message and self.text in message:
            self.token.cancel()


def test_mlp_and_logreg_report_epochs_and_iterations(prepared: PreparedDataset) -> None:
    request = _request(mode="binary", channels=("mlp", "logreg"))
    data = build_training_data(prepared, request)
    sink = _Sink()
    run = train_all(data, request, data_request=prepared.request, dataset_fingerprint=prepared.fingerprint,
                    progress=sink)
    assert run.ok_channels() == ["mlp", "logreg"]
    epochs = [m for k, m in sink.messages if k == "mlp" and m and m.startswith("Epoch ")]
    assert epochs and epochs[0].startswith("Epoch 1 of up to 15; validation score ")
    assert len(epochs) == run.channels["mlp"].extra["n_iter"]
    assert ("logreg", "Iteration 1 (at most 200)") in sink.messages


def test_a_cancel_stops_the_mlp_after_its_current_epoch(prepared: PreparedDataset) -> None:
    request = _request(mode="binary", channels=("mlp", "logreg"))
    data = build_training_data(prepared, request)
    token = CancelToken()
    sink = _Sink(token, "mlp", "Fitting on")
    run = train_all(data, request, data_request=prepared.request, dataset_fingerprint=prepared.fingerprint,
                    progress=sink, cancel=token)
    assert run.channels["mlp"].status == "cancelled" and run.channels["logreg"].status == "skipped"
    assert not [m for k, m in sink.messages if k == "mlp" and m and m.startswith("Epoch ")], \
        "the first epoch's check stopped the fit"


def test_a_cancel_stops_logistic_regression_at_its_next_iteration(prepared: PreparedDataset,
                                                                  monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    real_hook = train_mod._IterationReporter.on_fit_task_end

    def counted(self: Any, estimator: Any, context: Any) -> bool:
        calls.append(getattr(context, "task_name", ""))
        return real_hook(self, estimator, context)

    monkeypatch.setattr(train_mod._IterationReporter, "on_fit_task_end", counted)
    request = _request(mode="multiclass", channels=("logreg",))
    data = build_training_data(prepared, request)
    token = CancelToken()
    run = train_all(data, request, data_request=prepared.request, dataset_fingerprint=prepared.fingerprint,
                    progress=_Sink(token, "logreg", "Fitting on"), cancel=token)
    assert run.channels["logreg"].status == "cancelled"
    assert calls == ["lbfgs-iter"], "stopped at the end of the first L-BFGS iteration"


def test_a_cancel_while_scoring_stops_the_channel(prepared: PreparedDataset) -> None:
    request = _request(mode="binary", channels=("forest", "logreg"))
    data = build_training_data(prepared, request)
    token = CancelToken()
    run = train_all(data, request, data_request=prepared.request, dataset_fingerprint=prepared.fingerprint,
                    progress=_Sink(token, "forest", "Scoring"), cancel=token)
    assert run.channels["forest"].status == "cancelled" and run.channels["forest"].proba is None
    assert run.channels["logreg"].status == "skipped" and run.cancelled


def test_scoring_in_batches_matches_one_call(prepared: PreparedDataset,
                                              runs: dict[str, tuple[TrainingData, TrainingRun]],
                                              monkeypatch: pytest.MonkeyPatch) -> None:
    data, run = runs["multiclass"]
    monkeypatch.setattr(train_mod, "SCORE_BLOCK", 50)
    model = run.channels["forest"].estimator
    seen: list[tuple[int, int]] = []
    proba, seconds = train_mod.score_in_blocks(model, data.X_test, after_block=lambda done, n: seen.append((done, n)))
    assert seconds > 0
    n = len(data.X_test)
    assert seen == [(min(stop, n), n) for stop in range(50, n + 50, 50)]  # fixed blocks of 50 rows
    # The forest sums its trees on several threads, so two whole calls can already differ in the last bit.
    np.testing.assert_allclose(proba, model.predict_proba(data.X_test), rtol=0, atol=1e-12)


def test_an_iteration_limit_is_noted_on_every_fit(prepared: PreparedDataset, monkeypatch: pytest.MonkeyPatch) -> None:
    """The note comes from the fitted model, so it appears every time (warning filters show a repeat only once)."""
    real_builder = zoo.BUILDERS["logreg"]

    def two_iterations(ctx: zoo.BuildContext) -> object:
        pipeline = real_builder(ctx)
        pipeline.named_steps["model"].set_params(max_iter=2)
        return pipeline

    monkeypatch.setitem(zoo.BUILDERS, "logreg", two_iterations)
    for _ in range(2):
        _, run = _fit(prepared, _request(mode="multiclass", channels=("logreg",)))
        logreg = run.channels["logreg"]
        assert logreg.ok and logreg.extra["hit_iteration_limit"]
        convergence = [n for n in logreg.notes if "ConvergenceWarning" in n]
        assert convergence == ["ConvergenceWarning: stopped at the limit of 2 iterations before converging."]


def test_feature_quantiles_handle_empty_columns_without_warnings() -> None:
    X = np.array([[1, np.nan, np.inf], [2, np.nan, 3], [3, np.nan, 5]], dtype=np.float32)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        q = train_mod.feature_quantiles(X)
    assert caught == []
    assert q.shape == (101, 3) and q.dtype == np.float32
    assert np.isnan(q[:, 1]).all()
    assert (q[0, 0], q[-1, 0], q[0, 2], q[-1, 2]) == (1, 3, 3, 5)


# --------------------------------------------------------------------------------------------------------------
# SVM calibration: rare classes and tiny samples
# --------------------------------------------------------------------------------------------------------------
def test_calibration_draw_gives_rare_classes_a_floor_but_never_more_than_half() -> None:
    y = np.array([0] * 900 + [1] * 95 + [2] * 40 + [3] * 3 + [4] * 2)
    picked = train_mod._calibration_take(y, 250, seed=1)
    assert len(picked) == 250 and len(np.unique(picked)) == 250 and np.all(np.diff(picked) > 0)
    taken = np.bincount(y[picked], minlength=5)
    assert taken[2] == 20 and taken[3] == 1 and taken[4] == 1
    assert taken[1] == 23, "half of 95 rounded down would be 47; its proportional share stays below that"
    assert (np.bincount(y, minlength=5) - taken >= np.bincount(y, minlength=5) // 2).all()
    # Floors can make a tiny draw larger than asked, but never take more than half of a class.
    tiny = train_mod._calibration_take(np.array([0] * 8 + [1] * 8), 3, seed=2)
    assert np.bincount(np.array([0] * 8 + [1] * 8)[tiny]).tolist() == [4, 4]


def _blobs(sizes: list[int], seed: int, spread: float) -> tuple[np.ndarray, np.ndarray]:
    """Gaussian clusters, one per class, in 8 dimensions."""
    rng = np.random.default_rng(seed)
    centers = rng.normal(scale=2.0, size=(len(sizes), 8))
    X = np.vstack([rng.normal(loc=c, scale=spread, size=(n, 8)) for c, n in zip(centers, sizes)]).astype(np.float32)
    y = np.concatenate([np.full(n, k) for k, n in enumerate(sizes)]).astype(np.int64)
    return X, y


def test_svm_calibration_keeps_a_rare_class_readable() -> None:
    X_train, y_train = _blobs([2_400, 600, 50], seed=4, spread=1.6)
    X_test, y_test = _blobs([800, 200, 40], seed=4, spread=1.6)
    ctx = BuildContext(n_classes=3, seed=4, profile="test", svm_cap=2_000)
    weights = zoo.channel_weights("svm", y_train)
    fitted, info = fit_model("svm", zoo.build_estimator("svm", ctx), X_train, y_train, weights, ctx=ctx)
    per_class = info["extra"]["calibration_rows_per_class"]
    assert 20 <= per_class[2] <= 25, per_class
    assert not [n for n in info["notes"] if "least populated" in n or "n_splits" in n], info["notes"]
    raw = fitted.estimator.estimator.predict(X_test)  # the frozen SVM pipeline inside the calibration
    calibrated = fitted.predict_proba(X_test).argmax(axis=1)
    rare = y_test == 2
    raw_recall, calibrated_recall = float(np.mean(raw[rare] == 2)), float(np.mean(calibrated[rare] == 2))
    assert calibrated_recall >= raw_recall - 0.15, (raw_recall, calibrated_recall)
    assert calibrated_recall > 0.5


@pytest.mark.parametrize("sizes", [[8, 8], [22, 22], [8] * 12], ids=["binary-8+8", "binary-22+22", "12-classes-x8"])
def test_svm_fits_on_tiny_training_splits(sizes: list[int]) -> None:
    X, y = _blobs(sizes, seed=6, spread=0.8)
    ctx = BuildContext(n_classes=len(sizes), seed=6, profile="test")
    fitted, info = fit_model("svm", zoo.build_estimator("svm", ctx), X, y, zoo.channel_weights("svm", y), ctx=ctx)
    proba = fitted.predict_proba(X)
    assert proba.shape == (len(y), len(sizes))
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)
    assert info["rows_used"] + info["extra"]["calibration_rows"] == len(y)
    assert min(info["extra"]["calibration_rows_per_class"]) >= 1
    assert not [n for n in info["notes"] if "Warning" in n], info["notes"]
