"""Readings of fitted channels: hand-computed metrics, curves, leaderboard, permutation importance (held-out rows
only), cross-validation (training rows only), and the background task wrapper.

Fits use ``profile="test"`` models on about 1,500 generated flows, once per mode for the whole module.
"""

from __future__ import annotations

import threading
from dataclasses import replace
from itertools import product
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from graticule import evaluate
from graticule.data.prepare import DataRequest, PreparedDataset
from graticule.evaluate import ChannelEvaluation, EvaluationTask
from graticule.models import train
from graticule.models.jobs import CancelToken, TrainingCancelled
from graticule.models.train import TrainingRun, TrainRequest
from graticule.models.zoo import MODEL_KEYS
from tests.helpers import shared_fit, shared_sample

pytestmark = pytest.mark.unit
SEED = 7


# --------------------------------------------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def prepared() -> PreparedDataset:
    """About 1,500 synthetic flows (six classes), shared with the other modules that use this sample."""
    return shared_sample(DataRequest(source="synthetic", synthetic_flows=1_500, seed=SEED))


@pytest.fixture(scope="module")
def runs(prepared: PreparedDataset) -> dict[str, TrainingRun]:
    """All five channels fitted once per mode (test profile; each fit made once per session, see shared_fit)."""
    return {mode: shared_fit(prepared, TrainRequest(mode=mode, profile="test", seed=SEED))  # type: ignore[arg-type]
            for mode in ("binary", "multiclass")}


def _fits() -> int:
    return sum(train.FIT_CALLS.values())


def _pair_auc(positive: np.ndarray, score: np.ndarray) -> float:
    """ROC-AUC by counting pairs: share of (positive, negative) pairs ranked correctly, ties counting half."""
    pos, neg = score[positive], score[~positive]
    wins = sum(1.0 if p > n else 0.5 if p == n else 0.0 for p, n in product(pos, neg))
    return wins / (len(pos) * len(neg))


# --------------------------------------------------------------------------------------------------------------
# Metrics against hand-computed examples
# --------------------------------------------------------------------------------------------------------------
def test_binary_metrics_match_a_hand_computed_example() -> None:
    y = np.array([0, 0, 1, 1, 1, 0])
    pred = np.array([0, 1, 1, 1, 0, 0])
    attack = np.array([0.1, 0.6, 0.8, 0.7, 0.4, 0.2])
    proba = np.column_stack([1 - attack, attack])
    m = evaluate.classification_metrics(y, pred, proba, 2)
    # TP 2, FP 1, FN 1, TN 2.
    assert m["accuracy"] == pytest.approx(4 / 6)
    assert m["balanced_accuracy"] == pytest.approx((2 / 3 + 2 / 3) / 2)
    assert m["precision"] == pytest.approx(2 / 3) and m["recall"] == pytest.approx(2 / 3)
    assert m["f1"] == pytest.approx(2 / 3)
    # Pairs (positive, negative) ranked correctly: 3 + 3 + 2 of 9.
    assert m["roc_auc"] == pytest.approx(8 / 9)
    # Ranked by score: P P N P N N -> precision at each positive 1, 1, 3/4.
    assert m["average_precision"] == pytest.approx((1 + 1 + 0.75) / 3)
    assert m["f1_macro"] == pytest.approx(2 / 3)


def test_three_class_metrics_match_a_hand_computed_example() -> None:
    y = np.array([0, 0, 0, 1, 1, 2, 2, 2, 2])
    pred = np.array([0, 1, 0, 1, 1, 2, 0, 2, 1])
    rng = np.random.default_rng(3)
    proba = rng.dirichlet(np.ones(3), size=len(y))
    proba[np.arange(len(y)), pred] += 1.0  # the predicted class is the most probable
    proba /= proba.sum(axis=1, keepdims=True)
    m = evaluate.classification_metrics(y, pred, proba, 3)
    # Per class: precision 2/3, 1/2, 1; recall 2/3, 1, 1/2; F1 2/3 each.
    assert m["accuracy"] == pytest.approx(6 / 9)
    assert m["balanced_accuracy"] == pytest.approx((2 / 3 + 1 + 0.5) / 3)
    assert m["precision_macro"] == pytest.approx((2 / 3 + 0.5 + 1) / 3)
    assert m["recall_macro"] == pytest.approx((2 / 3 + 1 + 0.5) / 3)
    assert m["f1_macro"] == pytest.approx(2 / 3) and m["f1_weighted"] == pytest.approx(2 / 3)
    assert m["precision_weighted"] == pytest.approx((3 * 2 / 3 + 2 * 0.5 + 4 * 1) / 9)
    assert m["recall_weighted"] == pytest.approx(6 / 9)
    one_vs_rest = [_pair_auc(y == c, proba[:, c]) for c in range(3)]
    assert m["roc_auc"] == pytest.approx(np.mean(one_vs_rest))
    assert "precision" not in m  # the single-class readings exist for two classes only

    table = evaluate.per_class_metrics(y, pred, proba, ["BENIGN", "Bot", "DDoS"])
    assert list(table.columns) == ["class", "support", "precision", "recall", "f1", "roc_auc", "average_precision"]
    assert table["support"].tolist() == [3, 2, 4]
    np.testing.assert_allclose(table["precision"], [2 / 3, 0.5, 1.0])
    np.testing.assert_allclose(table["recall"], [2 / 3, 1.0, 0.5])
    np.testing.assert_allclose(table["f1"], [2 / 3] * 3)
    np.testing.assert_allclose(table["roc_auc"], one_vs_rest)


def test_confusion_rows_are_normalised_and_empty_rows_stay_zero() -> None:
    counts = np.array([[2, 1, 0], [0, 2, 0], [1, 1, 2], [0, 0, 0]])
    norm = evaluate.normalise_rows(counts)
    np.testing.assert_allclose(norm[:3], [[2 / 3, 1 / 3, 0], [0, 1, 0], [0.25, 0.25, 0.5]])
    np.testing.assert_allclose(norm[:3].sum(axis=1), 1.0)
    assert (norm[3] == 0).all()


def test_missing_classes_and_empty_input_give_nan_not_errors() -> None:
    y = np.array([0, 0, 1, 1])
    pred = np.array([0, 2, 1, 1])
    proba = np.eye(3)[pred]
    m = evaluate.classification_metrics(y, pred, proba, 3)
    assert np.isnan(m["roc_auc"])  # class 2 never occurs, so the one-vs-rest macro AUC is undefined
    assert np.isfinite(m["average_precision"])
    empty = evaluate.classification_metrics([], [], np.empty((0, 2)), 2)
    assert all(np.isnan(v) for v in empty.values())


# --------------------------------------------------------------------------------------------------------------
# Curves
# --------------------------------------------------------------------------------------------------------------
def test_downsampling_keeps_endpoints_and_at_most_400_points() -> None:
    x = np.linspace(0, 1, 5_000) ** 3
    y = np.sqrt(np.linspace(0, 1, 5_000))
    xs, ys = evaluate.downsample_curve(x, y)
    assert 2 < len(xs) <= 400 and len(xs) == len(ys)
    assert (xs[0], ys[0]) == (x[0], y[0]) and (xs[-1], ys[-1]) == (x[-1], y[-1])
    assert np.all(np.diff(xs) >= 0)  # order kept
    short_x, short_y = evaluate.downsample_curve(x[:50], y[:50])
    assert np.array_equal(short_x, x[:50]) and np.array_equal(short_y, y[:50])


def test_class_curves_are_bounded_and_named() -> None:
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, size=20_000)
    score = np.clip(y * 0.3 + rng.random(20_000) * 0.7, 0, 1)
    roc, pr = evaluate.class_curves(y, np.column_stack([1 - score, score]), ("Normal", "Attack"))
    assert list(roc) == ["Attack"] and list(pr) == ["Attack"]
    curve = roc["Attack"]
    assert len(curve) <= 400 and len(pr["Attack"]) <= 400
    assert (curve["fpr"].iloc[0], curve["tpr"].iloc[0]) == (0.0, 0.0)
    assert (curve["fpr"].iloc[-1], curve["tpr"].iloc[-1]) == (1.0, 1.0)
    assert pr["Attack"]["recall"].iloc[0] == 1.0 and pr["Attack"]["recall"].iloc[-1] == 0.0
    multi_y = rng.integers(0, 3, size=600)
    roc3, _ = evaluate.class_curves(multi_y, rng.dirichlet(np.ones(3), size=600), ("BENIGN", "Bot", "DDoS"))
    assert list(roc3) == ["BENIGN", "Bot", "DDoS"]


# --------------------------------------------------------------------------------------------------------------
# Whole-run readings
# --------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("mode", ["binary", "multiclass"])
def test_evaluate_run_reads_every_channel_once(runs: dict[str, TrainingRun], mode: str) -> None:
    run = runs[mode]
    fits = _fits()
    evals = evaluate.evaluate_run(run)
    assert list(evals) == list(MODEL_KEYS)
    k = len(run.data.classes)
    for key, ev in evals.items():
        assert ev.key == key and ev.n_test == len(run.data.y_test) and ev.classes == run.data.classes
        assert ev.confusion.shape == (k, k) and ev.confusion.sum() == len(run.data.y_test)
        np.testing.assert_allclose(ev.confusion_norm.sum(axis=1), 1.0)
        assert ev.metrics["balanced_accuracy"] == pytest.approx(run.channels[key].extra["metrics"]["balanced_accuracy"])
        assert 0 <= ev.metrics["roc_auc"] <= 1 and 0 <= ev.metrics["average_precision"] <= 1
        assert ev.per_class["support"].sum() == len(run.data.y_test)
        assert all(len(frame) <= 400 for frame in [*ev.roc.values(), *ev.pr.values()])
        assert ev.throughput_fps > 0 and ev.single_flow_ms > 0
        if key in ("forest", "xgboost"):
            assert ev.native_importance is not None
            assert set(ev.native_importance["feature"]) == set(run.data.feature_names)
            assert ev.native_importance["importance"].sum() == pytest.approx(1.0)
            assert ev.native_importance["importance"].is_monotonic_decreasing
        else:
            assert ev.native_importance is None
    if mode == "binary":
        assert all(list(ev.roc) == ["Attack"] for ev in evals.values())
        assert {"precision", "recall", "f1"} <= set(evals["forest"].metrics)
    else:
        assert set(evals["forest"].roc) == set(run.data.classes)
    # Kept on the run: a second call returns the very same objects, and nothing was fitted.
    again = evaluate.evaluate_run(run)
    assert all(again[key] is evals[key] for key in evals)
    assert evaluate.cached_evaluations(run) is not None and _fits() == fits
    assert evaluate.held_out_counts(run) == {c: int(n) for c, n in run.data.reports["class_counts_test"].items()}


def test_runs_without_held_out_rows_have_no_readings(runs: dict[str, TrainingRun]) -> None:
    run = runs["binary"]
    data = replace(run.data, X_test=np.empty((0, run.data.n_features), dtype=np.float32),
                   y_test=np.empty(0, dtype=np.int64))
    hollow = replace(run, data=data)
    assert not evaluate.has_test_rows(hollow)
    assert evaluate.evaluate_run(hollow) == {}
    with pytest.raises(ValueError, match="no held-out rows"):
        evaluate.evaluate_channel(hollow, "forest")
    failed = replace(run, channels={**run.channels, "svm": replace(run.channels["svm"], status="failed")})
    with pytest.raises(ValueError, match="not fitted"):
        evaluate.evaluate_channel(failed, "svm")


def _fake_eval(key: str, balanced: float) -> ChannelEvaluation:
    metrics = {name: 0.9 for name in ("accuracy", "f1_macro", "f1_weighted", "precision", "recall", "f1",
                                      "roc_auc", "average_precision")}
    metrics["balanced_accuracy"] = balanced
    return ChannelEvaluation(key=key, metrics=metrics, per_class=pd.DataFrame(), confusion=np.zeros((2, 2)),
                             confusion_norm=np.zeros((2, 2)), roc={}, pr={}, native_importance=None,
                             throughput_fps=1000.0, single_flow_ms=1.0)


def test_leaderboard_orders_by_balanced_accuracy_and_measures_the_gap(runs: dict[str, TrainingRun]) -> None:
    run = runs["binary"]
    evals = {"logreg": _fake_eval("logreg", 0.95), "forest": _fake_eval("forest", 0.97),
             "svm": _fake_eval("svm", 0.99), "xgboost": _fake_eval("xgboost", 0.97)}
    board = evaluate.leaderboard(evals, run)
    assert board["key"].tolist() == ["svm", "forest", "xgboost", "logreg"]  # ties keep the channel order
    assert board.columns[:4].tolist() == ["key", "Channel", "Balanced accuracy", "Accuracy"]
    np.testing.assert_allclose(board["Gap to best"], [0.0, 0.02, 0.02, 0.04])
    assert board.loc[0, "Channel"] == "CH3 RBF SVM"
    assert board.loc[0, "Rows used"] == run.channels["svm"].rows_used
    assert board.loc[0, "Fit s"] == pytest.approx(run.channels["svm"].fit_seconds)
    multi = evaluate.leaderboard(evaluate.evaluate_run(runs["multiclass"]), runs["multiclass"])
    assert {"F1 macro", "F1 weighted"} <= set(multi.columns) and "F1 (attack)" not in multi.columns
    assert multi["Balanced accuracy"].is_monotonic_decreasing and multi["Gap to best"].iloc[0] == 0
    per_class = evaluate.per_class_frame(evaluate.evaluate_run(runs["multiclass"]))
    assert len(per_class) == 5 * len(runs["multiclass"].data.classes)


# --------------------------------------------------------------------------------------------------------------
# Permutation importance: held-out rows only
# --------------------------------------------------------------------------------------------------------------
def test_permutation_importance_covers_every_feature_and_uses_only_test_rows(
        runs: dict[str, TrainingRun], monkeypatch: pytest.MonkeyPatch) -> None:
    run = runs["multiclass"]
    seen: list[np.ndarray] = []
    real = evaluate.permutation_importance

    def spy(estimator: object, X: np.ndarray, y: np.ndarray, **kwargs: object) -> object:
        seen.append(np.array(X, copy=True))
        return real(estimator, X, y, **kwargs)

    monkeypatch.setattr(evaluate, "permutation_importance", spy)
    fits = _fits()
    calls: list[float] = []
    frame = evaluate.permutation_importance_for(run, "logreg", max_rows=300, repeats=2,
                                                progress=lambda _m, f: calls.append(f))
    assert set(frame["feature"]) == set(run.data.feature_names) and len(frame) == run.data.n_features
    assert frame["importance"].is_monotonic_decreasing and (frame["std"] >= 0).all()
    assert frame.attrs["rows"] == 300 and frame.attrs["repeats"] == 2 and frame.attrs["seconds"] > 0
    assert len(calls) == 1 + run.data.n_features * 2 and calls[-1] == pytest.approx(1.0)
    assert _fits() == fits  # nothing refitted
    test_rows = {row.tobytes() for row in np.asarray(run.data.X_test, dtype=np.float32)}
    assert len(seen) == 1 and all(row.tobytes() in test_rows for row in seen[0])
    # Poisoning the training rows changes nothing: they play no part.
    poisoned = replace(run, data=replace(run.data, X_train=np.full_like(run.data.X_train, np.nan)))
    again = evaluate.permutation_importance_for(poisoned, "logreg", max_rows=300, repeats=2)
    pd.testing.assert_frame_equal(again, frame)


def test_permutation_plan_holds_the_svm_to_2000_rows_and_3_repeats() -> None:
    fake = SimpleNamespace(data=SimpleNamespace(y_test=np.zeros(10_000)))
    assert evaluate.permutation_plan(fake, "svm") == (2_000, 3)  # type: ignore[arg-type]
    assert evaluate.permutation_plan(fake, "forest") == (5_000, 5)  # type: ignore[arg-type]
    small = SimpleNamespace(data=SimpleNamespace(y_test=np.zeros(700)))
    assert evaluate.permutation_plan(small, "mlp", repeats=2) == (700, 2)  # type: ignore[arg-type]


def test_permutation_importance_can_be_cancelled(runs: dict[str, TrainingRun]) -> None:
    token = CancelToken()
    token.cancel()
    with pytest.raises(TrainingCancelled):
        evaluate.permutation_importance_for(runs["binary"], "logreg", max_rows=200, repeats=2, cancel=token)


# --------------------------------------------------------------------------------------------------------------
# Cross-validation: training rows only
# --------------------------------------------------------------------------------------------------------------
def test_cross_validation_uses_only_training_rows(runs: dict[str, TrainingRun],
                                                  monkeypatch: pytest.MonkeyPatch) -> None:
    run = runs["binary"]
    fitted_on: list[np.ndarray] = []
    real = train.fit_model

    def spy(key: str, estimator: object, X: np.ndarray, *args: object, **kwargs: object) -> object:
        fitted_on.append(np.array(X, copy=True))
        return real(key, estimator, X, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(train, "fit_model", spy)
    fits = _fits()
    cv = evaluate.cross_validate_run(run, k=3, max_rows=600, channels=("forest", "logreg"))
    assert cv["key"].tolist() == ["forest", "logreg"] and (cv["Folds"] == 3).all() and (cv["Rows"] == 600).all()
    assert cv.attrs["k"] == 3 and not cv.attrs["cancelled"] and len(cv.attrs["folds"]) == 6
    assert ((cv["Balanced accuracy mean"] > 0.5) & (cv["Balanced accuracy std"] >= 0)).all()
    assert _fits() - fits == 6
    train_rows = {row.tobytes() for row in np.asarray(run.data.X_train, dtype=np.float32)}
    assert len(fitted_on) == 6 and all(row.tobytes() in train_rows for X in fitted_on for row in X)

    # A poisoned test split (garbage rows, every label flipped) leaves the result untouched.
    data = run.data
    poisoned = replace(run, data=replace(data, X_test=np.full_like(data.X_test, 1e9),
                                         y_test=(1 - data.y_test).astype(np.int64)))
    again = evaluate.cross_validate_run(poisoned, k=3, max_rows=600, channels=("forest", "logreg"))
    columns = [c for c in cv.columns if c != "Fit s mean"]
    pd.testing.assert_frame_equal(again[columns], cv[columns])


def _tiny_run(class_sizes: dict[int, int]) -> SimpleNamespace:
    """A stand-in run whose training split has the given rows per class (well separated)."""
    rng = np.random.default_rng(5)
    y = np.concatenate([np.full(n, c, dtype=np.int64) for c, n in class_sizes.items()])
    X = (rng.normal(size=(len(y), 4)) + y[:, None] * 4.0).astype(np.float32)
    data = SimpleNamespace(X_train=X, y_train=y, X_test=X[:6], y_test=y[:6], classes=("BENIGN", "Bot", "Rare"),
                           feature_names=("a", "b", "c", "d"))
    result = SimpleNamespace(fit_seconds=0.05, rows_used=len(y), predict_seconds=0.001, status="ok")
    return SimpleNamespace(data=data, request=TrainRequest(mode="multiclass", profile="test", seed=SEED),
                           channels={"logreg": result, "forest": result},
                           ok_channels=lambda: ["forest", "logreg"])


def test_folds_are_reduced_when_a_class_is_tiny() -> None:
    run = _tiny_run({0: 60, 1: 40, 2: 3})
    plan = evaluate.plan_cross_validation(run, k=5)  # type: ignore[arg-type]
    assert plan.k == 3 and plan.k_requested == 5 and "Rare has only 3 training rows" in plan.note
    cv = evaluate.cross_validate_run(run, k=5, channels=("logreg",))  # type: ignore[arg-type]
    assert cv.attrs["k"] == 3 and cv["Folds"].tolist() == [3] and "reduced from 5 to 3" in cv.attrs["note"]
    with pytest.raises(ValueError, match="only 1 training row"):
        evaluate.plan_cross_validation(_tiny_run({0: 30, 1: 30, 2: 1}), k=5)  # type: ignore[arg-type]


def test_cross_validation_stops_when_cancelled_mid_way() -> None:
    run = _tiny_run({0: 60, 1: 40, 2: 30})
    token = CancelToken()
    messages: list[str] = []

    def progress(message: str, fraction: float) -> None:
        messages.append(message)
        if message.endswith("fold 1 of 3 done"):
            token.cancel()

    fits = _fits()
    cv = evaluate.cross_validate_run(run, k=3, channels=("forest", "logreg"), progress=progress,  # type: ignore[arg-type]
                                     cancel=token)
    assert cv.attrs["cancelled"] and cv.empty and cv.attrs["folds"] == []
    assert _fits() - fits == 1  # one fold of the first channel, then nothing more


def test_cv_estimate_grows_with_folds(runs: dict[str, TrainingRun]) -> None:
    run = runs["binary"]
    three = evaluate.estimate_cv_seconds(run, 3, 600)
    ten = evaluate.estimate_cv_seconds(run, 10, 600)
    assert 0 < three < ten
    assert evaluate.estimate_cv_seconds(run, 5, 600, channels=("logreg",)) < evaluate.estimate_cv_seconds(run, 5, 600)


def test_results_are_kept_with_the_run(runs: dict[str, TrainingRun]) -> None:
    run = replace(runs["binary"])
    assert evaluate.stored_cross_validation(run) is None and evaluate.stored_permutations(run) == {}
    frame = pd.DataFrame({"key": ["forest"]})
    evaluate.remember_cross_validation(run, frame)
    evaluate.remember_permutation(run, "svm", frame)
    assert evaluate.stored_cross_validation(run) is frame and evaluate.stored_permutations(run)["svm"] is frame


# --------------------------------------------------------------------------------------------------------------
# Background tasks
# --------------------------------------------------------------------------------------------------------------
def test_tasks_record_their_outcome_inline_and_in_the_background() -> None:
    done = EvaluationTask("cv", "run-1", lambda progress, cancel: (progress("half", 0.5), 42)[1])
    assert done.run_inline() == 42 and done.snapshot().state == "done" and done.snapshot().fraction == 1.0
    assert evaluate.get_task(done.task_id) is done
    evaluate.forget_task(done.task_id)
    assert evaluate.get_task(done.task_id) is None

    failing = EvaluationTask("perm", "run-1", lambda progress, cancel: 1 / 0)
    assert failing.run_inline() is None
    assert failing.snapshot().state == "failed" and "ZeroDivisionError" in (failing.error or "")

    started = threading.Event()

    def slow(progress: object, cancel: CancelToken) -> None:
        started.set()
        while not cancel.cancelled:
            threading.Event().wait(0.01)
        raise TrainingCancelled("stopped")

    task = EvaluationTask("cv", "run-1", slow)
    task.start()
    assert started.wait(10)
    assert not task.finished and task.snapshot().state == "running"
    task.cancel()
    assert task.wait(10)
    snap = task.snapshot()
    assert snap.state == "cancelled" and snap.finished and snap.elapsed > 0
    with pytest.raises(RuntimeError):
        task.start()


def test_exclusive_tasks_share_the_work_slot_with_fits() -> None:
    """Cross-validation and permutation importance run as exclusive tasks: one at a time, never alongside a fit
    (which takes the same slot), and the slot is given back however the task ends."""
    from graticule.models.jobs import JobBusyError, slot_holder

    started, release = threading.Event(), threading.Event()

    def held(progress: object, cancel: CancelToken) -> int:
        started.set()
        release.wait(10)
        return 1

    first = EvaluationTask("cv", "run-1", held, label="cross-validation", exclusive=True)
    first.start()
    try:
        assert started.wait(10) and slot_holder() == "a cross-validation"
        second = EvaluationTask("perm", "run-1", lambda progress, cancel: 2, label="CH1", exclusive=True)
        with pytest.raises(JobBusyError, match="A cross-validation is running"):
            second.run_inline()
        assert evaluate.get_task(second.task_id) is None and second.snapshot().state == "queued"
        free = EvaluationTask("perm", "run-1", lambda progress, cancel: 3)  # not exclusive: no slot needed
        assert free.run_inline() == 3
    finally:
        release.set()
    assert first.wait(10) and first.snapshot().state == "done" and slot_holder() is None
    failing = EvaluationTask("cv", "run-1", lambda progress, cancel: 1 / 0, exclusive=True)
    assert failing.run_inline() is None and failing.snapshot().state == "failed" and slot_holder() is None


def test_cv_notes_say_when_readings_are_not_comparable(runs: dict[str, TrainingRun]) -> None:
    run = runs["multiclass"]
    n_train = len(run.data.y_train)
    assert evaluate.plan_cross_validation(run, 3, n_train).note == ""  # every training row, chosen columns
    smaller = evaluate.plan_cross_validation(run, 3, n_train // 2)
    assert "class mix differs from the held-out rows" in smaller.note
    topk = replace(run, request=replace(run.request, feature_mode="topk"))
    assert "chosen on all training rows" in evaluate.plan_cross_validation(topk, 3, n_train).note


def test_multiclass_leaderboard_shows_macro_and_weighted_readings(runs: dict[str, TrainingRun]) -> None:
    from graticule import viz

    run = runs["multiclass"]
    board = evaluate.leaderboard(evaluate.evaluate_run(run), run)
    weighted = {"F1 weighted", "Precision weighted", "Recall weighted"}
    assert weighted | {"F1 macro", "Precision macro", "Recall macro"} <= set(board.columns)
    np.testing.assert_allclose(board["Recall weighted"], board["Accuracy"])  # support-weighted recall is accuracy
    assert weighted <= set(viz.SCORE_COLUMNS)  # so the dot plot and the number formats pick them up
