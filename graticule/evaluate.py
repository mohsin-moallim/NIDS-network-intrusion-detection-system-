"""Readings of fitted channels on the held-out test rows, and the measurements 03 Measure takes on demand.

Every reading here comes from the TEST split, which no fitting step ever saw: the predictions and probabilities the
trainer stored for each channel are compared with the true classes. The two measurements that fit or re-score
models are explicit and bounded:

* :func:`permutation_importance_for` shuffles one feature at a time in a rare-aware draw of held-out rows (at most
  5,000; 2,000 for the kernel SVM) and measures the drop in balanced accuracy. It never fits anything.
* :func:`cross_validate_run` refits fresh copies of the channels in stratified folds of a rare-aware draw of the
  TRAINING rows (at most 50,000). The test rows play no part in it.

:func:`evaluate_run` computes each channel's :class:`ChannelEvaluation` once and keeps it on the run object, so
showing the readings again never recomputes them. Results of the on-demand measurements can be kept with the run
the same way (:func:`remember_cross_validation`, :func:`remember_permutation`).

Long measurements run as an :class:`EvaluationTask` (a daemon thread with progress, elapsed time and a cancel
token, or inline on the calling thread). The task never touches the web framework; the UI polls its snapshot. An
exclusive task holds the process-wide work slot of :mod:`graticule.models.jobs` while it runs, so it never runs
alongside a fit or another exclusive measurement (a second one is turned away with
:class:`~graticule.models.jobs.JobBusyError`).
"""

from __future__ import annotations

import threading
import time
import traceback
import uuid
from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold

from graticule.models.jobs import CancelToken, JobBusyError, TrainingCancelled, busy_message, claim_slot, release_slot
from graticule.theme import CHANNEL_BY_KEY

if TYPE_CHECKING:
    from graticule.models.train import TrainingRun

#: Channel keys in their fixed order (kept here as well, so this module never imports the trainer at load time:
#: the trainer imports this module).
CHANNEL_ORDER: tuple[str, ...] = ("forest", "xgboost", "svm", "mlp", "logreg")
#: Most points kept per ROC or precision-recall curve.
MAX_CURVE_POINTS = 400
#: Single-row predictions timed for the single-flow latency (after one warm-up call): at most LATENCY_CALLS, but
#: no more once LATENCY_BUDGET_SECONDS have been spent and at least LATENCY_MIN_CALLS were made (a forest that
#: starts its threads for every call takes tens of milliseconds per flow, and its median settles quickly).
LATENCY_CALLS = 30
LATENCY_MIN_CALLS = 5
LATENCY_BUDGET_SECONDS = 0.25
#: Test rows scored to time throughput when the run holds no stored scoring time.
THROUGHPUT_ROWS = 5_000
#: Permutation importance on the kernel SVM: at most this many held-out rows and repeats.
SVM_PERMUTATION_ROWS = 2_000
SVM_PERMUTATION_REPEATS = 3
#: Channels whose models already predict on several threads (permutation importance scores them one at a time).
THREADED_CHANNELS: frozenset[str] = frozenset({"forest", "xgboost"})
#: Seconds of fixed cost per cross-validation fit (building the pipeline, weights, scoring set-up).
CV_OVERHEAD_SECONDS = 0.05
#: Attribute names under which results are kept on a run object.
EVALUATIONS_ATTR = "evaluations"
CROSS_VALIDATION_ATTR = "cv_readings"
PERMUTATION_ATTR = "permutation_readings"

ProgressFn = Callable[[str, float], None]
TaskState = Literal["queued", "running", "done", "failed", "cancelled"]

_EVAL_LOCK = threading.RLock()


# --------------------------------------------------------------------------------------------------------------
# Headline numbers (used by the trainer right after each fit)
# --------------------------------------------------------------------------------------------------------------
def quick_metrics(y_true: npt.ArrayLike, y_pred: npt.ArrayLike, n_classes: int) -> dict[str, float]:
    """Accuracy, balanced accuracy and macro/weighted F1 for class codes 0..n_classes-1.

    Balanced accuracy is the mean recall over the classes present in ``y_true`` (the scikit-learn definition).
    F1 scores cover every class code; a class that is never predicted and never true scores 0 rather than raising.
    Empty input gives NaN readings. With ``zero_division=0`` given, scikit-learn raises no warning here, so nothing
    touches the process-wide warning filters (this runs on the fit thread as well as on the app's threads).
    """
    truth = np.asarray(y_true, dtype=np.int64).reshape(-1)
    guess = np.asarray(y_pred, dtype=np.int64).reshape(-1)
    if truth.shape != guess.shape:
        raise ValueError("y_true and y_pred must have the same length.")
    if truth.size == 0:
        nan = float("nan")
        return {"accuracy": nan, "balanced_accuracy": nan, "f1_macro": nan, "f1_weighted": nan}
    accuracy = float(np.mean(truth == guess))
    recalls = [float(np.mean(guess[truth == c] == c)) for c in np.unique(truth)]
    labels = list(range(int(n_classes)))
    f1_macro = float(f1_score(truth, guess, labels=labels, average="macro", zero_division=0))
    f1_weighted = float(f1_score(truth, guess, labels=labels, average="weighted", zero_division=0))
    return {"accuracy": accuracy, "balanced_accuracy": float(np.mean(recalls)), "f1_macro": f1_macro,
            "f1_weighted": f1_weighted}


# --------------------------------------------------------------------------------------------------------------
# Metrics from labels and probabilities
# --------------------------------------------------------------------------------------------------------------
def _codes(values: npt.ArrayLike) -> np.ndarray:
    """Class codes as a flat int64 array."""
    return np.asarray(values, dtype=np.int64).reshape(-1)


def _scores(proba: npt.ArrayLike, n_rows: int, n_classes: int) -> np.ndarray:
    """Probabilities as float64 (n x K) with every row summing to exactly 1 (float32 rounding removed)."""
    values = np.array(proba, dtype=np.float64, copy=True)
    if values.shape != (n_rows, n_classes):
        raise ValueError(f"Expected probabilities of shape ({n_rows}, {n_classes}), got {values.shape}.")
    values = np.clip(np.nan_to_num(values, nan=0.0), 0.0, None)
    totals = values.sum(axis=1, keepdims=True)
    totals[totals <= 0] = 1.0
    return values / totals


def _one_vs_rest_auc(is_class: np.ndarray, score: np.ndarray) -> float:
    """ROC-AUC of one class against the rest; NaN unless both sides are present."""
    if is_class.all() or not is_class.any():
        return float("nan")
    return float(roc_auc_score(is_class, score))


def _one_vs_rest_ap(is_class: np.ndarray, score: np.ndarray) -> float:
    """Average precision of one class against the rest; NaN when the class has no rows."""
    if not is_class.any():
        return float("nan")
    return float(average_precision_score(is_class, score))


def classification_metrics(y_true: npt.ArrayLike, y_pred: npt.ArrayLike, proba: npt.ArrayLike,
                           n_classes: int) -> dict[str, float]:
    """Every headline reading of one channel on labelled rows.

    Keys: ``accuracy``, ``balanced_accuracy``, ``f1_macro``, ``f1_weighted`` (as :func:`quick_metrics`),
    ``precision_macro``, ``recall_macro``, ``precision_weighted``, ``recall_weighted``, ``roc_auc`` and
    ``average_precision``. With two classes, ``precision``, ``recall`` and ``f1`` are those of class code 1 (the
    Attack class in binary mode), ROC-AUC and average precision score class 1's probability. With more classes,
    ROC-AUC is the one-vs-rest macro average (NaN unless every class occurs in ``y_true``) and average precision
    the mean of the one-vs-rest values over the classes that occur. Macro and weighted averages cover every class
    code, so a class that is never predicted counts as 0 rather than being skipped.
    """
    truth, guess = _codes(y_true), _codes(y_pred)
    k = int(n_classes)
    if truth.shape != guess.shape:
        raise ValueError("y_true and y_pred must have the same length.")
    out = quick_metrics(truth, guess, k)
    nan = float("nan")
    if truth.size == 0:
        out.update({name: nan for name in ("precision_macro", "recall_macro", "precision_weighted",
                                           "recall_weighted", "roc_auc", "average_precision")})
        if k == 2:
            out.update(precision=nan, recall=nan, f1=nan)
        return out
    labels = list(range(k))
    for average in ("macro", "weighted"):
        p, r, _, _ = precision_recall_fscore_support(truth, guess, labels=labels, average=average, zero_division=0)
        out[f"precision_{average}"] = float(p)
        out[f"recall_{average}"] = float(r)
    scores = _scores(proba, len(truth), k)
    if k == 2:
        p, r, f, _ = precision_recall_fscore_support(truth, guess, labels=[1], average=None, zero_division=0)
        out["precision"], out["recall"], out["f1"] = float(p[0]), float(r[0]), float(f[0])
        out["roc_auc"] = _one_vs_rest_auc(truth == 1, scores[:, 1])
        out["average_precision"] = _one_vs_rest_ap(truth == 1, scores[:, 1])
        return out
    present = np.unique(truth)
    if len(present) == k:
        out["roc_auc"] = float(roc_auc_score(truth, scores, multi_class="ovr", average="macro", labels=labels))
    else:
        out["roc_auc"] = nan
    aps = [_one_vs_rest_ap(truth == c, scores[:, c]) for c in present]
    out["average_precision"] = float(np.mean(aps)) if aps else nan
    return out


def per_class_metrics(y_true: npt.ArrayLike, y_pred: npt.ArrayLike, proba: npt.ArrayLike,
                      classes: Sequence[str]) -> pd.DataFrame:
    """One row per class in code order: class, support, precision, recall, f1, roc_auc, average_precision.

    ``support`` counts the class's rows in ``y_true``. ROC-AUC and average precision are one-vs-rest on that
    class's probability (NaN when the class has no rows, or for ROC-AUC when every row is of that class).
    """
    truth, guess = _codes(y_true), _codes(y_pred)
    k = len(classes)
    labels = list(range(k))
    if truth.size == 0:
        p = r = f = np.zeros(k)
        s = np.zeros(k, dtype=np.int64)
        aucs = aps = [float("nan")] * k
    else:
        p, r, f, s = precision_recall_fscore_support(truth, guess, labels=labels, average=None, zero_division=0)
        scores = _scores(proba, len(truth), k)
        aucs = [_one_vs_rest_auc(truth == c, scores[:, c]) for c in labels]
        aps = [_one_vs_rest_ap(truth == c, scores[:, c]) for c in labels]
    return pd.DataFrame({
        "class": pd.Series([str(c) for c in classes], dtype="str"),
        "support": np.asarray(s, dtype=np.int64),
        "precision": np.asarray(p, dtype=np.float64),
        "recall": np.asarray(r, dtype=np.float64),
        "f1": np.asarray(f, dtype=np.float64),
        "roc_auc": np.asarray(aucs, dtype=np.float64),
        "average_precision": np.asarray(aps, dtype=np.float64),
    })


def normalise_rows(matrix: npt.ArrayLike) -> np.ndarray:
    """Each row of a confusion matrix divided by its total (float64); a row with no rows stays all zeros."""
    counts = np.asarray(matrix, dtype=np.float64)
    totals = counts.sum(axis=1, keepdims=True)
    return np.divide(counts, totals, out=np.zeros_like(counts), where=totals > 0)


def downsample_curve(x: npt.ArrayLike, y: npt.ArrayLike,
                     max_points: int = MAX_CURVE_POINTS) -> tuple[np.ndarray, np.ndarray]:
    """Thin a curve to at most ``max_points`` points, keeping both endpoints and its shape.

    Points are picked at even steps of distance travelled along the curve (in the units of the plot, both axes
    spanning 0..1), so steep corners keep as many points as long flat stretches. Curves that are short enough
    come back unchanged.
    """
    xs = np.asarray(x, dtype=np.float64).reshape(-1)
    ys = np.asarray(y, dtype=np.float64).reshape(-1)
    if xs.shape != ys.shape:
        raise ValueError("x and y must have the same length.")
    n = len(xs)
    limit = max(int(max_points), 2)
    if n <= limit:
        return xs, ys
    steps = np.hypot(np.diff(xs), np.diff(ys))
    travelled = np.concatenate([[0.0], np.cumsum(steps)])
    total = float(travelled[-1])
    if total <= 0 or not np.isfinite(total):
        picks = np.unique(np.linspace(0, n - 1, limit).round().astype(np.int64))
    else:
        picks = np.searchsorted(travelled, np.linspace(0.0, total, limit), side="left")
        picks = np.clip(picks, 0, n - 1)
        picks[0], picks[-1] = 0, n - 1
        picks = np.unique(picks)
    return xs[picks], ys[picks]


def curve_names(classes: Sequence[str]) -> list[tuple[int, str]]:
    """The (class code, curve name) pairs a channel gets curves for: class 1 alone with two classes, else all."""
    if len(classes) == 2:
        return [(1, str(classes[1]))]
    return [(code, str(name)) for code, name in enumerate(classes)]


def class_curves(y_true: npt.ArrayLike, proba: npt.ArrayLike, classes: Sequence[str], *,
                 max_points: int = MAX_CURVE_POINTS) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    """One-vs-rest ROC and precision-recall curves, at most ``max_points`` points each.

    With two classes there is one curve per kind, named after class 1 ("Attack" in binary mode); with more, one per
    class that has both positive and negative rows. ROC frames hold ``fpr``/``tpr``; precision-recall frames hold
    ``recall``/``precision`` in the order scikit-learn returns them (recall falling from 1 to 0).
    """
    truth = _codes(y_true)
    k = len(classes)
    roc: dict[str, pd.DataFrame] = {}
    pr: dict[str, pd.DataFrame] = {}
    if truth.size == 0:
        return roc, pr
    scores = _scores(proba, len(truth), k)
    for code, name in curve_names(classes):
        positive = truth == code
        if positive.all() or not positive.any():
            continue
        fpr, tpr, _ = roc_curve(positive, scores[:, code], drop_intermediate=True)
        fpr, tpr = downsample_curve(fpr, tpr, max_points)
        roc[name] = pd.DataFrame({"fpr": fpr, "tpr": tpr})
        precision, recall, _ = precision_recall_curve(positive, scores[:, code], drop_intermediate=True)
        recall, precision = downsample_curve(recall, precision, max_points)
        pr[name] = pd.DataFrame({"recall": recall, "precision": precision})
    return roc, pr


# --------------------------------------------------------------------------------------------------------------
# Importance and timing
# --------------------------------------------------------------------------------------------------------------
def _final_model(estimator: Any) -> Any:
    """The last step of a pipeline (the estimator itself when it is not a pipeline)."""
    steps = getattr(estimator, "steps", None)
    return steps[-1][1] if steps else estimator


def native_importance(estimator: Any, key: str, feature_names: Sequence[str]) -> pd.DataFrame | None:
    """The model's own importance per feature, normalised to sum to 1, largest first; None when it has none.

    Random forest: mean decrease in impurity (``feature_importances_``). XGBoost: total gain over all splits on the
    feature (features never used score 0). The other channels have no built-in importance; permutation importance
    covers every channel.
    """
    names = [str(n) for n in feature_names]
    model = _final_model(estimator)
    values: np.ndarray | None = None
    if key == "forest" and hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_, dtype=np.float64)
    elif key == "xgboost" and hasattr(model, "get_booster"):
        gains = model.get_booster().get_score(importance_type="total_gain")
        values = np.zeros(len(names), dtype=np.float64)
        position = {name: i for i, name in enumerate(names)}
        for name, gain in gains.items():
            index = position.get(name)
            if index is None and name.startswith("f") and name[1:].isdigit():
                index = int(name[1:])
            if index is not None and 0 <= index < len(names):
                values[index] = float(gain)
    if values is None or values.shape != (len(names),):
        return None
    total = float(values.sum())
    if total > 0:
        values = values / total
    frame = pd.DataFrame({"feature": pd.Series(names, dtype="str"), "importance": values})
    return frame.sort_values("importance", ascending=False, kind="stable").reset_index(drop=True)


def single_flow_latency_ms(estimator: Any, X: np.ndarray, calls: int = LATENCY_CALLS, *,
                           budget_seconds: float = LATENCY_BUDGET_SECONDS) -> float:
    """Median milliseconds for ``predict_proba`` on one row, over up to ``calls`` calls (after one warm-up call).

    Rows are taken in turn from ``X``. The calls stop early once ``budget_seconds`` have been spent and at least
    :data:`LATENCY_MIN_CALLS` were made, so a slow channel does not hold up the page. Returns NaN when ``X`` has no
    rows.
    """
    rows = np.asarray(X)
    if rows.shape[0] == 0:
        return float("nan")
    estimator.predict_proba(rows[:1])
    times: list[float] = []
    for i in range(max(int(calls), 1)):
        one = rows[i % rows.shape[0]: i % rows.shape[0] + 1]
        mark = time.perf_counter()
        estimator.predict_proba(one)
        times.append(time.perf_counter() - mark)
        if len(times) >= LATENCY_MIN_CALLS and sum(times) >= budget_seconds:
            break
    return float(np.median(times) * 1000.0)


def _throughput(result: Any, n_test: int, estimator: Any, X_test: np.ndarray) -> float:
    """Flows per second when scoring the test rows in batches.

    The trainer timed exactly that while scoring the test split (batches of at least 5,000 rows), so its figure
    is used; only a run without it is timed again, on at most :data:`THROUGHPUT_ROWS` test rows.
    """
    extra = getattr(result, "extra", None) or {}
    stored = extra.get("flows_per_second")
    if isinstance(stored, (int, float)) and not isinstance(stored, bool) and stored > 0:
        return float(stored)
    seconds = float(getattr(result, "predict_seconds", 0.0) or 0.0)
    if seconds > 0 and n_test > 0:
        return float(n_test / seconds)
    rows = np.asarray(X_test)[:THROUGHPUT_ROWS]
    if rows.shape[0] == 0:
        return float("nan")
    mark = time.perf_counter()
    estimator.predict_proba(rows)
    return float(rows.shape[0] / max(time.perf_counter() - mark, 1e-9))


# --------------------------------------------------------------------------------------------------------------
# Channel evaluations
# --------------------------------------------------------------------------------------------------------------
@dataclass
class ChannelEvaluation:
    """Every reading of one fitted channel on the held-out test rows.

    Attributes:
        key: channel key.
        metrics: see :func:`classification_metrics`.
        per_class: see :func:`per_class_metrics`.
        confusion: counts, rows = true class, columns = predicted class, in class-code order.
        confusion_norm: ``confusion`` with each row divided by its total (row share).
        roc: curve name ("Attack", or each class name) -> frame of ``fpr``/``tpr`` (at most 400 points).
        pr: curve name -> frame of ``recall``/``precision`` (at most 400 points).
        native_importance: feature/importance frame (forest impurity, XGBoost total gain), or None.
        throughput_fps: flows per second when scoring test rows in batches.
        single_flow_ms: median latency of scoring one flow, in milliseconds.
        classes: class names in code order.
        n_test: number of held-out rows behind these readings.
        seconds: time taken to compute this evaluation.
    """

    key: str
    metrics: dict[str, float]
    per_class: pd.DataFrame
    confusion: np.ndarray
    confusion_norm: np.ndarray
    roc: dict[str, pd.DataFrame]
    pr: dict[str, pd.DataFrame]
    native_importance: pd.DataFrame | None
    throughput_fps: float
    single_flow_ms: float
    classes: tuple[str, ...] = ()
    n_test: int = 0
    seconds: float = 0.0

    @property
    def label(self) -> str:
        """Badge and name of the channel, e.g. ``"CH2 XGBoost"``."""
        return channel_label(self.key)


def channel_label(key: str) -> str:
    """Badge and name of a channel key, e.g. ``"CH2 XGBoost"`` (the key itself when unknown)."""
    style = CHANNEL_BY_KEY.get(key)
    return style.label if style is not None else key


def has_test_rows(run: "TrainingRun") -> bool:
    """True when ``run`` holds held-out rows to measure on (a run loaded from disk may have none)."""
    flag = getattr(run, "has_test_rows", None)
    if flag is not None and not bool(flag):
        return False
    return len(run.data.y_test) > 0


def _ordered(keys: Sequence[str]) -> list[str]:
    """Channel keys in the fixed CH1..CH5 order (unknown keys last, as given)."""
    known = [k for k in CHANNEL_ORDER if k in keys]
    return known + [k for k in keys if k not in CHANNEL_ORDER]


def _channel_scores(result: Any, X_test: np.ndarray, n_classes: int) -> tuple[np.ndarray, np.ndarray]:
    """The stored test-set probabilities and predictions of a channel (predicted again only if missing)."""
    proba = getattr(result, "proba", None)
    if proba is None:
        raw = np.asarray(result.estimator.predict_proba(X_test), dtype=np.float64)
        model_classes = getattr(result.estimator, "classes_", None)
        if raw.shape[1] != n_classes and model_classes is not None:
            full = np.zeros((raw.shape[0], n_classes), dtype=np.float64)
            full[:, np.asarray(model_classes, dtype=np.int64)] = raw
            raw = full
        proba = raw
    proba = np.asarray(proba)
    y_pred = getattr(result, "y_pred", None)
    y_pred = np.asarray(proba).argmax(axis=1) if y_pred is None else np.asarray(y_pred)
    return proba, _codes(y_pred)


def evaluate_channel(run: "TrainingRun", key: str) -> ChannelEvaluation:
    """Compute every reading of channel ``key`` from its stored test-set predictions.

    Nothing is fitted; the model is only called to time single-flow scoring (and to score the test rows when a
    run carries no stored probabilities). Raises ``KeyError`` for a channel the run does not hold and
    ``ValueError`` for a channel that was not fitted or a run without held-out rows.
    """
    started = time.perf_counter()
    result = run.channels[key]
    if getattr(result, "status", "ok") != "ok" or result.estimator is None:
        raise ValueError(f"{channel_label(key)} was not fitted, so it has no readings.")
    if not has_test_rows(run):
        raise ValueError("This run holds no held-out rows to measure on.")
    data = run.data
    classes = tuple(str(c) for c in data.classes)
    k = len(classes)
    y_test = _codes(data.y_test)
    X_test = np.asarray(data.X_test)
    proba, y_pred = _channel_scores(result, X_test, k)
    matrix = confusion_matrix(y_test, y_pred, labels=list(range(k))).astype(np.int64)
    roc, pr = class_curves(y_test, proba, classes)
    return ChannelEvaluation(
        key=key,
        metrics=classification_metrics(y_test, y_pred, proba, k),
        per_class=per_class_metrics(y_test, y_pred, proba, classes),
        confusion=matrix,
        confusion_norm=normalise_rows(matrix),
        roc=roc,
        pr=pr,
        native_importance=native_importance(result.estimator, key, data.feature_names),
        throughput_fps=_throughput(result, len(y_test), result.estimator, X_test),
        single_flow_ms=single_flow_latency_ms(result.estimator, X_test),
        classes=classes,
        n_test=int(len(y_test)),
        seconds=time.perf_counter() - started,
    )


def evaluate_run(run: "TrainingRun") -> dict[str, ChannelEvaluation]:
    """Readings of every fitted channel of ``run``, in channel order, each computed once and kept on the run.

    The results live in ``run.evaluations`` (a dict keyed by channel), so later calls (any page, any session)
    return them without recomputing. A run without held-out rows gives an empty mapping. Thread-safe.
    """
    if not has_test_rows(run):
        return {}
    with _EVAL_LOCK:
        cache = getattr(run, EVALUATIONS_ATTR, None)
        if not isinstance(cache, dict):
            cache = {}
            setattr(run, EVALUATIONS_ATTR, cache)
        keys = _ordered(list(run.ok_channels()))
        for key in keys:
            if key not in cache:
                cache[key] = evaluate_channel(run, key)
        return {key: cache[key] for key in keys if key in cache}


def cached_evaluations(run: "TrainingRun") -> dict[str, ChannelEvaluation] | None:
    """The readings already kept on ``run`` by :func:`evaluate_run`, or None when none were computed yet."""
    cache = getattr(run, EVALUATIONS_ATTR, None)
    return dict(cache) if isinstance(cache, dict) and cache else None


def held_out_counts(run: "TrainingRun") -> dict[str, int]:
    """Held-out rows per class, in class-code order (classes with no rows included as 0)."""
    classes = [str(c) for c in run.data.classes]
    counts = np.bincount(_codes(run.data.y_test), minlength=len(classes)) if len(run.data.y_test) else \
        np.zeros(len(classes), dtype=np.int64)
    return {name: int(counts[i]) for i, name in enumerate(classes)}


def held_out_repeats(run: "TrainingRun") -> dict[str, int] | None:
    """How much recorded traffic the held-out rows stand for, or None when the run carries no repeat counts.

    Exact repeats are merged before the split, so every held-out row is one distinct flow and every reading counts
    it once, however often it occurred in the files. Returns ``rows`` (held-out rows), ``flows`` (rows of the
    source files they stand for, repeats included), ``largest`` (the most any one row stands for) and
    ``repeated`` (held-out rows standing for more than one recorded flow).
    """
    copies = getattr(run.data, "test_copies", None)
    if copies is None or len(copies) == 0 or len(copies) != len(run.data.y_test):
        return None
    values = np.asarray(copies, dtype=np.int64)
    return {"rows": int(len(values)), "flows": int(values.sum()), "largest": int(values.max()),
            "repeated": int((values > 1).sum())}


def repeats_sentence(repeats: Mapping[str, int] | None) -> str | None:
    """Plain words on what a held-out row stands for (see :func:`held_out_repeats`); None when nothing repeats."""
    if not repeats or repeats.get("flows", 0) <= repeats.get("rows", 0):
        return None
    return (f"Each held-out row is a distinct flow: exact repeats were merged before the split, so these "
            f"{repeats['rows']:,} rows stand for {repeats['flows']:,} recorded flows (one of them for "
            f"{repeats['largest']:,}). Every reading counts a distinct flow once. A whole file scored at 05 Assay "
            "counts every repeat, so a channel that misses a much-repeated flow reads lower there than here.")


# --------------------------------------------------------------------------------------------------------------
# Comparison tables
# --------------------------------------------------------------------------------------------------------------
#: Leaderboard score columns per mode: (metric key, column title). Balanced accuracy always comes first.
BINARY_SCORES: tuple[tuple[str, str], ...] = (
    ("balanced_accuracy", "Balanced accuracy"), ("accuracy", "Accuracy"), ("precision", "Precision (attack)"),
    ("recall", "Recall (attack)"), ("f1", "F1 (attack)"), ("f1_macro", "F1 macro"), ("roc_auc", "ROC-AUC"),
    ("average_precision", "Average precision"),
)
MULTICLASS_SCORES: tuple[tuple[str, str], ...] = (
    ("balanced_accuracy", "Balanced accuracy"), ("accuracy", "Accuracy"), ("f1_macro", "F1 macro"),
    ("f1_weighted", "F1 weighted"), ("precision_macro", "Precision macro"),
    ("precision_weighted", "Precision weighted"), ("recall_macro", "Recall macro"),
    ("recall_weighted", "Recall weighted"), ("roc_auc", "ROC-AUC"), ("average_precision", "Average precision"),
)
#: Leaderboard columns after the scores.
LEADERBOARD_TAIL: tuple[str, ...] = ("Gap to best", "Fit s", "Flows/s", "Single-flow ms", "Rows used",
                                     "Training rows")


def score_columns(mode: str) -> tuple[tuple[str, str], ...]:
    """The (metric key, column title) pairs of the leaderboard for ``mode`` ("binary" or "multiclass")."""
    return BINARY_SCORES if mode == "binary" else MULTICLASS_SCORES


def leaderboard(evals: Mapping[str, ChannelEvaluation], run: "TrainingRun") -> pd.DataFrame:
    """One row per evaluated channel, best balanced accuracy first (ties keep the channel order).

    Columns: ``key``, ``Channel`` (badge and name), the scores of :func:`score_columns` for the run's mode, then
    ``Gap to best`` (best balanced accuracy minus this one), ``Fit s``, ``Flows/s``, ``Single-flow ms``,
    ``Rows used`` (training rows the channel was fitted on) and ``Training rows`` (size of the training split).
    """
    scores = score_columns(run.request.mode)
    rows: list[dict[str, Any]] = []
    for key in _ordered(list(evals)):
        ev = evals[key]
        result = run.channels.get(key)
        row: dict[str, Any] = {"key": key, "Channel": channel_label(key)}
        for metric, title in scores:
            row[title] = float(ev.metrics.get(metric, float("nan")))
        row["Fit s"] = float(getattr(result, "fit_seconds", float("nan")))
        row["Flows/s"] = float(ev.throughput_fps)
        row["Single-flow ms"] = float(ev.single_flow_ms)
        row["Rows used"] = int(getattr(result, "rows_used", 0) or 0)
        row["Training rows"] = int(getattr(result, "rows_available", 0) or 0)
        rows.append(row)
    columns = ["key", "Channel", *[title for _, title in scores], *LEADERBOARD_TAIL]
    frame = pd.DataFrame(rows, columns=columns)
    if frame.empty:
        return frame
    frame = frame.sort_values("Balanced accuracy", ascending=False, kind="stable", na_position="last")
    best = float(frame["Balanced accuracy"].max())
    frame["Gap to best"] = best - frame["Balanced accuracy"]
    return frame.reset_index(drop=True)[columns]


def per_class_frame(evals: Mapping[str, ChannelEvaluation]) -> pd.DataFrame:
    """The per-class readings of every channel stacked into one table (channel order, then class order)."""
    parts = []
    for key in _ordered(list(evals)):
        table = evals[key].per_class.copy()
        table.insert(0, "Channel", channel_label(key))
        table.insert(0, "key", key)
        parts.append(table)
    if not parts:
        return pd.DataFrame(columns=["key", "Channel", "class", "support", "precision", "recall", "f1", "roc_auc",
                                     "average_precision"])
    return pd.concat(parts, ignore_index=True)


# --------------------------------------------------------------------------------------------------------------
# Permutation importance (held-out rows only)
# --------------------------------------------------------------------------------------------------------------
def _balanced_accuracy(truth: np.ndarray, guess: np.ndarray) -> float:
    """Mean recall over the classes present in ``truth`` (no warnings)."""
    present = np.unique(truth)
    return float(np.mean([np.mean(guess[truth == c] == c) for c in present])) if present.size else float("nan")


def permutation_plan(run: "TrainingRun", key: str, *, max_rows: int = 5_000,
                     repeats: int = 5) -> tuple[int, int]:
    """Held-out rows and repeats a permutation measurement of ``key`` uses (the kernel SVM is held to 2,000 x 3)."""
    rows, reps = int(max_rows), int(repeats)
    if key == "svm":
        rows, reps = min(rows, SVM_PERMUTATION_ROWS), min(reps, SVM_PERMUTATION_REPEATS)
    return max(min(rows, len(run.data.y_test)), 0), max(reps, 1)


def permutation_importance_for(
    run: "TrainingRun",
    key: str,
    *,
    max_rows: int = 5_000,
    repeats: int = 5,
    seed: int | None = None,
    progress: ProgressFn | None = None,
    cancel: CancelToken | None = None,
) -> pd.DataFrame:
    """Drop in balanced accuracy when each feature's values are shuffled among held-out rows.

    The rows are a rare-aware draw of at most ``max_rows`` TEST rows (every class keeps its share of a floor), so
    the measurement never touches training rows and never fits anything. Each feature is shuffled ``repeats``
    times; the kernel SVM is held to 2,000 rows and 3 repeats. Features are scored on several threads
    (``joblib.parallel_config(backend="threading")``), except for the tree channels, whose models already predict on
    several threads. ``progress`` gets (message, fraction) after each scoring and ``cancel`` is checked before each
    (raising :class:`~graticule.models.jobs.TrainingCancelled`).

    Returns one row per feature, largest drop first: ``feature``, ``importance`` (mean drop), ``std`` (spread over
    the repeats). ``attrs`` records ``key``, ``rows``, ``repeats``, ``baseline`` (balanced accuracy before any
    shuffle), ``seconds`` and ``seed``.
    """
    import joblib

    from graticule.data.sampling import sample_positions

    started = time.perf_counter()
    result = run.channels[key]
    if getattr(result, "status", "ok") != "ok" or result.estimator is None:
        raise ValueError(f"{channel_label(key)} was not fitted, so its importance cannot be measured.")
    if not has_test_rows(run):
        raise ValueError("This run holds no held-out rows to measure on.")
    seed = int(run.request.seed if seed is None else seed)
    n_rows, n_repeats = permutation_plan(run, key, max_rows=max_rows, repeats=repeats)
    y_all = _codes(run.data.y_test)
    picked, _ = sample_positions(y_all, n_rows, seed)
    X = np.array(np.asarray(run.data.X_test)[picked], dtype=np.float32, copy=True)
    y = y_all[picked]
    names = [str(n) for n in run.data.feature_names]
    total = 1 + len(names) * n_repeats
    done = [0]
    lock = threading.Lock()
    baseline: list[float] = []

    def scorer(estimator: Any, X_part: np.ndarray, y_part: np.ndarray) -> float:
        if cancel is not None and cancel.cancelled:
            raise TrainingCancelled("The permutation measurement was cancelled.")
        score = _balanced_accuracy(_codes(y_part), _codes(estimator.predict(X_part)))
        with lock:
            done[0] += 1
            count = done[0]
            if count == 1:
                baseline.append(score)
        if progress is not None:
            progress(f"{count:,} of {total:,} scorings", count / total)
        return score

    # The tree channels already score on several threads of their own; the others score one feature per thread.
    n_jobs = 1 if key in THREADED_CHANNELS else -1
    with joblib.parallel_config(backend="threading"):
        bunch = permutation_importance(result.estimator, X, y, scoring=scorer, n_repeats=n_repeats, n_jobs=n_jobs,
                                       random_state=seed)
    frame = pd.DataFrame({
        "feature": pd.Series(names, dtype="str"),
        "importance": np.asarray(bunch.importances_mean, dtype=np.float64),
        "std": np.asarray(bunch.importances_std, dtype=np.float64),
    }).sort_values("importance", ascending=False, kind="stable").reset_index(drop=True)
    frame.attrs.update({"key": key, "rows": int(len(y)), "repeats": n_repeats,
                        "baseline": baseline[0] if baseline else float("nan"),
                        "seconds": time.perf_counter() - started, "seed": seed})
    return frame


# --------------------------------------------------------------------------------------------------------------
# Cross-validation (training rows only)
# --------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class CVPlan:
    """What a cross-validation will do: rows drawn from the training split, folds and channels."""

    rows: int
    rows_available: int
    k_requested: int
    k: int
    channels: tuple[str, ...]
    note: str


def _cv_channels(run: "TrainingRun", channels: Sequence[str] | None) -> tuple[str, ...]:
    """Channels to cross-validate: those asked for (in channel order), else every fitted one."""
    fitted = list(run.ok_channels())
    wanted = fitted if channels is None else [str(c) for c in channels]
    return tuple(_ordered([k for k in wanted if k in CHANNEL_ORDER]))


def _cv_draw(run: "TrainingRun", max_rows: int) -> np.ndarray:
    """Positions (into the training split) of the rare-aware draw cross-validation runs on."""
    from graticule.data.sampling import sample_positions

    y = _codes(run.data.y_train)
    picked, _ = sample_positions(y, min(int(max_rows), len(y)), int(run.request.seed))
    return picked


def plan_cross_validation(run: "TrainingRun", k: int = 5, max_rows: int = 50_000,
                          channels: Sequence[str] | None = None) -> CVPlan:
    """Work out the rows and folds of :func:`cross_validate_run` without fitting anything.

    The number of folds is reduced when a class has fewer rows in the draw than folds (each fold must hold every
    class); ``note`` says so. ``note`` also warns when the readings are not comparable with the held-out ones: a
    Top-K run chose its columns on every training row (the validation folds included), which flatters the folds,
    and a draw smaller than the training split keeps rare classes, so its class mix differs from the held-out rows.
    Raises ``ValueError`` when there are no training rows or a class has a single row.
    """
    n_available = len(run.data.y_train)
    if n_available == 0:
        raise ValueError("This run holds no training rows to cross-validate on.")
    if int(k) < 2:
        raise ValueError("Cross-validation needs at least 2 folds.")
    picked = _cv_draw(run, max_rows)
    y = _codes(run.data.y_train)[picked]
    counts = np.bincount(y, minlength=len(run.data.classes))
    present = counts[counts > 0]
    smallest = int(present.min()) if present.size else 0
    k_used = int(min(int(k), smallest))
    note = ""
    if k_used < int(k):
        name = str(run.data.classes[int(np.flatnonzero(counts == smallest)[0])])
        if k_used < 2:
            raise ValueError(f"Cross-validation is not possible: {name} has only {smallest} training row.")
        note = (f"Folds reduced from {int(k)} to {k_used}: {name} has only {smallest} training rows in the draw, "
                "and every fold must hold each class.")
    caveats = [note] if note else []
    if getattr(run.request, "feature_mode", "curated") == "topk":
        caveats.append("The Top-K columns were chosen on all training rows, the validation folds included, so these "
                       "readings lean optimistic.")
    if len(picked) < n_available:
        caveats.append("The draw keeps rare classes, so its class mix differs from the held-out rows: compare "
                       "balanced accuracy with the readings above, not accuracy or F1.")
    note = " ".join(caveats)
    return CVPlan(rows=int(len(picked)), rows_available=int(n_available), k_requested=int(k), k=k_used,
                  channels=_cv_channels(run, channels), note=note)


def estimate_cv_seconds(run: "TrainingRun", k: int, max_rows: int = 50_000, *,
                        channels: Sequence[str] | None = None) -> float:
    """Rough seconds :func:`cross_validate_run` will take, scaled from the run's own fit and scoring times.

    Each fold fits on (k-1)/k of the draw. Fit time is taken as proportional to the rows fitted on, except for the
    kernel SVM, whose rows stay under its cap and whose time grows with the square of its rows. Scoring time is
    proportional to the rows scored. A rough guide only (the machine may be busier or idler than during the fit).
    """
    from graticule.models.zoo import BuildContext

    try:
        plan = plan_cross_validation(run, k, max_rows, channels)
    except ValueError:
        return 0.0
    fold_train = plan.rows * (plan.k - 1) / plan.k
    fold_test = plan.rows / plan.k
    n_test = max(len(run.data.y_test), 1)
    total = 0.0
    for key in plan.channels:
        result = run.channels.get(key)
        fit_s = float(getattr(result, "fit_seconds", 0.0) or 0.0)
        used = max(int(getattr(result, "rows_used", 0) or 0), 1)
        if key == "svm":
            cap = BuildContext(n_classes=max(len(run.data.classes), 2), seed=0, profile=run.request.profile,
                               svm_cap=int(run.request.svm_cap)).effective_svm_cap
            rows = min(fold_train, cap)
            scale = (rows / used) ** 2
        else:
            scale = fold_train / used
        predict_s = float(getattr(result, "predict_seconds", 0.0) or 0.0)
        total += plan.k * (fit_s * scale + predict_s * fold_test / n_test + CV_OVERHEAD_SECONDS)
    return float(total)


def _fold_proba(model: Any, X: np.ndarray, n_classes: int) -> np.ndarray:
    """Class probabilities of a fitted fold model, columns in class-code order."""
    raw = np.asarray(model.predict_proba(X), dtype=np.float64)
    if raw.shape[1] != n_classes:
        full = np.zeros((raw.shape[0], n_classes), dtype=np.float64)
        full[:, np.asarray(model.classes_, dtype=np.int64)] = raw
        raw = full
    return raw


CV_COLUMNS: tuple[str, ...] = (
    "key", "Channel", "Folds", "Rows", "Accuracy mean", "Accuracy std", "Balanced accuracy mean",
    "Balanced accuracy std", "F1 macro mean", "F1 macro std", "Fit s mean", "Error",
)
#: Metrics reported by cross-validation: (metric key, column stem).
CV_METRICS: tuple[tuple[str, str], ...] = (("accuracy", "Accuracy"), ("balanced_accuracy", "Balanced accuracy"),
                                           ("f1_macro", "F1 macro"))


def cross_validate_run(
    run: "TrainingRun",
    *,
    k: int = 5,
    max_rows: int = 50_000,
    channels: Sequence[str] | None = None,
    progress: ProgressFn | None = None,
    cancel: CancelToken | None = None,
) -> pd.DataFrame:
    """Stratified k-fold cross-validation of fresh channel copies on the TRAINING rows only.

    The rows are a rare-aware draw of at most ``max_rows`` training rows (seeded with the run's seed). Folds are
    stratified and shuffled; ``k`` drops automatically when a class has fewer rows than folds (see
    :func:`plan_cross_validation`). Every fold builds a fresh pipeline (:func:`graticule.models.zoo.build_estimator`)
    with the run's options and fits it through :func:`graticule.models.train.fit_model`, with the same capped
    balanced weights as the fit (recomputed on the fold), so the SVM cap, early stopping and calibration all come
    from the fold's own training part. The held-out test rows are never used.

    ``progress`` receives (message, fraction) after each fold; ``cancel`` is checked before each fold and inside
    each fit. A cancel ends the work early: channels that finished every fold are kept, ``attrs["cancelled"]`` is
    True. A channel that fails is reported in its ``Error`` column and the others carry on.

    Returns one row per channel (see :data:`CV_COLUMNS`; std is the sample standard deviation over folds).
    ``attrs`` holds ``k_requested``, ``k``, ``rows``, ``rows_available``, ``note``, ``cancelled``, ``seconds``,
    ``seed`` and ``folds`` (one plain dict per finished fold: key, fold, accuracy, balanced_accuracy, f1_macro,
    fit_seconds).
    """
    import joblib

    from graticule.models import zoo
    from graticule.models.train import fit_model

    started = time.perf_counter()
    plan = plan_cross_validation(run, k, max_rows, channels)
    seed = int(run.request.seed)
    picked = _cv_draw(run, max_rows)
    X = np.asarray(run.data.X_train)[picked]
    y = _codes(run.data.y_train)[picked]
    n_classes = len(run.data.classes)
    folds = list(StratifiedKFold(n_splits=plan.k, shuffle=True, random_state=seed).split(np.zeros(len(y)), y))
    ctx = run.request.build_context(n_classes)
    balanced = bool(run.request.balanced)
    total_steps = max(len(plan.channels) * plan.k, 1)
    step = 0
    cancelled = False
    rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    with joblib.parallel_config(backend="threading"):
        for key in plan.channels:
            label = channel_label(key)
            scores: list[dict[str, float]] = []
            error: str | None = None
            for index, (train_idx, test_idx) in enumerate(folds):
                if cancel is not None and cancel.cancelled:
                    cancelled = True
                    break
                if progress is not None:
                    progress(f"{label}: fold {index + 1} of {plan.k}", step / total_steps)
                try:
                    estimator = zoo.build_estimator(key, ctx)
                    weights = zoo.channel_weights(key, y[train_idx], balanced)
                    mark = time.perf_counter()
                    model, _info = fit_model(key, estimator, X[train_idx], y[train_idx], weights, ctx=ctx,
                                             cancel=cancel)
                    fit_seconds = time.perf_counter() - mark
                    guess = _fold_proba(model, X[test_idx], n_classes).argmax(axis=1)
                except TrainingCancelled:
                    cancelled = True
                    break
                except Exception as exc:  # noqa: BLE001 - one failing channel must not stop the others
                    error = f"{type(exc).__name__}: {exc}"
                    break
                reading = quick_metrics(y[test_idx], guess, n_classes)
                reading["fit_seconds"] = fit_seconds
                scores.append(reading)
                fold_rows.append({"key": key, "fold": index + 1, **{m: float(reading[m]) for m, _ in CV_METRICS},
                                  "fit_seconds": float(fit_seconds)})
                step += 1
                if progress is not None:
                    progress(f"{label}: fold {index + 1} of {plan.k} done", step / total_steps)
            if cancelled:
                fold_rows = [r for r in fold_rows if r["key"] != key]
                break
            row: dict[str, Any] = {"key": key, "Channel": label, "Folds": len(scores), "Rows": int(len(y)),
                                   "Error": error}
            for metric, stem in CV_METRICS:
                values = np.array([s[metric] for s in scores], dtype=np.float64)
                row[f"{stem} mean"] = float(values.mean()) if values.size else float("nan")
                row[f"{stem} std"] = float(values.std(ddof=1)) if values.size > 1 else float("nan")
            fit_times = [s["fit_seconds"] for s in scores]
            row["Fit s mean"] = float(np.mean(fit_times)) if fit_times else float("nan")
            rows.append(row)
    frame = pd.DataFrame(rows, columns=list(CV_COLUMNS))
    frame.attrs.update({
        "k_requested": plan.k_requested, "k": plan.k, "rows": plan.rows, "rows_available": plan.rows_available,
        "note": plan.note, "cancelled": cancelled, "seconds": time.perf_counter() - started, "seed": seed,
        "folds": fold_rows,
    })
    return frame


def cv_fold_frame(cv: pd.DataFrame) -> pd.DataFrame:
    """The per-fold readings kept in ``cv.attrs["folds"]`` as a table (empty when there are none)."""
    folds = cv.attrs.get("folds") or []
    columns = ["key", "fold", *[m for m, _ in CV_METRICS], "fit_seconds"]
    return pd.DataFrame(list(folds), columns=columns)


# --------------------------------------------------------------------------------------------------------------
# Results kept with a run
# --------------------------------------------------------------------------------------------------------------
def remember_cross_validation(run: "TrainingRun", frame: pd.DataFrame) -> None:
    """Keep a cross-validation result with ``run`` (replacing an earlier one)."""
    setattr(run, CROSS_VALIDATION_ATTR, frame)


def stored_cross_validation(run: "TrainingRun") -> pd.DataFrame | None:
    """The cross-validation result kept with ``run``, or None."""
    value = getattr(run, CROSS_VALIDATION_ATTR, None)
    return value if isinstance(value, pd.DataFrame) else None


def remember_permutation(run: "TrainingRun", key: str, frame: pd.DataFrame) -> None:
    """Keep a permutation-importance result for channel ``key`` with ``run``."""
    with _EVAL_LOCK:
        store = getattr(run, PERMUTATION_ATTR, None)
        if not isinstance(store, dict):
            store = {}
            setattr(run, PERMUTATION_ATTR, store)
        store[key] = frame


def stored_permutations(run: "TrainingRun") -> dict[str, pd.DataFrame]:
    """Permutation-importance results kept with ``run``, by channel key (empty when none)."""
    store = getattr(run, PERMUTATION_ATTR, None)
    return dict(store) if isinstance(store, dict) else {}


# --------------------------------------------------------------------------------------------------------------
# Background tasks
# --------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class TaskSnapshot:
    """An immutable view of an :class:`EvaluationTask`'s progress."""

    task_id: str
    kind: str
    run_id: str
    state: TaskState
    fraction: float
    message: str
    elapsed: float
    error: str | None

    @property
    def finished(self) -> bool:
        """True once the task has reached a final state."""
        return self.state in ("done", "failed", "cancelled")


_TASKS: "OrderedDict[str, EvaluationTask]" = OrderedDict()
_TASKS_LOCK = threading.Lock()
_KEEP_TASKS = 12


class EvaluationTask:
    """One long measurement (cross-validation, permutation importance) with progress, elapsed time and cancel.

    ``work`` is called with a progress function ``(message, fraction)`` and a :class:`CancelToken`; its return value
    becomes ``result``. The task runs on a daemon thread (:meth:`start`) or on the calling thread
    (:meth:`run_inline`); either way the outcome is recorded rather than raised: ``state`` ends as ``"done"``,
    ``"cancelled"`` (the work raised :class:`TrainingCancelled`, or returned after a cancel) or ``"failed"`` (with
    ``error``). Tasks are kept in a small process registry (:func:`get_task`) so a page can find one by its id.

    An ``exclusive`` task takes the process-wide work slot (:func:`graticule.models.jobs.claim_slot`, described by
    ``holder``) when it starts and gives it back when it ends; :meth:`start` and :meth:`run_inline` raise
    :class:`~graticule.models.jobs.JobBusyError` (and the task stays unstarted) while a fit or another exclusive
    task holds it.
    """

    def __init__(self, kind: str, run_id: str, work: Callable[[ProgressFn, CancelToken], Any], *,
                 label: str = "", exclusive: bool = False, holder: str | None = None) -> None:
        self.task_id = f"{kind}-{uuid.uuid4().hex[:12]}"
        self.kind = kind
        self.run_id = run_id
        self.label = label
        self.exclusive = bool(exclusive)
        self.holder = holder or f"a {label or kind}"
        self._holds_slot = False
        self.result: Any = None
        self.error: str | None = None
        self.extra: dict[str, Any] = {}
        self._work = work
        self._token = CancelToken()
        self._lock = threading.Lock()
        self._state: TaskState = "queued"
        self._fraction = 0.0
        self._message = "Waiting to start"
        self._started: float | None = None
        self._ended: float | None = None
        self._done = threading.Event()
        self._launched = False
        with _TASKS_LOCK:
            _TASKS[self.task_id] = self
            while len(_TASKS) > _KEEP_TASKS:
                oldest = next((t for t in _TASKS.values() if t.finished), None)
                if oldest is None:
                    break
                _TASKS.pop(oldest.task_id, None)

    def _progress(self, message: str, fraction: float) -> None:
        """Record progress (thread-safe; the fraction never moves backwards)."""
        with self._lock:
            self._message = str(message)
            self._fraction = max(self._fraction, float(min(max(fraction, 0.0), 1.0)))

    def _claim(self) -> None:
        """Mark the task as running (a task runs once); an exclusive task first takes the work slot."""
        with self._lock:
            if self._launched:
                raise RuntimeError("This task has already been started.")
            busy = self.exclusive and not claim_slot(self.holder)
            if not busy:
                self._holds_slot = self.exclusive
                self._launched = True
                self._state = "running"
                self._started = time.perf_counter()
                self._message = "Starting"
        if busy:
            forget_task(self.task_id)  # a task turned away never runs; it must not linger in the registry
            raise JobBusyError(busy_message())

    def _execute(self) -> None:
        """Run the work and record its outcome."""
        state: TaskState = "failed"
        try:
            result = self._work(self._progress, self._token)
            self.result = result
            cancelled = bool(getattr(result, "attrs", {}).get("cancelled")) if result is not None else False
            state = "cancelled" if (cancelled or (self._token.cancelled and result is None)) else "done"
        except TrainingCancelled:
            state = "cancelled"
        except BaseException as exc:  # noqa: BLE001 - the outcome is reported to the page, never raised in a thread
            state = "failed"
            self.error = f"{type(exc).__name__}: {exc}"
            self.extra["traceback"] = traceback.format_exc()
        finally:
            with self._lock:
                self._state = state
                self._ended = time.perf_counter()
                if state == "done":
                    self._fraction = 1.0
                    self._message = "Done"
                elif state == "cancelled":
                    self._message = "Cancelled"
                else:
                    self._message = "Failed"
                release = self._holds_slot
                self._holds_slot = False
            if release:
                release_slot()
            self._done.set()

    def start(self) -> None:
        """Run the work on a daemon thread (raises :class:`~graticule.models.jobs.JobBusyError` when an exclusive
        task finds the work slot taken)."""
        self._claim()
        threading.Thread(target=self._execute, name=f"graticule-{self.task_id}", daemon=True).start()

    def run_inline(self) -> Any:
        """Run the work on the calling thread; returns ``result`` (None when it failed or was cancelled early)."""
        self._claim()
        self._execute()
        return self.result

    def cancel(self) -> None:
        """Ask the work to stop at its next check."""
        self._token.cancel()
        with self._lock:
            if self._state == "running":
                self._message = "Cancelling..."

    def wait(self, timeout: float | None = None) -> bool:
        """Block until the task has finished (or ``timeout`` seconds passed); True when it finished."""
        return self._done.wait(timeout)

    @property
    def finished(self) -> bool:
        """True once the task has reached a final state."""
        return self._done.is_set()

    def snapshot(self) -> TaskSnapshot:
        """An immutable copy of the current progress."""
        with self._lock:
            started, ended = self._started, self._ended
            elapsed = 0.0 if started is None else (ended if ended is not None else time.perf_counter()) - started
            return TaskSnapshot(task_id=self.task_id, kind=self.kind, run_id=self.run_id, state=self._state,
                                fraction=self._fraction, message=self._message, elapsed=elapsed, error=self.error)


def get_task(task_id: str | None) -> EvaluationTask | None:
    """The task registered under ``task_id`` (None when unknown or forgotten)."""
    if task_id is None:
        return None
    with _TASKS_LOCK:
        return _TASKS.get(task_id)


def forget_task(task_id: str) -> None:
    """Drop a task from the registry (its result stays wherever it was stored)."""
    with _TASKS_LOCK:
        _TASKS.pop(task_id, None)


__all__ = [
    "CHANNEL_ORDER", "CVPlan", "ChannelEvaluation", "EvaluationTask", "TaskSnapshot", "cached_evaluations",
    "channel_label", "class_curves", "classification_metrics", "cross_validate_run", "cv_fold_frame",
    "downsample_curve", "estimate_cv_seconds", "evaluate_channel", "evaluate_run", "forget_task", "get_task",
    "has_test_rows", "held_out_counts", "held_out_repeats", "leaderboard", "native_importance", "normalise_rows",
    "per_class_frame",
    "per_class_metrics", "permutation_importance_for", "permutation_plan", "plan_cross_validation",
    "quick_metrics", "remember_cross_validation", "remember_permutation", "score_columns",
    "repeats_sentence", "single_flow_latency_ms", "stored_cross_validation", "stored_permutations",
]
