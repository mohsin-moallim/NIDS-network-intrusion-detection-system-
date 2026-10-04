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

Distinct flows and recorded traffic. Exact repeats are merged before the split, so every reading above counts each
DISTINCT flow once, however often the files recorded it; that stays the primary reading. :func:`traffic_readings`
adds a second, clearly labelled ESTIMATE of how the channels read the recorded traffic itself: each held-out row
``i`` is weighted by ``w_i = copies_i / f_c(i)``, where ``copies_i`` is how many rows of the cleaned files (after the
bad-value strategy) the row stands for once every exact repeat is counted (``TrainingData.test_copies``, which also
folds in the rows merged over the chosen columns at 02 Fit) and ``f_c`` is the share of the row's class that the
01 Sample draw kept (rows of that class in the sample divided by its distinct rows available before sampling; 1 when
the class was taken whole). The sum of the weights estimates the held-out share of the recorded traffic, and the
weighted readings estimate what a channel would read over all recorded flows of the selected files, leaving out rows
dropped for bad values (and rows a non-default conflict policy removed). Like :func:`evaluate_run`, it is computed
once per run from the stored test-set probabilities and kept on the run; nothing is refitted or rescored.

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
from dataclasses import dataclass, field
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

from graticule.data.sampling import apply_class_options
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
    """How many rows of the cleaned files the held-out rows stand for, or None when the run carries no repeat counts.

    Repeats are merged before the split (exact ones at 01 Sample, rows identical over the chosen columns at 02 Fit),
    so every held-out row is one distinct flow and every reading counts it once, however often it occurred in the
    files. Returns ``rows`` (held-out rows), ``flows`` (rows of the cleaned files they stand for, every copy
    counted: the recorded copies, NOT scaled up for the flows 01 Sample left out; see :func:`traffic_summary` for
    that estimate), ``largest`` (the most copies any one row stands for) and ``repeated`` (held-out rows standing
    for more than one row of the files).
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
    return (f"Each held-out row is a distinct flow: repeats were merged before the split (exact ones, and rows "
            f"identical over the chosen columns), so these {repeats['rows']:,} rows stand for "
            f"{repeats['flows']:,} rows of the cleaned files (one of them for {repeats['largest']:,} copies). Every "
            "reading counts a distinct flow once. A whole file scored at 05 Assay counts every repeat, so a channel "
            "that misses a much-repeated flow reads lower there than here.")


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
# Recorded-traffic readings (an estimate; see the module notes)
# --------------------------------------------------------------------------------------------------------------
#: Attribute under which the recorded-traffic readings are kept on a run.
TRAFFIC_ATTR = "traffic_evaluations"
#: Prefix of the recorded-traffic columns of the CSV exports and of their keys in a saved set's manifest.
TRAFFIC_PREFIX = "traffic_"
#: Column (and manifest key) holding the sum of the weights: the recorded flows the held-out rows stand for.
TRAFFIC_FLOWS_COLUMN = "traffic_flows_represented"
#: Attribute under which the weights, their summary and the heavy-flow account are kept on a run (computed once).
TRAFFIC_BASIS_ATTR = "traffic_basis"
#: The estimate is marked as resting on a few flows (and the pages warn) when the verdicts on the heavily repeated
#: flows that missed the held-out rows could move weighted accuracy or weighted balanced accuracy by at least this
#: much (see :class:`HeavyClass`), when the verdict on one held-out row can (see :class:`TrafficSummary`), or when at
#: most :data:`CONCENTRATED_ROWS` rows carry half of the weight. On Wednesday nine DoS Hulk flows, each recorded
#: 1,317 to 9,329 times, carry 16.5 % of the attack traffic; whether any of them lands among the held-out rows is
#: chance (at the 200,000-row budget, seed 42 and binary mode one of them does: 1,485 copies, an estimated 4,627
#: recorded flows, 2.8 % of the weight).
SWING_LIMIT = 0.01
CONCENTRATED_ROWS = 25
#: Per-class columns of the recorded-traffic readings (``flows`` is the weighted count of the class's rows).
TRAFFIC_PER_CLASS_COLUMNS: tuple[str, ...] = ("class", "flows", "precision", "recall", "f1", "roc_auc",
                                              "average_precision")


def _weights(weights: npt.ArrayLike, n_rows: int) -> np.ndarray:
    """Row weights as a flat float64 array of length ``n_rows``; raises ``ValueError`` unless finite and >= 0."""
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    if w.shape != (n_rows,):
        raise ValueError(f"Expected {n_rows} weights, got {w.size}.")
    if not np.isfinite(w).all() or (w < 0).any():
        raise ValueError("Weights must be finite and not negative.")
    return w


def _weighted_auc(is_class: np.ndarray, score: np.ndarray, w: np.ndarray) -> float:
    """Weighted ROC-AUC of one class against the rest; NaN unless both sides carry weight."""
    inside, outside = float(w[is_class].sum()), float(w[~is_class].sum())
    if inside <= 0 or outside <= 0:
        return float("nan")
    return float(roc_auc_score(is_class, score, sample_weight=w))


def _weighted_ap(is_class: np.ndarray, score: np.ndarray, w: np.ndarray) -> float:
    """Weighted average precision of one class against the rest; NaN when the class carries no weight."""
    if float(w[is_class].sum()) <= 0:
        return float("nan")
    return float(average_precision_score(is_class, score, sample_weight=w))


def weighted_classification_metrics(y_true: npt.ArrayLike, y_pred: npt.ArrayLike, proba: npt.ArrayLike,
                                    n_classes: int, weights: npt.ArrayLike) -> dict[str, float]:
    """The readings of :func:`classification_metrics` (same keys) with row ``i`` counted ``weights[i]`` times.

    Accuracy is the weight of the rows read right over the total weight; balanced accuracy the mean, over the classes
    whose rows carry weight, of each class's weighted recall. Precision, recall and F1 (binary: class code 1; else
    macro and weighted averages, the latter by weighted class totals), ROC-AUC and average precision pass the weights
    to scikit-learn as ``sample_weight``. With whole-number weights every reading equals that of
    :func:`classification_metrics` on the rows repeated that many times. No weight at all gives NaN readings.
    """
    truth, guess = _codes(y_true), _codes(y_pred)
    k = int(n_classes)
    if truth.shape != guess.shape:
        raise ValueError("y_true and y_pred must have the same length.")
    w = _weights(weights, len(truth))
    nan = float("nan")
    names = ["accuracy", "balanced_accuracy", "f1_macro", "f1_weighted", "precision_macro", "recall_macro",
             "precision_weighted", "recall_weighted", "roc_auc", "average_precision"]
    total = float(w.sum())
    if truth.size == 0 or total <= 0:
        out = {name: nan for name in names}
        if k == 2:
            out.update(precision=nan, recall=nan, f1=nan)
        return out
    present = [int(c) for c in np.unique(truth) if float(w[truth == c].sum()) > 0]
    recalls = [float(w[(truth == c) & (guess == c)].sum()) / float(w[truth == c].sum()) for c in present]
    labels = list(range(k))
    out: dict[str, float] = {
        "accuracy": float(w[truth == guess].sum()) / total,
        "balanced_accuracy": float(np.mean(recalls)),
        "f1_macro": float(f1_score(truth, guess, labels=labels, average="macro", sample_weight=w, zero_division=0)),
        "f1_weighted": float(f1_score(truth, guess, labels=labels, average="weighted", sample_weight=w,
                                      zero_division=0)),
    }
    for average in ("macro", "weighted"):
        p, r, _, _ = precision_recall_fscore_support(truth, guess, labels=labels, average=average, sample_weight=w,
                                                     zero_division=0)
        out[f"precision_{average}"] = float(p)
        out[f"recall_{average}"] = float(r)
    scores = _scores(proba, len(truth), k)
    if k == 2:
        p, r, f, _ = precision_recall_fscore_support(truth, guess, labels=[1], average=None, sample_weight=w,
                                                     zero_division=0)
        out["precision"], out["recall"], out["f1"] = float(p[0]), float(r[0]), float(f[0])
        out["roc_auc"] = _weighted_auc(truth == 1, scores[:, 1], w)
        out["average_precision"] = _weighted_ap(truth == 1, scores[:, 1], w)
        return out
    if len(present) == k:
        out["roc_auc"] = float(roc_auc_score(truth, scores, multi_class="ovr", average="macro", labels=labels,
                                             sample_weight=w))
    else:
        out["roc_auc"] = nan
    aps = [_weighted_ap(truth == c, scores[:, c], w) for c in present]
    out["average_precision"] = float(np.mean(aps)) if aps else nan
    return out


def weighted_per_class_metrics(y_true: npt.ArrayLike, y_pred: npt.ArrayLike, proba: npt.ArrayLike,
                               classes: Sequence[str], weights: npt.ArrayLike) -> pd.DataFrame:
    """One row per class in code order, weighted like :func:`weighted_classification_metrics`.

    Columns (:data:`TRAFFIC_PER_CLASS_COLUMNS`): class, flows (the summed weight of the class's rows, in place of
    :func:`per_class_metrics`' support), precision, recall, f1, roc_auc and average_precision (one-vs-rest on that
    class's probability; NaN when the class carries no weight).
    """
    truth, guess = _codes(y_true), _codes(y_pred)
    k = len(classes)
    labels = list(range(k))
    w = _weights(weights, len(truth))
    if truth.size == 0 or float(w.sum()) <= 0:
        p = r = f = np.zeros(k)
        s = np.zeros(k, dtype=np.float64)
        aucs = aps = [float("nan")] * k
    else:
        p, r, f, _ = precision_recall_fscore_support(truth, guess, labels=labels, average=None, sample_weight=w,
                                                     zero_division=0)
        s = np.array([float(w[truth == c].sum()) for c in labels], dtype=np.float64)
        scores = _scores(proba, len(truth), k)
        aucs = [_weighted_auc(truth == c, scores[:, c], w) for c in labels]
        aps = [_weighted_ap(truth == c, scores[:, c], w) for c in labels]
    return pd.DataFrame({
        "class": pd.Series([str(c) for c in classes], dtype="str"),
        "flows": np.asarray(s, dtype=np.float64),
        "precision": np.asarray(p, dtype=np.float64),
        "recall": np.asarray(r, dtype=np.float64),
        "f1": np.asarray(f, dtype=np.float64),
        "roc_auc": np.asarray(aucs, dtype=np.float64),
        "average_precision": np.asarray(aps, dtype=np.float64),
    })


def weighted_standard_errors(y_true: npt.ArrayLike, y_pred: npt.ArrayLike,
                             weights: npt.ArrayLike) -> dict[str, float]:
    """Rough standard errors of weighted accuracy and weighted balanced accuracy (keys ``accuracy`` and
    ``balanced_accuracy``), from the spread among the rows given.

    Each is a weighted ratio, so its error is linearised: for accuracy ``p = sum(w e) / sum(w)`` (``e`` = 1 when a
    row is read right) the error is ``sqrt(sum(w^2 (e - p)^2)) / sum(w)``; each class's weighted recall gets the same
    within its own rows, and balanced accuracy, their mean over K classes, the root of their summed squares over K.
    With unit weights the accuracy error is the familiar ``sqrt(p (1 - p) / n)``. A few heavy rows among those given
    make it large. It can only see the rows it is given: for the recorded-traffic estimate that means the held-out
    rows, so heavily repeated flows that missed them leave no trace in it (see :func:`heavy_flow_bounds`, which
    accounts for them). NaN without weight.
    """
    truth, guess = _codes(y_true), _codes(y_pred)
    w = _weights(weights, len(truth))
    right = (truth == guess).astype(np.float64)
    total = float(w.sum())
    nan = float("nan")
    if total <= 0:
        return {"accuracy": nan, "balanced_accuracy": nan}
    p = float((w * right).sum()) / total
    accuracy = float(np.sqrt(np.sum(w ** 2 * (right - p) ** 2))) / total
    parts = []
    for c in np.unique(truth):
        inside = truth == c
        weight = float(w[inside].sum())
        if weight <= 0:
            continue
        recall = float((w[inside] * right[inside]).sum()) / weight
        parts.append(float(np.sum(w[inside] ** 2 * (right[inside] - recall) ** 2)) / weight ** 2)
    balanced = float(np.sqrt(np.sum(parts))) / len(parts) if parts else nan
    return {"accuracy": accuracy, "balanced_accuracy": balanced}


def weighted_confusion(y_true: npt.ArrayLike, y_pred: npt.ArrayLike, n_classes: int,
                       weights: npt.ArrayLike) -> np.ndarray:
    """Confusion matrix (rows: true class, columns: predicted, code order) of summed weights, as float64."""
    truth, guess = _codes(y_true), _codes(y_pred)
    w = _weights(weights, len(truth))
    k = int(n_classes)
    matrix = np.zeros((k, k), dtype=np.float64)
    if truth.size:
        np.add.at(matrix, (truth, guess), w)
    return matrix


def _sampling_scales(run: "TrainingRun") -> tuple[dict[str, float], bool] | None:
    """Per class drawn at 01 Sample, distinct rows available divided by rows kept (1 / f_c), with the Web Attack
    merge flag; None when the run does not record its sampling shares."""
    shares = (getattr(run.data, "reports", None) or {}).get("sampling")
    if not isinstance(shares, Mapping):
        return None
    before, after = shares.get("before"), shares.get("after")
    if not isinstance(before, Mapping) or not isinstance(after, Mapping):
        return None
    scales: dict[str, float] = {}
    for name, available in before.items():
        try:
            total, kept = int(available), int(after.get(name, 0) or 0)
        except (TypeError, ValueError):
            return None
        if total > 0 and kept > 0:
            scales[str(name)] = total / kept
    return scales, bool(shares.get("merge_web_attacks", False))


def sampling_fractions(run: "TrainingRun") -> dict[str, float] | None:
    """f_c for every class drawn at 01 Sample: rows kept in the sample divided by distinct rows available before the
    draw (1.0 for a class taken whole), from the run's reports; None when the run does not record them."""
    found = _sampling_scales(run)
    if found is None:
        return None
    return {name: 1.0 / scale for name, scale in found[0].items()}


def _traffic_basis(run: "TrainingRun") -> tuple[np.ndarray | None, str | None]:
    """(weights, None) when the run can be weighted to its recorded traffic, else (None, the reason in words)."""
    if not has_test_rows(run):
        return None, "This run holds no held-out rows."
    data = run.data
    n = len(data.y_test)
    copies = getattr(data, "test_copies", None)
    if copies is None:
        return None, ("This run does not record how many recorded flows each held-out row stands for (its sample "
                      "was drawn without repeat counts), so only the distinct-flow readings exist.")
    copies = np.asarray(copies, dtype=np.float64).reshape(-1)
    if len(copies) != n or not np.isfinite(copies).all() or (copies < 1).any():
        return None, "The repeat counts kept with this run do not match its held-out rows."
    found = _sampling_scales(run)
    if found is None:
        return None, ("This run does not record how 01 Sample thinned each class (it was fitted before Graticule "
                      "kept those shares); fit it again to add the recorded-traffic estimate.")
    scales, merged = found
    labels = np.asarray(getattr(data, "detailed_test_labels", np.empty(0)), dtype=object).astype(str)
    if len(labels) != n:
        return None, "The detailed labels kept with this run do not match its held-out rows."
    strata = apply_class_options(pd.Series(labels, dtype="str"), merge_web_attacks=merged).to_numpy(dtype=object)
    names, inverse = np.unique(strata.astype(str), return_inverse=True)
    missing = [str(name) for name in names if str(name) not in scales]
    if missing:
        return None, f"The sampling report of this run holds no share for {', '.join(missing)}."
    factor = np.array([scales[str(name)] for name in names], dtype=np.float64)
    return copies * factor[inverse], None


@dataclass(frozen=True)
class HeavyClass:
    """The heavily repeated flows of one class of the target (an entry of ``TrainingData.reports["heavy_flows"]``,
    see :func:`graticule.models.train.heavy_flow_report`).

    01 Sample knows how often the cleaned files recorded EVERY distinct flow it drew from, sampled or not, so these
    counts cover the flows the sample left out as well as those it kept. A flow is heavily repeated in a class when
    the files recorded it at least ``threshold`` times.

    Attributes:
        name: class of the target.
        recorded: rows of the cleaned files in this class (every copy counted).
        threshold: copies from which a flow counts as heavily repeated in this class.
        flows: heavily repeated flows the cleaned files hold in this class.
        copies: their rows in the cleaned files.
        held_out: how many of them are among the held-out rows.
        held_out_copies: the copies of those held out.
    """

    name: str
    recorded: int
    threshold: int
    flows: int
    copies: int
    held_out: int
    held_out_copies: int

    @property
    def unknown_copies(self) -> int:
        """Copies of the heavily repeated flows that are NOT among the held-out rows (trained on, or never sampled):
        no held-out verdict covers them."""
        return max(int(self.copies) - int(self.held_out_copies), 0)

    @property
    def unknown_share(self) -> float:
        """:attr:`unknown_copies` over the class's recorded rows: how far their verdicts alone can move the class's
        recall over the recorded traffic."""
        return self.unknown_copies / self.recorded if self.recorded > 0 else 0.0


def _heavy_account(data: Any, n_rows: int) -> tuple[tuple[HeavyClass, ...], np.ndarray] | None:
    """The run's heavy-flow account per target class (code order) and a mask of the held-out rows that hold a
    heavily repeated flow; None when the run does not record it or it does not fit the held-out rows."""
    report = (getattr(data, "reports", None) or {}).get("heavy_flows")
    known = report.get("classes") if isinstance(report, Mapping) else None
    if not isinstance(known, Mapping):
        return None
    account: list[HeavyClass] = []
    mask = np.zeros(int(n_rows), dtype=bool)
    try:
        for name in data.classes:
            entry = known.get(str(name))
            if not isinstance(entry, Mapping):
                return None
            account.append(HeavyClass(
                name=str(name), recorded=int(entry.get("recorded", 0)), threshold=int(entry.get("threshold", 0)),
                flows=int(entry.get("flows", 0)), copies=int(entry.get("copies", 0)),
                held_out=int(entry.get("held_out", 0)), held_out_copies=int(entry.get("held_out_copies", 0))))
            rows = np.asarray(list(entry.get("rows") or ()), dtype=np.int64)
            if rows.size and (int(rows.min()) < 0 or int(rows.max()) >= n_rows):
                return None
            mask[rows] = True
    except (TypeError, ValueError):
        return None
    if not account or any(c.recorded <= 0 for c in account):
        return None
    return tuple(account), mask


@dataclass(frozen=True)
class TrafficSummary:
    """What the recorded-traffic weights of a run's held-out rows add up to (see :func:`traffic_weights`), and how
    much of the estimate rests on a few flows.

    Attributes:
        rows: held-out rows.
        flows: the sum of the weights: the estimated recorded flows the held-out rows stand for (an estimate of the
            held-out share of the recorded traffic of the classes in play).
        repeats: the sum of the repeat counts alone: rows of the cleaned files behind the held-out rows themselves
            (recorded copies, not scaled up for the flows 01 Sample left out).
        by_class: summed weight per class of the target, in class-code order.
        largest: the largest single weight (estimated recorded flows of the heaviest held-out row).
        largest_share: ``largest / flows``: how far the verdict on that one row can move weighted accuracy.
        balanced_swing: the most the verdict on any one row can move weighted balanced accuracy (its weight over its
            class's weight, divided by the number of classes).
        rows_for_half: the fewest held-out rows whose weights add up to at least half of ``flows``.
        sampled_classes: sampling share f_c of every class drawn at 01 Sample (1.0 when taken whole).
        largest_copies: copies in the cleaned files of the heaviest held-out row.
        largest_fraction: the sampling share of that row's class (``largest = largest_copies / largest_fraction``).
        recorded: rows of the cleaned files in the classes of the target (None when the run does not record them).
        test_share: the share of every class held out at 02 Fit (None when unknown).
        heavy: the heavy-flow account per class of the target (empty when the run does not record it).
    """

    rows: int
    flows: float
    repeats: int
    by_class: dict[str, float]
    largest: float
    largest_share: float
    balanced_swing: float
    rows_for_half: int
    sampled_classes: dict[str, float]
    largest_copies: int = 0
    largest_fraction: float = 1.0
    recorded: int | None = None
    test_share: float | None = None
    heavy: tuple[HeavyClass, ...] = ()

    @property
    def heavy_accuracy_swing(self) -> float:
        """How far the verdicts on the heavily repeated flows that missed the held-out rows can move weighted
        accuracy over the recorded traffic (their copies over all recorded rows of the target's classes)."""
        total = sum(c.recorded for c in self.heavy)
        return sum(c.unknown_copies for c in self.heavy) / total if total > 0 else 0.0

    @property
    def heavy_balanced_swing(self) -> float:
        """How far those verdicts can move weighted balanced accuracy (the mean over classes of their share)."""
        return float(np.mean([c.unknown_share for c in self.heavy])) if self.heavy else 0.0

    @property
    def heavy_concentrated(self) -> bool:
        """True when heavily repeated flows that missed the held-out rows could move a reading by
        :data:`SWING_LIMIT` or more: neither the estimate nor its standard errors can see them."""
        return max(self.heavy_accuracy_swing, self.heavy_balanced_swing) >= SWING_LIMIT

    @property
    def rows_concentrated(self) -> bool:
        """True when a handful of held-out rows carry much of the weight (see :data:`SWING_LIMIT`)."""
        return (self.largest_share >= SWING_LIMIT or self.balanced_swing >= SWING_LIMIT
                or self.rows_for_half <= CONCENTRATED_ROWS)

    @property
    def concentrated(self) -> bool:
        """True when the estimate rests on a few flows: heavily repeated flows missing from the held-out rows
        (:attr:`heavy_concentrated`) or a few heavy held-out rows (:attr:`rows_concentrated`)."""
        return self.heavy_concentrated or self.rows_concentrated


def _summary(run: "TrainingRun", w: np.ndarray,
             heavy: tuple[tuple[HeavyClass, ...], np.ndarray] | None) -> TrafficSummary:
    """The :class:`TrafficSummary` of the weights ``w`` of ``run`` (``heavy`` from :func:`_heavy_account`)."""
    classes = [str(c) for c in run.data.classes]
    codes = _codes(run.data.y_test)
    copies = np.asarray(run.data.test_copies, dtype=np.int64)
    total = float(w.sum())
    ordered = np.sort(w)[::-1]
    half = int(np.searchsorted(np.cumsum(ordered), total / 2.0, side="left")) + 1 if total > 0 else 0
    heaviest = int(np.argmax(w)) if len(w) else -1
    largest = float(w[heaviest]) if heaviest >= 0 else 0.0
    by_class = np.bincount(codes, weights=w, minlength=len(classes)) if len(w) else np.zeros(len(classes))
    present = int((by_class > 0).sum())
    per_class_share = np.divide(w, by_class[codes], out=np.zeros_like(w), where=by_class[codes] > 0)
    swing = float(per_class_share.max()) / present if len(w) and present else 0.0
    account = heavy[0] if heavy is not None else ()
    share = getattr(getattr(run, "request", None), "test_share", None)
    return TrafficSummary(
        rows=int(len(w)), flows=total, repeats=int(copies.sum()),
        by_class={name: float(w[codes == i].sum()) for i, name in enumerate(classes)},
        largest=largest, largest_share=largest / total if total > 0 else 0.0, balanced_swing=swing,
        rows_for_half=min(half, len(w)), sampled_classes=dict(sampling_fractions(run) or {}),
        largest_copies=int(copies[heaviest]) if heaviest >= 0 else 0,
        largest_fraction=float(copies[heaviest]) / largest if heaviest >= 0 and largest > 0 else 1.0,
        recorded=sum(c.recorded for c in account) if account else None,
        test_share=float(share) if isinstance(share, (int, float)) else None, heavy=tuple(account),
    )


@dataclass(frozen=True)
class _TrafficBasis:
    """What every recorded-traffic reading of a run starts from, computed once and kept on the run."""

    data: Any
    weights: np.ndarray | None
    reason: str | None
    summary: TrafficSummary | None
    heavy: tuple[tuple[HeavyClass, ...], np.ndarray] | None


def _basis(run: "TrainingRun") -> _TrafficBasis:
    """The run's weights, unavailable reason, summary and heavy-flow account: computed on the first call and kept on
    the run (:data:`TRAFFIC_BASIS_ATTR`), so redrawing a page never computes them again. Kept per data object: a run
    given other matrices gets a fresh basis. Thread-safe."""
    data = getattr(run, "data", None)
    kept = getattr(run, TRAFFIC_BASIS_ATTR, None)
    if isinstance(kept, _TrafficBasis) and kept.data is data:
        return kept
    with _EVAL_LOCK:
        kept = getattr(run, TRAFFIC_BASIS_ATTR, None)
        if isinstance(kept, _TrafficBasis) and kept.data is data:
            return kept
        weights, reason = _traffic_basis(run)
        heavy = summary = None
        if weights is not None:
            weights.setflags(write=False)
            heavy = _heavy_account(data, len(weights))
            summary = _summary(run, weights, heavy)
        kept = _TrafficBasis(data=data, weights=weights, reason=reason, summary=summary, heavy=heavy)
        try:
            setattr(run, TRAFFIC_BASIS_ATTR, kept)
        except AttributeError:  # an object that takes no new attributes: computed again next time
            pass
        return kept


def traffic_weights(run: "TrainingRun") -> np.ndarray | None:
    """The recorded-traffic weight of every held-out row (float64, test-row order), or None when unavailable.

    ``w_i = copies_i / f_c(i)``: the rows of the cleaned files (after the bad-value strategy) held-out row ``i``
    stands for once every repeat is counted (``TrainingData.test_copies``: its exact repeats, and the rows merged
    with it at 02 Fit because they are identical over the chosen columns), divided by the 01 Sample share f_c of the
    row's detailed class (its sampling class when the Web Attack types were drawn as one; 1 when the class was taken
    whole). Rows merged at 02 Fit that carried other detailed labels of the same target class are scaled by the kept
    row's share. Computed once per run (a fresh copy is returned). :func:`traffic_unavailable_reason` says why None
    was returned.
    """
    weights = _basis(run).weights
    return None if weights is None else np.array(weights, dtype=np.float64, copy=True)


def traffic_unavailable_reason(run: "TrainingRun") -> str | None:
    """Why ``run`` has no recorded-traffic readings, in plain words; None when it has them."""
    return _basis(run).reason


def traffic_summary(run: "TrainingRun") -> TrafficSummary | None:
    """Totals and concentration of the recorded-traffic weights of ``run`` (computed once per run); None when they
    are unavailable."""
    return _basis(run).summary


def heavy_flow_bounds(run: "TrainingRun", y_pred: npt.ArrayLike,
                      weights: npt.ArrayLike | None = None) -> dict[str, tuple[float, float]] | None:
    """Where weighted accuracy and balanced accuracy over the recorded traffic can lie for the verdicts ``y_pred``
    on the held-out rows, given that no held-out verdict covers the heavily repeated flows that missed them.

    Per class of the target, with R its rows in the cleaned files and U the copies of its heavily repeated flows
    that are not held out (:class:`HeavyClass`): the held-out rows that hold a heavily repeated flow count with their
    own copies, exactly; the other held-out rows give the class's rate on the remaining rows (weighted by
    ``weights``, default :func:`traffic_weights`); and the U copies are counted as all misread (low) or all read
    right (high). Accuracy weighs the classes by R, balanced accuracy averages them. Returns ``{"accuracy": (low,
    high), "balanced_accuracy": (low, high)}``, or None when the run does not record its heavy-flow account. The range
    leaves out the ordinary spread of the other rows (the standard errors cover that).
    """
    basis = _basis(run)
    if basis.weights is None or basis.heavy is None:
        return None
    account, holds_heavy = basis.heavy
    truth, guess = _codes(run.data.y_test), _codes(y_pred)
    if truth.shape != guess.shape:
        raise ValueError("y_pred must hold one verdict per held-out row.")
    w = basis.weights if weights is None else _weights(weights, len(truth))
    copies = np.asarray(run.data.test_copies, dtype=np.float64)
    right = truth == guess
    lows: list[float] = []
    highs: list[float] = []
    sizes: list[float] = []
    for code, entry in enumerate(account):
        inside = truth == code
        heavy_rows, light = inside & holds_heavy, inside & ~holds_heavy
        recorded = float(entry.recorded)
        known, known_right = float(copies[heavy_rows].sum()), float(copies[heavy_rows & right].sum())
        unknown = float(entry.unknown_copies)
        rest = max(recorded - unknown - known, 0.0)
        light_weight = float(w[light].sum())
        if light_weight > 0:
            low = (float(w[light & right].sum()) / light_weight * rest + known_right) / recorded
            high = low + unknown / recorded
        else:
            low, high = known_right / recorded, (known_right + rest + unknown) / recorded
        lows.append(min(max(low, 0.0), 1.0))
        highs.append(min(max(high, 0.0), 1.0))
        sizes.append(recorded)
    size = np.asarray(sizes)
    return {
        "accuracy": (float(np.dot(size, lows) / size.sum()), float(np.dot(size, highs) / size.sum())),
        "balanced_accuracy": (float(np.mean(lows)), float(np.mean(highs))),
    }


def flows_text(value: float) -> str:
    """A count of ESTIMATED flows for reading, always marked as approximate and rounded to three significant
    figures: ``"about 812"``, ``"about 4,630"``, ``"about 166,000"``, ``"about 1.3 million"``."""
    number = float(value)
    if not np.isfinite(number):
        return "n/a"
    if number < 1_000_000:
        whole = int(round(number))
        digits = 3 - len(str(abs(whole))) if whole else 0
        return f"about {int(round(number, min(digits, 0))):,}"
    return f"about {number / 1_000_000:,.1f} million"


def traffic_sentence(summary: TrafficSummary) -> str:
    """How the recorded-traffic estimate is made and what it stands for, in plain sentences.

    It keeps the two counts apart: the rows of the cleaned files the held-out rows stand for (recorded copies,
    exact) and the recorded flows they are estimated to stand for once the classes 01 Sample thinned are scaled
    back up.
    """
    text = ("Each held-out row is weighted by the recorded flows it stands for: its copies in the cleaned files "
            "(its exact repeats, and rows identical to it over the chosen columns), scaled up by how much 01 Sample "
            f"thinned its class. The {summary.rows:,} held-out rows stand for {summary.repeats:,} rows of the "
            f"cleaned files and for {flows_text(summary.flows)} recorded flows once the thinned classes are scaled "
            "back up")
    if summary.recorded and summary.test_share:
        text += (f": an estimate of the held-out share ({summary.test_share:.0%}) of the {summary.recorded:,} "
                 "flows the cleaned files hold in these classes")
    return text + (". These readings estimate how each channel reads the traffic as recorded, not only its distinct "
                   "flows.")


def _heaviest_row_text(summary: TrafficSummary) -> str:
    """The sentence on the heaviest held-out row (and on a few rows carrying half the weight)."""
    if summary.largest_fraction < 1.0 - 1e-9:
        made = (f"its {summary.largest_copies:,} copies in the cleaned files divided by its class's sampling share, "
                f"{summary.largest_fraction:.3f}")
    else:
        made = f"its {summary.largest_copies:,} copies in the cleaned files (its class was taken whole)"
    text = (f"The heaviest held-out row weighs {flows_text(summary.largest)} estimated recorded flows ({made}), "
            f"{summary.largest_share:.1%} of the weight, so its verdict alone moves weighted accuracy by up to "
            f"{summary.largest_share:.3f}")
    if summary.balanced_swing > summary.largest_share:
        text += f", and one row's verdict moves weighted balanced accuracy by up to {summary.balanced_swing:.3f}"
    if summary.rows_for_half <= CONCENTRATED_ROWS:
        rows = "row carries" if summary.rows_for_half == 1 else "rows carry"
        text += f"; {summary.rows_for_half:,} {rows} half of all the weight"
    text += "."
    if summary.heavy_concentrated:
        text += (" The range table counts every held-out row that holds a heavily repeated flow with its own copies, "
                 "not scaled up, so where an estimate falls outside its range, the range is the better guide.")
    return text


def _heavy_text(summary: TrafficSummary) -> str:
    """The sentences on heavily repeated flows that missed the held-out rows."""
    parts = []
    for entry in summary.heavy:
        if entry.unknown_copies <= 0:
            continue
        missing = entry.flows - entry.held_out
        if entry.held_out == 0:
            held = "none of them is among the held-out rows, so no held-out verdict covers them"
        else:
            held = (f"only {entry.held_out:,} of them {'is' if entry.held_out == 1 else 'are'} among the held-out "
                    f"rows, so no held-out verdict covers the other {missing:,}")
        flows = "flow" if entry.flows == 1 else "flows"
        parts.append(f"{entry.flows:,} {entry.name} {flows}, each recorded at least {entry.threshold:,} times, "
                     f"{'makes' if entry.flows == 1 else 'make'} up {entry.copies:,} of the class's "
                     f"{entry.recorded:,} rows in the cleaned files ({entry.copies / entry.recorded:.1%}), and "
                     f"{held} ({entry.unknown_copies:,} rows, {entry.unknown_share:.1%} of the class)")
    text = "Heavily repeated flows decide much of this estimate. " + "; ".join(parts) + ". "
    return text + (f"Their verdicts alone could move accuracy over the recorded traffic by up to "
                   f"{summary.heavy_accuracy_swing:.3f} and balanced accuracy by up to "
                   f"{summary.heavy_balanced_swing:.3f}, which the standard errors cannot show (they see only the "
                   "held-out rows). The range table gives where each channel's readings can lie; scoring the whole "
                   "files at 05 Assay gives the actual figure.")


def concentration_sentence(summary: TrafficSummary) -> str | None:
    """A warning when the estimate rests on a few flows; None when the weight is spread out.

    It names the heavily repeated flows of the cleaned files that missed the held-out rows (their verdicts are
    unknown, see :func:`heavy_flow_bounds`) and the heaviest held-out row when one row's verdict can move a reading
    by :data:`SWING_LIMIT` or more.
    """
    if not summary.concentrated:
        return None
    parts = []
    if summary.heavy_concentrated:
        parts.append(_heavy_text(summary))
    if summary.rows_concentrated:
        parts.append(_heaviest_row_text(summary))
    if not summary.heavy:
        parts.append("Whether heavily repeated flows land among the held-out rows is chance, and this run does not "
                     "record those that did not, so these estimates are rough.")
    elif not summary.heavy_concentrated:
        parts.append("The estimates are rough.")
    return " ".join(parts)


@dataclass
class TrafficEvaluation:
    """Readings of one fitted channel weighted to the recorded traffic its held-out rows stand for (an estimate).

    Attributes:
        key: channel key.
        metrics: as :func:`classification_metrics` (same keys), weighted (:func:`weighted_classification_metrics`).
        per_class: see :func:`weighted_per_class_metrics` (``flows`` in place of ``support``).
        confusion: estimated recorded flows (float64), rows = true class, columns = predicted class.
        confusion_norm: ``confusion`` with each row divided by its total (row share).
        flows: the sum of the weights (recorded flows the held-out rows stand for).
        flows_by_class: summed weight per true class, in class-code order.
        errors: rough standard errors of the weighted ``accuracy`` and ``balanced_accuracy``
            (:func:`weighted_standard_errors`), from the spread among the held-out rows only.
        bounds: ``(low, high)`` of ``accuracy`` and ``balanced_accuracy`` over the recorded traffic, given that no
            held-out verdict covers the heavily repeated flows that missed the held-out rows
            (:func:`heavy_flow_bounds`); empty when the run does not record its heavy-flow account.
        classes: class names in code order.
        n_test: number of held-out rows behind these readings.
        seconds: time taken to compute them.
    """

    key: str
    metrics: dict[str, float]
    per_class: pd.DataFrame
    confusion: np.ndarray
    confusion_norm: np.ndarray
    flows: float
    flows_by_class: dict[str, float]
    errors: dict[str, float] = field(default_factory=dict)
    classes: tuple[str, ...] = ()
    n_test: int = 0
    seconds: float = 0.0
    bounds: dict[str, tuple[float, float]] = field(default_factory=dict)

    @property
    def label(self) -> str:
        """Badge and name of the channel, e.g. ``"CH2 XGBoost"``."""
        return channel_label(self.key)


def traffic_channel(run: "TrainingRun", key: str, weights: npt.ArrayLike | None = None) -> TrafficEvaluation:
    """The recorded-traffic readings of channel ``key`` from its stored test-set predictions (nothing is fitted).

    ``weights`` defaults to :func:`traffic_weights`. Raises ``KeyError`` for a channel the run does not hold and
    ``ValueError`` for a channel that was not fitted or a run without weights.
    """
    started = time.perf_counter()
    result = run.channels[key]
    if getattr(result, "status", "ok") != "ok" or result.estimator is None:
        raise ValueError(f"{channel_label(key)} was not fitted, so it has no readings.")
    if weights is None:
        basis = _basis(run)
        weights = basis.weights
        if weights is None:
            raise ValueError(basis.reason or "No recorded-traffic weights.")
    data = run.data
    classes = tuple(str(c) for c in data.classes)
    k = len(classes)
    y_test = _codes(data.y_test)
    w = _weights(weights, len(y_test))
    proba, y_pred = _channel_scores(result, np.asarray(data.X_test), k)
    matrix = weighted_confusion(y_test, y_pred, k, w)
    return TrafficEvaluation(
        key=key,
        metrics=weighted_classification_metrics(y_test, y_pred, proba, k, w),
        per_class=weighted_per_class_metrics(y_test, y_pred, proba, classes, w),
        confusion=matrix,
        confusion_norm=normalise_rows(matrix),
        flows=float(w.sum()),
        flows_by_class={name: float(w[y_test == i].sum()) for i, name in enumerate(classes)},
        errors=weighted_standard_errors(y_test, y_pred, w),
        classes=classes,
        n_test=int(len(y_test)),
        seconds=time.perf_counter() - started,
        bounds=heavy_flow_bounds(run, y_pred, w) or {},
    )


def traffic_readings(run: "TrainingRun") -> dict[str, TrafficEvaluation] | None:
    """Recorded-traffic readings of every fitted channel of ``run`` (channel order), or None when unavailable.

    Computed once per channel from the stored test-set probabilities and kept on the run (``run.traffic_evaluations``)
    like :func:`evaluate_run`'s readings, so later calls from any page or session return them as they are (the
    weights behind them are kept on the run too, so a later call computes nothing).
    :func:`traffic_unavailable_reason` says why None was returned. Thread-safe.
    """
    weights = _basis(run).weights
    if weights is None:
        return None
    keys = _ordered(list(run.ok_channels()))
    kept = getattr(run, TRAFFIC_ATTR, None)
    if isinstance(kept, dict) and all(key in kept for key in keys):
        return {key: kept[key] for key in keys}
    with _EVAL_LOCK:
        cache = getattr(run, TRAFFIC_ATTR, None)
        if not isinstance(cache, dict):
            cache = {}
            setattr(run, TRAFFIC_ATTR, cache)
        keys = _ordered(list(run.ok_channels()))
        for key in keys:
            if key not in cache:
                cache[key] = traffic_channel(run, key, weights)
        return {key: cache[key] for key in keys if key in cache}


def cached_traffic_readings(run: "TrainingRun") -> dict[str, TrafficEvaluation] | None:
    """The recorded-traffic readings already kept on ``run``, or None when none were computed yet."""
    cache = getattr(run, TRAFFIC_ATTR, None)
    return dict(cache) if isinstance(cache, dict) and cache else None


#: Standard-error columns of the recorded-traffic leaderboard: (key of ``TrafficEvaluation.errors``, column title).
TRAFFIC_ERROR_COLUMNS: tuple[tuple[str, str], ...] = (("balanced_accuracy", "Balanced accuracy s.e."),
                                                      ("accuracy", "Accuracy s.e."))


def traffic_leaderboard(traffic: Mapping[str, TrafficEvaluation], run: "TrainingRun",
                        evals: Mapping[str, ChannelEvaluation] | None = None) -> pd.DataFrame:
    """:func:`leaderboard` over the recorded-traffic readings, best weighted balanced accuracy first.

    The columns are those of :func:`leaderboard` with the two standard errors of :data:`TRAFFIC_ERROR_COLUMNS`
    after the scores. The scores and ``Gap to best`` are the weighted ones; fit time, rows used and training rows
    come from the run, and ``Flows/s`` and ``Single-flow ms`` from ``evals`` (the distinct-flow readings) when given,
    else NaN (scoring speed does not depend on how the rows are counted).
    """
    scores = score_columns(run.request.mode)
    rows: list[dict[str, Any]] = []
    for key in _ordered(list(traffic)):
        reading = traffic[key]
        result = run.channels.get(key)
        timing = (evals or {}).get(key)
        row: dict[str, Any] = {"key": key, "Channel": channel_label(key)}
        for metric, title in scores:
            row[title] = float(reading.metrics.get(metric, float("nan")))
        for metric, title in TRAFFIC_ERROR_COLUMNS:
            row[title] = float(reading.errors.get(metric, float("nan")))
        row["Fit s"] = float(getattr(result, "fit_seconds", float("nan")))
        row["Flows/s"] = float(timing.throughput_fps) if timing is not None else float("nan")
        row["Single-flow ms"] = float(timing.single_flow_ms) if timing is not None else float("nan")
        row["Rows used"] = int(getattr(result, "rows_used", 0) or 0)
        row["Training rows"] = int(getattr(result, "rows_available", 0) or 0)
        rows.append(row)
    columns = ["key", "Channel", *[title for _, title in scores], *[title for _, title in TRAFFIC_ERROR_COLUMNS],
               *LEADERBOARD_TAIL]
    frame = pd.DataFrame(rows, columns=columns)
    if frame.empty:
        return frame
    frame = frame.sort_values("Balanced accuracy", ascending=False, kind="stable", na_position="last")
    frame["Gap to best"] = float(frame["Balanced accuracy"].max()) - frame["Balanced accuracy"]
    return frame.reset_index(drop=True)[columns]


def traffic_metric_columns(mode: str) -> list[tuple[str, str]]:
    """(metric key, export column) pairs of the recorded-traffic readings for ``mode``, e.g.
    ``("balanced_accuracy", "traffic_balanced_accuracy")``, in leaderboard order."""
    return [(metric, f"{TRAFFIC_PREFIX}{metric}") for metric, _ in score_columns(mode)]


#: Export columns of the standard errors: (key of ``TrafficEvaluation.errors``, column).
TRAFFIC_ERROR_EXPORTS: tuple[tuple[str, str], ...] = (
    ("balanced_accuracy", f"{TRAFFIC_PREFIX}balanced_accuracy_se"), ("accuracy", f"{TRAFFIC_PREFIX}accuracy_se"))
#: Export columns of the heavy-flow ranges: (key of ``TrafficEvaluation.bounds``, low column, high column).
TRAFFIC_BOUND_EXPORTS: tuple[tuple[str, str, str], ...] = (
    ("balanced_accuracy", f"{TRAFFIC_PREFIX}balanced_accuracy_low", f"{TRAFFIC_PREFIX}balanced_accuracy_high"),
    ("accuracy", f"{TRAFFIC_PREFIX}accuracy_low", f"{TRAFFIC_PREFIX}accuracy_high"))


def bound_values(bounds: Mapping[str, Any] | None) -> dict[str, float]:
    """The heavy-flow ranges as export columns (:data:`TRAFFIC_BOUND_EXPORTS`), NaN where a range is missing."""
    out: dict[str, float] = {}
    for metric, low, high in TRAFFIC_BOUND_EXPORTS:
        pair = (bounds or {}).get(metric)
        out[low], out[high] = (float(pair[0]), float(pair[1])) if pair is not None else (float("nan"), float("nan"))
    return out


def traffic_columns(traffic: Mapping[str, TrafficEvaluation], mode: str) -> pd.DataFrame:
    """One row per channel: ``key``, :data:`TRAFFIC_FLOWS_COLUMN`, a ``traffic_<metric>`` column per score of
    :func:`score_columns`, the two standard errors (``traffic_balanced_accuracy_se``, ``traffic_accuracy_se``) and
    the heavy-flow ranges of :data:`TRAFFIC_BOUND_EXPORTS` (NaN when the run does not record them), channel order,
    for joining onto a leaderboard export."""
    pairs = traffic_metric_columns(mode)
    rows = []
    for key in _ordered(list(traffic)):
        reading = traffic[key]
        row: dict[str, Any] = {"key": key, TRAFFIC_FLOWS_COLUMN: float(reading.flows)}
        for metric, column in pairs:
            row[column] = float(reading.metrics.get(metric, float("nan")))
        for metric, column in TRAFFIC_ERROR_EXPORTS:
            row[column] = float(reading.errors.get(metric, float("nan")))
        row.update(bound_values(reading.bounds))
        rows.append(row)
    bound_columns = [column for _, low, high in TRAFFIC_BOUND_EXPORTS for column in (low, high)]
    return pd.DataFrame(rows, columns=["key", TRAFFIC_FLOWS_COLUMN, *[c for _, c in pairs],
                                       *[c for _, c in TRAFFIC_ERROR_EXPORTS], *bound_columns])


def traffic_per_class_frame(traffic: Mapping[str, TrafficEvaluation]) -> pd.DataFrame:
    """The per-class recorded-traffic readings of every channel stacked into one table (channel, then class order):
    ``key``, ``Channel`` and :data:`TRAFFIC_PER_CLASS_COLUMNS`."""
    parts = []
    for key in _ordered(list(traffic)):
        table = traffic[key].per_class.copy()
        table.insert(0, "Channel", channel_label(key))
        table.insert(0, "key", key)
        parts.append(table)
    if not parts:
        return pd.DataFrame(columns=["key", "Channel", *TRAFFIC_PER_CLASS_COLUMNS])
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
    "CHANNEL_ORDER", "CVPlan", "ChannelEvaluation", "EvaluationTask", "TRAFFIC_FLOWS_COLUMN", "TRAFFIC_PREFIX",
    "TaskSnapshot", "TrafficEvaluation", "TrafficSummary", "cached_evaluations", "cached_traffic_readings",
    "channel_label", "class_curves", "classification_metrics", "concentration_sentence", "cross_validate_run",
    "cv_fold_frame", "downsample_curve", "estimate_cv_seconds", "evaluate_channel", "evaluate_run", "flows_text",
    "forget_task", "get_task", "has_test_rows", "held_out_counts", "held_out_repeats", "leaderboard",
    "native_importance", "normalise_rows", "per_class_frame",
    "per_class_metrics", "permutation_importance_for", "permutation_plan", "plan_cross_validation",
    "quick_metrics", "remember_cross_validation", "remember_permutation", "sampling_fractions", "score_columns",
    "repeats_sentence", "single_flow_latency_ms", "stored_cross_validation", "stored_permutations",
    "traffic_channel", "traffic_columns", "traffic_leaderboard", "traffic_metric_columns", "traffic_per_class_frame",
    "traffic_readings", "traffic_sentence", "traffic_summary", "traffic_unavailable_reason", "traffic_weights",
    "weighted_classification_metrics", "weighted_confusion", "weighted_per_class_metrics",
    "weighted_standard_errors",
]
