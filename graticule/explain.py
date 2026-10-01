"""04 Probe's core: one flow read by every fitted channel, and which features moved a channel's reading.

Scoring. :func:`score_flow` asks each chosen channel for its class probabilities on a single flow and combines
them into the consensus (:func:`graticule.models.verdict.combine`): equal weight per channel, the most probable
class, and how many channels picked that class on their own. Probabilities are tidied exactly as the trainer tidies
the held-out readings, and a held-out flow named by its position takes the very readings the run stored for it (a
neural net or logistic regression scoring one row alone can differ from its block-scored reading in the last
float digits). Nothing is fitted.

Two ways to say which features mattered:

* **XGBoost exact contributions** (:func:`xgboost_contributions`, CH2 only). XGBoost can split the raw score a
  flow receives (the margin, on the log-odds scale) among the features along the branches the flow follows
  through each tree, plus a constant bias. The parts add up exactly to the margin, so nothing is estimated. For a
  binary run the margin is the log-odds of "Attack", and asking for the other class flips every sign; in a
  multi-class run each class has its own margin, and a softmax over them gives the probabilities.
* **Reference swap** (:func:`reference_swap`, any channel, approximate). For every feature in turn, the flow is
  copied once per background flow with only that feature replaced by the background flow's value, and the channel
  scores all copies in ONE batched call. A feature's contribution is the probability of the class for the flow
  itself minus the mean probability over its swapped copies: positive means the flow's own value pushed the
  channel towards the class. Each feature is swapped alone, so an effect that needs two features to move together
  (an interaction) is not seen, and features that carry the same information can share or hide each other's
  credit. The background is the run's reference sample (training rows), or, for a channel set loaded from disk
  without its training rows, synthetic vectors read off the saved training quantiles
  (:func:`background_from_quantiles`), never held-out rows.

:func:`typical_flow` gives a "typical" flow to start from: the per-feature median of one class's rows in the
reference sample, or the overall training median read off the quantiles when no rows are in memory.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd

from graticule.data import sampling
from graticule.models.train import QUANTILE_LEVELS, REFERENCE_ROWS, _tidy_proba
from graticule.models.verdict import Consensus, alert_flags, combine
from graticule.models.verdict import attack_probability as attack_probability_of
from graticule.persist import deterministic
from graticule.schema import is_normal_traffic
from graticule.theme import CHANNEL_BY_KEY

if TYPE_CHECKING:
    from graticule.models.train import TrainingRun

Method = Literal["xgboost_exact", "reference_swap"]
#: Units of each method's contributions.
UNITS_LOG_ODDS = "log-odds"
UNITS_PROBABILITY = "probability points"
#: Background flows a reference swap uses at most.
DEFAULT_BACKGROUND = 32
#: Synthetic background vectors drawn from the quantiles of a run that holds no training rows.
QUANTILE_BACKGROUND_ROWS = 256
#: Features shown in a contribution chart.
TOP_FEATURES = 12
#: Row of the quantile table holding the median (levels 0, 1, ..., 100 %).
MEDIAN_ROW = int(np.argmin(np.abs(QUANTILE_LEVELS - 0.5)))


# --------------------------------------------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True, eq=False)
class Explanation:
    """Why one channel read one flow the way it did, feature by feature.

    Attributes:
        method: ``"xgboost_exact"`` or ``"reference_swap"``.
        channel: channel key (``"xgboost"`` ...).
        class_name: the class whose score is explained (for a binary XGBoost explanation, "Attack" unless the other
            class was asked for).
        units: ``"log-odds"`` (exact) or ``"probability points"`` (reference swap; 0.10 = ten points).
        table: one row per feature: ``feature``, ``value`` (the flow's own value) and ``contribution`` (positive
            pushes towards ``class_name``), largest absolute contribution first.
        bias: exact method only: the constant part of the margin (the same for every flow); None otherwise.
        output: what is explained: the margin in log-odds (exact; equals ``bias`` plus every contribution) or the
            channel's probability of ``class_name`` for this flow (reference swap).
        background_rows: reference swap only: how many background flows each feature was swapped with.
    """

    method: Method
    channel: str
    class_name: str
    units: str
    table: pd.DataFrame
    bias: float | None = None
    output: float = float("nan")
    background_rows: int = 0

    @property
    def approximate(self) -> bool:
        """True for the reference swap (an estimate that ignores interactions), False for exact contributions."""
        return self.method == "reference_swap"

    def top(self, n: int = TOP_FEATURES) -> pd.DataFrame:
        """The ``n`` features with the largest absolute contribution (a copy, largest first)."""
        return self.table.head(max(int(n), 0)).copy()

    def rest(self, n: int = TOP_FEATURES) -> float:
        """Sum of the contributions of the features outside the top ``n``."""
        return float(self.table["contribution"].iloc[max(int(n), 0):].sum())

    def total(self) -> float:
        """Sum of every feature's contribution (plus the bias, for the exact method)."""
        return float(self.table["contribution"].sum()) + (float(self.bias) if self.bias is not None else 0.0)


@dataclass(frozen=True, eq=False)
class FlowVerdict:
    """Every chosen channel's reading of one flow, and their consensus.

    Attributes:
        classes: class names in code order.
        channels: channel keys scored, in channel order.
        proba: channel key -> class probabilities of the flow (float32, length K, summing to 1).
        consensus: the combined reading (:class:`~graticule.models.verdict.Consensus`, one flow).
    """

    classes: tuple[str, ...]
    channels: tuple[str, ...]
    proba: dict[str, np.ndarray]
    consensus: Consensus

    def label_index(self, key: str) -> int:
        """Class code channel ``key`` reads (the most probable class)."""
        return int(np.argmax(self.proba[key]))

    def label(self, key: str) -> str:
        """Class name channel ``key`` reads."""
        return self.classes[self.label_index(key)]

    def probability(self, key: str, class_index: int | None = None) -> float:
        """Channel ``key``'s probability of ``class_index`` (default: of its own verdict)."""
        index = self.label_index(key) if class_index is None else int(class_index)
        return float(self.proba[key][index])

    @property
    def normal_index(self) -> int | None:
        """Code of the normal-traffic class (BENIGN or Normal), or None when the run has none."""
        return next((i for i, name in enumerate(self.classes) if is_normal_traffic(name)), None)

    def _attack_values(self, key: str | None) -> np.ndarray:
        """The shared attack probability (float32, one value) of channel ``key`` or (None) the consensus."""
        values = self.consensus.proba[0] if key is None else self.proba[key]
        return attack_probability_of(values, self.normal_index)

    def attack_probability(self, key: str | None = None) -> float:
        """Probability that the flow is an attack, for channel ``key`` or (None) the consensus.

        Two classes (binary runs): the probability of the attack class. More classes: one minus the probability
        of the normal class. NaN when the run has no normal class. Computed by
        :func:`graticule.models.verdict.attack_probability`, as at 05 Assay and 06 Sweep.
        """
        if self.normal_index is None:
            return float("nan")
        return float(self._attack_values(key)[0])

    def raises_alert(self, key: str | None, threshold: float) -> bool:
        """True when channel ``key`` (None: the consensus) raises a high-confidence alert on the flow: its verdict
        is an attack class and its attack probability is at least ``threshold`` (the rule 05 Assay and 06 Sweep
        use, :func:`graticule.models.verdict.alert_flags`)."""
        index = self.consensus_index if key is None else self.label_index(key)
        # Without a normal class every flow is read as an attack with certainty (as 05 Assay and 06 Sweep read it).
        return bool(alert_flags(self._attack_values(key), np.asarray([index]), self.normal_index, threshold)[0])

    def top_classes(self, key: str | None = None, n: int = 3) -> list[tuple[str, float]]:
        """The ``n`` most probable classes (name, probability) for channel ``key`` or (None) the consensus."""
        values = np.asarray(self.consensus.proba[0] if key is None else self.proba[key], dtype=np.float64)
        order = np.argsort(-values, kind="stable")[: max(int(n), 0)]
        return [(self.classes[int(i)], float(values[int(i)])) for i in order]

    @property
    def consensus_index(self) -> int:
        """Class code of the consensus."""
        return int(self.consensus.label_index[0])

    @property
    def consensus_label(self) -> str:
        """Class name of the consensus."""
        return self.classes[self.consensus_index]

    @property
    def consensus_probability(self) -> float:
        """The consensus probability of the consensus class."""
        return float(self.consensus.proba[0, self.consensus_index])

    @property
    def agreement(self) -> int:
        """How many channels read the consensus class on their own."""
        return int(self.consensus.agreement[0])


# --------------------------------------------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------------------------------------------
def channel_label(key: str) -> str:
    """Badge and name of a channel, e.g. ``"CH2 XGBoost"`` (the key itself when unknown)."""
    style = CHANNEL_BY_KEY.get(key)
    return style.label if style is not None else key


def _badge(key: str) -> str:
    """Badge of a channel, e.g. ``"CH2"`` (the key itself when unknown)."""
    style = CHANNEL_BY_KEY.get(key)
    return style.badge if style is not None else key


def _as_row(x_row: Any, n_features: int) -> np.ndarray:
    """One flow as a float32 (1 x n_features) matrix; raises ``ValueError`` for the wrong width."""
    row = np.asarray(x_row, dtype=np.float32).reshape(1, -1)
    if row.shape[1] != int(n_features):
        raise ValueError(f"The flow has {row.shape[1]} values, but the channel reads {int(n_features)} features.")
    return row


def row_digest(x_row: Any) -> str:
    """A short fingerprint of a flow's values (float32; every NaN alike, -0.0 as +0.0), for caching readings."""
    values = np.array(x_row, dtype=np.float32, copy=True).reshape(-1)
    values[values == 0] = 0.0
    values[np.isnan(values)] = np.nan
    return hashlib.sha1(np.ascontiguousarray(values, dtype="<f4").tobytes()).hexdigest()[:16]


def _split_pipeline(estimator: Any) -> tuple[Any, Any | None]:
    """(final model, the steps before it or None) of a pipeline; (estimator, None) for a bare model."""
    steps = getattr(estimator, "steps", None)
    if steps:
        return steps[-1][1], (estimator[:-1] if len(steps) > 1 else None)
    return estimator, None


def iteration_window(model: Any) -> tuple[int, int]:
    """The boosting rounds an XGBoost classifier predicts with: up to its best round when early stopping recorded
    one, else every round (``(0, 0)``), exactly as its own ``predict_proba`` chooses."""
    try:
        return 0, int(model.best_iteration) + 1
    except AttributeError:
        return 0, 0


def _class_column(estimator: Any, class_index: int) -> int | None:
    """Column of ``predict_proba`` holding class code ``class_index`` (None when the model never saw it)."""
    classes = getattr(estimator, "classes_", None)
    if classes is None:
        return int(class_index)
    matches = np.flatnonzero(np.asarray(classes).astype(np.int64) == int(class_index))
    return int(matches[0]) if matches.size else None


def _contribution_table(names: Sequence[str], values: np.ndarray, contributions: np.ndarray) -> pd.DataFrame:
    """Feature, value and contribution, largest absolute contribution first (ties keep the feature order)."""
    frame = pd.DataFrame({
        "feature": pd.Series([str(n) for n in names], dtype="str"),
        "value": np.asarray(values, dtype=np.float64),
        "contribution": np.asarray(contributions, dtype=np.float64),
    })
    order = np.argsort(-np.abs(frame["contribution"].to_numpy()), kind="stable")
    return frame.iloc[order].reset_index(drop=True)


# --------------------------------------------------------------------------------------------------------------
# Explanations
# --------------------------------------------------------------------------------------------------------------
def xgboost_contributions(
    estimator: Any,
    x_row: np.ndarray,
    feature_names: Sequence[str],
    class_index: int,
    *,
    channel: str = "xgboost",
    class_name: str | None = None,
) -> Explanation:
    """Exact per-feature contributions of an XGBoost channel to one flow's margin (log-odds) for ``class_index``.

    The flow first goes through the pipeline's steps before the model (the sanitiser: float32, ±inf to missing),
    then the booster splits its margin among the features (``pred_contribs``), using the same boosting rounds as
    the channel's own predictions (:func:`iteration_window`). The bias is returned separately; contributions plus
    bias equal the margin (``output``). A binary model has one margin, that of class 1 ("Attack"); for class 0
    every sign is flipped. ``class_name`` labels the result (default: the class code as text). Raises
    ``TypeError`` for a channel that is not XGBoost and ``ValueError`` for a class the model does not have.
    """
    import xgboost

    model, prefix = _split_pipeline(estimator)
    if not hasattr(model, "get_booster"):
        raise TypeError("Exact contributions need an XGBoost channel (CH2); use the reference swap for the others.")
    names = [str(n) for n in feature_names]
    row = _as_row(x_row, len(names))
    ready = np.asarray(prefix.transform(row) if prefix is not None else row, dtype=np.float32)
    if ready.shape != row.shape:
        raise ValueError("The steps before the XGBoost model changed the number of features.")
    booster = model.get_booster()
    window = iteration_window(model)
    matrix = xgboost.DMatrix(ready, missing=np.nan)
    parts = np.asarray(booster.predict(matrix, pred_contribs=True, strict_shape=True, iteration_range=window),
                       dtype=np.float64)
    margins = np.asarray(booster.predict(matrix, output_margin=True, strict_shape=True, iteration_range=window),
                         dtype=np.float64)
    groups = parts.shape[1]
    index = int(class_index)
    if groups == 1:
        if index not in (0, 1):
            raise ValueError(f"A binary model has classes 0 and 1, not {index}.")
        sign = 1.0 if index == 1 else -1.0
        values, margin = sign * parts[0, 0], sign * float(margins[0, 0])
    else:
        if not 0 <= index < groups:
            raise ValueError(f"The model has {groups} classes; class {index} does not exist.")
        values, margin = parts[0, index], float(margins[0, index])
    return Explanation(
        method="xgboost_exact", channel=channel, class_name=class_name if class_name is not None else str(index),
        units=UNITS_LOG_ODDS, table=_contribution_table(names, row[0], values[:-1]), bias=float(values[-1]),
        output=margin,
    )


def reference_swap(
    estimator: Any,
    x_row: np.ndarray,
    background: np.ndarray,
    feature_names: Sequence[str],
    class_index: int,
    *,
    max_background: int = DEFAULT_BACKGROUND,
    seed: int = 0,
    channel: str = "",
    class_name: str | None = None,
) -> Explanation:
    """Approximate per-feature contributions of any channel to its probability of ``class_index`` for one flow.

    At most ``max_background`` background flows are used (a seeded draw when there are more). For each feature j
    and background flow b, a copy of the flow takes b's value for j; every copy that differs from the flow is
    scored, together with the flow itself, in ONE ``predict_proba`` call. The contribution of j is the mean over b
    of p(class | flow) - p(class | flow with j swapped): positive means the flow's own value pushed the channel
    towards the class. A copy equal to the flow (the same value, or both missing) contributes exactly 0 without
    being scored. Features are swapped one at a time, so interactions between features are not captured: read
    the result as approximate. Raises ``ValueError`` for an empty background or mismatched widths.
    """
    names = [str(n) for n in feature_names]
    n_features = len(names)
    row = _as_row(x_row, n_features)[0]
    pool = np.asarray(background, dtype=np.float32)
    if pool.ndim != 2 or pool.shape[1] != n_features:
        raise ValueError(f"The background must be a matrix with {n_features} columns.")
    if pool.shape[0] == 0:
        raise ValueError("The reference swap needs at least one background flow.")
    limit = max(int(max_background), 1)
    if pool.shape[0] > limit:
        rng = np.random.default_rng(int(seed))
        pool = pool[np.sort(rng.choice(pool.shape[0], size=limit, replace=False))]
    n_background = pool.shape[0]
    same = (pool == row[np.newaxis, :]) | (np.isnan(pool) & np.isnan(row)[np.newaxis, :])
    feature_at, background_at = np.nonzero(~same.T)  # every (feature, background flow) pair that changes the flow
    batch = np.repeat(row[np.newaxis, :], 1 + len(feature_at), axis=0)
    batch[1 + np.arange(len(feature_at)), feature_at] = pool[background_at, feature_at]
    column = _class_column(estimator, class_index)
    if column is None:
        scores = np.zeros(len(batch), dtype=np.float64)
    else:
        scores = np.asarray(estimator.predict_proba(batch), dtype=np.float64)[:, column]
    own = float(scores[0])
    drops = np.zeros((n_features, n_background), dtype=np.float64)
    drops[feature_at, background_at] = own - scores[1:]
    return Explanation(
        method="reference_swap", channel=channel,
        class_name=class_name if class_name is not None else str(int(class_index)), units=UNITS_PROBABILITY,
        table=_contribution_table(names, row, drops.mean(axis=1)), output=own, background_rows=int(n_background),
    )


def background_from_quantiles(quantiles: np.ndarray, n: int, seed: int) -> np.ndarray:
    """``n`` synthetic background vectors (float32, n x n_features) read off per-feature training quantiles.

    Each value follows its feature's training distribution (the quantile curve at an independent random level),
    so no vector is, or copies, a dataset row. Used for channel sets loaded from disk without their training rows;
    these are the same kind of vectors :func:`graticule.persist.restore_run` keeps as such a run's reference
    sample. A feature without finite quantiles gives NaN (missing) throughout.
    """
    from graticule.persist import quantile_vectors

    return quantile_vectors(np.asarray(quantiles, dtype=np.float32), int(n), int(seed))


def background_for(run: "TrainingRun") -> tuple[np.ndarray, str]:
    """The background a reference swap on ``run`` draws from, with a short description of it.

    A run holding its training rows (fitted here, or loaded with its rows rebuilt) uses its reference sample of
    training rows; a run loaded without them uses :func:`background_from_quantiles` (seeded with the run's seed).
    """
    if len(run.data.y_train) > 0 and len(run.reference_sample) > 0:
        rows = len(run.reference_sample)
        return (np.asarray(run.reference_sample, dtype=np.float32),
                f"training rows from the run's reference sample of {rows:,}")
    vectors = background_from_quantiles(run.feature_quantiles, QUANTILE_BACKGROUND_ROWS, int(run.request.seed))
    return vectors, ("synthetic flows read off the saved training quantiles (the run was loaded without its "
                     "training rows)")


# --------------------------------------------------------------------------------------------------------------
# Scoring one flow, and typical flows
# --------------------------------------------------------------------------------------------------------------
def _stored_reading(run: "TrainingRun", key: str, row: np.ndarray, test_index: int | None) -> np.ndarray | None:
    """The probabilities channel ``key`` recorded for held-out row ``test_index`` when the run was scored, or None
    when there is no such row, the row holds other values than ``row``, or the channel kept no readings."""
    if test_index is None:
        return None
    X_test = run.data.X_test
    index = int(test_index)
    stored = run.channels[key].proba
    if not 0 <= index < len(X_test) or stored is None or len(stored) != len(X_test):
        return None
    if not np.array_equal(np.asarray(X_test[index], dtype=np.float32), row[0], equal_nan=True):
        return None
    return np.asarray(stored[index], dtype=np.float32).copy()


def score_flow(run: "TrainingRun", x_row: np.ndarray, channels: Sequence[str], *,
               test_index: int | None = None) -> FlowVerdict:
    """Score one flow with the chosen fitted channels of ``run`` and combine them into the consensus.

    Channels that are not fitted in the run are left out; ``ValueError`` when none remains or the flow has the
    wrong number of values. Probabilities are tidied as the trainer tidies the held-out readings (float32, in
    class-code order, rows summing to 1). When ``test_index`` names the held-out row the flow came from (and the
    values match it), each channel's reading is the one stored for that row when the run was scored, so the flow
    gets exactly the verdict 03 Measure counted. Any other flow is scored now. Nothing is fitted.
    """
    wanted = {str(k) for k in channels}
    keys = tuple(k for k in run.ok_channels() if k in wanted)
    if not keys:
        raise ValueError("Choose at least one fitted channel to read the flow.")
    classes = tuple(str(c) for c in run.data.classes)
    row = _as_row(x_row, len(run.data.feature_names))
    proba: dict[str, np.ndarray] = {}
    for key in keys:
        stored = _stored_reading(run, key, row, test_index)
        if stored is not None:
            proba[key] = stored
            continue
        estimator = run.channels[key].estimator
        # One flow: a forest sums its trees on one thread (reproducible to the last bit, and much quicker than
        # starting a pool of threads for a single row).
        with deterministic(estimator):
            raw = estimator.predict_proba(row)
        proba[key] = _tidy_proba(raw, getattr(estimator, "classes_", None), len(classes))[0]
    return FlowVerdict(classes=classes, channels=keys, proba=proba, consensus=combine(proba))


def _reference_labels(run: "TrainingRun") -> np.ndarray | None:
    """Class codes of the rows in ``run.reference_sample``, or None when they cannot be recovered.

    The trainer draws the reference rows with a seeded sampler over the training classes; the same draw is made
    again here and accepted only when it gives exactly the stored rows.
    """
    y_train = np.asarray(run.data.y_train)
    reference = np.asarray(run.reference_sample, dtype=np.float32)
    if len(y_train) == 0 or len(reference) == 0:
        return None
    n_ref = min(REFERENCE_ROWS, len(y_train))
    positions, _ = sampling.sample_positions(y_train, n_ref, int(run.request.seed))
    positions = np.asarray(positions, dtype=np.int64)
    if len(positions) != len(reference):
        return None
    if not np.array_equal(np.asarray(run.data.X_train, dtype=np.float32)[positions], reference, equal_nan=True):
        return None
    return y_train[positions].astype(np.int64)


def _column_medians(X: np.ndarray) -> np.ndarray:
    """Per-column medians ignoring missing and infinite values (float32; NaN for a column with no finite value)."""
    frame = pd.DataFrame(np.where(np.isinf(X), np.nan, np.asarray(X, dtype=np.float64)))
    return frame.median(axis=0, skipna=True).to_numpy(dtype=np.float32)


def typical_flow(run: "TrainingRun", class_index: int | None = None) -> tuple[np.ndarray, str]:
    """A typical flow (float32 vector) and where it comes from.

    With training rows in memory: the per-feature median of the reference-sample rows of class ``class_index``
    (of every reference row when None; when the classes of the reference rows cannot be recovered, of that class's
    training rows). Without them (a set loaded from disk alone): the overall training median, read off the saved
    quantiles, whatever ``class_index`` says. The vector is a summary, not a dataset row.
    """
    classes = [str(c) for c in run.data.classes]
    if len(run.data.y_train) == 0 or len(run.reference_sample) == 0:
        median = np.asarray(run.feature_quantiles, dtype=np.float32)[MEDIAN_ROW].copy()
        return median, "the overall training median of every feature (read off the saved quantiles)"
    reference = np.asarray(run.reference_sample, dtype=np.float32)
    if class_index is None:
        return _column_medians(reference), f"the median of every feature over {len(reference):,} reference rows"
    index = int(class_index)
    if not 0 <= index < len(classes):
        raise ValueError(f"The run has {len(classes)} classes; class {index} does not exist.")
    labels = _reference_labels(run)
    if labels is not None and np.any(labels == index):
        rows = reference[labels == index]
        where = f"the median of every feature over the {len(rows):,} {classes[index]} rows of the reference sample"
        return _column_medians(rows), where
    mask = np.asarray(run.data.y_train) == index
    rows = np.asarray(run.data.X_train, dtype=np.float32)[mask]
    if len(rows) == 0:
        raise ValueError(f"The run has no training rows of {classes[index]}.")
    return _column_medians(rows), f"the median of every feature over the {len(rows):,} {classes[index]} training rows"


# --------------------------------------------------------------------------------------------------------------
# Plain words
# --------------------------------------------------------------------------------------------------------------
def _names_text(names: Sequence[str]) -> str:
    """``"A"``, ``"A and B"``, ``"A, B and C"``."""
    names = [str(n) for n in names]
    if len(names) <= 1:
        return "".join(names)
    return ", ".join(names[:-1]) + " and " + names[-1]


def reading_sentence(explanation: Explanation, *, towards: str, sign: int = 1, n: int = 2) -> str:
    """One plain sentence naming the features that pushed the channel hardest towards ``towards``.

    ``sign`` is +1 when ``towards`` is the explained class (features with positive contributions) and -1 when it
    is the other side (negative contributions, e.g. "Normal" for a binary explanation of "Attack"). Example:
    "Flow IAT Max and Init_Win_bytes_backward pushed CH2 towards DoS Hulk the most."
    """
    badge = _badge(explanation.channel)
    signed = explanation.table.assign(pushed=explanation.table["contribution"] * (1 if sign >= 0 else -1))
    helping = signed[signed["pushed"] > 0].sort_values("pushed", ascending=False, kind="stable")
    against = signed[signed["pushed"] < 0].sort_values("pushed", ascending=True, kind="stable")
    if helping.empty and against.empty:
        return f"No single feature moved {badge}'s reading: changing any one of them alone leaves it where it is."
    if helping.empty:
        lead = _names_text(list(against["feature"].head(max(int(n), 1))))
        return f"No feature pushed {badge} towards {towards}; {lead} pulled it away the most."
    lead = _names_text(list(helping["feature"].head(max(int(n), 1))))
    return f"{lead} pushed {badge} towards {towards} the most."


__all__ = [
    "DEFAULT_BACKGROUND", "Explanation", "FlowVerdict", "Method", "TOP_FEATURES", "UNITS_LOG_ODDS",
    "UNITS_PROBABILITY", "background_for", "background_from_quantiles", "channel_label", "iteration_window",
    "reading_sentence", "reference_swap", "row_digest", "score_flow", "typical_flow", "xgboost_contributions",
]
