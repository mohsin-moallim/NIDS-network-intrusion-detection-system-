"""02 Fit: from a prepared sample to training/test matrices, and from those to fitted channels.

Order of work (each step looks only at what it is allowed to see, so nothing leaks from the test rows):

1. Target for the mode (:func:`graticule.data.sampling.target_for_mode`): binary maps BENIGN to "Normal" and every
   other label to "Attack"; multi-class keeps the classes and leaves out those below the minimum class count. In both
   modes a class below the hard floor of 10 rows is left out.
2. Feature columns (:func:`graticule.features.select_features`), with degenerate columns detected on the rows that
   take part.
3. Model-space de-duplication: rows identical over the chosen columns and the target are reduced to one, and rows
   identical over the chosen columns but with different targets (conflicts) are counted and handled by the conflict
   policy. Dropping columns (the port, say) can make distinct flows identical, so this is done again here even
   though 01 Sample removed exact duplicates over all 77 features. Classes that fall below their minimum because
   of it are left out and reported.
4. Stratified split (:func:`graticule.data.sampling.stratified_split`).
5. Top-K only: the ranking is computed on the TRAINING rows alone (de-duplication in step 3 used every candidate
   column), and the report counts test rows whose K-column vector also occurs among the training rows.

Then :func:`train_all` fits each requested channel in the fixed channel order. :func:`fit_model` is the only place
any estimator is fitted, the Top-K ranking model included (under the key :data:`RANKING_KEY`); it counts its calls
in :data:`FIT_CALLS` (tests spy on it). A channel that fails is recorded as failed with its error and the run
carries on with the next one.

Cancelling. The cancel token is checked between channels, forest chunks, boosting rounds (also those of the Top-K
ranking), MLP epochs, logistic-regression iterations and batches of test rows being scored, so a cancel takes
effect within one such step. The steps that cannot be interrupted are the kernel SVM's own fit and the matrix
copies.

Warnings. Each channel records the warnings raised on its own thread (:func:`graticule.models.jobs.thread_warnings`,
which leaves the process's warning state alone) and keeps them as notes, apart from a few that say nothing about
the readings. Whether a model stopped at its iteration limit is read from the fitted model, so that note never
depends on warning filters.
"""

from __future__ import annotations

import secrets
import threading
import time
import traceback
import warnings
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.exceptions import ConvergenceWarning
from sklearn.frozen import FrozenEstimator
from sklearn.pipeline import Pipeline
from xgboost.callback import TrainingCallback

from graticule.data import sampling
from graticule.data.clean import (
    conflict_keep_mask,
    conflict_mask,
    copies_per_kept_row,
    find_degenerate_columns,
    first_occurrences,
    hashes_with_labels,
    row_hashes,
)
from graticule.data.prepare import DataRequest, PreparedDataset
from graticule.data.sampling import HARD_FLOOR, SingleClassError
from graticule.evaluate import quick_metrics
from graticule.features import (
    DEFAULT_K,
    FEATURE_MODES,
    RANK_ROUNDS,
    FeatureChoice,
    FeatureMode,
    rank_features,
    select_features,
)
from graticule.models import zoo
from graticule.models.jobs import (
    STAGE_KEY,
    CancelToken,
    ProgressSink,
    TrainingCancelled,
    find_hooks,
    register_hooks,
    release_hooks,
    thread_warnings,
)
from graticule.models.zoo import MODEL_KEYS, BuildContext, ObservableMLP, Profile, WholeSliceSplit
from graticule.schema import DESTINATION_PORT, FEATURES, LABEL

Mode = Literal["binary", "multiclass"]
ConflictPolicy = Literal["keep", "majority", "drop"]
#: ``"not_saved"`` marks a channel of a run loaded from disk that was fitted but is never kept in saved sets
#: (see :data:`graticule.persist.UNSAVED_CHANNELS`).
ChannelStatus = Literal["ok", "failed", "cancelled", "skipped", "not_saved"]
RunOrigin = Literal["fitted", "loaded"]

#: Calls of :func:`fit_model` per key: one per channel fit, and one under :data:`RANKING_KEY` per Top-K ranking.
#: Tests read and reset it.
FIT_CALLS: Counter[str] = Counter()
_FIT_CALLS_LOCK = threading.Lock()
#: The :data:`FIT_CALLS` key of the small XGBoost model that ranks features for Top-K
#: (:func:`graticule.features.rank_features`, fitted through :func:`fit_model` like every other model).
RANKING_KEY = "topk_ranking"

#: Training rows kept for explanations (rare-aware draw).
REFERENCE_ROWS = 2_000
#: Quantile levels stored per feature (0 %, 1 %, ..., 100 %).
QUANTILE_LEVELS: np.ndarray = np.linspace(0.0, 1.0, 101)
#: Share of the training rows XGBoost holds back to decide when to stop adding trees.
XGB_EVAL_SHARE = 0.10
#: Largest number of training rows used to calibrate the SVM's probabilities.
CALIBRATION_ROWS = 5_000
#: Calibration rows each class gives at least (but never more than half of its training rows, so the SVM still
#: sees the other half): a rare class's sigmoid then rests on a few dozen rows rather than on one or two.
CALIBRATION_FLOOR = 20
#: Rows per block when test rows are scored (:func:`score_in_blocks`), with a cancel check between blocks. The
#: blocks are fixed, so the fit and a run restored from disk call each model on exactly the same row ranges: a
#: neural net's matrix products can round differently for another batch size, so equal blocks give equal readings.
SCORE_BLOCK = 5_000
#: Warnings left out of a channel's notes because they say nothing about its readings. The frozen SVM cannot take
#: sample weights, so scikit-learn warns that the weights go to the calibration sigmoid only: that is the intent.
QUIET_WARNINGS: tuple[str, ...] = ("does not appear to accept sample_weight",)
#: Rows used to rank features for top-K, by mode. Ranking fits one small booster per class and round, so its cost
#: grows with the number of classes. Measured on a 200,000-row sample of all eight recorded files (12 classes after
#: the minimum count of 50; 8 logical CPUs): 50,000 rows take 19.1 s multi-class (20,000 rows: 6.8 s) and 2.2 s
#: binary. Both stay under the 30 s limit set for this stage, so both modes keep 50,000 rows.
TOPK_MAX_ROWS: dict[str, int] = {"binary": 50_000, "multiclass": 50_000}


# --------------------------------------------------------------------------------------------------------------
# Requests and results
# --------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class TrainRequest:
    """Everything 02 Fit needs to know besides the sample itself.

    ``channels`` is normalised to the fixed channel order without repeats. ``svm_cap`` is the most rows the SVM
    trains on; ``profile="test"`` shrinks every model for fast tests.
    """

    mode: Mode = "binary"
    feature_mode: FeatureMode = "curated"
    top_k: int = DEFAULT_K
    include_port: bool = False
    channels: tuple[str, ...] = MODEL_KEYS
    balanced: bool = True
    min_class_count: int = 50
    svm_cap: int = 20_000
    test_share: float = 0.25
    seed: int = 42
    conflict_policy: ConflictPolicy = "keep"
    profile: Profile = "full"

    def __post_init__(self) -> None:
        """Validate the options and put ``channels`` in the fixed order."""
        if self.mode not in ("binary", "multiclass"):
            raise ValueError(f"Unknown mode {self.mode!r}; choose 'binary' or 'multiclass'.")
        if self.feature_mode not in FEATURE_MODES:
            raise ValueError(f"Unknown feature mode {self.feature_mode!r}.")
        if self.conflict_policy not in ("keep", "majority", "drop"):
            raise ValueError(f"Unknown conflict policy {self.conflict_policy!r}.")
        if self.profile not in zoo.PROFILES:
            raise ValueError(f"Unknown profile {self.profile!r}.")
        if isinstance(self.top_k, bool) or int(self.top_k) < 1:
            raise ValueError("Top-K needs K of at least 1.")
        if not 0.0 < float(self.test_share) < 1.0:
            raise ValueError("The test share must be between 0 and 1.")
        if int(self.svm_cap) < 1:
            raise ValueError("The SVM row cap must be at least 1.")
        if int(self.min_class_count) < 1:
            raise ValueError("The minimum class count must be at least 1.")
        requested = [str(c) for c in self.channels]
        unknown = [c for c in requested if c not in MODEL_KEYS]
        if unknown:
            raise ValueError(f"Unknown channel(s): {', '.join(unknown)}.")
        ordered = tuple(k for k in MODEL_KEYS if k in requested)
        if not ordered:
            raise ValueError("Choose at least one channel to fit.")
        object.__setattr__(self, "channels", ordered)

    def build_context(self, n_classes: int) -> BuildContext:
        """The :class:`~graticule.models.zoo.BuildContext` for a target with ``n_classes`` classes."""
        return BuildContext(n_classes=int(n_classes), seed=int(self.seed), profile=self.profile,
                            svm_cap=int(self.svm_cap))

    def to_dict(self) -> dict[str, Any]:
        """The request as plain values (for manifests and the run history)."""
        out = asdict(self)
        out["channels"] = list(self.channels)
        return out


@dataclass
class TrainingData:
    """Training and test matrices for one fit, with the choices and reports that produced them.

    ``train_rows``/``test_rows`` are positions (``iloc``) into ``prepared.frame``; ``X_*`` are float32 with columns
    in ``feature_names`` order; ``y_*`` are int64 codes into ``classes``. ``reports`` holds plain values: ``rows``,
    ``target`` (classes and dropped classes), ``model_space_duplicates`` (per class), ``conflicts``,
    ``topk_overlap`` (None unless top-K), ``class_counts_train``/``class_counts_test`` and ``seconds``.
    ``test_copies`` (int64 per held-out row, or None when the sample carries no counts) is how many rows of the
    source files each held-out row stands for, once every exact repeat (also over the chosen columns) is counted.
    """

    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    classes: tuple[str, ...]
    detailed_test_labels: np.ndarray
    feature_names: tuple[str, ...]
    feature_choice: FeatureChoice
    train_rows: np.ndarray
    test_rows: np.ndarray
    reports: dict[str, Any] = field(default_factory=dict)
    test_copies: np.ndarray | None = field(default=None, repr=False)

    @property
    def n_classes(self) -> int:
        """Number of classes in the target."""
        return len(self.classes)

    @property
    def n_features(self) -> int:
        """Number of feature columns."""
        return len(self.feature_names)


@dataclass
class ChannelResult:
    """The outcome of fitting one channel.

    ``estimator`` is the fitted model (a pipeline, or for the SVM a calibrated wrapper around the fitted pipeline);
    ``proba`` holds float32 class probabilities on the test rows (n_test x K, rows summing to 1) and ``y_pred`` their
    argmax. ``rows_used`` is the number of training rows the model was fitted on, ``rows_available`` the size of
    the training split. ``extra`` holds plain values: ``metrics`` (see :func:`graticule.evaluate.quick_metrics`),
    ``flows_per_second`` and model details such as ``best_iteration``, ``n_trees`` or ``svm_rows_used``.
    """

    key: str
    status: ChannelStatus
    estimator: Any | None = None
    fit_seconds: float = 0.0
    rows_used: int = 0
    rows_available: int = 0
    notes: list[str] = field(default_factory=list)
    error: str | None = None
    y_pred: np.ndarray | None = None
    proba: np.ndarray | None = None
    predict_seconds: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        """True when the channel was fitted and scored."""
        return self.status == "ok"


@dataclass
class TrainingRun:
    """A finished fit: the request, the data it used and one :class:`ChannelResult` per requested channel.

    ``seconds`` covers the channel loop only (fitting, scoring, reference rows); the time spent building the
    matrices (including any Top-K ranking) is ``data.reports["seconds"]``, and :attr:`total_seconds` adds both.

    ``origin`` is ``"fitted"`` for a run fitted in this process and ``"loaded"`` for one restored from a saved
    bundle (:func:`graticule.persist.restore_run`); ``bundle_path`` is the folder the run was saved to or loaded
    from (None while unsaved). A loaded run may come without its held-out rows (see :attr:`has_test_rows`).
    """

    run_id: str
    created_utc: str
    request: TrainRequest
    data_request: DataRequest
    dataset_fingerprint: str
    data: TrainingData
    channels: dict[str, ChannelResult]
    seconds: float
    reference_sample: np.ndarray
    feature_quantiles: np.ndarray
    origin: RunOrigin = "fitted"
    bundle_path: str | None = None

    @property
    def has_test_rows(self) -> bool:
        """True when the held-out rows are in memory (False for a bundle loaded without its data folder)."""
        return len(self.data.y_test) > 0

    def ok_channels(self) -> list[str]:
        """Keys of the channels that were fitted successfully, in channel order."""
        return [k for k in MODEL_KEYS if k in self.channels and self.channels[k].ok]

    @property
    def cancelled(self) -> bool:
        """True when the fit was cancelled before every requested channel finished."""
        return any(r.status in ("cancelled", "skipped") for r in self.channels.values())

    @property
    def classes(self) -> tuple[str, ...]:
        """Class names, in code order."""
        return self.data.classes

    @property
    def prep_seconds(self) -> float:
        """Seconds spent building the training and test matrices (target, de-duplication, split, Top-K ranking)."""
        value = (self.data.reports or {}).get("seconds")
        return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else 0.0

    @property
    def total_seconds(self) -> float:
        """The whole fit as the user waited for it: building the matrices plus the channel loop."""
        return self.prep_seconds + float(self.seconds)


def new_run_id() -> str:
    """A fresh run id: UTC date and time plus four random hex digits, e.g. ``20261001-153012-ab12``."""
    return f"{datetime.now(timezone.utc):%Y%m%d-%H%M%S}-{secrets.token_hex(2)}"


# --------------------------------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------------------------------
def _notify(progress: ProgressSink | None, key: str, **kwargs: Any) -> None:
    """Send a progress update when a sink is attached."""
    if progress is not None:
        progress.update(key, **kwargs)


def _check(cancel: CancelToken | None) -> None:
    """Raise :class:`TrainingCancelled` once the cancel token has fired."""
    if cancel is not None and cancel.cancelled:
        raise TrainingCancelled("The fit was cancelled.")


def _counts(names: np.ndarray | pd.Series, order: list[str] | tuple[str, ...] | None = None) -> dict[str, int]:
    """Rows per class as plain ints, in ``order`` when given (else largest first)."""
    counts = pd.Series(np.asarray(names, dtype=object)).value_counts()
    if order is None:
        return {str(k): int(v) for k, v in counts.items()}
    return {str(name): int(counts.get(name, 0)) for name in order}


def _matrix(frame: pd.DataFrame, columns: tuple[str, ...] | list[str], rows: np.ndarray) -> np.ndarray:
    """Rows ``rows`` of ``columns`` as a fresh, writeable, C-ordered float32 matrix (one column at a time)."""
    out = np.empty((len(rows), len(columns)), dtype=np.float32)
    for j, name in enumerate(columns):
        out[:, j] = frame[name].to_numpy(dtype=np.float32, na_value=np.nan)[rows]
    return out


def _stratified_take(y: np.ndarray, size: int, seed: int, *, leave: int = 1) -> np.ndarray:
    """Seeded stratified draw of ``size`` row indices (fewer only when the classes cannot give more).

    Rows are shared in proportion to class size (largest-remainder rounding). Every class with more than ``leave``
    rows gives at least one row, and every class keeps at least ``leave`` rows out of the draw. Returns sorted
    indices.
    """
    n = len(y)
    if n == 0 or size <= 0:
        return np.empty(0, dtype=np.int64)
    classes, inverse = np.unique(y, return_inverse=True)
    sizes = np.bincount(inverse, minlength=len(classes)).astype(np.int64)
    room = np.maximum(sizes - leave, 0)
    target = int(min(size, room.sum()))
    exact = sizes * (target / n)
    quota = np.minimum(np.maximum(np.floor(exact).astype(np.int64), (room > 0).astype(np.int64)), room)
    order = np.argsort(-(exact - np.floor(exact)), kind="stable")
    while quota.sum() < target:
        grew = False
        for c in order:
            if quota.sum() >= target:
                break
            if quota[c] < room[c]:
                quota[c] += 1
                grew = True
        if not grew:
            break
    while quota.sum() > target:
        c = int(np.argmax(np.where(quota > 1, quota, -1)))
        if quota[c] <= 1:
            break
        quota[c] -= 1
    rng = np.random.default_rng(seed)
    picked = [rng.choice(np.flatnonzero(inverse == c), size=int(quota[c]), replace=False)
              for c in range(len(classes)) if quota[c] > 0]
    return np.sort(np.concatenate(picked)).astype(np.int64) if picked else np.empty(0, dtype=np.int64)


def _calibration_take(y: np.ndarray, size: int, seed: int, *, floor: int = CALIBRATION_FLOOR) -> np.ndarray:
    """Seeded rare-aware draw of about ``size`` row indices for calibrating the SVM (sorted).

    Each class gives its proportional share, but at least ``floor`` rows, and never more than half of its rows (the
    SVM keeps the rest). The large classes give up what the floors add, so the draw stays at ``size`` rows; only
    when the floors alone need more (tiny samples) is it larger, and it never takes more than half of any class.
    Every class with at least two rows gives at least one.
    """
    n = len(y)
    if n == 0 or size <= 0:
        return np.empty(0, dtype=np.int64)
    classes, inverse = np.unique(y, return_inverse=True)
    sizes = np.bincount(inverse, minlength=len(classes)).astype(np.int64)
    half = sizes // 2
    base = np.minimum(half, int(floor))
    total = int(min(max(int(size), int(base.sum())), int(half.sum())))
    want = np.minimum(np.maximum(sizes * (total / n), base), half).astype(np.float64)
    excess = float(want.sum()) - total
    if excess > 0:
        above = want - base
        want = base + above * (1.0 - excess / float(above.sum()))
    quota = np.maximum(np.floor(want + 1e-9).astype(np.int64), base)
    order = np.argsort(-(want - quota), kind="stable")
    while quota.sum() < total:
        grew = False
        for c in order:
            if quota.sum() >= total:
                break
            if quota[c] < half[c]:
                quota[c] += 1
                grew = True
        if not grew:
            break
    rng = np.random.default_rng(seed)
    picked = [rng.choice(np.flatnonzero(inverse == c), size=int(quota[c]), replace=False)
              for c in range(len(classes)) if quota[c] > 0]
    return np.sort(np.concatenate(picked)).astype(np.int64) if picked else np.empty(0, dtype=np.int64)


def _short_warning(message: warnings.WarningMessage) -> str:
    """One line for a captured warning: its category and the first line of its text."""
    text = str(message.message).strip().splitlines()[0] if str(message.message).strip() else ""
    if len(text) > 180:
        text = text[:177] + "..."
    return f"{message.category.__name__}: {text}"


def _warning_notes(caught: list[warnings.WarningMessage], *,
                   skip: tuple[type[Warning], ...] = ()) -> list[str]:
    """Distinct one-line notes for the warnings recorded while a channel was fitted (first five).

    Warnings matching :data:`QUIET_WARNINGS`, and those of the categories in ``skip`` (already reported another
    way), are left out.
    """
    notes: list[str] = []
    for message in caught:
        if skip and issubclass(message.category, skip):
            continue
        if any(quiet in str(message.message) for quiet in QUIET_WARNINGS):
            continue
        line = _short_warning(message)
        if line not in notes:
            notes.append(line)
    return notes[:5]


def _tidy_proba(proba: np.ndarray, model_classes: np.ndarray | None, n_classes: int) -> np.ndarray:
    """Probabilities as float32 (n x K) in code order, clipped at 0 and renormalised so every row sums to 1."""
    values = np.asarray(proba, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("predict_proba returned an array that is not two-dimensional.")
    if values.shape[1] != n_classes:
        if model_classes is None:
            raise ValueError("The model's probabilities do not cover every class.")
        full = np.zeros((values.shape[0], n_classes), dtype=np.float64)
        full[:, np.asarray(model_classes, dtype=np.int64)] = values
        values = full
    values = np.clip(np.nan_to_num(values, nan=0.0), 0.0, None)
    totals = values.sum(axis=1, keepdims=True)
    empty = totals[:, 0] <= 0
    if empty.any():
        values[empty] = 1.0 / n_classes
        totals[empty] = 1.0
    return (values / totals).astype(np.float32)


# --------------------------------------------------------------------------------------------------------------
# Training data
# --------------------------------------------------------------------------------------------------------------
def build_training_data(
    prepared: PreparedDataset,
    request: TrainRequest,
    *,
    progress: ProgressSink | None = None,
    cancel: CancelToken | None = None,
    ranking: Sequence[tuple[str, float]] | None = None,
) -> TrainingData:
    """Build the training and test matrices for ``request`` (see the module notes for the order of work).

    Raises :class:`~graticule.data.sampling.SingleClassError` with a message for the user when fewer than two
    classes remain (for example BENIGN-only data in binary mode). ``progress`` receives stage updates under
    ``STAGE_KEY``; ``cancel`` is checked between steps.

    ``ranking`` (Top-K only) is a feature ranking recorded by an earlier build of the very same matrices (a saved
    run's ``feature_choice.ranking``): it is used as it is instead of ranking the features again, so rebuilding a
    saved run fits no ranking model. Callers must still check that the columns and rows come out as recorded.
    """
    started = time.perf_counter()
    frame = prepared.frame
    stage = "preparing"
    _notify(progress, STAGE_KEY, status=stage, fraction=0.02, message="Building the target")
    labels = prepared.labels()
    target = sampling.target_for_mode(labels, request.mode, min_class_count=int(request.min_class_count))
    positions = np.flatnonzero(target.keep).astype(np.int64)
    names = target.target.to_numpy(dtype=object)
    _check(cancel)

    # Feature space (degenerate columns found on the rows taking part).
    rows_frame = frame if len(positions) == len(frame) else frame[list(FEATURES)].iloc[positions]
    degenerate = find_degenerate_columns(rows_frame, FEATURES)
    topk = request.feature_mode == "topk"
    if topk:
        # De-duplicate over every candidate column (plus the port when it was opted in); rank without the port.
        space = select_features("all", degenerate=degenerate, include_port=request.include_port).columns
        ranked = [c for c in space if c != DESTINATION_PORT]
        choice: FeatureChoice | None = None
    else:
        choice = select_features(request.feature_mode, degenerate=degenerate, include_port=request.include_port)
        space = choice.columns
    _notify(progress, STAGE_KEY, status=stage, fraction=0.15,
            message=f"Removing rows that repeat over the {len(space)} chosen columns")

    # Model-space de-duplication over (space + target), then conflicting targets.
    feature_hashes = row_hashes(rows_frame, list(space))
    del rows_frame
    full_hashes = hashes_with_labels(feature_hashes, pd.Series(names, dtype="str"))
    keep = first_occurrences(full_hashes)
    copies = copies_per_kept_row(full_hashes, keep)
    # How many rows of the source files each kept row stands for (repeats removed at 01 Sample and here).
    base = getattr(prepared, "copies", None)
    flow_copies = (copies_per_kept_row(full_hashes, keep, np.asarray(base, dtype=np.int64)[positions])
                   if base is not None and len(base) == len(frame) else None)
    removed = names[~keep]
    duplicates = {
        "columns": len(space),
        "rows_before": int(len(keep)),
        "rows_removed": int((~keep).sum()),
        "rows_after": int(keep.sum()),
        "by_class": _counts(removed) if len(removed) else {},
    }
    positions, names, feature_hashes = positions[keep], names[keep], feature_hashes[keep]
    in_conflict, groups = conflict_mask(feature_hashes, names)
    survive = conflict_keep_mask(feature_hashes, names, request.conflict_policy, copies)
    conflicts = {
        "groups": int(groups),
        "rows": int(in_conflict.sum()),
        "by_class": _counts(names[in_conflict]) if groups else {},
        "policy": request.conflict_policy,
        "rows_removed": int((~survive).sum()),
    }
    positions, names = positions[survive], names[survive]
    if flow_copies is not None:
        flow_copies = flow_copies[survive]
    _check(cancel)

    # Classes may have shrunk below their minimum: check them again on the rows that are left.
    threshold = int(request.min_class_count) if request.mode == "multiclass" else HARD_FLOOR
    try:
        final = sampling.target_for_mode(pd.Series(names, dtype="str"), "multiclass", min_class_count=threshold)
    except SingleClassError as exc:
        raise SingleClassError(
            "After removing rows that repeat over the chosen feature columns, fewer than two classes have enough "
            f"rows. {exc}", exc.present, exc.dropped) from exc
    positions = positions[final.keep]
    if flow_copies is not None:
        flow_copies = np.asarray(flow_copies, dtype=np.int64)[final.keep]
    codes = final.codes
    classes = tuple(final.classes)
    dropped_after = dict(final.dropped)

    # Stratified split.
    _notify(progress, STAGE_KEY, status=stage, fraction=0.25, message="Splitting into training and test rows")
    split = sampling.stratified_split(codes, float(request.test_share), int(request.seed))
    train_rows, test_rows = positions[split.train], positions[split.test]
    y_train, y_test = codes[split.train].astype(np.int64), codes[split.test].astype(np.int64)
    _check(cancel)

    # Top-K: rank on the training rows only (a callback reports the rounds and stops them when cancelled).
    overlap: dict[str, Any] | None = None
    if topk:
        max_rows = TOPK_MAX_ROWS.get(request.mode, 50_000)
        reused = ranking is not None
        if ranking is not None:
            ranking = [(str(name), float(score)) for name, score in ranking]
            rank_seconds = 0.0
        else:
            label = f"Ranking {len(ranked)} features on up to {min(max_rows, len(train_rows)):,} training rows"
            _notify(progress, STAGE_KEY, status=stage, fraction=0.3, message=label)
            mark = time.perf_counter()
            hook_id = register_hooks(progress, cancel)
            reporter = _RoundReporter(hook_id, STAGE_KEY, RANK_ROUNDS, start=0.3, span=0.55, label=label)
            try:
                ranking = rank_features(_matrix(frame, ranked, train_rows), y_train, ranked,
                                        seed=int(request.seed), max_rows=max_rows, callbacks=[reporter])
            finally:
                release_hooks(hook_id)
            rank_seconds = time.perf_counter() - mark
            if reporter.cancelled:
                raise TrainingCancelled("The fit was cancelled while the features were being ranked.")
        _check(cancel)
        choice = select_features("topk", degenerate=degenerate, include_port=request.include_port, ranking=ranking,
                                 k=int(request.top_k))
        chosen = frame[list(choice.columns)]
        train_k = row_hashes(chosen.iloc[train_rows], list(choice.columns))
        test_k = row_hashes(chosen.iloc[test_rows], list(choice.columns))
        del chosen
        seen = np.isin(test_k, train_k)
        overlap = {
            "k": int(request.top_k),
            "columns": len(choice.columns),
            "candidates": len(ranked),
            "kept": len([c for c in choice.columns if c != DESTINATION_PORT]),
            "port_added": DESTINATION_PORT in choice.columns,
            "ranked_on_rows": int(min(max_rows, len(train_rows))),
            "ranking_seconds": round(rank_seconds, 3),
            "ranking_reused": reused,
            "test_rows": int(len(test_rows)),
            "test_rows_seen_in_train": int(seen.sum()),
            "share": float(seen.mean()) if len(seen) else 0.0,
            "by_class": _counts(np.asarray(classes, dtype=object)[y_test[seen]], classes) if seen.any() else {},
        }
    assert choice is not None
    columns = tuple(choice.columns)

    _notify(progress, STAGE_KEY, status=stage, fraction=0.9, message="Copying the training and test matrices")
    X_train = _matrix(frame, columns, train_rows)
    X_test = _matrix(frame, columns, test_rows)
    detailed = frame[LABEL].to_numpy(dtype=object)[test_rows].astype(str)
    class_names = np.asarray(classes, dtype=object)
    reports: dict[str, Any] = {
        "rows": {
            "prepared": int(len(frame)),
            "in_target": int(target.keep.sum()),
            "after_dedupe": int(len(names)),
            "used": int(len(positions)),
            "train": int(len(train_rows)),
            "test": int(len(test_rows)),
        },
        "target": {
            "mode": request.mode,
            "classes": list(classes),
            "counts": {str(k): int(v) for k, v in final.counts.items()},
            "dropped": {str(k): int(v) for k, v in target.dropped.items()},
            "dropped_reasons": {str(k): str(v) for k, v in target.dropped_reasons.items()},
            "dropped_after_dedupe": {str(k): int(v) for k, v in dropped_after.items()},
            "dropped_after_dedupe_reasons": {str(k): str(v) for k, v in final.dropped_reasons.items()},
        },
        "model_space_duplicates": duplicates,
        "conflicts": conflicts,
        "topk_overlap": overlap,
        "degenerate": {"constant": list(degenerate.constant), "duplicate_of": dict(degenerate.duplicate_of)},
        "class_counts_train": _counts(class_names[y_train], classes),
        "class_counts_test": _counts(class_names[y_test], classes),
        "seconds": round(time.perf_counter() - started, 3),
    }
    _notify(progress, STAGE_KEY, status=stage, fraction=1.0,
            message=f"{len(train_rows):,} training and {len(test_rows):,} test rows ready")
    return TrainingData(
        X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test, classes=classes,
        detailed_test_labels=detailed, feature_names=columns, feature_choice=choice,
        train_rows=train_rows.astype(np.int64), test_rows=test_rows.astype(np.int64), reports=reports,
        test_copies=None if flow_copies is None else flow_copies[split.test],
    )


# --------------------------------------------------------------------------------------------------------------
# Fitting
# --------------------------------------------------------------------------------------------------------------
class _RoundReporter(TrainingCallback):
    """XGBoost callback holding only plain values; progress and cancel are looked up in the hooks registry.

    Progress for ``key`` moves from ``start`` to ``start + span``; ``label`` (when given) prefixes the message.
    ``cancelled`` turns True when the callback stopped the boosting because the fit was cancelled.
    """

    def __init__(self, hook_id: str, key: str, total: int, *, start: float = 0.05, span: float = 0.9,
                 label: str | None = None) -> None:
        super().__init__()
        self.hook_id = hook_id
        self.key = key
        self.total = int(total)
        self.start = float(start)
        self.span = float(span)
        self.label = label
        self.cancelled = False

    def after_iteration(self, model: Any, epoch: int, evals_log: Any) -> bool:
        """Report the round just finished; returning True stops training (used for cancel)."""
        progress, cancel = find_hooks(self.hook_id)
        done = epoch + 1
        if progress is not None and (done % 5 == 0 or done == self.total):
            text = (f"Round {done} of up to {self.total}" if self.label is None
                    else f"{self.label}: round {done} of {self.total}")
            progress.update(self.key, fraction=self.start + self.span * done / max(self.total, 1), message=text)
        if cancel is not None and cancel.cancelled:
            self.cancelled = True
            return True
        return False


class _IterationReporter:
    """scikit-learn fit callback for logistic regression's L-BFGS: progress per iteration and a cancel check.

    It holds only plain values (progress and cancel are looked up in the hooks registry) and is detached from the
    model right after the fit. Raising :class:`TrainingCancelled` from the hook ends the fit.
    """

    def __init__(self, hook_id: str, key: str, limit: int) -> None:
        self.hook_id = hook_id
        self.key = key
        self.limit = max(int(limit), 1)
        self.iterations = 0
        self._last_report = 0.0

    def setup(self, estimator: Any, context: Any) -> None:
        """Called by scikit-learn before the fit (nothing to prepare)."""

    def teardown(self, estimator: Any, context: Any) -> None:
        """Called by scikit-learn after the fit (nothing to clean up)."""

    def on_fit_task_begin(self, estimator: Any, context: Any) -> None:
        """Called by scikit-learn when a fit task starts (nothing to do)."""

    def on_fit_task_end(self, estimator: Any, context: Any) -> bool:
        """After each L-BFGS iteration: stop the fit when it has been cancelled, else report now and then."""
        if getattr(context, "task_name", None) != "lbfgs-iter":
            return False
        self.iterations += 1
        progress, cancel = find_hooks(self.hook_id)
        if cancel is not None and cancel.cancelled:
            raise TrainingCancelled("The fit was cancelled.")
        now = time.perf_counter()
        if progress is not None and now - self._last_report >= 0.5:
            self._last_report = now
            progress.update(self.key, fraction=0.05 + 0.9 * min(self.iterations / self.limit, 1.0),
                            message=f"Iteration {self.iterations:,} (at most {self.limit:,})")
        return False


def _epoch_observer(hook_id: str, key: str, limit: int) -> Callable[[Any], None]:
    """The per-epoch hook for :class:`~graticule.models.zoo.ObservableMLP`: a cancel check, then progress."""
    def observe(model: Any) -> None:
        progress, cancel = find_hooks(hook_id)
        if cancel is not None and cancel.cancelled:
            raise TrainingCancelled("The fit was cancelled.")
        if progress is not None:
            epoch = int(getattr(model, "n_iter_", 0) or 0)
            scores = getattr(model, "validation_scores_", None) or []
            score = f"; validation score {float(scores[-1]):.4f}" if scores else ""
            progress.update(key, fraction=0.05 + 0.9 * min(epoch / max(limit, 1), 1.0),
                            message=f"Epoch {epoch} of up to {limit}{score}")

    return observe


def _fit_prefix(estimator: Pipeline, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Fit every step before the model on the training rows and return the transformed matrix."""
    if len(estimator.steps) == 1:
        return X
    return estimator[:-1].fit_transform(X, y)


def _subset_weights(sample_weight: np.ndarray, y: np.ndarray, cap: float) -> np.ndarray:
    """Weights for a subset of training rows: unit weights stay unit, otherwise balanced weights are recomputed
    on the subset (its class mix differs from the full training split)."""
    if sample_weight.size == 0 or np.allclose(sample_weight, sample_weight[0]):
        return np.ones(len(y), dtype=np.float64)
    return zoo.balanced_weights(y, cap)


def _fit_forest(estimator: Pipeline, X: np.ndarray, y: np.ndarray, w: np.ndarray, ctx: BuildContext,
                progress: ProgressSink | None, cancel: CancelToken | None) -> dict[str, Any]:
    """Grow the forest in warm-start chunks with progress and a cancel check between chunks."""
    Xt = _fit_prefix(estimator, X, y)
    model = estimator[-1]
    total = int(model.n_estimators)
    chunk = zoo.FOREST_CHUNK if ctx.profile == "full" else max(total // 2, 1)
    model.set_params(warm_start=True)
    grown = 0
    while grown < total:
        _check(cancel)
        grown = min(total, grown + chunk)
        model.set_params(n_estimators=grown)
        model.fit(Xt, y, sample_weight=w)
        _notify(progress, "forest", fraction=0.02 + 0.96 * grown / total, message=f"{grown} of {total} trees")
    model.set_params(warm_start=False)
    return {"rows_used": len(y), "notes": [], "extra": {"n_trees": len(model.estimators_)}}


def _fit_xgboost(estimator: Pipeline, X: np.ndarray, y: np.ndarray, w: np.ndarray, ctx: BuildContext,
                 hook_id: str) -> dict[str, Any]:
    """Boost with early stopping on a stratified 10 % slice of the training rows."""
    Xt = _fit_prefix(estimator, X, y)
    model = estimator[-1]
    eval_idx = _stratified_take(y, int(round(len(y) * XGB_EVAL_SHARE)), ctx.seed + 1, leave=1)
    fit_mask = np.ones(len(y), dtype=bool)
    fit_mask[eval_idx] = False
    total = int(model.n_estimators)
    reporter = _RoundReporter(hook_id, "xgboost", total)
    model.set_params(callbacks=[reporter])
    try:
        model.fit(Xt[fit_mask], y[fit_mask], sample_weight=w[fit_mask], eval_set=[(Xt[eval_idx], y[eval_idx])],
                  sample_weight_eval_set=[w[eval_idx]], verbose=False)
    finally:
        model.set_params(callbacks=None)
    if reporter.cancelled:
        raise TrainingCancelled("The fit was cancelled.")
    rounds = int(model.get_booster().num_boosted_rounds())
    best = int(getattr(model, "best_iteration", rounds - 1))
    notes = [f"Early stopping on {len(eval_idx):,} held-back training rows: best round {best + 1} of {rounds}."]
    return {"rows_used": int(fit_mask.sum()), "notes": notes,
            "extra": {"best_iteration": best, "rounds": rounds, "early_stopping_rows": int(len(eval_idx))}}


def _fit_svm(estimator: Pipeline, X: np.ndarray, y: np.ndarray, w: np.ndarray, ctx: BuildContext,
             progress: ProgressSink | None, cancel: CancelToken | None) -> tuple[Any, dict[str, Any]]:
    """Fit the SVM on a capped rare-aware subset, then calibrate a sigmoid on separate training rows.

    The calibration rows (at most 5,000) are drawn first, rare-aware (:func:`_calibration_take`: every class gives
    at least 20 rows, or half of its rows when it has fewer than 40), and the SVM draws its rows from the rest. The
    calibration uses one split over all of its rows (:class:`~graticule.models.zoo.WholeSliceSplit`).
    """
    n = len(y)
    cap = ctx.effective_svm_cap
    cal_size = min(CALIBRATION_ROWS, max(n - cap, n // 5))
    cal_idx = _calibration_take(y, cal_size, ctx.seed + 2)
    rest = np.setdiff1d(np.arange(n, dtype=np.int64), cal_idx, assume_unique=True)
    budget = min(cap, len(rest))
    picked, _ = sampling.sample_positions(y[rest], budget, ctx.seed)
    svm_idx = rest[picked]
    weight_cap = zoo.MODEL_SPECS["svm"].weight_cap
    _notify(progress, "svm", fraction=0.05, message=f"Fitting on {len(svm_idx):,} rows")
    Xt = _fit_prefix(estimator, X[svm_idx], y[svm_idx])
    estimator[-1].fit(Xt, y[svm_idx], sample_weight=_subset_weights(w, y[svm_idx], weight_cap))
    _check(cancel)
    _notify(progress, "svm", fraction=0.85, message=f"Calibrating on {len(cal_idx):,} other rows")
    # The frozen SVM is never refitted, so one split over every calibration row is all the calibration needs; the
    # default five folds would only demand five rows of every class. Its warning that the weights reach only the
    # sigmoid is expected (see QUIET_WARNINGS).
    calibrated = CalibratedClassifierCV(FrozenEstimator(estimator), method="sigmoid", cv=WholeSliceSplit())
    calibrated.fit(X[cal_idx], y[cal_idx], sample_weight=_subset_weights(w, y[cal_idx], weight_cap))
    per_class = np.bincount(y[cal_idx], minlength=ctx.n_classes)
    notes = []
    if len(svm_idx) < n:
        reason = f"SVM cap {cap:,}" if len(svm_idx) >= cap else "the rest calibrate its probabilities"
        notes.append(f"Trained on {len(svm_idx):,} of {n:,} training rows ({reason}).")
    notes.append(f"Probabilities calibrated (sigmoid) on {len(cal_idx):,} training rows the SVM did not see, at "
                 f"least {int(per_class[per_class > 0].min()) if per_class.any() else 0:,} of every class.")
    extra = {"svm_rows_used": int(len(svm_idx)), "calibration_rows": int(len(cal_idx)), "svm_cap": int(cap),
             "calibration_rows_per_class": [int(v) for v in per_class],
             "support_vectors": int(estimator[-1].n_support_.sum())}
    return calibrated, {"rows_used": int(len(svm_idx)), "notes": notes, "extra": extra}


def _fit_plain(key: str, estimator: Pipeline, X: np.ndarray, y: np.ndarray, w: np.ndarray,
               progress: ProgressSink | None, hook_id: str) -> dict[str, Any]:
    """One ordinary fit of the whole pipeline (MLP, logistic regression or any other channel).

    The MLP reports every epoch and logistic regression (L-BFGS) every iteration, both with a cancel check; any other
    model is one uninterrupted call. Whether the model stopped at its iteration limit is read from the fitted
    model and becomes a ConvergenceWarning note.
    """
    _notify(progress, key, fraction=0.05, message=f"Fitting on {len(y):,} rows")
    Xt = _fit_prefix(estimator, X, y)
    model = estimator[-1]
    limit = getattr(model, "max_iter", None)
    if isinstance(model, ObservableMLP):
        model.epoch_observer = _epoch_observer(hook_id, key, int(limit or 1))
        try:
            model.fit(Xt, y, sample_weight=w)
        finally:
            model.__dict__.pop("epoch_observer", None)
    elif hasattr(model, "set_callbacks") and getattr(model, "solver", None) == "lbfgs":
        model.set_callbacks(_IterationReporter(hook_id, key, int(limit or 1)))
        try:
            model.fit(Xt, y, sample_weight=w)
        finally:
            model.set_callbacks()
    else:
        model.fit(Xt, y, sample_weight=w)
    extra: dict[str, Any] = {}
    notes: list[str] = []
    n_iter = getattr(model, "n_iter_", None)
    if n_iter is not None:
        extra["n_iter"] = int(np.max(np.asarray(n_iter)))
        extra["hit_iteration_limit"] = bool(limit is not None and extra["n_iter"] >= int(limit))
        if extra["hit_iteration_limit"]:
            what = "epochs" if isinstance(model, ObservableMLP) else "iterations"
            notes.append(f"ConvergenceWarning: stopped at the limit of {int(limit):,} {what} before converging.")
    best = getattr(model, "best_validation_score_", None)
    if best is not None:
        extra["best_validation_score"] = float(best)
    return {"rows_used": len(y), "notes": notes, "extra": extra}


def fit_model(
    key: str,
    estimator: Any,
    X: np.ndarray,
    y: np.ndarray,
    sample_weight: np.ndarray,
    *,
    ctx: BuildContext,
    progress: ProgressSink | None = None,
    cancel: CancelToken | None = None,
    job_id: str | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Fit channel ``key`` on the training rows and return (fitted model, info). The ONLY place models are fitted.

    Every call adds one to ``FIT_CALLS[key]``. ``key`` :data:`RANKING_KEY` fits the Top-K ranking model (any
    estimator with ``fit(X, y, sample_weight=...)``, used as it is). Channel specifics: the forest grows in
    warm-start chunks of 25 trees; XGBoost holds back a stratified 10 % of the rows for early stopping (its callback
    holds only plain values and is removed from the fitted model); the SVM trains on at most
    ``ctx.effective_svm_cap`` rows (rare-aware draw) and returns a
    ``CalibratedClassifierCV(FrozenEstimator(pipeline), method="sigmoid")`` fitted, over one split, on up to 5,000
    other training rows drawn rare-aware (see :func:`_calibration_take`); the MLP and logistic regression report
    every epoch or iteration. When the SVM sees a subset, balanced weights are recomputed on that
    subset (unit weights stay unit). ``info`` holds ``rows_used``, ``notes`` (including the warnings raised on this
    thread during the fit, see :func:`graticule.models.jobs.thread_warnings`) and ``extra``.

    Raises :class:`TrainingCancelled` when ``cancel`` fires before the model is complete (checked before the fit,
    between forest chunks, boosting rounds, MLP epochs and L-BFGS iterations, and between the SVM's fit and its
    calibration). A model that was completed is returned even if the cancel came during its last step.
    """
    with _FIT_CALLS_LOCK:
        FIT_CALLS[key] += 1
    X = np.asarray(X)
    y = np.asarray(y, dtype=np.int64)
    weights = np.asarray(sample_weight, dtype=np.float64)
    if len(X) != len(y) or len(y) != len(weights):
        raise ValueError("X, y and sample_weight must have the same number of rows.")
    hook_id = register_hooks(progress, cancel, job_id=job_id)
    try:
        with thread_warnings() as caught:
            _check(cancel)
            if key == RANKING_KEY:
                estimator.fit(X, y, sample_weight=weights)
                info = {"rows_used": len(y), "notes": [], "extra": {}}
                fitted: Any = estimator
            elif key == "forest":
                info = _fit_forest(estimator, X, y, weights, ctx, progress, cancel)
                fitted = estimator
            elif key == "xgboost":
                info = _fit_xgboost(estimator, X, y, weights, ctx, hook_id)
                fitted = estimator
            elif key == "svm":
                fitted, info = _fit_svm(estimator, X, y, weights, ctx, progress, cancel)
            else:
                info = _fit_plain(key, estimator, X, y, weights, progress, hook_id)
                fitted = estimator
    finally:
        release_hooks(hook_id)
    # A model that reports its own iteration limit already has a convergence note.
    reported = (ConvergenceWarning,) if info.get("extra", {}).get("hit_iteration_limit") else ()
    notes = list(info.get("notes", []))
    notes.extend(n for n in _warning_notes(caught, skip=reported) if n not in notes)
    info["notes"] = notes
    return fitted, info


def score_in_blocks(
    model: Any,
    X: np.ndarray,
    *,
    cancel: CancelToken | None = None,
    after_block: Callable[[int, int], None] | None = None,
) -> tuple[np.ndarray, float]:
    """``model.predict_proba(X)`` in fixed blocks of :data:`SCORE_BLOCK` rows; returns (probabilities, seconds).

    ``cancel`` is checked before each block (raising :class:`TrainingCancelled`), so even the kernel SVM stays
    interruptible, and ``after_block`` receives (rows scored so far, rows in all) after each one. The block
    boundaries depend on nothing but the number of rows, so whoever scores the same rows this way (the fit, or a
    run restored from disk) gets the very same numbers. The seconds count prediction calls only.
    """
    n = len(X)
    block = max(int(SCORE_BLOCK), 1)
    parts: list[np.ndarray] = []
    spent = 0.0
    for start in range(0, n, block):
        _check(cancel)
        stop = min(n, start + block)
        mark = time.perf_counter()
        parts.append(np.asarray(model.predict_proba(X[start:stop])))
        spent += time.perf_counter() - mark
        if after_block is not None:
            after_block(stop, n)
    if not parts:
        return np.asarray(model.predict_proba(X)), spent
    return (parts[0] if len(parts) == 1 else np.vstack(parts)), spent


def _score_in_batches(key: str, model: Any, X: np.ndarray, progress: ProgressSink | None,
                      cancel: CancelToken | None) -> tuple[np.ndarray, float]:
    """Score the test rows for channel ``key`` (:func:`score_in_blocks`), reporting progress between blocks."""
    _notify(progress, key, fraction=0.98, message=f"Scoring {len(X):,} test rows")

    def report(done: int, total: int) -> None:
        if done < total:
            _notify(progress, key, message=f"Scoring the test rows: {done:,} of {total:,}")

    return score_in_blocks(model, X, cancel=cancel, after_block=report)


def _train_channel(key: str, data: TrainingData, request: TrainRequest, ctx: BuildContext,
                   progress: ProgressSink | None, cancel: CancelToken | None, job_id: str | None) -> ChannelResult:
    """Build, fit and score one channel; failures and cancels are recorded, never raised."""
    n_train = len(data.y_train)
    result = ChannelResult(key=key, status="failed", rows_available=n_train)
    _notify(progress, key, status="running", fraction=0.0, message="Starting")
    started = time.perf_counter()
    with thread_warnings() as caught:
        try:
            estimator = zoo.build_estimator(key, ctx)
            weights = zoo.channel_weights(key, data.y_train, request.balanced)
            fitted, info = fit_model(key, estimator, data.X_train, data.y_train, weights, ctx=ctx,
                                     progress=progress, cancel=cancel, job_id=job_id)
            result.fit_seconds = time.perf_counter() - started
            raw, result.predict_seconds = _score_in_batches(key, fitted, data.X_test, progress, cancel)
            result.proba = _tidy_proba(raw, getattr(fitted, "classes_", None), data.n_classes)
            result.y_pred = result.proba.argmax(axis=1).astype(np.int64)
            result.estimator = fitted
            result.rows_used = int(info.get("rows_used", n_train))
            result.notes.extend(info.get("notes", []))
            result.extra.update(info.get("extra", {}))
            if request.balanced:
                capped = zoo.capped_classes(data.y_train, zoo.MODEL_SPECS[key].weight_cap)
                if capped:
                    names = ", ".join(data.classes[int(c)] for c in capped)
                    result.notes.append(f"Row weight capped at {zoo.MODEL_SPECS[key].weight_cap:g} for: {names}.")
            else:
                result.notes.append("Unweighted: every training row counts once.")
            result.extra["metrics"] = quick_metrics(data.y_test, result.y_pred, data.n_classes)
            result.extra["flows_per_second"] = (float(len(data.y_test) / result.predict_seconds)
                                                if result.predict_seconds > 0 else None)
            result.status = "ok"
        except TrainingCancelled:
            result.status = "cancelled"
            result.fit_seconds = time.perf_counter() - started
            result.notes.append("Cancelled before this channel finished.")
        except Exception as exc:  # noqa: BLE001 - one failing channel must not stop the others
            result.status = "failed"
            result.fit_seconds = time.perf_counter() - started
            result.error = f"{type(exc).__name__}: {exc}"
            result.extra["traceback"] = traceback.format_exc()
    result.notes.extend(n for n in _warning_notes(caught) if n not in result.notes)
    message = {"ok": f"Fitted in {result.fit_seconds:.1f} s", "failed": "Failed",
               "cancelled": "Cancelled"}.get(result.status, result.status)
    _notify(progress, key, status=result.status, fraction=1.0, message=message)
    return result


def train_all(
    data: TrainingData,
    request: TrainRequest,
    *,
    data_request: DataRequest,
    dataset_fingerprint: str,
    progress: ProgressSink | None = None,
    cancel: CancelToken | None = None,
    job_id: str | None = None,
) -> TrainingRun:
    """Fit every requested channel (in the fixed channel order) and return the run.

    A failing channel is recorded with ``status="failed"`` and the run continues. When ``cancel`` fires, the
    channel being fitted or scored is marked ``"cancelled"`` and the ones not yet started ``"skipped"``; a cancel
    that comes after the last channel has finished changes nothing. The run also keeps a rare-aware reference sample
    of at most 2,000 training rows and the per-feature training quantiles (0..100 %). ``seconds`` times this loop;
    :attr:`TrainingRun.total_seconds` adds the time spent building the matrices.
    """
    started = time.perf_counter()
    run_id = new_run_id()
    created = datetime.now(timezone.utc).isoformat(timespec="seconds")
    ctx = request.build_context(data.n_classes)
    _notify(progress, STAGE_KEY, status="fitting", fraction=0.0, message="Fitting channels")
    results: dict[str, ChannelResult] = {}
    for key in MODEL_KEYS:
        if key not in request.channels:
            continue
        if cancel is not None and cancel.cancelled:
            results[key] = ChannelResult(key=key, status="skipped", rows_available=len(data.y_train),
                                         notes=["Not started: the fit was cancelled."])
            _notify(progress, key, status="skipped", fraction=1.0, message="Skipped (cancelled)")
            continue
        _notify(progress, STAGE_KEY, status="fitting", message=f"Fitting {key}")
        results[key] = _train_channel(key, data, request, ctx, progress, cancel, job_id)

    _notify(progress, STAGE_KEY, status="finishing", fraction=0.2, message="Keeping reference rows and quantiles")
    n_ref = min(REFERENCE_ROWS, len(data.y_train))
    ref_rows, _ = sampling.sample_positions(data.y_train, n_ref, int(request.seed))
    reference = np.array(data.X_train[ref_rows], dtype=np.float32, copy=True)
    quantiles = feature_quantiles(data.X_train)
    _notify(progress, STAGE_KEY, status="finishing", fraction=1.0, message="Finished")
    return TrainingRun(
        run_id=run_id, created_utc=created, request=request, data_request=data_request,
        dataset_fingerprint=dataset_fingerprint, data=data, channels=results,
        seconds=time.perf_counter() - started, reference_sample=reference, feature_quantiles=quantiles,
    )


def feature_quantiles(X: np.ndarray) -> np.ndarray:
    """Per-feature quantiles of ``X`` at 0, 1, ..., 100 % ignoring missing values: float32, shape (101, n_features).

    Infinite values count as missing; a column with no finite value gives NaN throughout.
    """
    values = np.asarray(X, dtype=np.float32)
    result = np.full((len(QUANTILE_LEVELS), values.shape[1]), np.nan, dtype=np.float32)
    if values.shape[0] == 0:
        return result
    if np.isinf(values).any():
        values = np.where(np.isinf(values), np.float32(np.nan), values)
    # Columns without any finite value stay NaN; leaving them out of nanquantile avoids its all-NaN warning (this
    # runs on the fit thread, where changing the process-wide warning filters is not safe).
    usable = ~np.isnan(values).all(axis=0)
    if usable.any():
        result[:, usable] = np.nanquantile(values[:, usable], QUANTILE_LEVELS, axis=0)
    return result
