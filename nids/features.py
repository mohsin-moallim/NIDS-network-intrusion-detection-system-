"""Feature-set selection: which flow columns a channel is allowed to see.

Three modes are offered.

* ``curated`` - a fixed list of 28 columns picked for what each one measures (see
  :data:`nids.schema.CURATED_GROUPS`). It is chosen without looking at any data, so it cannot leak anything.
* ``all`` - every numeric column except the destination port and the columns found to be degenerate (constant or
  exact copies of another column) on the data in hand.

Degenerate columns are best passed as the :class:`~nids.data.clean.DegenerateReport` itself. Constant columns
are then always left out, but a column that merely repeats another one is left out only when the column it repeats
is also chosen. In the real files, for example, ``SYN Flag Count`` is an exact copy of ``Fwd PSH Flags``: the
``all`` set keeps only the earlier column, while the curated set, which does not contain ``Fwd PSH Flags``, keeps
``SYN Flag Count``. A plain collection of names is treated more bluntly: every name in it is left out.
* ``topk`` - the K best columns of a ranking made by :func:`rank_features` on the training split only.

The destination port is never picked automatically. A port number is an identifier, not a measurement, and a model
that leans on it tends to memorise which services the lab happened to attack rather than how attacks behave. It
can be added to any set by explicit opt-in (``include_port=True``), and the UI shows a badge when it is.
"""

from __future__ import annotations

import math
import re
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
import pandas as pd

from nids import schema
from nids.data.clean import DegenerateReport

FeatureMode = Literal["curated", "all", "topk"]
FEATURE_MODES: tuple[FeatureMode, ...] = ("curated", "all", "topk")
DEFAULT_K = 20
#: Boosting rounds of the small ranking model behind Top-K.
RANK_ROUNDS = 120

_BOOSTER_NAME = re.compile(r"f(\d+)")


@dataclass(frozen=True)
class FeatureChoice:
    """The outcome of a feature-set selection.

    Attributes:
        mode: the selection mode that produced it.
        columns: the chosen column names, in the order models will receive them.
        include_port: True when ``Destination Port`` is among ``columns`` (only ever by explicit opt-in).
        k: the requested K for ``topk``; None for the other modes.
        ranking: for ``topk``, the full ranking used as (name, score) pairs, best first (score NaN when only
            names were supplied); None for the other modes.
        dropped_degenerate: degenerate columns that would otherwise have been chosen and were left out.
    """

    mode: FeatureMode
    columns: tuple[str, ...]
    include_port: bool
    k: int | None
    ranking: tuple[tuple[str, float], ...] | None
    dropped_degenerate: tuple[str, ...]

    @property
    def n_features(self) -> int:
        """Number of chosen columns."""
        return len(self.columns)


def _schema_order(names: Sequence[str]) -> list[str]:
    """Sort names by their position in ``schema.FEATURES``; unknown names follow in their given order."""
    rank = {name: i for i, name in enumerate(schema.FEATURES)}
    known = sorted((n for n in names if n in rank), key=rank.__getitem__)
    return known + [n for n in names if n not in rank]


def _as_pairs(ranking: Sequence[str] | Sequence[tuple[str, float]]) -> list[tuple[str, float]]:
    """Accept either ranked names or (name, score) pairs and return (name, score) pairs."""
    pairs: list[tuple[str, float]] = []
    for item in ranking:
        if isinstance(item, str):
            pairs.append((item, math.nan))
        else:
            name, score = item
            pairs.append((str(name), float(score)))
    return pairs


def _degenerate_parts(degenerate: Collection[str] | DegenerateReport) -> tuple[set[str], dict[str, str]]:
    """Split ``degenerate`` into names that are always left out and a copy map (column -> the column it repeats).

    A :class:`DegenerateReport` (or any object with ``constant`` and ``duplicate_of``) gives both parts; a plain
    collection of names gives only the first, so every listed name is left out.
    """
    constant = getattr(degenerate, "constant", None)
    duplicate_of = getattr(degenerate, "duplicate_of", None)
    if constant is not None and duplicate_of is not None:
        return {str(c) for c in constant}, {str(k): str(v) for k, v in dict(duplicate_of).items()}
    return {str(name) for name in degenerate}, {}  # type: ignore[union-attr]


def _original_of(name: str, copies: dict[str, str]) -> str:
    """The column that ``name`` is an exact copy of (following chains), or ``name`` itself."""
    seen = {name}
    while name in copies and copies[name] not in seen:
        name = copies[name]
        seen.add(name)
    return name


def select_features(
    mode: FeatureMode,
    *,
    degenerate: Collection[str] | DegenerateReport = (),
    include_port: bool = False,
    ranking: Sequence[str] | Sequence[tuple[str, float]] | None = None,
    k: int = DEFAULT_K,
    available: Sequence[str] = schema.FEATURES,
) -> FeatureChoice:
    """Choose the columns a channel will be trained on.

    ``curated`` keeps the curated columns that are available and not degenerate; ``all`` keeps every available
    column except ``Destination Port`` and the degenerate ones (both in ``schema.FEATURES`` order). ``topk`` walks
    ``ranking`` (ranked names, or the (name, score) pairs returned by :func:`rank_features`) best first and keeps
    the first ``k`` usable names, skipping unavailable and degenerate names and never counting the port towards
    ``k``. ``include_port=True`` appends ``Destination Port`` to any mode when it is available and not degenerate.

    ``degenerate`` is either the :class:`~nids.data.clean.DegenerateReport` found on the data (preferred) or a
    collection of column names. With a report, constant columns are always left out and, of a group of identical
    columns, only the first one met in selection order is kept, so a curated column is never lost merely because it
    copies a column outside the curated set. With plain names, every listed column is left out.

    Raises ValueError for an unknown mode, a missing ranking or ``k < 1`` in ``topk``, or an empty selection.
    """
    if mode not in FEATURE_MODES:
        raise ValueError(f"Unknown feature mode {mode!r}; choose one of {', '.join(FEATURE_MODES)}.")
    port = schema.DESTINATION_PORT
    usable = set(available)
    always_out, copies = _degenerate_parts(degenerate)
    picked: list[str] = []
    dropped: list[str] = []
    taken: set[str] = set()
    stored_ranking: tuple[tuple[str, float], ...] | None = None
    chosen_k: int | None = None

    def admit(name: str) -> bool:
        """Pick ``name`` unless it is degenerate here; record it as dropped otherwise."""
        original = _original_of(name, copies)
        if name in always_out or original in taken:
            dropped.append(name)
            return False
        taken.add(original)
        picked.append(name)
        return True

    if mode == "topk":
        if ranking is None:
            raise ValueError("Top-K needs a feature ranking computed on the training split.")
        if isinstance(k, bool) or not isinstance(k, (int, np.integer)) or k < 1:
            raise ValueError("k must be a positive integer.")
        chosen_k = int(k)
        pairs = _as_pairs(ranking)
        stored_ranking = tuple(pairs)
        seen: set[str] = set()
        for name, _score in pairs:
            if len(picked) >= chosen_k:
                break
            if name == port or name not in usable or name in seen:
                continue
            seen.add(name)
            admit(name)
    else:
        pool = schema.CURATED if mode == "curated" else tuple(dict.fromkeys(available))
        for name in _schema_order([c for c in pool if c in usable and c != port]):
            admit(name)

    port_in = bool(include_port and port in usable and admit(port))
    if not picked:
        raise ValueError("No usable feature columns are left for this selection.")
    return FeatureChoice(
        mode=mode,
        columns=tuple(picked),
        include_port=port_in,
        k=chosen_k,
        ranking=stored_ranking,
        dropped_degenerate=tuple(dropped),
    )


def _stratified_rows(codes: np.ndarray, max_rows: int, seed: int) -> np.ndarray:
    """Row indices of a seeded stratified subsample of at most ``max_rows`` rows (every class keeps a row)."""
    n = codes.size
    if n <= max_rows:
        return np.arange(n)
    sizes = np.bincount(codes)
    exact = sizes * (max_rows / n)
    quota = np.floor(exact).astype(np.int64)
    short = max_rows - int(quota.sum())
    if short > 0:
        quota[np.argsort(-(exact - quota), kind="stable")[:short]] += 1
    quota = np.minimum(np.maximum(quota, 1), sizes)
    excess = int(quota.sum()) - max_rows
    while excess > 0 and quota.max() > 1:
        quota[int(np.argmax(quota))] -= 1
        excess -= 1
    rng = np.random.default_rng(seed)
    chosen = [
        rng.choice(np.flatnonzero(codes == c), size=int(quota[c]), replace=False) for c in range(sizes.size)
    ]
    return np.sort(np.concatenate(chosen))


def rank_features(
    X_train: np.ndarray | pd.DataFrame,
    y_train: npt.ArrayLike,
    feature_names: Sequence[str],
    *,
    seed: int,
    max_rows: int = 50_000,
    callbacks: Sequence[Any] | None = None,
) -> list[tuple[str, float]]:
    """Rank features by the total gain they earn in a small XGBoost model; return every feature, best first.

    Only ever pass rows from the TRAINING split. Ranking on data that includes test rows lets the test set choose
    the features and makes every later test reading optimistic.

    A seeded stratified subsample of at most ``max_rows`` rows is drawn, labels are encoded 0..K-1, rows are
    weighted so each class carries equal total weight, and a compact gradient-boosted model is fitted
    (120 rounds, depth 6). Each feature's score is the total gain of all splits on it; features never used score
    0.0. ``X_train`` may be an array whose columns follow ``feature_names`` or a DataFrame holding those columns;
    +/-inf values are treated as missing. Ties keep ``feature_names`` order, so the result is deterministic.

    ``callbacks`` are XGBoost training callbacks for the ranking model (progress, or stopping it early when a fit
    is cancelled; a callback that stops it leaves a ranking from the rounds done so far). The model is thrown away
    afterwards, so they are never kept anywhere. Like every model NIDS fits, it is fitted through
    :func:`nids.models.train.fit_model`, so ``FIT_CALLS`` counts it (key ``RANKING_KEY``).
    """
    from sklearn.utils.class_weight import compute_sample_weight
    from xgboost import XGBClassifier

    # Imported here: the trainer imports this module, so a module-level import would be circular.
    from nids.models.train import RANKING_KEY, fit_model
    from nids.models.zoo import BuildContext

    names = [str(n) for n in feature_names]
    if len(set(names)) != len(names):
        raise ValueError("feature_names must be unique.")
    if isinstance(X_train, pd.DataFrame):
        missing = [c for c in names if c not in X_train.columns]
        if missing:
            raise ValueError(f"X_train lacks columns: {', '.join(missing)}.")
        X = X_train.loc[:, names].to_numpy(dtype=np.float32, copy=True)
    else:
        X = np.array(X_train, dtype=np.float32, copy=True)
    if X.ndim != 2 or X.shape[1] != len(names):
        raise ValueError("X_train must be two-dimensional with one column per feature name.")
    y = np.asarray(y_train)
    if y.ndim != 1 or y.shape[0] != X.shape[0]:
        raise ValueError("y_train must hold one label per row of X_train.")
    if max_rows < 2:
        raise ValueError("max_rows must be at least 2.")
    classes, codes = np.unique(y, return_inverse=True)
    if classes.size < 2:
        raise ValueError("Ranking features needs at least two classes in the training rows.")

    rows = _stratified_rows(codes.astype(np.int64), max_rows, seed)
    X = X[rows]
    codes = codes[rows]
    X[~np.isfinite(X)] = np.nan
    model = XGBClassifier(
        tree_method="hist",
        n_estimators=RANK_ROUNDS,
        max_depth=6,
        learning_rate=0.2,
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs=4,
        random_state=seed,
        callbacks=list(callbacks) if callbacks else None,
    )
    fit_model(RANKING_KEY, model, X, codes, compute_sample_weight("balanced", codes),
              ctx=BuildContext(n_classes=int(classes.size), seed=int(seed)))
    scores = model.get_booster().get_score(importance_type="total_gain")

    # The model was fitted on a plain array, so the booster names columns by position ("f0", "f1", ...). Map by
    # that position only: matching the user's own names could pick the wrong column when a name looks like "f3".
    gains = np.zeros(len(names))
    for key, value in scores.items():
        gain = float(np.sum(value))  # a plain float, or one value per target for multi-output boosters
        match = _BOOSTER_NAME.fullmatch(key)
        if match and int(match.group(1)) < len(names):
            gains[int(match.group(1))] = gain
    order = sorted(range(len(names)), key=lambda i: (-gains[i], i))
    return [(names[i], float(gains[i])) for i in order]
