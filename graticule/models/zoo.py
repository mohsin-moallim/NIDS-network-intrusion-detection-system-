"""The five channels: what each model is, how its pipeline is built, and the sample weights it trains with.

Every channel is an unfitted scikit-learn :class:`~sklearn.pipeline.Pipeline` whose first step is a
:class:`~graticule.models.transforms.FlowSanitizer` (float32, ±inf to NaN) and whose last step is named
``"model"``. Models that are sensitive to the scale of their inputs (the SVM, the MLP and logistic regression) add a
median imputer, a signed logarithm and a standard scaler in between. The tree models need none of that: XGBoost and
scikit-learn's random forest both route missing values natively (checked for scikit-learn 1.9: the forest declares
``allow_nan``), so no imputer is ever inserted in front of them, whatever the bad-value strategy of the sample.

Weighting. Every channel trains with the same kind of SAMPLE weights (never ``class_weight=``): balanced weights
``n / (K * n_c)`` so that each class carries the same total weight, with a per-row cap (100, or 50 for the MLP,
whose stochastic updates are more easily thrown off by a few very heavy rows) so that a class with a handful of rows
cannot dominate, and the whole set rescaled to sum to ``n``. See :func:`balanced_weights`.

Profiles. ``"full"`` uses the hyperparameters of the specification; ``"test"`` shrinks every model (a few trees,
rounds or iterations) so that the test suite can train all five channels in seconds.

Two small helpers live here because fitted channels keep them: :class:`ObservableMLP` (scikit-learn's MLP with a
per-epoch hook, so a fit can report progress and stop when cancelled; training itself is unchanged) and
:class:`WholeSliceSplit` (the single split used to calibrate the frozen SVM).
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

from graticule.models.transforms import FlowSanitizer, make_signed_log

#: Channel keys in their fixed order (the same order as ``graticule.theme.CHANNELS``).
MODEL_KEYS: tuple[str, ...] = ("forest", "xgboost", "svm", "mlp", "logreg")
Profile = Literal["full", "test"]
PROFILES: tuple[str, ...] = ("full", "test")

#: Per-row weight caps for the balanced sample weights.
WEIGHT_CAP = 100.0
MLP_WEIGHT_CAP = 50.0
#: The forest grows in chunks of this many trees, with progress and a cancel check between chunks.
FOREST_CHUNK = 25
#: In the test profile the SVM never sees more than this many rows, whatever the requested cap.
TEST_SVM_CAP = 1_000
#: Name of the final step of every channel pipeline.
MODEL_STEP = "model"


@dataclass(frozen=True)
class BuildContext:
    """What a builder needs to know: number of classes, random seed, size profile and the SVM row cap."""

    n_classes: int
    seed: int
    profile: Profile = "full"
    svm_cap: int = 20_000

    def __post_init__(self) -> None:
        """Reject impossible contexts early, with a readable message."""
        if int(self.n_classes) < 2:
            raise ValueError("A channel needs at least two classes to tell apart.")
        if self.profile not in PROFILES:
            raise ValueError(f"Unknown profile {self.profile!r}; choose 'full' or 'test'.")
        if int(self.svm_cap) < 1:
            raise ValueError("The SVM row cap must be at least 1.")

    @property
    def effective_svm_cap(self) -> int:
        """Rows the SVM may be trained on: the requested cap, lowered to 1,000 in the test profile."""
        cap = int(self.svm_cap)
        return min(cap, TEST_SVM_CAP) if self.profile == "test" else cap


@dataclass(frozen=True)
class ModelSpec:
    """Static description of one channel.

    Attributes:
        key: channel key (``forest``, ``xgboost``, ``svm``, ``mlp`` or ``logreg``).
        family: short family name for tables.
        description: one sentence on what the model does and how it is set up here.
        scaled: True when the pipeline imputes, log-compresses and standardises before the model.
        weight_cap: largest weight a single training row may receive from :func:`balanced_weights`.
    """

    key: str
    family: str
    description: str
    scaled: bool
    weight_cap: float


MODEL_SPECS: dict[str, ModelSpec] = {
    "forest": ModelSpec(
        "forest", "Tree ensemble",
        "150 fully grown trees (leaves of at least 2 rows), each on a weighted half-size bootstrap draw and "
        "choosing among sqrt(features) columns at every split; grown 25 trees at a time.",
        scaled=False, weight_cap=WEIGHT_CAP),
    "xgboost": ModelSpec(
        "xgboost", "Gradient boosting",
        "Up to 300 boosted trees of depth 8 (histogram method, learning rate 0.15, 80% row and column "
        "subsampling), stopped early when a 10% slice of the training rows stops improving.",
        scaled=False, weight_cap=WEIGHT_CAP),
    "svm": ModelSpec(
        "svm", "Kernel machine",
        "Support vector machine with a Gaussian (RBF) kernel, C = 10, on a capped rare-aware subset of the "
        "training rows; probabilities from a sigmoid fitted on separate training rows.",
        scaled=True, weight_cap=WEIGHT_CAP),
    "mlp": ModelSpec(
        "mlp", "Neural network",
        "Two hidden layers (128 and 64 units) trained with Adam in batches of 512, stopping when a 10% "
        "validation slice stops improving.",
        scaled=True, weight_cap=MLP_WEIGHT_CAP),
    "logreg": ModelSpec(
        "logreg", "Linear model",
        "Multinomial logistic regression (L-BFGS, C = 1) on log-compressed, standardised features: the "
        "linear baseline.",
        scaled=True, weight_cap=WEIGHT_CAP),
}


class ObservableMLP(MLPClassifier):
    """scikit-learn's :class:`~sklearn.neural_network.MLPClassifier`, plus a hook called after every training epoch.

    The trainer sets ``epoch_observer`` (a callable taking the model) just before ``fit`` and removes it right
    after, so a fitted model never holds it. The observer reports progress and raises to stop the fit when it is
    cancelled. Parameters, training and predictions are exactly those of ``MLPClassifier``: the hook runs after the
    library's own end-of-epoch bookkeeping (the early-stopping check), which is the one per-epoch step it exposes.
    """

    epoch_observer: Callable[[MLPClassifier], None] | None = None

    def _update_no_improvement_count(self, early_stopping: bool, X: Any, y: Any, sample_weight: Any) -> None:
        """The library's end-of-epoch bookkeeping, then the observer (when one is set)."""
        super()._update_no_improvement_count(early_stopping, X, y, sample_weight)
        observer = self.__dict__.get("epoch_observer")
        if observer is not None:
            observer(self)


class WholeSliceSplit:
    """A single "split" whose training and test parts are both every row.

    The SVM's probabilities are calibrated by ``CalibratedClassifierCV(FrozenEstimator(svm), cv=WholeSliceSplit())``.
    A frozen model is never refitted, so cross-validation folds would only repeat its decision values while
    demanding several rows of every class in every fold (and warning, or failing outright, when a rare class has
    fewer than five calibration rows). One split over all calibration rows gives the same decision values without
    that demand.
    """

    def split(self, X: Any, y: Any = None, groups: Any = None) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Yield one (train, test) pair of index arrays, both holding every row of ``X``."""
        rows = np.arange(len(X), dtype=np.int64)
        yield rows, rows

    def get_n_splits(self, X: Any = None, y: Any = None, groups: Any = None) -> int:
        """The number of splits: always 1."""
        return 1

    def __repr__(self) -> str:
        """``WholeSliceSplit()``."""
        return "WholeSliceSplit()"


def _scaled_steps() -> list[tuple[str, object]]:
    """Sanitise, fill gaps with the training median, log-compress and standardise."""
    return [
        ("sanitize", FlowSanitizer()),
        ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("log", make_signed_log()),
        ("scale", StandardScaler()),
    ]


def _build_forest(ctx: BuildContext) -> Pipeline:
    """Random forest; the fit grows it in warm-start chunks (see ``graticule.models.train``)."""
    trees = 150 if ctx.profile == "full" else 20
    model = RandomForestClassifier(
        n_estimators=trees, max_features="sqrt", min_samples_leaf=2, max_samples=0.5, bootstrap=True,
        n_jobs=-1, random_state=ctx.seed,
    )
    return Pipeline([("sanitize", FlowSanitizer()), (MODEL_STEP, model)])


def _build_xgboost(ctx: BuildContext) -> Pipeline:
    """Gradient-boosted trees with early stopping (the eval slice is chosen at fit time)."""
    full = ctx.profile == "full"
    binary = ctx.n_classes == 2
    model = XGBClassifier(
        tree_method="hist", max_depth=8, learning_rate=0.15, n_estimators=300 if full else 30,
        early_stopping_rounds=20 if full else 10, subsample=0.8, colsample_bytree=0.8, n_jobs=4,
        random_state=ctx.seed, objective="binary:logistic" if binary else "multi:softprob",
        eval_metric="logloss" if binary else "mlogloss",
    )
    return Pipeline([("sanitize", FlowSanitizer()), (MODEL_STEP, model)])


def _build_svm(ctx: BuildContext) -> Pipeline:
    """RBF support vector machine (row cap and probability calibration are applied at fit time)."""
    model = SVC(kernel="rbf", C=10.0, gamma="scale", cache_size=1024, random_state=ctx.seed)
    return Pipeline([*_scaled_steps(), (MODEL_STEP, model)])


def _build_mlp(ctx: BuildContext) -> Pipeline:
    """Multi-layer perceptron with early stopping (an :class:`ObservableMLP`, so its epochs can be followed)."""
    full = ctx.profile == "full"
    model = ObservableMLP(
        hidden_layer_sizes=(128, 64) if full else (32,), solver="adam", alpha=1e-4, batch_size=512,
        max_iter=60 if full else 15, early_stopping=True, n_iter_no_change=5, random_state=ctx.seed,
    )
    return Pipeline([*_scaled_steps(), (MODEL_STEP, model)])


def _build_logreg(ctx: BuildContext) -> Pipeline:
    """Logistic regression (no ``penalty`` or ``n_jobs`` arguments: both are deprecated in scikit-learn 1.9)."""
    model = LogisticRegression(solver="lbfgs", C=1.0, max_iter=1000 if ctx.profile == "full" else 200,
                               random_state=ctx.seed)
    return Pipeline([*_scaled_steps(), (MODEL_STEP, model)])


#: One builder per channel key. Tests replace entries here to simulate a channel that cannot be built.
BUILDERS: dict[str, Callable[[BuildContext], Pipeline]] = {
    "forest": _build_forest,
    "xgboost": _build_xgboost,
    "svm": _build_svm,
    "mlp": _build_mlp,
    "logreg": _build_logreg,
}


def build_estimator(key: str, ctx: BuildContext) -> Pipeline:
    """Return a fresh, unfitted pipeline for channel ``key`` (raises ``KeyError`` for an unknown key)."""
    if key not in BUILDERS:
        raise KeyError(f"Unknown channel {key!r}; choose one of {', '.join(MODEL_KEYS)}.")
    return BUILDERS[key](ctx)


def balanced_weights(y: npt.ArrayLike, cap: float) -> np.ndarray:
    """Balanced per-row weights with a per-row cap, summing to ``len(y)`` (float64).

    Uncapped, row ``i`` of class ``c`` weighs ``n / (K * n_c)``, so every class carries total weight ``n / K``.
    When that would exceed ``cap`` for a small class, the class's rows weigh exactly ``cap`` and the weight it gives
    up is shared equally among the remaining classes, repeating until no row exceeds the cap. This is "cap, then
    rescale to sum to n" solved exactly, so the cap still holds after rescaling. With classes of similar size
    nothing is capped and the weights are the plain balanced ones. ``cap`` must be at least 1 (a cap of 1 gives
    unit weights).
    """
    labels = np.asarray(y)
    n = labels.shape[0]
    if float(cap) < 1.0:
        raise ValueError("The weight cap must be at least 1.")
    if n == 0:
        return np.empty(0, dtype=np.float64)
    _, inverse, counts = np.unique(labels, return_inverse=True, return_counts=True)
    counts = counts.astype(np.float64)
    capped = np.zeros(counts.size, dtype=bool)
    while True:
        free = ~capped
        budget = n - float(cap) * float(counts[capped].sum())
        per_row = np.where(capped, float(cap), budget / (int(free.sum()) * counts))
        over = free & (per_row > float(cap))
        if not over.any():
            break
        capped |= over
    weights = per_row[inverse.reshape(-1)]
    return weights * (n / weights.sum())


def channel_weights(key: str, y: npt.ArrayLike, balanced: bool = True) -> np.ndarray:
    """Training weights for channel ``key``: capped balanced weights, or unit weights when ``balanced`` is False."""
    labels = np.asarray(y)
    if not balanced:
        return np.ones(labels.shape[0], dtype=np.float64)
    return balanced_weights(labels, MODEL_SPECS[key].weight_cap)


def capped_classes(y: npt.ArrayLike, cap: float) -> list[int]:
    """Classes (label values) whose rows hit the cap in :func:`balanced_weights`, smallest class first."""
    labels = np.asarray(y)
    if labels.shape[0] == 0:
        return []
    weights = balanced_weights(labels, cap)
    values, first = np.unique(labels, return_index=True)
    hit = [(int(np.sum(labels == v)), v) for v, i in zip(values, first) if weights[i] >= float(cap) * (1 - 1e-9)]
    return [v.item() if hasattr(v, "item") else v for _, v in sorted(hit, key=lambda t: t[0])]
