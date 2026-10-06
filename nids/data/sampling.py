"""Classes, targets, the rare-aware sampler and the stratified train/test split.

The sampler keeps rare attack classes visible: every class first receives up to a floor of rows, and only the rest
of the row budget is shared in proportion to class size. The split guarantees at least two training and two test
rows for every class, so every class can be both learned and measured.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

from nids.schema import ATTACK, BENIGN, NORMAL, WEB_ATTACK, is_normal_traffic

Mode = Literal["binary", "multiclass"]
HARD_FLOOR = 10
MIN_FLOOR_ROWS = 1_000
FLOOR_SHARE = 0.02


class SingleClassError(ValueError):
    """Raised when fewer than two classes remain for the chosen mode; the message is written for the user."""

    def __init__(self, message: str, present: Sequence[str] = (), dropped: dict[str, int] | None = None) -> None:
        super().__init__(message)
        self.present = list(present)
        self.dropped = dict(dropped or {})


# --------------------------------------------------------------------------------------------------------------
# Classes and targets
# --------------------------------------------------------------------------------------------------------------
def apply_class_options(labels: pd.Series, *, merge_web_attacks: bool) -> pd.Series:
    """Return the class labels to train on: with ``merge_web_attacks`` the three Web Attack types become one."""
    if not merge_web_attacks:
        return labels
    is_web = labels.str.startswith(WEB_ATTACK).fillna(False).to_numpy(dtype=bool)
    if not is_web.any():
        return labels
    merged = labels.to_numpy(dtype=object, na_value="").copy()
    merged[is_web] = WEB_ATTACK
    return pd.Series(merged, index=labels.index, name=labels.name, dtype="str")


def class_order(classes: Sequence[str]) -> list[str]:
    """Canonical class order: normal traffic first, then the rest alphabetically (defines label codes 0..K-1)."""
    unique = sorted(set(classes), key=lambda c: (not is_normal_traffic(c), c.lower(), c))
    return unique


@dataclass
class TargetResult:
    """The target for one detection mode: which rows take part, their class names and codes, and what was left out."""

    mode: str
    target: pd.Series
    keep: np.ndarray
    classes: list[str]
    codes: np.ndarray
    counts: dict[str, int]
    dropped: dict[str, int] = field(default_factory=dict)
    dropped_reasons: dict[str, str] = field(default_factory=dict)

    @property
    def n_classes(self) -> int:
        """Number of classes in the target."""
        return len(self.classes)


def _is_normal(name: str) -> bool:
    """True for the normal-traffic class under either naming (``BENIGN`` or ``Normal``, any letter case)."""
    return is_normal_traffic(name)


def _too_few_remedy(reasons: dict[str, str], names: list[str]) -> str:
    """What to change so classes left out for having too few rows can take part."""
    below_minimum = any("minimum" in reasons.get(name, "") for name in names)
    if below_minimum:
        return ("Lower the minimum class count (it applies in multi-class mode), or add files with more of these "
                "flows.")
    return "Every class needs at least the hard floor of rows; add files with more of these flows."


def _single_class_message(labels: pd.Series, present: list[str], dropped: dict[str, int],
                          reasons: dict[str, str]) -> str:
    """User-facing explanation when fewer than two classes are left.

    It separates data that truly holds one kind of traffic (add other files) from data where the other kind is
    present but was left out for having too few rows (lower the minimum, or add more of those flows).
    """
    if not present:
        if len(labels) == 0:
            text = "The sample holds no rows to fit on."
        else:
            text = f"No class has enough rows to fit on. {_too_few_remedy(reasons, list(dropped))}"
    elif dropped:
        # Another class is in the data but was left out for having too few rows: say so, and how to keep it.
        if _is_normal(present[0]):
            kind, names = "attack flows", list(dropped)
        elif any(_is_normal(name) for name in dropped):
            kind, names = f"normal ({BENIGN}) flows", [name for name in dropped if _is_normal(name)]
        else:
            kind, names = "other classes", list(dropped)
        n_rows = sum(dropped[name] for name in names)
        text = (f"Only one class has enough rows: {present[0]}. The sample does hold {kind}, but too few to fit on "
                f"({n_rows:,} rows). {_too_few_remedy(reasons, names)}")
    elif _is_normal(present[0]):
        text = (f"Only one class in this sample: every row is {BENIGN}. A detector needs attack flows to measure "
                "against. Add a file that contains attacks, or draw a synthetic sample.")
    else:
        who = present[0] if present[0] != ATTACK else "an attack"
        text = (f"Only one class in this sample: every row is {who}. A detector needs normal ({BENIGN}) flows as "
                "well. Add a file with normal traffic, or draw a synthetic sample.")
    if dropped:
        parts = [f"{name} ({count:,} rows, {reasons.get(name, 'too few rows')})" for name, count in dropped.items()]
        text += " Left out: " + "; ".join(parts) + "."
    return text


def target_for_mode(
    labels: pd.Series, mode: Mode, *, min_class_count: int, hard_floor: int = HARD_FLOOR
) -> TargetResult:
    """Build the target for ``mode``.

    ``binary`` maps normal traffic (BENIGN, or a label already reading "Normal", in any letter case) to "Normal"
    and every other label to "Attack". ``multiclass`` keeps the labels as they
    are and leaves out classes with fewer than ``min_class_count`` rows. In both modes a class with fewer than
    ``hard_floor`` rows is left out. Raises :class:`SingleClassError` when fewer than two classes remain.
    """
    if mode not in ("binary", "multiclass"):
        raise ValueError(f"Unknown mode: {mode!r}")
    values = pd.Series(labels.to_numpy(dtype=object, na_value=""), index=labels.index)
    if mode == "binary":
        lookup = {v: (NORMAL if is_normal_traffic(str(v)) else ATTACK) for v in values.unique()}
        series = values.map(lookup).astype("str")
        threshold = hard_floor
    else:
        series = values.astype("str")
        threshold = max(int(min_class_count), hard_floor)
    counts = series.value_counts()
    dropped: dict[str, int] = {}
    reasons: dict[str, str] = {}
    for name, count in counts.items():
        if count < threshold:
            dropped[str(name)] = int(count)
            if count < hard_floor:
                reasons[str(name)] = f"below the hard floor of {hard_floor} rows per class"
            else:
                reasons[str(name)] = f"below the minimum of {threshold} rows per class"
    present = class_order([str(c) for c in counts.index if str(c) not in dropped])
    if len(present) < 2:
        raise SingleClassError(_single_class_message(labels, present, dropped, reasons), present, dropped)
    keep = ~series.isin(list(dropped)).to_numpy(dtype=bool)
    target = series[keep]
    lookup = {name: code for code, name in enumerate(present)}
    codes = target.map(lookup).to_numpy(dtype=np.int64)
    kept_counts = {name: int(counts[name]) for name in present}
    return TargetResult(mode=mode, target=target, keep=keep, classes=present, codes=codes, counts=kept_counts,
                        dropped=dropped, dropped_reasons=reasons)


# --------------------------------------------------------------------------------------------------------------
# Rare-aware sampling
# --------------------------------------------------------------------------------------------------------------
@dataclass
class SamplingReport:
    """How the row budget was shared between classes, with per-class counts before and after."""

    budget: int
    floor: int
    floor_shrunk: bool
    sampled: bool
    rows_before: int
    rows_after: int
    before: dict[str, int]
    after: dict[str, int]
    seed: int = 0

    @property
    def rows_removed(self) -> int:
        """Rows left out by sampling."""
        return self.rows_before - self.rows_after


def _factorize(values: pd.Series | np.ndarray | Sequence[object]) -> tuple[np.ndarray, list[object]]:
    """Integer code per row plus the distinct values, in order of first appearance."""
    if isinstance(values, pd.Series):
        codes, uniques = pd.factorize(values, sort=False)
    else:
        codes, uniques = pd.factorize(pd.Series(np.asarray(values, dtype=object)), sort=False)
    if (codes < 0).any():
        raise ValueError("Class labels must not be missing.")
    return np.asarray(codes), list(uniques)


def default_floor(budget: int) -> int:
    """Rows every class receives before the rest is shared: max(1,000, 2% of the budget)."""
    return int(max(MIN_FLOOR_ROWS, round(FLOOR_SHARE * budget)))


def _largest_remainder(total: int, weights: dict[str, float]) -> dict[str, int]:
    """Split ``total`` into integers proportional to ``weights`` (largest-remainder rounding, ties by name)."""
    weight_sum = float(sum(weights.values()))
    if total <= 0 or weight_sum <= 0:
        return {k: 0 for k in weights}
    raw = {k: total * w / weight_sum for k, w in weights.items()}
    base = {k: int(np.floor(v)) for k, v in raw.items()}
    left = total - sum(base.values())
    order = sorted(weights, key=lambda k: (-(raw[k] - base[k]), k))
    for k in order[:left]:
        base[k] += 1
    return base


def allocate_quotas(counts: dict[str, int], budget: int, floor: int | None = None) -> tuple[dict[str, int], int, bool]:
    """Rows to take from each class so the total is exactly ``budget`` (or everything when the data is smaller).

    Each class first gets min(rows, floor). When floor × classes exceeds the budget, the floor shrinks to
    budget // classes, so small classes still keep all their rows. The remaining budget is shared in proportion to
    each class's rows not yet taken. Returns (quotas, floor used, whether the floor shrank).
    """
    counts = {k: int(v) for k, v in counts.items() if int(v) > 0}
    total = sum(counts.values())
    floor_used = default_floor(budget) if floor is None else int(floor)
    if total <= budget:
        return dict(counts), floor_used, False
    shrunk = False
    if floor_used * len(counts) > budget:
        floor_used = budget // len(counts)
        shrunk = True
    quotas = {k: min(n, floor_used) for k, n in counts.items()}
    remainder = budget - sum(quotas.values())
    spare = {k: counts[k] - quotas[k] for k in counts if counts[k] > quotas[k]}
    extra = _largest_remainder(remainder, {k: float(v) for k, v in spare.items()})
    for k, add in extra.items():
        quotas[k] += min(add, spare[k])
    return quotas, floor_used, shrunk


def sample_positions(
    labels: pd.Series | np.ndarray, budget: int, seed: int, floor: int | None = None
) -> tuple[np.ndarray, SamplingReport]:
    """Choose row positions for a rare-aware sample of at most ``budget`` rows (sorted, deterministic for a seed)."""
    codes, uniques = _factorize(labels)
    names = [str(u) for u in uniques]
    before = {names[i]: int(n) for i, n in enumerate(np.bincount(codes, minlength=len(names)))}
    quotas, floor_used, shrunk = allocate_quotas(before, int(budget), floor)
    rng = np.random.default_rng(seed)
    chosen: list[np.ndarray] = []
    for name in sorted(names):
        positions = np.flatnonzero(codes == names.index(name))
        take = quotas.get(name, 0)
        if take >= len(positions):
            chosen.append(positions)
        elif take > 0:
            chosen.append(rng.choice(positions, size=take, replace=False))
    picked = np.sort(np.concatenate(chosen)) if chosen else np.empty(0, dtype=np.int64)
    after = {name: int(quotas.get(name, 0)) for name in before}
    order = sorted(before, key=lambda k: (-before[k], k))
    report = SamplingReport(
        budget=int(budget), floor=floor_used, floor_shrunk=shrunk, sampled=len(picked) < len(codes),
        rows_before=len(codes), rows_after=len(picked), before={k: before[k] for k in order},
        after={k: after[k] for k in order}, seed=int(seed),
    )
    return picked.astype(np.int64), report


def rare_aware_sample(
    df: pd.DataFrame, label_col: str, budget: int, seed: int, floor: int | None = None
) -> tuple[pd.DataFrame, SamplingReport]:
    """Draw a rare-aware sample of ``df`` (see :func:`allocate_quotas`); row order is preserved."""
    positions, report = sample_positions(df[label_col], budget, seed, floor)
    return df.iloc[positions], report


# --------------------------------------------------------------------------------------------------------------
# Stratified split
# --------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class SplitIndices:
    """Row positions of the training and test parts of a split (both sorted)."""

    train: np.ndarray
    test: np.ndarray


def stratified_split(
    y: pd.Series | np.ndarray | Sequence[object] | int, test_share: float, seed: int, *, min_rows: int = 2
) -> SplitIndices:
    """Split rows into train and test so that every class has at least ``min_rows`` rows on each side.

    ``y`` holds the class of each row; an integer means that many rows without classes. Each class sends
    round(n × test_share) rows to the test part, clamped to leave ``min_rows`` on both sides. Raises ``ValueError``
    naming the class when a class is too small to split.
    """
    if not 0.0 < float(test_share) < 1.0:
        raise ValueError("The test share must be between 0 and 1.")
    codes, uniques = _factorize(np.zeros(int(y), dtype=np.int64) if isinstance(y, (int, np.integer)) else y)
    names = [str(u) for u in uniques]
    rng = np.random.default_rng(seed)
    train_parts: list[np.ndarray] = []
    test_parts: list[np.ndarray] = []
    too_small = [(names[i], int(n)) for i, n in enumerate(np.bincount(codes, minlength=len(names))) if n < 2 * min_rows]
    if too_small:
        detail = ", ".join(f"{name} has {n}" for name, n in too_small)
        raise ValueError(f"Some classes are too small to split ({detail}); each needs at least {2 * min_rows} rows: "
                         f"{min_rows} to train and {min_rows} to test.")
    for name in sorted(names):
        positions = np.flatnonzero(codes == names.index(name))
        n = len(positions)
        n_test = int(min(max(round(n * float(test_share)), min_rows), n - min_rows))
        shuffled = rng.permutation(positions)
        test_parts.append(shuffled[:n_test])
        train_parts.append(shuffled[n_test:])
    train = np.sort(np.concatenate(train_parts)) if train_parts else np.empty(0, dtype=np.int64)
    test = np.sort(np.concatenate(test_parts)) if test_parts else np.empty(0, dtype=np.int64)
    return SplitIndices(train=train.astype(np.int64), test=test.astype(np.int64))
