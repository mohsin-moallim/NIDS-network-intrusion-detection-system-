"""Cleaning steps for flow records: bad values, duplicate rows, conflicting labels and degenerate columns.

Every function here returns a new frame (inputs are never modified; frames may be shared by a cache) together
with a small report of what it changed, broken down by class where that helps to judge the effect.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

from graticule.schema import FEATURES, LABEL

NonFinitePolicy = Literal["drop", "impute", "recompute"]
ConflictPolicy = Literal["keep", "majority", "drop"]
CONFLICT_POLICIES: tuple[str, ...] = ("keep", "majority", "drop")

BYTES_PER_S = "Flow Bytes/s"
PACKETS_PER_S = "Flow Packets/s"
DURATION = "Flow Duration"
FWD_BYTES, BWD_BYTES = "Total Length of Fwd Packets", "Total Length of Bwd Packets"
FWD_PACKETS, BWD_PACKETS = "Total Fwd Packets", "Total Backward Packets"
# Flow Duration is recorded in microseconds; a zero (or negative) duration is floored to 1 µs when rates are rebuilt.
DURATION_FLOOR_US = 1.0
_HASH_CHUNK = 262_144


def _features_in(df: pd.DataFrame) -> list[str]:
    """Known feature columns present in ``df``, in catalogue order."""
    return [c for c in FEATURES if c in df.columns]


def _class_counts(labels: pd.Series | None, mask: np.ndarray | None = None) -> dict[str, int]:
    """Row counts per class for the rows selected by ``mask`` (all rows when ``mask`` is None)."""
    if labels is None:
        return {}
    chosen = labels if mask is None else labels[mask]
    return {str(k): int(v) for k, v in chosen.value_counts(sort=True).items()}


def _add_counts(a: dict[str, int], b: dict[str, int]) -> dict[str, int]:
    """Sum two count dictionaries, largest first."""
    out = dict(a)
    for key, value in b.items():
        out[key] = out.get(key, 0) + value
    return dict(sorted(out.items(), key=lambda kv: (-kv[1], kv[0])))


# --------------------------------------------------------------------------------------------------------------
# Infinite and missing values
# --------------------------------------------------------------------------------------------------------------
@dataclass
class NonFiniteReport:
    """Effect of the infinite/missing-value strategy: affected rows per class and bad cells per column."""

    strategy: str
    rows_before: int = 0
    rows_affected: int = 0
    rows_dropped: int = 0
    rows_after: int = 0
    by_class: dict[str, int] = field(default_factory=dict)
    by_column: dict[str, int] = field(default_factory=dict)
    infinite_cells: int = 0
    missing_cells: int = 0
    recomputed: dict[str, int] = field(default_factory=dict)
    rows_left_with_gaps: int = 0

    def combined(self, other: "NonFiniteReport") -> "NonFiniteReport":
        """Return the sum of this report and ``other`` (used to total the per-file reports)."""
        return NonFiniteReport(
            strategy=self.strategy,
            rows_before=self.rows_before + other.rows_before,
            rows_affected=self.rows_affected + other.rows_affected,
            rows_dropped=self.rows_dropped + other.rows_dropped,
            rows_after=self.rows_after + other.rows_after,
            by_class=_add_counts(self.by_class, other.by_class),
            by_column=_add_counts(self.by_column, other.by_column),
            infinite_cells=self.infinite_cells + other.infinite_cells,
            missing_cells=self.missing_cells + other.missing_cells,
            recomputed=_add_counts(self.recomputed, other.recomputed),
            rows_left_with_gaps=self.rows_left_with_gaps + other.rows_left_with_gaps,
        )


def nonfinite_mask(features: pd.DataFrame | np.ndarray) -> np.ndarray:
    """Boolean array marking rows that hold at least one infinite or missing value."""
    if isinstance(features, np.ndarray):
        values = features.reshape(len(features), -1)
        return ~np.isfinite(values).all(axis=1)
    mask = np.zeros(len(features), dtype=bool)
    for pos in range(features.shape[1]):
        column = features.iloc[:, pos]
        if not pd.api.types.is_numeric_dtype(column):
            column = pd.to_numeric(column, errors="coerce")
        values = column.to_numpy(dtype=np.float64 if column.dtype != np.float32 else np.float32, na_value=np.nan)
        mask |= ~np.isfinite(values)
    return mask


def _replace_infinities(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """Return ``df`` with ±inf turned into NaN in ``columns`` (only columns that hold an infinity are rewritten)."""
    out = df
    for column in columns:
        values = df[column].to_numpy()
        infinite = np.isinf(values)
        if infinite.any():
            if out is df:
                out = df.copy(deep=False)
            out[column] = np.where(infinite, np.float32(np.nan), values).astype(values.dtype, copy=False)
    return out


def _recompute_rates(df: pd.DataFrame, bad_cells: dict[str, np.ndarray]) -> tuple[pd.DataFrame, dict[str, int]]:
    """Rebuild non-finite Flow Bytes/s and Flow Packets/s cells from the byte/packet totals and the duration."""
    needed = {DURATION, FWD_BYTES, BWD_BYTES, FWD_PACKETS, BWD_PACKETS}
    if not needed.issubset(df.columns):
        return df, {}
    duration = np.maximum(df[DURATION].to_numpy(dtype=np.float64), DURATION_FLOOR_US)
    totals = {
        BYTES_PER_S: df[FWD_BYTES].to_numpy(dtype=np.float64) + df[BWD_BYTES].to_numpy(dtype=np.float64),
        PACKETS_PER_S: df[FWD_PACKETS].to_numpy(dtype=np.float64) + df[BWD_PACKETS].to_numpy(dtype=np.float64),
    }
    out = df
    rebuilt: dict[str, int] = {}
    for column, total in totals.items():
        bad = bad_cells.get(column)
        if column not in df.columns or bad is None or not bad.any():
            continue
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            fresh = (total / duration * 1e6).astype(np.float32)
        if out is df:
            out = df.copy(deep=False)
        current = df[column].to_numpy()
        out[column] = np.where(bad, fresh, current).astype(np.float32, copy=False)
        rebuilt[column] = int(bad.sum())
    return out, rebuilt


def apply_nonfinite_strategy(
    df: pd.DataFrame, strategy: NonFinitePolicy, *, label_col: str = LABEL
) -> tuple[pd.DataFrame, NonFiniteReport]:
    """Handle infinite and missing feature values.

    * ``drop``: remove every row holding a non-finite feature value.
    * ``impute``: keep the rows; ±inf becomes NaN, to be filled by each channel's median imputer (fitted on the
      training split only).
    * ``recompute``: rebuild non-finite ``Flow Bytes/s`` and ``Flow Packets/s`` cells from the byte and packet totals
      and ``Flow Duration`` (microseconds, floored at 1 µs); anything still non-finite becomes NaN, as in ``impute``.
      Finite rate values from the flow meter are left untouched.
    """
    if strategy not in ("drop", "impute", "recompute"):
        raise ValueError(f"Unknown strategy for infinite and missing values: {strategy!r}")
    columns = _features_in(df)
    labels = df[label_col] if label_col in df.columns else None
    report = NonFiniteReport(strategy=strategy, rows_before=len(df))
    row_mask = np.zeros(len(df), dtype=bool)
    bad_cells: dict[str, np.ndarray] = {}
    for column in columns:
        values = df[column].to_numpy()
        bad = ~np.isfinite(values)
        if bad.any():
            bad_cells[column] = bad
            row_mask |= bad
            n_inf = int(np.isinf(values).sum())
            report.infinite_cells += n_inf
            report.missing_cells += int(bad.sum()) - n_inf
            report.by_column[column] = int(bad.sum())
    report.by_column = dict(sorted(report.by_column.items(), key=lambda kv: (-kv[1], kv[0])))
    report.rows_affected = int(row_mask.sum())
    report.by_class = _class_counts(labels, row_mask)

    if strategy == "drop":
        out = df.loc[~row_mask] if report.rows_affected else df
        report.rows_dropped = report.rows_affected
    else:
        out = df
        if strategy == "recompute":
            out, report.recomputed = _recompute_rates(out, bad_cells)
        out = _replace_infinities(out, [c for c in columns if c in bad_cells])
        report.rows_left_with_gaps = int(nonfinite_mask(out[columns]).sum()) if bad_cells else 0
    report.rows_after = len(out)
    return out, report


# --------------------------------------------------------------------------------------------------------------
# Duplicate rows
# --------------------------------------------------------------------------------------------------------------
@dataclass
class DuplicateReport:
    """Rows removed as exact duplicates at one stage, per class."""

    stage: str
    rows_before: int = 0
    rows_removed: int = 0
    rows_after: int = 0
    by_class: dict[str, int] = field(default_factory=dict)

    def combined(self, other: "DuplicateReport") -> "DuplicateReport":
        """Return the sum of this report and ``other`` (used to total the per-file reports)."""
        return DuplicateReport(
            stage=self.stage,
            rows_before=self.rows_before + other.rows_before,
            rows_removed=self.rows_removed + other.rows_removed,
            rows_after=self.rows_after + other.rows_after,
            by_class=_add_counts(self.by_class, other.by_class),
        )


def _normalised_for_hash(column: pd.Series) -> pd.Series | np.ndarray:
    """Float columns: -0.0 becomes +0.0 and every non-finite value the same NaN, so equal readings hash equally."""
    if pd.api.types.is_float_dtype(column):
        values = column.to_numpy()
        zero = values.dtype.type(0)
        with np.errstate(invalid="ignore"):
            return np.where(np.isfinite(values), values + zero, values.dtype.type(np.nan))
    return column


def row_hashes(df: pd.DataFrame, columns: Sequence[str]) -> np.ndarray:
    """64-bit hash of each row over ``columns`` (``pd.util.hash_pandas_object`` with ``index=False``).

    Before hashing, -0.0 becomes +0.0 and ±inf/NaN become one NaN. Rows are processed in chunks to bound memory.
    Two different rows share a hash with probability about n²/2⁶⁵, negligible for these data sizes.
    """
    columns = list(columns)
    out = np.empty(len(df), dtype=np.uint64)
    for start in range(0, len(df), _HASH_CHUNK):
        part = df.iloc[start : start + _HASH_CHUNK]
        normalised = pd.DataFrame({c: _normalised_for_hash(part[c]) for c in columns}, copy=False)
        out[start : start + len(part)] = pd.util.hash_pandas_object(normalised, index=False).to_numpy()
    return out


def _plain_series(values: pd.Series | np.ndarray) -> pd.Series:
    """``values`` as a Series with a fresh 0..n-1 index, so it lines up with plain arrays."""
    if isinstance(values, pd.Series):
        return values.reset_index(drop=True)
    return pd.Series(np.asarray(values))


def hashes_with_labels(feature_hashes: np.ndarray, labels: pd.Series | np.ndarray) -> np.ndarray:
    """Combine per-row feature hashes with the labels into one 64-bit row hash (features + label identity)."""
    pair = pd.DataFrame({"h": pd.Series(np.asarray(feature_hashes, dtype=np.uint64)), "y": _plain_series(labels)})
    return pd.util.hash_pandas_object(pair, index=False).to_numpy()


def first_occurrences(hashes: np.ndarray) -> np.ndarray:
    """Boolean array that is True for the first row carrying each hash (the row a de-duplication keeps)."""
    return ~pd.Index(np.asarray(hashes, dtype=np.uint64)).duplicated(keep="first")


def copies_per_kept_row(hashes: np.ndarray, keep: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    """For each kept row, how many rows (or how much weight) shared its hash before de-duplication."""
    series = pd.Series(np.ones(len(hashes), dtype=np.int64) if weights is None else np.asarray(weights))
    totals = series.groupby(np.asarray(hashes, dtype=np.uint64), sort=False).sum()
    return totals.reindex(np.asarray(hashes)[keep]).to_numpy()


def drop_duplicates_by_hash(
    df: pd.DataFrame,
    columns: Sequence[str],
    stage_name: str,
    *,
    hashes: np.ndarray | None = None,
    label_col: str = LABEL,
) -> tuple[pd.DataFrame, DuplicateReport]:
    """Remove rows identical to an earlier row over ``columns`` (first occurrence kept); counts removals per class.

    ``hashes`` may be supplied when already computed for exactly these columns.
    """
    h = row_hashes(df, columns) if hashes is None else hashes
    keep = first_occurrences(h)
    labels = df[label_col] if label_col in df.columns else None
    removed = int((~keep).sum())
    report = DuplicateReport(stage=stage_name, rows_before=len(df), rows_removed=removed,
                             rows_after=len(df) - removed, by_class=_class_counts(labels, ~keep) if removed else {})
    return (df.loc[keep] if removed else df), report


# --------------------------------------------------------------------------------------------------------------
# Conflicting labels
# --------------------------------------------------------------------------------------------------------------
@dataclass
class ConflictReport:
    """Groups of rows with identical features but different labels, and what the chosen policy did with them."""

    groups: int = 0
    rows: int = 0
    by_class: dict[str, int] = field(default_factory=dict)
    policy: str = "keep"
    rows_removed: int = 0


def conflict_mask(feature_hashes: np.ndarray, labels: pd.Series | np.ndarray) -> tuple[np.ndarray, int]:
    """Rows whose feature hash is shared by rows with a different label, and the number of such groups."""
    frame = pd.DataFrame({"h": pd.Series(np.asarray(feature_hashes, dtype=np.uint64)), "y": _plain_series(labels)})
    distinct = frame.drop_duplicates()
    per_hash = distinct["h"].value_counts()
    conflicting = per_hash.index[per_hash.to_numpy() > 1]
    mask = frame["h"].isin(conflicting).to_numpy()
    return mask, int(len(conflicting))


def conflicting_feature_rows(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    *,
    label_col: str = LABEL,
    feature_hashes: np.ndarray | None = None,
) -> ConflictReport:
    """Count groups of rows that are identical over ``feature_cols`` but carry different labels."""
    h = row_hashes(df, feature_cols) if feature_hashes is None else feature_hashes
    labels = df[label_col]
    mask, groups = conflict_mask(h, labels)
    return ConflictReport(groups=groups, rows=int(mask.sum()), by_class=_class_counts(labels, mask))


def conflict_keep_mask(
    feature_hashes: np.ndarray,
    labels: pd.Series | np.ndarray,
    policy: ConflictPolicy,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """Which rows survive a conflict policy.

    ``keep`` keeps everything. ``drop`` removes every row of a conflicting group. ``majority`` keeps, in each
    conflicting group, the rows carrying the label with the largest total weight (``weights`` default to 1 per row,
    e.g. the number of copies seen before de-duplication); a group whose top labels tie has no majority and is
    removed entirely.
    """
    if policy not in CONFLICT_POLICIES:
        raise ValueError(f"Unknown conflict policy: {policy!r}")
    n = len(feature_hashes)
    if policy == "keep" or n == 0:
        return np.ones(n, dtype=bool)
    mask, groups = conflict_mask(feature_hashes, labels)
    if groups == 0:
        return np.ones(n, dtype=bool)
    if policy == "drop":
        return ~mask
    frame = pd.DataFrame({
        "h": np.asarray(feature_hashes, dtype=np.uint64)[mask],
        "y": np.asarray(labels, dtype=object)[mask],
        "w": (np.ones(n) if weights is None else np.asarray(weights, dtype=np.float64))[mask],
    })
    totals = frame.groupby(["h", "y"], sort=False)["w"].sum().reset_index()
    best = totals.groupby("h")["w"].transform("max")
    leaders = totals[totals["w"] == best]
    unique_leader = leaders[~leaders["h"].duplicated(keep=False)]
    winners = set(zip(unique_leader["h"].to_numpy().tolist(), unique_leader["y"].tolist()))
    survives = np.fromiter(((h, y) in winners for h, y in zip(frame["h"].tolist(), frame["y"].tolist())),
                           dtype=bool, count=len(frame))
    keep = np.ones(n, dtype=bool)
    keep[np.flatnonzero(mask)] = survives
    return keep


def resolve_conflicts(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    policy: ConflictPolicy,
    *,
    label_col: str = LABEL,
    weights: np.ndarray | None = None,
) -> tuple[pd.DataFrame, ConflictReport]:
    """Apply a conflict policy (``keep``, ``majority`` or ``drop``; see :func:`conflict_keep_mask`)."""
    h = row_hashes(df, feature_cols)
    report = conflicting_feature_rows(df, feature_cols, label_col=label_col, feature_hashes=h)
    report.policy = policy
    keep = conflict_keep_mask(h, df[label_col], policy, weights)
    report.rows_removed = int((~keep).sum())
    return (df.loc[keep] if report.rows_removed else df), report


# --------------------------------------------------------------------------------------------------------------
# Degenerate columns
# --------------------------------------------------------------------------------------------------------------
@dataclass
class DegenerateReport:
    """Columns that carry no information in the data at hand: constant ones and exact copies of earlier ones."""

    constant: list[str] = field(default_factory=list)
    duplicate_of: dict[str, str] = field(default_factory=dict)

    @property
    def excluded(self) -> list[str]:
        """Every column to leave out of the "all numeric" feature set.

        For other feature sets pass the report itself to :func:`graticule.features.select_features`: a copied
        column should only be left out when the column it copies is chosen too (a curated set may hold the copy
        without the original).
        """
        return list(self.constant) + list(self.duplicate_of)


def _is_constant(values: np.ndarray) -> bool:
    """One distinct value, where NaN counts as a value of its own."""
    if len(values) == 0:
        return True
    missing = np.isnan(values)
    n_missing = int(missing.sum())
    if n_missing == len(values):
        return True
    if n_missing:
        return False
    return bool(values.min() == values.max())


def find_degenerate_columns(df: pd.DataFrame, columns: Iterable[str]) -> DegenerateReport:
    """Detect constant columns and columns identical to an earlier column (NaN equal to NaN)."""
    report = DegenerateReport()
    signatures: dict[tuple, list[str]] = {}
    for column in columns:
        values = df[column].to_numpy(dtype=np.float64, na_value=np.nan)
        if _is_constant(values):
            report.constant.append(column)
            continue
        finite = values[np.isfinite(values)]
        signature = (int(np.isnan(values).sum()), float(finite.sum()) if len(finite) else 0.0,
                     float(finite.min()) if len(finite) else 0.0, float(finite.max()) if len(finite) else 0.0)
        for earlier in signatures.get(signature, []):
            other = df[earlier].to_numpy(dtype=np.float64, na_value=np.nan)
            if np.array_equal(values, other, equal_nan=True):
                report.duplicate_of[column] = earlier
                break
        else:
            signatures.setdefault(signature, []).append(column)
    return report
