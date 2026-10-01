"""The 01 Sample procedure: read, clean, de-duplicate and sample flow records into a :class:`PreparedDataset`.

Order of work (nothing here looks at a train/test split, so nothing can leak from test data):

1. per file: read and normalise labels (:func:`read_source_file`), then work out which rows survive the
   infinite/missing-value strategy and the removal of duplicates within the file (:func:`stage_rows`);
2. remove duplicates across files (the earliest file keeps its copy; files are taken in capture order);
3. optionally merge the three Web Attack classes (the detailed label stays on every row);
4. count rows with identical features but different labels, and apply the conflict policy (default: keep);
5. draw a rare-aware sample.

The per-file steps never copy a file's rows: :func:`stage_rows` returns row positions plus hashes, and only the
sampled rows are copied out of the file frames at the end. The value repairs of the impute and rebuild strategies
are applied to those copied rows then; every repair works row by row, so the result is the same as repairing the
whole file first. This keeps the file frames strategy-independent, so the UI can cache one read per file and reuse
it for any strategy. Frames passed in may be shared and are never modified here.
"""

from __future__ import annotations

import ctypes
import hashlib
import importlib
import inspect
import json
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from graticule.data.clean import (
    CONFLICT_POLICIES,
    ConflictReport,
    DegenerateReport,
    DuplicateReport,
    NonFiniteReport,
    apply_nonfinite_strategy,
    conflict_keep_mask,
    conflict_mask,
    copies_per_kept_row,
    find_degenerate_columns,
    first_occurrences,
    hashes_with_labels,
    row_hashes,
)
from graticule.data.reader import READER_VERSION, DataFileError, FileReadReport, normalize_labels, read_flow_csv
from graticule.data.sampling import (
    SamplingReport,
    SingleClassError,
    apply_class_options,
    class_order,
    sample_positions,
    target_for_mode,
)
from graticule.schema import EXPECTED_BY_NAME, EXPECTED_FILES, FEATURES, LABEL, is_benign
from graticule.settings import NONFINITE_STRATEGIES

# Part of the UI cache key for per-file results: bump when the per-file stage changes.
STAGE_VERSION = f"{READER_VERSION}.2"
FILE_COL = "_file"
ROW_COL = "_row"
SYNTHETIC_FILE = "synthetic"
SourceKind = Literal["cicids", "synthetic"]
ProgressFn = Callable[[str, float], None]
CLASS_TABLE_COLUMNS: tuple[str, ...] = ("Class", "Kind", "Available", "In sample", "Share of sample")


def file_order_key(name: str) -> tuple[int, str]:
    """Capture order for the eight known files (Monday first); other files follow by name."""
    order = [f.name.lower() for f in EXPECTED_FILES]
    lowered = name.lower()
    return (order.index(lowered) if lowered in order else len(order), lowered)


@dataclass(frozen=True)
class DataRequest:
    """Everything that decides the prepared dataset. Equal requests on the same files give identical results."""

    source: SourceKind = "cicids"
    data_dir: str | None = None
    files: tuple[str, ...] = ()
    row_budget: int = 200_000
    nonfinite_strategy: str = "drop"
    merge_web_attacks: bool = False
    seed: int = 42
    synthetic_flows: int = 40_000
    synthetic_attack_share: float = 0.35
    conflict_policy: str = "keep"

    def __post_init__(self) -> None:
        """Validate the request and put the file list in canonical (capture) order."""
        if self.source not in ("cicids", "synthetic"):
            raise ValueError(f"Unknown data source: {self.source!r}")
        if self.nonfinite_strategy not in NONFINITE_STRATEGIES:
            raise ValueError(f"Unknown strategy for infinite and missing values: {self.nonfinite_strategy!r}")
        if self.conflict_policy not in CONFLICT_POLICIES:
            raise ValueError(f"Unknown conflict policy: {self.conflict_policy!r}")
        if int(self.row_budget) < 1:
            raise ValueError("The row budget must be at least 1.")
        if self.source == "synthetic":
            object.__setattr__(self, "files", ())
            object.__setattr__(self, "data_dir", None)
        else:
            object.__setattr__(self, "files", tuple(sorted(set(self.files), key=file_order_key)))

    def fingerprint_fields(self) -> dict[str, Any]:
        """The request as a plain dict for fingerprints and manifests. The folder path is left out on purpose:
        moving the data folder does not change the data."""
        fields_ = asdict(self)
        fields_.pop("data_dir", None)
        fields_["files"] = list(self.files)
        return fields_

    def describe(self) -> str:
        """One line naming the source and options, e.g. for captions and reports."""
        if self.source == "synthetic":
            where = f"{self.synthetic_flows:,} synthetic flows ({self.synthetic_attack_share:.0%} attacks)"
        else:
            where = f"{len(self.files)} CIC-IDS2017 file{'s' if len(self.files) != 1 else ''}"
        merge = ", Web Attack types merged" if self.merge_web_attacks else ""
        return (f"{where}; bad values: {self.nonfinite_strategy}; conflicts: {self.conflict_policy}{merge}; "
                f"budget {self.row_budget:,} rows; seed {self.seed}")


@dataclass
class FileStageReport:
    """What the per-file stage did to one file: reading, bad values and duplicates within the file.

    ``seconds`` is the time the bad-value and duplicate steps took when they were computed (reading is timed in
    ``read.seconds``). Both describe that first computation; the time a particular draw spent on the file is in
    :attr:`PreparedDataset.file_timings`.
    """

    name: str
    read: FileReadReport
    nonfinite: NonFiniteReport
    duplicates: DuplicateReport
    rows_out: int
    seconds: float


@dataclass(frozen=True)
class FileStage:
    """Which rows of one file survive the per-file steps, held as positions rather than as a copy of the rows.

    ``positions`` are ascending row positions (``iloc``) in the frame as read; ``feature_hashes`` the 64-bit hash
    of each surviving row's features after the strategy's repairs; ``copies`` how many identical rows (features and
    label) the file held for each surviving row. Treat the arrays as read-only: the UI caches and shares them.
    """

    positions: np.ndarray
    feature_hashes: np.ndarray
    copies: np.ndarray
    report: FileStageReport


@dataclass(frozen=True)
class FileTiming:
    """Wall time one source took in a particular draw, and whether its read was reused from an earlier draw."""

    seconds: float
    reused: bool


FileReader = Callable[[Path], tuple[pd.DataFrame, FileReadReport]]
FileStager = Callable[[Path, pd.DataFrame, FileReadReport, str], FileStage]


def read_source_file(path: Path) -> tuple[pd.DataFrame, FileReadReport]:
    """Read one file to sample from: every one of the 77 features and a ``Label`` column are required.

    The frame is what :func:`~graticule.data.reader.read_flow_csv` returns (index = 0-based data-row position in
    the file). It does not depend on the bad-value strategy, so a cache can share it between draws; treat it as
    read-only. Raises :class:`~graticule.data.reader.DataFileError` when features are missing.
    """
    frame, report = read_flow_csv(Path(path), require_label=True)
    if report.missing_features:
        shown = ", ".join(report.missing_features[:4])
        raise DataFileError(
            f"{report.name} lacks {len(report.missing_features)} of the {len(FEATURES)} flow features "
            f"(for example {shown}), so it does not look like a CIC-IDS2017 MachineLearningCSV file."
        )
    return frame, report


def _nonfinite_rows(frame: pd.DataFrame) -> np.ndarray:
    """Rows holding an infinite or missing feature value (column by column, without copying the frame)."""
    mask = np.zeros(len(frame), dtype=bool)
    for feature in FEATURES:
        mask |= ~np.isfinite(frame[feature].to_numpy())
    return mask


@dataclass(frozen=True)
class RowHashes:
    """The identity of every row of one file as read, which no bad-value strategy changes (except ``recompute``
    for the few rows it rebuilds): ``features`` hashes the 77 features (-0.0 as +0.0, every infinite or missing value
    alike), ``rows`` hashes the features together with the label. Both are uint64 arrays, one entry per row."""

    features: np.ndarray
    rows: np.ndarray


def hash_rows(frame: pd.DataFrame) -> RowHashes:
    """Hash every row of a frame as read (see :class:`RowHashes`); the slow part of the per-file stage."""
    features = row_hashes(frame, FEATURES)
    return RowHashes(features=features, rows=hashes_with_labels(features, frame[LABEL]))


def stage_rows(frame: pd.DataFrame, read_report: FileReadReport, strategy: str, *,
               hashes: RowHashes | None = None) -> FileStage:
    """Apply the bad-value strategy and the within-file de-duplication to one file, without copying its rows.

    Only the rows holding bad values are copied, to count them and (for ``recompute``) to hash their rebuilt
    values. Under ``drop`` those rows are left out; under ``impute`` and ``recompute`` they stay, and their repairs
    are applied when the sampled rows are copied out (see the module notes). ``hashes`` may pass the result of
    :func:`hash_rows` for this frame (a cache does, so changing strategy does not hash the file again); every hash is
    a function of its own row only, so a subset of rows keeps its hashes. ``frame`` is not modified.
    """
    if strategy not in NONFINITE_STRATEGIES:
        raise ValueError(f"Unknown strategy for infinite and missing values: {strategy!r}")
    started = time.perf_counter()
    bad = _nonfinite_rows(frame)
    bad_positions = np.flatnonzero(bad)
    repaired, bad_report = apply_nonfinite_strategy(frame.iloc[bad_positions], strategy)  # type: ignore[arg-type]
    nonfinite = replace(bad_report, rows_before=len(frame), rows_after=len(frame) - bad_report.rows_dropped)

    known = hashes if hashes is not None else hash_rows(frame)
    if len(known.features) != len(frame) or len(known.rows) != len(frame):
        raise ValueError("The row hashes do not belong to this frame.")
    feature_hashes, full_hashes = known.features, known.rows
    if strategy == "recompute" and len(bad_positions):
        # Rebuilt rates change these rows' values, so they get fresh hashes (on copies: the inputs may be shared).
        fresh = row_hashes(repaired, FEATURES)
        feature_hashes, full_hashes = feature_hashes.copy(), full_hashes.copy()
        feature_hashes[bad_positions] = fresh
        full_hashes[bad_positions] = hashes_with_labels(fresh, frame[LABEL].iloc[bad_positions])
    del repaired
    candidates = np.flatnonzero(~bad) if strategy == "drop" else np.arange(len(frame))
    candidate_hashes = full_hashes[candidates]
    keep = first_occurrences(candidate_hashes)
    copies = copies_per_kept_row(candidate_hashes, keep).astype(np.int32)
    removed = int((~keep).sum())
    removed_labels = frame[LABEL].iloc[candidates[~keep]] if removed else None
    duplicates = DuplicateReport(
        stage="within file", rows_before=len(candidates), rows_removed=removed, rows_after=len(candidates) - removed,
        by_class=({str(k): int(v) for k, v in removed_labels.value_counts(sort=True).items()}
                  if removed_labels is not None else {}),
    )
    positions = candidates[keep].astype(np.int64)
    report = FileStageReport(name=read_report.name, read=read_report, nonfinite=nonfinite, duplicates=duplicates,
                             rows_out=len(positions), seconds=time.perf_counter() - started)
    return FileStage(positions=positions, feature_hashes=feature_hashes[positions], copies=copies, report=report)


def load_file_stage(path: Path, strategy: str) -> tuple[pd.DataFrame, FileStage]:
    """Read one file and run the per-file steps on it (no caching): the frame as read plus its :class:`FileStage`."""
    frame, read_report = read_source_file(Path(path))
    return frame, stage_rows(frame, read_report, strategy)


def _stage_without_cache(path: Path, frame: pd.DataFrame, read_report: FileReadReport, strategy: str) -> FileStage:
    """Default :data:`FileStager`: run :func:`stage_rows` (the path only matters to caching stagers)."""
    return stage_rows(frame, read_report, strategy)


def _load_generator() -> Callable[..., pd.DataFrame]:
    """The synthetic flow generator (imported lazily so this module works before the generator exists)."""
    try:
        module = importlib.import_module("graticule.data.synthetic")
    except ImportError as exc:
        raise DataFileError("The synthetic flow generator is not available in this build.") from exc
    return module.generate


def _accepts_progress(function: Callable[..., Any]) -> bool:
    """True when ``function`` takes a ``progress`` keyword (the real generator does; stand-ins may not)."""
    try:
        parameters = inspect.signature(function).parameters
    except (TypeError, ValueError):
        return False
    return "progress" in parameters or any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values())


def _conform_generated(raw: pd.DataFrame) -> tuple[pd.DataFrame, FileReadReport]:
    """Give generated flows the same shape as a file read by :func:`read_flow_csv`."""
    started = time.perf_counter()
    missing = [f for f in FEATURES if f not in raw.columns]
    if missing or LABEL not in raw.columns:
        raise DataFileError(f"The synthetic generator returned an unexpected table (missing: {missing[:4]}).")
    block = np.empty((len(FEATURES), len(raw)), dtype=np.float32)
    for i, feature in enumerate(FEATURES):
        block[i] = raw[feature].to_numpy(dtype=np.float32, na_value=np.nan)
    frame = pd.DataFrame(block.T, columns=list(FEATURES), copy=False)
    labels = normalize_labels(raw[LABEL].reset_index(drop=True))
    frame[LABEL] = labels
    empty = labels.eq("").to_numpy()
    if empty.any():
        frame = frame.loc[~empty]
    counts = frame[LABEL].value_counts()
    report = FileReadReport(
        name=SYNTHETIC_FILE, rows_read=len(raw), rows_kept=len(frame), encoding="n/a", engine="generator",
        seconds=time.perf_counter() - started, empty_labels=int(empty.sum()), has_label=True,
        label_counts={str(k): int(v) for k, v in counts.items()},
    )
    return frame, report


# --------------------------------------------------------------------------------------------------------------
# Peak memory (Windows working set)
# --------------------------------------------------------------------------------------------------------------
class _MemoryCounters(ctypes.Structure):
    """PROCESS_MEMORY_COUNTERS from psapi.h."""

    _fields_ = [
        ("cb", ctypes.c_uint32),
        ("PageFaultCount", ctypes.c_uint32),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
    ]


_PSAPI: Any = None


def _psapi_reader() -> Callable[[], tuple[int, int]] | None:
    """A function returning (working set, peak working set) in bytes, or None off Windows or on failure."""
    global _PSAPI
    if sys.platform != "win32":
        return None
    if _PSAPI is None:
        try:
            from ctypes import wintypes

            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            psapi = ctypes.WinDLL("psapi", use_last_error=True)
            kernel32.GetCurrentProcess.restype = wintypes.HANDLE
            kernel32.GetCurrentProcess.argtypes = []
            psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(_MemoryCounters), wintypes.DWORD]
            psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
            _PSAPI = (kernel32, psapi)
        except (OSError, AttributeError):
            _PSAPI = False
    if not _PSAPI:
        return None
    kernel32, psapi = _PSAPI

    def read() -> tuple[int, int]:
        counters = _MemoryCounters()
        counters.cb = ctypes.sizeof(_MemoryCounters)
        if not psapi.GetProcessMemoryInfo(kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            raise OSError("GetProcessMemoryInfo failed")
        return int(counters.WorkingSetSize), int(counters.PeakWorkingSetSize)

    return read


def working_set_mb() -> tuple[float, float] | None:
    """Current and peak working set of this process in MB (Windows only; None elsewhere)."""
    reader = _psapi_reader()
    if reader is None:
        return None
    try:
        current, peak = reader()
    except OSError:
        return None
    return current / 2**20, peak / 2**20


class PeakMemoryWatch:
    """Context manager measuring the highest working set reached while it is open (Windows only).

    If the process-wide peak rises during the block, that new peak is exact; otherwise a background thread's samples
    (every ``interval`` seconds) give the highest value seen. The working set covers the whole process, so the peak
    includes whatever the process already held; ``start_mb`` (the working set on entry) lets a caller report the
    rise caused by the block. Both are None where they cannot be measured.
    """

    def __init__(self, interval: float = 0.05) -> None:
        self._interval = interval
        self._reader = _psapi_reader()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._peak_before = 0
        self._sampled = 0
        self.peak_mb: float | None = None
        self.start_mb: float | None = None

    def _poll(self) -> None:
        """Sample the working set until stopped."""
        assert self._reader is not None
        while not self._stop.wait(self._interval):
            try:
                self._sampled = max(self._sampled, self._reader()[0])
            except OSError:
                return

    def __enter__(self) -> "PeakMemoryWatch":
        if self._reader is not None:
            try:
                current, self._peak_before = self._reader()
            except OSError:
                self._reader = None
                return self
            self._sampled = current
            self.start_mb = current / 2**20
            self._thread = threading.Thread(target=self._poll, name="graticule-memory-watch", daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if self._reader is None:
            return
        try:
            current, peak_after = self._reader()
        except OSError:
            return
        best = peak_after if peak_after > self._peak_before else max(self._sampled, current)
        self.peak_mb = best / 2**20


# --------------------------------------------------------------------------------------------------------------
# The prepared dataset
# --------------------------------------------------------------------------------------------------------------
@dataclass
class PreparedDataset:
    """The result of 01 Sample: the sampled rows plus a full account of how they were obtained.

    ``frame`` holds the 77 float32 features, ``Label`` (the detailed class, never merged), ``_file`` (categorical
    source file name, "synthetic" for generated flows) and ``_row`` (int32 0-based data-row position in that file,
    or the generator's row number). No other copy of the data is kept.

    ``peak_memory_mb`` is the highest working set of the whole process during the draw, so in a long-running
    server it includes memory held before the draw began (cached files, other samples); ``memory_start_mb`` is
    the working set when the draw began, and :attr:`memory_rise_mb` the difference.
    """

    frame: pd.DataFrame
    request: DataRequest
    file_reports: list[FileStageReport]
    nonfinite: NonFiniteReport
    within_duplicates: DuplicateReport
    across_duplicates: DuplicateReport
    conflicts: ConflictReport
    degenerate: DegenerateReport
    sampling: SamplingReport
    class_counts: dict[str, int]
    single_class: bool
    single_class_message: str | None
    fingerprint: str
    seconds: float
    peak_memory_mb: float | None
    step_seconds: dict[str, float] = field(default_factory=dict)
    file_timings: dict[str, FileTiming] = field(default_factory=dict)
    memory_start_mb: float | None = None

    def features(self) -> pd.DataFrame:
        """The 77 float32 feature columns (a read-only view: copy before modifying)."""
        return self.frame[list(FEATURES)]

    def labels(self, detailed: bool = False) -> pd.Series:
        """Class of every row: merged Web Attack classes when the request asked for it, unless ``detailed``."""
        detailed_labels = self.frame[LABEL]
        if detailed:
            return detailed_labels
        return apply_class_options(detailed_labels, merge_web_attacks=self.request.merge_web_attacks)

    def provenance(self) -> pd.DataFrame:
        """Source file and original row position of every row."""
        return self.frame[[FILE_COL, ROW_COL]]

    @property
    def rows_read(self) -> int:
        """Data rows read from all files (or generated)."""
        return sum(r.read.rows_read for r in self.file_reports)

    @property
    def empty_labels(self) -> int:
        """Rows dropped because their label was empty."""
        return sum(r.read.empty_labels for r in self.file_reports)

    @property
    def rows_kept(self) -> int:
        """Rows left after cleaning and de-duplication, before sampling."""
        return self.sampling.rows_before

    @property
    def rows_sampled(self) -> int:
        """Rows in the sample."""
        return len(self.frame)

    @property
    def memory_rise_mb(self) -> float | None:
        """How far the process working set rose above its level at the start of the draw (None if unknown)."""
        if self.peak_memory_mb is None or self.memory_start_mb is None:
            return None
        return max(self.peak_memory_mb - self.memory_start_mb, 0.0)

    def reconciliation(self) -> dict[str, int]:
        """Where every row read went; the parts add up to ``rows_read``."""
        return {
            "rows_read": self.rows_read,
            "empty_labels": self.empty_labels,
            "bad_value_rows_dropped": self.nonfinite.rows_dropped,
            "duplicates_within_files": self.within_duplicates.rows_removed,
            "duplicates_across_files": self.across_duplicates.rows_removed,
            "conflicting_rows_removed": self.conflicts.rows_removed,
            "removed_by_sampling": self.sampling.rows_removed,
            "rows_sampled": self.rows_sampled,
        }

    def summary_rows(self) -> list[dict[str, object]]:
        """Headline readings for display and reports: a list of {"Reading", "Value", "Note"} rows.

        The readings reconcile: "Rows read" minus every reading marked as removed gives "Rows kept". Empty labels
        appear only when there were some; conflicting rows only when the conflict policy can remove rows.
        """
        nf = self.nonfinite
        rows: list[dict[str, object]] = [
            {"Reading": "Rows read", "Value": self.rows_read, "Note": f"{len(self.file_reports)} source(s)"},
        ]
        if self.empty_labels:
            rows.append({"Reading": "Empty labels", "Value": self.empty_labels, "Note": "rows removed"})
        if nf.strategy == "drop":
            rows.append({"Reading": "Bad-value rows", "Value": nf.rows_dropped,
                         "Note": "rows removed (infinite or missing values)"})
        else:
            note = {"impute": "rows kept; gaps filled per channel",
                    "recompute": "rows kept; rates rebuilt"}[nf.strategy]
            rows.append({"Reading": "Bad-value rows", "Value": nf.rows_affected, "Note": note})
        rows += [
            {"Reading": "Duplicates within files", "Value": self.within_duplicates.rows_removed,
             "Note": "rows removed"},
            {"Reading": "Duplicates across files", "Value": self.across_duplicates.rows_removed,
             "Note": "rows removed"},
        ]
        if self.conflicts.policy != "keep":
            rows.append({"Reading": "Conflicting rows", "Value": self.conflicts.rows_removed,
                         "Note": f"rows removed ({self.conflicts.policy} policy)"})
        rows += [
            {"Reading": "Rows kept", "Value": self.rows_kept, "Note": "after the removals above, before sampling"},
            {"Reading": "Rows sampled", "Value": self.rows_sampled, "Note": f"budget {self.sampling.budget:,}"},
            {"Reading": "Classes", "Value": len(self.class_counts), "Note": "in the sample"},
        ]
        return rows

    def file_table(self) -> pd.DataFrame:
        """One row per source: rows read, bad values, duplicates, rows kept and sampled, encoding, engine, the time
        this draw spent on it (an engine marked "cached" means the file was read by an earlier draw) and every
        repair made while reading."""
        sampled = self.frame[FILE_COL].astype("str").value_counts()
        rows = []
        for r in self.file_reports:
            known = EXPECTED_BY_NAME.get(r.name.lower())
            timing = self.file_timings.get(r.name)
            seconds = timing.seconds if timing is not None else r.read.seconds + r.seconds
            engine = r.read.engine + (" (cached)" if timing is not None and timing.reused else "")
            rows.append({
                "File": r.name,
                "Session": known.day if known else ("generated" if r.name == SYNTHETIC_FILE else "other"),
                "Rows read": r.read.rows_read,
                "Bad-value rows": r.nonfinite.rows_affected,
                "Duplicates in file": r.duplicates.rows_removed,
                "Rows after file stage": r.rows_out,
                "Rows in sample": int(sampled.get(r.name, 0)),
                "Encoding": r.read.encoding,
                "Engine": engine,
                "Seconds": round(seconds, 2),
                "File fixes": "; ".join(r.read.fixes()) or "none",
            })
        return pd.DataFrame(rows)

    def class_table(self) -> pd.DataFrame:
        """One row per class: kind, rows available after cleaning, rows in the sample and share of the sample."""
        total = max(self.rows_sampled, 1)
        rows = [
            {"Class": name, "Kind": "Normal" if is_benign(name) else "Attack", "Available": available,
             "In sample": self.sampling.after.get(name, 0),
             "Share of sample": self.sampling.after.get(name, 0) / total}
            for name, available in self.sampling.before.items()
        ]
        return pd.DataFrame(rows, columns=list(CLASS_TABLE_COLUMNS))


# --------------------------------------------------------------------------------------------------------------
# The procedure
# --------------------------------------------------------------------------------------------------------------
@dataclass
class _Source:
    """One file (or the generator's output) after the per-file steps of a draw."""

    name: str
    frame: pd.DataFrame
    stage: FileStage
    timing: FileTiming


def _collect_cicids(request: DataRequest, reader: FileReader, stager: FileStager,
                    notify: ProgressFn) -> list[_Source]:
    """Read every requested file and run the per-file steps on it, timing each file for this draw."""
    if not request.data_dir:
        raise DataFileError("No data folder is set. Choose one on the Bench, or draw a synthetic sample.")
    folder = Path(request.data_dir)
    if not folder.is_dir():
        raise DataFileError(f"Data folder not found: {folder}")
    if not request.files:
        raise DataFileError("Choose at least one file to sample from.")
    sources = []
    count = len(request.files)
    for i, name in enumerate(request.files):
        if Path(name).name != name:
            raise DataFileError(f"Not a plain file name: {name}")
        path = folder / name
        if not path.is_file():
            raise DataFileError(f"File not found in the data folder: {name}")
        notify(f"Reading {name} ({i + 1} of {count})", 0.8 * i / count)
        started = time.perf_counter()
        frame, read_report = reader(path)
        # A fresh read cannot finish sooner than its own inner timing, so a quicker answer came from a cache.
        reused = time.perf_counter() - started < read_report.seconds
        notify(f"Checking {name} for bad values and repeated rows", 0.8 * (i + 0.6) / count)
        stage = stager(path, frame, read_report, request.nonfinite_strategy)
        timing = FileTiming(seconds=time.perf_counter() - started, reused=reused)
        sources.append(_Source(name, frame, stage, timing))
        cached = ", read earlier" if reused else ""
        notify(f"{name}: {stage.report.rows_out:,} of {read_report.rows_read:,} rows kept "
               f"({timing.seconds:.1f} s{cached})", 0.8 * (i + 1) / count)
    return sources


def _collect_synthetic(request: DataRequest, notify: ProgressFn) -> list[_Source]:
    """Generate flows and run them through the same per-file steps as a real file."""
    started = time.perf_counter()
    total = int(request.synthetic_flows)
    notify(f"Generating {total:,} synthetic flows", 0.05)
    generate = _load_generator()
    options: dict[str, Any] = {}
    if _accepts_progress(generate):
        def generated(done: int, of: int) -> None:
            notify(f"Generating synthetic flows: {done:,} of {of:,}", 0.05 + 0.6 * done / max(of, 1))

        options["progress"] = generated
    raw = generate(total, seed=int(request.seed), attack_share=float(request.synthetic_attack_share), **options)
    notify("Checking the generated flows for bad values and repeated rows", 0.7)
    frame, read_report = _conform_generated(raw)
    del raw
    stage = stage_rows(frame, read_report, request.nonfinite_strategy)
    timing = FileTiming(seconds=time.perf_counter() - started, reused=False)
    notify(f"{SYNTHETIC_FILE}: {stage.report.rows_out:,} of {read_report.rows_read:,} generated rows kept "
           f"({timing.seconds:.1f} s)", 0.8)
    return [_Source(SYNTHETIC_FILE, frame, stage, timing)]


def _fingerprint(request: DataRequest, frame: pd.DataFrame) -> str:
    """sha256 over the request (without the folder path), the sorted (file, row) pairs and their labels."""
    digest = hashlib.sha256()
    digest.update(json.dumps(request.fingerprint_fields(), sort_keys=True).encode("utf-8"))
    files = frame[FILE_COL].astype("str").to_numpy()
    rows = frame[ROW_COL].to_numpy()
    labels = frame[LABEL].astype("str").to_numpy()
    for name in sorted(set(files.tolist())):
        chosen = np.flatnonzero(files == name)
        order = chosen[np.argsort(rows[chosen], kind="stable")]
        digest.update(b"\x00" + name.encode("utf-8") + b"\x00")
        digest.update(rows[order].astype("<i4").tobytes())
        digest.update("\n".join(labels[order].tolist()).encode("utf-8"))
    return digest.hexdigest()


def _single_class_note(classes: pd.Series) -> str | None:
    """User-facing warning when the sample holds fewer than two classes, else None."""
    try:
        target_for_mode(classes, "multiclass", min_class_count=1, hard_floor=1)
    except SingleClassError as exc:
        return str(exc)
    return None


def _nothing_left_message(sources: list[_Source], nonfinite: NonFiniteReport, within: DuplicateReport,
                          across: DuplicateReport, conflicts: ConflictReport) -> str:
    """Explain, from the counts, why no row survived cleaning (and what to change)."""
    rows_read = sum(s.stage.report.read.rows_read for s in sources)
    if rows_read == 0:
        which = "chosen file holds" if len(sources) == 1 else "chosen files hold"
        return f"The {which} no data rows, so there is nothing to sample."
    empty = sum(s.stage.report.read.empty_labels for s in sources)
    reasons = []
    hints = []
    if empty:
        reasons.append(f"{empty:,} had an empty label")
    if nonfinite.rows_dropped:
        reasons.append(f"{nonfinite.rows_dropped:,} held infinite or missing values, which the drop strategy removes")
        hints.append("keep those rows by choosing impute or rebuild for infinite and missing values")
    repeats = within.rows_removed + across.rows_removed
    if repeats:
        reasons.append(f"{repeats:,} repeated another row exactly")
    if conflicts.rows_removed:
        reasons.append(f"{conflicts.rows_removed:,} were identical flows with different labels, removed by the "
                       f"{conflicts.policy} policy")
        hints.append("keep conflicting rows with the keep policy")
    text = f"No rows are left to sample: of the {rows_read:,} rows read, " + "; ".join(reasons) + "."
    if hints:
        text += " To keep them, " + ", or ".join(hints) + "."
    elif empty:
        text += " Check that the file's Label column is filled in."
    return text


def prepare_dataset(
    request: DataRequest,
    *,
    read_file: FileReader | None = None,
    stage_file: FileStager | None = None,
    progress: ProgressFn | None = None,
) -> PreparedDataset:
    """Run the whole 01 Sample procedure for ``request`` and return the prepared, sampled dataset.

    ``read_file`` replaces :func:`read_source_file` and ``stage_file`` replaces :func:`stage_rows` (the UI passes
    cached versions; frames they return are shared and never modified here). ``progress`` receives (message,
    fraction done) updates. Raises :class:`~graticule.data.reader.DataFileError` with a readable message when a
    folder or file is missing or unusable, or when no row survives cleaning.
    """
    notify: ProgressFn = progress or (lambda message, fraction: None)
    started = time.perf_counter()
    steps: dict[str, float] = {}
    strategy = request.nonfinite_strategy
    with PeakMemoryWatch() as watch:
        if request.source == "synthetic":
            sources = _collect_synthetic(request, notify)
        else:
            sources = _collect_cicids(request, read_file or read_source_file, stage_file or _stage_without_cache,
                                      notify)
        steps["files"] = time.perf_counter() - started
        names = [s.name for s in sources]
        reports = [s.stage.report for s in sources]
        nonfinite = NonFiniteReport(strategy=strategy)
        within = DuplicateReport(stage="within file")
        for report in reports:
            nonfinite = nonfinite.combined(report.nonfinite)
            within = within.combined(report.duplicates)

        # Cross-file duplicates: the earliest file keeps its copy.
        mark = time.perf_counter()
        notify("Removing duplicates across files", 0.84)
        sizes = [len(s.stage.positions) for s in sources]
        file_id = np.repeat(np.arange(len(sources), dtype=np.int32), sizes)
        local = np.concatenate([np.arange(n, dtype=np.int64) for n in sizes]) if sources else np.empty(0, np.int64)
        detailed = (pd.concat([s.frame[LABEL].iloc[s.stage.positions] for s in sources], ignore_index=True)
                    if sources else pd.Series([], dtype="str"))
        fhash = (np.concatenate([s.stage.feature_hashes for s in sources]) if sources
                 else np.empty(0, np.uint64))
        copies = np.concatenate([s.stage.copies for s in sources]) if sources else np.empty(0, np.int32)
        full_hashes = hashes_with_labels(fhash, detailed)
        keep = first_occurrences(full_hashes)
        removed = int((~keep).sum())
        across = DuplicateReport(
            stage="across files", rows_before=len(keep), rows_removed=removed, rows_after=len(keep) - removed,
            by_class={str(k): int(v) for k, v in detailed[~keep].value_counts().items()} if removed else {},
        )
        weights = copies_per_kept_row(full_hashes, keep, copies)
        file_id, local, fhash = file_id[keep], local[keep], fhash[keep]
        detailed = detailed[keep].reset_index(drop=True)
        del full_hashes, copies
        steps["across_files"] = time.perf_counter() - mark

        # Class options, then conflicting labels on the classes that will be trained on.
        mark = time.perf_counter()
        classes = apply_class_options(detailed, merge_web_attacks=request.merge_web_attacks)
        notify("Checking for identical flows with different labels", 0.88)
        in_conflict, groups = conflict_mask(fhash, classes)
        conflicts = ConflictReport(
            groups=groups, rows=int(in_conflict.sum()), policy=request.conflict_policy,
            by_class={str(k): int(v) for k, v in classes[in_conflict].value_counts().items()} if groups else {},
        )
        survive = conflict_keep_mask(fhash, classes, request.conflict_policy, weights)  # type: ignore[arg-type]
        conflicts.rows_removed = int((~survive).sum())
        if conflicts.rows_removed:
            file_id, local = file_id[survive], local[survive]
            classes = classes[survive].reset_index(drop=True)
        steps["conflicts"] = time.perf_counter() - mark
        if len(classes) == 0:
            raise DataFileError(_nothing_left_message(sources, nonfinite, within, across, conflicts))

        # Rare-aware sample, then copy only the chosen rows out of each file frame (applying the repairs).
        mark = time.perf_counter()
        notify("Drawing the rare-aware sample", 0.92)
        positions, sampling_report = sample_positions(classes, int(request.row_budget), int(request.seed))
        file_id, local = file_id[positions], local[positions]
        sampled_classes = classes.iloc[positions].reset_index(drop=True)
        parts = []
        for index, source in enumerate(sources):
            at = source.stage.positions[local[file_id == index]]
            part = source.frame.iloc[at][[*FEATURES, LABEL]]
            if strategy != "drop":
                part, _ = apply_nonfinite_strategy(part, strategy)  # type: ignore[arg-type]
            part = part.reset_index(drop=True)
            part[ROW_COL] = source.frame.index.to_numpy()[at].astype(np.int32)
            parts.append(part)
        out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=[*FEATURES, LABEL, ROW_COL])
        out.insert(len(FEATURES) + 1, FILE_COL, pd.Categorical.from_codes(file_id, categories=names))
        out[ROW_COL] = out[ROW_COL].astype(np.int32)
        steps["sampling"] = time.perf_counter() - mark

        mark = time.perf_counter()
        degenerate = find_degenerate_columns(out, FEATURES)
        counts = sampled_classes.value_counts()
        class_counts = {name: int(counts[name]) for name in class_order([str(c) for c in counts.index])}
        note = _single_class_note(sampled_classes)
        fingerprint = _fingerprint(request, out)
        steps["summaries"] = time.perf_counter() - mark
        del parts

    notify("Sample ready", 1.0)
    return PreparedDataset(
        frame=out, request=request, file_reports=reports, nonfinite=nonfinite, within_duplicates=within,
        across_duplicates=across, conflicts=conflicts, degenerate=degenerate, sampling=sampling_report,
        class_counts=class_counts, single_class=note is not None, single_class_message=note,
        fingerprint=fingerprint, seconds=time.perf_counter() - started, peak_memory_mb=watch.peak_mb,
        step_seconds=steps, file_timings={s.name: s.timing for s in sources}, memory_start_mb=watch.start_mb,
    )
