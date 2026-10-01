"""Read one CIC-IDS2017-style flow CSV into a float32 feature frame with clean labels.

The published files have three quirks this module absorbs: header names carry stray leading spaces, the
``Fwd Header Length`` column appears twice, and the Web Attack labels hold a stand-in character where a dash
was (U+FFFD in the UTF-8 copies, byte 0x96 in Windows-1252 copies). Rates such as ``Flow Bytes/s`` can also be
the text ``Infinity``, which is parsed as ``inf``.

Reading tries the fast pyarrow engine first and falls back to pandas' C engine; encodings are tried in the order
UTF-8, Windows-1252, Latin-1. The same function reads files on disk and uploaded files (for batch scoring), where
the ``Label`` column is optional.
"""

from __future__ import annotations

import io
import re
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO, Literal

import numpy as np
import pandas as pd

from graticule.schema import FEATURE_SET, FEATURES, LABEL, normalize_label

# Bump when reading or cleaning logic changes, so cached per-file results are rebuilt.
READER_VERSION = "1"
ENCODINGS: tuple[str, ...] = ("utf-8", "cp1252", "latin-1")
Engine = Literal["auto", "pyarrow", "c"]
Source = Path | str | bytes | BinaryIO

# The C engine renames a repeated header "X" to "X.1", "X.2", ...
_MANGLED = re.compile(r"^(?P<base>.+?)\.(?P<n>\d+)$")
_DECODE_HINTS = ("utf8", "utf-8", "decode", "codec", "encoding", "unicode")


class DataFileError(ValueError):
    """A file could not be read, or does not have the columns Graticule needs. The message is shown to the user."""


@dataclass
class FileReadReport:
    """What happened while reading one file: sizes, how it was decoded, and every column or label fix applied."""

    name: str
    rows_read: int = 0
    rows_kept: int = 0
    encoding: str = ""
    engine: str = ""
    seconds: float = 0.0
    duplicate_columns_dropped: list[str] = field(default_factory=list)
    duplicate_columns_equal: bool = True
    duplicate_columns_mismatched: list[str] = field(default_factory=list)
    missing_features: list[str] = field(default_factory=list)
    extra_columns: list[str] = field(default_factory=list)
    empty_labels: int = 0
    label_counts: dict[str, int] = field(default_factory=dict)
    has_label: bool = False
    non_numeric_cells: int = 0
    fallback_note: str | None = None

    @property
    def features_found(self) -> int:
        """Number of the 77 known feature columns present in the file."""
        return len(FEATURES) - len(self.missing_features)

    def fixes(self, *, include_routine: bool = True) -> list[str]:
        """Plain-language list of every repair made while reading, for the sample sheet.

        A repeated column whose copy matched the first one is the routine quirk of the published files; pass
        ``include_routine=False`` to leave it out and keep only the repairs worth a reader's attention.
        """
        out: list[str] = []
        mismatched = set(self.duplicate_columns_mismatched)
        for name in self.duplicate_columns_dropped:
            if name in mismatched:
                out.append(f"repeated {name} dropped (it differed from the first copy, which was kept)")
            elif include_routine:
                out.append(f"repeated {name} dropped (identical copy)")
        if self.extra_columns:
            shown = ", ".join(self.extra_columns[:4]) + (", ..." if len(self.extra_columns) > 4 else "")
            plural = "s" if len(self.extra_columns) != 1 else ""
            out.append(f"{len(self.extra_columns)} unknown column{plural} ignored ({shown})")
        if self.non_numeric_cells:
            plural = "s" if self.non_numeric_cells != 1 else ""
            out.append(f"{self.non_numeric_cells:,} text cell{plural} in number columns turned into gaps")
        if self.empty_labels:
            plural = "s" if self.empty_labels != 1 else ""
            out.append(f"{self.empty_labels:,} row{plural} with an empty label dropped")
        if self.fallback_note:
            out.append(f"read by the fallback parser ({self.fallback_note})")
        return out


def _source_label(source: Source, name: str | None) -> str:
    """A short display name for ``source``."""
    if name:
        return name
    if isinstance(source, (str, Path)):
        return Path(source).name
    return str(getattr(source, "name", "") or "uploaded file")


def _as_input(source: Source) -> Path | bytes:
    """Return a path for files on disk, or the raw bytes for anything file-like (read once, retried from memory)."""
    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.is_file():
            raise DataFileError(f"File not found: {path}")
        return path
    if isinstance(source, (bytes, bytearray, memoryview)):
        return bytes(source)
    if hasattr(source, "read"):
        if hasattr(source, "seek"):
            try:
                source.seek(0)
            except (OSError, ValueError):
                pass
        data = source.read()
        if isinstance(data, str):
            data = data.encode("utf-8")
        return bytes(data)
    raise TypeError(f"Cannot read flows from a {type(source).__name__}")


def _looks_like_decoding_problem(exc: BaseException) -> bool:
    """True when a parser error message points at text decoding rather than the CSV structure."""
    text = str(exc).lower()
    return any(hint in text for hint in _DECODE_HINTS)


def _holds_undecoded_bytes(frame: pd.DataFrame) -> bool:
    """pyarrow returns columns of raw ``bytes`` when text is not valid UTF-8; treat that as a decoding failure."""
    for pos, dtype in enumerate(frame.dtypes):
        if dtype != np.dtype(object):
            continue
        values = frame.iloc[:, pos].dropna()
        if len(values) and isinstance(values.iloc[0], (bytes, bytearray)):
            return True
    return False


def _parse(data: Path | bytes, engine: str, encoding: str, nrows: int | None) -> pd.DataFrame:
    """One parsing attempt with a given engine and encoding."""
    handle: str | io.BytesIO = str(data) if isinstance(data, Path) else io.BytesIO(data)
    if engine == "pyarrow":
        return pd.read_csv(handle, engine="pyarrow", encoding=encoding)
    return pd.read_csv(handle, engine="c", encoding=encoding, index_col=False, nrows=nrows)


def _read_raw(data: Path | bytes, engine: Engine, nrows: int | None) -> tuple[pd.DataFrame, str, str, str | None]:
    """Try each engine and encoding in turn; return the raw frame, engine used, encoding used and a fallback note.

    A file the operating system refuses to open (held by another program, or no permission) is reported as such at
    once, since no other engine or encoding can help.
    """
    engines = ["pyarrow", "c"] if engine == "auto" else [engine]
    if nrows is not None:
        engines = ["c"]  # the pyarrow engine cannot stop after n rows
    note: str | None = None
    last_error: BaseException | None = None
    for eng in engines:
        for encoding in ENCODINGS:
            try:
                raw = _parse(data, eng, encoding, nrows)
            except PermissionError as exc:
                where = data.name if isinstance(data, Path) else "The file"
                raise DataFileError(
                    f"{where} cannot be opened: another program may be holding it, or access is denied. "
                    "Close it elsewhere and try again."
                ) from exc
            except UnicodeDecodeError as exc:
                last_error = exc
                continue
            except ImportError as exc:  # pyarrow missing: use the C engine
                last_error = exc
                note = f"{eng} engine unavailable ({exc})"
                break
            except (ValueError, OSError) as exc:
                last_error = exc
                if isinstance(exc, ValueError) and _looks_like_decoding_problem(exc):
                    continue
                note = f"{eng} engine failed: {str(exc).splitlines()[0][:200] if str(exc) else type(exc).__name__}"
                break
            if _holds_undecoded_bytes(raw):
                del raw
                last_error = UnicodeError(f"text is not valid {encoding}")
                continue
            return raw, eng, encoding, note
    detail = str(last_error).splitlines()[0][:300] if last_error else "unknown error"
    raise DataFileError(f"Could not read the file as CSV: {detail}")


def _clean_name(name: object) -> str:
    """Header name with the byte-order mark and surrounding spaces removed."""
    return str(name).replace("﻿", "").strip()


def _columns_equal(a: pd.Series, b: pd.Series) -> bool:
    """True when two columns hold the same values (numerically when possible, NaN equal to NaN)."""
    if pd.api.types.is_numeric_dtype(a) or pd.api.types.is_numeric_dtype(b):
        x = pd.to_numeric(a, errors="coerce").to_numpy(dtype=np.float64)
        y = pd.to_numeric(b, errors="coerce").to_numpy(dtype=np.float64)
        return bool(np.array_equal(x, y, equal_nan=True))
    return bool(a.reset_index(drop=True).equals(b.reset_index(drop=True)))


def _to_float32(column: pd.Series) -> tuple[np.ndarray, int]:
    """Convert one column to float32; text that is not a number becomes NaN. Returns values and the bad-cell count."""
    if pd.api.types.is_bool_dtype(column) or pd.api.types.is_numeric_dtype(column):
        with np.errstate(over="ignore", invalid="ignore"):
            return column.to_numpy(dtype=np.float32, na_value=np.nan), 0
    parsed = pd.to_numeric(column, errors="coerce")
    text = column.astype("str").str.strip()
    blank = column.isna() | text.eq("") | text.str.lower().isin(("nan", "na", "null", "none"))
    bad = int((parsed.isna() & ~blank).sum())
    with np.errstate(over="ignore", invalid="ignore"):
        return parsed.to_numpy(dtype=np.float32, na_value=np.nan), bad


def normalize_labels(values: pd.Series) -> pd.Series:
    """Apply :func:`graticule.schema.normalize_label` to every value (once per distinct value); missing becomes ""."""
    uniques = values.unique()
    mapping = {u: ("" if pd.isna(u) else normalize_label(u)) for u in uniques}
    return values.map(mapping).astype("str")


def _dedupe_columns(raw: pd.DataFrame) -> tuple[list[tuple[str, int]], list[str], list[str]]:
    """Pick the first occurrence of each stripped header name by position; return kept (name, position) pairs,
    the repeated names dropped, and the repeated names whose values differ from the column they repeat."""
    names = [_clean_name(c) for c in raw.columns]
    known = FEATURE_SET | {LABEL}
    first_at: dict[str, int] = {}
    kept: list[tuple[str, int]] = []
    dropped: list[str] = []
    mismatched: list[str] = []
    for pos, name in enumerate(names):
        base = name
        if base not in first_at and base not in known:
            match = _MANGLED.match(name)
            if match and match.group("base").strip() in first_at:
                base = match.group("base").strip()
        if base in first_at:
            dropped.append(base)
            if not _columns_equal(raw.iloc[:, first_at[base]], raw.iloc[:, pos]) and base not in mismatched:
                mismatched.append(base)
            continue
        first_at[base] = pos
        kept.append((base, pos))
    return kept, dropped, mismatched


def read_flow_csv(
    source: Source,
    *,
    require_label: bool = True,
    engine: Engine = "auto",
    nrows: int | None = None,
    name: str | None = None,
) -> tuple[pd.DataFrame, FileReadReport]:
    """Read one flow CSV (a path, raw bytes or an uploaded file object).

    Returns a frame holding the known feature columns that are present, in :data:`graticule.schema.FEATURES`
    order and as float32, followed by ``Label`` (pandas ``str``, normalised) when the file has one. The frame's
    index is each row's 0-based position among the file's data rows, so provenance survives any later filtering.
    Rows whose label is empty are counted; they are dropped when ``require_label`` is True (training data) and
    kept with an empty label otherwise (uploads to score). Missing feature columns are reported, not invented; the
    caller decides whether they matter. ``nrows`` reads only the first rows (C engine), for quick checks.
    """
    started = time.perf_counter()
    label = _source_label(source, name)
    data = _as_input(source)
    raw, used_engine, used_encoding, note = _read_raw(data, engine, nrows)
    del data
    report = FileReadReport(name=label, rows_read=len(raw), encoding=used_encoding, engine=used_engine,
                            fallback_note=note)

    kept, dropped, mismatched = _dedupe_columns(raw)
    report.duplicate_columns_dropped = dropped
    report.duplicate_columns_mismatched = mismatched
    report.duplicate_columns_equal = not mismatched
    position = dict(kept)
    label_name = next((n for n, _ in kept if n.lower() == LABEL.lower()), None)
    present = [f for f in FEATURES if f in position]
    report.missing_features = [f for f in FEATURES if f not in position]
    report.extra_columns = [n for n, _ in kept if n not in FEATURE_SET and n != label_name]
    report.has_label = label_name is not None
    if require_label and label_name is None:
        raise DataFileError(f"{label} has no Label column, so it cannot be used to fit or measure channels.")

    # Build the float32 block column by column (one contiguous row per feature), then release the wide frame.
    n_rows = len(raw)
    block = np.empty((len(present), n_rows), dtype=np.float32)
    bad_cells = 0
    for i, feature in enumerate(present):
        block[i], bad = _to_float32(raw.iloc[:, position[feature]])
        bad_cells += bad
    report.non_numeric_cells = bad_cells
    labels = normalize_labels(raw.iloc[:, position[label_name]]) if label_name is not None else None
    del raw

    frame = pd.DataFrame(block.T, columns=present, copy=False)
    if labels is not None:
        label_values = labels.to_numpy()
        empty = labels.eq("").to_numpy()
        frame[LABEL] = pd.Series(label_values, dtype="str")
        report.empty_labels = int(empty.sum())
        if report.empty_labels and require_label:
            frame = frame.loc[~empty]
        counts = frame.loc[frame[LABEL] != "", LABEL].value_counts(sort=True)
        report.label_counts = {str(k): int(v) for k, v in counts.items()}
    report.rows_kept = len(frame)
    report.seconds = time.perf_counter() - started
    return frame, report


def feature_columns(frame: pd.DataFrame, candidates: Iterable[str] = FEATURES) -> list[str]:
    """The known feature columns present in ``frame``, in catalogue order."""
    return [c for c in candidates if c in frame.columns]
