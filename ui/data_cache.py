"""Streamlit caching for the per-file work of 01 Sample (the prepared dataset itself lives in :mod:`ui.state`).

Three process-wide caches (``st.cache_resource``: one shared object, no copy per session):

* the file as read (:func:`nids.data.prepare.read_source_file`), which does not depend on any option, so a
  file is read once however often the strategy or the budget changes;
* the hash of every row (:func:`nids.data.prepare.hash_rows`), the slow part of the per-file stage, which no
  strategy changes, so switching strategy does not hash a file again (16 bytes per row);
* the per-file stage for one bad-value strategy (:func:`nids.data.prepare.stage_rows`): row positions and
  hashes only, a few bytes per row, so every strategy of every file fits.

Keys include the file's size and modification time and the reader/stage versions, so an edited file or new code is
never served stale results. Cached objects are shared between sessions and must never be modified.
:func:`clear_file_cache` releases them (the 01 Sample station offers this as a button).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from nids.data.prepare import (
    STAGE_VERSION,
    FileReader,
    FileStage,
    FileStager,
    RowHashes,
    hash_rows,
    read_source_file,
    stage_rows,
)
from nids.data.reader import READER_VERSION, FileReadReport

# Room for the eight published files plus a few other CSVs, so drawing from all of them never evicts one that
# the same draw needs again; three strategies per file for the small stage entries.
READ_ENTRIES = 12
STAGE_ENTRIES = 3 * READ_ENTRIES


@st.cache_resource(max_entries=READ_ENTRIES, show_spinner=False)
def cached_read(path_str: str, size: int, mtime_ns: int, reader_version: str) -> tuple[pd.DataFrame, FileReadReport]:
    """One file as read, for one file version (``size``/``mtime_ns``/``reader_version`` are key parts)."""
    return read_source_file(Path(path_str))


@st.cache_resource(max_entries=READ_ENTRIES, show_spinner=False)
def cached_hashes(path_str: str, size: int, mtime_ns: int, stage_version: str, _frame: pd.DataFrame) -> RowHashes:
    """Row hashes of one file version; the frame is not part of the key (the path, size, modification time and
    version fix it)."""
    return hash_rows(_frame)


@st.cache_resource(max_entries=STAGE_ENTRIES, show_spinner=False)
def cached_stage(path_str: str, size: int, mtime_ns: int, strategy: str, stage_version: str,
                 _frame: pd.DataFrame, _read_report: FileReadReport) -> FileStage:
    """Per-file stage of one file version for one strategy; the frame and report are not part of the key (they
    are fixed by the path, size, modification time and version)."""
    hashes = cached_hashes(path_str, size, mtime_ns, stage_version, _frame)
    return stage_rows(_frame, _read_report, strategy, hashes=hashes)


def _key(path: Path) -> tuple[str, int, int]:
    """Resolved path, size and modification time of a file: the part of every cache key that names its version."""
    resolved = Path(path).resolve()
    stat = resolved.stat()
    return str(resolved), int(stat.st_size), int(stat.st_mtime_ns)


def cached_reader(path: Path) -> tuple[pd.DataFrame, FileReadReport]:
    """Drop-in replacement for :func:`nids.data.prepare.read_source_file` that goes through the cache."""
    path_str, size, mtime_ns = _key(path)
    return cached_read(path_str, size, mtime_ns, READER_VERSION)


def cached_stager(path: Path, frame: pd.DataFrame, read_report: FileReadReport, strategy: str) -> FileStage:
    """Drop-in :data:`nids.data.prepare.FileStager` that goes through the cache."""
    path_str, size, mtime_ns = _key(path)
    return cached_stage(path_str, size, mtime_ns, strategy, STAGE_VERSION, frame, read_report)


def file_reader() -> FileReader:
    """The reader the 01 Sample station passes to ``prepare_dataset``."""
    return cached_reader


def file_stager() -> FileStager:
    """The per-file stage the 01 Sample station passes to ``prepare_dataset``."""
    return cached_stager


def clear_file_cache() -> None:
    """Forget every cached file and per-file stage (frees their memory; the next draw reads the files again)."""
    cached_read.clear()
    cached_hashes.clear()
    cached_stage.clear()
