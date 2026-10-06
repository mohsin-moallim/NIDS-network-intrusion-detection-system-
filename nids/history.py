"""Run history: one line per finished fit, kept in a small SQLite file that survives restarts.

The file is ``run_history/runs.sqlite3`` (git ignores it). Each line summarises a run: when it was fitted, from which
source and files, in which mode and feature set, how many training and test rows it used, which channels were
fitted, the best channel by balanced accuracy, every channel's headline metrics and the settings, as JSON, the
duration, and where the run was saved (once it is).

Every call opens its own connection and closes it before returning, and the journal stays in SQLite's default
rollback mode (no write-ahead log), so no handle or side file outlives a call: Windows can delete or move the file
whenever the app is idle. Only plain values are stored; never a dataset row.
"""

from __future__ import annotations

import json
import math
import sqlite3
from collections.abc import Iterator, Mapping
from contextlib import closing, contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

from nids import settings as settings_mod

if TYPE_CHECKING:
    from nids.models.train import TrainingRun

#: Name of the history file inside ``settings.HISTORY_DIR``.
HISTORY_FILE = "runs.sqlite3"
#: Columns of a history line, in table order.
COLUMNS: tuple[str, ...] = (
    "run_id", "created_utc", "source", "files", "mode", "feature_mode", "rows_train", "rows_test", "channels",
    "best_channel", "best_balanced_accuracy", "metrics_json", "settings_json", "seconds", "saved_path",
)
#: The order channels are listed in (same as ``nids.models.zoo.MODEL_KEYS``; repeated here so this module
#: stays light to import).
CHANNEL_ORDER: tuple[str, ...] = ("forest", "xgboost", "svm", "mlp", "logreg")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    created_utc TEXT NOT NULL,
    source TEXT NOT NULL,
    files TEXT NOT NULL,
    mode TEXT NOT NULL,
    feature_mode TEXT NOT NULL,
    rows_train INTEGER NOT NULL,
    rows_test INTEGER NOT NULL,
    channels TEXT NOT NULL,
    best_channel TEXT,
    best_balanced_accuracy REAL,
    metrics_json TEXT NOT NULL,
    settings_json TEXT NOT NULL,
    seconds REAL NOT NULL,
    saved_path TEXT
)
"""


def _number(value: Any) -> float | None:
    """A finite float, or None for missing, non-numeric or non-finite values."""
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _ordered(keys: Any) -> list[str]:
    """Channel keys in the fixed CH1..CH5 order, unknown keys after them in their own order."""
    keys = [str(k) for k in keys]
    return [k for k in CHANNEL_ORDER if k in keys] + [k for k in keys if k not in CHANNEL_ORDER]


def best_reading(run: "TrainingRun") -> tuple[str | None, float | None]:
    """The fitted channel with the highest balanced accuracy on the held-out rows, and that value.

    Read from each channel's stored metrics (``extra["metrics"]``); ties go to the lower channel number. Returns
    ``(None, None)`` when no fitted channel has a balanced accuracy.
    """
    best: tuple[str | None, float | None] = (None, None)
    for key in _ordered(run.channels):
        result = run.channels[key]
        if result.status != "ok":
            continue
        value = _number(((result.extra or {}).get("metrics") or {}).get("balanced_accuracy"))
        if value is not None and (best[1] is None or value > best[1]):
            best = (key, value)
    return best


def _plain_metrics(metrics: Mapping[str, Any] | None) -> dict[str, float | None]:
    """Metric values as plain floats (None where missing or not finite)."""
    return {str(k): _number(v) for k, v in (metrics or {}).items()}


def run_summary(run: "TrainingRun", *, saved_path: Path | str | None = None) -> dict[str, Any]:
    """The history line for ``run`` as a dict keyed by :data:`COLUMNS` (plain values only)."""
    request = run.request
    data_request = run.data_request
    metrics = {
        key: {"status": run.channels[key].status,
              **_plain_metrics((run.channels[key].extra or {}).get("metrics"))}
        for key in _ordered(run.channels)
    }
    settings_blob = {
        "train_request": request.to_dict(),
        "data_request": data_request.fingerprint_fields(),
        "data_dir": data_request.data_dir,
        "dataset_fingerprint": run.dataset_fingerprint,
    }
    best_key, best_value = best_reading(run)
    files = ", ".join(data_request.files) if data_request.files else (
        "generated" if data_request.source == "synthetic" else "")
    reports = run.data.reports or {}
    rows = reports.get("rows") or {}
    rows_train = len(run.data.y_train) or int(rows.get("train", 0) or 0)
    rows_test = len(run.data.y_test) or int(rows.get("test", 0) or 0)
    seconds = float(getattr(run, "total_seconds", run.seconds))
    return {
        "run_id": str(run.run_id),
        "created_utc": str(run.created_utc),
        "source": str(data_request.source),
        "files": files,
        "mode": str(request.mode),
        "feature_mode": str(request.feature_mode),
        "rows_train": int(rows_train),
        "rows_test": int(rows_test),
        "channels": ", ".join(run.ok_channels()),
        "best_channel": best_key,
        "best_balanced_accuracy": best_value,
        "metrics_json": json.dumps(metrics, sort_keys=True, allow_nan=False),
        "settings_json": json.dumps(settings_blob, sort_keys=True, allow_nan=False, default=str),
        "seconds": round(seconds, 3),
        "saved_path": None if saved_path is None else str(saved_path),
    }


class RunHistory:
    """The run history file. Cheap to create: nothing is opened until a method is called.

    ``db_path`` defaults to ``settings.HISTORY_DIR / "runs.sqlite3"`` (looked up when the object is created, so
    tests can point ``HISTORY_DIR`` elsewhere); the folder and the file are created on first use.
    """

    def __init__(self, db_path: Path | None = None) -> None:
        self.path = Path(db_path) if db_path is not None else Path(settings_mod.HISTORY_DIR) / HISTORY_FILE

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        """An open connection inside a transaction: committed on success, rolled back on error, always closed."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.path, timeout=10.0)) as conn:
            conn.execute("PRAGMA journal_mode=DELETE").fetchall()
            with conn:
                conn.execute(_SCHEMA)
                yield conn

    def record(self, run: "TrainingRun", *, saved_path: Path | None = None) -> None:
        """Add the line for ``run``. Recording the same run again changes nothing, except that a ``saved_path``
        given now is stored."""
        line = run_summary(run, saved_path=saved_path)
        names = ", ".join(COLUMNS)
        marks = ", ".join("?" for _ in COLUMNS)
        with self._connection() as conn:
            conn.execute(f"INSERT OR IGNORE INTO runs ({names}) VALUES ({marks})", [line[c] for c in COLUMNS])
            if saved_path is not None:
                conn.execute("UPDATE runs SET saved_path = ? WHERE run_id = ?", (str(saved_path), line["run_id"]))

    def mark_saved(self, run_id: str, path: Path | str) -> None:
        """Note where run ``run_id`` was saved (nothing happens when the run is not in the history)."""
        with self._connection() as conn:
            conn.execute("UPDATE runs SET saved_path = ? WHERE run_id = ?", (str(path), str(run_id)))

    def list(self, limit: int | None = 200) -> pd.DataFrame:
        """The newest ``limit`` lines (every line when ``limit`` is None), newest first, with the columns of
        :data:`COLUMNS`."""
        query = f"SELECT {', '.join(COLUMNS)} FROM runs ORDER BY created_utc DESC, rowid DESC"
        with self._connection() as conn:
            if limit is None:
                rows = conn.execute(query).fetchall()
            else:
                rows = conn.execute(query + " LIMIT ?", (max(int(limit), 0),)).fetchall()
        frame = pd.DataFrame(rows, columns=list(COLUMNS))
        for column in ("rows_train", "rows_test"):
            frame[column] = frame[column].astype("int64")
        for column in ("best_balanced_accuracy", "seconds"):
            frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("float64")
        return frame

    def get(self, run_id: str) -> dict[str, Any] | None:
        """The line of run ``run_id`` as a dict (JSON columns left as text), or None when it is not recorded."""
        with self._connection() as conn:
            row = conn.execute(f"SELECT {', '.join(COLUMNS)} FROM runs WHERE run_id = ?", (str(run_id),)).fetchone()
        return None if row is None else dict(zip(COLUMNS, row))

    def delete(self, run_id: str) -> None:
        """Remove the line of run ``run_id`` (nothing happens when it is not recorded)."""
        with self._connection() as conn:
            conn.execute("DELETE FROM runs WHERE run_id = ?", (str(run_id),))

    def clear(self) -> None:
        """Remove every line (the file itself stays)."""
        with self._connection() as conn:
            conn.execute("DELETE FROM runs")

    def count(self) -> int:
        """Number of recorded runs."""
        with self._connection() as conn:
            return int(conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0])
