"""Bench settings: where the data lives and the defaults every station starts from.

Settings are stored in ``local_settings.json`` next to the project (git ignores it). The data folder can also
come from the ``NIDS_DATA_DIR`` environment variable; a folder typed into the app takes precedence over it.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Literal

from nids.schema import EXPECTED_BY_NAME, EXPECTED_FILES

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SETTINGS_FILE = PROJECT_ROOT / "local_settings.json"
MODELS_DIR = PROJECT_ROOT / "saved_models"
HISTORY_DIR = PROJECT_ROOT / "run_history"
ENV_DATA_DIR = "NIDS_DATA_DIR"

NonFiniteStrategy = Literal["drop", "impute", "recompute"]
NONFINITE_STRATEGIES: tuple[str, ...] = ("drop", "impute", "recompute")
SVM_CAP_RANGE = (2_000, 50_000)


@dataclass
class AppSettings:
    """User-adjustable defaults. Every field has a safe default so a missing or partial file still works."""

    data_dir: str | None = None
    nonfinite_strategy: NonFiniteStrategy = "drop"
    row_budget: int = 200_000
    svm_cap: int = 20_000
    seed: int = 42
    alert_threshold: float = 0.90
    min_class_count: int = 50
    test_share: float = 0.25
    merge_web_attacks: bool = False

    def validated(self) -> "AppSettings":
        """Return a copy with every value clamped to its allowed range."""
        strategy = self.nonfinite_strategy if self.nonfinite_strategy in NONFINITE_STRATEGIES else "drop"
        return AppSettings(
            data_dir=(self.data_dir or "").strip() or None,
            nonfinite_strategy=strategy,  # type: ignore[arg-type]
            row_budget=int(min(max(int(self.row_budget), 1_000), 5_000_000)),
            svm_cap=int(min(max(int(self.svm_cap), SVM_CAP_RANGE[0]), SVM_CAP_RANGE[1])),
            seed=int(self.seed) % (2**31 - 1),
            alert_threshold=float(min(max(float(self.alert_threshold), 0.5), 0.999)),
            min_class_count=int(min(max(int(self.min_class_count), 10), 100_000)),
            test_share=float(min(max(float(self.test_share), 0.1), 0.5)),
            merge_web_attacks=bool(self.merge_web_attacks),
        )


def load_settings(path: Path | None = None) -> AppSettings:
    """Read settings from ``path`` (default ``local_settings.json``); unknown keys are ignored, bad files give defaults."""
    target = path or SETTINGS_FILE
    try:
        raw = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return AppSettings()
    if not isinstance(raw, dict):
        return AppSettings()
    known = {f.name for f in fields(AppSettings)}
    try:
        return AppSettings(**{k: v for k, v in raw.items() if k in known}).validated()
    except (TypeError, ValueError):
        return AppSettings()


def atomic_write_text(target: Path, text: str, retries: int = 3) -> None:
    """Write ``text`` to ``target`` via a temporary file and an atomic rename, retrying briefly on Windows locks."""
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
        replace_with_retry(tmp, target, retries)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)


def replace_with_retry(source: Path, target: Path, retries: int = 3) -> None:
    """``os.replace`` that retries a few times, because antivirus scans can briefly lock files on Windows."""
    for attempt in range(retries + 1):
        try:
            os.replace(source, target)
            return
        except PermissionError:
            if attempt == retries:
                raise
            time.sleep(0.2 * (attempt + 1))


def save_settings(settings: AppSettings, path: Path | None = None) -> Path:
    """Validate and store ``settings``; returns the file written."""
    target = path or SETTINGS_FILE
    atomic_write_text(target, json.dumps(asdict(settings.validated()), indent=2))
    return target


@dataclass(frozen=True)
class DataDirResolution:
    """Which data folder is in use, where that choice came from, and which expected files were found."""

    path: Path | None
    source: Literal["setting", "env", "none"]
    found: tuple[Path, ...]
    missing: tuple[str, ...]
    other_csvs: tuple[Path, ...]
    problem: str | None = None

    @property
    def usable(self) -> bool:
        """True when the folder exists and holds at least one CSV file."""
        return self.path is not None and (bool(self.found) or bool(self.other_csvs))


def list_csv_files(folder: Path) -> list[Path]:
    """Return ``*.csv`` files directly inside ``folder`` (not recursive), sorted by name."""
    return sorted((p for p in folder.glob("*.csv") if p.is_file()), key=lambda p: p.name.lower())


def resolve_data_dir(settings: AppSettings, environ: dict[str, str] | None = None) -> DataDirResolution:
    """Work out which data folder to use: the in-app setting, else ``NIDS_DATA_DIR``, else none (synthetic mode)."""
    env = os.environ if environ is None else environ
    candidate: str | None = None
    source: Literal["setting", "env", "none"] = "none"
    if settings.data_dir and settings.data_dir.strip():
        candidate, source = settings.data_dir.strip(), "setting"
    elif env.get(ENV_DATA_DIR, "").strip():
        candidate, source = env[ENV_DATA_DIR].strip(), "env"
    if candidate is None:
        return DataDirResolution(None, "none", (), tuple(f.name for f in EXPECTED_FILES), ())
    folder = Path(candidate.strip('"')).expanduser()
    if not folder.is_dir():
        return DataDirResolution(
            None, source, (), tuple(f.name for f in EXPECTED_FILES), (), problem=f"Folder not found: {folder}"
        )
    csvs = list_csv_files(folder)
    found = tuple(p for p in csvs if p.name.lower() in EXPECTED_BY_NAME)
    found_names = {p.name.lower() for p in found}
    missing = tuple(f.name for f in EXPECTED_FILES if f.name.lower() not in found_names)
    others = tuple(p for p in csvs if p.name.lower() not in EXPECTED_BY_NAME)
    problem = None if csvs else "The folder exists but contains no CSV files."
    return DataDirResolution(folder, source, found, missing, others, problem)
