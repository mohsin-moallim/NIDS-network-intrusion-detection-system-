"""Shared pytest setup: isolation from the developer's own settings, and access to the real dataset when present."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

import graticule.settings as settings_mod

# Resolved once, before any test clears the environment: an explicit --data-dir, else NIDS_DATA_DIR,
# else the folder saved in local_settings.json. Real-data tests skip when none of these points at a folder.
_REAL_DATA_DIR: Path | None = None


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register ``--data-dir`` for real-data tests.

    Give it as one token, ``"--data-dir=<folder>"``. Written as two arguments, pytest reads the folder as a test path
    while it is still looking for its configuration (before this file is loaded), settles on the wrong root and then
    rejects ``--data-dir`` as unrecognised.
    """
    parser.addoption(
        "--data-dir",
        action="store",
        default=None,
        help='Folder with the CIC-IDS2017 CSV files; write it as one token: "--data-dir=<folder>".',
    )


def pytest_configure(config: pytest.Config) -> None:
    """Work out where the real dataset is (if anywhere) before tests start changing the environment."""
    global _REAL_DATA_DIR
    candidate = config.getoption("--data-dir") or os.environ.get(settings_mod.ENV_DATA_DIR)
    if not candidate:
        candidate = settings_mod.load_settings().data_dir
    if candidate and Path(candidate).is_dir():
        _REAL_DATA_DIR = Path(candidate)


@pytest.fixture
def real_data_dir() -> Path:
    """Path to the real CIC-IDS2017 folder; skips the test when it is not available."""
    if _REAL_DATA_DIR is None:
        pytest.skip('CIC-IDS2017 folder not available (set NIDS_DATA_DIR or pass "--data-dir=<folder>")')
    return _REAL_DATA_DIR


@pytest.fixture(autouse=True)
def quick_latency_timing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Time single-flow latency on a 20 ms budget in tests (the app allows 0.25 s per channel).

    The reading stays a real median of at least five timed calls; only the extra calls a forest would get (each one
    starts a pool of threads) are skipped, which keeps every readings-taking test about 0.2 s quicker.
    """
    from graticule import evaluate

    defaults = evaluate.single_flow_latency_ms.__kwdefaults__
    if defaults is not None and "budget_seconds" in defaults:
        monkeypatch.setitem(defaults, "budget_seconds", 0.02)


@pytest.fixture(autouse=True)
def isolated_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point every settings, model and history path at a temporary folder and clear NIDS_DATA_DIR."""
    monkeypatch.delenv(settings_mod.ENV_DATA_DIR, raising=False)
    monkeypatch.setattr(settings_mod, "SETTINGS_FILE", tmp_path / "local_settings.json")
    monkeypatch.setattr(settings_mod, "MODELS_DIR", tmp_path / "saved_models")
    monkeypatch.setattr(settings_mod, "HISTORY_DIR", tmp_path / "run_history")
    monkeypatch.setenv("GRATICULE_SYNC_TRAINING", "1")
    return tmp_path
