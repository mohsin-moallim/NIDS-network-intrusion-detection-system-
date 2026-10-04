"""Shared pytest setup: isolation from the developer's own settings, and access to the real dataset when present."""

from __future__ import annotations

import functools
import itertools
import os
from collections.abc import Iterator
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


@pytest.fixture(autouse=True, scope="session")
def library_versions_read_once() -> Iterator[None]:
    """Read the installed library versions once per session instead of at every save and load of a channel set.

    Each read searches the metadata of every installed package (about 15 ms on Windows, some sixty times in a
    default run), and the versions cannot change while the tests run. Every caller still gets its own copy; tests that
    simulate another version replace ``persist.library_versions`` themselves, as before.
    """
    from graticule import persist

    real = persist.library_versions
    versions = real()

    @functools.wraps(real)
    def read_once() -> dict[str, str]:
        return dict(versions)

    persist.library_versions = read_once
    try:
        yield
    finally:
        persist.library_versions = real


@pytest.fixture(autouse=True)
def quick_latency_timing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Time single-flow latency with at most three calls (at least two) on a 20 ms budget in tests; the app makes up
    to 30 calls (at least five) within 0.25 s per channel.

    The reading stays a real median of timed ``predict_proba`` calls after the warm-up call (tests only check that it
    is a positive number). One call of a forest starts a pool of threads (about 17 ms) and one of a scaled pipeline
    takes about 3 ms, so the readings of a five-channel run take about 0.1 s less than with five to 30 calls.
    """
    from graticule import evaluate

    latency = evaluate.single_flow_latency_ms
    if latency.__kwdefaults__ is not None and "budget_seconds" in latency.__kwdefaults__:
        monkeypatch.setitem(latency.__kwdefaults__, "budget_seconds", 0.02)
    if latency.__defaults__ is not None and len(latency.__defaults__) == 1:
        monkeypatch.setattr(latency, "__defaults__", (3,))
    if hasattr(evaluate, "LATENCY_MIN_CALLS"):
        monkeypatch.setattr(evaluate, "LATENCY_MIN_CALLS", 2)


#: Boosting rounds of the feature-ranking model in tests that ask for :func:`quick_ranking` (the app uses 120).
QUICK_RANK_ROUNDS = 20


@pytest.fixture
def quick_ranking(monkeypatch: pytest.MonkeyPatch) -> int:
    """Rank features (Top-K) with a 20-round model instead of the app's 120 rounds; returns the round count.

    Like ``profile="test"`` for the channels: the ranking runs exactly as in the app (same rows, weights, depth,
    gains and progress reporting), on a smaller model, so a test that ranks takes a fraction of a second. The
    real-data tests rank with the full model.
    """
    from graticule import features
    from graticule.models import train

    monkeypatch.setattr(features, "RANK_ROUNDS", QUICK_RANK_ROUNDS)
    monkeypatch.setattr(train, "RANK_ROUNDS", QUICK_RANK_ROUNDS)
    return QUICK_RANK_ROUNDS


#: Numbers the private folders :func:`isolated_settings` makes for tests that do not use ``tmp_path``.
_ISOLATED = itertools.count(1)


@pytest.fixture(scope="session")
def isolation_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """One folder under pytest's temporary root for the private folders of tests that do not use ``tmp_path``."""
    return tmp_path_factory.mktemp("isolated")


@pytest.fixture(autouse=True)
def isolated_settings(request: pytest.FixtureRequest, isolation_root: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point every settings, model and history path at a temporary folder and clear NIDS_DATA_DIR.

    The folder is the test's ``tmp_path`` when the test (or one of its fixtures) uses it; any other test gets a new,
    empty folder of its own under :func:`isolation_root`. Both are private to the test; the second skips pytest's
    numbered-folder bookkeeping, which costs a few milliseconds per test on Windows (seconds over the suite).
    """
    if "tmp_path" in request.fixturenames:
        folder: Path = request.getfixturevalue("tmp_path")
    else:
        folder = isolation_root / f"t{next(_ISOLATED):04d}"
        folder.mkdir()
    monkeypatch.delenv(settings_mod.ENV_DATA_DIR, raising=False)
    monkeypatch.setattr(settings_mod, "SETTINGS_FILE", folder / "local_settings.json")
    monkeypatch.setattr(settings_mod, "MODELS_DIR", folder / "saved_models")
    monkeypatch.setattr(settings_mod, "HISTORY_DIR", folder / "run_history")
    monkeypatch.setenv("GRATICULE_SYNC_TRAINING", "1")
    return folder
