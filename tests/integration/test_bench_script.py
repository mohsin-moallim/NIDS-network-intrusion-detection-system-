"""``scripts/bench.py`` end to end on synthetic flows (no data folder is configured in tests), with tiny models.

The full 200,000-row timing run lives in ``tests/slow/test_benchmark.py``; this keeps the command line, the table
and the notes working in the default suite.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

pytestmark = pytest.mark.integration
ROOT = Path(__file__).resolve().parents[2]


def _bench_module() -> ModuleType:
    """Load scripts/bench.py as a module (the scripts folder is not a package)."""
    spec = importlib.util.spec_from_file_location("nids_bench_script_quick", ROOT / "scripts" / "bench.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_bench_command_line_prints_the_table_on_synthetic_flows(capsys: pytest.CaptureFixture[str]) -> None:
    bench = _bench_module()
    code = bench.main(["--rows", "2000", "--profile", "test", "--channels", "logreg,forest,svm", "--quiet"])
    assert code == 0
    out = capsys.readouterr().out
    assert "Source: synthetic; mode binary; " in out and " rows prepared (" in out
    lines = {line.split()[0]: line.split() for line in out.splitlines() if line.split()[:2][-1:] == ["ok"]}
    assert list(lines) == ["forest", "svm", "logreg"]  # fixed channel order, whatever order was asked for
    for key, cells in lines.items():
        accuracy, balanced = float(cells[-2]), float(cells[-1])
        assert 0.5 < balanced <= 1.0 and 0.5 < accuracy <= 1.0, key
    assert "svm: Trained on 1,000 of" in out and "(SVM cap 1,000)" in out
    assert "Peak working set:" in out


def test_bench_rejects_a_missing_folder(tmp_path: Path) -> None:
    bench = _bench_module()
    with pytest.raises(SystemExit, match="Data folder not found"):
        bench.main(["--data-dir", str(tmp_path / "nowhere"), "--quiet"])


@pytest.mark.parametrize("args, message", [
    (["--files", "Nope.csv"], "File not found in the data folder: Nope.csv"),
    (["--channels", "knn"], "Unknown channel(s): knn."),
], ids=["unknown-file", "unknown-channel"])
def test_bench_turns_bad_options_into_a_short_message(tmp_path: Path, capsys: pytest.CaptureFixture[str],
                                                      args: list[str], message: str) -> None:
    bench = _bench_module()
    with pytest.raises(SystemExit) as info:
        bench.main([*args, "--data-dir", str(tmp_path), "--rows", "2000", "--profile", "test", "--quiet"])
    assert info.value.code == 2
    err = capsys.readouterr().err
    assert err.strip() == f"bench.py: {message}" and "Traceback" not in err
