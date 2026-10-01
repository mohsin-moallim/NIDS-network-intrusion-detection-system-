"""The 200,000-row Wednesday benchmark of all five channels (run with ``-m slow``; needs the CIC-IDS2017 folder).

It drives ``scripts/bench.py`` and checks the timing budget: each tree channel must fit in under 90 s in binary mode
and under 180 s in multi-class mode, and the SVM must say that it was trained on a capped subset. Run with ``-s`` to
see the timing tables.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

pytestmark = [pytest.mark.slow, pytest.mark.realdata]
ROOT = Path(__file__).resolve().parents[2]
WEDNESDAY = "Wednesday-workingHours.pcap_ISCX.csv"
LIMITS = {"binary": 90.0, "multiclass": 180.0}


def _bench_module() -> ModuleType:
    """Load scripts/bench.py as a module (the scripts folder is not a package)."""
    spec = importlib.util.spec_from_file_location("graticule_bench_script", ROOT / "scripts" / "bench.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("mode", ["binary", "multiclass"])
def test_wednesday_200k_benchmark(real_data_dir: Path, mode: str) -> None:
    if not (real_data_dir / WEDNESDAY).is_file():
        pytest.skip(f"{WEDNESDAY} is not in the data folder")
    bench = _bench_module()
    result = bench.run_benchmark(data_dir=str(real_data_dir), files=[WEDNESDAY], rows=200_000, mode=mode,
                                 quiet=True)
    print(f"\n{mode}: {result.rows_prepared:,} rows prepared in {result.prepare_seconds:.1f} s; "
          f"{result.rows_train:,} train / {result.rows_test:,} test; {len(result.classes)} classes; "
          f"peak working set {result.peak_working_set_mb or float('nan'):,.0f} MB")
    print(bench.format_table(result))
    rows = {row["channel"]: row for row in result.rows}
    assert all(row["status"] == "ok" for row in rows.values()), {k: r["error"] for k, r in rows.items()}
    for key in ("forest", "xgboost"):
        assert rows[key]["fit_seconds"] < LIMITS[mode], (key, rows[key]["fit_seconds"])
    assert any("cap" in note.lower() for note in rows["svm"]["notes"]), rows["svm"]["notes"]
    assert rows["svm"]["rows_used"] <= 20_000 < result.rows_train
