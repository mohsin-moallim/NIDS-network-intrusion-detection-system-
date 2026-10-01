"""Timing benchmark for 02 Fit: prepare a sample, fit the chosen channels and print one line per channel.

Usage (from the project root)::

    .\\.venv\\Scripts\\python.exe scripts\\bench.py --files Wednesday-workingHours.pcap_ISCX.csv --rows 200000 \\
        --mode binary --channels forest,xgboost,svm,mlp,logreg

The data folder is ``--data-dir`` when given, otherwise the folder configured in the app (the Bench setting, then
the ``NIDS_DATA_DIR`` environment variable). Without a folder the benchmark draws synthetic flows instead. The
table lists, per channel: rows the model was fitted on, fit seconds, prediction throughput on the test rows (flows
per second), accuracy and balanced accuracy; the peak working set of the process is printed underneath (Windows).
No data is written anywhere. Problems with the options or the data (an unknown file or channel, a sample with one
class) end the script with a one-line message and exit code 2 instead of a traceback.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from graticule.data.prepare import DataRequest, PreparedDataset, prepare_dataset, working_set_mb  # noqa: E402
from graticule.features import FEATURE_MODES  # noqa: E402
from graticule.models.train import TrainingRun, TrainRequest, build_training_data, train_all  # noqa: E402
from graticule.models.zoo import MODEL_KEYS  # noqa: E402
from graticule.schema import EXPECTED_BY_NAME  # noqa: E402
from graticule.settings import list_csv_files, load_settings, resolve_data_dir  # noqa: E402


@dataclass
class BenchmarkResult:
    """Everything one benchmark run measured."""

    source: str
    prepare_seconds: float
    data_seconds: float
    rows_prepared: int
    rows_train: int
    rows_test: int
    classes: tuple[str, ...]
    run: TrainingRun
    peak_working_set_mb: float | None
    rows: list[dict[str, Any]] = field(default_factory=list)


class _ConsoleProgress:
    """Prints one line whenever a channel or stage changes status."""

    def __init__(self, quiet: bool) -> None:
        self.quiet = quiet
        self.last: dict[str, str] = {}

    def update(self, key: str, *, status: str | None = None, fraction: float | None = None,
               message: str | None = None) -> None:
        """Print status changes (not every fraction update)."""
        if self.quiet or status is None or self.last.get(key) == status:
            return
        self.last[key] = status
        print(f"  [{time.strftime('%H:%M:%S')}] {key}: {status}" + (f" - {message}" if message else ""), flush=True)


def resolve_folder(data_dir: str | None) -> Path | None:
    """The data folder to use: ``data_dir`` if given, else the app's configured folder, else None (synthetic)."""
    if data_dir:
        folder = Path(data_dir).expanduser()
        if not folder.is_dir():
            raise SystemExit(f"Data folder not found: {folder}")
        return folder
    resolution = resolve_data_dir(load_settings())
    return resolution.path if resolution.usable else None


def prepare(folder: Path | None, files: list[str], rows: int, seed: int, strategy: str,
            merge_web: bool = False) -> tuple[PreparedDataset, str]:
    """Prepare the sample from ``files`` in ``folder`` (all known files when empty), or synthetic flows."""
    if folder is None:
        request = DataRequest(source="synthetic", synthetic_flows=max(rows, 1_000), row_budget=rows, seed=seed,
                              nonfinite_strategy=strategy, merge_web_attacks=merge_web)
        return prepare_dataset(request), "synthetic"
    if not files:
        csvs = list_csv_files(folder)
        known = [p.name for p in csvs if p.name.lower() in EXPECTED_BY_NAME]
        files = known or [p.name for p in csvs]
    request = DataRequest(source="cicids", data_dir=str(folder), files=tuple(files), row_budget=rows, seed=seed,
                          nonfinite_strategy=strategy, merge_web_attacks=merge_web)
    return prepare_dataset(request), f"{len(request.files)} file(s) from {folder}"


def run_benchmark(
    *,
    data_dir: str | None = None,
    files: list[str] | None = None,
    rows: int = 200_000,
    mode: str = "binary",
    channels: tuple[str, ...] = MODEL_KEYS,
    feature_mode: str = "curated",
    top_k: int = 20,
    profile: str = "full",
    seed: int = 42,
    svm_cap: int = 20_000,
    strategy: str = "drop",
    quiet: bool = False,
) -> BenchmarkResult:
    """Prepare, build the matrices and fit; returns the measurements (prints progress unless ``quiet``).

    The fit options are checked before any data is read, so a bad option fails at once (``ValueError``).
    """
    request = TrainRequest(mode=mode, feature_mode=feature_mode, top_k=top_k, channels=tuple(channels),  # type: ignore[arg-type]
                           profile=profile, seed=seed, svm_cap=svm_cap)  # type: ignore[arg-type]
    folder = resolve_folder(data_dir)
    started = time.perf_counter()
    prepared, source = prepare(folder, list(files or []), rows, seed, strategy)
    prepare_seconds = time.perf_counter() - started
    if not quiet:
        print(f"Prepared {len(prepared.frame):,} rows ({source}) in {prepare_seconds:.1f} s", flush=True)
    sink = _ConsoleProgress(quiet)
    mark = time.perf_counter()
    data = build_training_data(prepared, request, progress=sink)
    data_seconds = time.perf_counter() - mark
    if not quiet:
        print(f"Matrices: {len(data.y_train):,} train / {len(data.y_test):,} test rows, {data.n_features} features, "
              f"{data.n_classes} classes ({data_seconds:.1f} s)", flush=True)
    run = train_all(data, request, data_request=prepared.request, dataset_fingerprint=prepared.fingerprint,
                    progress=sink)
    memory = working_set_mb()
    table = []
    for key, result in run.channels.items():
        metrics = result.extra.get("metrics", {})
        table.append({
            "channel": key,
            "status": result.status,
            "rows_used": result.rows_used,
            "fit_seconds": result.fit_seconds,
            "flows_per_second": result.extra.get("flows_per_second"),
            "accuracy": metrics.get("accuracy"),
            "balanced_accuracy": metrics.get("balanced_accuracy"),
            "notes": list(result.notes),
            "error": result.error,
        })
    return BenchmarkResult(
        source=source, prepare_seconds=prepare_seconds, data_seconds=data_seconds, rows_prepared=len(prepared.frame),
        rows_train=len(data.y_train), rows_test=len(data.y_test), classes=data.classes, run=run,
        peak_working_set_mb=memory[1] if memory else None, rows=table,
    )


def format_table(result: BenchmarkResult) -> str:
    """The per-channel table as fixed-width text."""
    header = f"{'channel':<9} {'status':<9} {'rows used':>10} {'fit s':>8} {'predict flows/s':>16} " \
             f"{'accuracy':>9} {'bal. acc.':>9}"
    lines = [header, "-" * len(header)]
    for row in result.rows:
        def number(value: float | None, spec: str) -> str:
            return format(value, spec) if isinstance(value, (int, float)) else "-"

        lines.append(
            f"{row['channel']:<9} {row['status']:<9} {row['rows_used']:>10,} {row['fit_seconds']:>8.1f} "
            f"{number(row['flows_per_second'], ',.0f'):>16} {number(row['accuracy'], '.4f'):>9} "
            f"{number(row['balanced_accuracy'], '.4f'):>9}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Time the Graticule channels on a prepared sample.")
    parser.add_argument("--files", default="", help="Comma-separated CSV names in the data folder (default: all).")
    parser.add_argument("--rows", type=int, default=200_000, help="Row budget of the sample.")
    parser.add_argument("--mode", choices=("binary", "multiclass"), default="binary")
    parser.add_argument("--channels", default=",".join(MODEL_KEYS), help="Comma-separated channel keys.")
    parser.add_argument("--feature-mode", choices=FEATURE_MODES, default="curated")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--profile", choices=("full", "test"), default="full")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--svm-cap", type=int, default=20_000)
    parser.add_argument("--strategy", choices=("drop", "impute", "recompute"), default="drop")
    parser.add_argument("--data-dir", default=None, help="Data folder (default: the app setting or NIDS_DATA_DIR).")
    parser.add_argument("--quiet", action="store_true", help="Print only the final table.")
    args = parser.parse_args(argv)
    files = [f.strip() for f in args.files.split(",") if f.strip()]
    channels = tuple(c.strip() for c in args.channels.split(",") if c.strip())
    try:
        result = run_benchmark(data_dir=args.data_dir, files=files, rows=args.rows, mode=args.mode,
                               channels=channels, feature_mode=args.feature_mode, top_k=args.top_k,
                               profile=args.profile, seed=args.seed, svm_cap=args.svm_cap, strategy=args.strategy,
                               quiet=args.quiet)
    except ValueError as exc:  # DataFileError and SingleClassError are ValueErrors too
        print(f"bench.py: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(2) from None
    print()
    print(f"Source: {result.source}; mode {args.mode}; {result.rows_prepared:,} rows prepared "
          f"({result.prepare_seconds:.1f} s); {result.rows_train:,} train / {result.rows_test:,} test; "
          f"{len(result.classes)} classes; matrices {result.data_seconds:.1f} s")
    print(format_table(result))
    for row in result.rows:
        for note in row["notes"]:
            print(f"  {row['channel']}: {note}")
        if row["error"]:
            print(f"  {row['channel']}: ERROR {row['error']}")
    peak = result.peak_working_set_mb
    print(f"Peak working set: {peak:,.0f} MB" if peak is not None else "Peak working set: not available")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
