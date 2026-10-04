"""The whole 01 Sample procedure on small fixture folders: order of steps, counts, provenance and determinism."""

from __future__ import annotations

import shutil
import sys
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from graticule.data import prepare
from graticule.data.prepare import DataRequest, prepare_dataset, read_source_file, stage_rows
from graticule.data.reader import DataFileError, FileReadReport
from graticule.data.sampling import SingleClassError, target_for_mode
from graticule.schema import FEATURES, LABEL
from tests.helpers import fake_generator, flow_row, make_rows, with_values, write_cic_csv

pytestmark = pytest.mark.unit
MON = "Monday-WorkingHours.pcap_ISCX.csv"
TUE = "Tuesday-WorkingHours.pcap_ISCX.csv"
THU = "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv"
INF = float("inf")


def build_folder(root: Path) -> dict[str, list[dict[str, object]]]:
    """Three small files with known problems:

    Monday: 40 normal rows, rows 40, 41 repeat row 3, row 43 repeats row 7, row 42 has infinite rates.
    Tuesday: 30 FTP-Patator rows, row 30 repeats Monday row 5, row 31 has Monday row 7's features but another label.
    Thursday: 10 normal rows and 38 Web Attack rows (Windows-1252 file, dash byte 0x96 in the labels).
    """
    mon = make_rows({"BENIGN": 40}, start=0)
    mon += [dict(mon[3]), dict(mon[3])]
    mon.append(with_values(flow_row("BENIGN", 500), {"Flow Bytes/s": INF, "Flow Packets/s": INF}))
    mon.append(dict(mon[7]))
    tue = make_rows({"FTP-Patator": 30}, start=100)
    tue.append(dict(mon[5]))
    tue.append(with_values(mon[7], {LABEL: "FTP-Patator"}))
    thu = make_rows({"BENIGN": 10, "Web Attack - Brute Force": 15, "Web Attack - XSS": 12,
                     "Web Attack - Sql Injection": 11}, start=200)
    write_cic_csv(root / MON, mon)
    write_cic_csv(root / TUE, tue)
    write_cic_csv(root / THU, thu, encoding="cp1252", label_style="cp1252")
    return {MON: mon, TUE: tue, THU: thu}


@pytest.fixture(scope="module")
def default_files(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, dict[str, list[dict[str, object]]]]:
    """The three files of :func:`build_folder` written once for the checks that only read them, with their rows."""
    root = tmp_path_factory.mktemp("prepare_default") / "data"
    return root, build_folder(root)


@pytest.fixture
def folder(default_files: tuple[Path, dict[str, list[dict[str, object]]]]) -> Path:
    """The folder of :func:`build_folder`'s three files, shared by the checks that only read it (written once)."""
    return default_files[0]


@pytest.fixture(scope="module")
def default_sample(default_files: tuple[Path, dict[str, list[dict[str, object]]]]) -> prepare.PreparedDataset:
    """The three files prepared with every option at its default, once for the checks that only read the result."""
    return prepare_dataset(_request(default_files[0]))


def _request(folder: Path, **changes: object) -> DataRequest:
    base = dict(source="cicids", data_dir=str(folder), files=(MON, TUE, THU))
    base.update(changes)
    return DataRequest(**base)  # type: ignore[arg-type]


def test_counts_reconcile(default_sample: prepare.PreparedDataset) -> None:
    ds = default_sample
    rec = ds.reconciliation()
    assert rec == {
        "rows_read": 44 + 32 + 48, "empty_labels": 0, "bad_value_rows_dropped": 1, "duplicates_within_files": 3,
        "duplicates_across_files": 1, "conflicting_rows_removed": 0, "removed_by_sampling": 0, "rows_sampled": 119,
    }
    assert rec["rows_read"] == sum(v for k, v in rec.items() if k != "rows_read")
    assert ds.within_duplicates.by_class == {"BENIGN": 3}
    assert ds.across_duplicates.by_class == {"BENIGN": 1}
    assert ds.nonfinite.by_class == {"BENIGN": 1}
    assert ds.conflicts.groups == 1 and ds.conflicts.rows == 2
    assert ds.class_counts == {"BENIGN": 50, "FTP-Patator": 31, "Web Attack - Brute Force": 15,
                               "Web Attack - Sql Injection": 11, "Web Attack - XSS": 12}
    assert not ds.single_class and ds.single_class_message is None
    assert ds.rows_kept == 119 and not ds.sampling.sampled


def test_frame_layout_and_provenance(default_files: tuple[Path, dict[str, list[dict[str, object]]]],
                                     default_sample: prepare.PreparedDataset) -> None:
    rows = default_files[1]
    ds = default_sample
    frame = ds.frame
    assert list(frame.columns) == [*FEATURES, LABEL, prepare.FILE_COL, prepare.ROW_COL]
    assert all(frame[c].dtype == np.float32 for c in FEATURES)
    assert frame[prepare.ROW_COL].dtype == np.int32
    assert list(frame[prepare.FILE_COL].cat.categories) == [MON, TUE, THU]  # capture order
    kept = {(str(f), int(r)) for f, r in zip(frame[prepare.FILE_COL], frame[prepare.ROW_COL])}
    assert (MON, 3) in kept and not {(MON, 40), (MON, 41), (MON, 43)} & kept  # repeats within Monday
    assert (MON, 42) not in kept  # infinite rates, dropped
    assert (MON, 5) in kept and (TUE, 30) not in kept  # the earliest file keeps a cross-file duplicate
    # Provenance leads back to the exact values written for that row.
    row = frame[(frame[prepare.FILE_COL] == TUE) & (frame[prepare.ROW_COL] == 31)].iloc[0]
    assert row[LABEL] == "FTP-Patator"
    expected = rows[TUE][31]
    for feature in ("Flow Duration", "Total Fwd Packets", "Idle Min"):
        assert row[feature] == np.float32(expected[feature])
    assert set(frame.loc[frame[prepare.FILE_COL] == THU, LABEL]) >= {"Web Attack - Brute Force"}


def test_views_and_tables(default_sample: prepare.PreparedDataset) -> None:
    ds = default_sample
    assert list(ds.features().columns) == list(FEATURES)
    assert ds.labels().equals(ds.frame[LABEL])
    assert list(ds.provenance().columns) == [prepare.FILE_COL, prepare.ROW_COL]
    assert [r["Reading"] for r in ds.summary_rows()][0] == "Rows read"
    table = ds.file_table()
    assert table["File"].tolist() == [MON, TUE, THU]
    assert table["Rows in sample"].tolist() == [40, 31, 48]
    assert table["Encoding"].tolist() == ["utf-8", "utf-8", "cp1252"]
    classes = ds.class_table()
    assert classes.loc[0, "Class"] == "BENIGN" and classes.loc[0, "Kind"] == "Normal"
    assert classes["In sample"].sum() == ds.rows_sampled


def _removed_readings_add_up(ds: prepare.PreparedDataset) -> None:
    """Rows read minus every reading marked as removed equals rows kept; rows kept minus sampling gives the sample."""
    readings = {r["Reading"]: r for r in ds.summary_rows()}
    removed = sum(int(r["Value"]) for r in readings.values() if str(r["Note"]).startswith("rows removed"))
    assert readings["Rows read"]["Value"] - removed == readings["Rows kept"]["Value"]
    assert readings["Rows kept"]["Value"] - ds.sampling.rows_removed == readings["Rows sampled"]["Value"]


def test_headline_readings_reconcile_with_empty_labels_and_conflict_removals(tmp_path: Path) -> None:
    root = tmp_path / "data"
    mon = make_rows({"BENIGN": 40}) + [flow_row("", 900), flow_row("", 901)]
    mon += [with_values(flow_row("BENIGN", 700), {"Flow Bytes/s": INF})]
    tue = make_rows({"FTP-Patator": 30}, start=100) + [with_values(mon[7], {LABEL: "FTP-Patator"})]
    write_cic_csv(root / MON, mon)
    write_cic_csv(root / TUE, tue)
    for policy in ("keep", "majority", "drop"):
        for strategy in ("drop", "impute"):
            ds = prepare_dataset(DataRequest(source="cicids", data_dir=str(root), files=(MON, TUE),
                                             conflict_policy=policy, nonfinite_strategy=strategy, row_budget=50))
            readings = {r["Reading"]: r["Value"] for r in ds.summary_rows()}
            assert readings["Empty labels"] == 2
            assert ("Conflicting rows" in readings) == (policy != "keep")
            _removed_readings_add_up(ds)
    ds = prepare_dataset(DataRequest(source="cicids", data_dir=str(root), files=(MON, TUE), conflict_policy="drop"))
    assert {r["Reading"]: r["Value"] for r in ds.summary_rows()}["Conflicting rows"] == 2
    clean = prepare_dataset(DataRequest(source="cicids", data_dir=str(root), files=(TUE,)))
    assert "Empty labels" not in {r["Reading"] for r in clean.summary_rows()}


def test_file_table_lists_the_repairs_made_while_reading(tmp_path: Path) -> None:
    root = tmp_path / "data"
    rows = make_rows({"BENIGN": 20, "DoS Hulk": 20})
    rows[3] = with_values(rows[3], {"Flow IAT Mean": "abc"})
    rows += [flow_row("", 500)]
    write_cic_csv(root / TUE, rows, duplicate_mismatch=True, extra_columns={"Timestamp": "7/4/2017 9:00"})
    write_cic_csv(root / MON, make_rows({"BENIGN": 10}, start=300))
    ds = prepare_dataset(DataRequest(source="cicids", data_dir=str(root), files=(MON, TUE)))
    fixes = dict(zip(ds.file_table()["File"], ds.file_table()["File fixes"]))
    assert fixes[MON] == "repeated Fwd Header Length dropped (identical copy)"
    assert fixes[TUE] == ("repeated Fwd Header Length dropped (it differed from the first copy, which was kept); "
                          "1 unknown column ignored (Timestamp); 1 text cell in number columns turned into gaps; "
                          "1 row with an empty label dropped")


@pytest.mark.parametrize(
    ("rows", "strategy", "policy", "message"),
    [
        ([with_values(flow_row("BENIGN", k), {"Flow Bytes/s": INF}) for k in range(10)], "drop", "keep",
         r"of the 10 rows read, 10 held infinite or missing values.*choosing impute or rebuild"),
        ([], "drop", "keep", "The chosen file holds no data rows"),
        ([flow_row("", k) for k in range(6)], "drop", "keep", r"6 had an empty label\. Check that the file's Label"),
        ([flow_row("BENIGN", 1), with_values(flow_row("BENIGN", 1), {LABEL: "Bot"})], "drop", "drop",
         r"2 were identical flows with different labels, removed by the drop policy.*keep policy"),
    ],
    ids=["all-bad-values", "header-only", "empty-labels", "all-conflicting"],
)
def test_no_rows_left_is_a_readable_error_not_an_empty_sample(tmp_path: Path, rows: list[dict[str, object]],
                                                               strategy: str, policy: str, message: str) -> None:
    root = tmp_path / "data"
    write_cic_csv(root / MON, rows)
    request = DataRequest(source="cicids", data_dir=str(root), files=(MON,), nonfinite_strategy=strategy,
                          conflict_policy=policy)
    with pytest.raises(DataFileError, match=message):
        prepare_dataset(request)


def test_class_table_has_fixed_columns_even_when_empty(default_sample: prepare.PreparedDataset) -> None:
    ds = default_sample
    empty = replace(ds, sampling=replace(ds.sampling, before={}, after={}))
    table = empty.class_table()
    assert list(table.columns) == list(prepare.CLASS_TABLE_COLUMNS) and table.empty
    assert list(ds.class_table().columns) == list(prepare.CLASS_TABLE_COLUMNS)


def test_conflict_policies(folder: Path) -> None:
    majority = prepare_dataset(_request(folder, conflict_policy="majority"))
    # Monday held two copies of the normal version, Tuesday one labelled FTP-Patator: the normal label wins.
    assert majority.conflicts.rows_removed == 1
    assert majority.class_counts["FTP-Patator"] == 30
    dropped = prepare_dataset(_request(folder, conflict_policy="drop"))
    assert dropped.conflicts.rows_removed == 2
    rec = dropped.reconciliation()
    assert rec["rows_read"] == sum(v for k, v in rec.items() if k != "rows_read")


def test_web_attack_merge_keeps_detailed_labels(folder: Path) -> None:
    ds = prepare_dataset(_request(folder, merge_web_attacks=True))
    assert ds.class_counts == {"BENIGN": 50, "FTP-Patator": 31, "Web Attack": 38}
    assert set(ds.labels()) == {"BENIGN", "FTP-Patator", "Web Attack"}
    assert {"Web Attack - XSS", "Web Attack - Sql Injection"} <= set(ds.labels(detailed=True))
    assert "Web Attack - XSS" in set(ds.frame[LABEL])


def test_sampling_is_rare_aware_and_deterministic(folder: Path) -> None:
    ds = prepare_dataset(_request(folder, row_budget=60, seed=11))
    assert ds.rows_sampled == 60 and ds.sampling.sampled
    assert ds.sampling.floor_shrunk and ds.sampling.floor == 12  # five classes share 60 rows
    assert ds.class_counts["Web Attack - Sql Injection"] == 11  # smaller than the floor: kept whole
    again = prepare_dataset(_request(folder, row_budget=60, seed=11))
    assert again.fingerprint == ds.fingerprint
    pd.testing.assert_frame_equal(again.frame, ds.frame)
    other = prepare_dataset(_request(folder, row_budget=60, seed=12))
    assert other.fingerprint != ds.fingerprint
    rec = ds.reconciliation()
    assert rec["removed_by_sampling"] == 119 - 60


def test_fingerprint_ignores_where_the_folder_is(folder: Path, default_sample: prepare.PreparedDataset,
                                                 tmp_path: Path) -> None:
    moved = tmp_path / "moved"
    shutil.copytree(folder, moved)
    assert default_sample.request.data_dir == str(folder)  # the folder's own sample, prepared once for the module
    assert default_sample.fingerprint == prepare_dataset(_request(moved)).fingerprint


def test_file_order_is_canonical(folder: Path) -> None:
    assert _request(folder, files=(THU, MON, TUE)).files == (MON, TUE, THU)
    assert DataRequest(source="synthetic", files=(MON,), data_dir="x").files == ()


def test_injected_reader_stager_and_progress(folder: Path) -> None:
    reads: list[str] = []
    stages: list[tuple[str, str]] = []
    events: list[tuple[str, float]] = []

    def reader(path: Path) -> tuple[pd.DataFrame, FileReadReport]:
        reads.append(path.name)
        return read_source_file(path)

    def stager(path: Path, frame: pd.DataFrame, report: FileReadReport, strategy: str) -> prepare.FileStage:
        stages.append((path.name, strategy))
        return stage_rows(frame, report, strategy)

    prepare_dataset(_request(folder), read_file=reader, stage_file=stager,
                    progress=lambda m, f: events.append((m, f)))
    assert reads == [MON, TUE, THU]
    assert stages == [(MON, "drop"), (TUE, "drop"), (THU, "drop")]
    fractions = [f for _, f in events]
    assert fractions == sorted(fractions) and fractions[-1] == 1.0
    assert sum("rows kept" in m for m, _ in events) == 3
    assert any(m.startswith("Checking") for m, _ in events)  # progress inside each file, not only after it


def test_shared_reads_serve_every_strategy_unchanged(folder: Path, default_sample: prepare.PreparedDataset) -> None:
    """One read per file serves all strategies: the frames are never modified, and results match fresh reads."""
    cache: dict[str, tuple[pd.DataFrame, FileReadReport]] = {}

    def reader(path: Path) -> tuple[pd.DataFrame, FileReadReport]:
        if path.name not in cache:
            cache[path.name] = read_source_file(path)
        return cache[path.name]

    first = prepare_dataset(_request(folder, row_budget=50), read_file=reader)
    snapshot = {name: frame.copy() for name, (frame, _) in cache.items()}
    second = prepare_dataset(_request(folder, row_budget=50), read_file=reader)
    assert first.fingerprint == second.fingerprint
    assert default_sample.request == _request(folder)  # "drop" with every other option at its default
    for strategy in ("impute", "recompute", "drop"):
        shared = prepare_dataset(_request(folder, nonfinite_strategy=strategy), read_file=reader)
        # A fresh preparation without the shared reads (the default one is prepared once for the module).
        fresh = default_sample if strategy == "drop" else prepare_dataset(_request(folder, nonfinite_strategy=strategy))
        pd.testing.assert_frame_equal(shared.frame, fresh.frame)
        assert shared.reconciliation() == fresh.reconciliation()
    for name, (frame, _) in cache.items():
        pd.testing.assert_frame_equal(frame, snapshot[name])


def test_per_file_seconds_belong_to_this_draw(folder: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A reused read reports this draw's time and is marked cached, instead of repeating the first read's time.

    The preparation runs on a stand-in clock that only a fresh read moves (by 5 s), so the check does not depend
    on how busy the machine is.
    """
    clock = {"now": 1_000.0}
    monkeypatch.setattr(prepare, "time", SimpleNamespace(perf_counter=lambda: clock["now"]))
    cache: dict[str, tuple[pd.DataFrame, FileReadReport]] = {}

    def reader(path: Path) -> tuple[pd.DataFrame, FileReadReport]:
        if path.name not in cache:
            frame, report = read_source_file(path)
            clock["now"] += 5.0  # a slow file, timed inside the "read"
            cache[path.name] = (frame, replace(report, seconds=5.0))
        return cache[path.name]

    first = prepare_dataset(_request(folder), read_file=reader)
    second = prepare_dataset(_request(folder), read_file=reader)
    assert not any(t.reused for t in first.file_timings.values())
    assert all(t.reused for t in second.file_timings.values())
    table = second.file_table()
    assert all(engine.endswith("(cached)") for engine in table["Engine"])
    assert (table["Seconds"] < 5.0).all() and (first.file_table()["Seconds"] >= 5.0).all()
    assert all(not engine.endswith("(cached)") for engine in first.file_table()["Engine"])


def test_stage_rows_keeps_positions_not_copies(folder: Path) -> None:
    frame, report = read_source_file(folder / MON)
    stage = stage_rows(frame, report, "drop")
    assert stage.positions.dtype == np.int64 and np.all(np.diff(stage.positions) > 0)
    assert stage.report.rows_out == len(stage.positions) == 40  # 44 rows - 3 repeats - 1 with infinite rates
    assert stage.report.nonfinite.rows_before == 44 and stage.report.nonfinite.rows_after == 43
    assert 42 not in stage.positions and not {40, 41, 43} & set(stage.positions.tolist())
    kept = stage_rows(frame, report, "impute")
    assert 42 in kept.positions and kept.report.nonfinite.rows_after == 44
    assert stage.copies[list(stage.positions).index(3)] == 3  # row 3 stood for itself and two repeats
    with pytest.raises(ValueError):
        stage_rows(frame, report, "zero")


def test_shared_row_hashes_give_the_same_stage_and_stay_unchanged(folder: Path) -> None:
    frame, report = read_source_file(folder / MON)
    hashes = prepare.hash_rows(frame)
    before = (hashes.features.copy(), hashes.rows.copy())
    for strategy in ("drop", "impute", "recompute"):
        shared = stage_rows(frame, report, strategy, hashes=hashes)
        own = stage_rows(frame, report, strategy)
        np.testing.assert_array_equal(shared.positions, own.positions)
        np.testing.assert_array_equal(shared.feature_hashes, own.feature_hashes)
        np.testing.assert_array_equal(shared.copies, own.copies)
        assert shared.report.duplicates == own.report.duplicates
    np.testing.assert_array_equal(hashes.features, before[0])  # recompute rehashed copies, not the shared arrays
    np.testing.assert_array_equal(hashes.rows, before[1])
    recompute = stage_rows(frame, report, "recompute", hashes=hashes)
    assert recompute.feature_hashes[list(recompute.positions).index(42)] != hashes.features[42]
    with pytest.raises(ValueError, match="do not belong"):
        stage_rows(frame.iloc[:10], report, "drop", hashes=hashes)


def test_benign_only_sample_is_flagged(folder: Path) -> None:
    ds = prepare_dataset(_request(folder, files=(MON,)))
    assert ds.single_class
    assert ds.single_class_message.startswith("Only one class in this sample: every row is BENIGN.")
    with pytest.raises(SingleClassError, match="every row is BENIGN"):
        target_for_mode(ds.labels(), "binary", min_class_count=50)


def test_impute_and_recompute_keep_the_bad_row(folder: Path) -> None:
    impute = prepare_dataset(_request(folder, files=(MON,), nonfinite_strategy="impute"))
    bad = impute.frame[impute.frame[prepare.ROW_COL] == 42]
    assert len(bad) == 1 and np.isnan(bad["Flow Bytes/s"].iloc[0])
    assert impute.nonfinite.rows_dropped == 0 and impute.nonfinite.rows_affected == 1
    recompute = prepare_dataset(_request(folder, files=(MON,), nonfinite_strategy="recompute"))
    fixed = recompute.frame[recompute.frame[prepare.ROW_COL] == 42]
    assert np.isfinite(fixed["Flow Bytes/s"].iloc[0]) and np.isfinite(fixed["Flow Packets/s"].iloc[0])
    assert recompute.nonfinite.recomputed == {"Flow Bytes/s": 1, "Flow Packets/s": 1}


def test_errors_are_readable(tmp_path: Path) -> None:
    folder = tmp_path / "data"
    build_folder(folder)  # a folder of the test's own: a narrow file is added below
    with pytest.raises(DataFileError, match="not found"):
        prepare_dataset(_request(tmp_path / "missing"))
    with pytest.raises(DataFileError, match="not found in the data folder"):
        prepare_dataset(_request(folder, files=("Friday-WorkingHours-Morning.pcap_ISCX.csv",)))
    with pytest.raises(DataFileError, match="at least one file"):
        prepare_dataset(_request(folder, files=()))
    with pytest.raises(DataFileError, match="No data folder"):
        prepare_dataset(DataRequest(source="cicids", files=(MON,)))
    write_cic_csv(folder / "narrow.csv", make_rows({"BENIGN": 3}), drop_columns=["Idle Min", "Idle Max"])
    with pytest.raises(DataFileError, match="lacks 2 of the 77"):
        prepare_dataset(_request(folder, files=("narrow.csv",)))
    with pytest.raises(ValueError):
        DataRequest(nonfinite_strategy="zero")
    with pytest.raises(ValueError):
        DataRequest(conflict_policy="vote")


def test_synthetic_source_runs_the_same_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(prepare, "_load_generator", lambda: fake_generator)
    request = DataRequest(source="synthetic", synthetic_flows=3_000, row_budget=1_000, seed=4)
    ds = prepare_dataset(request)
    assert ds.rows_sampled == 1_000
    assert set(ds.frame[prepare.FILE_COL].astype(str)) == {prepare.SYNTHETIC_FILE}
    assert ds.frame[prepare.ROW_COL].between(0, 2_999).all()
    assert ds.nonfinite.rows_affected == 3 and ds.nonfinite.rows_dropped == 3
    assert "BENIGN" in ds.class_counts and len(ds.class_counts) == 6
    assert prepare_dataset(request).fingerprint == ds.fingerprint
    assert ds.file_table().loc[0, "Engine"] == "generator"


def test_synthetic_progress_moves_during_generation(monkeypatch: pytest.MonkeyPatch) -> None:
    from graticule.data import synthetic

    monkeypatch.setattr(synthetic, "BLOCK_FLOWS", 1_000)
    events: list[tuple[str, float]] = []
    ds = prepare_dataset(DataRequest(source="synthetic", synthetic_flows=3_000, row_budget=2_000, seed=2),
                         progress=lambda m, f: events.append((m, f)))
    generating = [m for m, _ in events if m.startswith("Generating synthetic flows:")]
    assert generating == [f"Generating synthetic flows: {d:,} of 3,000" for d in (1_000, 2_000, 3_000)]
    fractions = [f for _, f in events]
    assert fractions == sorted(fractions) and fractions[-1] == 1.0
    assert any("rows kept" in m for m, _ in events)
    assert ds.file_timings[prepare.SYNTHETIC_FILE].seconds > 0 and not ds.file_timings[prepare.SYNTHETIC_FILE].reused


def test_missing_generator_gives_a_readable_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "graticule.data.synthetic", None)
    with pytest.raises(DataFileError, match="generator is not available"):
        prepare_dataset(DataRequest(source="synthetic", synthetic_flows=2_000))


def test_real_generator_matches_the_agreed_interface() -> None:
    pytest.importorskip("graticule.data.synthetic")
    ds = prepare_dataset(DataRequest(source="synthetic", synthetic_flows=4_000, row_budget=2_000, seed=3))
    assert ds.rows_sampled <= 2_000
    assert "BENIGN" in ds.class_counts and len(ds.class_counts) >= 2
    rec = ds.reconciliation()
    assert rec["rows_read"] == sum(v for k, v in rec.items() if k != "rows_read")


def test_peak_memory_watch() -> None:
    with prepare.PeakMemoryWatch() as watch:
        block = np.ones((2_000, 2_000))
        time.sleep(0.25)  # held long enough for the sampler, in case the process peak was already higher
        del block
    if sys.platform == "win32":
        assert watch.peak_mb is not None and watch.peak_mb > 10
        assert watch.start_mb is not None and watch.peak_mb - watch.start_mb > 25  # the 32 MB block shows as a rise
        assert prepare.working_set_mb() is not None
    else:
        assert watch.peak_mb is None and watch.start_mb is None


def test_memory_rise_is_reported_next_to_the_process_peak(default_sample: prepare.PreparedDataset) -> None:
    ds = default_sample
    if ds.peak_memory_mb is None:
        assert ds.memory_rise_mb is None
    else:
        assert ds.memory_start_mb is not None and 0 <= ds.memory_rise_mb <= ds.peak_memory_mb
