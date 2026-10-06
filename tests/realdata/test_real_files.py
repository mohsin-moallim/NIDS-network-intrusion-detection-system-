"""Checks against the real CIC-IDS2017 files (skipped when the folder is not available).

Kept quick: headers are checked on the first rows only, and the full procedure runs on single files. The two tests
that read a whole large file are also marked ``slow``, so the default run stays short; ``-m realdata`` runs them.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nids.data.clean import find_degenerate_columns
from nids.data.prepare import DataRequest, prepare_dataset, read_source_file
from nids.data.reader import FileReadReport, read_flow_csv
from nids.data.sampling import SingleClassError, target_for_mode
from nids.features import select_features
from nids.schema import CURATED, DESTINATION_PORT, EXPECTED_FILES, FEATURES, KNOWN_LABELS, LABEL

pytestmark = pytest.mark.realdata
WEDNESDAY = "Wednesday-workingHours.pcap_ISCX.csv"
MONDAY = "Monday-WorkingHours.pcap_ISCX.csv"
THURSDAY_WEB = "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv"


def _path(folder: Path, name: str) -> Path:
    path = folder / name
    if not path.is_file():
        pytest.skip(f"{name} is not in the data folder")
    return path


@pytest.mark.parametrize("name", [f.name for f in EXPECTED_FILES])
def test_header_is_repaired_in_every_file(real_data_dir: Path, name: str) -> None:
    frame, report = read_flow_csv(_path(real_data_dir, name), nrows=3_000)
    assert list(frame.columns) == [*FEATURES, LABEL]
    assert report.duplicate_columns_dropped == ["Fwd Header Length"]
    assert report.duplicate_columns_equal
    assert report.missing_features == [] and report.extra_columns == []
    assert set(frame[LABEL]) <= set(KNOWN_LABELS)


def test_thursday_web_attack_labels_are_normalised(real_data_dir: Path) -> None:
    frame, report = read_flow_csv(_path(real_data_dir, THURSDAY_WEB))
    assert report.engine == "pyarrow" and report.encoding == "utf-8"
    counts = report.label_counts
    assert counts["Web Attack - Brute Force"] == 1_507
    assert counts["Web Attack - XSS"] == 652
    assert counts["Web Attack - Sql Injection"] == 21
    assert not frame[LABEL].str.contains("�").any()


@pytest.mark.slow  # reads the whole 215 MB Wednesday file three times (about 10 s)
def test_wednesday_prepares_and_reconciles(real_data_dir: Path) -> None:
    _path(real_data_dir, WEDNESDAY)
    reads: dict[str, tuple[pd.DataFrame, FileReadReport]] = {}

    def remembered(path: Path) -> tuple[pd.DataFrame, FileReadReport]:
        if path.name not in reads:
            reads[path.name] = read_source_file(path)
        return reads[path.name]

    request = DataRequest(source="cicids", data_dir=str(real_data_dir), files=(WEDNESDAY,))
    ds = prepare_dataset(request, read_file=remembered)
    rec = ds.reconciliation()
    assert rec["rows_read"] == 692_703
    assert rec["rows_read"] == (ds.rows_sampled + rec["bad_value_rows_dropped"] + rec["duplicates_within_files"]
                                + rec["duplicates_across_files"] + rec["removed_by_sampling"] + rec["empty_labels"]
                                + rec["conflicting_rows_removed"])
    assert ds.rows_sampled == 200_000
    assert ds.nonfinite.by_column["Flow Packets/s"] == 1_297
    assert ds.class_counts["Heartbleed"] == 11  # far below the floor, so every row is kept
    assert all(ds.frame[c].dtype == np.float32 for c in FEATURES)
    assert {"Bwd PSH Flags", "Fwd URG Flags"} <= set(ds.degenerate.constant)
    # Feature sets built from the report: "all" holds no constant or repeated column; the curated set only loses
    # constant columns, never a column that merely copies one outside the curated set (e.g. SYN Flag Count).
    everything = select_features("all", degenerate=ds.degenerate)
    assert DESTINATION_PORT not in everything.columns
    recheck = find_degenerate_columns(ds.frame, everything.columns)
    assert recheck.constant == [] and recheck.duplicate_of == {}
    curated = select_features("curated", degenerate=ds.degenerate)
    assert set(curated.columns) == set(CURATED) - set(ds.degenerate.constant)
    again = prepare_dataset(request, read_file=remembered)
    assert again.fingerprint == ds.fingerprint
    # The second draw reused the read, and its per-file time says so instead of repeating the first read's time.
    assert not ds.file_timings[WEDNESDAY].reused and again.file_timings[WEDNESDAY].reused
    assert again.file_table()["Engine"].tolist() == ["pyarrow (cached)"]
    assert again.file_timings[WEDNESDAY].seconds < ds.file_timings[WEDNESDAY].seconds
    # Another strategy on the same read: the repairs reach only the rows that needed them.
    rebuilt = prepare_dataset(replace(request, nonfinite_strategy="recompute"), read_file=remembered)
    assert rebuilt.nonfinite.rows_dropped == 0 and rebuilt.nonfinite.rows_affected == ds.nonfinite.rows_affected
    assert np.isfinite(rebuilt.frame["Flow Packets/s"].to_numpy()).all()


@pytest.mark.slow  # reads the whole Monday file (about 3 s)
def test_monday_alone_cannot_train_a_binary_detector(real_data_dir: Path) -> None:
    _path(real_data_dir, MONDAY)
    ds = prepare_dataset(DataRequest(source="cicids", data_dir=str(real_data_dir), files=(MONDAY,),
                                     row_budget=20_000))
    assert ds.single_class and set(ds.class_counts) == {"BENIGN"}
    with pytest.raises(SingleClassError) as info:
        target_for_mode(ds.labels(), "binary", min_class_count=50)
    assert "every row is BENIGN" in str(info.value)


#: Rows per label in each published file, as the dataset documents them (duplicates included, labels tidied).
EXPECTED_LABEL_COUNTS: dict[str, dict[str, int]] = {
    "Monday-WorkingHours.pcap_ISCX.csv": {"BENIGN": 529_918},
    "Tuesday-WorkingHours.pcap_ISCX.csv": {"BENIGN": 432_074, "FTP-Patator": 7_938, "SSH-Patator": 5_897},
    "Wednesday-workingHours.pcap_ISCX.csv": {"BENIGN": 440_031, "DoS Hulk": 231_073, "DoS GoldenEye": 10_293,
                                             "DoS slowloris": 5_796, "DoS Slowhttptest": 5_499, "Heartbleed": 11},
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv": {
        "BENIGN": 168_186, "Web Attack - Brute Force": 1_507, "Web Attack - XSS": 652,
        "Web Attack - Sql Injection": 21},
    "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv": {"BENIGN": 288_566, "Infiltration": 36},
    "Friday-WorkingHours-Morning.pcap_ISCX.csv": {"BENIGN": 189_067, "Bot": 1_966},
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv": {"BENIGN": 127_537, "PortScan": 158_930},
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv": {"BENIGN": 97_718, "DDoS": 128_027},
}


def test_expected_label_counts_name_every_file() -> None:
    assert set(EXPECTED_LABEL_COUNTS) == {f.name for f in EXPECTED_FILES}


@pytest.mark.slow  # reads every whole file (about 10 s for all eight)
@pytest.mark.parametrize("name", [f.name for f in EXPECTED_FILES])
def test_every_file_holds_its_documented_label_counts(real_data_dir: Path, name: str) -> None:
    _, report = read_source_file(_path(real_data_dir, name))
    assert report.label_counts == EXPECTED_LABEL_COUNTS[name]
    assert report.empty_labels == 0
