"""Reading flow CSVs: header fixes, encodings, bad values, labels and uploads."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from graticule.data.reader import DataFileError, read_flow_csv
from graticule.schema import FEATURES, LABEL
from tests.helpers import default_rows, flow_row, make_rows, with_values, write_cic_csv

pytestmark = pytest.mark.unit


def test_header_layout_is_repaired(tmp_path: Path) -> None:
    path = write_cic_csv(tmp_path / "day.csv")
    frame, report = read_flow_csv(path)
    assert list(frame.columns) == [*FEATURES, LABEL]
    assert all(frame[c].dtype == np.float32 for c in FEATURES)
    assert pd.api.types.is_string_dtype(frame[LABEL])
    assert report.duplicate_columns_dropped == ["Fwd Header Length"]
    assert report.duplicate_columns_equal is True
    assert report.missing_features == [] and report.extra_columns == []
    assert report.rows_read == report.rows_kept == len(default_rows())
    assert report.engine == "pyarrow" and report.encoding == "utf-8"


@pytest.mark.parametrize("leading_spaces", [True, False])
def test_header_spacing_does_not_matter(tmp_path: Path, leading_spaces: bool) -> None:
    path = write_cic_csv(tmp_path / "day.csv", leading_spaces=leading_spaces)
    frame, _ = read_flow_csv(path)
    assert list(frame.columns) == [*FEATURES, LABEL]


def test_both_engines_agree(tmp_path: Path) -> None:
    path = write_cic_csv(tmp_path / "day.csv")
    fast, fast_report = read_flow_csv(path, engine="pyarrow")
    slow, slow_report = read_flow_csv(path, engine="c")
    assert fast_report.engine == "pyarrow" and slow_report.engine == "c"
    assert slow_report.duplicate_columns_dropped == ["Fwd Header Length"]  # the C engine renames it ".1"
    pd.testing.assert_frame_equal(fast, slow)


def test_repeated_column_mismatch_is_reported_and_first_copy_kept(tmp_path: Path) -> None:
    rows = make_rows({"BENIGN": 3})
    path = write_cic_csv(tmp_path / "day.csv", rows, duplicate_mismatch=True)
    frame, report = read_flow_csv(path)
    assert report.duplicate_columns_equal is False
    assert report.duplicate_columns_mismatched == ["Fwd Header Length"]
    assert frame["Fwd Header Length"].iloc[0] == np.float32(rows[0]["Fwd Header Length"])
    assert report.fixes(include_routine=False) == [
        "repeated Fwd Header Length dropped (it differed from the first copy, which was kept)"
    ]


def test_fixes_list_every_repair_in_plain_words(tmp_path: Path) -> None:
    rows = [with_values(flow_row("BENIGN", 1), {"Flow Duration": "abc"}), flow_row("", 2), flow_row("DoS Hulk", 3)]
    path = write_cic_csv(tmp_path / "day.csv", rows, extra_columns={"Timestamp": "7/7/2017 9:00"})
    _, report = read_flow_csv(path)
    assert report.fixes() == [
        "repeated Fwd Header Length dropped (identical copy)",
        "1 unknown column ignored (Timestamp)",
        "1 text cell in number columns turned into gaps",
        "1 row with an empty label dropped",
    ]
    assert report.fixes(include_routine=False) == report.fixes()[1:]
    _, clean = read_flow_csv(write_cic_csv(tmp_path / "clean.csv", with_duplicate_column=False))
    assert clean.fixes() == []


def test_a_file_held_by_another_program_is_not_called_malformed(tmp_path: Path,
                                                                monkeypatch: pytest.MonkeyPatch) -> None:
    path = write_cic_csv(tmp_path / "Tuesday-WorkingHours.pcap_ISCX.csv")
    calls: list[str] = []

    def locked(*args: object, **kwargs: object) -> pd.DataFrame:
        calls.append(str(kwargs.get("engine")))
        raise PermissionError(13, "Permission denied", str(path))

    monkeypatch.setattr(pd, "read_csv", locked)
    with pytest.raises(DataFileError) as info:
        read_flow_csv(path)
    message = str(info.value)
    assert message.startswith("Tuesday-WorkingHours.pcap_ISCX.csv cannot be opened")
    assert "another program" in message and "as CSV" not in message
    assert calls == ["pyarrow"]  # no pointless retries with other engines or encodings


def test_bad_values_become_inf_and_nan(tmp_path: Path) -> None:
    rows = [with_values(flow_row("BENIGN", 1), {"Flow Bytes/s": float("inf"), "Flow Packets/s": float("-inf")}),
            with_values(flow_row("BENIGN", 2), {"Flow Bytes/s": float("nan"), "Flow Packets/s": None})]
    frame, _ = read_flow_csv(write_cic_csv(tmp_path / "day.csv", rows))
    assert frame["Flow Bytes/s"].iloc[0] == np.inf
    assert frame["Flow Packets/s"].iloc[0] == -np.inf
    assert np.isnan(frame["Flow Bytes/s"].iloc[1]) and np.isnan(frame["Flow Packets/s"].iloc[1])


def test_non_numeric_text_is_coerced_and_counted(tmp_path: Path) -> None:
    rows = [with_values(flow_row("BENIGN", 1), {"Flow Duration": "abc"}), flow_row("BENIGN", 2)]
    frame, report = read_flow_csv(write_cic_csv(tmp_path / "day.csv", rows))
    assert frame["Flow Duration"].dtype == np.float32
    assert np.isnan(frame["Flow Duration"].iloc[0])
    assert report.non_numeric_cells == 1


def test_replacement_character_labels_are_normalised(tmp_path: Path) -> None:
    path = write_cic_csv(tmp_path / "thu.csv", label_style="fffd")
    assert "�".encode("utf-8") in path.read_bytes()
    frame, report = read_flow_csv(path)
    assert report.encoding == "utf-8"
    assert set(frame[LABEL]) == {"BENIGN", "DoS Hulk", "Web Attack - Brute Force", "Web Attack - XSS"}
    assert report.label_counts["Web Attack - Brute Force"] == 2


@pytest.mark.parametrize("engine", ["auto", "c"])
def test_cp1252_dash_byte_falls_back_to_cp1252(tmp_path: Path, engine: str) -> None:
    path = write_cic_csv(tmp_path / "thu.csv", encoding="cp1252", label_style="cp1252")
    assert b"\x96" in path.read_bytes()
    frame, report = read_flow_csv(path, engine=engine)  # type: ignore[arg-type]
    assert report.encoding == "cp1252"
    assert "Web Attack - Brute Force" in set(frame[LABEL])


def test_latin1_is_the_last_resort(tmp_path: Path) -> None:
    rows = make_rows({"BENIGN": 2}) + [flow_row("Odd\x81Class", 9)]
    path = write_cic_csv(tmp_path / "odd.csv", rows, encoding="latin-1")
    frame, report = read_flow_csv(path)
    assert report.encoding == "latin-1"
    assert "Odd\x81Class" in set(frame[LABEL])


def test_byte_order_mark_is_ignored(tmp_path: Path) -> None:
    path = write_cic_csv(tmp_path / "bom.csv", bom=True, leading_spaces=False)
    assert path.read_bytes().startswith(b"\xef\xbb\xbf")
    for engine in ("pyarrow", "c"):
        frame, report = read_flow_csv(path, engine=engine)  # type: ignore[arg-type]
        assert report.missing_features == []
        assert frame.columns[0] == "Destination Port"


def test_empty_labels_are_dropped_and_index_keeps_row_positions(tmp_path: Path) -> None:
    rows = make_rows({"BENIGN": 5})
    rows[1] = with_values(rows[1], {LABEL: ""})
    rows[3] = with_values(rows[3], {LABEL: None})
    frame, report = read_flow_csv(write_cic_csv(tmp_path / "day.csv", rows))
    assert report.empty_labels == 2
    assert report.rows_read == 5 and report.rows_kept == 3
    assert frame.index.tolist() == [0, 2, 4]
    # Uploads to be scored keep unlabelled rows.
    upload, upload_report = read_flow_csv(write_cic_csv(tmp_path / "up.csv", rows), require_label=False)
    assert len(upload) == 5 and upload_report.empty_labels == 2
    assert upload[LABEL].tolist()[1] == "" and upload_report.label_counts == {"BENIGN": 3}


def test_uploads_from_bytes_and_file_objects(tmp_path: Path) -> None:
    data = write_cic_csv(tmp_path / "up.csv").read_bytes()
    from_bytes, report_a = read_flow_csv(data, name="upload.csv")
    from_handle, report_b = read_flow_csv(io.BytesIO(data))
    pd.testing.assert_frame_equal(from_bytes, from_handle)
    assert report_a.name == "upload.csv"


def test_upload_without_label_is_allowed_when_not_required(tmp_path: Path) -> None:
    data = write_cic_csv(tmp_path / "up.csv", include_label=False).read_bytes()
    frame, report = read_flow_csv(data, require_label=False)
    assert not report.has_label and LABEL not in frame.columns
    assert list(frame.columns) == list(FEATURES)
    with pytest.raises(DataFileError, match="no Label column"):
        read_flow_csv(data, require_label=True)


def test_missing_and_extra_columns_are_reported(tmp_path: Path) -> None:
    dropped = ["Destination Port", "Idle Min"]
    path = write_cic_csv(tmp_path / "up.csv", drop_columns=dropped, extra_columns={"Flow ID": "1-2-3"})
    frame, report = read_flow_csv(path, require_label=False)
    assert report.missing_features == dropped
    assert report.extra_columns == ["Flow ID"]
    assert report.features_found == len(FEATURES) - 2
    assert list(frame.columns) == [f for f in FEATURES if f not in dropped] + [LABEL]


def test_nrows_reads_only_the_first_rows(tmp_path: Path) -> None:
    path = write_cic_csv(tmp_path / "day.csv", extra_rows=20)
    frame, report = read_flow_csv(path, nrows=5)
    assert len(frame) == 5 and report.engine == "c"


def test_missing_file_and_unreadable_content(tmp_path: Path) -> None:
    with pytest.raises(DataFileError, match="not found"):
        read_flow_csv(tmp_path / "nope.csv")
    with pytest.raises(DataFileError):
        read_flow_csv(b"")
