"""Cleaning: bad-value strategies, row hashes and duplicates, conflicting labels, degenerate columns."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from graticule.data import clean
from graticule.schema import FEATURES, LABEL
from tests.helpers import feature_frame, flow_row, make_rows, with_values

pytestmark = pytest.mark.unit
INF = float("inf")


def _with_bad_values() -> pd.DataFrame:
    rows = make_rows({"BENIGN": 4, "DoS Hulk": 2})
    rows[0] = with_values(rows[0], {"Flow Bytes/s": INF, "Flow Packets/s": INF})
    rows[4] = with_values(rows[4], {"Flow Bytes/s": float("nan")})
    rows[5] = with_values(rows[5], {"Idle Mean": -INF})
    return feature_frame(rows)


def test_nonfinite_mask_marks_rows_with_any_bad_value() -> None:
    frame = _with_bad_values()
    assert clean.nonfinite_mask(frame[list(FEATURES)]).tolist() == [True, False, False, False, True, True]
    assert clean.nonfinite_mask(frame[list(FEATURES)].to_numpy()).tolist() == [True, False, False, False, True, True]


def test_drop_strategy_removes_rows_and_counts_per_class_and_column() -> None:
    frame = _with_bad_values()
    before = frame.copy()
    out, report = clean.apply_nonfinite_strategy(frame, "drop")
    assert len(out) == 3 and report.rows_dropped == report.rows_affected == 3
    assert report.by_class == {"DoS Hulk": 2, "BENIGN": 1}
    assert report.by_column == {"Flow Bytes/s": 2, "Flow Packets/s": 1, "Idle Mean": 1}
    assert report.infinite_cells == 3 and report.missing_cells == 1
    assert np.isfinite(out[list(FEATURES)].to_numpy()).all()
    pd.testing.assert_frame_equal(frame, before)  # the input is never modified


def test_impute_strategy_keeps_rows_and_turns_infinities_into_gaps() -> None:
    frame = _with_bad_values()
    out, report = clean.apply_nonfinite_strategy(frame, "impute")
    assert len(out) == len(frame) and report.rows_dropped == 0
    values = out[list(FEATURES)].to_numpy()
    assert not np.isinf(values).any()
    assert np.isnan(out["Flow Bytes/s"].iloc[0]) and np.isnan(out["Idle Mean"].iloc[5])
    assert report.rows_left_with_gaps == 3
    assert np.isinf(frame["Flow Bytes/s"].iloc[0])  # input untouched


def test_recompute_strategy_rebuilds_rates_with_a_one_microsecond_floor() -> None:
    zero = with_values(flow_row("DoS Hulk", 1), {
        "Flow Duration": 0.0, "Total Length of Fwd Packets": 60.0, "Total Length of Bwd Packets": 40.0,
        "Total Fwd Packets": 1.0, "Total Backward Packets": 1.0, "Flow Bytes/s": INF, "Flow Packets/s": INF,
    })
    gap = with_values(flow_row("BENIGN", 2), {
        "Flow Duration": 2_000_000.0, "Total Length of Fwd Packets": 3_000.0, "Total Length of Bwd Packets": 1_000.0,
        "Total Fwd Packets": 3.0, "Total Backward Packets": 1.0, "Flow Bytes/s": float("nan"),
    })
    fine = flow_row("BENIGN", 3)
    other = with_values(flow_row("BENIGN", 4), {"Idle Max": INF})
    frame = feature_frame([zero, gap, fine, other])
    out, report = clean.apply_nonfinite_strategy(frame, "recompute")
    assert out["Flow Bytes/s"].iloc[0] == pytest.approx(100 / 1 * 1e6)
    assert out["Flow Packets/s"].iloc[0] == pytest.approx(2 / 1 * 1e6)
    assert out["Flow Bytes/s"].iloc[1] == pytest.approx(4_000 / 2_000_000 * 1e6)
    assert out["Flow Bytes/s"].iloc[2] == frame["Flow Bytes/s"].iloc[2]  # finite readings are not touched
    assert np.isnan(out["Idle Max"].iloc[3])  # other columns: gaps for the imputer
    assert report.recomputed == {"Flow Bytes/s": 2, "Flow Packets/s": 1}
    assert report.rows_left_with_gaps == 1 and len(out) == 4


def test_unknown_strategy_is_rejected() -> None:
    with pytest.raises(ValueError):
        clean.apply_nonfinite_strategy(_with_bad_values(), "zero")  # type: ignore[arg-type]


def test_row_hashes_treat_signed_zero_and_all_bad_values_alike() -> None:
    a = with_values(flow_row("BENIGN", 1), {"Idle Std": 0.0, "Flow Bytes/s": INF})
    b = with_values(flow_row("BENIGN", 1), {"Idle Std": -0.0, "Flow Bytes/s": float("nan")})
    c = with_values(flow_row("BENIGN", 1), {"Idle Std": 1.0, "Flow Bytes/s": INF})
    frame = feature_frame([a, b, c])
    hashes = clean.row_hashes(frame, [*FEATURES, LABEL])
    assert hashes.dtype == np.uint64
    assert hashes[0] == hashes[1] != hashes[2]


def test_row_hashes_include_the_label_when_asked() -> None:
    frame = feature_frame([flow_row("BENIGN", 1), flow_row("PortScan", 1)])
    assert clean.row_hashes(frame, FEATURES)[0] == clean.row_hashes(frame, FEATURES)[1]
    assert clean.row_hashes(frame, [*FEATURES, LABEL])[0] != clean.row_hashes(frame, [*FEATURES, LABEL])[1]
    feature_hashes = clean.row_hashes(frame, FEATURES)
    combined = clean.hashes_with_labels(feature_hashes, frame[LABEL])
    assert combined[0] != combined[1]


def test_duplicates_are_removed_keeping_the_first_and_counted_per_class() -> None:
    rows = make_rows({"BENIGN": 3, "DoS Hulk": 2})
    rows += [dict(rows[0]), dict(rows[0]), dict(rows[3])]
    frame = feature_frame(rows)
    out, report = clean.drop_duplicates_by_hash(frame, [*FEATURES, LABEL], "within file")
    assert report.rows_removed == 3 and report.rows_after == 5
    assert report.by_class == {"BENIGN": 2, "DoS Hulk": 1}
    assert out.index.tolist() == [0, 1, 2, 3, 4]


def test_copies_per_kept_row_counts_every_repeat() -> None:
    hashes = np.array([5, 7, 5, 5, 9, 7], dtype=np.uint64)
    keep = clean.first_occurrences(hashes)
    assert keep.tolist() == [True, True, False, False, True, False]
    assert clean.copies_per_kept_row(hashes, keep).tolist() == [3, 2, 1]
    weights = np.array([1, 2, 1, 4, 1, 1])
    assert clean.copies_per_kept_row(hashes, keep, weights).tolist() == [6, 3, 1]


def _conflicting() -> pd.DataFrame:
    shared = flow_row("BENIGN", 1)
    rows = [shared, with_values(shared, {LABEL: "PortScan"}), flow_row("BENIGN", 2), flow_row("PortScan", 3)]
    other = flow_row("BENIGN", 4)
    rows += [other, with_values(other, {LABEL: "DoS Hulk"})]
    return feature_frame(rows)


def test_conflicting_rows_are_counted() -> None:
    report = clean.conflicting_feature_rows(_conflicting(), FEATURES)
    assert report.groups == 2 and report.rows == 4
    assert report.by_class == {"BENIGN": 2, "DoS Hulk": 1, "PortScan": 1}


def test_conflict_policies() -> None:
    frame = _conflicting()
    kept, report = clean.resolve_conflicts(frame, FEATURES, "keep")
    assert len(kept) == 6 and report.rows_removed == 0 and report.policy == "keep"
    dropped, report = clean.resolve_conflicts(frame, FEATURES, "drop")
    assert dropped.index.tolist() == [2, 3] and report.rows_removed == 4
    # Every group ties one-to-one, so there is no majority and the groups go.
    tied, _ = clean.resolve_conflicts(frame, FEATURES, "majority")
    assert tied.index.tolist() == [2, 3]
    # With weights (copies seen before de-duplication) the heavier label wins its group.
    weighted, report = clean.resolve_conflicts(frame, FEATURES, "majority",
                                               weights=np.array([3, 1, 1, 1, 1, 2]))
    assert weighted.index.tolist() == [0, 2, 3, 5] and report.rows_removed == 2


def test_degenerate_columns() -> None:
    frame = feature_frame(make_rows({"BENIGN": 5}))
    frame["Bwd PSH Flags"] = np.float32(0.0)
    frame["Fwd URG Flags"] = np.float32(np.nan)
    frame["CWE Flag Count"] = np.array([np.nan, 1, 1, 1, 1], dtype=np.float32)  # NaN is a value of its own
    frame["Subflow Fwd Packets"] = frame["Total Fwd Packets"]
    report = clean.find_degenerate_columns(frame, FEATURES)
    assert report.constant == ["Bwd PSH Flags", "Fwd URG Flags"]
    assert report.duplicate_of["Subflow Fwd Packets"] == "Total Fwd Packets"
    assert "CWE Flag Count" not in report.constant
    assert set(report.excluded) >= {"Bwd PSH Flags", "Fwd URG Flags", "Subflow Fwd Packets"}
