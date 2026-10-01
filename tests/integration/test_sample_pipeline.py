"""01 Sample with the real synthetic generator, and the hand-over from a prepared sample to feature selection.

These tests join modules written separately: the generator (``graticule.data.synthetic``), the preparation procedure
(``graticule.data.prepare``), the degenerate-column check (``graticule.data.clean``), the class/split helpers
(``graticule.data.sampling``) and feature selection (``graticule.features``). Nothing is written to disk.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from graticule import schema
from graticule.data import prepare
from graticule.data.clean import find_degenerate_columns
from graticule.data.prepare import DataRequest, PreparedDataset, prepare_dataset
from graticule.data.sampling import class_order, stratified_split, target_for_mode
from graticule.data.synthetic import SYNTHETIC_CLASSES, generate
from graticule.features import rank_features, select_features
from graticule.schema import FEATURES, LABEL, RATE_COLUMNS

pytestmark = pytest.mark.integration
PORT = schema.DESTINATION_PORT
FLOWS, BUDGET, SEED, SHARE = 6_000, 3_000, 11, 0.35


def _request(strategy: str = "drop", **changes: object) -> DataRequest:
    base: dict[str, object] = dict(source="synthetic", synthetic_flows=FLOWS, row_budget=BUDGET, seed=SEED,
                                   synthetic_attack_share=SHARE, nonfinite_strategy=strategy)
    base.update(changes)
    return DataRequest(**base)  # type: ignore[arg-type]


@pytest.fixture(scope="module")
def generated() -> pd.DataFrame:
    """The flows the prepared samples below are drawn from (same arguments as the request)."""
    return generate(FLOWS, seed=SEED, attack_share=SHARE)


@pytest.fixture(scope="module")
def prepared() -> dict[str, PreparedDataset]:
    """One prepared synthetic sample per bad-value strategy."""
    return {strategy: prepare_dataset(_request(strategy)) for strategy in ("drop", "impute", "recompute")}


# ---------------------------------------------------------------------------------------------- generator → prepare
@pytest.mark.parametrize("strategy", ["drop", "impute", "recompute"])
def test_synthetic_sample_has_the_prepared_layout(prepared: dict[str, PreparedDataset], strategy: str) -> None:
    ds = prepared[strategy]
    frame = ds.frame
    assert list(frame.columns) == [*FEATURES, LABEL, prepare.FILE_COL, prepare.ROW_COL]
    assert all(frame[c].dtype == np.float32 for c in FEATURES)
    assert pd.api.types.is_string_dtype(frame[LABEL])
    assert list(frame[prepare.FILE_COL].cat.categories) == [prepare.SYNTHETIC_FILE]
    assert frame[prepare.ROW_COL].dtype == np.int32
    assert frame[prepare.ROW_COL].is_unique and frame[prepare.ROW_COL].between(0, FLOWS - 1).all()
    assert ds.rows_read == FLOWS and ds.rows_sampled == BUDGET
    assert set(ds.class_counts) == set(SYNTHETIC_CLASSES)
    assert list(ds.class_counts) == class_order(list(ds.class_counts)) and next(iter(ds.class_counts)) == "BENIGN"
    assert sum(ds.class_counts.values()) == BUDGET
    rec = ds.reconciliation()
    assert rec["rows_read"] == sum(v for k, v in rec.items() if k != "rows_read")
    assert not ds.single_class
    table = ds.file_table()
    assert table["Session"].tolist() == ["generated"] and table["Rows in sample"].tolist() == [BUDGET]
    assert ds.class_table()["In sample"].sum() == BUDGET


def test_zero_duration_flows_meet_the_same_bad_value_rules_as_real_files(
    prepared: dict[str, PreparedDataset],
) -> None:
    # Like the recorded files, only the two whole-flow rates are ever non-finite.
    for strategy, ds in prepared.items():
        assert ds.nonfinite.rows_affected > 0, strategy
        assert set(ds.nonfinite.by_column) <= set(RATE_COLUMNS), strategy
    drop, impute, recompute = prepared["drop"], prepared["impute"], prepared["recompute"]
    assert drop.nonfinite.rows_dropped == drop.nonfinite.rows_affected
    assert np.isfinite(drop.features().to_numpy()).all()
    impute_values = impute.features().to_numpy()
    assert not np.isinf(impute_values).any()
    gaps = [c for c in FEATURES if impute.frame[c].isna().any()]
    assert gaps and set(gaps) <= set(RATE_COLUMNS)
    # Rebuilding the rates from the totals and the duration fills every gap, as it does on the real files.
    assert recompute.nonfinite.rows_left_with_gaps == 0
    assert sum(recompute.nonfinite.recomputed.values()) > 0
    assert np.isfinite(recompute.features().to_numpy()).all()


@pytest.mark.parametrize("strategy", ["drop", "impute", "recompute"])
def test_provenance_points_back_to_the_generated_flow(
    prepared: dict[str, PreparedDataset], generated: pd.DataFrame, strategy: str
) -> None:
    ds = prepared[strategy]
    rows = ds.frame[prepare.ROW_COL].to_numpy()
    source = generated.iloc[rows]
    assert source[LABEL].tolist() == ds.frame[LABEL].tolist()
    untouched = [c for c in FEATURES if c not in RATE_COLUMNS] if strategy != "drop" else list(FEATURES)
    np.testing.assert_array_equal(ds.frame[untouched].to_numpy(), source[untouched].to_numpy())
    if strategy == "recompute":
        duration = np.maximum(source["Flow Duration"].to_numpy(np.float64), 1.0)
        packets = (source["Total Fwd Packets"].to_numpy(np.float64)
                   + source["Total Backward Packets"].to_numpy(np.float64))
        rebuilt = ~np.isfinite(source["Flow Packets/s"].to_numpy())
        assert rebuilt.any()
        np.testing.assert_allclose(ds.frame["Flow Packets/s"].to_numpy()[rebuilt],
                                   (packets / duration * 1e6)[rebuilt], rtol=1e-6)


def test_synthetic_preparation_is_deterministic(prepared: dict[str, PreparedDataset]) -> None:
    again = prepare_dataset(_request("drop"))
    assert again.fingerprint == prepared["drop"].fingerprint
    pd.testing.assert_frame_equal(again.frame, prepared["drop"].frame)
    other = prepare_dataset(_request("drop", seed=SEED + 1))
    assert other.fingerprint != prepared["drop"].fingerprint


def test_small_synthetic_request_keeps_every_row() -> None:
    ds = prepare_dataset(DataRequest(source="synthetic", synthetic_flows=2_000, seed=5))
    assert not ds.sampling.sampled and ds.rows_sampled == ds.rows_kept
    assert set(ds.class_counts) == set(SYNTHETIC_CLASSES)


# ---------------------------------------------------------------------------------------------- prepare → features
def _no_constant_or_copied_columns(frame: pd.DataFrame, columns: tuple[str, ...]) -> None:
    """Every chosen column varies in ``frame`` and no two chosen columns are identical."""
    values = frame.loc[:, list(columns)].to_numpy(dtype=np.float64, na_value=np.nan)
    for i, name in enumerate(columns):
        column = values[:, i]
        assert not np.array_equal(column, np.full_like(column, column[0]), equal_nan=True), f"{name} is constant"
        for j in range(i):
            assert not np.array_equal(column, values[:, j], equal_nan=True), f"{name} repeats {columns[j]}"


def test_degenerate_report_of_a_synthetic_sample_feeds_the_feature_sets(prepared: dict[str, PreparedDataset]) -> None:
    ds = prepared["drop"]
    report = ds.degenerate
    bulk = [c for c in FEATURES if "Bulk" in c]
    assert set(bulk) <= set(report.constant)
    assert report.duplicate_of["Subflow Fwd Packets"] == "Total Fwd Packets"

    everything = select_features("all", degenerate=report)
    assert everything.columns == tuple(c for c in FEATURES if c != PORT and c not in report.excluded)
    assert select_features("all", degenerate=report.excluded).columns == everything.columns
    assert set(everything.dropped_degenerate) == set(report.excluded) - {PORT}
    _no_constant_or_copied_columns(ds.frame, everything.columns)

    curated = select_features("curated", degenerate=report)
    assert set(curated.columns) <= set(schema.CURATED) and PORT not in curated.columns
    _no_constant_or_copied_columns(ds.frame, curated.columns)

    with_port = select_features("all", degenerate=report, include_port=True)
    assert with_port.columns == (*everything.columns, PORT) and with_port.include_port


def test_a_curated_copy_of_a_non_curated_column_is_kept() -> None:
    # In the recorded files SYN Flag Count is an exact copy of Fwd PSH Flags (and ECE of RST Flag Count).
    rng = np.random.default_rng(3)
    frame = pd.DataFrame(rng.integers(0, 50, size=(200, len(FEATURES))).astype(np.float32), columns=list(FEATURES))
    frame["SYN Flag Count"] = frame["Fwd PSH Flags"]
    frame["ECE Flag Count"] = frame["RST Flag Count"]
    frame["Bwd PSH Flags"] = np.float32(0)
    report = find_degenerate_columns(frame, FEATURES)
    assert report.duplicate_of == {"SYN Flag Count": "Fwd PSH Flags", "ECE Flag Count": "RST Flag Count"}
    assert report.constant == ["Bwd PSH Flags"]

    curated = select_features("curated", degenerate=report)
    assert "SYN Flag Count" in curated.columns and len(curated.columns) == len(schema.CURATED)
    assert curated.dropped_degenerate == ()
    everything = select_features("all", degenerate=report)
    assert {"SYN Flag Count", "ECE Flag Count", "Bwd PSH Flags"}.isdisjoint(everything.columns)
    assert {"Fwd PSH Flags", "RST Flag Count"} <= set(everything.columns)
    assert everything.dropped_degenerate == ("Bwd PSH Flags", "SYN Flag Count", "ECE Flag Count")
    # A plain list of names is applied bluntly: every listed column is left out, even from the curated set.
    assert "SYN Flag Count" not in select_features("curated", degenerate=report.excluded).columns
    # Top-K keeps whichever copy ranks first and skips the other.
    ranking = ["SYN Flag Count", "Bwd PSH Flags", "Fwd PSH Flags", "Flow Duration", "Idle Mean"]
    top = select_features("topk", ranking=ranking, k=3, degenerate=report)
    assert top.columns == ("SYN Flag Count", "Flow Duration", "Idle Mean")
    assert top.dropped_degenerate == ("Bwd PSH Flags", "Fwd PSH Flags")


@pytest.mark.parametrize("mode", ["binary", "multiclass"])
def test_sample_to_training_matrix(prepared: dict[str, PreparedDataset], mode: str) -> None:
    """The steps 02 Fit will take: target, split, ranking on the training part only, top-K selection."""
    ds = prepared["drop"]
    target = target_for_mode(ds.labels(), mode, min_class_count=50)  # type: ignore[arg-type]
    expected = ["Normal", "Attack"] if mode == "binary" else list(SYNTHETIC_CLASSES[:1]) + sorted(
        SYNTHETIC_CLASSES[1:], key=str.lower)
    assert target.classes == expected and target.keep.all()
    split = stratified_split(target.codes, 0.25, SEED)
    assert np.intersect1d(split.train, split.test).size == 0
    assert split.train.size + split.test.size == ds.rows_sampled
    for code in range(target.n_classes):
        assert (target.codes[split.test] == code).sum() >= 2 and (target.codes[split.train] == code).sum() >= 2

    pool = select_features("all", degenerate=ds.degenerate)
    train_rows = ds.frame.iloc[split.train]
    ranking = rank_features(train_rows[list(pool.columns)], target.codes[split.train], pool.columns, seed=SEED,
                            max_rows=1_500)
    assert [name for name, _ in ranking] != [] and {name for name, _ in ranking} == set(pool.columns)
    top = select_features("topk", ranking=ranking, k=10, degenerate=ds.degenerate)
    assert len(top.columns) == 10 and PORT not in top.columns
    assert set(top.columns) <= set(pool.columns)
    X = ds.frame[list(top.columns)].to_numpy(dtype=np.float32)
    assert X.shape == (ds.rows_sampled, 10) and np.isfinite(X).all()
