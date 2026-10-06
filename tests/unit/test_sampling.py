"""Class options, targets, the rare-aware sampler and the stratified split."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nids.data import sampling
from nids.data.sampling import SingleClassError

pytestmark = pytest.mark.unit


def _labels(spec: dict[str, int]) -> pd.Series:
    values = [name for name, count in spec.items() for _ in range(count)]
    return pd.Series(values, dtype="str")


def test_web_attack_merge() -> None:
    labels = _labels({"BENIGN": 2, "Web Attack - XSS": 1, "Web Attack - Brute Force": 1, "Bot": 1})
    merged = sampling.apply_class_options(labels, merge_web_attacks=True)
    assert merged.tolist() == ["BENIGN", "BENIGN", "Web Attack", "Web Attack", "Bot"]
    assert sampling.apply_class_options(labels, merge_web_attacks=False).tolist() == labels.tolist()
    assert labels.iloc[2] == "Web Attack - XSS"  # input untouched


def test_class_order_puts_normal_first() -> None:
    assert sampling.class_order(["PortScan", "BENIGN", "Bot", "DDoS"]) == ["BENIGN", "Bot", "DDoS", "PortScan"]
    assert sampling.class_order(["Attack", "Normal"]) == ["Normal", "Attack"]


def test_binary_target() -> None:
    labels = _labels({"BENIGN": 30, "DoS Hulk": 12, "Heartbleed": 5})
    result = sampling.target_for_mode(labels, "binary", min_class_count=50)
    assert result.classes == ["Normal", "Attack"]
    assert result.counts == {"Normal": 30, "Attack": 17}  # min_class_count does not apply in binary mode
    assert result.codes.tolist() == [0] * 30 + [1] * 17
    assert result.keep.all() and not result.dropped


def test_multiclass_target_drops_small_classes_with_reasons() -> None:
    labels = _labels({"BENIGN": 80, "DoS Hulk": 60, "Bot": 20, "Heartbleed": 5})
    result = sampling.target_for_mode(labels, "multiclass", min_class_count=50)
    assert result.classes == ["BENIGN", "DoS Hulk"]
    assert result.dropped == {"Bot": 20, "Heartbleed": 5}
    assert "minimum of 50" in result.dropped_reasons["Bot"]
    assert "hard floor of 10" in result.dropped_reasons["Heartbleed"]
    assert int(result.keep.sum()) == 140 and len(result.target) == 140
    assert set(result.codes.tolist()) == {0, 1}


def test_hard_floor_applies_in_binary_mode() -> None:
    labels = _labels({"BENIGN": 40, "Heartbleed": 4})
    with pytest.raises(SingleClassError) as info:
        sampling.target_for_mode(labels, "binary", min_class_count=50)
    assert info.value.dropped == {"Attack": 4}


def test_benign_only_data_gives_a_friendly_single_class_error() -> None:
    labels = _labels({"BENIGN": 500})
    with pytest.raises(SingleClassError) as info:
        sampling.target_for_mode(labels, "binary", min_class_count=50)
    message = str(info.value)
    assert message.startswith("Only one class in this sample: every row is BENIGN.")
    assert "attack" in message and "synthetic" in message
    assert info.value.present == ["Normal"]


def test_attacks_left_out_for_size_are_not_called_absent() -> None:
    """When attack rows exist but are too few, the message says so and names the right remedy."""
    with pytest.raises(SingleClassError) as info:
        sampling.target_for_mode(_labels({"BENIGN": 20, "Heartbleed": 5}), "binary", min_class_count=50)
    message = str(info.value)
    assert "every row is BENIGN" not in message and "Add a file that contains attacks" not in message
    assert message.startswith("Only one class has enough rows: Normal. The sample does hold attack flows")
    assert "(5 rows)" in message and "hard floor" in message and "Left out: Attack (5 rows" in message

    with pytest.raises(SingleClassError) as info:
        sampling.target_for_mode(_labels({"BENIGN": 200, "Heartbleed": 11}), "multiclass", min_class_count=50)
    message = str(info.value)
    assert message.startswith("Only one class has enough rows: BENIGN.")
    assert "Lower the minimum class count" in message
    assert "Heartbleed (11 rows, below the minimum of 50 rows per class)" in message

    with pytest.raises(SingleClassError) as info:
        sampling.target_for_mode(_labels({"BENIGN": 12, "DoS Hulk": 300}), "multiclass", min_class_count=50)
    message = str(info.value)
    assert message.startswith("Only one class has enough rows: DoS Hulk. The sample does hold normal (BENIGN)")

    with pytest.raises(SingleClassError) as info:
        sampling.target_for_mode(_labels({"Bot": 300, "Heartbleed": 11}), "multiclass", min_class_count=50)
    assert "does hold other classes" in str(info.value)


def test_empty_labels_give_a_plain_message() -> None:
    with pytest.raises(SingleClassError, match="The sample holds no rows to fit on."):
        sampling.target_for_mode(_labels({}), "multiclass", min_class_count=1, hard_floor=1)


def test_default_floor() -> None:
    assert sampling.default_floor(200_000) == 4_000
    assert sampling.default_floor(10_000) == 1_000


def test_sampler_hits_the_budget_exactly_and_protects_rare_classes() -> None:
    labels = _labels({"BENIGN": 20_000, "DoS Hulk": 8_000, "Bot": 900, "Heartbleed": 11})
    positions, report = sampling.sample_positions(labels, budget=10_000, seed=7)
    assert len(positions) == 10_000 == report.rows_after
    assert np.all(np.diff(positions) > 0)
    taken = labels.iloc[positions].value_counts().to_dict()
    assert taken["Heartbleed"] == 11 and taken["Bot"] == 900  # below the floor: kept whole
    assert report.floor == 1_000 and not report.floor_shrunk
    # The rest of the budget follows class size.
    assert taken["BENIGN"] > taken["DoS Hulk"] > 1_000
    assert report.before["BENIGN"] == 20_000 and report.after == {k: taken[k] for k in report.after}


def test_floors_shrink_when_they_exceed_the_budget() -> None:
    labels = _labels({"A": 5_000, "B": 5_000, "C": 5_000, "D": 300})
    quotas, floor, shrunk = sampling.allocate_quotas(labels.value_counts().to_dict(), budget=2_000)
    assert shrunk and floor == 500
    assert sum(quotas.values()) == 2_000
    assert quotas["D"] == 300  # still whole: smaller than the lowered floor
    assert sorted(quotas[k] for k in "ABC") == [566, 567, 567]  # 1,700 left, shared equally, remainder by name


def test_sampler_keeps_everything_when_data_fits() -> None:
    labels = _labels({"BENIGN": 50, "Bot": 5})
    positions, report = sampling.sample_positions(labels, budget=1_000, seed=1)
    assert positions.tolist() == list(range(55)) and not report.sampled


def test_sampler_is_deterministic_for_a_seed() -> None:
    labels = _labels({"BENIGN": 5_000, "PortScan": 3_000, "Bot": 50})
    first, _ = sampling.sample_positions(labels, budget=2_500, seed=42)
    again, _ = sampling.sample_positions(labels, budget=2_500, seed=42)
    other, _ = sampling.sample_positions(labels, budget=2_500, seed=43)
    assert np.array_equal(first, again)
    assert not np.array_equal(first, other)


def test_rare_aware_sample_on_a_frame() -> None:
    frame = pd.DataFrame({"x": np.arange(3_000), "Label": _labels({"BENIGN": 2_500, "Bot": 500})})
    sample, report = sampling.rare_aware_sample(frame, "Label", budget=1_200, seed=3)
    assert len(sample) == 1_200 and report.after["Bot"] == 500
    assert sample.index.is_monotonic_increasing


def test_split_guarantees_two_rows_each_side() -> None:
    y = _labels({"BENIGN": 1_000, "Bot": 4, "Heartbleed": 5, "PortScan": 300})
    split = sampling.stratified_split(y, test_share=0.25, seed=0)
    assert len(np.intersect1d(split.train, split.test)) == 0
    assert np.array_equal(np.sort(np.concatenate([split.train, split.test])), np.arange(len(y)))
    for name in ("BENIGN", "Bot", "Heartbleed", "PortScan"):
        in_test = int((y.iloc[split.test] == name).sum())
        in_train = int((y.iloc[split.train] == name).sum())
        assert in_test >= 2 and in_train >= 2, name
    assert int((y.iloc[split.test] == "BENIGN").sum()) == 250


def test_split_is_deterministic_and_accepts_a_row_count() -> None:
    y = _labels({"BENIGN": 200, "Bot": 40})
    a = sampling.stratified_split(y, 0.3, seed=5)
    b = sampling.stratified_split(y.to_numpy(), 0.3, seed=5)
    assert np.array_equal(a.test, b.test) and np.array_equal(a.train, b.train)
    plain = sampling.stratified_split(100, 0.2, seed=1)
    assert len(plain.test) == 20 and len(plain.train) == 80


def test_split_refuses_classes_that_are_too_small() -> None:
    y = _labels({"BENIGN": 100, "Heartbleed": 3})
    with pytest.raises(ValueError, match="Heartbleed has 3"):
        sampling.stratified_split(y, 0.25, seed=0)
    with pytest.raises(ValueError):
        sampling.stratified_split(y, 1.5, seed=0)
