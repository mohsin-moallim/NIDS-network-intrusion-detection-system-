"""The synthetic packet simulator and its flow meter: shape, determinism, internal consistency and realism."""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pytest

from graticule import schema
from graticule.data import synthetic
from graticule.data.synthetic import ACK, FIN, PSH, RST, SYN, PacketTable, generate, measure

pytestmark = pytest.mark.unit

BULK_COLUMNS = [c for c in schema.FEATURES if "Bulk" in c]
ACTIVE_IDLE = [f"{kind} {stat}" for kind in ("Active", "Idle") for stat in ("Mean", "Std", "Max", "Min")]


@pytest.fixture(scope="module")
def flows() -> pd.DataFrame:
    """A mid-sized sample with extra blur, so refused connections and disguised attacks are present."""
    return generate(12_000, seed=11, blur=0.1)


def _f64(frame: pd.DataFrame, column: str) -> np.ndarray:
    return frame[column].to_numpy(dtype=np.float64)


# ------------------------------------------------------------------ shape and determinism


def test_columns_dtypes_and_order(flows: pd.DataFrame) -> None:
    assert list(flows.columns) == [*schema.FEATURES, schema.LABEL]
    assert all(flows[c].dtype == np.float32 for c in schema.FEATURES)
    assert pd.api.types.is_string_dtype(flows[schema.LABEL])
    assert isinstance(flows.index, pd.RangeIndex) and len(flows) == 12_000


def test_same_arguments_give_identical_frames() -> None:
    first = generate(3_000, seed=5)
    second = generate(3_000, seed=5)
    pd.testing.assert_frame_equal(first, second)
    other = generate(3_000, seed=6)
    assert not first.equals(other)
    assert not first.equals(generate(3_000, seed=5, blur=0.3))


def test_labels_and_attack_share() -> None:
    frame = generate(20_000, seed=3, attack_share=0.35)
    labels = frame[schema.LABEL]
    assert set(labels.unique()) == set(synthetic.SYNTHETIC_CLASSES)
    share = float((labels != schema.BENIGN).mean())
    assert abs(share - 0.35) <= 0.03
    counts = labels[labels != schema.BENIGN].value_counts(normalize=True)
    for name, weight in synthetic.ATTACK_MIX.items():
        assert abs(counts[name] - weight) < 0.03, name


@pytest.mark.parametrize(("share", "expected_attacks"), [(0.0, 0), (1.0, 500), (0.2, 100)])
def test_attack_share_extremes(share: float, expected_attacks: int) -> None:
    frame = generate(500, seed=1, attack_share=share)
    assert int((frame[schema.LABEL] != schema.BENIGN).sum()) == expected_attacks


def test_zero_flows_gives_empty_frame_with_all_columns() -> None:
    frame = generate(0, seed=1)
    assert list(frame.columns) == [*schema.FEATURES, schema.LABEL]
    assert frame.empty


@pytest.mark.parametrize(
    "kwargs", [{"n_flows": -1}, {"n_flows": 10, "attack_share": 1.5}, {"n_flows": 10, "blur": -0.1}]
)
def test_rejects_bad_arguments(kwargs: dict[str, float]) -> None:
    n = kwargs.pop("n_flows")
    with pytest.raises(ValueError):
        generate(n, seed=1, **kwargs)  # type: ignore[arg-type]


def test_blocks_are_deterministic_and_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    """Generation in several blocks: same arguments, same frame; exact attack count; progress after every block."""
    monkeypatch.setattr(synthetic, "BLOCK_FLOWS", 1_000)
    seen: list[tuple[int, int]] = []
    first = generate(2_500, seed=8, progress=lambda done, total: seen.append((done, total)))
    assert seen == [(1_000, 2_500), (2_000, 2_500), (2_500, 2_500)]
    pd.testing.assert_frame_equal(first, generate(2_500, seed=8))
    assert list(first.columns) == [*schema.FEATURES, schema.LABEL] and len(first) == 2_500
    assert int((first[schema.LABEL] != schema.BENIGN).sum()) == round(2_500 * 0.35)
    # Blocks get their own random streams: no block repeats another one's flows.
    feature_rows = first[list(schema.FEATURES)]
    assert not feature_rows.iloc[:1_000].reset_index(drop=True).equals(
        feature_rows.iloc[1_000:2_000].reset_index(drop=True))


# ------------------------------------------------------------------ flow-meter invariants on every row


def test_subflows_equal_totals(flows: pd.DataFrame) -> None:
    for sub, total in [
        ("Subflow Fwd Packets", "Total Fwd Packets"),
        ("Subflow Bwd Packets", "Total Backward Packets"),
        ("Subflow Fwd Bytes", "Total Length of Fwd Packets"),
        ("Subflow Bwd Bytes", "Total Length of Bwd Packets"),
    ]:
        assert np.array_equal(flows[sub].to_numpy(), flows[total].to_numpy()), sub


def test_max_mean_min_ordering(flows: pd.DataFrame) -> None:
    for prefix in ("Fwd Packet Length", "Bwd Packet Length", "Flow IAT", "Fwd IAT", "Bwd IAT", "Active", "Idle"):
        high, mean, low = (_f64(flows, f"{prefix} {s}") for s in ("Max", "Mean", "Min"))
        assert np.all(high >= mean) and np.all(mean >= low), prefix
    assert np.all(_f64(flows, "Max Packet Length") >= _f64(flows, "Packet Length Mean"))
    assert np.all(_f64(flows, "Packet Length Mean") >= _f64(flows, "Min Packet Length"))


def test_variance_is_std_squared(flows: pd.DataFrame) -> None:
    std = _f64(flows, "Packet Length Std")
    np.testing.assert_allclose(_f64(flows, "Packet Length Variance"), std**2, rtol=1e-3, atol=1e-3)


def test_rates_follow_totals_and_duration(flows: pd.DataFrame) -> None:
    duration = _f64(flows, "Flow Duration")
    fwd, bwd = _f64(flows, "Total Fwd Packets"), _f64(flows, "Total Backward Packets")
    total_bytes = _f64(flows, "Total Length of Fwd Packets") + _f64(flows, "Total Length of Bwd Packets")
    timed = duration > 0
    seconds = duration[timed] / 1e6
    np.testing.assert_allclose(_f64(flows, "Flow Bytes/s")[timed], total_bytes[timed] / seconds, rtol=1e-5)
    np.testing.assert_allclose(_f64(flows, "Flow Packets/s")[timed], (fwd + bwd)[timed] / seconds, rtol=1e-5)
    np.testing.assert_allclose(_f64(flows, "Fwd Packets/s")[timed], fwd[timed] / seconds, rtol=1e-5)
    np.testing.assert_allclose(_f64(flows, "Bwd Packets/s")[timed], bwd[timed] / seconds, rtol=1e-5)
    rate_columns = ["Flow Bytes/s", "Flow Packets/s", "Fwd Packets/s", "Bwd Packets/s"]
    assert np.isfinite(flows.loc[timed, rate_columns].to_numpy()).all()


def test_zero_duration_rates_are_inf_or_nan_by_rule(flows: pd.DataFrame) -> None:
    duration = _f64(flows, "Flow Duration")
    instant = duration == 0
    total_bytes = _f64(flows, "Total Length of Fwd Packets") + _f64(flows, "Total Length of Bwd Packets")
    byte_rate = _f64(flows, "Flow Bytes/s")
    with_bytes = instant & (total_bytes > 0)
    without = instant & (total_bytes == 0)
    assert with_bytes.any() and without.any(), "both zero-duration cases should occur"
    assert np.all(np.isposinf(byte_rate[with_bytes]))
    assert np.all(np.isnan(byte_rate[without]))
    assert np.all(np.isnan(byte_rate) == without)
    assert np.all(np.isposinf(byte_rate) == with_bytes)
    assert np.all(np.isposinf(_f64(flows, "Flow Packets/s")) == instant)
    for rate, count in [("Fwd Packets/s", "Total Fwd Packets"), ("Bwd Packets/s", "Total Backward Packets")]:
        values, packets = _f64(flows, rate), _f64(flows, count)
        # As in the recorded files, per-direction rates stay finite: 0 for zero duration or an empty direction.
        assert np.isfinite(values).all(), rate
        assert np.all(values[instant] == 0), rate
        assert np.all(values[packets == 0] == 0), rate


def test_iat_totals_fit_inside_the_flow(flows: pd.DataFrame) -> None:
    duration = _f64(flows, "Flow Duration")
    assert np.all(duration >= 0)
    for column in ("Fwd IAT Total", "Bwd IAT Total", "Flow IAT Max"):
        assert np.all(_f64(flows, column) <= duration * (1 + 1e-6)), column
    packets = _f64(flows, "Total Fwd Packets") + _f64(flows, "Total Backward Packets")
    several = packets >= 2
    np.testing.assert_allclose(
        (_f64(flows, "Flow IAT Mean") * (packets - 1))[several], duration[several], rtol=1e-4, atol=1.0
    )
    assert np.all(_f64(flows, "Flow IAT Mean")[~several] == 0)
    assert packets.max() <= synthetic.MAX_PACKETS
    assert packets.min() >= 2 and _f64(flows, "Total Fwd Packets").min() >= 1


def test_every_flow_has_two_packets_and_zero_duration_stays_rare() -> None:
    """Like the recorded files: no single-packet flows, and zero-duration flows about one in a thousand, so the
    default drop strategy does not thin out any class."""
    frame = generate(40_000, seed=42)
    packets = _f64(frame, "Total Fwd Packets") + _f64(frame, "Total Backward Packets")
    assert packets.min() >= 2
    instant = _f64(frame, "Flow Duration") == 0
    assert 0 < instant.mean() <= 0.01
    bad = ~np.isfinite(frame[list(schema.FEATURES)].to_numpy(dtype=np.float64)).all(axis=1)
    by_class = frame.loc[bad, schema.LABEL].value_counts()
    totals = frame[schema.LABEL].value_counts()
    assert all(by_class.get(name, 0) <= 0.02 * totals[name] for name in totals.index), by_class.to_dict()


def test_unanswered_probes_and_queries_are_retried() -> None:
    pure = generate(6_000, seed=21, blur=0.0)
    sweep = pure[pure[schema.LABEL] == "Sweep"]
    silent = sweep["Total Backward Packets"] == 0
    assert silent.any()
    assert (sweep.loc[silent, "Total Fwd Packets"] == 2).all() and (sweep.loc[silent, "SYN Flag Count"] == 2).all()
    assert (sweep.loc[silent, "Flow Duration"] >= 20_000).all()
    dns = pure[pure["min_seg_size_forward"] == 8]  # only UDP (DNS) packets carry 8-byte headers
    unanswered = dns["Total Backward Packets"] == 0
    assert unanswered.any() and (dns.loc[unanswered, "Total Fwd Packets"] == 2).all()
    assert (dns.loc[unanswered, "Flow Duration"] >= 1e6).all()


def test_initial_windows_are_minus_one_exactly_for_empty_directions_and_udp(flows: pd.DataFrame) -> None:
    udp = _f64(flows, "min_seg_size_forward") == 8  # only UDP packets carry 8-byte headers
    assert udp.any() and (~udp).any()
    no_bwd = _f64(flows, "Total Backward Packets") == 0
    assert np.array_equal(_f64(flows, "Init_Win_bytes_forward") == -1, udp)
    assert np.array_equal(_f64(flows, "Init_Win_bytes_backward") == -1, udp | no_bwd)
    assert np.all(flows.loc[udp, ["SYN Flag Count", "ACK Flag Count", "PSH Flag Count"]].to_numpy() == 0)
    assert set(flows.loc[udp, "Destination Port"].unique()) == {53.0}


def test_segment_sizes_ratio_and_averages(flows: pd.DataFrame) -> None:
    assert np.array_equal(flows["Avg Fwd Segment Size"].to_numpy(), flows["Fwd Packet Length Mean"].to_numpy())
    assert np.array_equal(flows["Avg Bwd Segment Size"].to_numpy(), flows["Bwd Packet Length Mean"].to_numpy())
    assert np.array_equal(flows["Average Packet Size"].to_numpy(), flows["Packet Length Mean"].to_numpy())
    fwd, bwd = _f64(flows, "Total Fwd Packets"), _f64(flows, "Total Backward Packets")
    assert np.array_equal(_f64(flows, "Down/Up Ratio"), np.floor(bwd / fwd))
    np.testing.assert_allclose(
        _f64(flows, "Fwd Packet Length Mean") * fwd, _f64(flows, "Total Length of Fwd Packets"), rtol=1e-5, atol=1e-3
    )
    assert np.all(_f64(flows, "act_data_pkt_fwd") <= fwd)
    for flag in ("FIN", "SYN", "RST", "PSH", "ACK", "URG", "CWE", "ECE"):
        assert np.all(_f64(flows, f"{flag} Flag Count") <= fwd + bwd), flag
    header = _f64(flows, "Fwd Header Length")
    assert np.all(header >= _f64(flows, "min_seg_size_forward") * fwd - 1e-6)


def test_bulk_columns_are_zero(flows: pd.DataFrame) -> None:
    assert len(BULK_COLUMNS) == 6
    assert (flows[BULK_COLUMNS].to_numpy() == 0).all()


def test_active_idle_only_when_a_long_gap_exists(flows: pd.DataFrame) -> None:
    longest = _f64(flows, "Flow IAT Max")
    quiet = longest <= synthetic.IDLE_GAP_US
    assert (~quiet).any() and quiet.any()
    assert (flows.loc[quiet, ACTIVE_IDLE].to_numpy() == 0).all()
    np.testing.assert_allclose(_f64(flows, "Idle Max")[~quiet], longest[~quiet], rtol=1e-6)
    assert np.all(_f64(flows, "Idle Min")[~quiet] > synthetic.IDLE_GAP_US * (1 - 1e-6))


# ------------------------------------------------------------------ the meter on a hand-built packet table


def _toy_table() -> PacketTable:
    """Flow 0: a TCP conversation with one 6 s silence. Flow 1: a lone UDP packet. Flow 2: a lone empty SYN."""
    return PacketTable(
        flow=np.array([0, 0, 0, 0, 0, 1, 2]),
        time_us=np.array([0, 100, 200, 6_000_200, 6_000_300, 0, 0], dtype=np.float64),
        backward=np.array([False, True, False, False, True, False, False]),
        payload=np.array([0, 0, 100, 50, 300, 40, 0], dtype=np.float64),
        header=np.array([20, 20, 20, 32, 20, 8, 24], dtype=np.float64),
        flags=np.array([SYN, SYN | ACK, ACK | PSH, ACK | PSH, ACK, 0, SYN]),
        window=np.array([1000, 2000, 5, 5, 7, 0, 1024]),
        port=np.array([80, 53, 22]),
        udp=np.array([False, True, False]),
    )


def test_meter_on_hand_built_flows() -> None:
    out = measure(_toy_table())
    assert list(out.columns) == list(schema.FEATURES) and len(out) == 3
    row = out.iloc[0].astype(np.float64)
    expected = {
        "Destination Port": 80,
        "Flow Duration": 6_000_300,
        "Total Fwd Packets": 3,
        "Total Backward Packets": 2,
        "Total Length of Fwd Packets": 150,
        "Total Length of Bwd Packets": 300,
        "Fwd Packet Length Max": 100,
        "Fwd Packet Length Min": 0,
        "Fwd Packet Length Mean": 50,
        "Fwd Packet Length Std": np.sqrt(5000 / 3),
        "Bwd Packet Length Mean": 150,
        "Bwd Packet Length Std": 150,
        "Flow Bytes/s": 450 / 6.0003,
        "Flow Packets/s": 5 / 6.0003,
        "Flow IAT Mean": 6_000_300 / 4,
        "Flow IAT Max": 6_000_000,
        "Flow IAT Min": 100,
        "Fwd IAT Total": 6_000_200,
        "Fwd IAT Min": 200,
        "Bwd IAT Total": 6_000_200,
        "Fwd PSH Flags": 2,
        "Bwd PSH Flags": 0,
        "Fwd Header Length": 72,
        "Bwd Header Length": 40,
        "Min Packet Length": 0,
        "Max Packet Length": 300,
        "Packet Length Mean": 90,
        "SYN Flag Count": 2,
        "ACK Flag Count": 4,
        "PSH Flag Count": 2,
        "FIN Flag Count": 0,
        "Down/Up Ratio": 0,
        "Init_Win_bytes_forward": 1000,
        "Init_Win_bytes_backward": 2000,
        "act_data_pkt_fwd": 2,
        "min_seg_size_forward": 20,
        "Active Mean": 150,
        "Active Std": 50,
        "Active Max": 200,
        "Active Min": 100,
        "Idle Mean": 6_000_000,
        "Idle Std": 0,
        "Idle Max": 6_000_000,
        "Idle Min": 6_000_000,
    }
    for column, value in expected.items():
        assert row[column] == pytest.approx(value, rel=1e-6), column
    assert row["Packet Length Variance"] == pytest.approx(row["Packet Length Std"] ** 2, rel=1e-5)

    lone_udp = out.iloc[1]
    assert lone_udp["Flow Duration"] == 0
    assert np.isposinf(lone_udp["Flow Bytes/s"]) and np.isposinf(lone_udp["Flow Packets/s"])
    assert lone_udp["Fwd Packets/s"] == 0 and lone_udp["Bwd Packets/s"] == 0
    assert lone_udp["Init_Win_bytes_forward"] == -1 and lone_udp["Init_Win_bytes_backward"] == -1
    assert (out.iloc[1][ACTIVE_IDLE] == 0).all() and lone_udp["Flow IAT Max"] == 0

    lone_syn = out.iloc[2]
    assert np.isnan(lone_syn["Flow Bytes/s"]) and np.isposinf(lone_syn["Flow Packets/s"])
    assert lone_syn["Init_Win_bytes_forward"] == 1024 and lone_syn["Init_Win_bytes_backward"] == -1
    assert lone_syn["min_seg_size_forward"] == 24


def test_meter_rejects_malformed_tables() -> None:
    good = _toy_table()
    unsorted = PacketTable(**{**good.__dict__, "flow": np.array([0, 0, 1, 0, 0, 1, 2])})
    with pytest.raises(ValueError, match="sorted"):
        measure(unsorted)
    gap = PacketTable(**{**good.__dict__, "port": np.array([80, 53, 22, 9]), "udp": np.zeros(4, dtype=bool)})
    with pytest.raises(ValueError, match="at least one packet"):
        measure(gap)
    backwards = good.time_us.copy()
    backwards[3] = 50
    with pytest.raises(ValueError, match="decrease"):
        measure(PacketTable(**{**good.__dict__, "time_us": backwards}))


# ------------------------------------------------------------------ profile realism


def test_profiles_look_like_their_descriptions() -> None:
    pure = generate(6_000, seed=21, blur=0.0)
    by = pure.groupby(schema.LABEL, observed=True)
    duration = by["Flow Duration"].median()
    assert 60e6 * 0.8 <= duration["Slow Drip"] <= 120e6 * 1.1
    assert duration["Flood"] < 5_000 and duration["Sweep"] < 1_000
    packets = pure["Total Fwd Packets"] + pure["Total Backward Packets"]
    assert packets[pure[schema.LABEL] == "Sweep"].max() <= 3
    guess = pure[pure[schema.LABEL] == "Credential Guess"]
    assert set(guess["Destination Port"].unique()) <= {21.0, 22.0}
    assert (guess["Total Fwd Packets"] >= 10).all()
    drip = pure[pure[schema.LABEL] == "Slow Drip"]
    assert (drip["Idle Mean"] >= 10e6 * 0.9).all()
    injection = pure[pure[schema.LABEL] == "Web Injection"]
    assert (injection["Fwd Packet Length Max"] >= 500).all()
    flood = pure[pure[schema.LABEL] == "Flood"]
    assert (flood["Destination Port"] == 80).all() and (flood["Fwd Packet Length Max"] <= 12).all()
    assert pure.loc[pure[schema.LABEL] == schema.BENIGN, "Destination Port"].isin([53, 80, 443]).mean() > 0.7


def _binary_xy(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    columns = [c for c in schema.FEATURES if c != schema.DESTINATION_PORT]
    X = frame[columns].to_numpy(dtype=np.float32, copy=True)
    X[~np.isfinite(X)] = np.nan
    return X, (frame[schema.LABEL] != schema.BENIGN).to_numpy()


def _forest_balanced_accuracy(blur: float) -> float:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import balanced_accuracy_score

    X_train, y_train = _binary_xy(generate(6_000, seed=101, blur=blur))
    X_test, y_test = _binary_xy(generate(4_000, seed=202, blur=blur))
    forest = RandomForestClassifier(n_estimators=60, n_jobs=-1, random_state=0).fit(X_train, y_train)
    return float(balanced_accuracy_score(y_test, forest.predict(X_test)))


def test_normal_and_attack_separate_clearly() -> None:
    assert _forest_balanced_accuracy(blur=0.04) >= 0.93


def test_blur_keeps_separation_imperfect() -> None:
    assert _forest_balanced_accuracy(blur=0.2) < 0.999


# ------------------------------------------------------------------ speed


def test_ten_thousand_flows_are_quick() -> None:
    generate(500, seed=0)  # warm-up
    start = time.perf_counter()
    generate(10_000, seed=0)
    assert time.perf_counter() - start < 3.0


@pytest.mark.slow
def test_forty_thousand_flows_under_eight_seconds() -> None:
    start = time.perf_counter()
    frame = generate(40_000, seed=0)
    assert time.perf_counter() - start < 8.0
    assert len(frame) == 40_000
