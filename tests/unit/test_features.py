"""Feature-set selection (curated / all / top-K, port opt-in, degenerate columns) and the gain-based ranking.

The ranking tests use the 20-round ranking model of ``quick_ranking`` (tests/conftest.py); the app's 120 rounds
are exercised by the real-data tests.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pandas as pd
import pytest

from graticule import schema
from graticule.data.synthetic import generate
from graticule.features import FeatureChoice, rank_features, select_features

pytestmark = [pytest.mark.unit, pytest.mark.usefixtures("quick_ranking")]

PORT = schema.DESTINATION_PORT
CURATED_IN_ORDER = tuple(c for c in schema.FEATURES if c in schema.CURATED)
ALL_BUT_PORT = tuple(c for c in schema.FEATURES if c != PORT)


# ------------------------------------------------------------------ select_features


def test_curated_mode() -> None:
    choice = select_features("curated")
    assert choice.mode == "curated"
    assert choice.columns == CURATED_IN_ORDER and choice.n_features == 28
    assert not choice.include_port and choice.k is None and choice.ranking is None
    assert choice.dropped_degenerate == ()


def test_all_mode_excludes_port() -> None:
    choice = select_features("all")
    assert choice.columns == ALL_BUT_PORT and len(choice.columns) == 76
    assert PORT not in choice.columns and not choice.include_port


@pytest.mark.parametrize("mode", ["curated", "all", "topk"])
def test_port_opt_in_in_every_mode(mode: str) -> None:
    ranking = [PORT, *reversed(ALL_BUT_PORT)]
    without = select_features(mode, ranking=ranking, k=5)  # type: ignore[arg-type]
    with_port = select_features(mode, ranking=ranking, k=5, include_port=True)  # type: ignore[arg-type]
    assert PORT not in without.columns and not without.include_port
    assert with_port.columns == (*without.columns, PORT) and with_port.include_port


def test_port_is_skipped_when_not_available() -> None:
    available = [c for c in schema.FEATURES if c != PORT]
    choice = select_features("all", include_port=True, available=available)
    assert PORT not in choice.columns and not choice.include_port


def test_degenerate_columns_are_removed_and_reported_in_all_mode() -> None:
    degenerate = {"Bwd PSH Flags", "Fwd Avg Bytes/Bulk", "Idle Std", "Not A Column"}
    choice = select_features("all", degenerate=degenerate)
    assert not degenerate & set(choice.columns)
    assert choice.dropped_degenerate == ("Bwd PSH Flags", "Fwd Avg Bytes/Bulk", "Idle Std")  # schema order
    assert len(choice.columns) == 73


def test_curated_mode_reports_only_curated_degenerate_columns() -> None:
    choice = select_features("curated", degenerate=["URG Flag Count", "Bwd PSH Flags"])
    assert "URG Flag Count" not in choice.columns and len(choice.columns) == 27
    assert choice.dropped_degenerate == ("URG Flag Count",)


def test_degenerate_port_is_not_added_even_on_request() -> None:
    choice = select_features("curated", degenerate=[PORT], include_port=True)
    assert PORT not in choice.columns and not choice.include_port
    assert choice.dropped_degenerate == (PORT,)


def test_available_restricts_every_mode() -> None:
    available = list(CURATED_IN_ORDER[:5]) + ["Idle Max", PORT]
    assert select_features("curated", available=available).columns == CURATED_IN_ORDER[:5]
    assert select_features("all", available=list(reversed(available))).columns == (*CURATED_IN_ORDER[:5], "Idle Max")


def test_topk_keeps_ranking_order_and_ignores_port_for_k() -> None:
    ranking = ["Flow Duration", PORT, "SYN Flag Count", "Init_Win_bytes_forward", "Idle Mean", "ACK Flag Count"]
    choice = select_features("topk", ranking=ranking, k=3)
    assert choice.columns == ("Flow Duration", "SYN Flag Count", "Init_Win_bytes_forward")
    assert choice.k == 3 and not choice.include_port
    with_port = select_features("topk", ranking=ranking, k=3, include_port=True)
    assert with_port.columns == ("Flow Duration", "SYN Flag Count", "Init_Win_bytes_forward", PORT)
    assert len(with_port.columns) == 4


def test_topk_skips_degenerate_and_unavailable_names() -> None:
    ranking = [
        ("Flow Duration", 9.0),
        ("Bwd PSH Flags", 8.0),
        ("Mystery", 7.0),
        ("Idle Mean", 6.0),
        ("FIN Flag Count", 5.0),
    ]
    choice = select_features("topk", ranking=ranking, k=2, degenerate=["Bwd PSH Flags", "FIN Flag Count"])
    assert choice.columns == ("Flow Duration", "Idle Mean")
    assert choice.dropped_degenerate == ("Bwd PSH Flags",)  # FIN was never reached
    assert choice.ranking == tuple(ranking)


def test_topk_with_short_ranking_takes_what_there_is() -> None:
    choice = select_features("topk", ranking=["Flow Duration", "Idle Mean"], k=20)
    assert choice.columns == ("Flow Duration", "Idle Mean") and choice.k == 20
    assert choice.ranking is not None and all(math.isnan(score) for _, score in choice.ranking)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"mode": "topk"}, "ranking"),
        ({"mode": "topk", "ranking": ["Flow Duration"], "k": 0}, "positive"),
        ({"mode": "everything"}, "Unknown feature mode"),
        ({"mode": "topk", "ranking": [PORT]}, "No usable"),
        ({"mode": "curated", "available": ["Destination Port"]}, "No usable"),
    ],
)
def test_selection_errors(kwargs: dict[str, object], message: str) -> None:
    mode = kwargs.pop("mode")
    with pytest.raises(ValueError, match=message):
        select_features(mode, **kwargs)  # type: ignore[arg-type]


def test_feature_choice_is_frozen() -> None:
    choice = select_features("curated")
    assert isinstance(choice, FeatureChoice)
    with pytest.raises(dataclasses.FrozenInstanceError):
        choice.columns = ()  # type: ignore[misc]


# ------------------------------------------------------------------ rank_features


@pytest.fixture(scope="module")
def training_rows() -> pd.DataFrame:
    return generate(3_000, seed=4)


def test_rank_features_is_deterministic_and_complete(training_rows: pd.DataFrame) -> None:
    names = list(ALL_BUT_PORT)
    X = training_rows[names].to_numpy()
    y = training_rows[schema.LABEL]
    first = rank_features(X, y, names, seed=7, max_rows=1_500)
    second = rank_features(X, y, names, seed=7, max_rows=1_500)
    assert first == second
    assert sorted(name for name, _ in first) == sorted(names)
    gains = [gain for _, gain in first]
    assert gains == sorted(gains, reverse=True)
    assert all(isinstance(g, float) and g >= 0 for g in gains) and gains[0] > 0
    constant = {c for c in names if training_rows[c].nunique(dropna=False) == 1}
    assert constant, "the synthetic data has constant bulk columns"
    assert all(gain == 0.0 for name, gain in first if name in constant)


def test_rank_features_accepts_dataframes_and_binary_labels(training_rows: pd.DataFrame) -> None:
    names = ["Flow Duration", "SYN Flag Count", "Init_Win_bytes_forward", "Fwd Avg Bulk Rate"]
    y = (training_rows[schema.LABEL] != schema.BENIGN).to_numpy()
    from_frame = rank_features(training_rows, y, names, seed=1)
    from_array = rank_features(training_rows[names].to_numpy(), y, names, seed=1)
    assert from_frame == from_array
    assert dict(from_frame)["Fwd Avg Bulk Rate"] == 0.0


def test_rank_features_puts_the_informative_column_first() -> None:
    rng = np.random.default_rng(0)
    n = 2_000
    y = rng.integers(0, 3, n)
    X = rng.normal(size=(n, 4)).astype(np.float32)
    X[:, 2] = y * 10 + rng.normal(scale=0.1, size=n)
    X[:5, 0] = np.inf  # infinities are treated as missing values, not errors
    ranking = rank_features(X, y, ["noise a", "noise b", "signal", "noise c"], seed=3)
    assert ranking[0][0] == "signal"


def test_rank_features_maps_gains_by_position_even_for_booster_like_names() -> None:
    """Names that look like the booster's own ("f0", "f1", ...) must not redirect gains to another column."""
    rng = np.random.default_rng(5)
    n = 2_000
    y = rng.integers(0, 2, n)
    X = rng.normal(size=(n, 5)).astype(np.float32)
    X[:, 0] = y * 10 + rng.normal(scale=0.1, size=n)
    plain = rank_features(X, y, ["a", "b", "c", "d", "e"], seed=2)
    tricky = rank_features(pd.DataFrame(X, columns=["f4", "f3", "f2", "f1", "f0"]), y,
                           ["f4", "f3", "f2", "f1", "f0"], seed=2)
    rename = dict(zip("abcde", ["f4", "f3", "f2", "f1", "f0"]))
    assert tricky == [(rename[name], gain) for name, gain in plain]
    assert tricky[0][0] == "f4"


def test_rank_features_subsamples_but_keeps_rare_classes() -> None:
    rng = np.random.default_rng(1)
    y = np.array(["common"] * 3_000 + ["rare"] * 4)
    X = rng.normal(size=(y.size, 3)).astype(np.float32)
    X[y == "rare", 1] += 50
    ranking = rank_features(X, y, ["a", "b", "c"], seed=2, max_rows=300)
    assert ranking[0][0] == "b"


@pytest.mark.parametrize(
    ("X", "y", "names", "message"),
    [
        (np.zeros((10, 2)), np.array([0, 1] * 5), ["a"], "one column per feature"),
        (np.zeros((10, 2)), np.array([0, 1] * 4), ["a", "b"], "one label per row"),
        (np.zeros((10, 2)), np.zeros(10), ["a", "b"], "two classes"),
        (np.zeros((10, 2)), np.array([0, 1] * 5), ["a", "a"], "unique"),
    ],
)
def test_rank_features_errors(X: np.ndarray, y: np.ndarray, names: list[str], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        rank_features(X, y, names, seed=0)


def test_rank_features_reports_missing_frame_columns() -> None:
    frame = pd.DataFrame({"a": [0.0, 1.0, 2.0, 3.0]})
    with pytest.raises(ValueError, match="lacks columns"):
        rank_features(frame, [0, 1, 0, 1], ["a", "b"], seed=0)
