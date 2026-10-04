"""Single-flow explanations and scoring (graticule.explain) and the contribution chart (graticule.viz).

* XGBoost exact contributions plus the bias equal the raw margin, binary and multi-class, also for a flow with
  missing and infinite values; the binary "Normal" side is the mirror image.
* The reference swap gives exactly zero when every background value equals the flow's own value, points the right
  way on a hand-made model whose answer is known, and scores everything in ONE batched call.
* Quantile backgrounds have the right shape, stay within each feature's training range and repeat for a seed.
* ``score_flow`` on held-out rows reproduces the verdicts the fit stored for them, for every channel.
* The contribution chart builds, validates against the Vega-Lite schema and uses the two ends of the diverging ramp.

Models are tiny (``profile="test"``) and data are random numbers or the synthetic generator; no dataset rows.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from graticule import explain, theme, viz
from graticule.data.prepare import DataRequest
from graticule.models import train, zoo
from graticule.models.train import TrainRequest, feature_quantiles
from graticule.models.zoo import BuildContext
from tests.helpers import shared_fit, shared_sample

pytestmark = pytest.mark.unit
NAMES = [f"f{i}" for i in range(6)]


def _toy(n_classes: int, seed: int = 3, n: int = 600) -> tuple[np.ndarray, np.ndarray]:
    """Random features with a class signal in the first two columns."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, len(NAMES))).astype(np.float32)
    score = X[:, 0] + 0.5 * X[:, 1]
    edges = np.quantile(score, np.linspace(0, 1, n_classes + 1)[1:-1])
    return X, np.digitize(score, edges).astype(np.int64)


#: XGBoost channels fitted by :func:`_fitted_xgboost`, one per class count (the tests only read them).
_XGBOOST_FITS: dict[int, Any] = {}


def _fitted_xgboost(n_classes: int) -> Any:
    """A test-profile XGBoost channel fitted through the trainer (early stopping sets its best round); fitted once
    per class count for the module and shared, so it must never be changed."""
    if n_classes not in _XGBOOST_FITS:
        X, y = _toy(n_classes)
        ctx = BuildContext(n_classes=n_classes, seed=0, profile="test")
        estimator = zoo.build_estimator("xgboost", ctx)
        fitted, info = train.fit_model("xgboost", estimator, X, y, zoo.channel_weights("xgboost", y), ctx=ctx)
        assert "best_iteration" in info["extra"]
        _XGBOOST_FITS[n_classes] = fitted
    return _XGBOOST_FITS[n_classes]


def _margins(pipeline: Any, rows: np.ndarray) -> np.ndarray:
    """The channel's own raw scores (log-odds), as its predict_proba would turn them into probabilities."""
    ready = pipeline[:-1].transform(rows)
    return np.asarray(pipeline[-1].predict(ready, output_margin=True), dtype=np.float64)


# --------------------------------------------------------------------------------------------------------------
# XGBoost exact contributions
# --------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("n_classes", [2, 3])
def test_xgboost_contributions_add_up_to_the_margin(n_classes: int) -> None:
    pipeline = _fitted_xgboost(n_classes)
    X, _ = _toy(n_classes, seed=9, n=8)
    X[0, 2] = np.nan
    X[1, 3] = np.inf  # the sanitiser turns it into a missing value before the booster sees it
    margins = _margins(pipeline, X)
    for i in range(len(X)):
        for k in range(n_classes):
            if n_classes == 2:
                expected = margins[i] if k == 1 else -margins[i]
            else:
                expected = margins[i, k]
            e = explain.xgboost_contributions(pipeline, X[i], NAMES, k, class_name=f"class {k}")
            assert e.method == "xgboost_exact" and e.units == explain.UNITS_LOG_ODDS and not e.approximate
            assert e.class_name == f"class {k}" and e.bias is not None
            assert e.output == pytest.approx(expected, abs=1e-4)
            assert e.total() == pytest.approx(expected, abs=1e-4)
            assert list(e.table.columns) == ["feature", "value", "contribution"]
            assert sorted(e.table["feature"]) == sorted(NAMES)
            sizes = e.table["contribution"].abs().to_numpy()
            assert np.all(sizes[:-1] >= sizes[1:])
    # The flow's own values are reported as given (the infinite value too).
    e = explain.xgboost_contributions(pipeline, X[1], NAMES, 1)
    assert np.isinf(e.table.set_index("feature").loc["f3", "value"])


def test_binary_normal_side_mirrors_the_attack_side() -> None:
    pipeline = _fitted_xgboost(2)
    row = _toy(2, seed=4, n=1)[0][0]
    attack = explain.xgboost_contributions(pipeline, row, NAMES, 1).table.set_index("feature")
    normal = explain.xgboost_contributions(pipeline, row, NAMES, 0).table.set_index("feature")
    assert np.allclose(attack.loc[NAMES, "contribution"], -normal.loc[NAMES, "contribution"])


def test_exact_contributions_refuse_other_channels_and_classes() -> None:
    X, y = _toy(2)
    ctx = BuildContext(n_classes=2, seed=0, profile="test")
    logreg, _ = train.fit_model("logreg", zoo.build_estimator("logreg", ctx), X, y, np.ones(len(y)), ctx=ctx)
    with pytest.raises(TypeError):
        explain.xgboost_contributions(logreg, X[0], NAMES, 1)
    with pytest.raises(ValueError):
        explain.xgboost_contributions(_fitted_xgboost(2), X[0], NAMES, 2)
    with pytest.raises(ValueError):
        explain.xgboost_contributions(_fitted_xgboost(2), X[0, :3], NAMES, 1)


# --------------------------------------------------------------------------------------------------------------
# Reference swap
# --------------------------------------------------------------------------------------------------------------
class _Logistic:
    """A hand-made binary model: P(class 1) = sigmoid(2 x0 - 3 x1); x2 plays no part. Counts its calls."""

    classes_ = np.array([0, 1])

    def __init__(self) -> None:
        self.calls: list[int] = []

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        self.calls.append(len(X))
        p = 1.0 / (1.0 + np.exp(-(2.0 * X[:, 0] - 3.0 * X[:, 1])))
        return np.column_stack([1.0 - p, p])


def _background(n: int = 50) -> np.ndarray:
    return np.random.default_rng(1).normal(size=(n, 3)).astype(np.float32)


def test_reference_swap_is_zero_when_the_background_equals_the_flow() -> None:
    model = _Logistic()
    row = np.array([0.7, -0.2, 3.0], dtype=np.float32)
    row_nan = np.array([0.7, np.nan, 3.0], dtype=np.float32)
    for flow in (row, row_nan):
        e = explain.reference_swap(model, flow, np.tile(flow, (10, 1)), ["a", "b", "c"], 1)
        assert (e.table["contribution"] == 0.0).all()
        assert e.method == "reference_swap" and e.approximate and e.units == explain.UNITS_PROBABILITY
    assert model.calls == [1, 1]  # only the flow itself had to be scored


def test_reference_swap_points_the_right_way() -> None:
    names = ["x0", "x1", "x2"]
    towards = np.array([2.0, -1.0, 0.0], dtype=np.float32)  # both x0 (high) and x1 (low) favour class 1
    for class_index, sign in ((1, 1.0), (0, -1.0)):
        e = explain.reference_swap(_Logistic(), towards, _background(), names, class_index, seed=3)
        c = e.table.set_index("feature")["contribution"]
        assert sign * c["x0"] > 0.05 and sign * c["x1"] > 0.05
        assert c["x2"] == 0.0  # a feature the model ignores never moves it
        assert e.table["feature"].iloc[-1] == "x2"
    away = np.array([-2.0, 1.0, 0.0], dtype=np.float32)
    c = explain.reference_swap(_Logistic(), away, _background(), names, 1).table.set_index("feature")["contribution"]
    assert c["x0"] < -0.05 and c["x1"] < -0.05
    # The output is the model's own probability of the class for the flow.
    e = explain.reference_swap(_Logistic(), towards, _background(), names, 1)
    assert e.output == pytest.approx(1.0 / (1.0 + np.exp(-7.0)))


def test_reference_swap_scores_everything_in_one_batched_call() -> None:
    model = _Logistic()
    row = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    background = _background(80)
    background[:, 2] = 0.5  # the third feature never differs: its copies are not scored
    e = explain.reference_swap(model, row, background, ["a", "b", "c"], 1, max_background=32, seed=7)
    assert len(model.calls) == 1
    assert model.calls[0] == 1 + 2 * 32
    assert e.background_rows == 32
    again = explain.reference_swap(_Logistic(), row, background, ["a", "b", "c"], 1, max_background=32, seed=7)
    pd.testing.assert_frame_equal(e.table, again.table)  # seeded draw of the background
    with pytest.raises(ValueError):
        explain.reference_swap(model, row, np.empty((0, 3), dtype=np.float32), ["a", "b", "c"], 1)
    with pytest.raises(ValueError):
        explain.reference_swap(model, row, background[:, :2], ["a", "b", "c"], 1)


# --------------------------------------------------------------------------------------------------------------
# Backgrounds, sentences, digests
# --------------------------------------------------------------------------------------------------------------
def test_quantile_background_shape_and_range() -> None:
    rng = np.random.default_rng(5)
    X = np.column_stack([rng.normal(size=400), rng.exponential(5.0, size=400), np.full(400, 7.0),
                         np.full(400, np.nan)]).astype(np.float32)
    quantiles = feature_quantiles(X)
    vectors = explain.background_from_quantiles(quantiles, 64, seed=2)
    assert vectors.shape == (64, 4) and vectors.dtype == np.float32
    finite = vectors[:, :3]
    assert np.all(finite >= np.nanmin(X[:, :3], axis=0) - 1e-6) and np.all(finite <= np.nanmax(X[:, :3], axis=0) + 1e-6)
    assert np.all(vectors[:, 2] == 7.0) and np.isnan(vectors[:, 3]).all()
    assert np.array_equal(vectors, explain.background_from_quantiles(quantiles, 64, seed=2), equal_nan=True)
    assert not np.array_equal(vectors, explain.background_from_quantiles(quantiles, 64, seed=3), equal_nan=True)
    assert len(np.unique(vectors[:, 0])) > 50  # a spread of values, not one row repeated


def test_reading_sentence_names_the_strongest_pushes() -> None:
    table = pd.DataFrame({"feature": ["Flow IAT Max", "Init_Win_bytes_backward", "SYN Flag Count", "Idle Mean"],
                          "value": [1.0, 2.0, 3.0, 4.0], "contribution": [0.9, 0.4, -0.6, 0.0]})
    e = explain.Explanation(method="xgboost_exact", channel="xgboost", class_name="DoS Hulk",
                            units=explain.UNITS_LOG_ODDS, table=table, bias=0.1)
    assert explain.reading_sentence(e, towards="DoS Hulk") == (
        "Flow IAT Max and Init_Win_bytes_backward pushed CH2 towards DoS Hulk the most.")
    assert explain.reading_sentence(e, towards="Normal", sign=-1) == (
        "SYN Flag Count pushed CH2 towards Normal the most.")
    # Only features pushing that way are named, however many are asked for.
    assert explain.reading_sentence(e, towards="DoS Hulk", n=3) == (
        "Flow IAT Max and Init_Win_bytes_backward pushed CH2 towards DoS Hulk the most.")
    none_up = e.table.assign(contribution=[-0.9, -0.4, -0.6, 0.0])
    e2 = explain.Explanation("reference_swap", "logreg", "Bot", explain.UNITS_PROBABILITY, none_up)
    assert explain.reading_sentence(e2, towards="Bot").startswith("No feature pushed CH5 towards Bot;")
    flat = explain.Explanation("reference_swap", "mlp", "Bot", explain.UNITS_PROBABILITY,
                               table.assign(contribution=0.0))
    assert explain.reading_sentence(flat, towards="Bot").startswith("No single feature moved CH4")
    assert e.top(2)["feature"].tolist() == ["Flow IAT Max", "Init_Win_bytes_backward"]
    assert e.rest(2) == pytest.approx(-0.6) and e.total() == pytest.approx(0.8)


def test_row_digest_ignores_nan_payload_and_signed_zero() -> None:
    a = np.array([0.0, np.nan, 1.5], dtype=np.float32)
    b = np.array([-0.0, np.float32(np.nan) * -1, 1.5], dtype=np.float32)
    assert explain.row_digest(a) == explain.row_digest(b)
    assert explain.row_digest(a) != explain.row_digest(np.array([0.0, np.nan, 1.25], dtype=np.float32))


# --------------------------------------------------------------------------------------------------------------
# Scoring one flow with a fitted run
# --------------------------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def binary_run() -> train.TrainingRun:
    """All five channels (test profile) fitted on a small synthetic sample (made once per session, see shared_fit)."""
    prepared = shared_sample(DataRequest(source="synthetic", synthetic_flows=1_500, seed=7))
    return shared_fit(prepared, TrainRequest(mode="binary", profile="test", seed=7))


@pytest.mark.integration
def test_score_flow_reproduces_the_stored_verdicts(binary_run: train.TrainingRun) -> None:
    run = binary_run
    fits = sum(train.FIT_CALLS.values())
    keys = run.ok_channels()
    assert keys == list(zoo.MODEL_KEYS)
    rows = range(0, len(run.data.y_test), max(1, len(run.data.y_test) // 20))  # 20 flows spread over the rows
    checked = 0
    for i in rows:
        verdict = explain.score_flow(run, run.data.X_test[i], keys)
        assert verdict.channels == tuple(keys) and verdict.classes == ("Normal", "Attack")
        for key in keys:
            stored = run.channels[key].proba[i]
            assert np.allclose(verdict.proba[key], stored, atol=1e-5), (key, i)
            top_two = np.sort(stored)[-2:]
            if top_two[1] - top_two[0] > 1e-4:  # a near-tie could round either way: compare clear verdicts
                assert verdict.label_index(key) == int(run.channels[key].y_pred[i]), (key, i)
                checked += 1
        mean = np.mean([verdict.proba[k] for k in keys], axis=0)
        assert verdict.consensus_index == int(np.argmax(mean))
        assert verdict.agreement == sum(verdict.label_index(k) == verdict.consensus_index for k in keys)
        assert verdict.attack_probability() == pytest.approx(float(mean[1]), abs=1e-6)
    assert checked > 0.9 * len(keys) * len(rows)
    one = explain.score_flow(run, run.data.X_test[0], ["logreg", "not-a-channel"])
    assert one.channels == ("logreg",) and one.consensus.voters == 1
    with pytest.raises(ValueError):
        explain.score_flow(run, run.data.X_test[0], ["nothing"])
    with pytest.raises(ValueError):
        explain.score_flow(run, run.data.X_test[0, :3], keys)
    assert sum(train.FIT_CALLS.values()) == fits


@pytest.mark.integration
def test_typical_flows_and_backgrounds(binary_run: train.TrainingRun) -> None:
    run = binary_run
    attack, where = explain.typical_flow(run, 1)
    assert attack.shape == (len(run.data.feature_names),) and "Attack rows of the reference sample" in where
    labels = explain._reference_labels(run)
    assert labels is not None and len(labels) == len(run.reference_sample)
    expected = np.nanmedian(np.asarray(run.reference_sample, dtype=np.float64)[labels == 1], axis=0)
    assert np.allclose(attack, expected.astype(np.float32), equal_nan=True)
    background, text = explain.background_for(run)
    assert background is run.reference_sample or np.array_equal(background, run.reference_sample, equal_nan=True)
    assert "reference sample" in text
    # A run without its rows (as a set loaded from disk alone): the quantiles stand in for both.
    from dataclasses import replace

    empty = replace(run.data, X_train=run.data.X_train[:0], y_train=run.data.y_train[:0],
                    X_test=run.data.X_test[:0], y_test=run.data.y_test[:0])
    hollow = replace(run, data=empty)
    median, where = explain.typical_flow(hollow, 1)
    assert np.array_equal(median, run.feature_quantiles[explain.MEDIAN_ROW], equal_nan=True)
    assert "quantiles" in where
    vectors, text = explain.background_for(hollow)
    assert vectors.shape == (explain.QUANTILE_BACKGROUND_ROWS, len(run.data.feature_names))
    assert "quantiles" in text
    e = explain.reference_swap(run.channels["forest"].estimator, median, vectors, run.data.feature_names, 1, seed=4)
    assert len(e.table) == len(run.data.feature_names) and e.background_rows == explain.DEFAULT_BACKGROUND


# --------------------------------------------------------------------------------------------------------------
# The chart
# --------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("mode", ["light", "dark"])
def test_contribution_chart_builds_with_the_diverging_ends(mode: str) -> None:
    rng = np.random.default_rng(0)
    table = pd.DataFrame({"feature": [f"Feature {i}" for i in range(20)], "value": rng.normal(size=20) * 1e5,
                          "contribution": rng.normal(size=20)})
    table.loc[3, "value"] = np.nan
    chart = viz.contribution_chart(table, mode, top=12, toward_label="Towards Attack", away_label="Towards Normal")
    spec = chart.to_dict()  # validates against the Vega-Lite schema
    rows = next(iter(spec["datasets"].values()))
    assert len(rows) == 12
    sizes = [abs(r["c"]) for r in rows]
    assert sizes == sorted(sizes, reverse=True)
    assert all(r["shown"][0] in "+-" for r in rows)
    ramp = theme.DIVERGING[mode]
    text = str(spec)
    assert ramp[0] in text and ramp[-1] in text
    colour = spec["layer"][0]["layer"][0]["encoding"]["color"]["scale"]
    assert colour["domain"] == ["Towards Attack", "Towards Normal"] and colour["range"] == [ramp[-1], ramp[0]]
    normal = viz.contribution_chart(table, mode, toward_label="Towards BENIGN (normal)", away_label="Away",
                                    toward_is_normal=True).to_dict()
    assert normal["layer"][0]["layer"][0]["encoding"]["color"]["scale"]["range"] == [ramp[0], ramp[-1]]
    assert viz.value_text(float("nan")) == "missing" and viz.value_text(1234.0) == "1,234"
    assert viz.value_text(0.000123456) == "0.0001235" and viz.value_text(float("-inf")) == "-inf"
    assert viz.value_text(638_812.25) == "638,812.2" and viz.value_text(549.3125) == "549.3"
    assert viz.value_text(-1.0) == "-1" and viz.value_text(3e17) == "3.000e+17"
    flat = viz.contribution_chart(table.assign(contribution=0.0), mode).to_dict()
    assert flat["layer"][0]["layer"][0]["encoding"]["x"]["scale"]["domain"] == [-1.0, 1.0]
