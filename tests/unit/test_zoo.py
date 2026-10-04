"""The channel registry: every channel builds, hyperparameters per profile, and the capped balanced weights."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from graticule import theme
from graticule.models import zoo
from graticule.models.transforms import FlowSanitizer
from graticule.models.zoo import MODEL_KEYS, MODEL_SPECS, BuildContext, balanced_weights, build_estimator

pytestmark = pytest.mark.unit


def test_channel_keys_follow_the_theme_order() -> None:
    assert MODEL_KEYS == tuple(c.key for c in theme.CHANNELS)
    assert set(MODEL_SPECS) == set(MODEL_KEYS) == set(zoo.BUILDERS)


@pytest.mark.parametrize("profile", ["full", "test"])
@pytest.mark.parametrize("n_classes", [2, 6])
@pytest.mark.parametrize("key", MODEL_KEYS)
def test_every_channel_builds_a_sanitised_pipeline(key: str, n_classes: int, profile: str) -> None:
    pipe = build_estimator(key, BuildContext(n_classes=n_classes, seed=7, profile=profile))  # type: ignore[arg-type]
    assert isinstance(pipe, Pipeline)
    assert isinstance(pipe.steps[0][1], FlowSanitizer)
    assert pipe.steps[-1][0] == zoo.MODEL_STEP
    names = [name for name, _ in pipe.steps]
    if MODEL_SPECS[key].scaled:
        assert names == ["sanitize", "impute", "log", "scale", "model"]
        assert isinstance(pipe.named_steps["impute"], SimpleImputer)
        assert pipe.named_steps["impute"].strategy == "median"
        assert isinstance(pipe.named_steps["scale"], StandardScaler)
    else:
        assert names == ["sanitize", "model"], "tree channels route missing values natively"
    params = pipe.named_steps["model"].get_params()
    assert params.get("class_weight") is None, "weighting is done with sample weights only"
    if "random_state" in params:
        assert params["random_state"] == 7


def test_full_profile_hyperparameters() -> None:
    ctx = BuildContext(n_classes=2, seed=42)
    forest = build_estimator("forest", ctx).named_steps["model"]
    assert (forest.n_estimators, forest.max_features, forest.min_samples_leaf, forest.max_samples, forest.n_jobs) \
        == (150, "sqrt", 2, 0.5, -1)
    xgb = build_estimator("xgboost", ctx).named_steps["model"].get_params()
    assert (xgb["tree_method"], xgb["max_depth"], xgb["learning_rate"], xgb["n_estimators"]) == ("hist", 8, 0.15, 300)
    assert (xgb["early_stopping_rounds"], xgb["subsample"], xgb["colsample_bytree"], xgb["n_jobs"]) == (20, 0.8, 0.8, 4)
    assert xgb["objective"] == "binary:logistic"
    multi = build_estimator("xgboost", BuildContext(n_classes=5, seed=42)).named_steps["model"]
    assert multi.get_params()["objective"] == "multi:softprob"
    svc = build_estimator("svm", ctx).named_steps["model"]
    assert (svc.kernel, svc.C, svc.gamma, svc.cache_size) == ("rbf", 10.0, "scale", 1024)
    assert svc.probability in (False, "deprecated"), "probabilities come from post-hoc calibration"
    mlp = build_estimator("mlp", ctx).named_steps["model"]
    assert (mlp.hidden_layer_sizes, mlp.solver, mlp.alpha, mlp.batch_size, mlp.max_iter) == ((128, 64), "adam", 1e-4,
                                                                                           512, 60)
    assert mlp.early_stopping and mlp.n_iter_no_change == 5
    logreg = build_estimator("logreg", ctx).named_steps["model"]
    assert (logreg.solver, logreg.C, logreg.max_iter) == ("lbfgs", 1.0, 1000)


def test_test_profile_shrinks_every_model() -> None:
    ctx = BuildContext(n_classes=3, seed=1, profile="test", svm_cap=20_000)
    forest = build_estimator("forest", ctx).named_steps["model"]
    assert (forest.n_estimators, forest.n_jobs) == (20, 1)  # one thread: no thread pool for every small call
    assert build_estimator("xgboost", ctx).named_steps["model"].n_estimators == 10
    mlp = build_estimator("mlp", ctx).named_steps["model"]
    assert mlp.hidden_layer_sizes == (32,) and mlp.max_iter == 15
    assert build_estimator("logreg", ctx).named_steps["model"].max_iter == 200
    assert ctx.effective_svm_cap == 1_000
    assert BuildContext(n_classes=3, seed=1, profile="test", svm_cap=400).effective_svm_cap == 400
    assert BuildContext(n_classes=3, seed=1, svm_cap=20_000).effective_svm_cap == 20_000


def test_builders_return_fresh_objects() -> None:
    ctx = BuildContext(n_classes=2, seed=0)
    a, b = build_estimator("logreg", ctx), build_estimator("logreg", ctx)
    assert a is not b and a.named_steps["log"] is not b.named_steps["log"]


def test_bad_contexts_and_keys_are_rejected() -> None:
    with pytest.raises(ValueError):
        BuildContext(n_classes=1, seed=0)
    with pytest.raises(ValueError):
        BuildContext(n_classes=2, seed=0, profile="huge")  # type: ignore[arg-type]
    with pytest.raises(KeyError):
        build_estimator("knn", BuildContext(n_classes=2, seed=0))


def test_balanced_weights_without_capping_are_n_over_k_nc() -> None:
    y = np.array([0] * 60 + [1] * 30 + [2] * 10)
    w = balanced_weights(y, cap=100)
    assert w.dtype == np.float64 and w.shape == (100,)
    assert w.sum() == pytest.approx(100.0)
    assert w[0] == pytest.approx(100 / (3 * 60)) and w[60] == pytest.approx(100 / (3 * 30))
    assert w[-1] == pytest.approx(100 / (3 * 10))
    totals = [w[y == c].sum() for c in range(3)]
    assert totals == pytest.approx([100 / 3] * 3), "every class carries the same total weight"


def test_balanced_weights_respect_the_cap_after_rescaling() -> None:
    # One row in 100,000 would get weight ~33,000 uncapped.
    y = np.array([0] * 90_000 + [1] * 9_999 + [2] * 1)
    for cap in (100.0, 50.0):
        w = balanced_weights(y, cap)
        assert w.sum() == pytest.approx(len(y))
        assert w.max() <= cap * (1 + 1e-9)
        assert w[-1] == pytest.approx(cap)
        # The two uncapped classes still share the remaining weight equally.
        assert w[y == 0].sum() == pytest.approx(w[y == 1].sum())


def test_balanced_weights_cascade_when_several_classes_hit_the_cap() -> None:
    y = np.array([0] * 10_000 + [1] * 3 + [2] * 5 + [3] * 400)
    w = balanced_weights(y, 20)
    assert w.sum() == pytest.approx(len(y))
    assert w.max() <= 20 * (1 + 1e-9)
    assert set(np.round(w[(y == 1) | (y == 2)], 9)) == {20.0}
    assert sorted(zoo.capped_classes(y, 20)) == [1, 2]


def test_balanced_weights_edge_cases() -> None:
    assert balanced_weights(np.array([], dtype=int), 100).shape == (0,)
    assert np.allclose(balanced_weights(np.array([3, 3, 3]), 100), 1.0)
    assert np.allclose(balanced_weights(np.array([0] * 9 + [1]), 1.0), 1.0), "a cap of 1 means unit weights"
    with pytest.raises(ValueError):
        balanced_weights(np.array([0, 1]), 0.5)


def test_channel_weights_use_the_channel_cap_or_unit_weights() -> None:
    y = np.array([0] * 9_990 + [1] * 10)
    assert zoo.channel_weights("mlp", y).max() == pytest.approx(50.0)
    assert zoo.channel_weights("forest", y).max() == pytest.approx(100.0)
    unit = zoo.channel_weights("svm", y, balanced=False)
    assert unit.dtype == np.float64 and np.all(unit == 1.0) and unit.sum() == len(y)
