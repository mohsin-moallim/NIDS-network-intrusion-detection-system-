"""FlowSanitizer and the signed logarithm: the first steps of every channel pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from graticule.models.transforms import SIGNED_LOG, FlowSanitizer, make_signed_log, signed_log1p

pytestmark = pytest.mark.unit


def test_sanitizer_turns_infinities_into_nan_and_returns_float32() -> None:
    raw = np.array([[1.0, np.inf], [-np.inf, 2.5], [np.nan, 3.0]], dtype=np.float64)
    out = FlowSanitizer().fit(raw).transform(raw)
    assert out.dtype == np.float32
    assert np.isnan(out[0, 1]) and np.isnan(out[1, 0]) and np.isnan(out[2, 0])
    assert out[1, 1] == np.float32(2.5) and out[2, 1] == np.float32(3.0)
    assert np.isinf(raw).sum() == 2, "the input must not be modified"


def test_sanitizer_leaves_a_read_only_input_alone() -> None:
    raw = np.array([[1.0, np.inf], [2.0, 3.0]], dtype=np.float32)
    raw.setflags(write=False)
    out = FlowSanitizer().fit_transform(raw)
    assert np.isnan(out[0, 1]) and np.isinf(raw[0, 1])


def test_sanitizer_records_feature_names_from_a_frame() -> None:
    frame = pd.DataFrame({"Flow Duration": [1.0, 2.0], "Flow Bytes/s": [np.inf, 4.0]}, dtype=np.float32)
    step = FlowSanitizer().fit(frame)
    assert step.n_features_in_ == 2
    assert list(step.feature_names_in_) == ["Flow Duration", "Flow Bytes/s"]
    assert list(step.get_feature_names_out()) == ["Flow Duration", "Flow Bytes/s"]
    assert np.isnan(step.transform(frame)[0, 1])


def test_sanitizer_names_out_without_recorded_names() -> None:
    step = FlowSanitizer().fit(np.zeros((3, 3)))
    assert list(step.get_feature_names_out()) == ["x0", "x1", "x2"]
    assert list(step.get_feature_names_out(["a", "b", "c"])) == ["a", "b", "c"]


def test_sanitizer_rejects_a_different_number_of_columns() -> None:
    step = FlowSanitizer().fit(np.zeros((3, 4)))
    with pytest.raises(ValueError):
        step.transform(np.zeros((3, 5)))


def test_sanitizer_must_be_fitted_first() -> None:
    from sklearn.exceptions import NotFittedError

    with pytest.raises(NotFittedError):
        FlowSanitizer().transform(np.zeros((2, 2)))


def test_signed_log_handles_negatives_zero_and_nan() -> None:
    values = np.array([[-1000.0, -1.0, 0.0, 1.0, 1000.0, np.nan]])
    out = signed_log1p(values)
    assert out.dtype == np.float32
    assert out[0, 2] == 0.0
    assert out[0, 3] == pytest.approx(np.log(2.0), rel=1e-6)
    assert out[0, 0] == pytest.approx(-out[0, 4]) and out[0, 1] == pytest.approx(-out[0, 3])
    assert np.isnan(out[0, 5])
    assert np.all(np.diff(out[0, :5]) > 0), "order must be kept"


def test_signed_log_step_passes_names_through_one_to_one() -> None:
    frame = pd.DataFrame({"a": [-1.0, 10.0], "b": [0.0, 100.0]})
    step = make_signed_log().fit(frame)
    assert list(step.get_feature_names_out()) == ["a", "b"]
    assert make_signed_log() is not make_signed_log(), "each pipeline needs its own step"
    assert clone(SIGNED_LOG).func is signed_log1p


def test_sanitizer_then_imputer_never_sees_an_infinity() -> None:
    X = np.array([[1.0, np.inf], [3.0, 5.0], [np.nan, 7.0]], dtype=np.float32)
    pipe = Pipeline([("sanitize", FlowSanitizer()), ("impute", SimpleImputer(strategy="median")),
                     ("log", make_signed_log())])
    out = pipe.fit_transform(X)
    assert np.isfinite(out).all()
    assert out.shape == (3, 2)
