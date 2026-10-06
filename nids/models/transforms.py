"""Transformers shared by the channel pipelines.

* :class:`FlowSanitizer` is the first step of every channel. Flow records can hold infinite rates (a flow whose
  packets share one timestamp has zero duration); the sanitiser turns ±inf into NaN so that the next step, a median
  imputer or a tree learner that understands missing values, deals with them in one consistent way. It also makes
  the matrix float32, the precision every channel works in.
* :func:`signed_log1p` compresses the heavy tails of byte counts, rates and inter-arrival times before scaling.
  Several flow columns use -1 as a marker and a few rates are negative in the recorded files, so a plain
  ``log1p`` would fail; the signed version keeps the sign and is symmetric around zero.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, OneToOneFeatureMixin, TransformerMixin
from sklearn.preprocessing import FunctionTransformer
from sklearn.utils.validation import check_is_fitted, validate_data


class FlowSanitizer(OneToOneFeatureMixin, TransformerMixin, BaseEstimator):
    """Turn a feature matrix into float32 with every ±inf replaced by NaN.

    Fitting only records the number of columns (and their names when given a DataFrame), so the sanitiser never
    learns anything from the data and cannot leak information between a training and a test split. NaN values pass
    through untouched. The input is never modified; a copy is made only when an infinity has to be replaced.
    """

    def fit(self, X: Any, y: Any = None) -> "FlowSanitizer":
        """Record ``n_features_in_`` (and ``feature_names_in_`` for a DataFrame); returns ``self``."""
        validate_data(self, X, reset=True, ensure_all_finite=False, dtype=np.float32)
        return self

    def transform(self, X: Any) -> np.ndarray:
        """Return ``X`` as a float32 array with ±inf replaced by NaN."""
        check_is_fitted(self, "n_features_in_")
        values = validate_data(self, X, reset=False, ensure_all_finite=False, dtype=np.float32)
        values = np.asarray(values, dtype=np.float32)
        infinite = np.isinf(values)
        if infinite.any():
            values = values.copy()
            values[infinite] = np.nan
        return values

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        """Output column names: the same as the input names (``x0``, ``x1``, ... when none were recorded)."""
        return super().get_feature_names_out(input_features)

    def __sklearn_tags__(self) -> Any:
        """Declare that missing and infinite values are accepted (they are what this step exists to handle)."""
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True
        tags.requires_fit = True
        return tags


def signed_log1p(X: np.ndarray) -> np.ndarray:
    """``sign(x) * log1p(|x|)`` element-wise, as float32; NaN stays NaN.

    Zero maps to zero, the transform is odd (f(-x) = -f(x)) and strictly increasing, so the order of values is kept
    while a range of twelve orders of magnitude shrinks to about 28 units.
    """
    values = np.asarray(X, dtype=np.float32)
    with np.errstate(invalid="ignore"):
        return (np.sign(values) * np.log1p(np.abs(values))).astype(np.float32, copy=False)


def make_signed_log() -> FunctionTransformer:
    """A fresh, unfitted pipeline step applying :func:`signed_log1p` (column names pass through one-to-one)."""
    return FunctionTransformer(signed_log1p, feature_names_out="one-to-one", validate=False,
                               accept_sparse=False, check_inverse=False)


#: An unfitted template of the signed-log step. Pipelines must not share one fitted step, so build steps with
#: :func:`make_signed_log` (or ``sklearn.base.clone`` this template) instead of inserting this object directly.
SIGNED_LOG: FunctionTransformer = make_signed_log()
