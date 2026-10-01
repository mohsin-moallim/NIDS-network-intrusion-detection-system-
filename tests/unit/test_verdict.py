"""The combined verdict: equal-weight mean of channel probabilities, argmax and agreement count."""

from __future__ import annotations

import numpy as np
import pytest

from graticule.models.verdict import Consensus, combine

pytestmark = pytest.mark.unit


def test_combine_averages_with_equal_weight() -> None:
    probas = {
        "forest": np.array([[0.9, 0.1], [0.4, 0.6], [0.2, 0.8]]),
        "xgboost": np.array([[0.7, 0.3], [0.6, 0.4], [0.1, 0.9]]),
        "logreg": np.array([[0.2, 0.8], [0.3, 0.7], [0.45, 0.55]]),
    }
    result = combine(probas)
    assert isinstance(result, Consensus)
    assert result.voters == 3
    assert result.proba.dtype == np.float32
    np.testing.assert_allclose(result.proba, np.mean(list(probas.values()), axis=0), atol=1e-7)
    assert result.label_index.tolist() == [0, 1, 1]
    assert result.agreement.tolist() == [2, 2, 3]
    assert result.agreement_text(0, ("Normal", "Attack")) == "2 of 3 channels read Normal"


def test_combine_accepts_a_single_flow_vector_and_multiclass() -> None:
    result = combine({"a": np.array([0.1, 0.2, 0.7]), "b": np.array([0.6, 0.3, 0.1])})
    assert result.proba.shape == (1, 3)
    assert result.label_index.tolist() == [2]
    assert result.agreement.tolist() == [1]
    one = combine({"a": np.array([[0.3, 0.7]])})
    assert one.agreement_text(0, ["Normal", "Attack"]) == "1 of 1 channel read Attack"


def test_combine_rejects_empty_or_mismatched_inputs() -> None:
    with pytest.raises(ValueError):
        combine({})
    with pytest.raises(ValueError):
        combine({"a": np.zeros((2, 2)), "b": np.zeros((3, 2))})
    with pytest.raises(ValueError):
        combine({"a": np.zeros((2, 2, 2))})


def test_consensus_is_immutable() -> None:
    result = combine({"a": np.array([[0.5, 0.5]])})
    with pytest.raises(AttributeError):
        result.voters = 2  # type: ignore[misc]
