"""The combined verdict: equal-weight mean of channel probabilities, argmax and agreement count."""

from __future__ import annotations

import numpy as np
import pytest

from nids.models.verdict import Consensus, alert_flags, combine

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


def test_an_alert_needs_an_attack_verdict_and_the_threshold() -> None:
    """The one alert rule of 04 Probe, 05 Assay and 06 Sweep."""
    attack = np.array([0.6, 0.6, 0.95, 0.4, np.nan])
    verdict = np.array([0, 2, 1, 1, 1])  # class 0 is normal traffic
    assert alert_flags(attack, verdict, 0, 0.5).tolist() == [False, True, True, False, False]
    assert alert_flags(attack, verdict, None, 0.5).tolist() == [True, True, True, False, False]  # no normal class
    # The comparison keeps the precision of the probabilities given (float32 against a float32 threshold).
    close = np.array([np.float32(0.9)], dtype=np.float32)
    assert alert_flags(close, np.array([1]), 0, 0.9).tolist() == [True]
    assert alert_flags(0.95, 1, 0, 0.9).tolist() == [True]  # one flow
    with pytest.raises(ValueError):
        alert_flags(np.zeros(2), np.zeros(3), 0, 0.5)


def test_the_probe_reads_alerts_by_the_same_rule() -> None:
    """A flow read as BENIGN with 1 - P(BENIGN) above the threshold raises no alert at 04 Probe either."""
    from nids.explain import FlowVerdict

    proba = {"a": np.array([0.4, 0.3, 0.3], dtype=np.float32), "b": np.array([0.1, 0.8, 0.1], dtype=np.float32)}
    verdict = FlowVerdict(classes=("BENIGN", "DoS", "PortScan"), channels=("a", "b"), proba=proba,
                          consensus=combine(proba))
    assert verdict.attack_probability("a") == pytest.approx(0.6)
    assert not verdict.raises_alert("a", 0.5) and verdict.raises_alert("b", 0.5)
    assert verdict.raises_alert(None, 0.5)  # the consensus reads DoS (0.55) at 1 - 0.25 = 0.75
    assert not verdict.raises_alert("b", 0.95)
