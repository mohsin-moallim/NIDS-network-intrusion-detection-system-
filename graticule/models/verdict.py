"""The combined verdict ("Consensus"): one reading made from several channels' class probabilities.

Each selected channel counts equally: the consensus probabilities are the plain mean of the channels' (n x K)
probability arrays, the consensus class is their argmax, and the agreement count says how many channels picked
that same class on their own (for example "4 of 5 channels read Attack").

The alert rule lives here too, so 04 Probe, 05 Assay and 06 Sweep raise the very same alerts: a flow raises a
high-confidence alert when its verdict is an attack class AND its attack probability (binary: P(Attack);
multi-class: 1 - P(normal class)) is at least the alert threshold (:func:`alert_flags`). Every station computes
the attack probability with :func:`attack_probability` and compares it with the threshold in 32-bit precision
(the precision every channel's probabilities are kept in), so a flow whose probability sits exactly on the
threshold gets the same answer everywhere, and a probability printed as 0.7 meets a threshold of 0.7.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

#: The alert rule in words (for help texts, captions and the PDF record).
ALERT_RULE = "an attack verdict whose attack probability is at least the alert threshold"


def attack_probability(proba: Any, normal_index: int | None) -> np.ndarray:
    """The attack probability of each flow (float32, one entry per flow) from its class probabilities.

    ``proba`` is an (n x K) array (a single flow may be a length-K vector) in class-code order. With two classes
    (a binary run, or a multi-class run left with one attack class): the probability of the attack class itself.
    With more: one minus the probability of the normal class ``normal_index``, kept within 0..1. Without a normal
    class (None) every flow is an attack with certainty (1.0). The arithmetic is done in float64 and the result
    kept as float32, the precision of the probabilities themselves, the same way at 04 Probe, 05 Assay and 06
    Sweep.
    """
    values = np.asarray(proba, dtype=np.float64)
    if values.ndim == 1:
        values = values[np.newaxis, :]
    if values.ndim != 2:
        raise ValueError("Class probabilities must be an (n x K) array.")
    if normal_index is None:
        return np.ones(values.shape[0], dtype=np.float32)
    normal = int(normal_index)
    if values.shape[1] == 2:
        return values[:, 1 - normal].astype(np.float32)
    return np.clip(1.0 - values[:, normal], 0.0, 1.0).astype(np.float32)


def alert_flags(attack_probability: Any, label_index: Any, normal_index: int | None, threshold: float
                ) -> np.ndarray:
    """Which flows raise a high-confidence alert (bool array, one entry per flow).

    A flow raises one when its verdict (``label_index``, class codes) is not the normal class ``normal_index`` AND
    its ``attack_probability`` is at least ``threshold``. With no normal class every verdict counts as an attack.
    NaN probabilities never alert. Probability and threshold are both compared as float32 (the precision the
    channels' probabilities are kept in), whatever precision they arrive in, so the same flow and threshold always
    give the same flag at every station.
    """
    probability = np.atleast_1d(np.asarray(attack_probability, dtype=np.float32))
    labels = np.atleast_1d(np.asarray(label_index)).astype(np.int64, copy=False)
    if probability.shape != labels.shape:
        raise ValueError("Give one attack probability and one verdict per flow.")
    attack_verdict = np.ones(labels.shape, dtype=bool) if normal_index is None else labels != int(normal_index)
    with np.errstate(invalid="ignore"):
        return attack_verdict & (probability >= np.float32(threshold))


@dataclass(frozen=True)
class Consensus:
    """The combined reading for n flows.

    Attributes:
        proba: (n x K) float32 mean of the channels' probabilities (rows sum to 1 when the inputs do).
        label_index: (n,) int64 consensus class code per flow (argmax of ``proba``).
        agreement: (n,) int64 number of channels whose own argmax equals the consensus class.
        voters: number of channels combined.
    """

    proba: np.ndarray
    label_index: np.ndarray
    agreement: np.ndarray
    voters: int

    def agreement_text(self, index: int, class_names: tuple[str, ...] | list[str]) -> str:
        """Plain words for flow ``index``, e.g. ``"4 of 5 channels read Attack"``."""
        name = class_names[int(self.label_index[index])]
        noun = "channel" if self.voters == 1 else "channels"
        return f"{int(self.agreement[index])} of {self.voters} {noun} read {name}"


def combine(probas: Mapping[str, np.ndarray]) -> Consensus:
    """Combine channels' class probabilities with equal weight.

    ``probas`` maps channel keys to (n x K) arrays that must all have the same shape (a single flow may be passed
    as a length-K vector). Raises ``ValueError`` when nothing is passed or the shapes differ.
    """
    if not probas:
        raise ValueError("Choose at least one channel to combine.")
    arrays = []
    for key, values in probas.items():
        array = np.asarray(values, dtype=np.float64)
        if array.ndim == 1:
            array = array[np.newaxis, :]
        if array.ndim != 2:
            raise ValueError(f"Probabilities of {key} must be an (n x K) array.")
        arrays.append(array)
    shape = arrays[0].shape
    if any(a.shape != shape for a in arrays):
        raise ValueError("Every channel must give probabilities for the same flows and classes.")
    stacked = np.stack(arrays)
    mean = stacked.mean(axis=0)
    label = mean.argmax(axis=1).astype(np.int64)
    own = stacked.argmax(axis=2)
    agreement = (own == label[np.newaxis, :]).sum(axis=0).astype(np.int64)
    return Consensus(proba=mean.astype(np.float32), label_index=label, agreement=agreement, voters=len(arrays))
