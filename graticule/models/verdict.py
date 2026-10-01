"""The combined verdict ("Consensus"): one reading made from several channels' class probabilities.

Each selected channel counts equally: the consensus probabilities are the plain mean of the channels' (n x K)
probability arrays, the consensus class is their argmax, and the agreement count says how many channels picked
that same class on their own (for example "4 of 5 channels read Attack").
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np


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
