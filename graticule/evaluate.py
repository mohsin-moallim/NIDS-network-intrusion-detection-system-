"""Readings of a fitted channel on the test rows.

For now this holds the headline numbers every fit reports right away (:func:`quick_metrics`); the full set of
readings (per-class tables, confusion matrices, curves, importance, cross-validation and timing) joins it later.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from sklearn.metrics import f1_score


def quick_metrics(y_true: npt.ArrayLike, y_pred: npt.ArrayLike, n_classes: int) -> dict[str, float]:
    """Accuracy, balanced accuracy and macro/weighted F1 for class codes 0..n_classes-1.

    Balanced accuracy is the mean recall over the classes present in ``y_true`` (the scikit-learn definition).
    F1 scores cover every class code; a class that is never predicted and never true scores 0 rather than raising.
    Empty input gives NaN readings. With ``zero_division=0`` given, scikit-learn raises no warning here, so nothing
    touches the process-wide warning filters (this runs on the fit thread as well as on the app's threads).
    """
    truth = np.asarray(y_true, dtype=np.int64).reshape(-1)
    guess = np.asarray(y_pred, dtype=np.int64).reshape(-1)
    if truth.shape != guess.shape:
        raise ValueError("y_true and y_pred must have the same length.")
    if truth.size == 0:
        nan = float("nan")
        return {"accuracy": nan, "balanced_accuracy": nan, "f1_macro": nan, "f1_weighted": nan}
    accuracy = float(np.mean(truth == guess))
    recalls = [float(np.mean(guess[truth == c] == c)) for c in np.unique(truth)]
    labels = list(range(int(n_classes)))
    f1_macro = float(f1_score(truth, guess, labels=labels, average="macro", zero_division=0))
    f1_weighted = float(f1_score(truth, guess, labels=labels, average="weighted", zero_division=0))
    return {"accuracy": accuracy, "balanced_accuracy": float(np.mean(recalls)), "f1_macro": f1_macro,
            "f1_weighted": f1_weighted}
