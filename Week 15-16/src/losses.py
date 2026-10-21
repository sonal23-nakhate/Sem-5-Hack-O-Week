"""Loss functions for comparing predictions with target values."""

import numpy as np


def mean_squared_error(
    predictions: np.ndarray, targets: np.ndarray
) -> float:
    """Return the average squared difference between predictions and targets."""
    predictions = np.asarray(predictions, dtype=float)
    targets = np.asarray(targets, dtype=float)
    return float(np.mean(np.square(predictions - targets)))


def binary_cross_entropy(
    probabilities: np.ndarray, targets: np.ndarray, epsilon: float = 1e-7
) -> float:
    """Binary cross-entropy, clipping probabilities before taking logarithms."""
    probabilities = np.asarray(probabilities, dtype=float)
    targets = np.asarray(targets, dtype=float)
    probabilities = np.clip(probabilities, epsilon, 1.0 - epsilon)
    terms = targets * np.log(probabilities) + (1.0 - targets) * np.log(
        1.0 - probabilities
    )
    return float(-np.mean(terms))
