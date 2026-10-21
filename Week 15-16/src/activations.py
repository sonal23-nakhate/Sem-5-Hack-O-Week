"""Common activation functions implemented with NumPy."""

import numpy as np


def sigmoid(values: np.ndarray) -> np.ndarray:
    """Map values to (0, 1), clipping inputs to avoid exponential overflow."""
    values = np.asarray(values, dtype=float)
    values = np.clip(values, -500.0, 500.0)
    return 1.0 / (1.0 + np.exp(-values))


def sigmoid_derivative_from_output(output: np.ndarray) -> np.ndarray:
    """Derivative of sigmoid when its already-computed output is available."""
    return output * (1.0 - output)


def relu(values: np.ndarray) -> np.ndarray:
    """Keep positive values and replace negative values with zero."""
    return np.maximum(0.0, np.asarray(values, dtype=float))


def relu_derivative(values: np.ndarray) -> np.ndarray:
    """Derivative of ReLU; its value at zero is defined here as zero."""
    return (np.asarray(values) > 0.0).astype(float)


def tanh(values: np.ndarray) -> np.ndarray:
    """Map values to (-1, 1)."""
    return np.tanh(np.asarray(values, dtype=float))


def tanh_derivative_from_output(output: np.ndarray) -> np.ndarray:
    """Derivative of tanh when its already-computed output is available."""
    return 1.0 - np.square(output)


def softmax(values: np.ndarray, axis: int = -1) -> np.ndarray:
    """Convert scores to probabilities with a max-shift for numerical stability."""
    values = np.asarray(values, dtype=float)
    shifted = values - np.max(values, axis=axis, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / np.sum(exponentials, axis=axis, keepdims=True)


def softmax_jacobian(probabilities: np.ndarray) -> np.ndarray:
    """Return the Jacobian of one softmax probability vector with respect to scores."""
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.ndim != 1:
        raise ValueError("probabilities must be a one-dimensional vector")
    return np.diag(probabilities) - np.outer(probabilities, probabilities)


def step(values: np.ndarray) -> np.ndarray:
    """Binary perceptron activation: return one at zero or above, otherwise zero."""
    return (np.asarray(values) >= 0.0).astype(int)
