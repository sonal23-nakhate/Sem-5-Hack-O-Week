"""A small feedforward neural network with manual backpropagation."""

from typing import Dict, List, Optional

import numpy as np

try:
    from .activations import sigmoid, sigmoid_derivative_from_output
    from .losses import binary_cross_entropy
    from .optimizers import Adam, SGD
except ImportError:  # Supports running this file directly as a script.
    from activations import sigmoid, sigmoid_derivative_from_output
    from losses import binary_cross_entropy
    from optimizers import Adam, SGD


class NeuralNetwork:
    """One-hidden-layer network for binary classification, built using NumPy."""

    def __init__(
        self,
        input_size: int = 2,
        hidden_size: int = 4,
        output_size: int = 1,
        optimizer: str = "adam",
        learning_rate: Optional[float] = None,
        seed: int = 42,
    ) -> None:
        if min(input_size, hidden_size, output_size) < 1:
            raise ValueError("Layer sizes must be at least 1")
        if output_size != 1:
            raise ValueError("This example supports one binary output")

        rng = np.random.default_rng(seed)
        self.parameters: Dict[str, np.ndarray] = {
            "W1": rng.normal(
                0.0, np.sqrt(2.0 / (input_size + hidden_size)),
                size=(input_size, hidden_size),
            ),
            "b1": np.zeros((1, hidden_size)),
            "W2": rng.normal(
                0.0, np.sqrt(2.0 / (hidden_size + output_size)),
                size=(hidden_size, output_size),
            ),
            "b2": np.zeros((1, output_size)),
        }
        self.optimizer_name = optimizer.lower()
        if self.optimizer_name == "sgd":
            self.optimizer = SGD(
                learning_rate if learning_rate is not None else 0.8
            )
        elif self.optimizer_name == "adam":
            self.optimizer = Adam(
                learning_rate if learning_rate is not None else 0.05
            )
        else:
            raise ValueError("optimizer must be 'sgd' or 'adam'")
        self.cache: Dict[str, np.ndarray] = {}

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """Compute predictions and cache values needed by backpropagation."""
        inputs = np.asarray(inputs, dtype=float)
        if inputs.ndim != 2 or inputs.shape[1] != self.parameters["W1"].shape[0]:
            raise ValueError("inputs must have shape (samples, input_size)")
        z1 = inputs @ self.parameters["W1"] + self.parameters["b1"]
        a1 = sigmoid(z1)
        z2 = a1 @ self.parameters["W2"] + self.parameters["b2"]
        a2 = sigmoid(z2)
        self.cache = {"X": inputs, "A1": a1, "A2": a2}
        return a2

    def backward(self, targets: np.ndarray) -> Dict[str, np.ndarray]:
        """Calculate parameter gradients using the chain rule."""
        if not self.cache:
            raise RuntimeError("Call forward(inputs) before backward(targets)")
        targets = np.asarray(targets, dtype=float).reshape(-1, 1)
        inputs = self.cache["X"]  # (samples, input_size)
        hidden_output = self.cache["A1"]  # (samples, hidden_size)
        prediction = self.cache["A2"]  # (samples, 1)
        sample_count = inputs.shape[0]
        if targets.shape != prediction.shape:
            raise ValueError("targets must contain one binary label per sample")

        # For sigmoid + BCE, dL/dZ2 simplifies to (prediction - target) / m.
        dz2 = (prediction - targets) / sample_count  # (samples, 1)
        dw2 = hidden_output.T @ dz2  # (hidden_size, 1)
        db2 = np.sum(dz2, axis=0, keepdims=True)  # (1, 1)

        # Chain the output error through W2 and the hidden sigmoid activation.
        da1 = dz2 @ self.parameters["W2"].T  # (samples, hidden_size)
        dz1 = da1 * sigmoid_derivative_from_output(hidden_output)
        dw1 = inputs.T @ dz1  # (input_size, hidden_size)
        db1 = np.sum(dz1, axis=0, keepdims=True)  # (1, hidden_size)
        return {"W1": dw1, "b1": db1, "W2": dw2, "b2": db2}

    def train(
        self,
        inputs: np.ndarray,
        targets: np.ndarray,
        epochs: int = 10000,
        print_every: int = 2000,
    ) -> List[float]:
        """Train with full-batch gradients and return loss sampled each epoch."""
        inputs = np.asarray(inputs, dtype=float)
        targets = np.asarray(targets, dtype=float).reshape(-1, 1)
        if len(inputs) != len(targets) or len(inputs) == 0:
            raise ValueError("inputs and targets must have the same nonzero size")
        if epochs < 1 or print_every < 0:
            raise ValueError("epochs must be positive and print_every nonnegative")

        history: List[float] = []
        for epoch in range(1, epochs + 1):
            predictions = self.forward(inputs)
            loss = binary_cross_entropy(predictions, targets)
            history.append(loss)
            gradients = self.backward(targets)
            self.optimizer.step(self.parameters, gradients)
            if print_every and (epoch == 1 or epoch % print_every == 0):
                print(
                    f"{self.optimizer_name.upper()} epoch {epoch:5d}/{epochs}: "
                    f"loss={loss:.6f}"
                )
        return history

    def predict_proba(self, inputs: np.ndarray) -> np.ndarray:
        return self.forward(inputs)

    def predict(self, inputs: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(inputs) >= threshold).astype(int)


def xor_dataset() -> tuple[np.ndarray, np.ndarray]:
    """Return the four input/target examples for the XOR truth table."""
    inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    targets = np.array([[0], [1], [1], [0]], dtype=float)
    return inputs, targets
