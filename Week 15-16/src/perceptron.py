"""A single-layer perceptron trained with the perceptron learning rule."""

import numpy as np

try:
    from .activations import step
except ImportError:  # Supports running this file directly as a script.
    from activations import step


class Perceptron:
    """A linear decision unit followed by a non-differentiable step function.

    A single perceptron can separate only linearly separable classes. XOR is
    not linearly separable, so it requires a network with a hidden layer.
    """

    def __init__(
        self, input_size: int, learning_rate: float = 0.1, seed: int = 42
    ) -> None:
        if input_size < 1:
            raise ValueError("input_size must be at least 1")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        rng = np.random.default_rng(seed)
        self.weights = rng.normal(0.0, 0.1, size=input_size)
        self.bias = 0.0
        self.learning_rate = learning_rate

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        inputs = np.asarray(inputs, dtype=float)
        return step(inputs @ self.weights + self.bias)

    def train(
        self, inputs: np.ndarray, targets: np.ndarray, epochs: int = 20
    ) -> None:
        inputs = np.asarray(inputs, dtype=float)
        targets = np.asarray(targets, dtype=int).reshape(-1)
        if inputs.ndim != 2 or inputs.shape[1] != self.weights.size:
            raise ValueError("inputs must have shape (samples, input_size)")
        if len(inputs) != len(targets):
            raise ValueError("inputs and targets must contain the same samples")
        if epochs < 1:
            raise ValueError("epochs must be at least 1")

        for _ in range(epochs):
            errors = targets - self.predict(inputs)
            self.weights += self.learning_rate * (inputs.T @ errors)
            self.bias += self.learning_rate * np.sum(errors)


def and_gate_demo() -> tuple[Perceptron, np.ndarray, np.ndarray]:
    """Train and return a perceptron and its predictions on the AND truth table."""
    inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    targets = np.array([0, 0, 0, 1])
    model = Perceptron(input_size=2, learning_rate=0.1, seed=42)
    model.train(inputs, targets, epochs=30)
    return model, inputs, model.predict(inputs)


if __name__ == "__main__":
    trained, examples, predictions = and_gate_demo()
    print("AND gate perceptron")
    print(f"Weights: {trained.weights}")
    print(f"Bias: {trained.bias:.3f}")
    print(f"Predictions: {predictions.tolist()}")
    print("Targets:     [0, 0, 0, 1]")
    print("A single perceptron cannot represent XOR because XOR is not linearly separable.")
