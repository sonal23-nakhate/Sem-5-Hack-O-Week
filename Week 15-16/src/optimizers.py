"""Parameter update algorithms implemented with NumPy."""

from typing import Dict

import numpy as np

ArrayDict = Dict[str, np.ndarray]


class SGD:
    """Stochastic Gradient Descent (used here with full-batch gradients).

    The learning rate controls the size of each step in the direction opposite
    the gradient. Steps that are too large can overshoot; small steps train
    slowly.
    """

    def __init__(self, learning_rate: float = 0.1) -> None:
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        self.learning_rate = learning_rate

    def step(self, parameters: ArrayDict, gradients: ArrayDict) -> None:
        for name, parameter in parameters.items():
            parameter -= self.learning_rate * gradients[name]


class Adam:
    """Adaptive Moment Estimation with per-parameter state and bias correction."""

    def __init__(
        self,
        learning_rate: float = 0.01,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
    ) -> None:
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if not 0 <= beta1 < 1 or not 0 <= beta2 < 1:
            raise ValueError("beta1 and beta2 must be in the interval [0, 1)")
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")

        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.time_step = 0
        self.first_moments: ArrayDict = {}
        self.second_moments: ArrayDict = {}

    def step(self, parameters: ArrayDict, gradients: ArrayDict) -> None:
        self.time_step += 1
        for name, parameter in parameters.items():
            gradient = gradients[name]
            if name not in self.first_moments:
                self.first_moments[name] = np.zeros_like(parameter)
                self.second_moments[name] = np.zeros_like(parameter)

            first = self.first_moments[name]
            second = self.second_moments[name]
            first *= self.beta1
            first += (1.0 - self.beta1) * gradient
            second *= self.beta2
            second += (1.0 - self.beta2) * np.square(gradient)

            corrected_first = first / (1.0 - self.beta1**self.time_step)
            corrected_second = second / (1.0 - self.beta2**self.time_step)
            parameter -= self.learning_rate * corrected_first / (
                np.sqrt(corrected_second) + self.epsilon
            )
