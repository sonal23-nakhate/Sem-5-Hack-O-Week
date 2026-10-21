"""Focused correctness checks for the educational examples."""

import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from activations import sigmoid, softmax, softmax_jacobian, step  # noqa: E402
from deployment import load_model, save_model, train_demo_model  # noqa: E402
from losses import binary_cross_entropy  # noqa: E402
from neural_network import NeuralNetwork, xor_dataset  # noqa: E402
from optimizers import Adam, SGD  # noqa: E402
from perceptron import and_gate_demo  # noqa: E402


class ProjectChecks(unittest.TestCase):
    def test_perceptron_learns_and(self) -> None:
        _, _, predictions = and_gate_demo()
        np.testing.assert_array_equal(predictions, [0, 0, 0, 1])

    def test_activation_functions_are_stable(self) -> None:
        values = sigmoid(np.array([-1000.0, 0.0, 1000.0]))
        self.assertTrue(np.all(np.isfinite(values)))
        self.assertAlmostEqual(values[1], 0.5)
        np.testing.assert_array_equal(step(np.array([-1.0, 0.0, 1.0])), [0, 1, 1])
        probabilities = softmax(np.array([[10000.0, 10001.0]]))
        self.assertAlmostEqual(float(np.sum(probabilities)), 1.0)
        jacobian = softmax_jacobian(probabilities[0])
        self.assertEqual(jacobian.shape, (2, 2))
        np.testing.assert_allclose(np.sum(jacobian, axis=0), 0.0, atol=1e-12)

    def test_backprop_gradients_match_finite_differences(self) -> None:
        inputs = np.array([[0.0, 1.0], [1.0, 0.0]])
        targets = np.array([[1.0], [1.0]])
        model = NeuralNetwork(hidden_size=3, seed=7)
        predictions = model.forward(inputs)
        gradients = model.backward(targets)
        epsilon = 1e-5
        parameter = model.parameters["W1"]
        original = parameter[0, 0]
        parameter[0, 0] = original + epsilon
        plus = binary_cross_entropy(model.forward(inputs), targets)
        parameter[0, 0] = original - epsilon
        minus = binary_cross_entropy(model.forward(inputs), targets)
        parameter[0, 0] = original
        numerical_gradient = (plus - minus) / (2.0 * epsilon)
        self.assertAlmostEqual(
            gradients["W1"][0, 0], numerical_gradient, places=5
        )

    def test_optimizers_update_and_adam_tracks_time(self) -> None:
        sgd_parameter = {"w": np.array([1.0])}
        SGD(learning_rate=0.1).step(sgd_parameter, {"w": np.array([2.0])})
        self.assertAlmostEqual(float(sgd_parameter["w"][0]), 0.8)

        adam_parameter = {"w": np.array([1.0])}
        adam = Adam(learning_rate=0.1)
        adam.step(adam_parameter, {"w": np.array([1.0])})
        adam.step(adam_parameter, {"w": np.array([1.0])})
        self.assertEqual(adam.time_step, 2)
        self.assertLess(float(adam_parameter["w"][0]), 1.0)
        self.assertEqual(set(adam.first_moments), {"w"})

    def test_network_learns_xor(self) -> None:
        inputs, targets = xor_dataset()
        model = NeuralNetwork(optimizer="adam", learning_rate=0.05, seed=42)
        model.train(inputs, targets, epochs=10000, print_every=0)
        np.testing.assert_array_equal(model.predict(inputs), targets.astype(int))

    def test_pickle_and_joblib_round_trip(self) -> None:
        model, inputs, _ = train_demo_model()
        expected = model.predict(inputs[:5])
        with TemporaryDirectory() as temporary_directory:
            for method, extension in (("pickle", ".pkl"), ("joblib", ".joblib")):
                path = Path(temporary_directory) / f"model{extension}"
                save_model(model, path, method)
                restored = load_model(path, method)
                np.testing.assert_array_equal(restored.predict(inputs[:5]), expected)
            with self.assertRaises(FileNotFoundError):
                load_model(Path(temporary_directory) / "missing.pkl", "pickle")


if __name__ == "__main__":
    unittest.main()
