"""Interactive demonstrations for the neural_network_basics project."""

from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np

try:
    from .activations import (
        relu,
        sigmoid,
        softmax,
        softmax_jacobian,
        step,
        tanh,
    )
    from .deployment import deployment_demo
    from .losses import binary_cross_entropy, mean_squared_error
    from .neural_network import NeuralNetwork, xor_dataset
    from .perceptron import and_gate_demo
except ImportError:  # Supports `python src/main.py` from the project root.
    from activations import relu, sigmoid, softmax, softmax_jacobian, step, tanh
    from deployment import deployment_demo
    from losses import binary_cross_entropy, mean_squared_error
    from neural_network import NeuralNetwork, xor_dataset
    from perceptron import and_gate_demo


PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_DIR / "outputs"


def show_perceptron() -> None:
    model, inputs, predictions = and_gate_demo()
    print("\nAND gate")
    print(f"Weights: {model.weights}")
    print(f"Bias: {model.bias:.3f}")
    print(f"Input rows: {inputs.astype(int).tolist()}")
    print(f"Predictions: {predictions.tolist()}")
    print("Targets:     [0, 0, 0, 1]")
    print("One perceptron draws one linear boundary; it cannot separate XOR.")


def show_activations() -> None:
    samples = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    print("\nActivation examples")
    print(f"Input:   {samples}")
    print(f"Sigmoid: {sigmoid(samples)} (often used for binary output probabilities)")
    print(f"ReLU:    {relu(samples)} (common hidden-layer activation)")
    print(f"Tanh:    {tanh(samples)} (zero-centered hidden-layer activation)")
    print(f"Step:    {step(samples)} (basic perceptron; not differentiable)")
    scores = np.array([[1.0, 2.0, 3.0]])
    print(f"Softmax({scores.tolist()}): {softmax(scores)} (multiclass probabilities)")
    print(f"Softmax Jacobian for that row:\n{softmax_jacobian(softmax(scores)[0])}")
    print(
        "The step function is suitable for a basic perceptron, but its zero "
        "gradient prevents gradient-based training. Smooth activations such as "
        "sigmoid, tanh, and ReLU are differentiable almost everywhere."
    )

    curve_x = np.linspace(-6, 6, 300)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.plot(curve_x, sigmoid(curve_x), label="Sigmoid")
    plt.plot(curve_x, relu(curve_x), label="ReLU")
    plt.plot(curve_x, tanh(curve_x), label="Tanh")
    plt.plot(curve_x, step(curve_x), label="Step")
    plt.title("Activation functions")
    plt.xlabel("Input")
    plt.ylabel("Output")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plot_path = OUTPUT_DIR / "activation_functions.png"
    plt.savefig(plot_path)
    print(f"Saved activation plot to {plot_path}")
    plt.show()
    plt.close()


def show_forward_and_loss() -> None:
    inputs, targets = xor_dataset()
    model = NeuralNetwork(seed=42, optimizer="adam")
    predictions = model.forward(inputs)
    print("\nForward propagation example (untrained network)")
    print(f"Predicted probabilities:\n{predictions.round(4)}")
    print(f"Targets:\n{targets.astype(int)}")
    print(f"Binary cross-entropy: {binary_cross_entropy(predictions, targets):.6f}")
    print(f"Mean squared error:   {mean_squared_error(predictions, targets):.6f}")


def train_xor(
    optimizer_name: str, epochs: int = 10000, print_every: int = 2000
) -> Tuple[NeuralNetwork, List[float], float]:
    inputs, targets = xor_dataset()
    model = NeuralNetwork(optimizer=optimizer_name, seed=42)
    history = model.train(inputs, targets, epochs=epochs, print_every=print_every)
    predictions = model.predict(inputs)
    accuracy = float(np.mean(predictions == targets))
    print("\nXOR results")
    print(f"Input rows:           {inputs.astype(int).tolist()}")
    print(f"Target labels:        {targets.astype(int).ravel().tolist()}")
    print(f"Predicted labels:     {predictions.ravel().tolist()}")
    probabilities = model.predict_proba(inputs).ravel().round(4).tolist()
    print(f"Predicted probabilities: {probabilities}")
    print(f"Classification accuracy: {accuracy * 100:.1f}%")
    return model, history, accuracy


def plot_loss(
    history: List[float], optimizer_name: str, filename: str
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.plot(np.arange(1, len(history) + 1), history, label=optimizer_name.upper())
    plt.title(f"XOR training loss - {optimizer_name.upper()}")
    plt.xlabel("Epoch")
    plt.ylabel("Binary cross-entropy")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    path = OUTPUT_DIR / filename
    plt.savefig(path)
    print(f"Saved loss plot to {path}")
    plt.show()
    plt.close()


def compare_optimizers() -> None:
    print("\nComparing both optimizers on the same XOR data and initialization")
    sgd_model, sgd_history, sgd_accuracy = train_xor(
        "sgd", epochs=10000, print_every=2500
    )
    adam_model, adam_history, adam_accuracy = train_xor(
        "adam", epochs=10000, print_every=2500
    )
    del sgd_model, adam_model
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.plot(sgd_history, label=f"SGD ({sgd_accuracy * 100:.0f}% accuracy)")
    plt.plot(adam_history, label=f"Adam ({adam_accuracy * 100:.0f}% accuracy)")
    plt.title("Optimizer comparison on XOR")
    plt.xlabel("Epoch")
    plt.ylabel("Binary cross-entropy")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    path = OUTPUT_DIR / "optimizer_comparison.png"
    plt.savefig(path)
    print(f"Saved comparison plot to {path}")
    plt.show()
    plt.close()
    print("Neither optimizer is always better; outcomes depend on data and settings.")


def print_menu() -> None:
    print(
        "\n=== Neural Network Basics ===\n"
        "1. Train and test a perceptron on the AND gate\n"
        "2. Display activation functions\n"
        "3. Demonstrate forward propagation and loss calculation\n"
        "4. Train a NumPy neural network on XOR using SGD\n"
        "5. Train a NumPy neural network on XOR using Adam\n"
        "6. Save and load a scikit-learn model using pickle and joblib\n"
        "7. Compare optimizer performance\n"
        "8. Exit"
    )


def main() -> None:
    actions = {
        "1": show_perceptron,
        "2": show_activations,
        "3": show_forward_and_loss,
        "4": lambda: (
            plot_loss(*_history_for_plot("sgd"), "xor_sgd_loss.png")
        ),
        "5": lambda: (
            plot_loss(*_history_for_plot("adam"), "xor_adam_loss.png")
        ),
        "6": deployment_demo,
        "7": compare_optimizers,
    }
    while True:
        print_menu()
        choice = input("Choose an option (1-8): ").strip()
        if choice == "8":
            print("Goodbye! Keep experimenting.")
            return
        action = actions.get(choice)
        if action is None:
            print("Please enter a number from 1 to 8.")
            continue
        try:
            action()
        except (ValueError, FileNotFoundError, RuntimeError) as error:
            print(f"Could not complete that demonstration: {error}")


def _history_for_plot(optimizer_name: str) -> Tuple[List[float], str]:
    _, history, _ = train_xor(optimizer_name)
    return history, optimizer_name


if __name__ == "__main__":
    main()
