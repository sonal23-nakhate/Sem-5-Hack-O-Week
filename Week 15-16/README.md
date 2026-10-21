# Neural Network Basics

A beginner-friendly, NumPy-first project for exploring model deployment,
perceptrons, activations, forward propagation, backpropagation, losses, SGD, and
Adam. The neural network implementation does its own forward pass, gradient
calculation, and parameter updates; it does not use an automatic-differentiation
framework.

## Learning objectives

After working through the demos, you should be able to:

- Explain how a perceptron makes a binary decision and why XOR needs a hidden
  layer.
- Calculate a neural network prediction with matrix multiplication and
  activations.
- Understand how a loss measures prediction error and how backpropagation uses
  the chain rule to calculate gradients.
- Compare basic SGD and Adam updates.
- Save a trained scikit-learn model and reload it to make matching predictions.

## Project layout

```text
neural_network_basics/
├── src/
│   ├── main.py              # Menu and demonstrations
│   ├── perceptron.py        # Perceptron and AND gate
│   ├── activations.py       # NumPy activation functions
│   ├── neural_network.py    # Forward/backward pass and training loop
│   ├── losses.py            # MSE and binary cross-entropy
│   ├── optimizers.py        # SGD and Adam
│   └── deployment.py        # scikit-learn, pickle, and joblib example
├── models/                  # Created serialized models are stored here
├── outputs/                 # Generated plots are stored here
├── tests/
│   └── test_project.py      # Small checks for core functionality
├── requirements.txt
├── README.md
└── .gitignore
```

## Core ideas and formulas

### Perceptrons and layers

A perceptron calculates a weighted sum and applies a step function:

```text
z = x · w + b
prediction = 1 if z >= 0 else 0
```

Here `x` is an input row, `w` is a vector of learned weights, and `b` is a
learned bias. The training rule adjusts parameters by
`w <- w + learning_rate * error * x` and
`b <- b + learning_rate * error`. The AND truth table is linearly separable;
XOR is not, so no single straight-line boundary can classify it correctly.

A multilayer network composes transformations. This project uses an input
layer, one hidden layer, and one binary output unit. The hidden layer gives the
model the ability to combine multiple learned boundaries.

### Activation functions

- **Sigmoid:** `1 / (1 + exp(-z))`, range `(0, 1)`. Common for a binary
  probability output. Its derivative from its output `a` is `a * (1 - a)`.
- **ReLU:** `max(0, z)`. A common hidden-layer activation; derivative is 0 for
  negative inputs and 1 for positive inputs (defined as 0 at exactly zero).
- **Tanh:** `(exp(z) - exp(-z)) / (exp(z) + exp(-z))`, range `(-1, 1)`.
  Sometimes used in hidden layers; derivative from output `a` is `1 - a²`.
- **Softmax:** `exp(z_i - max(z)) / sum_j exp(z_j - max(z))`, converts a row of
  scores into probabilities, often for mutually exclusive classes.
- **Step:** outputs 0 or 1 at a threshold. It is used by the basic perceptron
  but is not differentiable at its threshold and has zero gradient elsewhere.
  Gradient-based network training therefore uses smooth activations instead.

The softmax implementation subtracts the largest score before exponentiation.
Sigmoid inputs are clipped to avoid overflow in the exponential.
For a probability vector `p`, the softmax Jacobian is
`diag(p) - p @ p.T`; the example exposes this Jacobian for one row of scores.

### Forward propagation

For input batch `X` and layer parameters `W` (weights) and `b` (biases):

```text
Z1 = X @ W1 + b1        A1 = sigmoid(Z1)
Z2 = A1 @ W2 + b2        A2 = sigmoid(Z2)
```

`@` is matrix multiplication. A batch has one example per row; biases are
broadcast across those rows. `A2` contains the predicted probability for each
example. The network caches `X`, `A1`, and `A2` for the backward pass.

### Loss and backpropagation

Mean squared error (MSE) is `mean((prediction - target)²)`. Binary
cross-entropy (BCE) for target `y` and predicted probability `p` is:

```text
L = -mean(y * log(p) + (1-y) * log(1-p))
```

Targets are the correct labels; predictions are the model's current estimates.
The loss is smaller when probabilities agree with labels. Probabilities are
clipped before logarithms to prevent `log(0)`.

With a sigmoid output and BCE, the output-layer gradient simplifies to:

```text
dZ2 = (A2 - Y) / m
dW2 = A1.T @ dZ2             db2 = sum_rows(dZ2)
dA1 = dZ2 @ W2.T
dZ1 = dA1 * A1 * (1 - A1)
dW1 = X.T @ dZ1              db1 = sum_rows(dZ1)
```

`m` is the number of examples in the batch. Transposes align matrix
dimensions; summing over rows gives one bias gradient per neuron. These are
chain-rule derivatives. The training loop computes them and applies an
optimizer update without an autodiff library.

```text
Input Data
    → Forward Propagation
    → Prediction
    → Loss Calculation
    → Backpropagation
    → Optimizer Updates
    → Repeat Until Training Completes
    → Save Model
```

For the XOR example, the NumPy network uses sigmoid hidden/output units and BCE.
Its random initialization means changing the seed or hyperparameters can change
convergence; the supplied seed and training settings are tested to learn all
four XOR examples.

### SGD versus Adam

**SGD** applies `parameter <- parameter - learning_rate * gradient`. The
learning rate controls step size: too large may overshoot, while too small can
make learning slow. This project computes full-batch gradients, which is a
simple, reproducible SGD example.

**Adam** maintains an exponentially weighted average of gradients (`m`) and
squared gradients (`v`), then corrects their initial bias:

```text
m = beta1*m + (1-beta1)*gradient
v = beta2*v + (1-beta2)*gradient²
m_hat = m / (1-beta1^t)       v_hat = v / (1-beta2^t)
parameter <- parameter - learning_rate * m_hat / (sqrt(v_hat) + epsilon)
```

`t` is the update number; `beta1`, `beta2`, and `epsilon` control the moving
averages and numerical stability. Adam can work well with less manual tuning,
but neither optimizer is always better. Dataset and hyperparameter choices
matter, so option 7 compares both on the same XOR data and initialization.

### Saving and loading a scikit-learn model

`deployment.py` fits a standard-scaled Logistic Regression pipeline, writes it
as both `models/logistic_regression.pkl` and
`models/logistic_regression.joblib`, reloads both, and verifies predictions on
the same examples. Pickle is part of Python's standard library; joblib is
convenient for objects containing large NumPy arrays. **Never load pickle or
joblib files from untrusted sources**: deserialization can execute code.

The NumPy XOR network is trained in memory for learning purposes; the
serialization demonstration intentionally uses a scikit-learn estimator.

## Setup and run in VS Code (Windows PowerShell)

1. Open the `neural_network_basics` folder in VS Code.
2. Open **Terminal → New Terminal** and ensure the terminal is PowerShell.
3. Create and activate a virtual environment, install dependencies, and launch:

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   python src/main.py
   ```

4. Enter a menu number. Options 2, 4, 5, and 7 save plots in `outputs/`.
   Close each plot window to return to the menu.

If PowerShell blocks environment activation, do not permanently weaken system
policy. Instead, activate only for the current terminal process with:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

Or skip activation and run the virtual environment's interpreter directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe src\main.py
```

## Verification

Run the focused checks from the project root:

```powershell
python -m unittest discover -s tests -v
```

The checks cover the AND perceptron, stable activations, a finite-difference
gradient check, SGD and Adam updates/state, XOR learning, and pickle/joblib
round-trips including the missing-file error.

## Sample output

Exact floating-point values and loss progress can vary with dependency versions,
but the expected results include:

```text
AND gate predictions: [0, 0, 0, 1]
XOR target labels:    [0, 1, 1, 0]
XOR accuracy:         100.0%
pickle ... predictions after load: [ ... ] (identical)
joblib ... predictions after load: [ ... ] (identical)
```

## Troubleshooting

- **`ModuleNotFoundError`:** activate `.venv` and run `pip install -r
  requirements.txt` from the project root.
- **PowerShell activation is blocked:** use the process-only policy command
  above, or invoke `.venv\Scripts\python.exe` directly.
- **A plot window appears to pause the menu:** close the plot window; Matplotlib
  uses a blocking display so the plot can be inspected.
- **Model file not found:** choose menu option 6 first; it creates the model
  files in `models/`.
- **Different learning progress:** compare the loss curves and adjust the
  optimizer learning rate or epoch count in `main.py`; results depend on
  initialization and hyperparameters.
