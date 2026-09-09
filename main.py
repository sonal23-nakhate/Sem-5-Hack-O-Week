import random
from engine import Value, dot, matvec_mul, power_iteration
from models import LinearRegression, LogisticRegression, KNNClassifier, expand_polynomial_features

print("==================================================")
print("  WEEK 5 & 6: Neural Network & Linear Algebra     ")
print("==================================================")

# 1. XOR Neural Network Training
X_xor = [[0, 0], [0, 1], [1, 0], [1, 1]]
Y_xor = [0, 1, 1, 0]

random.seed(1337)
W1 = [[Value(random.uniform(-1, 1)) for _ in range(2)] for _ in range(4)]
b1 = [Value(0) for _ in range(4)]
W2 = [Value(random.uniform(-1, 1)) for _ in range(4)]
b2 = Value(0)

all_params = [p for row in W1 for p in row] + b1 + W2 + [b2]

lr = 0.5
for epoch in range(1000):
    total_loss = Value(0)
    for x_raw, y_target in zip(X_xor, Y_xor):
        x = [Value(x_raw[0]), Value(x_raw[1])]
        hidden_linear = matvec_mul(W1, x)
        hidden = [(h + b).sigmoid() for h, b in zip(hidden_linear, b1)]
        pred = (dot(W2, hidden) + b2).sigmoid()
        
        diff = pred + Value(-y_target)
        loss = diff * diff
        total_loss = total_loss + loss

    total_loss.backward()

    for p in all_params:
        p.data -= lr * p.grad
        p.grad = 0.0

print("Neural Net (XOR) Predictions:")
for x_raw, y_target in zip(X_xor, Y_xor):
    x = [Value(x_raw[0]), Value(x_raw[1])]
    hidden_linear = matvec_mul(W1, x)
    hidden = [(h + b).sigmoid() for h, b in zip(hidden_linear, b1)]
    pred = (dot(W2, hidden) + b2).sigmoid()
    print(f"  Input: {x_raw} | Target: {y_target} | Predicted: {pred.data:.4f}")

# 2. Eigenvalue Power Iteration
A = [[2.0, 1.0], [1.0, 2.0]]
val, vec = power_iteration(A)
print(f"\nPower Iteration -> Dominant Eigenvalue: {val:.4f} | Eigenvector: [{vec[0]:.4f}, {vec[1]:.4f}]")

print("\n==================================================")
print("  WEEK 7 & 8: Regression & Classification          ")
print("==================================================")

# 3. Ridge Regression
X_reg = [[1.0], [2.0], [3.0], [4.0]]
Y_reg = [3.0, 5.0, 7.0, 9.0]
lin_model = LinearRegression(n_features=1)
params = lin_model.w + [lin_model.b]

for epoch in range(200):
    loss = lin_model.loss(X_reg, Y_reg, penalty="ridge", alpha=0.01)
    loss.backward()
    for p in params:
        p.data -= 0.02 * p.grad
        p.grad = 0.0

print(f"Ridge Reg Prediction (x=5): {lin_model.forward([5.0]).data:.2f} (Target ~11.0)")

# 4. Polynomial Regression
X_poly_raw = [[1.0], [2.0], [3.0], [4.0]]
Y_poly = [1.0, 4.0, 9.0, 16.0]
X_poly = expand_polynomial_features(X_poly_raw, degree=2)
X_poly_norm = [[x1 / 4.0, x2 / 16.0] for x1, x2 in X_poly]

poly_model = LinearRegression(n_features=2)
params = poly_model.w + [poly_model.b]

for epoch in range(1000):
    loss = poly_model.loss(X_poly_norm, Y_poly)
    loss.backward()
    for p in params:
        p.data -= 0.01 * p.grad
        p.grad = 0.0

test_x_norm = [5.0 / 4.0, 25.0 / 16.0]
print(f"Polynomial Reg Prediction (x=5): {poly_model.forward(test_x_norm).data:.2f} (Target = 25.0)")

# 5. Logistic Regression
X_class = [[1.0], [2.0], [3.0], [4.0]]
Y_class = [0.0, 0.0, 1.0, 1.0]
log_model = LogisticRegression(n_features=1)
params = log_model.w + [log_model.b]

for epoch in range(500):
    loss = log_model.loss(X_class, Y_class)
    loss.backward()
    for p in params:
        p.data -= 0.2 * p.grad
        p.grad = 0.0

print(f"Logistic Reg Prob (x=1.5): {log_model.forward([1.5]).data:.4f} (Target Class 0)")
print(f"Logistic Reg Prob (x=3.5): {log_model.forward([3.5]).data:.4f} (Target Class 1)")

# 6. KNN Classifier
X_knn = [[1.0, 2.0], [1.5, 1.8], [5.0, 8.0], [6.0, 9.0]]
Y_knn = [0, 0, 1, 1]
knn = KNNClassifier(k=3)
knn.fit(X_knn, Y_knn)
test_point = [1.2, 1.9]
print(f"KNN Prediction for {test_point}: Class {knn.predict(test_point)}")