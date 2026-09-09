from models import LinearRegression, LogisticRegression, KNNClassifier, expand_polynomial_features

print("=== 1. Linear & Ridge/Lasso Regression ===")
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

print(f"Ridge Reg Prediction for x=5: {lin_model.forward([5.0]).data:.2f} (Target ~11.0)")

print("\n=== 2. Polynomial Regression ===")
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
print(f"Polynomial Reg Prediction for x=5 (x^2): {poly_model.forward(test_x_norm).data:.2f} (Target = 25.0)")

print("\n=== 3. Logistic Regression (Classification) ===")
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

print(f"Logistic Reg prob for x=1.5: {log_model.forward([1.5]).data:.4f} (Target Class 0)")
print(f"Logistic Reg prob for x=3.5: {log_model.forward([3.5]).data:.4f} (Target Class 1)")

print("\n=== 4. K-Nearest Neighbors (KNN) ===")
X_knn = [[1.0, 2.0], [1.5, 1.8], [5.0, 8.0], [6.0, 9.0]]
Y_knn = [0, 0, 1, 1]

knn = KNNClassifier(k=3)
knn.fit(X_knn, Y_knn)

test_point = [1.2, 1.9]
res = knn.predict(test_point)

print(f"Test Point: {test_point}")
print(f"Predicted Class: {res['predicted_class']}")
print(f"Class 1 Probability: {res['class_1_probability']:.2f}")
print("3 Nearest Neighbors (Distance, Label, Point):")
for dist, label, pt in res['neighbors']:
    print(f"  - Dist: {dist} | Label: {label} | Point: {pt}")