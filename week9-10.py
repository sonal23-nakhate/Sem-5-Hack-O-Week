import sys
sys.path.append("../Week 5-6")  # Points to models.py in the previous folder

import random
from models import (
    fill_missing_data, StandardScaler, MinMaxScaler,
    train_test_split, classification_metrics, calculate_roc_auc,
    k_fold_cross_validation, KNNClassifier
)

print("=== 1. Feature Engineering & Scaling ===")
X_raw = [
    [10.0, None],
    [20.0, 200.0],
    [30.0, 300.0],
    [None, 400.0],
    [50.0, 500.0]
]

# Impute missing values
X_clean = fill_missing_data(X_raw, strategy="mean")
print("Imputed Missing Data (Mean Strategy):")
for row in X_clean:
    print(f"  {row}")

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_clean)
print("\nStandardized Features (Mean=0, Std=1):")
for row in X_scaled:
    print(f"  [{row[0]:.2f}, {row[1]:.2f}]")

print("\n=== 2. Train/Test Split & Metrics ===")
X_data = [[1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0], [8.0]]
Y_data = [0, 0, 0, 0, 1, 1, 1, 1]

X_tr, X_te, Y_tr, Y_te = train_test_split(X_data, Y_data, test_size=0.25, seed=42)
print(f"Train size: {len(X_tr)} | Test size: {len(X_te)}")

# Mock binary classifier predictions & raw scores
y_true = [0, 0, 1, 1, 1, 0, 1, 0]
y_pred = [0, 0, 1, 1, 0, 0, 1, 1]
y_scores = [0.1, 0.2, 0.85, 0.9, 0.4, 0.3, 0.75, 0.6]

metrics = classification_metrics(y_true, y_pred)
auc_score = calculate_roc_auc(y_true, y_scores)

print(f"Confusion Matrix: {metrics['CM']}")
print(f"Accuracy:  {metrics['Accuracy']:.2f}")
print(f"Precision: {metrics['Precision']:.2f}")
print(f"Recall:    {metrics['Recall']:.2f}")
print(f"F1-Score:  {metrics['F1-Score']:.2f}")
print(f"ROC-AUC:   {auc_score:.4f}")

print("\n=== 3. 3-Fold Cross-Validation (KNN) ===")
X_cv = [[1.0, 2.0], [1.5, 1.8], [5.0, 8.0], [6.0, 9.0], [1.1, 2.1], [5.5, 8.5]]
Y_cv = [0, 0, 1, 1, 0, 1]

cv_acc = k_fold_cross_validation(X_cv, Y_cv, model_cls=lambda: KNNClassifier(k=3), k=3)
print(f"3-Fold CV Average Accuracy: {cv_acc * 100:.2f}%")