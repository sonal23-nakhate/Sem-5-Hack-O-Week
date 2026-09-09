import math
from engine import Value, dot

class LinearRegression:
    def __init__(self, n_features):
        self.w = [Value(0.0) for _ in range(n_features)]
        self.b = Value(0.0)

    def forward(self, x_vec):
        return dot(self.w, [Value(xi) for xi in x_vec]) + self.b

    def loss(self, X, Y, penalty="none", alpha=0.01):
        total_loss = Value(0.0)
        for x, y in zip(X, Y):
            pred = self.forward(x)
            diff = pred + Value(-y)
            total_loss = total_loss + (diff * diff)
        
        if penalty == "ridge":
            reg = sum((wi * wi for wi in self.w), Value(0.0))
            total_loss = total_loss + (Value(alpha) * reg)
        elif penalty == "lasso":
            reg = sum((Value(abs(wi.data)) for wi in self.w), Value(0.0))
            total_loss = total_loss + (Value(alpha) * reg)
            
        return total_loss

class LogisticRegression:
    def __init__(self, n_features):
        self.w = [Value(0.0) for _ in range(n_features)]
        self.b = Value(0.0)

    def forward(self, x_vec):
        z = dot(self.w, [Value(xi) for xi in x_vec]) + self.b
        return z.sigmoid()

    def loss(self, X, Y):
        total_loss = Value(0.0)
        for x, y in zip(X, Y):
            pred = self.forward(x)
            diff = pred + Value(-y)
            total_loss = total_loss + (diff * diff)
        return total_loss

class KNNClassifier:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, Y):
        self.X_train = X
        self.Y_train = Y

    def predict(self, x_test):
        distances = []
        for x_train, y_train in zip(self.X_train, self.Y_train):
            dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(x_test, x_train)))
            distances.append((dist, y_train, x_train))
        
        distances.sort(key=lambda item: item[0])
        neighbors = distances[:self.k]
        
        k_nearest_labels = [label for _, label, _ in neighbors]
        predicted_class = max(set(k_nearest_labels), key=k_nearest_labels.count)
        prob_class_1 = k_nearest_labels.count(1) / self.k

        return {
            "predicted_class": predicted_class,
            "class_1_probability": prob_class_1,
            "neighbors": [(round(dist, 4), label, point) for dist, label, point in neighbors]
        }

def expand_polynomial_features(X, degree=2):
    X_poly = []
    for x in X:
        row = [x[0] ** d for d in range(1, degree + 1)]
        X_poly.append(row)
    return X_poly

# --- Week 9-10: Preprocessing & Metrics ---

def fill_missing_data(X, strategy="mean"):
    n_rows = len(X)
    n_cols = len(X[0])
    X_filled = [row[:] for row in X]

    for j in range(n_cols):
        valid_vals = [X[i][j] for i in range(n_rows) if X[i][j] is not None]
        if not valid_vals:
            continue
        
        if strategy == "mean":
            fill_val = sum(valid_vals) / len(valid_vals)
        elif strategy == "median":
            sorted_vals = sorted(valid_vals)
            mid = len(sorted_vals) // 2
            fill_val = sorted_vals[mid] if len(sorted_vals) % 2 != 0 else (sorted_vals[mid-1] + sorted_vals[mid]) / 2.0

        for i in range(n_rows):
            if X_filled[i][j] is None:
                X_filled[i][j] = fill_val

    return X_filled

class StandardScaler:
    def fit_transform(self, X):
        n_cols = len(X[0])
        self.means = [sum(row[j] for row in X) / len(X) for j in range(n_cols)]
        self.stds = [
            math.sqrt(sum((row[j] - self.means[j]) ** 2 for row in X) / len(X)) or 1.0
            for j in range(n_cols)
        ]
        return [[(row[j] - self.means[j]) / self.stds[j] for j in range(n_cols)] for row in X]

class MinMaxScaler:
    def fit_transform(self, X):
        n_cols = len(X[0])
        self.mins = [min(row[j] for row in X) for j in range(n_cols)]
        self.maxs = [max(row[j] for row in X) for j in range(n_cols)]
        return [
            [
                (row[j] - self.mins[j]) / (self.maxs[j] - self.mins[j]) if self.maxs[j] != self.mins[j] else 0.0
                for j in range(n_cols)
            ]
            for row in X
        ]

def train_test_split(X, Y, test_size=0.25, seed=42):
    import random
    random.seed(seed)
    indices = list(range(len(X)))
    random.shuffle(indices)
    
    split_idx = int(len(X) * (1 - test_size))
    train_idx, test_idx = indices[:split_idx], indices[split_idx:]
    
    return [X[i] for i in train_idx], [X[i] for i in test_idx], [Y[i] for i in train_idx], [Y[i] for i in test_idx]

def confusion_matrix(y_true, y_pred):
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    return {"TP": tp, "TN": tn, "FP": fp, "FN": fn}

def classification_metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    tp, tn, fp, fn = cm["TP"], cm["TN"], cm["FP"], cm["FN"]
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / len(y_true) if len(y_true) > 0 else 0.0
    
    return {"Accuracy": accuracy, "Precision": precision, "Recall": recall, "F1-Score": f1, "CM": cm}

def calculate_roc_auc(y_true, y_scores):
    paired = sorted(zip(y_scores, y_true), key=lambda x: x[0], reverse=True)
    num_pos = sum(y_true)
    num_neg = len(y_true) - num_pos
    if num_pos == 0 or num_neg == 0:
        return 0.5

    tpr_list, fpr_list = [0.0], [0.0]
    tp, fp = 0, 0

    for score, label in paired:
        if label == 1:
            tp += 1
        else:
            fp += 1
        tpr_list.append(tp / num_pos)
        fpr_list.append(fp / num_neg)

    auc = 0.0
    for i in range(1, len(fpr_list)):
        auc += (fpr_list[i] - fpr_list[i-1]) * (tpr_list[i] + tpr_list[i-1]) / 2.0
    return auc

def k_fold_cross_validation(X, Y, model_cls, k=3):
    n = len(X)
    fold_size = n // k
    scores = []
    
    for i in range(k):
        val_start = i * fold_size
        val_end = val_start + fold_size if i != k - 1 else n
        
        X_val, Y_val = X[val_start:val_end], Y[val_start:val_end]
        X_train = X[:val_start] + X[val_end:]
        Y_train = Y[:val_start] + Y[val_end:]
        
        model = model_cls()
        model.fit(X_train, Y_train)
        
        preds = [model.predict(x)["predicted_class"] for x in X_val]
        acc = sum(1 for p, y in zip(preds, Y_val) if p == y) / len(Y_val)
        scores.append(acc)
        
    return sum(scores) / len(scores)