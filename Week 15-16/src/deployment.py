"""Train, save, reload, and compare a small scikit-learn classifier."""

import pickle
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"


def train_demo_model() -> tuple[Any, np.ndarray, np.ndarray]:
    """Fit a Logistic Regression pipeline on a reproducible sample dataset."""
    inputs, labels = make_classification(
        n_samples=200,
        n_features=4,
        n_informative=3,
        n_redundant=0,
        random_state=42,
    )
    model = make_pipeline(StandardScaler(), LogisticRegression(random_state=42))
    model.fit(inputs, labels)
    return model, inputs, labels


def save_model(model: Any, file_path: Path, method: str) -> None:
    """Serialize a model using pickle or joblib."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    if method == "pickle":
        with file_path.open("wb") as model_file:
            pickle.dump(model, model_file)
    elif method == "joblib":
        joblib.dump(model, file_path)
    else:
        raise ValueError("method must be 'pickle' or 'joblib'")


def load_model(file_path: Path, method: str) -> Any:
    """Load a serialized model, reporting missing files clearly.

    Pickle and joblib can execute code during deserialization. Only load files
    produced by a trusted source.
    """
    if not file_path.is_file():
        raise FileNotFoundError(
            f"Model file not found: {file_path}. Run the deployment demo first."
        )
    if method == "pickle":
        with file_path.open("rb") as model_file:
            return pickle.load(model_file)
    if method == "joblib":
        return joblib.load(file_path)
    raise ValueError("method must be 'pickle' or 'joblib'")


def deployment_demo() -> Dict[str, np.ndarray]:
    """Save with both formats and verify predictions survive a reload."""
    model, inputs, _ = train_demo_model()
    examples = inputs[:5]
    before = model.predict(examples)
    paths = {
        "pickle": MODEL_DIR / "logistic_regression.pkl",
        "joblib": MODEL_DIR / "logistic_regression.joblib",
    }
    results: Dict[str, np.ndarray] = {}
    for method, path in paths.items():
        save_model(model, path, method)
        restored = load_model(path, method)
        after = restored.predict(examples)
        if not np.array_equal(before, after):
            raise RuntimeError(f"{method} model predictions changed after loading")
        results[method] = after
        print(f"{method}: saved to {path}")
        print(f"  predictions before save: {before.tolist()}")
        print(f"  predictions after load:  {after.tolist()} (identical)")

    print(
        "pickle is built into Python; joblib is convenient for NumPy-heavy "
        "objects and can handle large arrays efficiently. Both require trusted files."
    )
    return results
