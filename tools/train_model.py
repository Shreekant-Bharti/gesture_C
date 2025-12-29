"""
train_model.py
Loads CSVs from gesture_data/, trains a RandomForest model, evaluates it,
and saves a single pickle file gesture_model.pkl.

Outputs:
- accuracy
- classification report
- confusion matrix
- per-class accuracy
"""

import os
import glob
import pickle
from typing import Tuple, List, Dict

import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

from config import DATA, MODEL


def load_dataset(data_dir: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Reads all CSV files. Label = filename without extension.
    Returns: X, y (string labels), labels_sorted
    """
    pattern = os.path.join(data_dir, "*.csv")
    files = sorted(glob.glob(pattern))

    if not files:
        raise FileNotFoundError(
            f"No CSV files found in '{data_dir}'. Run collect_data.py first."
        )

    X_list: List[np.ndarray] = []
    y_list: List[str] = []

    for fp in files:
        label = os.path.splitext(os.path.basename(fp))[0]
        try:
            data = np.genfromtxt(fp, delimiter=DATA.csv_delimiter, skip_header=1)
            if data.size == 0:
                print(f"[WARN] Empty file skipped: {fp}")
                continue
            if data.ndim == 1:
                data = data.reshape(1, -1)
            if data.shape[1] != 63:
                print(f"[WARN] Unexpected feature count in {fp}: {data.shape[1]} (expected 63). Skipping.")
                continue
            X_list.append(data.astype(np.float32))
            y_list.extend([label] * data.shape[0])
        except Exception as e:
            print(f"[WARN] Failed reading {fp}: {e}")

    if not X_list:
        raise RuntimeError("No valid training data loaded. Check your CSV files.")

    X = np.vstack(X_list)
    y = np.array(y_list, dtype=object)

    labels_sorted = sorted(list(set(y_list)))
    return X, y, labels_sorted


def per_class_accuracy(cm: np.ndarray, labels: List[str]) -> Dict[str, float]:
    acc = {}
    for i, label in enumerate(labels):
        total = cm[i].sum()
        correct = cm[i, i]
        acc[label] = float(correct / total) if total > 0 else 0.0
    return acc


def main() -> None:
    try:
        X, y, labels_sorted = load_dataset(DATA.data_dir)
    except Exception as e:
        print(f"[ERROR] {e}")
        return

    print(f"[INFO] Loaded dataset: X={X.shape}, classes={len(labels_sorted)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=MODEL.test_size,
        random_state=MODEL.random_state,
        stratify=y if len(set(y)) > 1 else None
    )

    # Pipeline = StandardScaler + RandomForest
    # Scaling helps a bit even for trees when features are normalized but not perfectly consistent.
    clf = Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("rf", RandomForestClassifier(
            n_estimators=MODEL.n_estimators,
            max_depth=MODEL.max_depth,
            min_samples_split=MODEL.min_samples_split,
            min_samples_leaf=MODEL.min_samples_leaf,
            class_weight=MODEL.class_weight,
            random_state=MODEL.random_state,
            n_jobs=-1
        ))
    ])

    print("[INFO] Training RandomForest...")
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print("\n[RESULT] Test Accuracy:", round(float(acc) * 100, 2), "%")

    print("\n[REPORT] Classification Report:")
    print(classification_report(y_test, y_pred, digits=4))

    labels_for_cm = sorted(list(set(y_test) | set(y_pred)))
    cm = confusion_matrix(y_test, y_pred, labels=labels_for_cm)

    print("\n[REPORT] Confusion Matrix (rows=true, cols=pred):")
    print("Labels:", labels_for_cm)
    print(cm)

    pca = per_class_accuracy(cm, labels_for_cm)
    print("\n[REPORT] Per-class Accuracy:")
    for k in labels_for_cm:
        print(f"- {k}: {pca[k]*100:.2f}%")

    # Save model bundle
    bundle = {
        "model": clf,
        "labels": labels_sorted,  # all seen labels
        "feature_count": 63,
    }

    try:
        with open(MODEL.model_path, "wb") as f:
            pickle.dump(bundle, f)
        print(f"\n[SAVED] Model saved to: {MODEL.model_path}")
    except Exception as e:
        print(f"[ERROR] Failed to save model: {e}")


if __name__ == "__main__":
    main()
