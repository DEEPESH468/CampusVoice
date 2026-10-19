"""Leakage-safe sentiment evaluation utilities."""
from __future__ import annotations

from typing import Sequence

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split

from src.sentiment import train_classifier, vader_score


def evaluate_predictions(y_true: Sequence[str], y_pred: Sequence[str]) -> dict[str, object]:
    labels = sorted(set(y_true) | set(y_pred))
    p, r, f, support = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    return {"accuracy": float(accuracy_score(y_true, y_pred)),
            "macro_precision": float(np.mean(p)), "macro_recall": float(np.mean(r)), "macro_f1": float(np.mean(f)),
            "per_class": {label: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(support[i])} for i, label in enumerate(labels)},
            "labels": labels, "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist()}


def compare_models(texts: Sequence[str], labels: Sequence[str], test_size: float = 0.25, random_state: int = 42) -> dict[str, object]:
    counts = {label: list(labels).count(label) for label in set(labels)}
    if len(texts) < 8 or len(counts) < 2 or min(counts.values()) < 2:
        raise ValueError("At least 8 labelled comments and 2 examples per class are required for a safe split.")
    x_train, x_test, y_train, y_test = train_test_split(list(texts), list(labels), test_size=test_size,
        random_state=random_state, stratify=list(labels))
    model = train_classifier(x_train, y_train, random_state)
    ml_pred = model.predict(x_test).tolist()
    vader_pred = [vader_score(text)["sentiment"] for text in x_test]
    return {"train_size": len(x_train), "test_size": len(x_test), "y_test": y_test,
            "texts_test": x_test, "logistic_regression": evaluate_predictions(y_test, ml_pred),
            "vader": evaluate_predictions(y_test, vader_pred), "model": model, "synthetic": False}
