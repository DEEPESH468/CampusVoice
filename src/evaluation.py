"""Leakage-safe sentiment evaluation utilities."""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

from src.sentiment import train_classifier, vader_score


def evaluate_predictions(y_true: Sequence[str], y_pred: Sequence[str]) -> dict[str, object]:
    labels = sorted(set(y_true) | set(y_pred))
    p, r, f, support = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    return {"accuracy": float(accuracy_score(y_true, y_pred)),
            "macro_precision": float(np.mean(p)), "macro_recall": float(np.mean(r)), "macro_f1": float(np.mean(f)),
            "per_class": {label: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(support[i])} for i, label in enumerate(labels)},
            "labels": labels, "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist()}


def compare_models(
    texts: Sequence[str], labels: Sequence[str], test_size: float = 0.2,
    random_state: int = 42, groups: Sequence[str] | None = None,
) -> dict[str, object]:
    """Evaluate on a reproducible stratified holdout.

    Exact comments (case and whitespace normalized) are deduplicated before
    splitting. Conflicting labels or topic groups for an identical comment are
    rejected because they make a leakage-safe, well-defined evaluation impossible.
    When groups are supplied, whole groups are held out together.
    """
    if len(texts) != len(labels):
        raise ValueError("Comments and sentiment labels must have the same number of rows.")
    if groups is not None and len(groups) != len(texts):
        raise ValueError("Comments and topic groups must have the same number of rows.")

    normalized_groups: list[str] | None = None
    if groups is not None:
        normalized_groups = []
        for group in groups:
            try:
                is_missing = bool(pd.isna(group))
            except (TypeError, ValueError):
                is_missing = False
            value = "" if is_missing else str(group).strip()
            if not value:
                raise ValueError("Topic groups must be present and non-empty for every labelled comment.")
            normalized_groups.append(value)

    clean_texts: list[str] = []
    clean_labels: list[str] = []
    clean_groups: list[str] = []
    seen: dict[str, int] = {}
    duplicates_removed = 0
    for i, (text, label) in enumerate(zip(texts, labels)):
        text = str(text).strip()
        label = str(label).strip()
        key = " ".join(text.casefold().split())
        if not key:
            continue
        if key in seen:
            prior = seen[key]
            if clean_labels[prior] != label:
                raise ValueError("An exact duplicate comment has conflicting sentiment labels; resolve it before evaluation.")
            if normalized_groups is not None and clean_groups[prior] != normalized_groups[i]:
                raise ValueError("An exact duplicate comment has conflicting topic groups; resolve it before evaluation.")
            duplicates_removed += 1
            continue
        seen[key] = len(clean_texts)
        clean_texts.append(text)
        clean_labels.append(label)
        if normalized_groups is not None:
            clean_groups.append(normalized_groups[i])

    texts, labels = clean_texts, clean_labels
    counts = {label: list(labels).count(label) for label in set(labels)}
    if len(texts) != len(labels):
        raise ValueError("Comments and sentiment labels must have the same number of rows.")
    if len(texts) < 8 or len(counts) < 2 or min(counts.values()) < 2:
        raise ValueError("At least 8 labelled comments, 2 sentiment classes, and 2 examples per class are needed for a stratified holdout.")
    n_classes = len(counts)
    n_test = int(np.ceil(len(texts) * test_size))
    if n_test < n_classes or len(texts) - n_test < n_classes:
        raise ValueError("Not enough rows for every class to appear in both training and test splits. Add labelled examples.")
    if groups is None:
        train_idx, test_idx = train_test_split(
            np.arange(len(texts)), test_size=test_size, random_state=random_state, stratify=list(labels))
        split_strategy = "stratified random split"
    else:
        # Five folds approximate an 80/20 split while ensuring a topic group
        # can never occur in both partitions. Pick the valid fold closest to target.
        group_count = len(set(clean_groups))
        if group_count < 2:
            raise ValueError("A group-aware holdout requires at least two distinct topic groups.")
        splitter = StratifiedGroupKFold(n_splits=min(5, group_count), shuffle=True, random_state=random_state)
        candidates = []
        y_array = np.asarray(labels)
        g_array = np.asarray(clean_groups)
        all_classes = set(counts)
        try:
            splits = splitter.split(np.zeros(len(texts)), y_array, g_array)
            for candidate_train, candidate_test in splits:
                if set(y_array[candidate_train]) == all_classes and set(y_array[candidate_test]) == all_classes:
                    candidates.append((candidate_train, candidate_test))
        except ValueError as exc:
            raise ValueError(f"Could not create a valid group-aware holdout: {exc}") from exc
        if not candidates:
            raise ValueError("Could not create a topic-group holdout containing every class in both partitions. Add more labelled examples across topic groups.")
        train_idx, test_idx = min(candidates, key=lambda pair: abs(len(pair[1]) / len(texts) - test_size))
        split_strategy = "stratified topic-group holdout"
    x_train = [texts[int(i)] for i in train_idx]
    x_test = [texts[int(i)] for i in test_idx]
    y_train = [labels[int(i)] for i in train_idx]
    y_test = [labels[int(i)] for i in test_idx]
    model = train_classifier(x_train, y_train, random_state)
    ml_pred = model.predict(x_test).tolist()
    vader_pred = [vader_score(text)["sentiment"] for text in x_test]
    return {"train_size": len(x_train), "test_size": len(x_test), "class_counts": counts,
            "duplicates_removed": duplicates_removed, "split_strategy": split_strategy,
            "train_groups": sorted(set(clean_groups[int(i)] for i in train_idx)) if groups is not None else None,
            "test_groups": sorted(set(clean_groups[int(i)] for i in test_idx)) if groups is not None else None,
            "train_class_counts": {label: y_train.count(label) for label in sorted(counts)},
            "test_class_counts": {label: y_test.count(label) for label in sorted(counts)}, "y_test": y_test,
            "texts_test": x_test, "logistic_regression": evaluate_predictions(y_test, ml_pred),
            "vader": evaluate_predictions(y_test, vader_pred),
            "classification_report": classification_report(y_test, ml_pred, labels=sorted(counts),
                target_names=sorted(counts), output_dict=True, zero_division=0),
            "model": model, "synthetic": False}
