"""VADER baseline and TF-IDF + Logistic Regression sentiment models."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import nltk
import numpy as np
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

LABELS = ("Negative", "Neutral", "Positive")
VADER_THRESHOLD = 0.05
_VADER: SentimentIntensityAnalyzer | None = None
_VADER_UNAVAILABLE = False
_FALLBACK_POSITIVE = {"good", "great", "excellent", "helpful", "clear", "useful", "supportive", "fast", "reliable", "engaging", "comfortable", "fair", "easy", "well"}
_FALLBACK_NEGATIVE = {"bad", "awful", "poor", "broken", "slow", "confusing", "incomplete", "outdated", "delayed", "fail", "fails", "rushed", "ignores", "damaged"}


def _vader_analyzer() -> SentimentIntensityAnalyzer:
    global _VADER, _VADER_UNAVAILABLE
    if _VADER is None:
        if _VADER_UNAVAILABLE:
            raise LookupError("VADER lexicon is not installed")
        try:
            _VADER = SentimentIntensityAnalyzer()
        except LookupError:
            # Keep offline installations usable. The caller switches to a small,
            # explicitly identified lexicon fallback rather than claiming VADER.
            _VADER_UNAVAILABLE = True
            raise
    return _VADER


def vader_score(text: str) -> dict[str, float | str]:
    try:
        scores = _vader_analyzer().polarity_scores(text)
        engine = "VADER"
        confidence = None
        compound = float(scores["compound"])
    except LookupError:
        tokens = [token.lower().strip(".,!?;:'\"()[]{}") for token in text.split()]
        pos = neg = 0
        for index, token in enumerate(tokens):
            negated = any(previous in {"not", "never", "no", "n't"} for previous in tokens[max(0, index - 3):index])
            if token in _FALLBACK_POSITIVE:
                neg += int(negated)
                pos += int(not negated)
            elif token in _FALLBACK_NEGATIVE:
                pos += int(negated)
                neg += int(not negated)
        raw = pos - neg
        compound = raw / (abs(raw) + 4.0) if raw else 0.0
        scores = {"neg": float(neg > pos), "neu": float(pos == neg), "pos": float(pos > neg), "compound": compound}
        engine = "built-in fallback (VADER lexicon unavailable)"
        confidence = None
    label = "Positive" if compound >= VADER_THRESHOLD else "Negative" if compound <= -VADER_THRESHOLD else "Neutral"
    return {"sentiment": label, "confidence": confidence, "engine": engine, "compound": compound, **scores}


def train_classifier(texts: Sequence[str], labels: Sequence[str], random_state: int = 42) -> Pipeline:
    if len(texts) != len(labels) or len(set(labels)) < 2:
        raise ValueError("Training requires matching comments and at least two sentiment classes.")
    model = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=20000, sublinear_tf=True)),
        ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=random_state)),
    ])
    model.fit(texts, labels)
    return model


def predict_classifier(model: Pipeline, texts: Sequence[str]) -> list[dict[str, object]]:
    if not texts:
        return []
    probabilities = model.predict_proba(texts)
    labels = model.predict(texts)
    return [{"sentiment": str(label), "confidence": float(np.max(prob)), "probabilities": {str(c): float(p) for c, p in zip(model.classes_, prob)}} for label, prob in zip(labels, probabilities)]
