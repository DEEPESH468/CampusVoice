"""NMF topic discovery over TF-IDF features with representative evidence."""
from __future__ import annotations

from typing import Sequence

import numpy as np
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer


def discover_topics(comments: Sequence[str], sentiments: Sequence[str] | None = None, max_topics: int = 5) -> list[dict[str, object]]:
    if len(comments) < 4:
        return []
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1, max_features=5000)
    try:
        matrix = vectorizer.fit_transform(comments)
    except ValueError:
        return []
    if matrix.shape[1] < 2:
        return []
    n_topics = min(max_topics, max(2, len(comments) // 5), matrix.shape[1])
    model = NMF(n_components=n_topics, init="nndsvda", random_state=42, max_iter=400)
    weights = model.fit_transform(matrix)
    terms = np.asarray(vectorizer.get_feature_names_out())
    assignment = weights.argmax(axis=1)
    output = []
    for topic_id in range(n_topics):
        indices = np.flatnonzero(assignment == topic_id)
        if len(indices) == 0:
            continue
        top_indices = model.components_[topic_id].argsort()[::-1][:5]
        top_words = terms[top_indices].tolist()
        indices = sorted(indices, key=lambda i: weights[i, topic_id], reverse=True)
        neg = sum(1 for i in indices if sentiments and sentiments[i] == "Negative")
        output.append({"topic_id": topic_id, "name": " / ".join(top_words[:3]), "terms": top_words,
                       "count": len(indices), "negative_count": neg,
                       "negative_proportion": neg / len(indices),
                       "representatives": [comments[i] for i in indices[:3]],
                       "term_weights": [float(model.components_[topic_id, i]) for i in top_indices],
                       "member_indices": [int(i) for i in indices]})
    return sorted(output, key=lambda t: int(t["count"]), reverse=True)
