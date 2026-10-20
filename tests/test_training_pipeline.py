import pandas as pd
import pytest

from src.data_loader import prepare_feedback
from src.evaluation import compare_models
from src.sentiment import predict_classifier, train_classifier


def labelled_examples():
    texts, labels = [], []
    examples = {
        "Positive": ["excellent helpful teaching", "great clear lecture", "supportive useful class", "engaging instructor"],
        "Negative": ["awful confusing lecture", "poor broken laboratory", "slow unhelpful service", "outdated course materials"],
        "Neutral": ["class met today", "lecture followed schedule", "course has weekly sessions", "assignment was submitted"],
    }
    for label, comments in examples.items():
        for comment in comments:
            texts.append(f"{comment} zxqmarker{len(texts)}")
            labels.append(label)
    return texts, labels


def test_training_and_prediction_on_rich_unlabelled_feedback():
    texts, labels = labelled_examples()
    model = train_classifier(texts, labels)
    rich, _ = prepare_feedback(pd.DataFrame({
        "feedback_id": ["R1", "R2"], "comment": ["excellent helpful class", "poor confusing lecture"],
        "department": ["Science", "Arts"], "category_hint": ["teaching", "course"],
    }))
    predictions = predict_classifier(model, rich.comment.tolist())
    assert [p["sentiment"] for p in predictions] == ["Positive", "Negative"]
    assert rich.category_hint.tolist() == ["teaching", "course"]


def test_one_class_cannot_be_held_out_evaluated():
    with pytest.raises(ValueError, match="2 sentiment classes"):
        compare_models([f"comment {i}" for i in range(9)], ["Positive"] * 9)


def test_holdout_evaluation_fits_vectorizer_without_test_comments():
    texts, labels = labelled_examples()
    result = compare_models(texts, labels)
    assert result["train_size"] + result["test_size"] == len(texts)
    assert sum(result["test_class_counts"].values()) == result["test_size"]
    vocab = result["model"].named_steps["tfidf"].vocabulary_
    # Each row has a unique token; at least one test-only marker must be absent from train-fitted vocabulary.
    test_markers = {word for text in result["texts_test"] for word in text.split() if word.startswith("zxqmarker")}
    assert test_markers.isdisjoint(vocab)
    assert "macro_f1" in result["logistic_regression"]
    assert "classification_report" in result
