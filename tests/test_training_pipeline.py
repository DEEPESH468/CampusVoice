import pandas as pd
import pytest
from pathlib import Path

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


def test_synthetic_demo_is_balanced_and_group_holdout_prevents_topic_leakage():
    demo = pd.read_csv(Path(__file__).parents[1] / "data" / "demo_labeled_training.csv")
    assert len(demo) >= 300
    assert demo.comment.nunique() == len(demo)
    assert demo.sentiment.value_counts().to_dict() == {"Positive": 100, "Negative": 100, "Neutral": 100}
    assert demo.topic_group.nunique() >= 5

    result = compare_models(demo.comment.tolist(), demo.sentiment.tolist(), groups=demo.topic_group.tolist())
    assert result["split_strategy"] == "stratified topic-group holdout"
    assert 0.15 <= result["test_size"] / len(demo) <= 0.25
    assert set(result["train_groups"]).isdisjoint(result["test_groups"])
    assert set(result["test_class_counts"]) == {"Positive", "Negative", "Neutral"}
    assert set(result["logistic_regression"]["per_class"]) == {"Positive", "Negative", "Neutral"}
    assert sum(x["support"] for x in result["logistic_regression"]["per_class"].values()) == result["test_size"]


def test_exact_duplicate_comments_are_removed_before_holdout():
    texts, labels = [], []
    for sentiment, stem in [("Positive", "helpful faculty"), ("Negative", "poor wifi"), ("Neutral", "class schedule")]:
        for i in range(6):
            texts.append(f"{stem} detail {i}")
            labels.append(sentiment)
    texts.append("  HELPFUL faculty   DETAIL 0 ")
    labels.append("Positive")
    result = compare_models(texts, labels)
    assert result["duplicates_removed"] == 1
    assert result["train_size"] + result["test_size"] == 18


def test_exact_duplicate_with_conflicting_sentiment_is_rejected():
    with pytest.raises(ValueError, match="conflicting sentiment"):
        compare_models(["Helpful faculty", " helpful   Faculty "] + [f"bad wifi {i}" for i in range(8)],
                       ["Positive", "Negative"] + ["Negative"] * 8)
