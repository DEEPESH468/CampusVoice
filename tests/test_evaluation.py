from src.evaluation import evaluate_predictions


def test_metrics_and_confusion_matrix():
    result = evaluate_predictions(["Positive", "Negative"], ["Positive", "Positive"])
    assert result["accuracy"] == 0.5
    assert "macro_f1" in result
    assert sum(sum(row) for row in result["confusion_matrix"]) == 2
