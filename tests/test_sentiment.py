from src.sentiment import train_classifier, predict_classifier


def test_classifier_outputs_confidence():
    texts = ["excellent helpful class", "great clear lesson", "awful confusing class", "very bad lecture", "ordinary class today", "regular lecture today"]
    labels = ["Positive", "Positive", "Negative", "Negative", "Neutral", "Neutral"]
    model = train_classifier(texts, labels)
    result = predict_classifier(model, ["helpful class"])
    assert result[0]["sentiment"] in labels
    assert 0 <= result[0]["confidence"] <= 1
