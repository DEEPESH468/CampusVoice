from src.preprocessing import normalize_text, tokenize


def test_normalize_and_tokenize_preserve_negation():
    assert normalize_text("  not   good ") == "not good"
    assert "not" in tokenize("This is not good!")
