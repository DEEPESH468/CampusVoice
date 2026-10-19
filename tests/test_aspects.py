from src.aspects import detect_aspects


def test_multi_aspect_and_unknown():
    result = detect_aspects("Teachers explain well but the laboratory computers are slow.")
    assert {row["aspect"] for row in result} == {"Teaching and faculty", "Laboratory and practical sessions"}
    assert detect_aspects("The weather is pleasant.")[0]["aspect"] == "Other or unclassified"
