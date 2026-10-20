from io import StringIO

import pandas as pd
import pytest

from src.data_loader import normalize_sentiment, prepare_feedback, read_feedback_csv


def test_prepare_feedback_skips_blank_but_preserves_repeated_comments():
    clean, stats = prepare_feedback(pd.DataFrame({"comment": [" okay ", None, "okay"]}))
    assert clean.comment.tolist() == ["okay", "okay"]
    assert stats["missing_comments"] == 1
    assert stats["duplicate_comments"] == 1
    assert stats["exact_duplicate_rows"] == 1
    assert stats["exact_duplicates_removed"] == 0


def test_optional_exact_duplicate_removal_and_feedback_id_report():
    source = pd.DataFrame({"feedback_id": ["id1", "id1", "id2"],
                           "comment": ["same", "same", "same"],
                           "department": ["CS", "CS", "Arts"]})
    preserved, stats = prepare_feedback(source)
    assert len(preserved) == 3
    assert stats["duplicate_comments"] == 2
    assert stats["duplicate_feedback_ids"] == 1
    removed, dedupe_stats = prepare_feedback(source, remove_exact_duplicates=True)
    assert len(removed) == 2
    assert dedupe_stats["exact_duplicates_removed"] == 1
    assert dedupe_stats["skipped_rows"] == 1


def test_read_feedback_csv_supports_both_duplicate_settings():
    csv_text = "feedback_id,comment,department\nf1,Same comment,CS\nf1,Same comment,CS\nf2,Same comment,Arts\n"
    kept, kept_stats = read_feedback_csv(StringIO(csv_text), remove_exact_duplicates=False)
    removed, removed_stats = read_feedback_csv(StringIO(csv_text), True)
    assert len(kept) == 3
    assert kept_stats["duplicate_comments"] == 2
    assert kept_stats["exact_duplicates_removed"] == 0
    assert len(removed) == 2
    assert removed_stats["exact_duplicate_rows"] == 1
    assert removed_stats["exact_duplicates_removed"] == 1


def test_demo_csv_supports_remove_exact_duplicates_argument():
    from pathlib import Path

    demo_path = Path(__file__).parents[1] / "data" / "demo_feedback.csv"
    default_rows, _ = read_feedback_csv(demo_path, remove_exact_duplicates=False)
    deduped_rows, stats = read_feedback_csv(demo_path, remove_exact_duplicates=True)
    assert len(default_rows) >= len(deduped_rows)
    assert stats["accepted_rows"] == len(deduped_rows)


def test_comment_required():
    with pytest.raises(ValueError, match="comment"):
        prepare_feedback(pd.DataFrame({"text": ["hi"]}))


def test_rich_schema_and_metadata_without_labels():
    columns = ["feedback_id", "comment", "department", "program", "semester", "academic_year",
               "course", "category_hint", "date", "source"]
    frame = pd.DataFrame([["F1", "  clear teaching ", "Science", "BSc", 2, "2025-26",
                           "NLP", "positive experience", "2025-03-01", "survey"]], columns=columns)
    clean, stats = prepare_feedback(frame)
    assert set(columns).issubset(clean.columns)
    assert "sentiment" not in clean
    assert clean.loc[0, "category_hint"] == "positive experience"
    assert stats["accepted_rows"] == 1


def test_label_aliases_and_invalid_date_are_reported():
    clean, stats = prepare_feedback(pd.DataFrame({"comment": ["good", "fine"], "sentiment": [" POS ", "neu"], "date": ["not-a-date", "2025-01-01"]}))
    assert clean.sentiment.tolist() == ["Positive", "Neutral"]
    assert stats["invalid_dates"] == 1
    assert normalize_sentiment("NEG") == "Negative"


def test_labelled_and_unlabelled_csv_loading_preserve_metadata_and_missing_values():
    labelled, _ = read_feedback_csv(StringIO(
        "feedback_id,comment,department,program,semester,course,category_hint,date,source,sentiment\n"
        "f1,Helpful class,CS,BSc,2,NLP,teaching,2025-01-01,survey,pos\n"
        "f2,,Arts,BA,1,Writing,resources,invalid,form,\n"))
    assert labelled.sentiment.tolist() == ["Positive"]
    assert labelled.loc[0, "category_hint"] == "teaching"
    assert "source" in labelled.columns and "course" in labelled.columns
    unlabelled, _ = read_feedback_csv(StringIO("comment,category_hint\nA comment,negative tag\n"))
    assert "sentiment" not in unlabelled.columns
    assert unlabelled.loc[0, "category_hint"] == "negative tag"


def test_invalid_sentiment_label_is_rejected():
    with pytest.raises(ValueError, match="Invalid sentiment"):
        prepare_feedback(pd.DataFrame({"comment": ["text"], "sentiment": ["happy"]}))


def test_empty_csv_frame_is_rejected():
    with pytest.raises(ValueError, match="no data rows"):
        prepare_feedback(pd.DataFrame(columns=["comment"]))
