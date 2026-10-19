import pandas as pd
import pytest

from src.data_loader import prepare_feedback


def test_prepare_feedback_skips_blank_and_duplicate():
    clean, stats = prepare_feedback(pd.DataFrame({"comment": [" okay ", None, "okay"]}))
    assert clean.comment.tolist() == ["okay"]
    assert stats["missing_comments"] == 1
    assert stats["duplicate_comments"] == 1


def test_comment_required():
    with pytest.raises(ValueError, match="comment"):
        prepare_feedback(pd.DataFrame({"text": ["hi"]}))
