"""CSV validation and normalization for feedback and labelled training data."""
from __future__ import annotations

from typing import Any

import pandas as pd

LABEL_ALIASES = {
    "positive": "Positive", "pos": "Positive", "+": "Positive",
    "negative": "Negative", "neg": "Negative", "-": "Negative",
    "neutral": "Neutral", "neu": "Neutral", "0": "Neutral",
}
KNOWN_METADATA = ["feedback_id", "department", "program", "semester", "academic_year", "course",
                  "category_hint", "date", "source"]


def normalize_sentiment(value: object) -> str | None:
    """Map supported label aliases to the canonical three class names."""
    if pd.isna(value) or not str(value).strip():
        return None
    label = str(value).strip().casefold()
    if label not in LABEL_ALIASES:
        raise ValueError(f"Invalid sentiment label {value!r}. Use Positive, Negative, or Neutral (pos/neg/neu are accepted).")
    return LABEL_ALIASES[label]


def prepare_feedback(
    frame: pd.DataFrame, remove_exact_duplicates: bool = False
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Clean comments and report duplicates without discarding repeated feedback by default.

    If ``remove_exact_duplicates`` is true, rows are deduplicated across every
    supplied column after comment whitespace, dates, and sentiment labels have
    been normalized. Matching comment text alone never removes a row.
    """
    if frame.empty or len(frame.columns) == 0:
        raise ValueError("The CSV contains no data rows.")
    frame = frame.rename(columns=lambda col: str(col).strip())
    if "comment" not in frame.columns:
        raise ValueError("CSV must include a 'comment' column containing feedback text.")
    data = frame.copy()
    data["comment"] = data["comment"].fillna("").astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    missing = int(data.comment.eq("").sum())
    data = data.loc[data.comment.ne("")].copy()
    duplicate_comments = int(data.comment.duplicated(keep="first").sum())
    invalid_dates = 0
    if "date" in data.columns:
        parsed = pd.to_datetime(data.date, errors="coerce", format="mixed")
        invalid_dates = int((data.date.notna() & data.date.astype(str).str.strip().ne("") & parsed.isna()).sum())
        data["date"] = parsed
    invalid_labels = 0
    if "sentiment" in data.columns:
        normalized: list[str | None] = []
        for label in data.sentiment:
            try:
                normalized.append(normalize_sentiment(label))
            except ValueError:
                invalid_labels += 1
                normalized.append(None)
        if invalid_labels:
            examples = data.loc[[x is None for x in normalized], "sentiment"].dropna().astype(str).unique()[:3]
            raise ValueError("Invalid sentiment labels: " + ", ".join(repr(x) for x in examples) + ". Allowed: Positive, Negative, Neutral (or pos/neg/neu).")
        data["sentiment"] = normalized
    # Comment repeats can be separate student reports. Only remove rows that
    # match across every supplied field, and only when the user opts in.
    exact_duplicate_rows = int(data.duplicated(keep="first").sum())
    duplicate_feedback_ids = 0
    if "feedback_id" in data.columns:
        ids = data.feedback_id.fillna("").astype(str).str.strip()
        duplicate_feedback_ids = int((ids.ne("") & ids.duplicated(keep="first")).sum())
    if remove_exact_duplicates:
        data = data.drop_duplicates(keep="first").copy()
    for column in KNOWN_METADATA:
        if column not in data.columns:
            data[column] = None
    # Keep the familiar columns first while retaining future/custom metadata too.
    ordered = ["feedback_id", "comment", *KNOWN_METADATA[1:], "sentiment"]
    columns = [c for c in ordered if c in data.columns] + [c for c in data.columns if c not in ordered]
    data = data[columns].reset_index(drop=True)
    removed_exact = exact_duplicate_rows if remove_exact_duplicates else 0
    return data, {"input_rows": len(frame), "missing_comments": missing,
                  "duplicate_comments": duplicate_comments, "exact_duplicate_rows": exact_duplicate_rows,
                  "exact_duplicates_removed": removed_exact, "duplicate_feedback_ids": duplicate_feedback_ids,
                  "invalid_dates": invalid_dates, "invalid_labels": invalid_labels, "accepted_rows": len(data),
                  "skipped_rows": missing + removed_exact}


def read_feedback_csv(source: Any, remove_exact_duplicates: bool = False) -> tuple[pd.DataFrame, dict[str, int]]:
    """Read and validate a feedback CSV, optionally removing full-row duplicates.

    The Boolean option is accepted positionally or by keyword for compatibility
    with callers that pass an ingestion setting through directly.
    """
    try:
        frame = pd.read_csv(source)
    except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError, ValueError) as exc:
        raise ValueError(f"Could not read CSV: {exc}") from exc
    return prepare_feedback(frame, remove_exact_duplicates=remove_exact_duplicates)


def manual_feedback(comment: str, department: str = "", semester: str = "") -> pd.DataFrame:
    frame = pd.DataFrame([{"comment": comment, "department": department or None, "semester": semester or None}])
    return prepare_feedback(frame)[0]
