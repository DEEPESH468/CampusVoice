"""Input validation for manual comments and uploaded CSV files."""
from __future__ import annotations

from io import BytesIO, StringIO
from typing import Any

import pandas as pd

OPTIONAL_COLUMNS = ["feedback_id", "department", "semester", "date"]


def prepare_feedback(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Return clean rows and counts for rows skipped during ingestion."""
    if "comment" not in frame.columns:
        raise ValueError("CSV must include a 'comment' column.")
    data = frame.copy()
    for column in OPTIONAL_COLUMNS:
        if column not in data.columns:
            data[column] = None
    data["comment"] = data["comment"].fillna("").astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    missing = int(data["comment"].eq("").sum())
    data = data.loc[data["comment"].ne("")].copy()
    duplicate = int(data["comment"].duplicated(keep="first").sum())
    data = data.drop_duplicates(subset=["comment"], keep="first")
    data = data[["feedback_id", "comment", "department", "semester", "date"]].reset_index(drop=True)
    return data, {"input_rows": len(frame), "missing_comments": missing, "duplicate_comments": duplicate, "accepted_rows": len(data)}


def read_feedback_csv(source: Any) -> tuple[pd.DataFrame, dict[str, int]]:
    try:
        frame = pd.read_csv(source)
    except (pd.errors.ParserError, UnicodeDecodeError, ValueError) as exc:
        raise ValueError(f"Could not read CSV: {exc}") from exc
    return prepare_feedback(frame)


def manual_feedback(comment: str, department: str = "", semester: str = "") -> pd.DataFrame:
    frame = pd.DataFrame([{"comment": comment, "department": department or None, "semester": semester or None}])
    return prepare_feedback(frame)[0]
