"""Evidence-linked suggestion generation. Suggestions require human review."""
from __future__ import annotations

import pandas as pd

ACTION_TEXT = {
    "Laboratory and practical sessions": "Review laboratory equipment condition and maintenance needs.",
    "Syllabus and curriculum": "Review course schedules and syllabus progress with the relevant department.",
    "Infrastructure and classrooms": "Investigate the facilities mentioned and assess repair or improvement needs.",
    "Internet and technology": "Check network coverage, capacity, and technology support in the reported areas.",
    "Teaching and faculty": "Review the reported teaching experience with appropriate, contextual evidence.",
    "Examinations and assessment": "Review assessment instructions, scheduling, and grading processes.",
    "Administration and student services": "Review the reported service process and expected response times.",
    "Library and learning resources": "Check availability and access to the learning resources mentioned.",
}


def generate_insights(frame: pd.DataFrame, minimum_count: int = 2) -> list[dict[str, object]]:
    results = []
    for aspect, group in frame.explode("aspects").dropna(subset=["aspects"]).groupby("aspects"):
        if aspect == "Other or unclassified":
            continue
        negative = group[group["sentiment"] == "Negative"]
        if len(negative) < minimum_count:
            continue
        results.append({"aspect": aspect, "negative_count": len(negative), "total_count": len(group),
                        "negative_proportion": len(negative) / len(group),
                        "suggestion": ACTION_TEXT.get(aspect, "Review the reported service area."),
                        "evidence": negative["comment"].head(3).tolist()})
    return sorted(results, key=lambda x: int(x["negative_count"]), reverse=True)
