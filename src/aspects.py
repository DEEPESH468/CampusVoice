"""Explainable phrase rules for multi-label college service detection."""
from __future__ import annotations

import re

from src.sentiment import vader_score

ASPECT_KEYWORDS: dict[str, list[str]] = {
    "Teaching and faculty": ["teacher", "teachers", "faculty", "professor", "lecturer", "teaching", "explain", "class delivery"],
    "Laboratory and practical sessions": ["lab", "laboratory", "practical", "experiment", "equipment", "computer systems"],
    "Syllabus and curriculum": ["syllabus", "curriculum", "course content", "coursework", "syllabus completion"],
    "Examinations and assessment": ["exam", "examination", "test", "assessment", "grading", "marks", "paper"],
    "Infrastructure and classrooms": ["classroom", "building", "infrastructure", "desk", "seat", "projector", "washroom", "toilet", "canteen"],
    "Internet and technology": ["internet", "wifi", "wi-fi", "network", "online portal", "website", "technology"],
    "Administration and student services": ["administration", "administrative", "office", "registration", "student services", "scholarship", "timetable", "schedule"],
    "Library and learning resources": ["library", "books", "journal", "study material", "learning resources", "reading room"],
}


def detect_aspects(comment: str, mappings: dict[str, list[str]] | None = None) -> list[dict[str, str]]:
    """Find all mapped categories and the matching evidence phrase."""
    text = comment.lower()
    matches: list[dict[str, str]] = []
    for category, keywords in (mappings or ASPECT_KEYWORDS).items():
        found = next((kw for kw in keywords if re.search(r"(?<!\w)" + re.escape(kw.lower()) + r"(?!\w)", text)), None)
        if found:
            # Estimate polarity from the local clause containing the matched phrase.
            # This is a lightweight heuristic, not a trained ABSA model.
            clauses = re.split(r"\b(?:but|however|although|whereas)\b|[.;!?]", comment, flags=re.IGNORECASE)
            clause = next((part.strip() for part in clauses if found.lower() in part.lower()), comment)
            matches.append({"aspect": category, "evidence": clause or found,
                            "sentiment": str(vader_score(clause)["sentiment"])})
    if not matches:
        return [{"aspect": "Other or unclassified", "evidence": "No configured aspect phrase matched", "sentiment": "Unclassified"}]
    return matches
