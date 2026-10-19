"""Downloadable Markdown summary generation."""
from __future__ import annotations

from datetime import date

import pandas as pd


def markdown_report(frame: pd.DataFrame, topics: list[dict[str, object]], insights: list[dict[str, object]], methodology: str) -> str:
    counts = frame["sentiment"].value_counts().to_dict() if not frame.empty else {}
    lines = ["# CampusVoice Analysis Report", "", f"Analysis date: {date.today().isoformat()}",
             f"Dataset size: {len(frame)} comments", "", "## Methodology", methodology, "",
             "## Sentiment counts"]
    lines.extend([f"- {label}: {counts.get(label, 0)}" for label in ["Positive", "Neutral", "Negative"]])
    lines += ["", "## Topics"]
    lines += [f"- {t['name']}: {t['count']} comments; terms: {', '.join(t['terms'])}" for t in topics] or ["No reliable topics were produced."]
    lines += ["", "## Suggestions for human review"]
    for item in insights:
        lines += [f"- **{item['aspect']}**: {item['suggestion']} ({item['negative_count']} negative / {item['total_count']} mentions)"]
        lines += [f"  - Evidence: {quote}" for quote in item["evidence"]]
    lines += ["", "## Limitations", "Automated labels, keyword aspects, and NMF topics are approximate. Synthetic training data is not validation on real student feedback. Suggestions need contextual human review. Data is processed locally by this application."]
    return "\n".join(lines)
