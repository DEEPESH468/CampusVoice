"""Text validation and preprocessing helpers used by CampusVoice."""
from __future__ import annotations

import re
from typing import Iterable

import nltk
from nltk.tokenize import wordpunct_tokenize


def normalize_text(text: object) -> str:
    """Normalize whitespace and Unicode-ish input without changing source evidence."""
    if text is None:
        return ""
    value = str(text).replace("\u00a0", " ")
    return re.sub(r"\s+", " ", value).strip()


def tokenize(text: str) -> list[str]:
    """Tokenize locally; wordpunct_tokenize requires no downloaded NLTK resources."""
    return [token.lower() for token in wordpunct_tokenize(normalize_text(text)) if token.strip()]


def topic_tokens(text: str, stop_words: Iterable[str] | None = None) -> list[str]:
    """Prepare topic tokens; sentiment negators are deliberately retained."""
    stops = set(stop_words or nltk.corpus.stopwords.words("english"))
    stops -= {"no", "not", "nor", "never", "n't"}
    return [t for t in tokenize(text) if re.search(r"[a-z]", t) and t not in stops]
