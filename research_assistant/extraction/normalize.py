"""Whitespace and PDF-hyphen normalization used by parsers and the evidence gate."""

from __future__ import annotations

import re

_HYPHEN_LINE = re.compile(r"(\w)-\s+(\w)")
_WS = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Join hyphenated line-breaks and collapse whitespace."""
    text = text.replace("\u00ad", "")
    text = _HYPHEN_LINE.sub(r"\1\2", text)
    return _WS.sub(" ", text).strip()


def normalize_heading(heading: str) -> str:
    text = heading.lower()
    text = re.sub(r"^(appendix|section|chapter)\s+", "", text)
    text = re.sub(r"^[\d.]+\s*", "", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return " ".join(text.split())


def headings_match(declared: str, actual: str) -> bool:
    left = normalize_heading(declared)
    right = normalize_heading(actual)
    if not left or not right:
        return False
    return left == right or left in right or right in left
