"""Parse and normalize arXiv identifiers.

Retrieval stores the versionless id. Extraction keeps the version so citations
point at the exact source that was parsed.
"""

from __future__ import annotations

import re

_VERSION_SUFFIX = re.compile(r"v(\d+)$", re.IGNORECASE)


def split_arxiv_id(url_or_id: str) -> tuple[str, str | None]:
    """Return (versionless_id, version) where version is like 'v2' or None."""
    text = _strip_url(url_or_id)
    match = _VERSION_SUFFIX.search(text)
    if not match:
        return text, None
    return text[: match.start()], f"v{match.group(1)}"


def normalize_arxiv_id(url_or_id: str) -> str:
    """Versionless arXiv id (2205.09329v2 → 2205.09329)."""
    core, _ = split_arxiv_id(url_or_id)
    return core


def format_version(version: str | None, default: str = "v1") -> str:
    if not version:
        return default
    text = str(version).strip()
    if not text:
        return default
    if text.isdigit():
        return f"v{text}"
    if text.lower().startswith("v") and text[1:].isdigit():
        return f"v{text[1:]}"
    return text


def paper_key(arxiv_id: str, version: str | None = None) -> str:
    """Canonical citation key: versionless id plus a vN suffix."""
    core, parsed = split_arxiv_id(arxiv_id)
    resolved = format_version(version or parsed)
    return f"{core}{resolved}"


def version_number(version: str | None) -> int:
    formatted = format_version(version)
    if formatted.lower().startswith("v") and formatted[1:].isdigit():
        return int(formatted[1:])
    return 0


def _strip_url(url_or_id: str) -> str:
    text = url_or_id.strip()
    if "arxiv.org/abs/" in text:
        text = text.rsplit("/abs/", 1)[-1]
    elif "arxiv.org/pdf/" in text:
        text = text.rsplit("/pdf/", 1)[-1]
        if text.endswith(".pdf"):
            text = text[: -len(".pdf")]
    elif "arxiv.org/e-print/" in text:
        text = text.rsplit("/e-print/", 1)[-1]
    text = text.replace("http://", "").replace("https://", "")
    if text.startswith("arxiv.org/"):
        text = text.split("/", 1)[-1]
    return text.strip()
