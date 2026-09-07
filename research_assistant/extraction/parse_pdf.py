"""Parse PDF papers into a PaperDocument."""

from __future__ import annotations

import logging
import re

from research_assistant.extraction.fetch import FetchedSource
from research_assistant.extraction.normalize import normalize_text
from research_assistant.extraction.types import PaperDocument, Section

logger = logging.getLogger(__name__)

_HEADING = re.compile(
    r"^(?:(?:\d+\.)+\d*|\d+)\s+[A-Z][\w\-].{1,80}$"
    r"|^(?:Abstract|Introduction|Related Work|Background|Method|Methods|"
    r"Approach|Experiments?|Results?|Evaluation|Discussion|Conclusion|"
    r"Conclusions|Limitations|References|Bibliography|Appendix)\b.*$",
    re.IGNORECASE,
)


class PdfParseError(RuntimeError):
    pass


def parse_pdf_source(source: FetchedSource) -> PaperDocument:
    try:
        import pymupdf
    except ImportError as exc:  # pragma: no cover
        raise PdfParseError("pymupdf is required for PDF fallback") from exc

    try:
        doc = pymupdf.open(source.path)
    except Exception as exc:
        raise PdfParseError(f"Failed to open PDF {source.path}: {exc}") from exc

    try:
        toc = doc.get_toc(simple=True) or []
        pages = []
        for page in doc:
            raw = page.get_text("text") or ""
            raw = re.sub(r"(\w)-\n(\w)", r"\1\2", raw)
            pages.append(raw)
        title = ""
        if doc.metadata:
            title = str(doc.metadata.get("title") or "")
    finally:
        doc.close()

    full = "\n".join(pages)
    if not normalize_text(full):
        raise PdfParseError(f"Empty PDF text for {source.arxiv_id}")

    if toc:
        sections = _sections_from_toc(toc, pages)
    else:
        sections = _sections_from_headings(full)

    abstract = ""
    for section in sections:
        if section.heading.lower().strip() == "abstract":
            abstract = section.text
            break
    if not abstract:
        abstract = _guess_abstract(full)
    if not title:
        title = _guess_title(pages[0] if pages else full)

    return PaperDocument(
        arxiv_id=source.arxiv_id,
        version=source.version,
        source_kind="pdf",
        title=normalize_text(title),
        abstract=normalize_text(abstract),
        sections=sections,
    )


def _sections_from_toc(toc: list, pages: list[str]) -> list[Section]:
    # toc entries: [level, title, page] with 1-based pages.
    if not toc:
        return []
    page_starts: list[int] = []
    cursor = 0
    joined_pages = []
    for text in pages:
        page_starts.append(cursor)
        joined_pages.append(text)
        cursor += len(text) + 1
    blob = "\n".join(joined_pages)
    sections: list[Section] = []
    for i, entry in enumerate(toc):
        level, heading, page = int(entry[0]), str(entry[1]), int(entry[2])
        start_page = max(page - 1, 0)
        if start_page >= len(page_starts):
            continue
        start = page_starts[start_page]
        if i + 1 < len(toc):
            next_page = max(int(toc[i + 1][2]) - 1, 0)
            end = page_starts[next_page] if next_page < len(page_starts) else len(blob)
        else:
            end = len(blob)
        body = normalize_text(blob[start:end])
        # Drop the heading if it is repeated at the start of the body.
        heading_n = normalize_text(heading)
        if body.lower().startswith(heading_n.lower()):
            body = body[len(heading_n) :].strip()
        if body:
            sections.append(Section(heading=heading_n or "Section", level=level, text=body))
    return sections or [Section(heading="Body", level=1, text=normalize_text(blob))]


def _sections_from_headings(full: str) -> list[Section]:
    lines = full.splitlines()
    marks: list[tuple[int, str]] = []
    for i, line in enumerate(lines):
        stripped = line.strip()
        if _HEADING.match(stripped) and len(stripped) < 90:
            marks.append((i, stripped))
    if not marks:
        return [Section(heading="Body", level=1, text=normalize_text(full))]
    sections: list[Section] = []
    if marks[0][0] > 0:
        preamble = normalize_text("\n".join(lines[: marks[0][0]]))
        if preamble:
            sections.append(Section(heading="Preamble", level=1, text=preamble))
    for i, (line_idx, heading) in enumerate(marks):
        end = marks[i + 1][0] if i + 1 < len(marks) else len(lines)
        body = normalize_text("\n".join(lines[line_idx + 1 : end]))
        if body:
            sections.append(Section(heading=normalize_text(heading), level=1, text=body))
    return sections


def _guess_abstract(full: str) -> str:
    match = re.search(
        r"(?:^|\n)\s*abstract\s*(?:\n|:)\s*(.+?)(?:\n\s*(?:1\s+)?introduction\b|\n\s*1\.)",
        full,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if match:
        return normalize_text(match.group(1))
    return ""


def _guess_title(first_page: str) -> str:
    for line in first_page.splitlines():
        stripped = line.strip()
        if 12 <= len(stripped) <= 180 and not stripped.lower().startswith("arxiv"):
            return stripped
    return ""
