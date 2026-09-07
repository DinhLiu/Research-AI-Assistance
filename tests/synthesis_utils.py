"""Shared builders for Stage 3 tests."""

from __future__ import annotations

from research_assistant.extraction.types import Claim, Evidence, ExtractedPaper, ExtractionResult, Mention


def evidence(quote: str = "quote", section: str = "Method") -> Evidence:
    return Evidence(
        quote=quote,
        section=section,
        source_kind="tex",
        start_char=0,
        end_char=min(len(quote), 8),
    )


def claim(text: str, quote: str | None = None) -> Claim:
    return Claim(text=text, evidence=[evidence(quote or text[:40])])


def mention(value: str, quote: str | None = None) -> Mention:
    return Mention(value=value, evidence=evidence(quote or value))


def paper(
    *,
    arxiv_id: str = "2205.09329",
    version: str = "v1",
    status: str = "ok",
    title: str = "A Paper",
    method: str | None = "Uses EL2N scores to prune the training set.",
    keywords: list[str] | None = None,
    datasets: list[str] | None = None,
    metrics: list[str] | None = None,
    limitations: list[str] | None = None,
    results: list[str] | None = None,
    problem: str | None = None,
    **kwargs,
) -> ExtractedPaper:
    return ExtractedPaper(
        arxiv_id=arxiv_id,
        version=version,
        title=title,
        status=status,
        method=claim(method) if method else None,
        method_keywords=[mention(item) for item in (keywords or [])],
        datasets=[mention(item) for item in (datasets or [])],
        metrics=[mention(item) for item in (metrics or [])],
        limitations=[claim(item) for item in (limitations or [])],
        results=[claim(item) for item in (results or [])],
        problem=claim(problem) if problem else None,
        **kwargs,
    )


def snapshot(papers: list[ExtractedPaper], topic: str = "dataset pruning") -> ExtractionResult:
    return ExtractionResult(topic=topic, records=papers, fingerprint="upstream-fp")
