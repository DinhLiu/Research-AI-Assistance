"""Section-aware regex/heuristic sentence candidates (no LLM)."""

from __future__ import annotations

import re

from research_assistant.extraction.normalize import normalize_heading, normalize_text
from research_assistant.extraction.sections import classify_heading
from research_assistant.extraction.types import CandidatePack, EvidenceSnippet, PaperDocument, PromptDocument, SourceKind

KIND_PREFIX = {
    "problem": "P",
    "method": "M",
    "result": "R",
    "limitation": "L",
    "contribution": "C",
}

METHOD_PATTERNS = [
    r"\bwe propose\b",
    r"\bwe introduce\b",
    r"\bour method\b",
    r"\bour approach\b",
    r"\bwe develop\b",
    r"\bwe present\b",
    r"\bwe prune\b",
    r"\bwe select\b",
]
PROBLEM_PATTERNS = [
    r"\bwe study\b",
    r"\bwe investigate\b",
    r"\bwe address\b",
    r"\bthe problem of\b",
    r"\bour goal is\b",
    r"\bwe consider\b",
]
CONTRIBUTION_PATTERNS = [
    r"\bour contributions?\b",
    r"\bwe make the following contributions\b",
    r"\bthis paper (?:makes|has) the following\b",
]
LIMITATION_PATTERNS = [
    r"\blimitation\b",
    r"\bdrawback\b",
    r"\bfuture work\b",
    r"\bunstable\b",
    r"\bdoes not\b",
    r"\bfail(?:s|ure)?\b",
]
RESULT_PATTERNS = [
    r"\bachieve",
    r"\baccuracy\b",
    r"\boutperform",
    r"\bwe evaluate\b",
    r"\bexperiments? (?:on|show)\b",
]

_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")
_MAX_PER_KIND = 8
_MAX_TOTAL = 28


def split_sentences(text: str) -> list[str]:
    text = normalize_text(text)
    if not text:
        return []
    parts = _SPLIT.split(text)
    out: list[str] = []
    for part in parts:
        sentence = part.strip()
        if 40 <= len(sentence) <= 600:
            out.append(sentence)
        elif 20 <= len(sentence) < 40 and out:
            out[-1] = f"{out[-1]} {sentence}".strip()
        elif 20 <= len(sentence) < 40:
            out.append(sentence)
    return out


def _score(sentence: str, patterns: list[str], section_kind: str, preferred: tuple[str, ...]) -> float:
    value = 0.0
    lowered = sentence.lower()
    for pattern in patterns:
        if re.search(pattern, lowered, flags=re.IGNORECASE):
            value += 4.0
            break
    if section_kind in preferred:
        value += 3.0
    if section_kind in {"related", "biblio", "appendix"}:
        value -= 4.0
    return value


def build_candidate_pack(
    paper: PaperDocument,
    prompt: PromptDocument,
    *,
    source_kind: SourceKind | None = None,
    datasets: list | None = None,
    metrics: list | None = None,
) -> CandidatePack:
    from research_assistant.extraction.deterministic import extract_datasets, extract_metrics

    kind_source = source_kind or prompt.source_kind
    selected_chars = sum(len(s.heading) + len(s.text) for s in prompt.selected_sections)
    buckets: dict[str, list[tuple[float, EvidenceSnippet]]] = {
        "problem": [],
        "method": [],
        "result": [],
        "limitation": [],
        "contribution": [],
    }
    by_sentence: dict[str, tuple[float, str, EvidenceSnippet]] = {}
    for section in prompt.selected_sections:
        section_kind = classify_heading(section.heading)
        for sentence in split_sentences(section.text):
            span = _span(section.text, sentence)
            best_kind = None
            best_score = 0.0
            specs = (
                ("problem", PROBLEM_PATTERNS, ("always", "other")),
                ("method", METHOD_PATTERNS, ("method", "always")),
                ("result", RESULT_PATTERNS, ("results",)),
                ("limitation", LIMITATION_PATTERNS, ("limits", "always")),
                ("contribution", CONTRIBUTION_PATTERNS, ("always", "method")),
            )
            for kind, patterns, preferred in specs:
                score = _score(sentence, patterns, section_kind, preferred)
                if score > best_score:
                    best_score = score
                    best_kind = kind
            if best_kind is None or best_score < 3.0:
                continue
            prefix = KIND_PREFIX[best_kind]
            snippet = EvidenceSnippet(
                id=f"{prefix}0",
                kind=best_kind,  # type: ignore[arg-type]
                sentence=sentence,
                section=section.heading,
                start_char=span[0] if span else None,
                end_char=span[1] if span else None,
                score=best_score,
            )
            prev = by_sentence.get(sentence)
            if prev is None or best_score > prev[0]:
                by_sentence[sentence] = (best_score, best_kind, snippet)

    for score, kind, snippet in by_sentence.values():
        buckets[kind].append((score, snippet))

    _backfill(buckets, prompt, "method", ("method", "always"))
    _backfill(buckets, prompt, "problem", ("always",))

    snippets: list[EvidenceSnippet] = []
    for kind in ("problem", "method", "result", "limitation", "contribution"):
        ranked = sorted(buckets[kind], key=lambda item: item[0], reverse=True)[:_MAX_PER_KIND]
        for i, (score, snippet) in enumerate(ranked, start=1):
            snippet.id = f"{KIND_PREFIX[kind]}{i}"
            snippet.score = score
            snippets.append(snippet)
            if len(snippets) >= _MAX_TOTAL:
                break

    ds = datasets if datasets is not None else extract_datasets(paper, prompt)
    ms = metrics if metrics is not None else extract_metrics(paper, prompt)
    pack = CandidatePack(
        arxiv_id=prompt.arxiv_id,
        version=prompt.version,
        title=prompt.title or paper.title,
        abstract=prompt.abstract or paper.abstract,
        source_kind=kind_source,
        snippets=snippets,
        datasets=ds,
        metrics=ms,
        selected_chars=selected_chars,
        local_confidence=_local_confidence(prompt, snippets, ds),
    )
    pack.candidate_chars = estimate_candidate_chars(pack)
    return pack


def estimate_candidate_chars(pack: CandidatePack) -> int:
    return sum(len(s.sentence) for s in pack.snippets)


def _backfill(
    buckets: dict[str, list[tuple[float, EvidenceSnippet]]],
    prompt: PromptDocument,
    kind: str,
    section_kinds: tuple[str, ...],
) -> None:
    if buckets[kind]:
        return
    prefix = KIND_PREFIX[kind]
    idx = 1
    for section in prompt.selected_sections:
        if classify_heading(section.heading) not in section_kinds:
            continue
        for sentence in split_sentences(section.text)[:4]:
            span = _span(section.text, sentence)
            buckets[kind].append(
                (
                    1.0,
                    EvidenceSnippet(
                        id=f"{prefix}{idx}",
                        kind=kind,  # type: ignore[arg-type]
                        sentence=sentence,
                        section=section.heading,
                        start_char=span[0] if span else None,
                        end_char=span[1] if span else None,
                        score=1.0,
                    ),
                )
            )
            idx += 1
            if idx > 4:
                return


def _span(haystack: str, needle: str) -> tuple[int, int] | None:
    idx = haystack.find(needle)
    if idx < 0:
        idx = normalize_text(haystack).find(normalize_text(needle))
        if idx < 0:
            return None
        if haystack != normalize_text(haystack):
            return None
    return idx, idx + len(needle)


def _local_confidence(prompt: PromptDocument, snippets: list[EvidenceSnippet], datasets: list) -> float:
    methods = [s for s in snippets if s.kind == "method" and s.score >= 6.0]
    if not methods:
        return 0.0
    heading = normalize_heading(methods[0].section)
    in_method = "method" in heading or "approach" in heading
    title = (prompt.title or "").lower()
    blob = methods[0].sentence.lower()
    named = any(token in blob and token in title for token in ("el2n", "grand", "coreset", "infobatch") if token in title)
    value = 0.55
    if in_method:
        value += 0.2
    if datasets:
        value += 0.1
    if named:
        value += 0.15
    if re.search(r"\bwe propose\b|\bwe introduce\b", blob):
        value += 0.1
    return round(min(value, 0.99), 3)
