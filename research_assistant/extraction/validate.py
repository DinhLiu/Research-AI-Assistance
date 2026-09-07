"""Evidence gate: LLM may only cite snippet IDs from the candidate pack."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from research_assistant.extraction.normalize import headings_match, normalize_text
from research_assistant.extraction.types import (
    CandidatePack,
    Claim,
    Evidence,
    EvidenceSnippet,
    EvidenceValidationResult,
    ExtractedPaper,
    IdRefClaim,
    IdRefMention,
    LlmBatchExtraction,
    LlmExtraction,
    Mention,
    PromptDocument,
    SourceKind,
)

_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


@dataclass
class GroundingReport:
    errors: list[str] = field(default_factory=list)
    relocated: int = 0
    proposed: int = 0
    kept: int = 0


def extract_json_object(raw: str) -> dict:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    match = _JSON_OBJECT.search(text)
    if not match:
        raise ValueError("LLM response did not contain a JSON object")
    payload = json.loads(match.group(0))
    if not isinstance(payload, dict):
        raise ValueError("JSON is not an object")
    return payload


def parse_llm_payload(raw: str) -> list[LlmExtraction]:
    payload = extract_json_object(raw)
    if "papers" in payload:
        batch = LlmBatchExtraction.model_validate(payload)
        return batch.papers
    return [LlmExtraction.model_validate(payload)]


def parse_candidate(raw: str) -> LlmExtraction:
    papers = parse_llm_payload(raw)
    if not papers:
        raise ValueError("JSON contained no paper objects")
    return papers[0]


def locate_quote(
    quote: str,
    declared_section: str,
    prompt: PromptDocument,
    *,
    full_text: str | None = None,
) -> EvidenceValidationResult:
    needle = normalize_text(quote)
    if not needle:
        return EvidenceValidationResult(valid=False)

    declared_hits = [
        section
        for section in prompt.selected_sections
        if headings_match(declared_section, section.heading)
    ]
    search_order = declared_hits + [
        section for section in prompt.selected_sections if section not in declared_hits
    ]
    for section in search_order:
        span = _find_span(section.text, needle)
        if span is None:
            continue
        exact = bool(declared_hits) and section in declared_hits
        return EvidenceValidationResult(
            valid=True,
            exact_section_match=exact,
            located_section=section.heading,
            start_char=span[0],
            end_char=span[1],
        )
    in_full = bool(full_text) and _find_span(full_text or "", needle) is not None
    return EvidenceValidationResult(valid=False, in_full_text=in_full)


def snippet_to_evidence(snippet: EvidenceSnippet, source_kind: SourceKind) -> Evidence:
    return Evidence(
        quote=snippet.sentence,
        section=snippet.section,
        source_kind=source_kind,
        start_char=snippet.start_char,
        end_char=snippet.end_char,
    )


def resolve_ids(
    ids: list[str],
    pack: CandidatePack,
    report: GroundingReport,
    allowed_paper: str,
) -> list[Evidence]:
    lookup = pack.lookup()
    evidence: list[Evidence] = []
    for ref in ids:
        report.proposed += 1
        key = str(ref).strip().upper()
        snippet = lookup.get(key) or lookup.get(str(ref).strip())
        if snippet is None:
            report.errors.append(f"Unknown evidence id {ref!r} for {allowed_paper}")
            continue
        report.kept += 1
        evidence.append(snippet_to_evidence(snippet, pack.source_kind))
    return evidence


def ground_id_claim(
    raw: IdRefClaim | None,
    pack: CandidatePack,
    report: GroundingReport,
) -> Claim | None:
    if raw is None:
        return None
    evidence = resolve_ids(raw.evidence_ids, pack, report, pack.arxiv_id)
    text = normalize_text(raw.text)
    if not text or not evidence:
        if text:
            report.errors.append(f"Dropped claim with no valid evidence ids: {text[:80]}")
        return None
    return Claim(text=text, evidence=evidence)


def ground_id_mention(
    raw: IdRefMention,
    pack: CandidatePack,
    report: GroundingReport,
) -> Mention | None:
    evidence = resolve_ids(raw.evidence_ids, pack, report, pack.arxiv_id)
    value = normalize_text(raw.value)
    if not value or not evidence:
        if value:
            report.errors.append(f"Dropped keyword {value!r}: bad evidence ids")
        return None
    return Mention(value=value, evidence=evidence[0])


def ground_llm(
    llm: LlmExtraction,
    pack: CandidatePack,
) -> tuple[ExtractedPaper, GroundingReport]:
    report = GroundingReport()
    if llm.arxiv_id:
        left = llm.arxiv_id.replace("v", "").split("v")[0]
        if pack.arxiv_id not in llm.arxiv_id and llm.arxiv_id not in pack.arxiv_id:
            # Soft check: allow missing version suffix.
            if normalize_text(llm.arxiv_id) not in {pack.arxiv_id, pack.arxiv_id + pack.version}:
                if not llm.arxiv_id.startswith(pack.arxiv_id) and not pack.arxiv_id.startswith(
                    re.sub(r"v\d+$", "", llm.arxiv_id)
                ):
                    report.errors.append(
                        f"arxiv_id mismatch: llm={llm.arxiv_id!r} pack={pack.arxiv_id!r}"
                    )
    problem = ground_id_claim(llm.problem, pack, report)
    method = ground_id_claim(llm.method, pack, report)
    results = [c for c in (ground_id_claim(x, pack, report) for x in llm.results) if c]
    limitations = [c for c in (ground_id_claim(x, pack, report) for x in llm.limitations) if c]
    contributions = [c for c in (ground_id_claim(x, pack, report) for x in llm.contributions) if c]
    keywords = [m for m in (ground_id_mention(x, pack, report) for x in llm.method_keywords) if m]

    status = "skipped"
    confidence = 0.0
    error = None
    if method is not None:
        status = "degraded" if pack.source_kind == "abstract" else "ok"
        confidence = _confidence(pack.source_kind, report)
        confidence = min(1.0, confidence + 0.05 * pack.local_confidence)
    else:
        error = "method missing after evidence-id gate"
        report.errors.append(error)

    paper = ExtractedPaper(
        arxiv_id=pack.arxiv_id,
        version=pack.version,
        title=pack.title,
        source_kind=pack.source_kind,
        status=status,
        problem=problem,
        method=method,
        results=results,
        limitations=limitations,
        contributions=contributions,
        datasets=list(pack.datasets),
        metrics=list(pack.metrics),
        method_keywords=keywords,
        confidence=round(min(confidence, 1.0), 3),
        error=error,
        units_proposed=report.proposed,
        units_kept=report.kept,
        relocated_evidence=0,
    )
    return paper, report


def paper_from_local_candidates(pack: CandidatePack) -> ExtractedPaper | None:
    """Skip the LLM when method/problem snippets are already high-confidence."""
    if pack.local_confidence < 0.92:
        return None
    methods = [s for s in pack.snippets if s.kind == "method"]
    problems = [s for s in pack.snippets if s.kind == "problem"]
    if not methods:
        return None
    method = Claim(
        text=methods[0].sentence,
        evidence=[snippet_to_evidence(methods[0], pack.source_kind)],
    )
    problem = None
    if problems:
        problem = Claim(
            text=problems[0].sentence,
            evidence=[snippet_to_evidence(problems[0], pack.source_kind)],
        )
    keywords = _keywords_from_snippets(methods, pack.source_kind)
    status: str = "degraded" if pack.source_kind == "abstract" else "ok"
    return ExtractedPaper(
        arxiv_id=pack.arxiv_id,
        version=pack.version,
        title=pack.title,
        source_kind=pack.source_kind,
        status=status,  # type: ignore[arg-type]
        problem=problem,
        method=method,
        datasets=list(pack.datasets),
        metrics=list(pack.metrics),
        method_keywords=keywords,
        confidence=pack.local_confidence,
        units_proposed=1,
        units_kept=1,
    )


def _keywords_from_snippets(snippets: list[EvidenceSnippet], source_kind: SourceKind) -> list[Mention]:
    found: dict[str, Mention] = {}
    token_re = re.compile(r"\b(?:EL2N|GraNd|CCS|InfoBatch|coreset|RRS)\b", re.IGNORECASE)
    for snippet in snippets:
        for match in token_re.finditer(snippet.sentence):
            value = match.group(0)
            if value.lower() == "coreset":
                value = "coreset"
            elif value.upper() == "EL2N":
                value = "EL2N"
            found.setdefault(
                value.lower(),
                Mention(value=value, evidence=snippet_to_evidence(snippet, source_kind)),
            )
    return list(found.values())


def _confidence(source_kind: SourceKind, report: GroundingReport) -> float:
    value = 1.0
    if source_kind == "pdf":
        value -= 0.1
    if source_kind == "abstract":
        value -= 0.35
    if report.proposed:
        value -= 0.25 * (1.0 - report.kept / report.proposed)
    return max(0.0, min(1.0, round(value, 3)))


def _find_span(haystack: str, needle: str) -> tuple[int, int] | None:
    if not needle:
        return None
    idx = haystack.find(needle)
    if idx >= 0:
        return idx, idx + len(needle)
    norm_hay = normalize_text(haystack)
    norm_needle = normalize_text(needle)
    idx = norm_hay.find(norm_needle)
    if idx < 0:
        return None
    if haystack == norm_hay:
        return idx, idx + len(norm_needle)
    return None
