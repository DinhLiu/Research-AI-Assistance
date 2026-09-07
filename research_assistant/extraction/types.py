"""Stage-2 data contracts: documents, claims, mentions, and extraction records."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

SourceKind = Literal["tex", "pdf", "abstract"]
ParsedSourceKind = Literal["tex", "pdf"]
PaperStatus = Literal["ok", "degraded", "skipped"]
AttemptOutcome = Literal["ok", "struct_fail", "evidence_fail", "llm_fail", "parse_fail"]


class Evidence(BaseModel):
    quote: str
    section: str
    source_kind: SourceKind
    start_char: int | None = None
    end_char: int | None = None


class Claim(BaseModel):
    text: str
    evidence: list[Evidence] = Field(min_length=1)


class Mention(BaseModel):
    value: str
    evidence: Evidence


class Section(BaseModel):
    heading: str
    level: int = 1
    text: str


class SectionPolicy(BaseModel):
    include_related_work: bool = False
    include_appendix: bool = False
    include_bibliography: bool = False
    abstract_only: bool = False
    max_input_chars: int = 24000
    max_input_tokens: int | None = None


class PaperDocument(BaseModel):
    """Parsed paper. Source of truth; not sent to the LLM as a whole."""

    arxiv_id: str
    version: str
    source_kind: ParsedSourceKind
    title: str = ""
    abstract: str = ""
    sections: list[Section] = Field(default_factory=list)

    @property
    def full_text(self) -> str:
        parts: list[str] = []
        if self.title:
            parts.append(self.title)
        if self.abstract:
            parts.append(self.abstract)
        for section in self.sections:
            parts.append(f"{section.heading}\n{section.text}".strip())
        return "\n\n".join(parts)


class PromptDocument(BaseModel):
    """Subset of a paper the extractor is allowed to see."""

    arxiv_id: str
    version: str
    title: str = ""
    abstract: str = ""
    source_kind: SourceKind
    selected_sections: list[Section] = Field(default_factory=list)
    selector_version: str = "1"
    policy: SectionPolicy = Field(default_factory=SectionPolicy)


class EvidenceSnippet(BaseModel):
    """A locally selected sentence the LLM may cite by id only."""

    id: str
    kind: Literal["problem", "method", "result", "limitation", "contribution"]
    sentence: str
    section: str
    start_char: int | None = None
    end_char: int | None = None
    score: float = 0.0


class CandidatePack(BaseModel):
    """Compressed paper view: snippets + deterministic mentions. Cached separately from LLM output."""

    arxiv_id: str
    version: str
    title: str = ""
    abstract: str = ""
    source_kind: SourceKind = "tex"
    snippets: list[EvidenceSnippet] = Field(default_factory=list)
    datasets: list[Mention] = Field(default_factory=list)
    metrics: list[Mention] = Field(default_factory=list)
    selected_chars: int = 0
    candidate_chars: int = 0
    local_confidence: float = 0.0

    def lookup(self) -> dict[str, EvidenceSnippet]:
        return {item.id: item for item in self.snippets}


class IdRefClaim(BaseModel):
    text: str
    evidence_ids: list[str] = Field(min_length=1)


class IdRefMention(BaseModel):
    value: str
    evidence_ids: list[str] = Field(min_length=1)


class LlmExtraction(BaseModel):
    arxiv_id: str | None = None
    problem: IdRefClaim | None = None
    method: IdRefClaim | None = None
    results: list[IdRefClaim] = Field(default_factory=list)
    limitations: list[IdRefClaim] = Field(default_factory=list)
    contributions: list[IdRefClaim] = Field(default_factory=list)
    method_keywords: list[IdRefMention] = Field(default_factory=list)


class LlmBatchExtraction(BaseModel):
    papers: list[LlmExtraction] = Field(default_factory=list)


class EvidenceValidationResult(BaseModel):
    valid: bool
    exact_section_match: bool = False
    located_section: str | None = None
    start_char: int | None = None
    end_char: int | None = None
    in_full_text: bool = False


class ExtractionAttempt(BaseModel):
    source_kind: SourceKind
    attempt_no: int
    outcome: AttemptOutcome
    validation_errors: list[str] = Field(default_factory=list)


class ExtractedPaper(BaseModel):
    arxiv_id: str
    version: str
    title: str = ""
    source_kind: SourceKind = "tex"
    status: PaperStatus = "skipped"
    problem: Claim | None = None
    method: Claim | None = None
    results: list[Claim] = Field(default_factory=list)
    limitations: list[Claim] = Field(default_factory=list)
    contributions: list[Claim] = Field(default_factory=list)
    datasets: list[Mention] = Field(default_factory=list)
    metrics: list[Mention] = Field(default_factory=list)
    method_keywords: list[Mention] = Field(default_factory=list)
    confidence: float = 0.0
    attempts: list[ExtractionAttempt] = Field(default_factory=list)
    error: str | None = None
    units_proposed: int = 0
    units_kept: int = 0
    relocated_evidence: int = 0
    fingerprint: str = ""


class ExtractionBundle(BaseModel):
    """Cached pair: what the model saw + what we accepted."""

    prompt_document: PromptDocument
    extracted: ExtractedPaper


class ExtractionResult(BaseModel):
    topic: str | None = None
    records: list[ExtractedPaper] = Field(default_factory=list)
    metrics: dict = Field(default_factory=dict)
    fingerprint: str = ""
