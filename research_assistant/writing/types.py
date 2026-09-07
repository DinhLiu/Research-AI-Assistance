"""Writing contracts. The model may paraphrase slots, never own provenance."""
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field


class WritingInputError(ValueError):
    pass


class ReviewClaim(BaseModel):
    claim_id: str
    section_id: str
    text: str
    kind: str
    subject_paper_keys: list[str]
    support_refs: list[str]
    source_claim_id: str | None = None
    scope: str = "extraction_snapshot"
    verification_needed: bool = True
    validation_status: str = "structurally_validated"
    generation: Literal["template", "llm"] = "template"


class ReviewSection(BaseModel):
    section_id: str
    title: str
    claim_ids: list[str] = Field(default_factory=list)


class WritingBatch(BaseModel):
    batch_id: str
    claim_ids: list[str]
    estimated_input_tokens: int
    estimated_output_tokens: int


class ReviewPlan(BaseModel):
    topic: str | None = None
    sections: list[ReviewSection] = Field(default_factory=list)
    slots: list[ReviewClaim] = Field(default_factory=list)
    batches: list[WritingBatch] = Field(default_factory=list)
    omitted: dict[str, str] = Field(default_factory=dict)
    evidence_ids: list[str] = Field(default_factory=list)
    unselected_evidence_ids: list[str] = Field(default_factory=list)
    evidence_warnings: dict[str, str] = Field(default_factory=dict)
    language: str
    target_words: int
    token_estimator: str = "utf8_bytes_plus_overhead_upper_estimate"


class WrittenClaim(BaseModel):
    model_config = ConfigDict(extra="forbid")
    claim_id: str
    text: str = Field(min_length=1)


class WrittenPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")
    claims: list[WrittenClaim]


class WritingExecution(BaseModel):
    run_id: str | None = None
    provider: str | None = None
    model: str | None = None
    logical_calls: int = 0
    repair_calls: int = 0
    http_attempts: int = 0
    reserved_tokens: int = 0
    transport_retries: int = 0
    input_tokens: int | None = None
    output_tokens: int | None = None
    cache_hits: int = 0
    queue_wait_seconds: float = 0
    api_seconds: float = 0
    seconds: float = 0
    stop_reason: str | None = None
    quota: dict = Field(default_factory=dict)


class WritingResult(BaseModel):
    schema_version: str = "1"
    writing_input_digest: str
    corpus_digest: str
    topic: str | None
    generation_status: Literal["complete", "partial", "fallback", "empty", "failed", "planned"]
    validation_status: str = "structurally_validated_not_semantically_verified"
    plan: ReviewPlan
    claims: list[ReviewClaim] = Field(default_factory=list)
    bibliography: list[dict] = Field(default_factory=list)
    coverage: dict = Field(default_factory=dict)
    validation_errors: list[str] = Field(default_factory=list)
    execution: WritingExecution = Field(default_factory=WritingExecution)
