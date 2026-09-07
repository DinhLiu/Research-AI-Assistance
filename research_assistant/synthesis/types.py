"""Stage-3 data contracts: inventory, cards, clusters, claims, and coverage."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Disposition = Literal[
    "accepted",
    "skipped",
    "missing_method",
    "empty_features",
    "duplicate",
    "superseded_version",
    "strict_ok_excluded",
]
EvidenceKind = Literal[
    "problem",
    "method",
    "result",
    "limitation",
    "contribution",
    "dataset",
    "metric",
    "method_keyword",
]
ClaimKind = Literal["shared", "difference", "subset"]
CoverageState = Literal["observed", "unknown", "explicitly_not_evaluated"]
SummaryStatus = Literal["not_requested", "unavailable", "complete", "partial", "failed"]
ValidationStatus = Literal["structurally_validated", "rejected"]
GapKind = Literal["documented_limitation", "relationship_not_observed"]
RelatedWorkPolicy = Literal["excluded", "included", "unknown"]


class SynthesisInputError(ValueError):
    """Invalid extraction snapshot: conflicts or unreadable input."""


class EvidenceUnit(BaseModel):
    unit_id: str
    paper_key: str
    field_path: str
    kind: EvidenceKind
    text: str
    quote: str | None = None
    section: str | None = None
    source_kind: str | None = None
    start_char: int | None = None
    end_char: int | None = None


class EvidenceRegistry(BaseModel):
    units: list[EvidenceUnit] = Field(default_factory=list)

    def lookup(self) -> dict[str, EvidenceUnit]:
        return {unit.unit_id: unit for unit in self.units}


class PaperCard(BaseModel):
    paper_key: str
    arxiv_id: str
    version: str
    title: str = ""
    status: str = "ok"
    source_kind: str = "tex"
    confidence: float = 0.0
    problem_text: str | None = None
    method_text: str = ""
    method_keywords: list[str] = Field(default_factory=list)
    datasets: list[str] = Field(default_factory=list)
    metrics: list[str] = Field(default_factory=list)
    results: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    contributions: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)


class InventoryItem(BaseModel):
    index: int
    paper_key: str
    arxiv_id: str
    version: str
    status: str
    disposition: Disposition
    reason: str
    title: str = ""


class ClusterAssignment(BaseModel):
    cluster_id: str
    label: str
    paper_keys: list[str]
    top_keywords: list[str] = Field(default_factory=list)
    size: int = 0
    mean_within_distance: float | None = None
    nearest_other_distance: float | None = None


class UnassignedRecord(BaseModel):
    paper_key: str
    reason: str


class SynthesisClaim(BaseModel):
    text: str
    kind: ClaimKind
    support_refs: list[str] = Field(default_factory=list)
    subject_paper_keys: list[str] = Field(default_factory=list)
    validation_status: ValidationStatus = "rejected"
    rejection_reason: str | None = None


class ClusterSummary(BaseModel):
    cluster_id: str
    claims: list[SynthesisClaim] = Field(default_factory=list)


class ComparisonClaim(BaseModel):
    text: str
    cluster_ids: list[str] = Field(default_factory=list)
    support_refs: list[str] = Field(default_factory=list)
    subject_paper_keys: list[str] = Field(default_factory=list)
    comparability: Literal["unknown", "comparable"] = "unknown"
    validation_status: ValidationStatus = "rejected"
    rejection_reason: str | None = None


class GapCandidate(BaseModel):
    kind: GapKind
    statement: str
    scope: str = "extraction_snapshot"
    supporting_refs: list[str] = Field(default_factory=list)
    inspected_paper_keys: list[str] = Field(default_factory=list)
    unknown_paper_keys: list[str] = Field(default_factory=list)
    verification_needed: bool = True


class PaperCoverage(BaseModel):
    paper_key: str
    dataset_mentions: list[str] = Field(default_factory=list)
    metric_mentions: list[str] = Field(default_factory=list)
    dataset_state: CoverageState = "unknown"
    metric_state: CoverageState = "unknown"
    limitation_state: CoverageState = "unknown"


class DescriptiveCoverage(BaseModel):
    papers: list[PaperCoverage] = Field(default_factory=list)
    snapshot_note: str = (
        "Coverage describes mentions in this extraction snapshot. "
        "Co-occurrence is not an evaluation relation."
    )


class Diagnostics(BaseModel):
    n_input: int = 0
    n_accepted: int = 0
    n_clustered: int = 0
    n_unassigned: int = 0
    n_dropped: int = 0
    n_clusters: int = 0
    n_singletons: int = 0
    silhouette: float | None = None
    silhouette_reason: str | None = None
    distance_threshold: float = 0.55
    distance_threshold_status: str = "provisional"
    keyword_weight: float = 0.5
    empty_reason: str | None = None
    ambiguity_warning: bool = False
    channel_used: str | None = None


class ExecutionInfo(BaseModel):
    provider: str | None = None
    model: str | None = None
    logical_calls: int = 0
    summary_status: SummaryStatus = "not_requested"
    failure_reason: str | None = None
    seconds: float = 0.0
    cache_cluster_hit: bool = False
    cache_narration_hit: bool = False
    prompt_omitted_units: int = 0


class LlmClaimIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: ClaimKind
    text: str
    subject_paper_keys: list[str] = Field(default_factory=list)
    support_refs: list[str] = Field(default_factory=list)


class LlmClusterSummaryIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cluster_id: str
    claims: list[LlmClaimIn] = Field(default_factory=list)


class LlmComparisonIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str
    cluster_ids: list[str] = Field(default_factory=list)
    subject_paper_keys: list[str] = Field(default_factory=list)
    support_refs: list[str] = Field(default_factory=list)


class LlmNarration(BaseModel):
    model_config = ConfigDict(extra="forbid")

    summaries: list[LlmClusterSummaryIn] = Field(default_factory=list)
    comparisons: list[LlmComparisonIn] = Field(default_factory=list)


class SynthesisResult(BaseModel):
    schema_version: str = "1"
    topic: str | None = None
    corpus_digest: str = ""
    upstream_pipeline_fingerprint: str = ""
    related_work_policy: RelatedWorkPolicy = "unknown"
    effective_config: dict = Field(default_factory=dict)
    input_inventory: list[InventoryItem] = Field(default_factory=list)
    paper_manifest: list[PaperCard] = Field(default_factory=list)
    assignments: list[ClusterAssignment] = Field(default_factory=list)
    unassigned: list[UnassignedRecord] = Field(default_factory=list)
    evidence_registry: EvidenceRegistry = Field(default_factory=EvidenceRegistry)
    descriptive_coverage: DescriptiveCoverage = Field(default_factory=DescriptiveCoverage)
    summaries: list[ClusterSummary] = Field(default_factory=list)
    comparisons: list[ComparisonClaim] = Field(default_factory=list)
    gap_candidates: list[GapCandidate] = Field(default_factory=list)
    diagnostics: Diagnostics = Field(default_factory=Diagnostics)
    execution: ExecutionInfo = Field(default_factory=ExecutionInfo)
