"""Structural grounding for synthesis claims. Not entailment."""

from __future__ import annotations

import re

from research_assistant.arxiv_ids import normalize_arxiv_id, paper_key, split_arxiv_id
from research_assistant.synthesis.types import (
    ClaimKind,
    ClusterAssignment,
    ComparisonClaim,
    ClusterSummary,
    EvidenceKind,
    EvidenceRegistry,
    EvidenceUnit,
    LlmNarration,
    SynthesisClaim,
)

_ARXIV_TOKEN = re.compile(r"\b(?:\d{4}\.\d{4,5}|[a-z-]+/\d{7})(?:v\d+)?\b", re.I)
_SCORE = re.compile(r"\d+(?:\.\d+)?\s*(?:%|pp\b|accuracy\b)", re.I)

SHARED_KINDS: frozenset[EvidenceKind] = frozenset(
    {"method", "method_keyword", "problem", "contribution"}
)
DIFFERENCE_KINDS: frozenset[EvidenceKind] = frozenset(
    {"method", "method_keyword", "result", "limitation", "contribution"}
)
COMPARISON_KINDS: frozenset[EvidenceKind] = frozenset({"method", "result", "limitation"})


def named_paper_keys(text: str, known_keys: set[str]) -> set[str]:
    found: set[str] = set()
    for token in _ARXIV_TOKEN.findall(text or ""):
        core, version = split_arxiv_id(token)
        # An explicit version must never resolve to evidence from another version.
        if version:
            found.add(paper_key(core, version))
            continue
        matches = [key for key in known_keys if normalize_arxiv_id(key) == core]
        if len(matches) == 1:
            found.add(matches[0])
        elif matches:
            found.add(token)
        else:
            found.add(paper_key(core, version))
    return found


def validate_claim(
    claim: SynthesisClaim,
    *,
    permitted_keys: set[str],
    cluster_keys: set[str] | None,
    registry: dict[str, EvidenceUnit],
    permitted_kinds: frozenset[EvidenceKind],
) -> SynthesisClaim:
    reasons: list[str] = []
    if not claim.text.strip():
        reasons.append("empty_text")
    subjects = list(dict.fromkeys(claim.subject_paper_keys))
    if not subjects:
        reasons.append("no_subjects")
    unknown_subjects = [key for key in subjects if key not in permitted_keys]
    if unknown_subjects:
        reasons.append(f"unknown_subjects:{','.join(unknown_subjects)}")

    named = named_paper_keys(claim.text, permitted_keys | set(subjects))
    extra_named = [key for key in named if key not in subjects]
    if extra_named:
        reasons.append(f"named_unsupported_subjects:{','.join(sorted(extra_named))}")

    supported_subjects: set[str] = set()
    for ref in claim.support_refs:
        unit = registry.get(ref)
        if unit is None:
            reasons.append(f"unknown_ref:{ref}")
            continue
        if unit.paper_key not in permitted_keys:
            reasons.append(f"cross_cluster_ref:{ref}")
            continue
        if unit.kind not in permitted_kinds:
            reasons.append(f"wrong_kind:{ref}:{unit.kind}")
            continue
        supported_subjects.add(unit.paper_key)

    missing = [key for key in subjects if key not in supported_subjects]
    if missing:
        reasons.append(f"unsupported_subjects:{','.join(missing)}")

    if claim.kind == "shared" and len(supported_subjects) < 2:
        reasons.append("shared_requires_two_subjects")
    if claim.kind == "difference" and len(supported_subjects) < 2:
        reasons.append("difference_requires_both_sides")
    if cluster_keys and set(subjects) == cluster_keys and missing:
        reasons.append("all_members_claim_incomplete")

    if reasons:
        return claim.model_copy(
            update={"validation_status": "rejected", "rejection_reason": "; ".join(dict.fromkeys(reasons))}
        )
    return claim.model_copy(update={"validation_status": "structurally_validated", "rejection_reason": None})


def validate_comparison(
    claim: ComparisonClaim,
    *,
    assignments: list[ClusterAssignment],
    registry: dict[str, EvidenceUnit],
) -> ComparisonClaim:
    by_id = {item.cluster_id: item for item in assignments}
    unknown_clusters = [cid for cid in claim.cluster_ids if cid not in by_id]
    reasons: list[str] = []
    if unknown_clusters:
        reasons.append(f"unknown_cluster_ids:{','.join(unknown_clusters)}")
    permitted: set[str] = set()
    for cid in claim.cluster_ids:
        cluster = by_id.get(cid)
        if cluster:
            permitted.update(cluster.paper_keys)
    if _SCORE.search(claim.text or ""):
        reasons.append("numeric_comparison_unsupported")
    checked = validate_claim(
        SynthesisClaim(
            text=claim.text,
            kind="difference",
            support_refs=claim.support_refs,
            subject_paper_keys=claim.subject_paper_keys,
        ),
        permitted_keys=permitted,
        cluster_keys=None,
        registry=registry,
        permitted_kinds=COMPARISON_KINDS,
    )
    if checked.rejection_reason:
        reasons.append(checked.rejection_reason)
    if reasons:
        return claim.model_copy(
            update={
                "validation_status": "rejected",
                "rejection_reason": "; ".join(dict.fromkeys(reasons)),
                "comparability": "unknown",
            }
        )
    return claim.model_copy(
        update={"validation_status": "structurally_validated", "rejection_reason": None, "comparability": "unknown"}
    )


def kinds_for(kind: ClaimKind) -> frozenset[EvidenceKind]:
    if kind == "difference":
        return DIFFERENCE_KINDS
    return SHARED_KINDS


def apply_narration(
    payload: LlmNarration,
    *,
    assignments: list[ClusterAssignment],
    registry: EvidenceRegistry,
) -> tuple[list[ClusterSummary], list[ComparisonClaim], list[str]]:
    errors: list[str] = []
    by_id = {item.cluster_id: item for item in assignments}
    lookup = registry.lookup()
    summaries: list[ClusterSummary] = []
    seen_clusters: set[str] = set()
    for raw in payload.summaries:
        cluster = by_id.get(raw.cluster_id)
        if cluster is None:
            errors.append(f"unknown_cluster_id:{raw.cluster_id}")
            continue
        seen_clusters.add(raw.cluster_id)
        permitted = set(cluster.paper_keys)
        claims: list[SynthesisClaim] = []
        for item in raw.claims:
            claim = SynthesisClaim(
                text=item.text,
                kind=item.kind,
                support_refs=list(item.support_refs),
                subject_paper_keys=list(item.subject_paper_keys),
            )
            checked = validate_claim(
                claim,
                permitted_keys=permitted,
                cluster_keys=permitted,
                registry=lookup,
                permitted_kinds=kinds_for(item.kind),
            )
            claims.append(checked)
            if checked.validation_status == "rejected":
                errors.append(f"{raw.cluster_id}:{checked.rejection_reason}")
        if not any(claim.validation_status == "structurally_validated" for claim in claims):
            errors.append(f"empty_cluster_summary:{raw.cluster_id}")
        summaries.append(ClusterSummary(cluster_id=raw.cluster_id, claims=claims))

    missing = [item.cluster_id for item in assignments if item.cluster_id not in seen_clusters]
    if missing:
        errors.append("missing_cluster_summaries:" + ",".join(missing))

    comparisons: list[ComparisonClaim] = []
    for item in payload.comparisons:
        claim = ComparisonClaim(
            text=item.text,
            cluster_ids=list(item.cluster_ids),
            support_refs=list(item.support_refs),
            subject_paper_keys=list(item.subject_paper_keys),
        )
        checked = validate_comparison(claim, assignments=assignments, registry=lookup)
        comparisons.append(checked)
        if checked.validation_status == "rejected":
            errors.append(f"comparison:{checked.rejection_reason}")
    return summaries, comparisons, errors


def keep_validated(
    summaries: list[ClusterSummary],
    comparisons: list[ComparisonClaim],
) -> tuple[list[ClusterSummary], list[ComparisonClaim], int, int]:
    kept_summaries = []
    kept_claims = 0
    rejected_claims = 0
    for summary in summaries:
        valid = [claim for claim in summary.claims if claim.validation_status == "structurally_validated"]
        rejected_claims += len(summary.claims) - len(valid)
        kept_claims += len(valid)
        kept_summaries.append(ClusterSummary(cluster_id=summary.cluster_id, claims=valid))
    valid_comparisons = [item for item in comparisons if item.validation_status == "structurally_validated"]
    rejected_claims += len(comparisons) - len(valid_comparisons)
    kept_claims += len(valid_comparisons)
    return kept_summaries, valid_comparisons, kept_claims, rejected_claims
