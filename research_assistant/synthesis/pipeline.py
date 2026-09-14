"""Offline clustering plus optional bounded narration."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Any

import numpy as np

from research_assistant.config import SynthesisConfig
from research_assistant.extraction.types import ExtractionResult
from research_assistant.extraction.validate import extract_json_object
from research_assistant.llm.client import complete, extraction_provider, model_for_provider, LlmError
from research_assistant.synthesis.cards import (
    build_inventory,
    build_registry_and_cards,
    mark_empty_features,
)
from research_assistant.synthesis.cluster import cluster_papers
from research_assistant.synthesis.coverage import build_coverage, build_gap_candidates
from research_assistant.synthesis.features import build_features
from research_assistant.synthesis.fingerprint import (
    cluster_cache_key,
    cluster_path,
    corpus_digest,
    effective_config,
    evidence_digest,
    load_result,
    narration_cache_key,
    narration_path,
    save_result,
)
from research_assistant.synthesis.prompt import (
    SYSTEM_PROMPT,
    build_comparison_prompt,
    build_prompt_batches,
    prompt_within_budget,
    repair_user_message,
)
from research_assistant.synthesis.types import (
    ClusterAssignment,
    ClusterSummary,
    ComparisonClaim,
    Diagnostics,
    EvidenceRegistry,
    ExecutionInfo,
    LlmClaimIn,
    LlmClusterSummaryIn,
    LlmNarration,
    PaperCard,
    SynthesisResult,
)
from research_assistant.synthesis.validate import apply_narration, keep_validated, validate_comparison

logger = logging.getLogger(__name__)

CompleteFn = Callable[..., str]


def synthesize(
    extracted: ExtractionResult,
    config: SynthesisConfig | None = None,
    *,
    complete_fn: CompleteFn | None = None,
) -> SynthesisResult:
    cfg = config or SynthesisConfig()
    t0 = time.perf_counter()
    snapshot = extracted if isinstance(extracted, ExtractionResult) else ExtractionResult.model_validate(extracted)
    digest = corpus_digest(snapshot.records)
    ckey = cluster_cache_key(digest, cfg)

    result = None
    cluster_hit = False
    if cfg.use_cache:
        cached = load_result(cluster_path(cfg.cache_dir, ckey))
        if cached is not None and cached.corpus_digest == digest:
            result = cached
            # Membership is order-independent; row indices and skip reasons are not.
            inventory, _accepted = build_inventory(snapshot.records, strict_ok=cfg.strict_ok)
            empty_keys = {item.paper_key for item in cached.input_inventory if item.disposition == "empty_features"}
            mark_empty_features(inventory, empty_keys)
            result.input_inventory = inventory
            cluster_hit = True
            result.summaries = []
            result.comparisons = []
            result.execution = ExecutionInfo(
                summary_status="not_requested",
                cache_cluster_hit=True,
            )

    if result is None:
        result = _cluster_offline(snapshot, cfg, digest)
        if cfg.use_cache:
            save_result(cluster_path(cfg.cache_dir, ckey), _cluster_snapshot(result))

    result.execution.cache_cluster_hit = cluster_hit
    result.topic = snapshot.topic
    result.upstream_pipeline_fingerprint = snapshot.fingerprint
    result.effective_config = effective_config(cfg)

    if not cfg.summarize:
        result.execution.summary_status = "not_requested"
        result.execution.seconds = round(time.perf_counter() - t0, 3)
        return result

    _run_narration(result, cfg, ckey, complete_fn)
    result.execution.cache_cluster_hit = cluster_hit
    result.execution.seconds = round(time.perf_counter() - t0, 3)
    return result


def _cluster_offline(snapshot: ExtractionResult, cfg: SynthesisConfig, digest: str) -> SynthesisResult:
    inventory, accepted = build_inventory(snapshot.records, strict_ok=cfg.strict_ok)
    registry, cards = build_registry_and_cards(accepted)
    empty_reason = None
    if not snapshot.records:
        empty_reason = "empty_input"
    elif not accepted:
        empty_reason = "no_usable_papers"

    features = build_features(
        [card.paper_key for card in cards],
        [card.method_text for card in cards],
        [card.method_keywords for card in cards],
        keyword_weight=cfg.keyword_weight,
        max_features=cfg.max_features,
    )
    empty_set = set(features.empty_keys)
    mark_empty_features(inventory, empty_set)
    clustered_cards = [card for card in cards if card.paper_key not in empty_set]
    if cards and not clustered_cards:
        empty_reason = empty_reason or "empty_features"

    key_to_row = {key: index for index, key in enumerate(features.paper_keys)}
    ordered_keys = [card.paper_key for card in clustered_cards]
    if ordered_keys:
        rows = [key_to_row[key] for key in ordered_keys]
        matrix = features.matrix[rows]
    else:
        matrix = np.zeros((0, 1), dtype=np.float64)

    assignments, unassigned, cluster_diag = cluster_papers(
        ordered_keys,
        matrix,
        clustered_cards,
        distance_threshold=cfg.distance_threshold,
        clustering_version=cfg.clustering_version,
        empty_keys=features.empty_keys,
    )
    coverage = build_coverage(cards)
    gaps = build_gap_candidates(cards, registry, coverage)
    n_clustered = sum(item.size for item in assignments)
    n_unassigned = len(unassigned)
    n_dropped = sum(1 for item in inventory if item.disposition not in {"accepted", "empty_features"})
    n_accepted = sum(1 for item in inventory if item.disposition in {"accepted", "empty_features"})
    diagnostics = Diagnostics(
        n_input=len(snapshot.records),
        n_accepted=n_accepted,
        n_clustered=n_clustered,
        n_unassigned=n_unassigned,
        n_dropped=n_dropped,
        n_clusters=cluster_diag.get("n_clusters") or len(assignments),
        n_singletons=cluster_diag.get("n_singletons") or 0,
        silhouette=cluster_diag.get("silhouette"),
        silhouette_reason=cluster_diag.get("silhouette_reason"),
        distance_threshold=cfg.distance_threshold,
        distance_threshold_status="provisional",
        keyword_weight=cfg.keyword_weight,
        empty_reason=empty_reason,
        ambiguity_warning=bool(cluster_diag.get("ambiguity_warning")),
        channel_used=features.channel_used,
    )
    return SynthesisResult(
        schema_version=cfg.schema_version,
        topic=snapshot.topic,
        corpus_digest=digest,
        upstream_pipeline_fingerprint=snapshot.fingerprint,
        related_work_policy="unknown",
        effective_config=effective_config(cfg),
        input_inventory=inventory,
        paper_manifest=cards,
        assignments=assignments,
        unassigned=unassigned,
        evidence_registry=registry,
        descriptive_coverage=coverage,
        gap_candidates=gaps,
        diagnostics=diagnostics,
        execution=ExecutionInfo(summary_status="not_requested"),
    )


def _cluster_snapshot(result: SynthesisResult) -> SynthesisResult:
    return result.model_copy(
        update={
            "summaries": [],
            "comparisons": [],
            "execution": ExecutionInfo(summary_status="not_requested"),
        }
    )


def _run_narration(
    result: SynthesisResult,
    cfg: SynthesisConfig,
    cluster_key: str,
    complete_fn: CompleteFn | None,
) -> None:
    edigest = evidence_digest(result.evidence_registry)
    provider = None
    model = "strict_evidence" if cfg.strict_evidence else "none"
    if complete_fn is not None:
        model = "injected"
    elif not cfg.strict_evidence:
        provider = extraction_provider(cfg.prefer_provider)
        model = f"{provider}:{model_for_provider(provider)}" if provider else "none"

    nkey = narration_cache_key(cluster_key, edigest, cfg, model=model, topic=result.topic)
    if cfg.use_cache and not cfg.strict_evidence:
        cached = load_result(narration_path(cfg.cache_dir, nkey))
        if (
            cached is not None
            and cached.corpus_digest == result.corpus_digest
            and cached.execution.summary_status in {"complete", "partial"}
        ):
            result.summaries = cached.summaries
            result.comparisons = cached.comparisons
            result.execution.summary_status = cached.execution.summary_status
            result.execution.provider = cached.execution.provider
            result.execution.model = cached.execution.model
            result.execution.prompt_omitted_units = cached.execution.prompt_omitted_units
            result.execution.narration_batches = cached.execution.narration_batches
            result.execution.successful_batches = cached.execution.successful_batches
            result.execution.failed_cluster_ids = cached.execution.failed_cluster_ids
            result.execution.comparison_status = cached.execution.comparison_status
            result.execution.failure_reason = cached.execution.failure_reason
            result.execution.cache_narration_hit = True
            result.execution.logical_calls = 0
            return

    if cfg.strict_evidence:
        payload = _template_narration(result.assignments, result.paper_manifest, result.evidence_registry)
        summaries, comparisons, errors = apply_narration(
            payload,
            assignments=result.assignments,
            registry=result.evidence_registry,
        )
        kept_s, kept_c, kept_n, rejected_n = keep_validated(summaries, comparisons)
        result.summaries = kept_s
        result.comparisons = kept_c
        result.execution.model = model
        result.execution.summary_status = "complete" if rejected_n == 0 and kept_n else "partial"
        if rejected_n and not kept_n:
            result.execution.summary_status = "failed"
            result.execution.failure_reason = "; ".join(errors[:8]) or "strict_evidence_failed"
        return

    if complete_fn is None and provider is None:
        result.execution.summary_status = "unavailable"
        result.execution.failure_reason = "no_api_key"
        result.execution.model = model
        return

    batches, oversized = build_prompt_batches(
        topic=result.topic,
        assignments=result.assignments,
        cards=result.paper_manifest,
        registry=result.evidence_registry,
        config=cfg,
    )
    result.execution.narration_batches = len(batches)
    result.execution.prompt_omitted_units = sum(batch[2] for batch in batches)
    result.execution.failed_cluster_ids = [cluster.cluster_id for cluster in oversized]
    result.execution.provider = provider
    result.execution.model = model
    if not batches:
        result.execution.summary_status = "failed"
        result.execution.failure_reason = "prompt_exceeds_budget"
        return

    completer = complete_fn or _bound_complete(cfg, provider)
    all_summaries: list[ClusterSummary] = []
    all_comparisons: list[ComparisonClaim] = []
    all_errors: list[str] = []
    rejected_n = 0
    remaining = max(0, int(cfg.max_logical_calls))
    for assignments, user_prompt, _omitted in batches:
        if remaining <= 0:
            result.execution.failed_cluster_ids.extend(cluster.cluster_id for cluster in assignments)
            all_errors.append("logical_call_budget_exhausted")
            continue
        summaries, comparisons, errors, calls = _complete_narration(
            completer,
            user_prompt,
            assignments,
            result.evidence_registry,
            cfg,
            max_calls=min(2, remaining),
        )
        remaining -= calls
        result.execution.logical_calls += calls
        kept_s, kept_c, kept_n, rejected = keep_validated(summaries, comparisons)
        rejected_n += rejected
        all_summaries.extend(kept_s)
        all_comparisons.extend(kept_c)
        all_errors.extend(errors)
        present = {item.cluster_id for item in summaries if any(claim.validation_status == "structurally_validated" for claim in item.claims)}
        missing = [cluster.cluster_id for cluster in assignments if cluster.cluster_id not in present]
        if missing:
            result.execution.failed_cluster_ids.extend(missing)
        else:
            result.execution.successful_batches += 1

    result.execution.failed_cluster_ids = list(dict.fromkeys(result.execution.failed_cluster_ids))
    result.summaries = _ensure_cluster_summaries(all_summaries, result.assignments)
    result.comparisons = all_comparisons

    if len(batches) > 1 and sum(bool(item.claims) for item in result.summaries) > 1:
        comparison_prompt = build_comparison_prompt(
            topic=result.topic,
            summaries=result.summaries,
            registry=result.evidence_registry,
            config=cfg,
        )
        if remaining <= 0 or not prompt_within_budget(comparison_prompt, cfg):
            result.execution.comparison_status = "skipped_budget"
        else:
            comparisons, errors, calls = _complete_comparisons(
                completer, comparison_prompt, result.assignments, result.evidence_registry
            )
            remaining -= calls
            result.execution.logical_calls += calls
            result.comparisons.extend(comparisons)
            all_errors.extend(errors)
            result.execution.comparison_status = "failed" if errors else "complete"

    kept_n = sum(len(item.claims) for item in result.summaries) + len(result.comparisons)
    if result.execution.logical_calls == 0:
        result.execution.summary_status = "failed"
        result.execution.failure_reason = "logical_call_budget_exhausted"
    elif kept_n == 0:
        result.execution.summary_status = "failed"
        result.execution.failure_reason = "; ".join(all_errors[:8]) or "no_valid_claims"
    elif rejected_n == 0 and not result.execution.failed_cluster_ids and not all_errors:
        result.execution.summary_status = "complete"
    else:
        result.execution.summary_status = "partial"
        reasons = all_errors[:8]
        if oversized:
            reasons.insert(0, "oversized_clusters:" + ",".join(cluster.cluster_id for cluster in oversized))
        result.execution.failure_reason = "; ".join(reasons) or "some_clusters_not_summarized"

    if cfg.use_cache and result.execution.summary_status in {"complete", "partial"}:
        save_result(narration_path(cfg.cache_dir, nkey), result)


def _complete_narration(
    completer: CompleteFn,
    user_prompt: str,
    assignments: list[ClusterAssignment],
    registry: EvidenceRegistry,
    cfg: SynthesisConfig,
    *,
    max_calls: int | None = None,
) -> tuple[list[ClusterSummary], list, list[str], int]:
    calls = 0
    last_summaries: list[ClusterSummary] = []
    last_comparisons: list = []
    last_errors: list[str] = []
    message = user_prompt
    budget = max(0, int(cfg.max_logical_calls if max_calls is None else max_calls))
    while calls < budget:
        calls += 1
        try:
            raw = completer(system=SYSTEM_PROMPT, user=message)
            payload = LlmNarration.model_validate(extract_json_object(raw))
        except (LlmError, TimeoutError) as exc:
            last_errors = [f"transport_fail:{exc}"]
            break
        except Exception as exc:
            last_errors = [f"parse_fail:{exc}"]
            message = repair_user_message(user_prompt, last_errors)
            continue
        summaries, comparisons, errors = apply_narration(
            payload, assignments=assignments, registry=registry
        )
        last_summaries, last_comparisons, last_errors = summaries, comparisons, errors
        if not errors:
            break
        message = repair_user_message(user_prompt, errors)
    return last_summaries, last_comparisons, last_errors, calls


def _complete_comparisons(
    completer: CompleteFn,
    user_prompt: str,
    assignments: list[ClusterAssignment],
    registry: EvidenceRegistry,
) -> tuple[list[ComparisonClaim], list[str], int]:
    try:
        raw = completer(system=SYSTEM_PROMPT, user=user_prompt)
        payload = LlmNarration.model_validate(extract_json_object(raw))
    except (LlmError, TimeoutError) as exc:
        return [], [f"comparison_transport_fail:{exc}"], 1
    except Exception as exc:
        return [], [f"comparison_parse_fail:{exc}"], 1

    lookup = registry.lookup()
    comparisons: list[ComparisonClaim] = []
    errors: list[str] = []
    for item in payload.comparisons:
        checked = validate_comparison(
            ComparisonClaim(
                text=item.text,
                cluster_ids=list(item.cluster_ids),
                subject_paper_keys=list(item.subject_paper_keys),
                support_refs=list(item.support_refs),
            ),
            assignments=assignments,
            registry=lookup,
        )
        if checked.validation_status == "structurally_validated":
            comparisons.append(checked)
        else:
            errors.append(f"comparison:{checked.rejection_reason}")
    return comparisons, errors, 1


def _ensure_cluster_summaries(
    summaries: list[ClusterSummary],
    assignments: list[ClusterAssignment],
) -> list[ClusterSummary]:
    by_id = {item.cluster_id: item for item in summaries}
    ordered = []
    for cluster in assignments:
        ordered.append(by_id.get(cluster.cluster_id) or ClusterSummary(cluster_id=cluster.cluster_id, claims=[]))
    return ordered


def _template_narration(
    assignments: list[ClusterAssignment],
    cards: list[PaperCard],
    registry: EvidenceRegistry,
) -> LlmNarration:
    by_key = {card.paper_key: card for card in cards}
    lookup = registry.lookup()
    summaries: list[LlmClusterSummaryIn] = []
    for cluster in assignments:
        claims: list[LlmClaimIn] = []
        if len(cluster.paper_keys) == 1:
            key = cluster.paper_keys[0]
            card = by_key[key]
            refs = [unit.unit_id for unit in lookup.values() if unit.paper_key == key and unit.kind == "method"]
            claims.append(
                LlmClaimIn(
                    kind="subset",
                    text=card.method_text,
                    subject_paper_keys=[key],
                    support_refs=refs[:1],
                )
            )
        else:
            refs: list[str] = []
            subjects: list[str] = []
            chunks: list[str] = []
            for key in cluster.paper_keys:
                card = by_key[key]
                method_refs = [
                    unit.unit_id for unit in lookup.values() if unit.paper_key == key and unit.kind == "method"
                ]
                if not method_refs:
                    continue
                refs.append(method_refs[0])
                subjects.append(key)
                chunks.append(f"{key}: {card.method_text}")
            kind = "shared" if len(subjects) >= 2 else "subset"
            claims.append(
                LlmClaimIn(
                    kind=kind,
                    text="Extracted method statements. " + " ".join(chunks),
                    subject_paper_keys=subjects,
                    support_refs=refs,
                )
            )
        summaries.append(LlmClusterSummaryIn(cluster_id=cluster.cluster_id, claims=claims))
    return LlmNarration(summaries=summaries, comparisons=[])


def _bound_complete(cfg: SynthesisConfig, provider: str | None) -> CompleteFn:
    def _inner(*, system: str, user: str, **_kwargs: Any) -> str:
        return complete(
            system=system,
            user=user,
            json_mode=True,
            temperature=0.0,
            timeout_s=cfg.llm_timeout_s,
            provider=provider,
        )

    return _inner
