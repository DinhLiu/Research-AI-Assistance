"""Structural checks, with conservative prose heuristics; not entailment."""
import re

from research_assistant.writing.locales import VI_RANK_PATTERN
from collections import Counter

from research_assistant.arxiv_ids import paper_key
from research_assistant.synthesis.types import SynthesisResult
from research_assistant.synthesis.validate import named_paper_keys
from research_assistant.writing.types import WritingInputError, WrittenPayload


def validate_input(snapshot: SynthesisResult):
    def require(condition, message):
        if not condition:
            raise WritingInputError(message)

    require(snapshot.schema_version == "1", "unsupported_synthesis_schema")
    require(bool(snapshot.corpus_digest.strip()), "missing_corpus_digest")
    keys = [p.paper_key for p in snapshot.paper_manifest]
    ids = [u.unit_id for u in snapshot.evidence_registry.units]
    clusters = [c.cluster_id for c in snapshot.assignments]
    for name, values in (("paper_key", keys), ("unit_id", ids), ("cluster_id", clusters)):
        require(len(values) == len(set(values)), f"duplicate_{name}")
    known, units = set(keys), snapshot.evidence_registry.lookup()
    for card in snapshot.paper_manifest:
        require(card.paper_key == paper_key(card.arxiv_id, card.version), "noncanonical_paper_key")
        require(bool(re.fullmatch(r"(?:\d{4}\.\d{4,5}|[a-z][a-z0-9.-]*/\d{7})v[1-9]\d*", card.paper_key, re.I)), "invalid_arxiv_identifier")
        require(all(ref in units and units[ref].paper_key == card.paper_key for ref in card.evidence_ids), "invalid_card_reference")
    for unit in units.values():
        require(unit.paper_key in known and bool(unit.text.strip()), "invalid_evidence_owner_or_text")
    assigned = []
    for cluster in snapshot.assignments:
        require(bool(cluster.paper_keys) and cluster.size == len(cluster.paper_keys), "invalid_cluster_size")
        require(set(cluster.paper_keys) <= known, "unknown_cluster_paper")
        assigned.extend(cluster.paper_keys)
    require(len(assigned) == len(set(assigned)), "duplicate_membership")
    unassigned = [p.paper_key for p in snapshot.unassigned]
    require(len(unassigned) == len(set(unassigned)), "duplicate_unassigned")
    require(not set(assigned) & set(unassigned), "assigned_and_unassigned")
    require(set(unassigned) <= known, "unknown_unassigned_paper")
    require(set(assigned) | set(unassigned) == known, "missing_membership_disposition")
    for summary in snapshot.summaries:
        require(summary.cluster_id in clusters, "unknown_summary_cluster")
        for claim in summary.claims:
            require(set(claim.support_refs) <= set(ids), "dangling_summary_ref")
    for comparison in snapshot.comparisons:
        require(set(comparison.support_refs) <= set(ids), "dangling_comparison_ref")
        require(set(comparison.cluster_ids) <= set(clusters), "unknown_comparison_cluster")
    for gap in snapshot.gap_candidates:
        require(set(gap.supporting_refs) <= set(ids), "dangling_gap_ref")


_UNSAFE = re.compile(r"https?://|www\.|\[[^\]]+\]|<[^>]+>|\n\s*#", re.I)
_RANK = re.compile(r"\b(outperform\w*|state.of.the.art|best performance|superior|no prior work|never studied|first ever)\b|" + VI_RANK_PATTERN, re.I)
_NUMERIC = re.compile(r"\d+(?:\.\d+)?\s*(?:%|pp\b|accuracy\b)", re.I)


def validate_written(payload: WrittenPayload, slots):
    allowed = {slot.claim_id: slot for slot in slots}
    counts = Counter(item.claim_id for item in payload.claims)
    valid, errors = {}, []
    for item in payload.claims:
        slot = allowed.get(item.claim_id)
        reason = None
        if slot is None:
            reason = "unknown_claim_id"
        elif counts[item.claim_id] != 1:
            reason = "duplicate_claim_id"
        elif not item.text.strip() or len(item.text) > 8000:
            reason = "empty_or_oversized_text"
        elif _UNSAFE.search(item.text) or _RANK.search(item.text) or _NUMERIC.search(item.text):
            reason = "unsupported_prose_or_citation"
        elif not named_paper_keys(item.text, set(slot.subject_paper_keys)) <= set(slot.subject_paper_keys):
            reason = "unsupported_named_paper"
        if reason:
            errors.append(f"{item.claim_id}:{reason}")
        else:
            valid[item.claim_id] = slot.model_copy(update={"text": item.text.strip(), "generation": "llm"})
    errors.extend(f"{key}:missing_claim" for key in allowed if key not in valid)
    return valid, list(dict.fromkeys(errors))
