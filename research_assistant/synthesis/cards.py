"""Inventory, paper cards, and the immutable evidence registry."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict

from research_assistant.arxiv_ids import format_version, normalize_arxiv_id, paper_key, split_arxiv_id, version_number
from research_assistant.extraction.types import Claim, Evidence, ExtractedPaper, Mention
from research_assistant.synthesis.types import (
    Disposition,
    EvidenceKind,
    EvidenceRegistry,
    EvidenceUnit,
    InventoryItem,
    PaperCard,
    SynthesisInputError,
)


def record_paper_key(paper: ExtractedPaper) -> str:
    core, parsed = split_arxiv_id(paper.arxiv_id)
    return paper_key(core, paper.version or parsed)


def consumed_payload(paper: ExtractedPaper) -> dict:
    def claim_dump(claim: Claim | None) -> dict | None:
        if claim is None:
            return None
        return {"text": claim.text, "evidence": [item.model_dump() for item in claim.evidence]}

    def mention_dump(items: list[Mention]) -> list[dict]:
        return [{"value": item.value, "evidence": item.evidence.model_dump()} for item in items]

    return {
        "arxiv_id": normalize_arxiv_id(paper.arxiv_id),
        "version": format_version(paper.version or split_arxiv_id(paper.arxiv_id)[1]),
        "status": paper.status,
        "title": paper.title,
        "source_kind": paper.source_kind,
        "confidence": paper.confidence,
        "problem": claim_dump(paper.problem),
        "method": claim_dump(paper.method),
        "results": [claim_dump(item) for item in paper.results],
        "limitations": [claim_dump(item) for item in paper.limitations],
        "contributions": [claim_dump(item) for item in paper.contributions],
        "datasets": mention_dump(paper.datasets),
        "metrics": mention_dump(paper.metrics),
        "method_keywords": mention_dump(paper.method_keywords),
    }


def payload_hash(paper: ExtractedPaper) -> str:
    raw = json.dumps(consumed_payload(paper), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def build_inventory(
    records: list[ExtractedPaper],
    *,
    strict_ok: bool = False,
) -> tuple[list[InventoryItem], list[ExtractedPaper]]:
    by_key: dict[str, list[tuple[int, ExtractedPaper, str]]] = defaultdict(list)
    for index, paper in enumerate(records):
        key = record_paper_key(paper)
        by_key[key].append((index, paper, payload_hash(paper)))

    for key, group in by_key.items():
        hashes = {item[2] for item in group}
        if len(hashes) > 1:
            raise SynthesisInputError(f"conflicting records for {key}")

    duplicate_indices: set[int] = set()
    for group in by_key.values():
        ordered = sorted(group, key=lambda item: item[0])
        for index, _paper, _digest in ordered[1:]:
            duplicate_indices.add(index)

    dispositions: dict[int, tuple[Disposition, str]] = {}
    eligible_by_core: dict[str, list[tuple[int, ExtractedPaper, str, int]]] = defaultdict(list)

    for index, paper in enumerate(records):
        key = record_paper_key(paper)
        if index in duplicate_indices:
            dispositions[index] = ("duplicate", "identical record already kept")
            continue
        if paper.status == "skipped":
            dispositions[index] = ("skipped", paper.error or "extraction skipped")
            continue
        if strict_ok and paper.status != "ok":
            dispositions[index] = ("strict_ok_excluded", f"status={paper.status}")
            continue
        if paper.status not in {"ok", "degraded"}:
            dispositions[index] = ("skipped", f"status={paper.status}")
            continue
        method_text = (paper.method.text if paper.method else "").strip()
        if not method_text:
            dispositions[index] = ("missing_method", "blank or missing method text")
            continue
        core = normalize_arxiv_id(paper.arxiv_id)
        version = format_version(paper.version or split_arxiv_id(paper.arxiv_id)[1])
        eligible_by_core[core].append((index, paper, key, version_number(version)))

    accepted_indices: set[int] = set()
    for group in eligible_by_core.values():
        best_version = max(item[3] for item in group)
        winners = [item for item in group if item[3] == best_version]
        winner = sorted(winners, key=lambda item: item[0])[0]
        accepted_indices.add(winner[0])
        for item in group:
            if item[0] != winner[0]:
                dispositions[item[0]] = ("superseded_version", f"kept {winner[2]}")

    inventory: list[InventoryItem] = []
    accepted: list[ExtractedPaper] = []
    for index, paper in enumerate(records):
        key = record_paper_key(paper)
        core = normalize_arxiv_id(paper.arxiv_id)
        version = format_version(paper.version or split_arxiv_id(paper.arxiv_id)[1])
        if index in accepted_indices:
            disposition, reason = "accepted", "eligible for clustering"
            accepted.append(paper)
        else:
            disposition, reason = dispositions.get(index, ("skipped", "not selected"))
        inventory.append(
            InventoryItem(
                index=index,
                paper_key=key,
                arxiv_id=core,
                version=version,
                status=paper.status,
                disposition=disposition,
                reason=reason,
                title=paper.title,
            )
        )
    accepted.sort(key=lambda paper: record_paper_key(paper))
    return inventory, accepted


def mark_empty_features(inventory: list[InventoryItem], empty_keys: set[str]) -> None:
    for item in inventory:
        if item.disposition == "accepted" and item.paper_key in empty_keys:
            item.disposition = "empty_features"
            item.reason = "zero or empty feature row"


def _unit_id(paper_key_value: str, field_path: str, text: str) -> str:
    digest = hashlib.sha256(
        f"{paper_key_value}|{field_path}|{text}".encode("utf-8")
    ).hexdigest()[:12]
    return f"e_{digest}"


def _units_from_claim(
    paper_key_value: str,
    field_path: str,
    kind: EvidenceKind,
    claim: Claim | None,
) -> list[EvidenceUnit]:
    if claim is None:
        return []
    primary = claim.evidence[0]
    return [
        EvidenceUnit(
            unit_id=_unit_id(paper_key_value, field_path, claim.text),
            paper_key=paper_key_value,
            field_path=field_path,
            kind=kind,
            text=claim.text,
            quote=primary.quote,
            section=primary.section,
            source_kind=primary.source_kind,
            start_char=primary.start_char,
            end_char=primary.end_char,
        )
    ]


def _units_from_mention(
    paper_key_value: str,
    field_path: str,
    kind: EvidenceKind,
    mention: Mention,
) -> list[EvidenceUnit]:
    evidence: Evidence = mention.evidence
    return [
        EvidenceUnit(
            unit_id=_unit_id(paper_key_value, field_path, mention.value),
            paper_key=paper_key_value,
            field_path=field_path,
            kind=kind,
            text=mention.value,
            quote=evidence.quote,
            section=evidence.section,
            source_kind=evidence.source_kind,
            start_char=evidence.start_char,
            end_char=evidence.end_char,
        )
    ]


def build_registry_and_cards(
    papers: list[ExtractedPaper],
) -> tuple[EvidenceRegistry, list[PaperCard]]:
    units: list[EvidenceUnit] = []
    cards: list[PaperCard] = []
    for paper in papers:
        key = record_paper_key(paper)
        paper_units: list[EvidenceUnit] = []
        paper_units.extend(_units_from_claim(key, "problem", "problem", paper.problem))
        paper_units.extend(_units_from_claim(key, "method", "method", paper.method))
        for index, claim in enumerate(paper.results):
            paper_units.extend(_units_from_claim(key, f"results[{index}]", "result", claim))
        for index, claim in enumerate(paper.limitations):
            paper_units.extend(_units_from_claim(key, f"limitations[{index}]", "limitation", claim))
        for index, claim in enumerate(paper.contributions):
            paper_units.extend(_units_from_claim(key, f"contributions[{index}]", "contribution", claim))
        for index, mention in enumerate(paper.method_keywords):
            paper_units.extend(_units_from_mention(key, f"method_keywords[{index}]", "method_keyword", mention))
        for index, mention in enumerate(paper.datasets):
            paper_units.extend(_units_from_mention(key, f"datasets[{index}]", "dataset", mention))
        for index, mention in enumerate(paper.metrics):
            paper_units.extend(_units_from_mention(key, f"metrics[{index}]", "metric", mention))
        units.extend(paper_units)
        cards.append(
            PaperCard(
                paper_key=key,
                arxiv_id=normalize_arxiv_id(paper.arxiv_id),
                version=format_version(paper.version or split_arxiv_id(paper.arxiv_id)[1]),
                title=paper.title,
                status=paper.status,
                source_kind=paper.source_kind,
                confidence=paper.confidence,
                problem_text=paper.problem.text if paper.problem else None,
                method_text=paper.method.text if paper.method else "",
                method_keywords=[item.value for item in paper.method_keywords],
                datasets=[item.value for item in paper.datasets],
                metrics=[item.value for item in paper.metrics],
                results=[item.text for item in paper.results],
                limitations=[item.text for item in paper.limitations],
                contributions=[item.text for item in paper.contributions],
                evidence_ids=[unit.unit_id for unit in paper_units],
            )
        )
    cards.sort(key=lambda card: card.paper_key)
    return EvidenceRegistry(units=units), cards
