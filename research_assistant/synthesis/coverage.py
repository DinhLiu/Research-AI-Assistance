"""Descriptive coverage and conservative gap candidates from extracted fields."""

from __future__ import annotations

from research_assistant.synthesis.types import (
    DescriptiveCoverage,
    EvidenceRegistry,
    GapCandidate,
    PaperCard,
    PaperCoverage,
)


SNAPSHOT_NOTE = (
    "Coverage describes mentions in this extraction snapshot. "
    "Co-occurrence is not an evaluation relation. "
    "Empty lists map to unknown, not to explicitly_not_evaluated."
)


def build_coverage(cards: list[PaperCard]) -> DescriptiveCoverage:
    papers: list[PaperCoverage] = []
    for card in cards:
        dataset_state = "observed" if card.datasets else "unknown"
        metric_state = "observed" if card.metrics else "unknown"
        limitation_state = "observed" if card.limitations else "unknown"
        papers.append(
            PaperCoverage(
                paper_key=card.paper_key,
                dataset_mentions=list(card.datasets),
                metric_mentions=list(card.metrics),
                dataset_state=dataset_state,
                metric_state=metric_state,
                limitation_state=limitation_state,
            )
        )
    return DescriptiveCoverage(papers=papers, snapshot_note=SNAPSHOT_NOTE)


def build_gap_candidates(
    cards: list[PaperCard],
    registry: EvidenceRegistry,
    coverage: DescriptiveCoverage,
) -> list[GapCandidate]:
    units = registry.lookup()
    candidates: list[GapCandidate] = []
    for card in cards:
        for field_index, text in enumerate(card.limitations):
            path = f"limitations[{field_index}]"
            refs = [
                unit.unit_id
                for unit in units.values()
                if unit.paper_key == card.paper_key and unit.field_path == path
            ]
            candidates.append(
                GapCandidate(
                    kind="documented_limitation",
                    statement=text,
                    scope="extraction_snapshot",
                    supporting_refs=refs,
                    inspected_paper_keys=[card.paper_key],
                    unknown_paper_keys=[],
                    verification_needed=True,
                )
            )

    unknown_datasets = [
        item.paper_key for item in coverage.papers if item.dataset_state == "unknown"
    ]
    if cards:
        candidates.append(
            GapCandidate(
                kind="relationship_not_observed",
                statement=(
                    "No supported method-dataset evaluation relations were recorded "
                    "in this extraction snapshot; dataset fields are mentions only."
                ),
                scope="extraction_snapshot",
                supporting_refs=[],
                inspected_paper_keys=[card.paper_key for card in cards],
                unknown_paper_keys=unknown_datasets,
                verification_needed=True,
            )
        )
    return candidates
