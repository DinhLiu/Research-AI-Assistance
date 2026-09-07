from __future__ import annotations

import json

from research_assistant.synthesis.cards import build_registry_and_cards
from research_assistant.synthesis.cluster import cluster_id_for
from research_assistant.synthesis.types import (
    ClusterAssignment,
    ComparisonClaim,
    LlmClaimIn,
    LlmClusterSummaryIn,
    LlmNarration,
    SynthesisClaim,
)
from research_assistant.synthesis.validate import apply_narration, validate_claim, validate_comparison

from tests.synthesis_utils import paper


def _cluster(keys: list[str], version: str = "1") -> ClusterAssignment:
    return ClusterAssignment(
        cluster_id=cluster_id_for(keys, version),
        label="test",
        paper_keys=keys,
        size=len(keys),
    )


def test_unknown_and_cross_cluster_refs_rejected():
    papers = [
        paper(arxiv_id="1111.00001", method="EL2N pruning.", keywords=["el2n"]),
        paper(arxiv_id="1111.00002", method="Random sampling.", keywords=["random"]),
    ]
    registry, _cards = build_registry_and_cards(papers)
    lookup = registry.lookup()
    a_method = next(unit.unit_id for unit in registry.units if unit.paper_key == "1111.00001v1" and unit.kind == "method")
    b_method = next(unit.unit_id for unit in registry.units if unit.paper_key == "1111.00002v1" and unit.kind == "method")
    cluster = _cluster(["1111.00001v1"])
    claim = SynthesisClaim(
        text="Both methods use EL2N.",
        kind="shared",
        subject_paper_keys=["1111.00001v1", "1111.00002v1"],
        support_refs=[a_method, b_method],
    )
    checked = validate_claim(
        claim,
        permitted_keys={"1111.00001v1"},
        cluster_keys={"1111.00001v1"},
        registry=lookup,
        permitted_kinds=frozenset({"method", "method_keyword", "problem", "contribution"}),
    )
    assert checked.validation_status == "rejected"
    assert "unknown_subjects" in (checked.rejection_reason or "")

    bad_ref = validate_claim(
        SynthesisClaim(
            text="Uses EL2N.",
            kind="subset",
            subject_paper_keys=["1111.00001v1"],
            support_refs=["e_does_not_exist"],
        ),
        permitted_keys={"1111.00001v1"},
        cluster_keys={"1111.00001v1"},
        registry=lookup,
        permitted_kinds=frozenset({"method"}),
    )
    assert "unknown_ref" in (bad_ref.rejection_reason or "")

    cross = validate_claim(
        SynthesisClaim(
            text="Uses EL2N.",
            kind="subset",
            subject_paper_keys=["1111.00001v1"],
            support_refs=[b_method],
        ),
        permitted_keys={"1111.00001v1"},
        cluster_keys={"1111.00001v1"},
        registry=lookup,
        permitted_kinds=frozenset({"method"}),
    )
    assert "cross_cluster_ref" in (cross.rejection_reason or "")


def test_named_unsupported_subject_rejected():
    papers = [paper(arxiv_id="1111.00001", method="EL2N pruning.", keywords=["el2n"])]
    registry, _cards = build_registry_and_cards(papers)
    method_id = next(unit.unit_id for unit in registry.units if unit.kind == "method")
    checked = validate_claim(
        SynthesisClaim(
            text="Unlike 1111.00099 this method uses EL2N.",
            kind="subset",
            subject_paper_keys=["1111.00001v1"],
            support_refs=[method_id],
        ),
        permitted_keys={"1111.00001v1"},
        cluster_keys={"1111.00001v1"},
        registry=registry.lookup(),
        permitted_kinds=frozenset({"method"}),
    )
    assert checked.validation_status == "rejected"
    assert "named_unsupported_subjects" in (checked.rejection_reason or "")


def test_invented_assignments_rejected_and_mixed_claims():
    papers = [
        paper(arxiv_id="1111.00001", method="EL2N pruning of examples.", keywords=["el2n"]),
        paper(arxiv_id="1111.00002", method="EL2N pruning with GraNd.", keywords=["el2n", "grand"]),
    ]
    registry, _cards = build_registry_and_cards(papers)
    keys = ["1111.00001v1", "1111.00002v1"]
    cluster = _cluster(keys)
    method_ids = [unit.unit_id for unit in registry.units if unit.kind == "method"]
    payload = LlmNarration(
        summaries=[
            LlmClusterSummaryIn(
                cluster_id=cluster.cluster_id,
                claims=[
                    LlmClaimIn(
                        kind="shared",
                        text="Both papers describe EL2N-based pruning.",
                        subject_paper_keys=keys,
                        support_refs=method_ids,
                    ),
                    LlmClaimIn(
                        kind="shared",
                        text="Unsupported claim.",
                        subject_paper_keys=keys,
                        support_refs=[],
                    ),
                ],
            )
        ]
    )
    summaries, _comparisons, errors = apply_narration(
        payload, assignments=[cluster], registry=registry
    )
    statuses = [claim.validation_status for claim in summaries[0].claims]
    assert "structurally_validated" in statuses
    assert "rejected" in statuses
    assert errors

    raw = json.dumps(
        {
            "summaries": [],
            "comparisons": [],
            "assignments": [{"cluster_id": "x", "paper_keys": ["1111.00001v1"]}],
        }
    )
    from pydantic import ValidationError
    from research_assistant.extraction.validate import extract_json_object
    from research_assistant.synthesis.types import LlmNarration as Narration

    try:
        Narration.model_validate(extract_json_object(raw))
        raised = False
    except ValidationError:
        raised = True
    assert raised


def test_numeric_comparison_rejected():
    papers = [
        paper(arxiv_id="1111.00001", method="EL2N pruning.", keywords=["el2n"], results=["Accuracy 90%"]),
        paper(arxiv_id="1111.00002", method="Random sampling.", keywords=["random"], results=["Accuracy 80%"]),
    ]
    registry, _cards = build_registry_and_cards(papers)
    a, b = _cluster(["1111.00001v1"]), _cluster(["1111.00002v1"])
    result_ids = [unit.unit_id for unit in registry.units if unit.kind == "result"]
    checked = validate_comparison(
        ComparisonClaim(
            text="1111.00001v1 is 10% accuracy better than 1111.00002v1.",
            cluster_ids=[a.cluster_id, b.cluster_id],
            subject_paper_keys=["1111.00001v1", "1111.00002v1"],
            support_refs=result_ids,
        ),
        assignments=[a, b],
        registry=registry.lookup(),
    )
    assert checked.validation_status == "rejected"
    assert "numeric_comparison_unsupported" in (checked.rejection_reason or "")
