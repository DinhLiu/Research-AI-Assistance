from __future__ import annotations

from research_assistant.config import SynthesisConfig
from research_assistant.synthesis.pipeline import synthesize

from tests.synthesis_utils import paper, snapshot


def test_missing_dataset_is_unknown_not_not_evaluated(tmp_path):
    result = synthesize(
        snapshot(
            [
                paper(
                    arxiv_id="1111.00001",
                    method="EL2N pruning evaluated conceptually on CIFAR-10 in the prose.",
                    keywords=["el2n"],
                    datasets=[],
                )
            ]
        ),
        SynthesisConfig(cache_dir=tmp_path, use_cache=False, summarize=False),
    )
    coverage = result.descriptive_coverage.papers[0]
    assert coverage.dataset_state == "unknown"
    assert coverage.dataset_mentions == []
    assert not any(
        "did not evaluate" in item.statement.lower() or "did not cover" in item.statement.lower()
        for item in result.gap_candidates
    )


def test_mentions_do_not_fabricate_evaluation_pairs(tmp_path):
    result = synthesize(
        snapshot(
            [
                paper(
                    arxiv_id="1111.00001",
                    method="EL2N pruning.",
                    keywords=["el2n"],
                    datasets=["CIFAR-10"],
                    metrics=["accuracy"],
                )
            ]
        ),
        SynthesisConfig(cache_dir=tmp_path, use_cache=False, summarize=False),
    )
    dump = result.model_dump()
    assert "evaluated_on" not in dump
    coverage = result.descriptive_coverage.papers[0]
    assert coverage.dataset_state == "observed"
    assert coverage.dataset_mentions == ["CIFAR-10"]
    note = result.descriptive_coverage.snapshot_note.lower()
    assert "co-occurrence is not an evaluation relation" in note
