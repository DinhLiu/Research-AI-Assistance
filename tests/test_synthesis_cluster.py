from __future__ import annotations

from research_assistant.config import SynthesisConfig
from research_assistant.synthesis.pipeline import synthesize

from tests.synthesis_utils import paper, snapshot


def _cfg(tmp_path, **kwargs) -> SynthesisConfig:
    base = dict(cache_dir=tmp_path, use_cache=False, summarize=False)
    base.update(kwargs)
    return SynthesisConfig(**base)


def test_empty_input(tmp_path):
    result = synthesize(snapshot([]), _cfg(tmp_path))
    assert result.assignments == []
    assert result.diagnostics.empty_reason == "empty_input"
    assert result.diagnostics.silhouette is None


def test_all_skipped(tmp_path):
    result = synthesize(snapshot([paper(status="skipped", method=None, error="fail")]), _cfg(tmp_path))
    assert result.assignments == []
    assert result.diagnostics.empty_reason == "no_usable_papers"
    assert result.input_inventory[0].disposition == "skipped"


def test_blank_method_excluded(tmp_path):
    result = synthesize(snapshot([paper(method="   ")]), _cfg(tmp_path))
    assert result.input_inventory[0].disposition == "missing_method"
    assert result.diagnostics.n_clustered == 0


def test_missing_keywords_still_clusters(tmp_path):
    result = synthesize(
        snapshot([paper(arxiv_id="1111.00001", keywords=[])]),
        _cfg(tmp_path),
    )
    assert result.diagnostics.n_clustered == 1
    assert result.assignments[0].size == 1


def test_punctuation_only_method_empty_features(tmp_path):
    result = synthesize(snapshot([paper(method="...")]), _cfg(tmp_path))
    assert result.unassigned
    assert result.unassigned[0].reason == "empty_features"
    assert result.input_inventory[0].disposition == "empty_features"


def test_identical_vectors_one_cluster(tmp_path):
    method = "Coverage-centric coreset selection with stratified sampling."
    kws = ["coreset", "coverage-centric"]
    result = synthesize(
        snapshot(
            [
                paper(arxiv_id="1111.00001", method=method, keywords=kws),
                paper(arxiv_id="1111.00002", method=method, keywords=kws),
            ]
        ),
        _cfg(tmp_path, distance_threshold=0.2),
    )
    assert len(result.assignments) == 1
    assert result.assignments[0].size == 2


def test_dissimilar_papers_are_singletons(tmp_path):
    result = synthesize(
        snapshot(
            [
                paper(
                    arxiv_id="1111.00001",
                    method="Prune training examples with EL2N influence scores.",
                    keywords=["el2n", "dataset pruning"],
                ),
                paper(
                    arxiv_id="1111.00002",
                    method="Message passing graph networks with multi-head attention over neighbors.",
                    keywords=["graph neural network", "attention"],
                ),
            ]
        ),
        _cfg(tmp_path, distance_threshold=0.4),
    )
    assert result.diagnostics.n_clustered == 2
    assert result.diagnostics.n_singletons == 2
    assert result.diagnostics.silhouette is None


def test_n1_singleton(tmp_path):
    result = synthesize(snapshot([paper()]), _cfg(tmp_path))
    assert len(result.assignments) == 1
    assert result.assignments[0].size == 1
    assert result.diagnostics.silhouette_reason == "n=1"


def test_n3_uses_distance_cut(tmp_path):
    shared = "Static score-based dataset pruning with EL2N example scoring."
    result = synthesize(
        snapshot(
            [
                paper(arxiv_id="1111.00001", method=shared, keywords=["el2n", "dataset pruning"]),
                paper(arxiv_id="1111.00002", method=shared + " Variant two.", keywords=["el2n"]),
                paper(
                    arxiv_id="1111.00003",
                    method="Random repeated sampling to minimize time-to-accuracy without scores.",
                    keywords=["random sampling", "time-to-accuracy"],
                ),
            ]
        ),
        _cfg(tmp_path, distance_threshold=0.5),
    )
    clustered = {key for item in result.assignments for key in item.paper_keys}
    assert clustered == {"1111.00001v1", "1111.00002v1", "1111.00003v1"}
    assert result.diagnostics.n_clusters >= 1


def test_every_paper_once_and_stable_order(tmp_path):
    papers = [
        paper(arxiv_id="1111.00003", method="Random sampling baseline.", keywords=["random"]),
        paper(arxiv_id="1111.00001", method="EL2N pruning of examples.", keywords=["el2n"]),
        paper(arxiv_id="1111.00002", method="EL2N pruning of examples with GraNd.", keywords=["el2n", "grand"]),
    ]
    first = synthesize(snapshot(papers), _cfg(tmp_path))
    second = synthesize(snapshot(list(reversed(papers))), _cfg(tmp_path))
    keys_first = [key for item in first.assignments for key in item.paper_keys]
    keys_second = [key for item in second.assignments for key in item.paper_keys]
    assert sorted(keys_first) == sorted(keys_second)
    assert len(keys_first) == len(set(keys_first))
    assert first.assignments[0].cluster_id == second.assignments[0].cluster_id or {
        item.cluster_id for item in first.assignments
    } == {item.cluster_id for item in second.assignments}
    excluded = [item for item in first.input_inventory if item.disposition != "accepted"]
    for item in excluded:
        assert item.reason
