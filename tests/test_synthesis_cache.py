from __future__ import annotations

import json

from research_assistant.config import SynthesisConfig
from research_assistant.synthesis.pipeline import synthesize

from tests.synthesis_utils import paper, snapshot


def _cfg(tmp_path, **kwargs) -> SynthesisConfig:
    base = dict(cache_dir=tmp_path, use_cache=True, summarize=False)
    base.update(kwargs)
    return SynthesisConfig(**base)


def test_same_upstream_fingerprint_different_records_miss_cache(tmp_path):
    first = snapshot([paper(arxiv_id="1111.00001", method="EL2N pruning.", keywords=["el2n"])])
    second = snapshot([paper(arxiv_id="1111.00002", method="Random sampling.", keywords=["random"])])
    assert first.fingerprint == second.fingerprint
    a = synthesize(first, _cfg(tmp_path))
    b = synthesize(second, _cfg(tmp_path))
    assert a.corpus_digest != b.corpus_digest
    assert b.execution.cache_cluster_hit is False
    assert a.paper_manifest[0].paper_key != b.paper_manifest[0].paper_key


def test_changed_evidence_invalidates_narration(tmp_path):
    calls = {"n": 0}

    def fake(*, system: str, user: str, **_kwargs) -> str:
        calls["n"] += 1
        return json.dumps({"summaries": [], "comparisons": []})

    base = paper(arxiv_id="1111.00001", method="EL2N pruning.", keywords=["el2n"])
    synthesize(snapshot([base]), _cfg(tmp_path, summarize=True), complete_fn=fake)
    first_calls = calls["n"]
    changed = paper(
        arxiv_id="1111.00001",
        method="EL2N pruning with a rewritten method sentence.",
        keywords=["el2n"],
    )
    synthesize(snapshot([changed]), _cfg(tmp_path, summarize=True), complete_fn=fake)
    assert calls["n"] > first_calls


def test_summarize_modes_are_distinct(tmp_path):
    extracted = snapshot([paper(method="EL2N pruning.", keywords=["el2n"])])
    offline = synthesize(extracted, _cfg(tmp_path, summarize=False))
    assert offline.execution.summary_status == "not_requested"

    def fake(*, system: str, user: str, **_kwargs) -> str:
        return json.dumps({"summaries": [], "comparisons": []})

    online = synthesize(extracted, _cfg(tmp_path, summarize=True), complete_fn=fake)
    assert online.execution.summary_status in {"failed", "partial", "complete"}
    assert online.execution.summary_status != "not_requested"


def test_corrupt_cache_is_recoverable(tmp_path):
    extracted = snapshot([paper(method="EL2N pruning.", keywords=["el2n"])])
    first = synthesize(extracted, _cfg(tmp_path))
    cache_files = list((tmp_path / "cluster").glob("*.json"))
    assert cache_files
    cache_files[0].write_text("{not json", encoding="utf-8")
    second = synthesize(extracted, _cfg(tmp_path))
    assert second.corpus_digest == first.corpus_digest
    assert second.execution.cache_cluster_hit is False
    assert second.assignments
