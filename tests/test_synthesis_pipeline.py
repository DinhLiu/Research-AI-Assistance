from __future__ import annotations

import json

from research_assistant.config import SynthesisConfig
from research_assistant.synthesis.cluster import cluster_id_for
from research_assistant.synthesis.pipeline import synthesize

from tests.synthesis_utils import paper, snapshot


def _cfg(tmp_path, **kwargs) -> SynthesisConfig:
    base = dict(cache_dir=tmp_path, use_cache=False, summarize=True, max_logical_calls=2)
    base.update(kwargs)
    return SynthesisConfig(**base)


def test_missing_key_keeps_deterministic_output(tmp_path, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    result = synthesize(snapshot([paper(method="EL2N pruning.", keywords=["el2n"])]), _cfg(tmp_path))
    assert result.assignments
    assert result.execution.summary_status == "unavailable"
    assert result.execution.failure_reason == "no_api_key"
    assert result.summaries == []


def test_timeout_preserves_clusters(tmp_path):
    def boom(*, system: str, user: str, **_kwargs) -> str:
        raise TimeoutError("deadline")

    result = synthesize(
        snapshot([paper(method="EL2N pruning.", keywords=["el2n"])]),
        _cfg(tmp_path),
        complete_fn=boom,
    )
    assert result.assignments
    assert result.execution.summary_status == "failed"
    assert result.execution.logical_calls == 1


def test_malformed_json_then_repair(tmp_path):
    extracted = snapshot(
        [paper(arxiv_id="1111.00001", method="EL2N pruning of examples.", keywords=["el2n"])]
    )
    dry = synthesize(extracted, _cfg(tmp_path, summarize=False))
    cluster = dry.assignments[0]
    method_id = next(unit.unit_id for unit in dry.evidence_registry.units if unit.kind == "method")
    calls = {"n": 0}

    def fake(*, system: str, user: str, **_kwargs) -> str:
        calls["n"] += 1
        if calls["n"] == 1:
            return "not json at all"
        return json.dumps(
            {
                "summaries": [
                    {
                        "cluster_id": cluster.cluster_id,
                        "claims": [
                            {
                                "kind": "subset",
                                "text": "EL2N pruning of examples.",
                                "subject_paper_keys": cluster.paper_keys,
                                "support_refs": [method_id],
                            }
                        ],
                    }
                ],
                "comparisons": [],
            }
        )

    result = synthesize(extracted, _cfg(tmp_path), complete_fn=fake)
    assert calls["n"] == 2
    assert result.execution.summary_status in {"complete", "partial"}
    assert result.summaries[0].claims
    assert result.summaries[0].claims[0].validation_status == "structurally_validated"


def test_repair_exhaustion_and_budget(tmp_path):
    def always_bad(*, system: str, user: str, **_kwargs) -> str:
        return "<<<"

    result = synthesize(
        snapshot([paper(method="EL2N pruning.", keywords=["el2n"])]),
        _cfg(tmp_path, max_logical_calls=2),
        complete_fn=always_bad,
    )
    assert result.execution.logical_calls == 2
    assert result.execution.summary_status == "failed"
    assert result.assignments


def test_oversized_prompt_skips_llm(tmp_path):
    called = {"n": 0}

    def fake(*, system: str, user: str, **_kwargs) -> str:
        called["n"] += 1
        raise AssertionError("should not be called")

    result = synthesize(
        snapshot([paper(method="EL2N pruning.", keywords=["el2n"])]),
        _cfg(tmp_path, max_prompt_chars=20),
        complete_fn=fake,
    )
    assert called["n"] == 0
    assert result.execution.failure_reason == "prompt_exceeds_budget"
    assert result.assignments


def test_strict_evidence_templates(tmp_path):
    result = synthesize(
        snapshot([paper(method="EL2N pruning of examples.", keywords=["el2n"])]),
        _cfg(tmp_path, strict_evidence=True),
    )
    assert result.execution.logical_calls == 0
    assert result.summaries
    assert result.summaries[0].claims
    assert result.summaries[0].claims[0].text.startswith("EL2N")
    assert result.summaries[0].claims[0].validation_status == "structurally_validated"


def test_cluster_id_stable_for_same_members():
    assert cluster_id_for(["b", "a"], "1") == cluster_id_for(["a", "b"], "1")
    assert cluster_id_for(["a"], "1") != cluster_id_for(["a"], "2")


def test_transport_error_is_not_content_repair(tmp_path):
    from research_assistant.llm.client import LlmError
    calls = []
    def fail(**kwargs):
        calls.append(1)
        raise LlmError("llm_http_401")
    result = synthesize(snapshot([paper()]), _cfg(tmp_path), complete_fn=fail)
    assert len(calls) == 1
    assert result.execution.summary_status == "failed"
    assert "transport_fail" in result.execution.failure_reason


def test_zero_logical_budget_never_calls(tmp_path):
    def fail(**kwargs):
        raise AssertionError("budget zero")
    result = synthesize(snapshot([paper()]), _cfg(tmp_path, max_logical_calls=0), complete_fn=fail)
    assert result.execution.logical_calls == 0
