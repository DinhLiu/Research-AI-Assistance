from __future__ import annotations

import json

import pytest

from research_assistant.config import ExtractionConfig
from research_assistant.extraction.fetch import FetchError
from research_assistant.extraction.fingerprint import pipeline_fingerprint
from research_assistant.extraction.pipeline import extract_one, extract_papers
from research_assistant.extraction.types import PaperDocument, PromptDocument, Section
from research_assistant.llm.client import RateLimitError
from research_assistant.retrieval.types import PaperHit


def _hit() -> PaperHit:
    return PaperHit(
        rank=1,
        row_id=-1,
        arxiv_id="0000.00001",
        title="Tiny Coreset Selection",
        abstract="We select a coreset of CIFAR-10 using EL2N scores.",
        latest_version="v1",
    )


def _ok_json(arxiv_id: str = "0000.00001") -> str:
    return json.dumps(
        {
            "arxiv_id": arxiv_id,
            "method": {
                "text": "Select a CIFAR-10 coreset with EL2N.",
                "evidence_ids": ["M1"],
            },
            "method_keywords": [{"value": "EL2N", "evidence_ids": ["M1"]}],
        }
    )


def _ok_batch(ids: list[str]) -> str:
    return json.dumps({"papers": [json.loads(_ok_json(aid)) for aid in ids]})


def _hit_n(n: int) -> PaperHit:
    return PaperHit(
        rank=n,
        row_id=-1,
        arxiv_id=f"0000.{n:05d}",
        title="Tiny Coreset Selection",
        abstract="We select a coreset of CIFAR-10 using EL2N scores.",
        latest_version="v1",
    )


def _cfg(tmp_path, **kwargs) -> ExtractionConfig:
    base = dict(
        cache_dir=tmp_path,
        allow_abstract_fallback=True,
        llm_concurrency=1,
        arxiv_delay_s=0.0,
        skip_llm_if_confident=False,
        llm_min_interval_s=0.0,
        max_llm_calls_per_run=50,
        max_retries=0,
    )
    base.update(kwargs)
    return ExtractionConfig(**base)


@pytest.fixture
def offline_fetch(monkeypatch):
    def fail(*_args, **_kwargs):
        raise FetchError("offline test")

    monkeypatch.setattr("research_assistant.extraction.pipeline.fetch_eprint", fail)
    monkeypatch.setattr("research_assistant.extraction.pipeline.fetch_pdf", fail)


def test_pipeline_skips_when_llm_always_fails(tmp_path, offline_fetch):
    def boom(*, system: str, user: str) -> str:
        raise RuntimeError("no model")

    result = extract_papers(
        [_hit()],
        _cfg(tmp_path, llm_batch_size=1),
        complete_fn=boom,
    )
    assert result.records[0].status == "skipped"
    assert result.metrics["skip_rate"] == 1.0
    assert result.metrics["llm_calls"] == 1


def test_abstract_fallback_succeeds_without_fetch(tmp_path, offline_fetch):
    calls = {"n": 0}

    def fake(*, system: str, user: str) -> str:
        calls["n"] += 1
        return _ok_json()

    result = extract_papers(
        [_hit()],
        _cfg(tmp_path, llm_batch_size=1),
        complete_fn=fake,
    )
    paper = result.records[0]
    assert paper.status in {"ok", "degraded"}
    assert paper.method is not None
    assert paper.method.evidence[0].quote
    assert "EL2N" in [m.value for m in paper.method_keywords]
    assert any(m.value == "CIFAR-10" for m in paper.datasets)
    assert calls["n"] == 1
    assert result.metrics["llm_calls"] == 1


def test_cache_hit_does_not_recall_llm(tmp_path, offline_fetch):
    calls = {"n": 0}

    def fake(*, system: str, user: str) -> str:
        calls["n"] += 1
        return _ok_json()

    cfg = _cfg(tmp_path, llm_batch_size=1)
    first = extract_papers([_hit()], cfg, complete_fn=fake)
    second = extract_papers([_hit()], cfg, complete_fn=fake)
    assert first.records[0].status == second.records[0].status
    assert second.metrics["cache_hits"] == 1
    assert second.metrics["llm_calls"] == 0
    assert calls["n"] == 1


def test_fingerprint_changes_with_model_and_policy():
    a = pipeline_fingerprint(ExtractionConfig(), "gemini:gemini-2.5-flash")
    b = pipeline_fingerprint(ExtractionConfig(), "openai:gpt-4o-mini")
    c = pipeline_fingerprint(ExtractionConfig(include_related_work=True), "gemini:gemini-2.5-flash")
    d = pipeline_fingerprint(ExtractionConfig(llm_batch_size=1), "gemini:gemini-2.5-flash")
    e = pipeline_fingerprint(ExtractionConfig(max_llm_calls_per_run=1), "gemini:gemini-2.5-flash")
    f = pipeline_fingerprint(ExtractionConfig(llm_min_interval_s=0.0), "gemini:gemini-2.5-flash")
    assert a != b
    assert a != c
    assert a != d
    assert a == e
    assert a == f


def test_microbatch_one_call_for_five_papers(tmp_path, offline_fetch):
    calls = {"n": 0}

    def fake(*, system: str, user: str) -> str:
        calls["n"] += 1
        return _ok_batch([f"0000.{i:05d}" for i in range(1, 6)])

    result = extract_papers(
        [_hit_n(i) for i in range(1, 6)],
        _cfg(tmp_path, llm_batch_size=5),
        complete_fn=fake,
    )
    assert calls["n"] == 1
    assert result.metrics["llm_calls"] == 1
    assert all(p.status in {"ok", "degraded"} for p in result.records)


def test_one_paper_failure_does_not_crash(tmp_path, offline_fetch):
    def fake(*, system: str, user: str) -> str:
        if "0000.00002" in user or "Second" in user:
            return "not json"
        return _ok_json()

    papers = [
        _hit(),
        PaperHit(
            rank=2,
            row_id=-1,
            arxiv_id="0000.00002",
            title="Second",
            abstract="Second abstract without enough method detail.",
            latest_version="v1",
        ),
    ]
    result = extract_papers(
        papers,
        _cfg(tmp_path, llm_batch_size=4),
        complete_fn=fake,
    )
    assert len(result.records) == 2
    assert {r.status for r in result.records} <= {"ok", "degraded", "skipped"}
    assert any(r.status == "skipped" for r in result.records)


def test_max_retries_zero_does_not_retry_evidence_fail(tmp_path, offline_fetch):
    calls = {"n": 0}

    def fake(*, system: str, user: str) -> str:
        calls["n"] += 1
        return json.dumps(
            {
                "papers": [
                    json.loads(_ok_json("0000.00001")),
                    {
                        "arxiv_id": "0000.00002",
                        "method": {
                            "text": "Does not cite a real snippet.",
                            "evidence_ids": ["M999"],
                        },
                    },
                ]
            }
        )

    result = extract_papers(
        [_hit_n(1), _hit_n(2)],
        _cfg(tmp_path, llm_batch_size=5, max_retries=0),
        complete_fn=fake,
    )
    assert calls["n"] == 1
    assert result.metrics["llm_calls"] == 1
    assert result.records[0].status in {"ok", "degraded"}
    assert result.records[1].status == "skipped"
    assert any(a.outcome == "evidence_fail" for a in result.records[1].attempts)


def test_max_retries_retries_evidence_fail_once(tmp_path, offline_fetch):
    calls = {"n": 0}

    def fake(*, system: str, user: str) -> str:
        calls["n"] += 1
        if calls["n"] == 1:
            return json.dumps(
                {
                    "papers": [
                        json.loads(_ok_json("0000.00001")),
                        {
                            "arxiv_id": "0000.00002",
                            "method": {
                                "text": "Does not cite a real snippet.",
                                "evidence_ids": ["M999"],
                            },
                        },
                    ]
                }
            )
        return _ok_json("0000.00002")

    result = extract_papers(
        [_hit_n(1), _hit_n(2)],
        _cfg(tmp_path, llm_batch_size=5, max_retries=1),
        complete_fn=fake,
    )
    assert calls["n"] == 2
    assert result.metrics["llm_calls"] == 2
    assert all(p.status in {"ok", "degraded"} for p in result.records)


def test_json_parse_fail_splits_when_budget_allows(tmp_path, offline_fetch):
    calls = {"n": 0}

    def fake(*, system: str, user: str) -> str:
        calls["n"] += 1
        if calls["n"] == 1:
            return "truncated {"
        if "0000.00001" in user and "0000.00002" not in user:
            return _ok_json("0000.00001")
        if "0000.00002" in user and "0000.00001" not in user:
            return _ok_json("0000.00002")
        return "truncated {"

    result = extract_papers(
        [_hit_n(1), _hit_n(2)],
        _cfg(tmp_path, llm_batch_size=5),
        complete_fn=fake,
    )
    assert calls["n"] == 3
    assert result.metrics["llm_calls"] == 3
    assert all(p.status in {"ok", "degraded"} for p in result.records)


def test_rate_limit_does_not_split_batch(tmp_path, offline_fetch):
    calls = {"n": 0}

    def fake(*, system: str, user: str) -> str:
        calls["n"] += 1
        raise RateLimitError("LLM HTTP 429: quota")

    result = extract_papers(
        [_hit_n(1), _hit_n(2)],
        _cfg(tmp_path, llm_batch_size=5),
        complete_fn=fake,
    )
    assert calls["n"] == 1
    assert result.metrics["llm_calls"] == 1
    assert all(r.status == "skipped" for r in result.records)


def test_call_budget_stops_further_batches(tmp_path, offline_fetch):
    calls = {"n": 0}

    def fake(*, system: str, user: str) -> str:
        calls["n"] += 1
        ids = [f"0000.{i:05d}" for i in range(1, 6) if f"0000.{i:05d}" in user]
        return _ok_batch(ids or ["0000.00001"])

    result = extract_papers(
        [_hit_n(i) for i in range(1, 7)],
        _cfg(tmp_path, llm_batch_size=5, max_llm_calls_per_run=1),
        complete_fn=fake,
    )
    assert calls["n"] == 1
    assert result.metrics["llm_calls"] == 1
    assert sum(r.status == "skipped" for r in result.records) == 1


def test_json_parse_fail_does_not_split_without_budget(tmp_path, offline_fetch):
    calls = {"n": 0}

    def fake(*, system: str, user: str) -> str:
        calls["n"] += 1
        return "truncated {"

    result = extract_papers(
        [_hit_n(1), _hit_n(2)],
        _cfg(tmp_path, llm_batch_size=5, max_llm_calls_per_run=1),
        complete_fn=fake,
    )
    assert calls["n"] == 1
    assert result.metrics["llm_calls"] == 1
    assert all(r.status == "skipped" for r in result.records)


def test_llm_min_interval_sleeps_between_calls(tmp_path, offline_fetch, monkeypatch):
    slept: list[float] = []
    monkeypatch.setattr("research_assistant.extraction.pipeline.time.sleep", slept.append)

    def fake(*, system: str, user: str) -> str:
        if "0000.00001" in user and "0000.00002" not in user:
            return _ok_json("0000.00001")
        return _ok_json("0000.00002")

    result = extract_papers(
        [_hit_n(1), _hit_n(2)],
        _cfg(tmp_path, llm_batch_size=1, llm_min_interval_s=5.0),
        complete_fn=fake,
    )
    assert result.metrics["llm_calls"] == 2
    assert slept and slept[0] >= 4.5


def test_extract_one_uses_injected_prompt_path(monkeypatch, tmp_path):
    from research_assistant.extraction import pipeline as pipe

    prompt = PromptDocument(
        arxiv_id="0000.00001",
        version="v1",
        title="Tiny Coreset Selection",
        abstract="We select a coreset of CIFAR-10 using EL2N scores.",
        source_kind="tex",
        selected_sections=[
            Section(
                heading="Abstract",
                text="We select a coreset of CIFAR-10 using EL2N scores.",
            )
        ],
    )
    parsed = PaperDocument(
        arxiv_id="0000.00001",
        version="v1",
        source_kind="tex",
        title=prompt.title,
        abstract=prompt.abstract,
        sections=prompt.selected_sections,
    )

    def stub(hit, source_kind, config, paper_doc):
        if source_kind != "tex":
            raise RuntimeError("should stop after tex")
        return prompt, parsed

    monkeypatch.setattr(pipe, "_prompt_for_source", stub)
    cfg = _cfg(tmp_path, llm_batch_size=1)
    paper, cached = extract_one(_hit(), cfg, "fp-test", lambda *, system, user: _ok_json())
    assert not cached
    assert paper.status == "ok"
    assert paper.source_kind == "tex"
    assert paper.method is not None
