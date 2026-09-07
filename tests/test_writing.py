import json
from dataclasses import replace

import pytest

from research_assistant.config import SynthesisConfig, WritingConfig
from research_assistant.llm.governor import Quota
from research_assistant.synthesis.pipeline import synthesize
from research_assistant.writing.pipeline import write_review
from research_assistant.writing.render import render_markdown
from research_assistant.writing.types import WritingInputError, WritingResult
from tests.synthesis_utils import paper, snapshot
from tests.test_llm_governor import setup


def synthesis(tmp_path, n=2):
    return synthesize(snapshot([paper(arxiv_id=f"2205.{i:05d}", limitations=["Limited evidence."],
                                      method="Uses EL2N scores to prune training data.") for i in range(n)]),
                      SynthesisConfig(cache_dir=tmp_path / "synthesis", use_cache=False))


def cfg(tmp_path, **kwargs):
    return WritingConfig(**dict({"cache_dir": tmp_path / "writing", "target_words": 100}, **kwargs))


def echo(*, system, user):
    data, _ = json.JSONDecoder().raw_decode(user)
    return json.dumps({"claims": [{"claim_id": s["claim_id"], "text": s["source_text"]} for s in data["slots"]]})


def live(tmp_path, snap, config, complete_fn=echo):
    _, gov = setup(tmp_path / "governor")
    return write_review(snap, config, complete_fn=complete_fn, governor=gov,
                        quota=Quota(6000, 1000000, 1000, safety=1, min_interval_s=0))


def test_offline_roundtrip_and_citations(tmp_path):
    result = write_review(synthesis(tmp_path), cfg(tmp_path))
    assert result.execution.http_attempts == 0
    assert result.generation_status == "fallback"
    assert result.coverage["supported_papers"] == 2
    assert WritingResult.model_validate_json(result.model_dump_json()) == result
    markdown = render_markdown(result)
    assert "https://arxiv.org/abs/2205.00000v1" in markdown
    assert "semantic support is not verified" in markdown


def test_one_call_and_cache_hit(tmp_path):
    snap = synthesis(tmp_path)
    config = cfg(tmp_path, use_llm=True)
    first = live(tmp_path, snap, config)
    assert first.execution.http_attempts == 1
    assert first.generation_status == "complete"
    second = live(tmp_path, snap, config, lambda **kw: pytest.fail("cache hit"))
    assert second.execution.http_attempts == 0
    assert second.generation_status == "complete"


def test_invalid_json_one_repair(tmp_path):
    calls = []
    def complete(**kwargs):
        calls.append(1)
        return "bad json" if len(calls) == 1 else echo(**kwargs)
    result = live(tmp_path, synthesis(tmp_path), cfg(tmp_path, use_llm=True), complete)
    assert len(calls) == 2
    assert result.execution.repair_calls == 1
    assert result.generation_status == "complete"


def test_invalid_claim_is_repaired_without_losing_valid_claim(tmp_path):
    calls = []
    def complete(**kwargs):
        response = json.loads(echo(**kwargs))
        calls.append(len(response["claims"]))
        if len(calls) == 1:
            response["claims"][0]["text"] = "Unknown paper 9999.99999v7 outperforms all prior work."
        return json.dumps(response)
    result = live(tmp_path, synthesis(tmp_path), cfg(tmp_path, use_llm=True), complete)
    assert calls[1] == 1
    assert result.generation_status == "complete"
    assert any("unsupported" in e for e in result.validation_errors)


def test_zero_budget_fallback_and_missing_quota(tmp_path, monkeypatch):
    snap = synthesis(tmp_path)
    result = live(tmp_path, snap, cfg(tmp_path, use_llm=True, max_http_attempts_per_run=0),
                  lambda **kw: pytest.fail("zero budget"))
    assert result.execution.http_attempts == 0
    assert result.generation_status == "fallback"
    for key in ("LLM_RPM", "LLM_TPM", "LLM_RPD"):
        monkeypatch.delenv(key, raising=False)
    result = write_review(snap, cfg(tmp_path, use_llm=True))
    assert "quota_configuration_required" in result.execution.stop_reason


def test_bad_identity_and_empty(tmp_path):
    snap = synthesis(tmp_path)
    snap.evidence_registry.units.append(snap.evidence_registry.units[0])
    with pytest.raises(WritingInputError, match="duplicate_unit_id"):
        write_review(snap, cfg(tmp_path))
    empty = write_review(synthesis(tmp_path, n=0), cfg(tmp_path, use_llm=True))
    assert empty.generation_status == "empty"
    assert empty.execution.http_attempts == 0


def test_oversized_pack_keeps_template_and_coverage(tmp_path):
    result = live(tmp_path, synthesis(tmp_path), cfg(tmp_path, use_llm=True, max_input_tokens=10),
                  lambda **kw: pytest.fail("oversize"))
    assert result.execution.http_attempts == 0
    assert result.plan.omitted
    assert result.coverage["supported_papers"] == 2


def test_resume_keeps_hard_attempt_budget(tmp_path):
    snap = synthesis(tmp_path)
    config = cfg(tmp_path, use_llm=True, run_id="resume", max_http_attempts_per_run=1)
    first = live(tmp_path, snap, config, lambda **kw: "bad")
    assert first.execution.http_attempts == 1
    resumed = live(tmp_path, snap, config, lambda **kw: pytest.fail("spent budget"))
    assert resumed.execution.http_attempts == 1
    assert resumed.generation_status == "fallback"


def test_cache_invalidated_by_language_and_evidence(tmp_path):
    snap = synthesis(tmp_path)
    first = live(tmp_path, snap, cfg(tmp_path, use_llm=True))
    changed = live(tmp_path, snap, cfg(tmp_path, use_llm=True, language="vi"))
    assert changed.execution.http_attempts == 1
    snap.evidence_registry.units[0].text += " Updated."
    other = live(tmp_path, snap, cfg(tmp_path, use_llm=True))
    assert other.execution.http_attempts == 1
    assert other.writing_input_digest != first.writing_input_digest


def test_cli_dry_run_and_offline(tmp_path, capsys):
    from research_assistant.writing.cli import main
    path = tmp_path / "input.json"
    path.write_text(synthesis(tmp_path).model_dump_json())
    md, js = tmp_path / "review.md", tmp_path / "review.json"
    assert main([str(path), "--dry-run", "--json-out", str(js)]) == 0
    assert json.loads(js.read_text())["generation_status"] == "planned"
    assert main([str(path), "--md-out", str(md), "--json-out", str(js)]) == 0
    assert md.read_text().startswith("# Literature review")
    assert main([str(path), "--target-words", "0"]) == 2


def test_default_length_small_snapshot_is_one_batch(tmp_path):
    result = write_review(synthesis(tmp_path), WritingConfig(cache_dir=tmp_path, dry_run=True))
    assert len(result.plan.batches) == 1
    assert not result.plan.omitted


def test_truncated_output_never_complete(tmp_path):
    from research_assistant.llm.client import Completion
    result = live(tmp_path, synthesis(tmp_path), cfg(tmp_path, use_llm=True),
                  lambda **kw: Completion(echo(**kw), "injected", "injected", finish_reason="length"))
    assert result.generation_status == "fallback"
    assert result.execution.http_attempts == 2
    assert "output_truncated" in result.validation_errors


def test_transport_retries_are_inside_attempt_cap(tmp_path):
    import httpx
    calls = []
    def broken(**kwargs):
        calls.append(1)
        raise httpx.ReadTimeout("provider unavailable")
    result = live(tmp_path, synthesis(tmp_path), cfg(tmp_path, use_llm=True), broken)
    assert result.generation_status == "fallback"
    assert len(calls) == 2
    assert result.execution.http_attempts == 2
    assert result.execution.transport_retries == 1
    assert result.execution.repair_calls == 0


def test_resume_does_not_repeat_exhausted_content_repair(tmp_path):
    snap = synthesis(tmp_path)
    config = cfg(tmp_path, use_llm=True, run_id="no_repeat")
    first = live(tmp_path, snap, config, lambda **kw: "bad")
    assert first.execution.http_attempts == 2
    resumed = live(tmp_path, snap, config, lambda **kw: pytest.fail("no more content retries"))
    assert resumed.execution.http_attempts == 2
    assert resumed.execution.repair_calls == 1


def test_full_batch_ceiling_and_partial_fallback(tmp_path):
    snap = synthesis(tmp_path, n=10)
    config = cfg(tmp_path, use_llm=True, max_input_tokens=3800)
    result = live(tmp_path, snap, config)
    assert len(result.plan.batches) == 3
    assert result.plan.omitted
    assert result.execution.http_attempts == 3
    assert result.generation_status == "partial"
    assert result.coverage["supported_papers"] == 10


def test_same_run_different_input_rejected(tmp_path):
    snap = synthesis(tmp_path)
    config = cfg(tmp_path, use_llm=True, run_id="fixed")
    live(tmp_path, snap, config)
    snap.topic = "new topic"
    with pytest.raises(WritingInputError, match="different_input"):
        live(tmp_path, snap, config)


def test_valid_json_but_wrong_refs_never_enters_prompt(tmp_path):
    snap = synthesis(tmp_path)
    snap.paper_manifest[0].evidence_ids = [snap.paper_manifest[1].evidence_ids[0]]
    with pytest.raises(WritingInputError, match="invalid_card_reference"):
        live(tmp_path, snap, cfg(tmp_path, use_llm=True))


def test_partial_cache_is_not_complete_and_corrupt_cache_ignored(tmp_path):
    snap = synthesis(tmp_path)
    config = cfg(tmp_path, use_llm=True, max_repair_calls=0)
    def partial(**kwargs):
        result = json.loads(echo(**kwargs))
        result["claims"] = result["claims"][:1]
        return json.dumps(result)
    first = live(tmp_path, snap, config, partial)
    assert first.generation_status == "partial"
    second = live(tmp_path, snap, config)
    assert second.execution.http_attempts == 1
    assert second.execution.cache_hits == 1
    assert second.generation_status == "complete"
    next((tmp_path / "writing/prose").glob("*.json")).write_text("corrupt")
    third = live(tmp_path, snap, config)
    assert third.execution.http_attempts == 1


def test_unsupported_papers_not_reported_empty_or_complete(tmp_path):
    snap = synthesis(tmp_path, n=1)
    snap.paper_manifest[0].evidence_ids = []
    snap.evidence_registry.units = []
    snap.gap_candidates = []
    result = write_review(snap, cfg(tmp_path))
    assert result.generation_status == "fallback"
    assert result.coverage["unsupported_paper_keys"]


def test_five_http_attempt_ceiling_includes_all_retries(tmp_path):
    import httpx
    calls = []
    def transient_then_success(**kwargs):
        calls.append(1)
        if len(calls) % 2:
            raise httpx.ReadTimeout("retry me")
        return echo(**kwargs)
    result = live(tmp_path, synthesis(tmp_path, n=10), cfg(tmp_path, use_llm=True, max_input_tokens=3800), transient_then_success)
    assert len(calls) == 5
    assert result.execution.http_attempts == 5
    assert result.generation_status == "partial"
    assert result.execution.stop_reason == "run_budget_exhausted"


def test_input_injection_in_arxiv_key_is_rejected(tmp_path):
    snap = synthesis(tmp_path, n=1)
    card = snap.paper_manifest[0]
    card.arxiv_id = "example) bad"
    card.paper_key = "example) badv1"
    with pytest.raises(WritingInputError, match="invalid_arxiv_identifier"):
        write_review(snap, cfg(tmp_path))


def test_empty_evaluation_rates_are_null(tmp_path):
    from evaluation.writing_scoring import score_writing
    result = write_review(synthesis(tmp_path, n=0), cfg(tmp_path))
    metrics = score_writing(result)
    assert metrics["paper_coverage"]["rate"] is None
    assert metrics["prose_coverage"]["rate"] is None
    assert metrics["semantic_support_rate"] is None


def test_concurrent_identical_reviews_single_flight(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Lock
    from research_assistant.llm.governor import Governor
    snap = synthesis(tmp_path)
    calls, lock = [], Lock()
    def counted(**kwargs):
        with lock:
            calls.append(1)
        return echo(**kwargs)
    def work(_):
        return write_review(snap, cfg(tmp_path, use_llm=True), complete_fn=counted,
                            governor=Governor(tmp_path / "state"),
                            quota=Quota(100000, 1000000, 10000, safety=1, min_interval_s=0))
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(work, range(2)))
    assert len(calls) == 1
    assert all(r.generation_status == "complete" for r in results)
    assert sorted(r.execution.http_attempts for r in results) == [0, 1]


def test_document_formatting_evidence_is_not_written_as_method(tmp_path):
    snap = synthesize(snapshot([paper(method="The abstract paragraph should be indented 1/2 inch on both margins.", limitations=[])]),
                      SynthesisConfig(cache_dir=tmp_path, use_cache=False))
    result = write_review(snap, cfg(tmp_path))
    assert result.plan.evidence_warnings
    assert not result.claims
    assert result.coverage["supported_papers"] == 0
    assert "abstract paragraph" not in render_markdown(result)
    assert result.generation_status == "fallback"


def test_invalid_quota_config_is_an_input_error(tmp_path, monkeypatch):
    for name, value in (("RPM", "not-a-number"), ("TPM", "10000"), ("RPD", "100")):
        monkeypatch.setenv("LLM_" + name, value)
    with pytest.raises(ValueError):
        write_review(synthesis(tmp_path), cfg(tmp_path, use_llm=True))
