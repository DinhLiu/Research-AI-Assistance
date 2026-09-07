"""Regression coverage for Stage 3 grounding and output correctness."""
import json
import pytest
from research_assistant.config import SynthesisConfig
from research_assistant.synthesis.pipeline import synthesize
from research_assistant.synthesis.prompt import build_user_prompt
from research_assistant.synthesis.types import SynthesisClaim
from research_assistant.synthesis.validate import validate_claim
from tests.synthesis_utils import paper, snapshot

@pytest.mark.parametrize('citation,expected', [('2205.09329v99', 'rejected'), ('2205.09329v1', 'structurally_validated'), ('2205.09329', 'structurally_validated')])
def test_citation_requires_exact_explicit_version(citation, expected):
    result = synthesize(snapshot([paper()]), SynthesisConfig(use_cache=False))
    key = result.paper_manifest[0].paper_key
    ref = next(u.unit_id for u in result.evidence_registry.units if u.kind == 'method')
    checked = validate_claim(SynthesisClaim(text=f'{citation} uses EL2N.', kind='subset', subject_paper_keys=[key], support_refs=[ref]), permitted_keys={key}, cluster_keys={key}, registry=result.evidence_registry.lookup(), permitted_kinds=frozenset({'method'}))
    assert checked.validation_status == expected


def test_cache_inventory_matches_current_rows(tmp_path):
    first = paper()
    other = paper(arxiv_id='2301.12345')
    empty = paper(arxiv_id='2301.12346', method='!!!')
    skipped = paper(arxiv_id='2301.12347', status='skipped', error='first reason')
    cfg = SynthesisConfig(cache_dir=tmp_path)
    before = synthesize(snapshot([first, other, empty, skipped, first]), cfg)
    current = snapshot([skipped.model_copy(update={'error': 'updated reason'}), empty, other, first, first])
    cached = synthesize(current, cfg)
    fresh = synthesize(current, SynthesisConfig(use_cache=False))
    assert cached.execution.cache_cluster_hit
    assert cached.assignments == before.assignments == fresh.assignments
    assert cached.input_inventory == fresh.input_inventory
    assert cached.input_inventory[0].reason == 'updated reason'
    assert cached.input_inventory[1].disposition == 'empty_features'
    assert cached.input_inventory[-1].disposition == 'duplicate'


def test_compact_prompt_keeps_references_and_can_generate_valid_summary():
    extracted = snapshot([paper(results=[f'Result {i} ' + 'x'*250 for i in range(20)])])
    dry = synthesize(extracted, SynthesisConfig(use_cache=False))
    cfg = SynthesisConfig(use_cache=False, summarize=True, max_prompt_chars=2500)
    prompt, _ = build_user_prompt(topic=dry.topic, assignments=dry.assignments, cards=dry.paper_manifest, registry=dry.evidence_registry, config=cfg)
    assert 'Quotes omitted to fit budget' in prompt
    assert len(prompt) <= cfg.max_prompt_chars
    for unit in dry.evidence_registry.units:
        assert f'[{unit.unit_id}]' in prompt
    method = next(u for u in dry.evidence_registry.units if u.kind == 'method')
    def complete(**kwargs):
        assert f'[{method.unit_id}]' in kwargs['user']
        return json.dumps({'summaries': [{'cluster_id': dry.assignments[0].cluster_id, 'claims': [{'kind': 'subset', 'text': method.text, 'subject_paper_keys': [method.paper_key], 'support_refs': [method.unit_id]}]}]})
    result = synthesize(extracted, cfg, complete_fn=complete)
    assert result.execution.summary_status == 'complete'
    assert result.execution.logical_calls == 1


@pytest.mark.parametrize('repair_succeeds', [False, True])
def test_empty_cluster_summary_requires_repair(repair_succeeds):
    extracted = snapshot([paper(), paper(arxiv_id='2301.12345', method='Quantum entanglement teleportation superconducting qubits')])
    dry = synthesize(extracted, SynthesisConfig(use_cache=False))
    assert len(dry.assignments) == 2
    calls = []
    def complete(**kwargs):
        calls.append(kwargs['user'])
        summaries = []
        for index, cluster in enumerate(dry.assignments):
            unit = next(u for u in dry.evidence_registry.units if u.paper_key == cluster.paper_keys[0] and u.kind == 'method')
            claims = [{'kind': 'subset', 'text': unit.text, 'subject_paper_keys': [unit.paper_key], 'support_refs': [unit.unit_id]}]
            if index == 1 and not (repair_succeeds and len(calls) == 2):
                claims = []
            summaries.append({'cluster_id': cluster.cluster_id, 'claims': claims})
        return json.dumps({'summaries': summaries})
    result = synthesize(extracted, SynthesisConfig(use_cache=False, summarize=True), complete_fn=complete)
    assert len(calls) == 2
    assert 'empty_cluster_summary:' in calls[1]
    assert result.execution.summary_status == ('complete' if repair_succeeds else 'partial')
    assert bool(result.summaries[1].claims) == repair_succeeds
