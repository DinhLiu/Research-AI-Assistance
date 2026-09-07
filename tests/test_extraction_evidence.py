from __future__ import annotations

from research_assistant.extraction.types import (
    CandidatePack,
    EvidenceSnippet,
    IdRefClaim,
    IdRefMention,
    LlmExtraction,
    PaperDocument,
    PromptDocument,
    Section,
    SectionPolicy,
)
from research_assistant.extraction.validate import ground_llm, locate_quote, parse_candidate


def _prompt() -> PromptDocument:
    return PromptDocument(
        arxiv_id="0000.00001",
        version="v1",
        title="Tiny Coreset Selection",
        abstract="We select a coreset of CIFAR-10 using EL2N scores.",
        source_kind="tex",
        selected_sections=[
            Section(
                heading="Method",
                text="We prune the training set with EL2N scores computed at epoch 10.",
            ),
            Section(
                heading="Experiments",
                text="On CIFAR-10, random selection achieves 83.2%. Our method reaches 90.1% accuracy.",
            ),
        ],
        policy=SectionPolicy(),
    )


def test_locate_quote_prefers_declared_section():
    prompt = _prompt()
    result = locate_quote("EL2N scores computed at epoch 10", "Method", prompt)
    assert result.valid
    assert result.exact_section_match
    assert result.located_section == "Method"
    assert result.start_char is not None
    assert prompt.selected_sections[0].text[result.start_char : result.end_char]


def test_locate_quote_relocates_to_other_selected_section():
    prompt = _prompt()
    result = locate_quote("random selection achieves 83.2%", "Method", prompt)
    assert result.valid
    assert not result.exact_section_match
    assert result.located_section == "Experiments"


def test_full_text_is_not_enough_to_pass():
    prompt = _prompt()
    paper = PaperDocument(
        arxiv_id="0000.00001",
        version="v1",
        source_kind="tex",
        sections=[
            *prompt.selected_sections,
            Section(heading="Related Work", text="GraNd and EL2N score examples early in training."),
        ],
    )
    result = locate_quote(
        "score examples early in training",
        "Experiments",
        prompt,
        full_text=paper.full_text,
    )
    assert not result.valid
    assert result.in_full_text


def test_ground_llm_resolves_evidence_ids():
    pack = CandidatePack(
        arxiv_id="0000.00001",
        version="v1",
        title="Tiny Coreset Selection",
        source_kind="tex",
        snippets=[
            EvidenceSnippet(
                id="M1",
                kind="method",
                sentence="We prune the training set with EL2N scores computed at epoch 10.",
                section="Method",
                start_char=0,
                end_char=10,
            )
        ],
    )
    llm = LlmExtraction(
        arxiv_id="0000.00001",
        method=IdRefClaim(text="Prune with EL2N.", evidence_ids=["M1"]),
        method_keywords=[IdRefMention(value="EL2N", evidence_ids=["M1"])],
    )
    paper, report = ground_llm(llm, pack)
    assert paper.status == "ok"
    assert paper.method is not None
    assert paper.method.evidence[0].quote.startswith("We prune")
    assert paper.method_keywords[0].value == "EL2N"
    assert report.kept >= 2


def test_unknown_evidence_id_is_dropped():
    pack = CandidatePack(
        arxiv_id="0000.00001",
        version="v1",
        source_kind="tex",
        snippets=[
            EvidenceSnippet(id="M1", kind="method", sentence="We propose X.", section="Method")
        ],
    )
    llm = LlmExtraction(
        method=IdRefClaim(text="Something", evidence_ids=["Z9"]),
    )
    paper, _ = ground_llm(llm, pack)
    assert paper.method is None
    assert paper.status == "skipped"


def test_parse_candidate_from_markdown_fence():
    raw = """```json
    {"method": {"text": "x", "evidence_ids": ["M1"]}}
    ```"""
    candidate = parse_candidate(raw)
    assert candidate.method is not None
    assert candidate.method.evidence_ids == ["M1"]
