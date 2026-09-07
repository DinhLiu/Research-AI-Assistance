from __future__ import annotations

from pathlib import Path

import pytest

from research_assistant.config import ExtractionConfig
from research_assistant.extraction.fetch import FetchedSource
from research_assistant.extraction.parse_tex import parse_tex_source
from research_assistant.extraction.sections import (
    build_prompt_document,
    classify_heading,
    policy_from_config,
    select_sections,
)

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "tex"


def _parse_fixture(tmp_path: Path):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    for name in ("main.tex", "extra.tex"):
        (source_dir / name).write_text((FIXTURE / name).read_text(encoding="utf-8"), encoding="utf-8")
    return parse_tex_source(
        FetchedSource("0000.00001", "v1", "tex", source_dir / "main.tex"),
        ExtractionConfig(cache_dir=tmp_path),
    )


def test_parse_tex_inlines_input_and_splits_sections(tmp_path: Path):
    paper = _parse_fixture(tmp_path)
    headings = [s.heading.lower() for s in paper.sections]
    assert paper.title.lower().startswith("tiny coreset")
    assert "cifar-10" in paper.abstract.lower()
    assert any("method" in h for h in headings)
    assert any("related" in h for h in headings)
    joined = " ".join(s.text for s in paper.sections)
    assert "subset selection is cheaper" in joined.lower()


def test_selector_drops_related_work_by_default(tmp_path: Path):
    paper = _parse_fixture(tmp_path)
    selected = select_sections(paper, policy_from_config(ExtractionConfig()))
    headings = [s.heading.lower() for s in selected]
    assert any("method" in h for h in headings)
    assert not any("related" in h for h in headings)

    included = select_sections(
        paper, policy_from_config(ExtractionConfig(include_related_work=True))
    )
    assert any("related" in s.heading.lower() for s in included)


def test_prompt_document_respects_char_budget(tmp_path: Path):
    paper = _parse_fixture(tmp_path)
    prompt = build_prompt_document(paper, ExtractionConfig(max_input_chars=180))
    used = sum(len(s.heading) + len(s.text) for s in prompt.selected_sections)
    assert used <= 220
    assert prompt.selected_sections


def test_classify_heading_kinds():
    assert classify_heading("3. Experiments") == "results"
    assert classify_heading("Related Work") == "related"
    assert classify_heading("References") == "biblio"


def test_abstract_only_keeps_just_abstract(tmp_path: Path):
    paper = _parse_fixture(tmp_path)
    selected = select_sections(paper, policy_from_config(ExtractionConfig(abstract_only=True)))
    assert len(selected) == 1
    assert selected[0].heading.lower() == "abstract"


def test_parse_pdf_from_synthetic_file(tmp_path: Path):
    pymupdf = pytest.importorskip("pymupdf")
    from research_assistant.extraction.parse_pdf import parse_pdf_source

    pdf_path = tmp_path / "paper.pdf"
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text(
        (72, 72),
        "Tiny Coreset Selection\n\nAbstract\nWe select a coreset of CIFAR-10.\n\n"
        "1 Introduction\nTraining is expensive.\n\n2 Method\nWe prune with EL2N.\n",
    )
    doc.save(pdf_path)
    doc.close()
    paper = parse_pdf_source(FetchedSource("0000.00001", "v1", "pdf", pdf_path))
    assert paper.source_kind == "pdf"
    assert paper.sections
    blob = " ".join(s.text for s in paper.sections).lower() + " " + paper.title.lower()
    assert "el2n" in blob or "cifar" in blob or "coreset" in blob
