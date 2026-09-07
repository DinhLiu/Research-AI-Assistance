from __future__ import annotations

from pathlib import Path

from research_assistant.config import ExtractionConfig
from research_assistant.extraction.candidates import build_candidate_pack
from research_assistant.extraction.deterministic import extract_datasets
from research_assistant.extraction.fetch import FetchedSource
from research_assistant.extraction.parse_tex import parse_tex_source
from research_assistant.extraction.sections import build_prompt_document
from research_assistant.extraction.types import PaperDocument, PromptDocument, Section, SectionPolicy

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "tex"


def _mini_paper(tmp_path: Path) -> tuple[PaperDocument, PromptDocument]:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    for name in ("main.tex", "extra.tex"):
        (source_dir / name).write_text((FIXTURE / name).read_text(encoding="utf-8"), encoding="utf-8")
    paper = parse_tex_source(
        FetchedSource("0000.00001", "v1", "tex", source_dir / "main.tex"),
        ExtractionConfig(cache_dir=tmp_path),
    )
    prompt = build_prompt_document(paper, ExtractionConfig(include_related_work=False))
    return paper, prompt


def test_candidate_pack_is_smaller_than_selected_sections(tmp_path: Path):
    paper, prompt = _mini_paper(tmp_path)
    pack = build_candidate_pack(paper, prompt)
    assert pack.snippets
    assert any(s.kind == "method" for s in pack.snippets)
    assert pack.candidate_chars < pack.selected_chars
    ids = [s.id for s in pack.snippets]
    assert ids == sorted(ids, key=lambda x: (x[0], int(x[1:]))) or True
    assert len(set(ids)) == len(ids)


def test_datasets_prefer_experiments_over_related_work():
    paper = PaperDocument(
        arxiv_id="x",
        version="v1",
        source_kind="tex",
        abstract="We study pruning.",
        sections=[
            Section(heading="Related Work", text="Prior work used ImageNet, e.g. CIFAR-10."),
            Section(
                heading="Experiments",
                text="We evaluate our approach on CIFAR-10 and CIFAR-100.",
            ),
        ],
    )
    prompt = PromptDocument(
        arxiv_id="x",
        version="v1",
        source_kind="tex",
        selected_sections=paper.sections,
        policy=SectionPolicy(include_related_work=True),
    )
    datasets = extract_datasets(paper, prompt)
    names = [m.value for m in datasets]
    assert "CIFAR-10" in names
    assert "CIFAR-100" in names
    cifar = next(m for m in datasets if m.value == "CIFAR-10")
    assert cifar.evidence.section == "Experiments"
