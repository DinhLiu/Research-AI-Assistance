"""Turn a PaperDocument into a PromptDocument under a SectionPolicy budget."""

from __future__ import annotations

from research_assistant.config import ExtractionConfig
from research_assistant.extraction.normalize import normalize_heading
from research_assistant.extraction.types import PaperDocument, PromptDocument, Section, SectionPolicy, SourceKind

ALWAYS = ("abstract", "introduction", "conclusion", "conclusions")
METHOD = ("method", "methods", "approach", "model", "architecture", "proposed")
RESULTS = ("experiment", "experiments", "result", "results", "evaluation", "ablation")
LIMITS = ("limitation", "limitations", "discussion", "future work")
RELATED = ("related work", "related works", "prior work", "background", "literature")
APPENDIX = ("appendix", "supplement", "supplementary", "supplemental")
BIBLIO = ("reference", "references", "bibliography", "acknowledgment", "acknowledgements")

PRIORITY = (
    ("always", 0),
    ("method", 1),
    ("results", 2),
    ("limits", 3),
    ("related", 4),
    ("other", 5),
    ("appendix", 6),
    ("biblio", 7),
)


def policy_from_config(config: ExtractionConfig) -> SectionPolicy:
    return SectionPolicy(
        include_related_work=config.include_related_work,
        include_appendix=config.include_appendix,
        include_bibliography=config.include_bibliography,
        abstract_only=config.abstract_only,
        max_input_chars=config.max_input_chars,
        max_input_tokens=config.max_input_tokens,
    )


def classify_heading(heading: str) -> str:
    key = normalize_heading(heading)
    if any(token == key or token in key for token in ALWAYS):
        return "always"
    if any(token == key or token in key for token in METHOD):
        return "method"
    if any(token == key or token in key for token in RESULTS):
        return "results"
    if any(token == key or token in key for token in LIMITS):
        return "limits"
    if any(token == key or token in key for token in RELATED):
        return "related"
    if any(token == key or token in key for token in APPENDIX):
        return "appendix"
    if any(token == key or token in key for token in BIBLIO):
        return "biblio"
    return "other"


def allowed(kind: str, policy: SectionPolicy) -> bool:
    if kind == "related":
        return policy.include_related_work
    if kind == "appendix":
        return policy.include_appendix
    if kind == "biblio":
        return policy.include_bibliography
    return True


def select_sections(paper: PaperDocument, policy: SectionPolicy) -> list[Section]:
    if policy.abstract_only:
        for section in _with_abstract(paper):
            if normalize_heading(section.heading) == "abstract":
                return [section]
        if paper.abstract:
            return [Section(heading="Abstract", level=0, text=paper.abstract)]
        return []

    ranked: list[tuple[int, int, Section]] = []
    for index, section in enumerate(_with_abstract(paper)):
        kind = classify_heading(section.heading)
        if not allowed(kind, policy):
            continue
        priority = next(p for name, p in PRIORITY if name == kind)
        ranked.append((priority, index, section))
    ranked.sort(key=lambda item: (item[0], item[1]))

    selected: list[Section] = []
    used = 0
    budget = max(policy.max_input_chars, 1)
    # max_input_tokens is reserved for a later tokenizer-backed budget.
    _ = policy.max_input_tokens
    for _, _, section in ranked:
        chunk = _section_chars(section)
        if selected and used + chunk > budget:
            remaining = budget - used
            if remaining < 400:
                continue
            trimmed = Section(
                heading=section.heading,
                level=section.level,
                text=_trim_text(section.text, remaining - len(section.heading) - 2),
            )
            if trimmed.text:
                selected.append(trimmed)
                used += _section_chars(trimmed)
            continue
        selected.append(section)
        used += chunk
        if used >= budget:
            break
    return selected


def build_prompt_document(
    paper: PaperDocument,
    config: ExtractionConfig,
    *,
    source_kind: SourceKind | None = None,
) -> PromptDocument:
    policy = policy_from_config(config)
    return PromptDocument(
        arxiv_id=paper.arxiv_id,
        version=paper.version,
        title=paper.title,
        abstract=paper.abstract,
        source_kind=source_kind or paper.source_kind,
        selected_sections=select_sections(paper, policy),
        selector_version=config.selector_version,
        policy=policy,
    )


def build_abstract_prompt(
    arxiv_id: str,
    version: str,
    title: str,
    abstract: str,
    config: ExtractionConfig,
) -> PromptDocument:
    policy = policy_from_config(config)
    text = abstract.strip() or title.strip()
    sections = [Section(heading="Abstract", level=0, text=text)] if text else []
    return PromptDocument(
        arxiv_id=arxiv_id,
        version=version,
        title=title,
        abstract=abstract,
        source_kind="abstract",
        selected_sections=sections,
        selector_version=config.selector_version,
        policy=policy,
    )


def _with_abstract(paper: PaperDocument) -> list[Section]:
    sections = list(paper.sections)
    has_abstract = any(normalize_heading(s.heading) == "abstract" for s in sections)
    if paper.abstract and not has_abstract:
        sections = [Section(heading="Abstract", level=0, text=paper.abstract)] + sections
    return sections


def _section_chars(section: Section) -> int:
    return len(section.heading) + 2 + len(section.text)


def _trim_text(text: str, limit: int) -> str:
    if limit <= 0:
        return ""
    if len(text) <= limit:
        return text
    clipped = text[:limit]
    if " " in clipped:
        clipped = clipped.rsplit(" ", 1)[0]
    return clipped.strip()
