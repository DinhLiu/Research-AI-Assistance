"""Dictionary + section-weighted mentions (datasets, metrics). No LLM."""

from __future__ import annotations

import re

from research_assistant.extraction.normalize import normalize_text
from research_assistant.extraction.sections import classify_heading
from research_assistant.extraction.types import Evidence, Mention, PaperDocument, PromptDocument, SourceKind

DATASET_PATTERNS: dict[str, list[str]] = {
    "CIFAR-10": [r"\bCIFAR[- ]?10\b"],
    "CIFAR-100": [r"\bCIFAR[- ]?100\b"],
    "ImageNet": [r"\bImageNet(?:-1[Kk])?\b"],
    "Tiny-ImageNet": [r"\bTiny[- ]ImageNet\b"],
    "SVHN": [r"\bSVHN\b"],
    "MNIST": [r"\bMNIST\b"],
    "CINIC-10": [r"\bCINIC[- ]?10\b"],
}

METRIC_PATTERNS: dict[str, list[str]] = {
    "accuracy": [r"\baccuracy\b", r"\btop-1\b", r"\bacc(?:uracy)?\b"],
    "error rate": [r"\berror rate\b"],
    "F1": [r"\bF1(?:-score)?\b"],
    "AUROC": [r"\bAUROC\b", r"\bAUC\b"],
    "training time": [r"\btraining time\b", r"\bwall[- ]clock\b"],
    "time-to-accuracy": [r"\btime-to-accuracy\b", r"\bTTA\b"],
}

_SECTION_WEIGHT = {
    "results": 4,
    "method": 2,
    "always": 3,
    "limits": 1,
    "other": 0,
    "related": -3,
    "appendix": -2,
    "biblio": -5,
}

_USE_CUES = (
    r"\bwe evaluate(?: our)?(?: approach| method)? on\b",
    r"\bexperiments? on\b",
    r"\btrained on\b",
    r"\bbenchmarks?\b",
    r"\bdatasets?\b",
)
_HEDGE_CUES = (r"\be\.g\.\b", r"\bfor example\b", r"\bsuch as\b", r"\bincluding\b")


def extract_datasets(paper: PaperDocument, prompt: PromptDocument) -> list[Mention]:
    return _extract_mentions(DATASET_PATTERNS, paper, prompt, min_score=2.5)


def extract_metrics(paper: PaperDocument, prompt: PromptDocument) -> list[Mention]:
    return _extract_mentions(METRIC_PATTERNS, paper, prompt, min_score=2.0)


def _extract_mentions(
    catalog: dict[str, list[str]],
    paper: PaperDocument,
    prompt: PromptDocument,
    *,
    min_score: float,
) -> list[Mention]:
    source_kind: SourceKind = prompt.source_kind
    found: dict[str, tuple[float, Mention]] = {}
    sections = list(prompt.selected_sections)
    if paper.abstract and not any(s.heading.lower() == "abstract" for s in sections):
        from research_assistant.extraction.types import Section

        sections = [Section(heading="Abstract", level=0, text=paper.abstract)] + sections
    for section in sections:
        weight = _SECTION_WEIGHT.get(classify_heading(section.heading), 0)
        text = section.text
        lowered = text.lower()
        for name, patterns in catalog.items():
            for pattern in patterns:
                for match in re.finditer(pattern, text, flags=re.IGNORECASE):
                    quote = _window(text, match.start(), match.end())
                    score = float(weight)
                    if any(re.search(cue, lowered) for cue in _USE_CUES):
                        score += 4.0
                    if any(re.search(cue, quote.lower()) for cue in _HEDGE_CUES):
                        score -= 1.0
                    if classify_heading(section.heading) == "results":
                        score += 1.0
                    if score < min_score:
                        continue
                    evidence = Evidence(
                        quote=normalize_text(quote),
                        section=section.heading,
                        source_kind=source_kind,
                        start_char=match.start(),
                        end_char=match.end(),
                    )
                    mention = Mention(value=name, evidence=evidence)
                    prev = found.get(name)
                    if prev is None or score > prev[0]:
                        found[name] = (score, mention)
    return [item[1] for _, item in sorted(found.items(), key=lambda kv: -kv[1][0])]


def _window(text: str, start: int, end: int, radius: int = 90) -> str:
    lo = max(0, start - radius)
    hi = min(len(text), end + radius)
    return normalize_text(text[lo:hi])
