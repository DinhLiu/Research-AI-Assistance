"""LLM prompt over compressed evidence IDs, not raw paper sections."""

from __future__ import annotations

from research_assistant.extraction.types import CandidatePack

PROMPT_VERSION = "2"

SYSTEM_PROMPT = """You extract grounded facts from numbered evidence sentences.
Return JSON only. For one paper:
{
  "arxiv_id": "...",
  "problem": {"text": "...", "evidence_ids": ["P1"]},
  "method": {"text": "...", "evidence_ids": ["M1", "M2"]},
  "results": [{"text": "...", "evidence_ids": ["R1"]}],
  "limitations": [{"text": "...", "evidence_ids": ["L1"]}],
  "contributions": [{"text": "...", "evidence_ids": ["C1"]}],
  "method_keywords": [{"value": "EL2N", "evidence_ids": ["M1"]}]
}
For multiple papers wrap as {"papers": [ ...one object per paper... ]}.
Rules:
- Use ONLY the provided evidence IDs. Do not invent IDs.
- evidence_ids must belong to that paper (same arxiv_id / PAPER block).
- Do not copy quotes; Python already has the sentence for each ID.
- method is required. If unsupported, use null and empty lists.
- method_keywords are short names that appear in the cited sentences.
- Do not add datasets or metrics (those are extracted locally).
"""


def render_candidate_pack(pack: CandidatePack) -> str:
    parts = [
        f'<PAPER id="{pack.arxiv_id}{pack.version}">',
        f"Title: {pack.title or '(unknown)'}",
    ]
    if pack.abstract:
        parts.append(f"Abstract: {pack.abstract[:400]}")
    if pack.datasets:
        parts.append("Deterministic datasets: " + ", ".join(m.value for m in pack.datasets))
    if pack.metrics:
        parts.append("Deterministic metrics: " + ", ".join(m.value for m in pack.metrics))
    grouped: dict[str, list] = {}
    for snippet in pack.snippets:
        grouped.setdefault(snippet.kind, []).append(snippet)
    for kind in ("problem", "method", "result", "limitation", "contribution"):
        items = grouped.get(kind) or []
        if not items:
            continue
        parts.append(f"{kind.upper()} candidates:")
        for snippet in items:
            parts.append(f"[{snippet.id}] ({snippet.section}) {snippet.sentence}")
    parts.append("</PAPER>")
    return "\n".join(parts)


def render_batch(packs: list[CandidatePack]) -> str:
    chunks = [
        "Extract one JSON object per paper. Cite only IDs from that paper's block.",
        "",
    ]
    for pack in packs:
        chunks.append(render_candidate_pack(pack))
        chunks.append("")
    return "\n".join(chunks).strip()


def retry_user_message(packs: list[CandidatePack], errors: list[str]) -> str:
    joined = "\n".join(f"- {err}" for err in errors[:12]) or "- previous output failed validation"
    return (
        f"{render_batch(packs)}\n\n"
        "The previous JSON failed validation:\n"
        f"{joined}\n"
        "Return corrected JSON using only the evidence IDs above."
    )
