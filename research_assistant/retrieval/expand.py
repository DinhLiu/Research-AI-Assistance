from __future__ import annotations

import json
import logging
import re
from collections.abc import Sequence

from research_assistant.llm.client import complete, expansion_provider

logger = logging.getLogger(__name__)

_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)

SYSTEM_PROMPT = """You expand a research topic into literature-search queries.
Return JSON only, no markdown, shape: {"queries": ["...", "..."]}.
Rules:
- Produce diverse queries: synonyms, abbreviations, adjacent methods, problem formulations.
- Each query is a short English search phrase (not a full paragraph).
- Do not number items. Do not repeat the original topic verbatim.
"""


def parse_query_json(raw: str, n: int) -> list[str]:
    text = raw.strip()
    match = _JSON_BLOCK.search(text)
    if not match:
        raise ValueError("LLM response did not contain a JSON object")
    payload = json.loads(match.group(0))
    queries = payload.get("queries")
    if not isinstance(queries, list) or not queries:
        raise ValueError("JSON missing a non-empty queries list")
    cleaned: list[str] = []
    seen: set[str] = set()
    for item in queries:
        q = " ".join(str(item).split())
        key = q.lower()
        if not q or key in seen:
            continue
        seen.add(key)
        cleaned.append(q)
        if len(cleaned) >= n:
            break
    if not cleaned:
        raise ValueError("No usable queries in LLM JSON")
    return cleaned


def template_queries(topic: str, n: int) -> list[str]:
    topic = " ".join(topic.split())
    templates = [
        topic,
        f"methods and algorithms for {topic}",
        f"{topic} in deep neural networks",
        f"empirical study of {topic}",
        f"{topic} survey benchmarks datasets",
    ]
    out: list[str] = []
    seen: set[str] = set()
    for q in templates:
        key = q.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(q)
        if len(out) >= n + 1:
            break
    return out


class QueryExpander:
    name = "base"

    def expand(self, topic: str, n: int = 4) -> list[str]:
        raise NotImplementedError


class TemplateQueryExpander(QueryExpander):
    name = "template"

    def expand(self, topic: str, n: int = 4) -> list[str]:
        return template_queries(topic, n)


class LlmQueryExpander(QueryExpander):
    def __init__(self, provider: str, timeout_s: float = 30.0):
        self.provider = provider
        self.timeout_s = timeout_s
        self.name = f"llm:{provider}"

    def expand(self, topic: str, n: int = 4) -> list[str]:
        user = (
            f"Research topic: {topic}\n"
            f"Generate {n} diverse literature-search queries."
        )
        raw = self._complete(user)
        variants = parse_query_json(raw, n)
        return _merge_original(topic, variants)

    def _complete(self, user: str) -> str:
        return complete(
            system=SYSTEM_PROMPT,
            user=user,
            json_mode=True,
            temperature=0.3,
            timeout_s=self.timeout_s,
            provider=self.provider,
        )


class FallbackQueryExpander(QueryExpander):
    """Try an LLM expander; fall back to templates on any failure."""

    def __init__(self, primary: QueryExpander, fallback: QueryExpander | None = None):
        self.primary = primary
        self.fallback = fallback or TemplateQueryExpander()
        self.name = primary.name

    def expand(self, topic: str, n: int = 4) -> list[str]:
        try:
            queries = self.primary.expand(topic, n)
            if queries:
                self.name = self.primary.name
                return queries
        except Exception as exc:
            logger.error("LLM query expansion failed: %s; falling back to templates", exc)
        self.name = self.fallback.name
        return self.fallback.expand(topic, n)


def build_expander(timeout_s: float = 30.0) -> QueryExpander:
    provider = expansion_provider()
    if provider:
        return FallbackQueryExpander(LlmQueryExpander(provider, timeout_s=timeout_s))
    logger.info("No LLM API key set; using template query expansion")
    return TemplateQueryExpander()


def _merge_original(topic: str, variants: Sequence[str]) -> list[str]:
    topic = " ".join(topic.split())
    out = [topic]
    seen = {topic.lower()}
    for q in variants:
        key = q.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(q)
    return out
