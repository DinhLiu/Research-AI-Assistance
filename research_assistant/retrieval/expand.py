from __future__ import annotations

import json
import logging
import os
import re
from collections.abc import Sequence

import httpx

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
        if self.provider == "groq":
            return _openai_compatible(
                base_url="https://api.groq.com/openai/v1",
                api_key=os.environ["GROQ_API_KEY"],
                model=os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant"),
                user=user,
                timeout_s=self.timeout_s,
            )
        if self.provider == "openai":
            return _openai_compatible(
                base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
                api_key=os.environ["OPENAI_API_KEY"],
                model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
                user=user,
                timeout_s=self.timeout_s,
            )
        if self.provider == "gemini":
            return _gemini_complete(user, timeout_s=self.timeout_s)
        raise ValueError(f"Unknown LLM provider: {self.provider}")


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
    if _env_key("GROQ_API_KEY"):
        return FallbackQueryExpander(LlmQueryExpander("groq", timeout_s=timeout_s))
    if _env_key("GEMINI_API_KEY") or _env_key("GOOGLE_API_KEY"):
        return FallbackQueryExpander(LlmQueryExpander("gemini", timeout_s=timeout_s))
    if _env_key("OPENAI_API_KEY"):
        return FallbackQueryExpander(LlmQueryExpander("openai", timeout_s=timeout_s))
    logger.info("No LLM API key set; using template query expansion")
    return TemplateQueryExpander()


def _env_key(name: str) -> str | None:
    raw = os.environ.get(name)
    if not raw:
        return None
    value = raw.strip().strip('"').strip("'")
    return value or None


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


def _openai_compatible(*, base_url: str, api_key: str, model: str, user: str, timeout_s: float) -> str:
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "temperature": 0.3,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ],
    }
    with httpx.Client(timeout=timeout_s) as client:
        response = client.post(
            url,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=payload,
        )
        response.raise_for_status()
        data = response.json()
    return data["choices"][0]["message"]["content"]


def _gemini_complete(user: str, timeout_s: float) -> str:
    api_key = _env_key("GEMINI_API_KEY") or _env_key("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY / GOOGLE_API_KEY is not set")
    model = (os.environ.get("GEMINI_MODEL") or "gemini-2.5-flash").strip()
    model = model.removeprefix("models/")
    logger.info("Calling Gemini model=%s", model)

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    headers = {
        "x-goog-api-key": api_key,
        "Content-Type": "application/json",
    }
    payload = {
        "system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "contents": [{"role": "user", "parts": [{"text": user}]}],
        "generationConfig": {
            "temperature": 0.3,
            "responseMimeType": "application/json",
        },
    }
    data = _gemini_post(url, headers, payload, timeout_s)
    text = _gemini_response_text(data)
    if not text.strip():
        raise RuntimeError(f"Gemini returned empty text: {data}")
    return text


def _gemini_post(url: str, headers: dict, payload: dict, timeout_s: float) -> dict:
    with httpx.Client(timeout=timeout_s) as client:
        response = client.post(url, headers=headers, json=payload)
        if response.status_code >= 400:
            raise RuntimeError(
                f"Gemini HTTP {response.status_code}: {_gemini_error_message(response)}"
            )
        data = response.json()
        if not isinstance(data, dict):
            raise RuntimeError("Gemini returned a non-object JSON body")
        return data


def _gemini_error_message(response: httpx.Response) -> str:
    try:
        payload = response.json()
        err = payload.get("error") if isinstance(payload, dict) else None
        if isinstance(err, dict) and err.get("message"):
            return str(err["message"])
    except Exception:
        pass
    return (response.text or "")[:400]


def _gemini_response_text(data: dict) -> str:
    candidates = data.get("candidates") or []
    if not candidates:
        feedback = data.get("promptFeedback") or data.get("error") or data
        raise RuntimeError(f"Gemini returned no candidates: {feedback}")
    parts = ((candidates[0].get("content") or {}).get("parts")) or []
    chunks: list[str] = []
    for part in parts:
        if not isinstance(part, dict):
            continue
        if part.get("thought"):
            continue
        text = part.get("text")
        if text:
            chunks.append(str(text))
    return "\n".join(chunks).strip()
