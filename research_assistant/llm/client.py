"""Shared LLM HTTP client for query expansion and extraction."""

from __future__ import annotations

import logging
import os
from collections.abc import Callable

import httpx

logger = logging.getLogger(__name__)

CompleteFn = Callable[..., str]


from dataclasses import dataclass
from uuid import uuid4

from research_assistant.llm.governor import (
    Governor, Quota, RunBudget, LlmError, RateLimitError, LlmBudgetExceeded, current_budget,
)


@dataclass
class Completion:
    text: str
    provider: str
    model: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    finish_reason: str | None = None

    @property
    def truncated(self):
        return self.finish_reason in {"length", "MAX_TOKENS"}


USER_AGENT = "research-assistant-agent/0.1"

DEFAULT_GROQ_MODEL = "llama-3.1-8b-instant"
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"


def env_key(name: str) -> str | None:
    raw = os.environ.get(name)
    if not raw:
        return None
    value = raw.strip().strip('"').strip("'")
    return value or None


def groq_model() -> str:
    return (os.environ.get("GROQ_MODEL") or DEFAULT_GROQ_MODEL).strip()


def gemini_model() -> str:
    model = (os.environ.get("GEMINI_MODEL") or DEFAULT_GEMINI_MODEL).strip()
    return model.removeprefix("models/")


def openai_model() -> str:
    return (os.environ.get("OPENAI_MODEL") or DEFAULT_OPENAI_MODEL).strip()


def provider_available(provider: str) -> bool:
    if provider == "groq":
        return bool(env_key("GROQ_API_KEY"))
    if provider == "gemini":
        return bool(env_key("GEMINI_API_KEY") or env_key("GOOGLE_API_KEY"))
    if provider == "openai":
        return bool(env_key("OPENAI_API_KEY"))
    return False


def expansion_provider() -> str | None:
    """Same cascade as stage-1 query expansion: Groq → Gemini → OpenAI."""
    for name in ("groq", "gemini", "openai"):
        if provider_available(name):
            return name
    return None


def extraction_provider(prefer: str = "gemini") -> str | None:
    """Gemini first (long context + JSON mode). Groq is last for extraction."""
    order = ["gemini", "openai", "groq"]
    prefer = (prefer or "gemini").strip().lower()
    if prefer in order:
        order = [prefer] + [name for name in order if name != prefer]
    for name in order:
        if provider_available(name):
            return name
    return None


def model_for_provider(provider: str) -> str:
    if provider == "groq":
        return groq_model()
    if provider == "gemini":
        return gemini_model()
    if provider == "openai":
        return openai_model()
    raise ValueError(f"Unknown LLM provider: {provider}")


def complete(**kwargs) -> str:
    """Compatibility wrapper; all callers share persistent admission control."""
    result = complete_result(**kwargs)
    if result.truncated:
        raise LlmError("output_truncated")
    return result.text


def estimate_tokens(text: str) -> int:
    # Conservative UTF-8 byte upper estimate, deliberately not len(text)/4.
    return len(text.encode("utf-8")) + 64


def complete_result(
    *, system: str, user: str, json_mode: bool = True,
    temperature: float = 0.0, timeout_s: float = 90.0,
    provider: str | None = None, max_output_tokens: int = 4096,
    governor: Governor | None = None, quota: Quota | None = None,
    budget: RunBudget | None = None,
) -> Completion:
    chosen = provider or expansion_provider()
    if not chosen or not provider_available(chosen):
        raise LlmError("no_api_key")
    if max_output_tokens <= 0:
        raise ValueError("max_output_tokens must be positive")
    model = model_for_provider(chosen)
    if chosen == "gemini":
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        headers = {"x-goog-api-key": env_key("GEMINI_API_KEY") or env_key("GOOGLE_API_KEY") or ""}
        generation = {"temperature": temperature, "maxOutputTokens": max_output_tokens}
        if json_mode:
            generation["responseMimeType"] = "application/json"
        payload = {
            "system_instruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": generation,
        }
    else:
        base = "https://api.groq.com/openai/v1" if chosen == "groq" else os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
        url = base.rstrip("/") + "/chat/completions"
        headers = {"Authorization": f"Bearer {env_key('GROQ_API_KEY' if chosen == 'groq' else 'OPENAI_API_KEY')}"}
        payload = {
            "model": model, "temperature": temperature,
            "max_tokens": max_output_tokens,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
    try:
        active_quota = quota or Quota.from_env()
    except ValueError as exc:
        raise LlmError("invalid_quota_configuration") from exc
    control = governor or Governor()
    active_budget = budget or current_budget() or RunBudget(str(uuid4()))
    headers.update({"Content-Type": "application/json", "User-Agent": USER_AGENT})
    group = os.environ.get("LLM_QUOTA_GROUP", chosen)
    with httpx.Client() as client:
        response = control.request(
            lambda timeout: client.post(url, headers=headers, json=payload, timeout=timeout),
            group=group, quota=active_quota, budget=active_budget,
            tokens=estimate_tokens(system + user) + max_output_tokens,
            timeout_s=timeout_s,
        )
    try:
        data = response.json()
        if chosen == "gemini":
            text = _gemini_response_text(data)
            usage = data.get("usageMetadata") or {}
            result = Completion(text, chosen, model, usage.get("promptTokenCount"),
                                usage.get("candidatesTokenCount"),
                                ((data.get("candidates") or [{}])[0]).get("finishReason"))
        else:
            choice = data["choices"][0]
            usage = data.get("usage") or {}
            result = Completion(choice["message"]["content"] or "", chosen, model,
                                usage.get("prompt_tokens"), usage.get("completion_tokens"), choice.get("finish_reason"))
        if not result.text.strip():
            raise ValueError("empty_response")
        return result
    except (ValueError, KeyError, TypeError, IndexError, AttributeError) as exc:
        raise LlmError("invalid_provider_response") from exc


def _gemini_response_text(data: dict) -> str:
    candidates = data.get("candidates") or []
    if not candidates:
        return ""
    parts = ((candidates[0].get("content") or {}).get("parts")) or []
    return "\n".join(str(part["text"]) for part in parts
                     if isinstance(part, dict) and part.get("text") and not part.get("thought")).strip()
