"""Shared LLM HTTP client for query expansion and extraction."""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Callable

import httpx

logger = logging.getLogger(__name__)

CompleteFn = Callable[..., str]


class RateLimitError(RuntimeError):
    """Provider returned 429/503; wait Retry-After instead of splitting the batch."""


class LlmBudgetExceeded(RuntimeError):
    """Hard cap on LLM calls for this run was reached."""


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


def complete(
    *,
    system: str,
    user: str,
    json_mode: bool = True,
    temperature: float = 0.0,
    timeout_s: float = 90.0,
    provider: str | None = None,
) -> str:
    chosen = provider or expansion_provider()
    if not chosen:
        raise RuntimeError("No LLM API key set (GROQ_API_KEY / GEMINI_API_KEY / OPENAI_API_KEY)")
    if chosen == "groq":
        return _openai_compatible(
            base_url="https://api.groq.com/openai/v1",
            api_key=env_key("GROQ_API_KEY") or "",
            model=groq_model(),
            system=system,
            user=user,
            json_mode=json_mode,
            temperature=temperature,
            timeout_s=timeout_s,
        )
    if chosen == "openai":
        return _openai_compatible(
            base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            api_key=env_key("OPENAI_API_KEY") or "",
            model=openai_model(),
            system=system,
            user=user,
            json_mode=json_mode,
            temperature=temperature,
            timeout_s=timeout_s,
        )
    if chosen == "gemini":
        return _gemini_complete(
            system=system,
            user=user,
            json_mode=json_mode,
            temperature=temperature,
            timeout_s=timeout_s,
        )
    raise ValueError(f"Unknown LLM provider: {chosen}")


def _openai_compatible(
    *,
    base_url: str,
    api_key: str,
    model: str,
    system: str,
    user: str,
    json_mode: bool,
    temperature: float,
    timeout_s: float,
) -> str:
    url = base_url.rstrip("/") + "/chat/completions"
    payload: dict = {
        "model": model,
        "temperature": temperature,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    with httpx.Client(timeout=timeout_s, headers={"User-Agent": USER_AGENT}) as client:
        response = _post_with_backoff(
            client,
            url,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=payload,
        )
        data = response.json()
    return data["choices"][0]["message"]["content"]


def _gemini_complete(
    *,
    system: str,
    user: str,
    json_mode: bool,
    temperature: float,
    timeout_s: float,
) -> str:
    api_key = env_key("GEMINI_API_KEY") or env_key("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY / GOOGLE_API_KEY is not set")
    model = gemini_model()
    logger.info("Calling Gemini model=%s", model)
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    headers = {
        "x-goog-api-key": api_key,
        "Content-Type": "application/json",
        "User-Agent": USER_AGENT,
    }
    generation: dict = {"temperature": temperature}
    if json_mode:
        generation["responseMimeType"] = "application/json"
    payload = {
        "system_instruction": {"parts": [{"text": system}]},
        "contents": [{"role": "user", "parts": [{"text": user}]}],
        "generationConfig": generation,
    }
    data = _gemini_post(url, headers, payload, timeout_s)
    text = _gemini_response_text(data)
    if not text.strip():
        raise RuntimeError(f"Gemini returned empty text: {data}")
    return text


def _gemini_post(url: str, headers: dict, payload: dict, timeout_s: float) -> dict:
    with httpx.Client(timeout=timeout_s) as client:
        response = _post_with_backoff(client, url, headers=headers, json=payload)
        data = response.json()
        if not isinstance(data, dict):
            raise RuntimeError("Gemini returned a non-object JSON body")
        return data


def _post_with_backoff(
    client: httpx.Client,
    url: str,
    *,
    headers: dict,
    json: dict,
    max_attempts: int = 4,
) -> httpx.Response:
    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            response = client.post(url, headers=headers, json=json)
        except httpx.HTTPError as exc:
            last_error = exc
            time.sleep(min(2 ** attempt, 20))
            continue
        if response.status_code in {429, 503}:
            retry_after = response.headers.get("Retry-After")
            try:
                wait_s = float(retry_after) if retry_after else min(2 ** attempt, 45)
            except ValueError:
                wait_s = min(2 ** attempt, 45)
            logger.warning("LLM HTTP %s; backing off %.1fs", response.status_code, wait_s)
            time.sleep(wait_s)
            last_error = RateLimitError(
                f"LLM HTTP {response.status_code}: {_gemini_error_message(response)}"
            )
            continue
        if response.status_code >= 400:
            raise RuntimeError(
                f"LLM HTTP {response.status_code}: {_gemini_error_message(response)}"
            )
        return response
    if isinstance(last_error, RateLimitError):
        raise last_error
    raise RuntimeError(f"LLM request failed after retries: {last_error}")


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
