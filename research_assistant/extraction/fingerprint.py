"""Pipeline fingerprint for extraction cache keys."""

from __future__ import annotations

import hashlib
import json

from research_assistant.config import ExtractionConfig
from research_assistant.extraction.prompt import PROMPT_VERSION
from research_assistant.llm.client import extraction_provider, model_for_provider


def fingerprint_model(config: ExtractionConfig, provider: str | None = None) -> str:
    chosen = provider or extraction_provider(config.prefer_provider) or "none"
    if chosen == "none":
        return "none"
    return f"{chosen}:{model_for_provider(chosen)}"


def pipeline_fingerprint(config: ExtractionConfig, model: str | None = None) -> str:
    payload = {
        "schema_version": config.schema_version,
        "prompt_version": config.prompt_version or PROMPT_VERSION,
        "selector_version": config.selector_version,
        "normalizer_version": config.normalizer_version,
        "candidate_version": getattr(config, "candidate_version", "1"),
        "llm_batch_size": getattr(config, "llm_batch_size", 1),
        "model": model or fingerprint_model(config),
        "policy": {
            "include_related_work": config.include_related_work,
            "include_appendix": config.include_appendix,
            "include_bibliography": config.include_bibliography,
            "abstract_only": config.abstract_only,
            "max_input_chars": config.max_input_chars,
            "max_input_tokens": config.max_input_tokens,
        },
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]
