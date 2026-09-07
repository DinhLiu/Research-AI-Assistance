"""Corpus digest, cluster/narration cache keys, and atomic cache IO."""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, is_dataclass
from pathlib import Path

from research_assistant.config import SynthesisConfig
from research_assistant.extraction.types import ExtractedPaper
from research_assistant.synthesis.cards import consumed_payload, record_paper_key
from research_assistant.synthesis.types import EvidenceRegistry, SynthesisResult

logger = logging.getLogger(__name__)


def corpus_digest(records: list[ExtractedPaper]) -> str:
    rows = []
    for paper in sorted(records, key=lambda item: (record_paper_key(item), item.status, item.title)):
        rows.append(consumed_payload(paper))
    raw = json.dumps(rows, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def evidence_digest(registry: EvidenceRegistry) -> str:
    rows = [
        {
            "unit_id": unit.unit_id,
            "paper_key": unit.paper_key,
            "field_path": unit.field_path,
            "kind": unit.kind,
            "text": unit.text,
            "quote": unit.quote,
        }
        for unit in sorted(registry.units, key=lambda item: item.unit_id)
    ]
    raw = json.dumps(rows, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]


def effective_config(config: SynthesisConfig) -> dict:
    payload = asdict(config) if is_dataclass(config) else dict(config.__dict__)
    payload["cache_dir"] = str(payload.get("cache_dir", ""))
    payload["distance_threshold_status"] = "provisional"
    return payload


def cluster_cache_key(digest: str, config: SynthesisConfig) -> str:
    payload = {
        "corpus": digest,
        "strict_ok": config.strict_ok,
        "distance_threshold": config.distance_threshold,
        "keyword_weight": config.keyword_weight,
        "max_features": config.max_features,
        "normalizer_version": config.normalizer_version,
        "clustering_version": config.clustering_version,
        "schema_version": config.schema_version,
        "card_version": config.card_version,
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]


def narration_cache_key(
    cluster_key: str,
    digest: str,
    config: SynthesisConfig,
    *,
    model: str,
    topic: str | None,
) -> str:
    payload = {
        "cluster_key": cluster_key,
        "evidence": digest,
        "topic": topic or "",
        "scope": "extraction_snapshot",
        "prompt_version": config.prompt_version,
        "validator_version": config.validator_version,
        "card_version": config.card_version,
        "model": model,
        "strict_evidence": config.strict_evidence,
        "max_prompt_chars": config.max_prompt_chars,
        "max_method_chars": config.max_method_chars,
        "max_problem_chars": config.max_problem_chars,
        "max_claim_chars": config.max_claim_chars,
        "max_claims_per_field": config.max_claims_per_field,
        "max_quote_chars": config.max_quote_chars,
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def cluster_path(cache_dir: Path, key: str) -> Path:
    return cache_dir / "cluster" / f"{key}.json"


def narration_path(cache_dir: Path, key: str) -> Path:
    return cache_dir / "narration" / f"{key}.json"


def load_result(path: Path) -> SynthesisResult | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return SynthesisResult.model_validate(payload)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        logger.warning("Ignoring corrupt synthesis cache %s: %s", path, exc)
        return None


def save_result(path: Path, result: SynthesisResult) -> None:
    atomic_write(path, result.model_dump_json(indent=2))
