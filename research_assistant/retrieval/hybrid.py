from __future__ import annotations

import logging
import os
import time
import xml.etree.ElementTree as ET
from urllib.parse import quote_plus

import httpx

from research_assistant.arxiv_ids import normalize_arxiv_id

logger = logging.getLogger(__name__)

ARXIV_ATOM = "{http://www.w3.org/2005/Atom}"
ARXIV_NS = "{http://arxiv.org/schemas/atom}"


def reciprocal_rank_fusion(rankings: list[list[str]], k: int = 60) -> dict[str, float]:
    """RRF over lists of arxiv_id already ordered best-first."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        seen: set[str] = set()
        for rank, doc_id in enumerate(ranking, start=1):
            if doc_id in seen:
                continue
            seen.add(doc_id)
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return scores


def search_arxiv_keyword(topic: str, n: int = 50, timeout_s: float = 20.0) -> list[str]:
    """Return arXiv ids from the official keyword API (best-effort)."""
    query = quote_plus(topic)
    url = (
        "https://export.arxiv.org/api/query"
        f"?search_query=all:{query}&start=0&max_results={int(n)}"
    )
    with httpx.Client(timeout=timeout_s, follow_redirects=True) as client:
        response = client.get(url, headers={"User-Agent": "research-assistant-agent/0.1"})
        response.raise_for_status()
        root = ET.fromstring(response.text)
    ids: list[str] = []
    for entry in root.findall(f"{ARXIV_ATOM}entry"):
        raw = (entry.findtext(f"{ARXIV_ATOM}id") or "").strip()
        arxiv_id = normalize_arxiv_id(raw)
        if arxiv_id:
            ids.append(arxiv_id)
    return ids


def semantic_scholar_api_key() -> str | None:
    for name in ("SEMANTIC_SCHOLAR_API_KEY", "S2_API_KEY"):
        raw = os.environ.get(name)
        if raw and raw.strip():
            return raw.strip().strip('"').strip("'")
    return None


def fetch_citation_counts(arxiv_ids: list[str], timeout_s: float = 30.0) -> dict[str, int]:
    """Lookup citationCount via Semantic Scholar /paper/batch (max 500 ids / request)."""
    if not arxiv_ids:
        return {}

    api_key = semantic_scholar_api_key()
    headers = {
        "User-Agent": "research-assistant-agent/0.1",
        "Content-Type": "application/json",
    }
    if api_key:
        headers["x-api-key"] = api_key
        logger.info("Semantic Scholar: API key loaded, sending one /paper/batch request")
    else:
        logger.warning(
            "Semantic Scholar: no API key loaded. Set SEMANTIC_SCHOLAR_API_KEY in .env; "
            "unauthenticated traffic is shared and often returns 429."
        )

    unique_ids = list(dict.fromkeys(str(x) for x in arxiv_ids if x))
    counts: dict[str, int] = {}
    # One request can carry the whole candidate pool; avoids bursting 1 req/s limit.
    batch_size = 500
    with httpx.Client(timeout=timeout_s, follow_redirects=True) as client:
        for start in range(0, len(unique_ids), batch_size):
            chunk = unique_ids[start : start + batch_size]
            rows = _post_paper_batch(client, chunk, headers)
            if rows is None:
                break
            for aid, row in zip(chunk, rows):
                if not isinstance(row, dict):
                    continue
                n = row.get("citationCount")
                if isinstance(n, bool):
                    continue
                if isinstance(n, int):
                    counts[aid] = n
                elif isinstance(n, float) and n.is_integer():
                    counts[aid] = int(n)
            if start + batch_size < len(unique_ids):
                time.sleep(1.1)

    logger.info(
        "Semantic Scholar: got citationCount for %s/%s papers",
        f"{len(counts):,}",
        f"{len(unique_ids):,}",
    )
    return counts


def _post_paper_batch(
    client: httpx.Client,
    arxiv_ids: list[str],
    headers: dict[str, str],
    max_attempts: int = 6,
) -> list | None:
    payload = {"ids": [f"ARXIV:{aid}" for aid in arxiv_ids]}
    url = "https://api.semanticscholar.org/graph/v1/paper/batch"
    last_error = None
    for attempt in range(1, max_attempts + 1):
        try:
            response = client.post(
                url,
                params={"fields": "citationCount,externalIds"},
                json=payload,
                headers=headers,
            )
        except httpx.HTTPError as exc:
            last_error = exc
            logger.warning("Semantic Scholar request failed (%s); retry %s/%s", exc, attempt, max_attempts)
            time.sleep(min(2 ** attempt, 20))
            continue

        if response.status_code == 429:
            retry_after = response.headers.get("Retry-After")
            try:
                wait_s = float(retry_after) if retry_after else min(2 ** attempt, 20)
            except ValueError:
                wait_s = min(2 ** attempt, 20)
            logger.warning(
                "Semantic Scholar 429 (attempt %s/%s); waiting %.1fs",
                attempt,
                max_attempts,
                wait_s,
            )
            time.sleep(wait_s)
            continue

        if response.status_code >= 400:
            logger.warning(
                "Semantic Scholar HTTP %s: %s",
                response.status_code,
                (response.text or "")[:300],
            )
            return None

        data = response.json()
        if not isinstance(data, list):
            logger.warning("Semantic Scholar batch returned non-list JSON: %s", type(data).__name__)
            return None
        return data

    logger.warning("Semantic Scholar lookup failed after retries: %s", last_error)
    return None


def _normalize_arxiv_id(url_or_id: str) -> str:
    """Backward-compatible alias used by tests."""
    return normalize_arxiv_id(url_or_id)
