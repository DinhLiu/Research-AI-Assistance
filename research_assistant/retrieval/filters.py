from __future__ import annotations

import numpy as np

from research_assistant.config import RetrievalConfig


def passes_filters(meta: dict, cfg: RetrievalConfig) -> bool:
    year = as_int(meta.get("year"))
    if cfg.year_from is not None and (year is None or year < cfg.year_from):
        return False
    if cfg.year_to is not None and (year is None or year > cfg.year_to):
        return False
    if cfg.categories:
        cats = set(str(meta.get("matched_categories") or "").split())
        if cats.isdisjoint(cfg.categories):
            return False
    return True


def as_int(value) -> int | None:
    if value is None:
        return None
    try:
        if isinstance(value, float) and np.isnan(value):
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def topk_overlap(broad_ids: list[str], final_ids: list[str]) -> float:
    if not final_ids:
        return 0.0
    return round(len(set(broad_ids) & set(final_ids)) / len(final_ids), 4)
