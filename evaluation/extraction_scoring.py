"""Offline scoring helpers for extraction evaluation (not part of the runtime package)."""

from __future__ import annotations

from collections.abc import Iterable


def _norm(value: str) -> str:
    return " ".join(value.lower().split())


def precision_recall(predicted: Iterable[str], gold: Iterable[str]) -> dict[str, float]:
    pred = {_norm(x) for x in predicted if str(x).strip()}
    gold_set = {_norm(x) for x in gold if str(x).strip()}
    if not pred:
        precision = 1.0 if not gold_set else 0.0
    else:
        precision = len(pred & gold_set) / len(pred)
    recall = len(pred & gold_set) / len(gold_set) if gold_set else 1.0
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return {
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "jaccard": round(len(pred & gold_set) / len(pred | gold_set), 3) if pred or gold_set else 1.0,
    }


def completeness(records: list[dict]) -> float:
    if not records:
        return 0.0
    ok = 0
    for row in records:
        if row.get("status") != "ok":
            continue
        if row.get("method") and row.get("results"):
            ok += 1
    return round(ok / len(records), 3)


def evidence_keep_rate(records: list[dict]) -> float:
    proposed = sum(int(r.get("units_proposed") or 0) for r in records)
    kept = sum(int(r.get("units_kept") or 0) for r in records)
    if not proposed:
        return 0.0
    return round(kept / proposed, 3)


def evidence_precision(labels: Iterable[str]) -> dict[str, float]:
    labels = [str(x) for x in labels]
    if not labels:
        return {"supported": 0.0, "partially_supported": 0.0, "unsupported": 0.0, "n": 0}
    n = len(labels)
    return {
        "supported": round(labels.count("supported") / n, 3),
        "partially_supported": round(labels.count("partially_supported") / n, 3),
        "unsupported": round(labels.count("unsupported") / n, 3),
        "n": n,
    }


def mention_values(paper: dict, field: str) -> list[str]:
    return [item.get("value", "") for item in paper.get(field) or [] if isinstance(item, dict)]
