"""Offline scoring helpers for synthesis evaluation (not part of the runtime package)."""

from __future__ import annotations

from collections import defaultdict
from itertools import combinations


def exclusive_labels(gold: dict) -> dict[str, str]:
    return {row["arxiv_id"]: row["exclusive_cluster"] for row in gold.get("papers", [])}


def _strip_version(key: str) -> str:
    if "v" in key:
        head, tail = key.rsplit("v", 1)
        if tail.isdigit():
            return head
    return key


def predicted_labels(assignments: list[dict], *, strip_version: bool = True) -> dict[str, str]:
    labels: dict[str, str] = {}
    for cluster in assignments:
        cid = cluster["cluster_id"]
        for key in cluster.get("paper_keys") or []:
            arxiv_id = _strip_version(key) if strip_version else key
            labels[arxiv_id] = cid
    return labels


def _pairs(labels: dict[str, str]) -> set[tuple[str, str]]:
    by_cluster: dict[str, list[str]] = defaultdict(list)
    for item, cluster in labels.items():
        by_cluster[cluster].append(item)
    pairs: set[tuple[str, str]] = set()
    for members in by_cluster.values():
        for left, right in combinations(sorted(members), 2):
            pairs.add((left, right))
    return pairs


def pairwise_precision_recall(predicted: dict[str, str], gold: dict[str, str]) -> dict[str, float]:
    keys = sorted(set(predicted) & set(gold))
    pred = {key: predicted[key] for key in keys}
    truth = {key: gold[key] for key in keys}
    pred_pairs = _pairs(pred)
    gold_pairs = _pairs(truth)
    if not pred_pairs:
        precision = 1.0 if not gold_pairs else 0.0
    else:
        precision = len(pred_pairs & gold_pairs) / len(pred_pairs)
    recall = len(pred_pairs & gold_pairs) / len(gold_pairs) if gold_pairs else 1.0
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return {
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "pred_pairs": len(pred_pairs),
        "gold_pairs": len(gold_pairs),
    }


def adjusted_rand_index(predicted: dict[str, str], gold: dict[str, str]) -> float | None:
    keys = sorted(set(predicted) & set(gold))
    if len(keys) < 2:
        return None
    try:
        from sklearn.metrics import adjusted_rand_score
    except ImportError:
        return None
    return round(float(adjusted_rand_score([gold[k] for k in keys], [predicted[k] for k in keys])), 3)


def singleton_rate(assignments: list[dict], n_clustered: int) -> float:
    if n_clustered <= 0:
        return 0.0
    singles = sum(1 for item in assignments if int(item.get("size") or len(item.get("paper_keys") or [])) == 1)
    return round(singles / n_clustered, 3)


def citation_keep_rate(summaries: list[dict], comparisons: list[dict] | None = None) -> dict:
    claims = []
    for summary in summaries or []:
        claims.extend(summary.get("claims") or [])
    claims.extend(comparisons or [])
    if not claims:
        return {"kept": 0, "total": 0, "rate": None}
    kept = sum(1 for claim in claims if claim.get("validation_status") == "structurally_validated")
    total = len(claims)
    return {"kept": kept, "total": total, "rate": round(kept / total, 3) if total else None}
