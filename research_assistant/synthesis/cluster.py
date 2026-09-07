"""Average-linkage agglomerative clustering on a sanitized cosine distance matrix."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_distances

from research_assistant.synthesis.features import normalize_keyword
from research_assistant.synthesis.types import ClusterAssignment, PaperCard, UnassignedRecord


def cluster_id_for(paper_keys: list[str], clustering_version: str) -> str:
    payload = json.dumps(
        {"keys": sorted(paper_keys), "v": clustering_version},
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def cosine_distance_matrix(matrix: np.ndarray) -> np.ndarray:
    distances = cosine_distances(matrix).astype(np.float64)
    distances = np.clip(distances, 0.0, None)
    distances = 0.5 * (distances + distances.T)
    np.fill_diagonal(distances, 0.0)
    return distances


def cluster_papers(
    paper_keys: list[str],
    matrix: np.ndarray,
    cards: list[PaperCard],
    *,
    distance_threshold: float,
    clustering_version: str,
    empty_keys: list[str] | None = None,
) -> tuple[list[ClusterAssignment], list[UnassignedRecord], dict]:
    unassigned = [UnassignedRecord(paper_key=key, reason="empty_features") for key in (empty_keys or [])]
    diagnostics = {
        "silhouette": None,
        "silhouette_reason": None,
        "ambiguity_warning": False,
        "n_clusters": 0,
        "n_singletons": 0,
        "n_clustered": 0,
    }
    if not paper_keys:
        return [], unassigned, diagnostics

    if len(paper_keys) == 1:
        assignments = [
            _assignment(
                [paper_keys[0]],
                cards,
                distances=np.zeros((1, 1)),
                clustering_version=clustering_version,
            )
        ]
        diagnostics.update({"n_clusters": 1, "n_singletons": 1, "n_clustered": 1, "silhouette_reason": "n=1"})
        return assignments, unassigned, diagnostics

    distances = cosine_distance_matrix(matrix)
    model = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=float(distance_threshold),
        metric="precomputed",
        linkage="average",
    )
    labels = model.fit_predict(distances)
    grouped: dict[int, list[int]] = defaultdict(list)
    for index, label in enumerate(labels):
        grouped[int(label)].append(index)

    assignments = []
    for indices in grouped.values():
        keys = [paper_keys[i] for i in indices]
        assignments.append(_assignment(keys, cards, distances, clustering_version, index_map=paper_keys))
    assignments.sort(key=lambda item: item.cluster_id)

    unique = {int(label) for label in labels}
    n_labels = len(unique)
    n_samples = len(paper_keys)
    if n_labels < 2 or n_labels >= n_samples:
        diagnostics["silhouette_reason"] = "label-count precondition not met (need 2..n-1 clusters)"
    else:
        score = float(silhouette_score(distances, labels, metric="precomputed"))
        diagnostics["silhouette"] = round(score, 4)
        diagnostics["ambiguity_warning"] = score < 0.15

    diagnostics["n_clusters"] = len(assignments)
    diagnostics["n_singletons"] = sum(1 for item in assignments if item.size == 1)
    diagnostics["n_clustered"] = n_samples
    return assignments, unassigned, diagnostics


def _assignment(
    keys: list[str],
    cards: list[PaperCard],
    distances: np.ndarray,
    clustering_version: str,
    index_map: list[str] | None = None,
) -> ClusterAssignment:
    ordered = sorted(keys)
    by_key = {card.paper_key: card for card in cards}
    others = [card.paper_key for card in cards if card.paper_key not in set(ordered)]
    label, top = _label(ordered, by_key, others)
    mean_within = None
    nearest = None
    if index_map is not None and distances.size:
        idxs = [index_map.index(key) for key in ordered]
        if len(idxs) >= 2:
            pairs = [distances[i, j] for a, i in enumerate(idxs) for j in idxs[a + 1 :]]
            mean_within = round(float(np.mean(pairs)), 4) if pairs else None
        outside = [i for i, key in enumerate(index_map) if key not in set(ordered)]
        if idxs and outside:
            nearest = round(float(min(distances[i, j] for i in idxs for j in outside)), 4)
    return ClusterAssignment(
        cluster_id=cluster_id_for(ordered, clustering_version),
        label=label,
        paper_keys=ordered,
        top_keywords=top,
        size=len(ordered),
        mean_within_distance=mean_within,
        nearest_other_distance=nearest,
    )


def _label(
    member_keys: list[str],
    by_key: dict[str, PaperCard],
    other_keys: list[str],
) -> tuple[str, list[str]]:
    in_counts: Counter[str] = Counter()
    out_counts: Counter[str] = Counter()
    for key in member_keys:
        card = by_key.get(key)
        if card is None:
            continue
        for keyword in {normalize_keyword(item) for item in card.method_keywords if normalize_keyword(item)}:
            in_counts[keyword] += 1
    for key in other_keys:
        card = by_key.get(key)
        if card is None:
            continue
        for keyword in {normalize_keyword(item) for item in card.method_keywords if normalize_keyword(item)}:
            out_counts[keyword] += 1
    scored = []
    for keyword, count in in_counts.items():
        contrast = count / (count + out_counts.get(keyword, 0))
        scored.append((-count, -contrast, keyword))
    scored.sort()
    top = [keyword for _count, _contrast, keyword in scored[:3]]
    if not top:
        return "unlabeled", []
    return "; ".join(top), top
