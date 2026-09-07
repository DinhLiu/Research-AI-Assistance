"""Two-channel TF-IDF features: method text and keyword phrases."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# Versioned with SynthesisConfig.normalizer_version. Keep this map narrow.
KEYWORD_ALIASES = {
    "data pruning": "dataset pruning",
    "data prune": "dataset pruning",
}


def normalize_text(value: str) -> str:
    text = unicodedata.normalize("NFKC", value or "")
    text = text.casefold()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_keyword(value: str) -> str:
    text = normalize_text(value).replace("_", " ")
    text = re.sub(r"[-–—]", " ", text)
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()
    return KEYWORD_ALIASES.get(text, text)


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return matrix / norms


def _keyword_analyzer(doc: str) -> list[str]:
    if not doc:
        return []
    return [token for token in doc.split("\t") if token]


@dataclass
class FeatureMatrix:
    paper_keys: list[str]
    matrix: np.ndarray
    empty_keys: list[str]
    channel_used: str
    settings: dict = field(default_factory=dict)


def build_features(
    paper_keys: list[str],
    method_texts: list[str],
    keyword_lists: list[list[str]],
    *,
    keyword_weight: float = 0.5,
    max_features: int = 2000,
) -> FeatureMatrix:
    if not paper_keys:
        return FeatureMatrix(
            paper_keys=[],
            matrix=np.zeros((0, 1), dtype=np.float64),
            empty_keys=[],
            channel_used="none",
            settings={"keyword_weight": keyword_weight, "max_features": max_features},
        )

    text_docs = [normalize_text(text) for text in method_texts]
    keyword_docs = ["\t".join(normalize_keyword(item) for item in kws if normalize_keyword(item)) for kws in keyword_lists]

    text_matrix = _fit_text(text_docs, max_features=max_features)
    keyword_matrix = _fit_keywords(keyword_docs, max_features=max_features)

    weight = float(np.clip(keyword_weight, 0.0, 1.0))
    parts: list[np.ndarray] = []
    channels: list[str] = []
    if text_matrix is not None:
        parts.append(text_matrix if keyword_matrix is None else np.sqrt(1.0 - weight) * text_matrix)
        channels.append("text")
    if keyword_matrix is not None:
        parts.append(keyword_matrix if text_matrix is None else np.sqrt(weight) * keyword_matrix)
        channels.append("keywords")

    if not parts:
        return FeatureMatrix(
            paper_keys=list(paper_keys),
            matrix=np.zeros((0, 1), dtype=np.float64),
            empty_keys=list(paper_keys),
            channel_used="none",
            settings={"keyword_weight": weight, "max_features": max_features},
        )

    combined = np.hstack(parts)
    row_norms = np.linalg.norm(combined, axis=1)
    empty_mask = row_norms < 1e-12
    empty_keys = [key for key, flag in zip(paper_keys, empty_mask) if flag]
    usable_idx = np.where(~empty_mask)[0]
    if usable_idx.size == 0:
        return FeatureMatrix(
            paper_keys=[],
            matrix=np.zeros((0, combined.shape[1]), dtype=np.float64),
            empty_keys=list(paper_keys),
            channel_used="+".join(channels),
            settings={"keyword_weight": weight, "max_features": max_features},
        )

    usable_keys = [paper_keys[i] for i in usable_idx]
    usable = _l2_normalize(combined[usable_idx])
    channel_used = "both" if len(channels) == 2 else channels[0]
    return FeatureMatrix(
        paper_keys=usable_keys,
        matrix=usable,
        empty_keys=empty_keys,
        channel_used=channel_used,
        settings={
            "keyword_weight": weight,
            "max_features": max_features,
            "text_ngrams": (1, 2),
            "sublinear_tf": True,
            "min_df": 1,
        },
    )


def _fit_text(docs: list[str], *, max_features: int) -> np.ndarray | None:
    if not any(docs):
        return None
    try:
        vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=1,
            sublinear_tf=True,
            max_features=max_features,
            token_pattern=r"(?u)\b\w+\b",
        )
        matrix = vectorizer.fit_transform(docs).toarray().astype(np.float64)
    except ValueError:
        return None
    if matrix.shape[1] == 0:
        return None
    return matrix


def _fit_keywords(docs: list[str], *, max_features: int) -> np.ndarray | None:
    if not any(docs):
        return None
    try:
        vectorizer = TfidfVectorizer(
            analyzer=_keyword_analyzer,
            min_df=1,
            sublinear_tf=True,
            max_features=max_features,
        )
        matrix = vectorizer.fit_transform(docs).toarray().astype(np.float64)
    except ValueError:
        return None
    if matrix.shape[1] == 0:
        return None
    return matrix
