from __future__ import annotations

import numpy as np

from research_assistant.config import RetrievalConfig
from research_assistant.retrieval.expand import TemplateQueryExpander
from research_assistant.retrieval.pipeline import RetrievalPipeline


class FakeEncoder:
    def encode(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        # Identical queries → first dim = 1 so rerank uses stored paper dim-0.
        return np.tile(np.array([[1.0, 0.0]], dtype=np.float32), (len(texts), 1))


class FakeCorpus:
    def search(self, queries, k):
        n = len(queries)
        scores = np.array([[0.9, 0.4]] * n, dtype=np.float32)
        ids = np.array([[0, 1]] * n, dtype=np.int64)
        return scores, ids

    def lookup_metadata(self, row_ids):
        catalog = {
            0: {
                "row_id": 0,
                "arxiv_id": "2205.09329",
                "title": "Dataset Pruning",
                "abstract": "Reduce training data.",
                "authors": "A B",
                "year": 2022,
                "matched_categories": "cs.LG",
                "latest_version": "v1",
                "doi": None,
            },
            1: {
                "row_id": 1,
                "arxiv_id": "2312.05599",
                "title": "Not All Data Matters",
                "abstract": "Adaptive data selection.",
                "authors": "C D",
                "year": 2023,
                "matched_categories": "cs.AI cs.LG",
                "latest_version": "v2",
                "doi": None,
            },
        }
        return [catalog[int(i)] for i in row_ids]

    def lookup_embeddings(self, row_ids):
        vecs = {
            0: np.array([1.0, 0.0], dtype=np.float32),
            1: np.array([0.0, 1.0], dtype=np.float32),
        }
        return np.stack([vecs[int(i)] for i in row_ids])

    def arxiv_id_to_row(self):
        return {"2205.09329": 0, "2312.05599": 1}


def test_pipeline_reranks_against_original_topic():
    pipe = RetrievalPipeline(
        RetrievalConfig(top_k=2, candidate_pool_size=2, n_query_variants=1, use_hybrid=False),
        corpus=FakeCorpus(),
        encoder=FakeEncoder(),
        expander=TemplateQueryExpander(),
    )
    result = pipe.search("dataset pruning")
    assert result.queries[0] == "dataset pruning"
    assert [p.arxiv_id for p in result.papers] == ["2205.09329", "2312.05599"]
    assert result.papers[0].rerank_score > result.papers[1].rerank_score
    assert result.metrics["unique_after_union"] == 2


def test_hybrid_keyword_hit_enters_pool(monkeypatch):
    class SparseCorpus(FakeCorpus):
        def search(self, queries, k):
            n = len(queries)
            scores = np.array([[0.9]] * n, dtype=np.float32)
            ids = np.array([[0]] * n, dtype=np.int64)
            return scores, ids

    monkeypatch.setattr(
        "research_assistant.retrieval.pipeline.search_arxiv_keyword",
        lambda topic, n=50: ["2312.05599"],
    )
    pipe = RetrievalPipeline(
        RetrievalConfig(top_k=2, candidate_pool_size=1, n_query_variants=1, use_hybrid=True),
        corpus=SparseCorpus(),
        encoder=FakeEncoder(),
        expander=TemplateQueryExpander(),
    )
    result = pipe.search("dataset pruning")
    ids = {p.arxiv_id for p in result.papers}
    assert "2205.09329" in ids
    assert "2312.05599" in ids
    assert result.metrics["keyword_added_to_pool"] == 1
    assert result.papers[0].arxiv_id == "2205.09329"
    assert result.papers[0].rerank_score > result.papers[1].rerank_score


def test_n_query_variants_zero_keeps_original_only():
    pipe = RetrievalPipeline(
        RetrievalConfig(top_k=2, candidate_pool_size=2, n_query_variants=0, use_hybrid=False),
        corpus=FakeCorpus(),
        encoder=FakeEncoder(),
        expander=TemplateQueryExpander(),
    )
    result = pipe.search("dataset pruning")
    assert result.queries == ["dataset pruning"]
    assert result.metrics["n_query_variants"] == 0


class InvertedFaissCorpus(FakeCorpus):
    def search(self, queries, k):
        n = len(queries)
        scores = np.array([[0.9, 0.4]] * n, dtype=np.float32)
        ids = np.array([[1, 0]] * n, dtype=np.int64)
        return scores, ids


def test_use_rerank_false_keeps_faiss_order():
    shared = dict(
        top_k=2,
        candidate_pool_size=2,
        n_query_variants=0,
        use_hybrid=False,
    )
    reranked = RetrievalPipeline(
        RetrievalConfig(use_rerank=True, **shared),
        corpus=InvertedFaissCorpus(),
        encoder=FakeEncoder(),
        expander=TemplateQueryExpander(),
    ).search("dataset pruning")
    faiss_order = RetrievalPipeline(
        RetrievalConfig(use_rerank=False, **shared),
        corpus=InvertedFaissCorpus(),
        encoder=FakeEncoder(),
        expander=TemplateQueryExpander(),
    ).search("dataset pruning")
    assert [p.arxiv_id for p in reranked.papers] == ["2205.09329", "2312.05599"]
    assert [p.arxiv_id for p in faiss_order.papers] == ["2312.05599", "2205.09329"]
    assert faiss_order.metrics["use_rerank"] is False

