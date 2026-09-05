from __future__ import annotations

import logging
import time
from collections import defaultdict

from research_assistant.config import RetrievalConfig
from research_assistant.retrieval.expand import QueryExpander, build_expander
from research_assistant.retrieval.filters import as_int, passes_filters, topk_overlap
from research_assistant.retrieval.hybrid import (
    fetch_citation_counts,
    reciprocal_rank_fusion,
    search_arxiv_keyword,
)
from research_assistant.retrieval.types import PaperHit, RetrievalResult

logger = logging.getLogger(__name__)


class RetrievalPipeline:
    def __init__(
        self,
        config: RetrievalConfig | None = None,
        *,
        corpus: ArtifactCorpus | None = None,
        encoder: Specter2QueryEncoder | None = None,
        expander: QueryExpander | None = None,
    ):
        self.config = config or RetrievalConfig()
        if corpus is None:
            from research_assistant.retrieval.corpus import ArtifactCorpus

            corpus = ArtifactCorpus(
                self.config.artifacts_dir,
                rebuild_index_if_missing=self.config.rebuild_index_if_missing,
            )
        if encoder is None:
            from research_assistant.retrieval.encoder import Specter2QueryEncoder

            encoder = Specter2QueryEncoder(
                device=self.config.device,
                use_fp16=self.config.use_fp16,
            )
        self.corpus = corpus
        self.encoder = encoder
        self.expander = expander or build_expander(timeout_s=self.config.llm_timeout_s)

    def search(self, topic: str) -> RetrievalResult:
        cfg = self.config
        t0 = time.perf_counter()
        if cfg.n_query_variants <= 0:
            queries = [" ".join(topic.split())]
        else:
            queries = self.expander.expand(topic, n=cfg.n_query_variants)
        logger.info("Expanded %s queries via %s", len(queries), self.expander.name)

        query_vecs = self.encoder.encode(queries)
        topic_vec = query_vecs[0:1]
        scores_mat, ids_mat = self.corpus.search(query_vecs, cfg.broad_k)

        best_score: dict[int, float] = {}
        retrieved_by: dict[int, list[str]] = defaultdict(list)
        for q_idx, query in enumerate(queries):
            for score, row_id in zip(scores_mat[q_idx], ids_mat[q_idx]):
                rid = int(row_id)
                if rid < 0:
                    continue
                if query not in retrieved_by[rid]:
                    retrieved_by[rid].append(query)
                prev = best_score.get(rid)
                if prev is None or float(score) > prev:
                    best_score[rid] = float(score)

        unique_after_union = len(best_score)
        meta_by_row = {
            item["row_id"]: item
            for item in self.corpus.lookup_metadata(list(best_score))
        }
        filtered_rows = [
            rid for rid in best_score if passes_filters(meta_by_row[rid], cfg)
        ]

        semantic_ranked = sorted(filtered_rows, key=lambda rid: best_score[rid], reverse=True)
        pool_ids = semantic_ranked[: cfg.candidate_pool_size]

        keyword_rank: dict[str, int] = {}
        keyword_added = 0
        if cfg.use_hybrid:
            id_map = self.corpus.arxiv_id_to_row()
            try:
                keyword_ids = search_arxiv_keyword(topic, n=cfg.arxiv_keyword_n)
            except Exception as exc:
                logger.warning("arXiv keyword search failed: %s", exc)
                keyword_ids = []
            mapped: list[tuple[str, int]] = []
            missing_rows: list[int] = []
            for arxiv_id in keyword_ids:
                rid = id_map.get(arxiv_id)
                if rid is None:
                    continue
                mapped.append((arxiv_id, rid))
                if rid not in meta_by_row:
                    missing_rows.append(rid)
            if missing_rows:
                for item in self.corpus.lookup_metadata(missing_rows):
                    meta_by_row[item["row_id"]] = item
                    best_score.setdefault(item["row_id"], 0.0)
                    retrieved_by.setdefault(item["row_id"], [])
            in_corpus: list[str] = []
            for arxiv_id, rid in mapped:
                if not passes_filters(meta_by_row[rid], cfg):
                    continue
                in_corpus.append(arxiv_id)
                if rid not in pool_ids:
                    pool_ids.append(rid)
                    keyword_added += 1
            keyword_rank = {aid: i + 1 for i, aid in enumerate(in_corpus)}

        if len(pool_ids) > cfg.candidate_pool_size + cfg.arxiv_keyword_n:
            pool_ids = pool_ids[: cfg.candidate_pool_size + cfg.arxiv_keyword_n]

        if not pool_ids:
            return RetrievalResult(
                topic=topic,
                queries=queries,
                expander=self.expander.name,
                candidate_pool_size=0,
                papers=[],
                metrics={
                    "n_queries": len(queries),
                    "unique_after_union": unique_after_union,
                    "after_filters": len(filtered_rows),
                    "candidate_pool": 0,
                    "seconds": round(time.perf_counter() - t0, 3),
                },
            )

        if cfg.use_rerank:
            pool_emb = self.corpus.lookup_embeddings(pool_ids)
            rerank_scores = (pool_emb @ topic_vec.T).reshape(-1)
            rerank_by_row = {rid: float(rerank_scores[i]) for i, rid in enumerate(pool_ids)}
        else:
            rerank_by_row = {rid: float(best_score[rid]) for rid in pool_ids}
        rerank_order = sorted(pool_ids, key=lambda rid: rerank_by_row[rid], reverse=True)

        citation_counts: dict[str, int] = {}
        if cfg.use_citations:
            arxiv_ids = [meta_by_row[rid]["arxiv_id"] for rid in rerank_order]
            citation_counts = fetch_citation_counts(arxiv_ids)

        rankings = [[meta_by_row[rid]["arxiv_id"] for rid in rerank_order]]
        if cfg.use_hybrid and keyword_rank:
            rankings.append(list(keyword_rank))
        if cfg.use_citations and citation_counts:
            cite_order = sorted(
                (rid for rid in rerank_order if meta_by_row[rid]["arxiv_id"] in citation_counts),
                key=lambda rid: citation_counts[meta_by_row[rid]["arxiv_id"]],
                reverse=True,
            )
            rankings.append([meta_by_row[rid]["arxiv_id"] for rid in cite_order])

        rrf_scores = reciprocal_rank_fusion(rankings, k=cfg.rrf_k) if len(rankings) > 1 else {}
        # Primary key is SPECTER2 cosine. RRF is a small additive boost (~0.01–0.03)
        # so keyword/citation evidence can move close ties, not replace semantic rank.
        def _fused(rid: int) -> float:
            aid = meta_by_row[rid]["arxiv_id"]
            return rerank_by_row[rid] + rrf_scores.get(aid, 0.0)

        final_order = sorted(rerank_order, key=_fused, reverse=True)

        papers: list[PaperHit] = []
        for rank, rid in enumerate(final_order[: cfg.top_k], start=1):
            meta = meta_by_row[rid]
            aid = str(meta["arxiv_id"])
            papers.append(
                PaperHit(
                    rank=rank,
                    row_id=rid,
                    arxiv_id=aid,
                    title=str(meta.get("title") or ""),
                    abstract=str(meta.get("abstract") or ""),
                    authors=str(meta.get("authors") or ""),
                    year=as_int(meta.get("year")),
                    matched_categories=str(meta.get("matched_categories") or ""),
                    latest_version=meta.get("latest_version"),
                    doi=meta.get("doi"),
                    broad_score=best_score.get(rid, 0.0),
                    rerank_score=rerank_by_row[rid],
                    rrf_score=rrf_scores.get(aid),
                    retrieved_by=retrieved_by.get(rid, []),
                    keyword_rank=keyword_rank.get(aid),
                    citation_count=citation_counts.get(aid),
                )
            )

        overlap = topk_overlap(
            [meta_by_row[rid]["arxiv_id"] for rid in semantic_ranked[: cfg.top_k]],
            [p.arxiv_id for p in papers],
        )
        metrics = {
            "n_queries": len(queries),
            "unique_after_union": unique_after_union,
            "after_filters": len(filtered_rows),
            "candidate_pool": len(pool_ids),
            "keyword_hits_in_corpus": len(keyword_rank),
            "keyword_added_to_pool": keyword_added,
            "citation_requested": len(rerank_order) if cfg.use_citations else 0,
            "citation_found": len(citation_counts),
            "top_k_overlap_broad_vs_final": overlap,
            "expander": self.expander.name,
            "use_rerank": cfg.use_rerank,
            "use_hybrid": cfg.use_hybrid,
            "n_query_variants": cfg.n_query_variants,
            "seconds": round(time.perf_counter() - t0, 3),
        }
        logger.info("Retrieval metrics: %s", metrics)
        return RetrievalResult(
            topic=topic,
            queries=queries,
            expander=self.expander.name,
            candidate_pool_size=len(pool_ids),
            papers=papers,
            metrics=metrics,
        )


def retrieve(topic: str, config: RetrievalConfig | None = None) -> RetrievalResult:
    return RetrievalPipeline(config).search(topic)
