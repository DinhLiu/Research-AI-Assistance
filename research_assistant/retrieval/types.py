from __future__ import annotations

from pydantic import BaseModel, Field


class PaperHit(BaseModel):
    rank: int
    row_id: int
    arxiv_id: str
    title: str
    abstract: str = ""
    authors: str = ""
    year: int | None = None
    matched_categories: str = ""
    latest_version: str | None = None
    doi: str | None = None
    broad_score: float = 0.0
    rerank_score: float = 0.0
    rrf_score: float | None = None
    retrieved_by: list[str] = Field(default_factory=list)
    keyword_rank: int | None = None
    citation_count: int | None = None


class RetrievalResult(BaseModel):
    topic: str
    queries: list[str]
    expander: str
    candidate_pool_size: int
    papers: list[PaperHit]
    metrics: dict = Field(default_factory=dict)
