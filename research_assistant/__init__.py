"""Research Assistance Agent — stage 1 is retrieve-then-rerank over SPECTER2 embeddings."""

from research_assistant.retrieval.pipeline import RetrievalPipeline, retrieve
from research_assistant.retrieval.types import PaperHit, RetrievalResult

__all__ = ["RetrievalPipeline", "retrieve", "PaperHit", "RetrievalResult"]
