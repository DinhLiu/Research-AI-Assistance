"""Stage-1 retrieval: query expansion → FAISS → SPECTER2 rerank → optional hybrid RRF."""

from research_assistant.retrieval.pipeline import RetrievalPipeline, retrieve
from research_assistant.retrieval.types import PaperHit, RetrievalResult

__all__ = ["RetrievalPipeline", "retrieve", "PaperHit", "RetrievalResult"]
