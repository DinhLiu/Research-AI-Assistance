"""Research Assistance Agent — retrieve-then-rerank, then grounded extraction."""

from research_assistant.extraction.pipeline import extract_papers
from research_assistant.extraction.types import ExtractedPaper, ExtractionResult
from research_assistant.retrieval.pipeline import RetrievalPipeline, retrieve
from research_assistant.retrieval.types import PaperHit, RetrievalResult

__all__ = [
    "RetrievalPipeline",
    "retrieve",
    "extract_papers",
    "PaperHit",
    "RetrievalResult",
    "ExtractedPaper",
    "ExtractionResult",
]
