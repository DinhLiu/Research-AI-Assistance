"""Research Assistance Agent — retrieve, extract, then synthesize."""

from research_assistant.extraction.pipeline import extract_papers
from research_assistant.extraction.types import ExtractedPaper, ExtractionResult
from research_assistant.retrieval.pipeline import RetrievalPipeline, retrieve
from research_assistant.retrieval.types import PaperHit, RetrievalResult
from research_assistant.synthesis.pipeline import synthesize
from research_assistant.synthesis.types import SynthesisResult

from research_assistant.writing import write_review, WritingResult

__all__ = [
    "write_review",
    "WritingResult",
    "RetrievalPipeline",
    "retrieve",
    "extract_papers",
    "synthesize",
    "PaperHit",
    "RetrievalResult",
    "ExtractedPaper",
    "ExtractionResult",
    "SynthesisResult",
]
