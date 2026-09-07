"""Example: extract grounded records from a retrieval JSON or a single arXiv id."""

from __future__ import annotations

from pathlib import Path

from research_assistant.config import ExtractionConfig
from research_assistant.extraction import extract_papers
from research_assistant.retrieval.types import PaperHit, RetrievalResult


if __name__ == "__main__":
    retrieval_path = Path("results/pruning.json")
    if retrieval_path.is_file():
        result = RetrievalResult.model_validate_json(retrieval_path.read_text(encoding="utf-8"))
        papers = result.papers[:5]
        topic = result.topic
    else:
        topic = "dataset pruning"
        papers = [
            PaperHit(
                rank=1,
                row_id=-1,
                arxiv_id="2205.09329",
                title="Dataset Pruning: Reducing Training Data by Examining Generalization Influence",
                latest_version="v1",
            )
        ]
    extracted = extract_papers(papers, ExtractionConfig(llm_concurrency=1), topic=topic)
    print(extracted.metrics)
    for paper in extracted.records:
        keywords = [m.value for m in paper.method_keywords]
        print(f"{paper.status:9} {paper.arxiv_id} method={paper.method is not None} keywords={keywords}")
