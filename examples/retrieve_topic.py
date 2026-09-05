"""Example: retrieve papers for a topic against local SPECTER2 artifacts."""

from research_assistant import retrieve
from research_assistant.config import RetrievalConfig


if __name__ == "__main__":
    topic = (
        "dataset pruning and data subset selection for deep neural networks"
    )
    result = retrieve(
        topic,
        RetrievalConfig(top_k=10, candidate_pool_size=100, use_hybrid=True),
    )
    print("Queries:", result.queries)
    print("Metrics:", result.metrics)
    for paper in result.papers:
        print(f"{paper.rank:2d}. {paper.rerank_score:.3f}  {paper.arxiv_id}  {paper.title}")
