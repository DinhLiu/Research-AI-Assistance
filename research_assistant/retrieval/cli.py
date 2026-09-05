from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from research_assistant.config import DEFAULT_ARTIFACTS_DIR, RetrievalConfig
from research_assistant.retrieval.pipeline import RetrievalPipeline


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Retrieve papers for a research topic (SPECTER2 retrieve-then-rerank).",
    )
    p.add_argument("topic", nargs="?", help="Research topic / query")
    p.add_argument("--topic", dest="topic_flag", help="Alternative to the positional topic")
    p.add_argument(
        "--artifacts",
        type=Path,
        default=DEFAULT_ARTIFACTS_DIR,
        help="Path to specter2_artifacts (manifest.json + embeddings/ + metadata/ + index/)",
    )
    p.add_argument("--top-k", type=int, default=25)
    p.add_argument("--broad-k", type=int, default=150, help="FAISS neighbors per expanded query")
    p.add_argument("--pool", type=int, default=200, help="Candidate pool size before rerank")
    p.add_argument("--variants", type=int, default=4, help="LLM/template query variants (plus original); 0 = original only")
    p.add_argument("--no-rerank", action="store_true", help="Keep FAISS order (skip original-query cosine rerank)")
    p.add_argument("--no-hybrid", action="store_true", help="Disable arXiv keyword RRF")
    p.add_argument(
        "--citations",
        "--citation",
        action="store_true",
        help="Blend Semantic Scholar citation ranks (optional; often 429 without an API key)",
    )
    p.add_argument("--year-from", type=int, default=None)
    p.add_argument("--year-to", type=int, default=None)
    p.add_argument("--category", action="append", default=[], help="Filter matched_categories (repeatable)")
    p.add_argument("--device", default="auto")
    p.add_argument("--json-out", type=Path, default=None)
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    topic = args.topic_flag or args.topic
    if not topic:
        build_parser().error("topic is required")

    config = RetrievalConfig(
        artifacts_dir=args.artifacts,
        broad_k=args.broad_k,
        candidate_pool_size=args.pool,
        top_k=args.top_k,
        n_query_variants=args.variants,
        use_rerank=not args.no_rerank,
        use_hybrid=not args.no_hybrid,
        use_citations=args.citations,
        year_from=args.year_from,
        year_to=args.year_to,
        categories=tuple(args.category),
        device=args.device,
    )
    result = RetrievalPipeline(config).search(topic)
    _print_table(result)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(result.model_dump_json(indent=2), encoding="utf-8")
        print(f"\nWrote {args.json_out}")
    return 0


def _print_table(result) -> None:
    print(f"Topic: {result.topic}")
    print(f"Expander: {result.expander}")
    print("Queries:")
    for q in result.queries:
        print(f"  - {q}")
    print("Metrics:", json.dumps(result.metrics, ensure_ascii=False))
    print()
    print(f"{'rank':>4}  {'rerank':>7}  {'broad':>7}  {'cites':>6}  {'year':>4}  {'arxiv_id':<12}  title")
    for paper in result.papers:
        year = "" if paper.year is None else str(paper.year)
        cites = "" if paper.citation_count is None else str(paper.citation_count)
        title = paper.title.replace("\n", " ")
        if len(title) > 88:
            title = title[:85] + "..."
        print(
            f"{paper.rank:4d}  {paper.rerank_score:7.3f}  {paper.broad_score:7.3f}  "
            f"{cites:>6}  {year:>4}  {paper.arxiv_id:<12}  {title}"
        )


if __name__ == "__main__":
    sys.exit(main())
