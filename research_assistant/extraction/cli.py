"""CLI for stage-2 extraction."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from research_assistant.arxiv_ids import format_version, split_arxiv_id
from research_assistant.config import DEFAULT_EXTRACTION_CACHE_DIR, ExtractionConfig
from research_assistant.extraction.pipeline import extract_papers
from research_assistant.retrieval.types import PaperHit, RetrievalResult


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Extract grounded claims from retrieved papers (TeX-first, PDF fallback).",
    )
    p.add_argument(
        "retrieval_json",
        nargs="?",
        type=Path,
        help="RetrievalResult JSON from python -m research_assistant.retrieval --json-out",
    )
    p.add_argument("--arxiv-id", action="append", default=[], help="Extract a single id (repeatable)")
    p.add_argument("--cache-dir", type=Path, default=DEFAULT_EXTRACTION_CACHE_DIR)
    p.add_argument("--json-out", type=Path, default=None)
    p.add_argument("--max-retries", type=int, default=0)
    p.add_argument("--concurrency", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=5)
    p.add_argument("--max-llm-calls", type=int, default=6)
    p.add_argument("--llm-interval", type=float, default=5.0)
    p.add_argument("--include-related-work", action="store_true")
    p.add_argument("--abstract-only", action="store_true")
    p.add_argument("--no-abstract-fallback", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    papers, topic = _load_papers(args)
    if not papers:
        build_parser().error("provide a retrieval JSON path or --arxiv-id")

    config = ExtractionConfig(
        cache_dir=args.cache_dir,
        max_retries=args.max_retries,
        llm_concurrency=args.concurrency,
        llm_batch_size=args.batch_size,
        max_llm_calls_per_run=args.max_llm_calls,
        llm_min_interval_s=args.llm_interval,
        include_related_work=args.include_related_work,
        abstract_only=args.abstract_only,
        allow_abstract_fallback=not args.no_abstract_fallback,
    )
    result = extract_papers(papers, config, topic=topic)
    _print_table(result)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(result.model_dump_json(indent=2), encoding="utf-8")
        print(f"\nWrote {args.json_out}")
    return 0


def _load_papers(args) -> tuple[list[PaperHit], str | None]:
    papers: list[PaperHit] = []
    topic = None
    if args.retrieval_json:
        payload = json.loads(Path(args.retrieval_json).read_text(encoding="utf-8"))
        result = RetrievalResult.model_validate(payload)
        topic = result.topic
        papers.extend(result.papers)
    for raw in args.arxiv_id:
        core, version = split_arxiv_id(raw)
        papers.append(
            PaperHit(
                rank=len(papers) + 1,
                row_id=-1,
                arxiv_id=core,
                title="",
                abstract="",
                latest_version=format_version(version),
            )
        )
    return papers, topic


def _print_table(result) -> None:
    print("Fingerprint:", result.fingerprint)
    print("Metrics:", json.dumps(result.metrics, ensure_ascii=False))
    print()
    print(f"{'status':<10} {'src':<8} {'conf':>5}  {'arxiv_id':<12}  title")
    for paper in result.records:
        title = (paper.title or "").replace("\n", " ")
        if len(title) > 72:
            title = title[:69] + "..."
        print(
            f"{paper.status:<10} {paper.source_kind:<8} {paper.confidence:5.2f}  "
            f"{paper.arxiv_id:<12}  {title}"
        )
        if paper.status == "skipped" and paper.error:
            print(f"           error: {paper.error[:160]}")


if __name__ == "__main__":
    sys.exit(main())
