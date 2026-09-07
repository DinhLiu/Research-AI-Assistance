"""CLI for stage-3 synthesis. Logs to stderr; JSON is written only to --json-out."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from research_assistant.config import DEFAULT_SYNTHESIS_CACHE_DIR, SynthesisConfig
from research_assistant.extraction.types import ExtractionResult
from research_assistant.synthesis.pipeline import synthesize
from research_assistant.synthesis.types import SynthesisInputError, SynthesisResult

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Cluster extracted papers by method and optionally narrate with citation gates.",
    )
    parser.add_argument(
        "extraction_json",
        type=Path,
        help="ExtractionResult JSON from python -m research_assistant.extraction --json-out",
    )
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_SYNTHESIS_CACHE_DIR)
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--strict-ok", action="store_true", help="Cluster only status=ok papers")
    parser.add_argument("--strict-evidence", action="store_true", help="Template wording; no LLM paraphrase")
    parser.add_argument("--distance-threshold", type=float, default=None)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--require-summary", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    try:
        payload = json.loads(args.extraction_json.read_text(encoding="utf-8"))
        extracted = ExtractionResult.model_validate(payload)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        logger.error("Invalid extraction JSON: %s", exc)
        return 2

    config = SynthesisConfig(
        cache_dir=args.cache_dir,
        summarize=bool(args.summarize or args.strict_evidence),
        strict_ok=args.strict_ok,
        strict_evidence=args.strict_evidence,
        use_cache=not args.no_cache,
    )
    if args.distance_threshold is not None:
        config.distance_threshold = args.distance_threshold

    try:
        result = synthesize(extracted, config)
    except SynthesisInputError as exc:
        logger.error("%s", exc)
        return 2

    _print_table(result)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(result.model_dump_json(indent=2), encoding="utf-8")
        print(f"Wrote {args.json_out}", file=sys.stderr)

    if args.require_summary and result.execution.summary_status not in {"complete", "partial"}:
        return 1
    return 0


def _print_table(result: SynthesisResult) -> None:
    print(f"corpus_digest: {result.corpus_digest}")
    print(f"summary_status: {result.execution.summary_status}")
    print(
        "diagnostics:",
        json.dumps(result.diagnostics.model_dump(), ensure_ascii=False),
    )
    print()
    print(f"{'cluster':<18} {'n':>3}  label")
    for cluster in result.assignments:
        print(f"{cluster.cluster_id:<18} {cluster.size:>3}  {cluster.label}")
        for key in cluster.paper_keys:
            print(f"  - {key}")
    if result.unassigned:
        print("unassigned:")
        for item in result.unassigned:
            print(f"  - {item.paper_key} ({item.reason})")


if __name__ == "__main__":
    sys.exit(main())
