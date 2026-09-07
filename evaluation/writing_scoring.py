"""Offline execution/coverage metrics; deliberately not an LLM quality judge."""
import argparse
import json
from pathlib import Path

from research_assistant.writing.types import WritingResult


def score_writing(result: WritingResult) -> dict:
    count = len(result.claims)
    total = result.coverage.get("total_papers", 0)
    supported = result.coverage.get("supported_papers", 0)
    return {
        "generation_status": result.generation_status,
        "http_attempts": result.execution.http_attempts,
        "transport_retries": result.execution.transport_retries,
        "repair_calls": result.execution.repair_calls,
        "input_tokens": result.execution.input_tokens,
        "output_tokens": result.execution.output_tokens,
        "reserved_tokens": result.execution.reserved_tokens,
        "api_seconds": result.execution.api_seconds,
        "queue_wait_seconds": result.execution.queue_wait_seconds,
        "seconds_this_invocation": result.execution.seconds,
        "paper_coverage": {"numerator": supported, "denominator": total,
                           "rate": supported / total if total else None},
        "prose_coverage": {"numerator": sum(c.generation == "llm" for c in result.claims), "denominator": count,
                           "rate": sum(c.generation == "llm" for c in result.claims) / count if count else None},
        "unselected_evidence_units": len(result.plan.unselected_evidence_ids),
        "budget_omitted_claims": len(result.plan.omitted),
        "semantic_support_rate": None,
        "semantic_support_note": "Requires independent human claim/evidence annotations; reference integrity is not entailment.",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("writing_json", type=Path)
    args = parser.parse_args(argv)
    result = WritingResult.model_validate_json(args.writing_json.read_text(encoding="utf-8"))
    print(json.dumps(score_writing(result), indent=2))


if __name__ == "__main__":
    main()
