"""Offline/dry-run/LLM writing CLI. JSON/Markdown artifacts are deterministic."""
import argparse
import json
import sys
from pathlib import Path

from research_assistant.config import WritingConfig
from research_assistant.writing.fingerprint import atomic_write
from research_assistant.writing.pipeline import write_review
from research_assistant.writing.render import render_markdown


def main(argv=None):
    p = argparse.ArgumentParser(description="Write a cited review from a SynthesisResult; offline by default.")
    p.add_argument("synthesis_json", type=Path)
    p.add_argument("--write", action="store_true", help="Enable bounded LLM writing; requires LLM_RPM/TPM/RPD")
    p.add_argument("--dry-run", action="store_true", help="Print packing/budgets; no network")
    p.add_argument("--language", choices=["en", "vi"], default="en")
    p.add_argument("--target-words", type=int, default=1500)
    p.add_argument("--md-out", type=Path)
    p.add_argument("--json-out", type=Path)
    p.add_argument("--cache-dir", type=Path, default=WritingConfig().cache_dir)
    p.add_argument("--no-cache", action="store_true")
    p.add_argument("--require-complete", action="store_true")
    p.add_argument("--run-id", help="Resume the same input/run without renewing attempts or deadline")
    p.add_argument("--provider", choices=["gemini", "openai", "groq"], default="gemini")
    p.add_argument("--max-http-attempts", type=int, default=5)
    p.add_argument("--max-input-tokens", type=int, default=20000)
    p.add_argument("--max-output-tokens", type=int, default=6000)
    p.add_argument("--context-tokens", type=int, default=32768)
    p.add_argument("--max-run-tokens", type=int, default=100000)
    args = p.parse_args(argv)
    try:
        cfg = WritingConfig(cache_dir=args.cache_dir, use_cache=not args.no_cache, use_llm=args.write,
                            dry_run=args.dry_run, language=args.language, target_words=args.target_words,
                            run_id=args.run_id, prefer_provider=args.provider,
                            max_http_attempts_per_run=args.max_http_attempts,
                            max_input_tokens=args.max_input_tokens, max_output_tokens=args.max_output_tokens,
                            context_tokens=args.context_tokens, max_run_tokens=args.max_run_tokens)
        result = write_review(json.loads(args.synthesis_json.read_text(encoding="utf-8")), cfg)
        if args.json_out:
            atomic_write(args.json_out, result.model_dump_json(indent=2))
        if args.md_out:
            atomic_write(args.md_out, render_markdown(result))
        if args.dry_run:
            print(json.dumps({"plan": result.plan.model_dump(), "config": {
                "max_http_attempts": cfg.max_http_attempts_per_run, "max_run_tokens": cfg.max_run_tokens,
                "max_input_tokens": cfg.max_input_tokens, "max_output_tokens": cfg.max_output_tokens,
                "context_tokens": cfg.context_tokens}, "quota": result.execution.quota,
                "stop_reason": result.execution.stop_reason}, ensure_ascii=False, indent=2))
        elif not args.md_out:
            print(render_markdown(result))
        print(f"status={result.generation_status} attempts={result.execution.http_attempts} "
              f"run_id={result.execution.run_id} reason={result.execution.stop_reason}", file=sys.stderr)
        return int(args.require_complete and result.generation_status not in {"complete", "empty"})
    except (OSError, ValueError) as exc:
        print(f"Invalid writing input/config: {exc}", file=sys.stderr)
        return 2
