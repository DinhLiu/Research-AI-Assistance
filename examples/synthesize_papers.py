"""Example: cluster extracted papers and optionally narrate."""

from __future__ import annotations

from pathlib import Path

from research_assistant.config import SynthesisConfig
from research_assistant.extraction.types import ExtractionResult
from research_assistant.synthesis import synthesize


if __name__ == "__main__":
    extraction_path = Path("results/pruning.extracted.json")
    if extraction_path.is_file():
        extracted = ExtractionResult.model_validate_json(extraction_path.read_text(encoding="utf-8"))
    else:
        from research_assistant.extraction.types import Claim, Evidence, ExtractedPaper, Mention

        def _ev(text: str) -> Evidence:
            return Evidence(quote=text, section="Method", source_kind="tex")

        extracted = ExtractionResult(
            topic="dataset pruning",
            records=[
                ExtractedPaper(
                    arxiv_id="2205.09329",
                    version="v1",
                    title="Dataset Pruning",
                    status="ok",
                    method=Claim(
                        text="Prune training examples by estimating generalization influence.",
                        evidence=[_ev("generalization influence")],
                    ),
                    method_keywords=[
                        Mention(value="dataset pruning", evidence=_ev("dataset pruning")),
                        Mention(value="EL2N", evidence=_ev("EL2N")),
                    ],
                )
            ],
        )
    result = synthesize(extracted, SynthesisConfig(summarize=False, use_cache=False))
    print(result.diagnostics)
    for cluster in result.assignments:
        print(cluster.cluster_id, cluster.label, cluster.paper_keys)
