"""python examples/write_review.py results/pruning.synthesis.json (offline)."""
import sys
from pathlib import Path
from research_assistant.config import WritingConfig
from research_assistant.synthesis.types import SynthesisResult
from research_assistant.writing import write_review
from research_assistant.writing.render import render_markdown

snapshot = SynthesisResult.model_validate_json(Path(sys.argv[1]).read_text())
print(render_markdown(write_review(snapshot, WritingConfig())))
