from __future__ import annotations

import ast
import json
from pathlib import Path

from research_assistant.synthesis.cli import main

from tests.synthesis_utils import paper, snapshot


def test_synthesis_sources_do_not_import_retrieval():
    root = Path("research_assistant/synthesis")
    for path in sorted(root.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert "retrieval" not in alias.name, path
            if isinstance(node, ast.ImportFrom) and node.module:
                assert "research_assistant.retrieval" not in node.module, path


def test_offline_cli_json_roundtrip(tmp_path):
    extracted = snapshot(
        [
            paper(arxiv_id="1111.00001", method="EL2N pruning of examples.", keywords=["el2n"]),
            paper(arxiv_id="1111.00002", method="Random repeated sampling.", keywords=["random sampling"]),
        ]
    )
    src = tmp_path / "extracted.json"
    out = tmp_path / "synthesis.json"
    src.write_text(extracted.model_dump_json(), encoding="utf-8")
    code = main([str(src), "--json-out", str(out), "--no-cache", "--strict-evidence"])
    assert code == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["corpus_digest"]
    assert payload["assignments"]
    lookup = {unit["unit_id"]: unit for unit in payload["evidence_registry"]["units"]}
    for summary in payload["summaries"]:
        for claim in summary["claims"]:
            assert claim["validation_status"] == "structurally_validated"
            for ref in claim["support_refs"]:
                assert ref in lookup
                assert lookup[ref]["paper_key"] in claim["subject_paper_keys"]


def test_invalid_json_exits_nonzero(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("{", encoding="utf-8")
    assert main([str(path)]) == 2


def test_require_summary_without_key(tmp_path, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    src = tmp_path / "extracted.json"
    src.write_text(snapshot([paper()]).model_dump_json(), encoding="utf-8")
    assert main([str(src), "--summarize", "--require-summary", "--no-cache"]) == 1
