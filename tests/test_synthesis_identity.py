from __future__ import annotations

import pytest

from research_assistant.config import SynthesisConfig
from research_assistant.synthesis.pipeline import synthesize
from research_assistant.synthesis.types import SynthesisInputError

from tests.synthesis_utils import paper, snapshot


def _cfg(tmp_path, **kwargs) -> SynthesisConfig:
    base = dict(cache_dir=tmp_path, use_cache=False, summarize=False)
    base.update(kwargs)
    return SynthesisConfig(**base)


def test_identical_duplicates_keep_one(tmp_path):
    dup = paper(arxiv_id="1111.00001", method="EL2N pruning.", keywords=["el2n"])
    result = synthesize(snapshot([dup, dup.model_copy()]), _cfg(tmp_path))
    assert result.diagnostics.n_clustered == 1
    assert [item.disposition for item in result.input_inventory].count("duplicate") == 1


def test_conflicting_same_version_raises(tmp_path):
    a = paper(arxiv_id="1111.00001", method="EL2N pruning.", keywords=["el2n"])
    b = paper(arxiv_id="1111.00001", method="Random sampling only.", keywords=["random"])
    with pytest.raises(SynthesisInputError, match="conflicting records"):
        synthesize(snapshot([a, b]), _cfg(tmp_path))


def test_highest_version_wins(tmp_path):
    v1 = paper(arxiv_id="1111.00001", version="v1", method="Old EL2N method.", keywords=["el2n"])
    v2 = paper(arxiv_id="1111.00001", version="v2", method="Updated EL2N method.", keywords=["el2n"])
    result = synthesize(snapshot([v1, v2]), _cfg(tmp_path))
    assert result.paper_manifest[0].paper_key == "1111.00001v2"
    assert result.input_inventory[0].disposition == "superseded_version"
    assert result.input_inventory[1].disposition == "accepted"


def test_strict_ok_excludes_degraded(tmp_path):
    result = synthesize(
        snapshot(
            [
                paper(arxiv_id="1111.00001", status="degraded", method="EL2N pruning.", keywords=["el2n"]),
                paper(arxiv_id="1111.00002", status="ok", method="EL2N pruning two.", keywords=["el2n"]),
            ]
        ),
        _cfg(tmp_path, strict_ok=True),
    )
    assert result.input_inventory[0].disposition == "strict_ok_excluded"
    assert result.diagnostics.n_clustered == 1
