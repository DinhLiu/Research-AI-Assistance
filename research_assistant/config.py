from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(REPO_ROOT / ".env", override=True)

DEFAULT_ARTIFACTS_DIR = REPO_ROOT / "data" / "specter2_artifacts"

BASE_MODEL = "allenai/specter2_base"
QUERY_ADAPTER = "allenai/specter2_adhoc_query"
MAX_LENGTH = 512


@dataclass
class RetrievalConfig:
    artifacts_dir: Path = DEFAULT_ARTIFACTS_DIR
    # Per expanded query, how many FAISS neighbors to pull.
    broad_k: int = 150
    # Union/dedup cap before rerank.
    candidate_pool_size: int = 200
    # Final papers handed to extraction.
    top_k: int = 25
    n_query_variants: int = 4
    # Cosine of the original topic vs stored paper embeddings. Off → keep FAISS order.
    use_rerank: bool = True
    use_hybrid: bool = True
    use_citations: bool = False
    rrf_k: int = 60
    year_from: int | None = None
    year_to: int | None = None
    categories: tuple[str, ...] = field(default_factory=tuple)
    device: str = "auto"
    use_fp16: bool = True
    llm_timeout_s: float = 30.0
    arxiv_keyword_n: int = 50
    rebuild_index_if_missing: bool = True
