from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(REPO_ROOT / ".env", override=True)

DEFAULT_ARTIFACTS_DIR = REPO_ROOT / "data" / "specter2_artifacts"
DEFAULT_EXTRACTION_CACHE_DIR = REPO_ROOT / "data" / "extraction_cache"

# Bump these when the corresponding algorithm/prompt/schema changes.
EXTRACTION_SCHEMA_VERSION = "2"
EXTRACTION_PROMPT_VERSION = "2"
EXTRACTION_SELECTOR_VERSION = "1"
EXTRACTION_NORMALIZER_VERSION = "1"
EXTRACTION_PARSED_VERSION = "1"
EXTRACTION_CANDIDATE_VERSION = "1"

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


@dataclass
class ExtractionConfig:
    cache_dir: Path = DEFAULT_EXTRACTION_CACHE_DIR
    arxiv_delay_s: float = 3.0
    llm_concurrency: int = 1
    max_retries: int = 0
    llm_timeout_s: float = 90.0
    allow_abstract_fallback: bool = True
    llm_batch_size: int = 5
    max_llm_calls_per_run: int = 6
    llm_min_interval_s: float = 5.0
    pdf_on_parse_fail_only: bool = True
    skip_llm_if_confident: bool = True
    candidate_version: str = EXTRACTION_CANDIDATE_VERSION
    include_related_work: bool = False
    include_appendix: bool = False
    include_bibliography: bool = False
    abstract_only: bool = False
    max_input_chars: int = 24000
    max_input_tokens: int | None = None
    max_tar_files: int = 400
    max_tar_bytes: int = 80_000_000
    max_tar_nesting: int = 8
    fetch_timeout_s: float = 60.0
    schema_version: str = EXTRACTION_SCHEMA_VERSION
    prompt_version: str = EXTRACTION_PROMPT_VERSION
    selector_version: str = EXTRACTION_SELECTOR_VERSION
    normalizer_version: str = EXTRACTION_NORMALIZER_VERSION
    parsed_version: str = EXTRACTION_PARSED_VERSION
    # Extraction prefers Gemini; Groq is last-resort for this stage.
    prefer_provider: str = "gemini"
