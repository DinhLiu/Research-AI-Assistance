from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(REPO_ROOT / ".env", override=True)

DEFAULT_ARTIFACTS_DIR = REPO_ROOT / "data" / "specter2_artifacts"
DEFAULT_EXTRACTION_CACHE_DIR = REPO_ROOT / "data" / "extraction_cache"
DEFAULT_SYNTHESIS_CACHE_DIR = REPO_ROOT / "data" / "synthesis_cache"

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


SYNTHESIS_SCHEMA_VERSION = "1"
SYNTHESIS_NORMALIZER_VERSION = "1"
SYNTHESIS_CLUSTERING_VERSION = "1"
SYNTHESIS_PROMPT_VERSION = "2"
SYNTHESIS_VALIDATOR_VERSION = "2"
SYNTHESIS_CARD_VERSION = "1"


@dataclass
class SynthesisConfig:
    cache_dir: Path = DEFAULT_SYNTHESIS_CACHE_DIR
    summarize: bool = False
    strict_ok: bool = False
    strict_evidence: bool = False
    # Provisional cosine-distance cut; calibrate before treating as validated.
    distance_threshold: float = 0.55
    keyword_weight: float = 0.5
    max_features: int = 2000
    use_cache: bool = True
    llm_timeout_s: float = 90.0
    max_logical_calls: int = 2
    max_prompt_chars: int = 24000
    max_method_chars: int = 800
    max_problem_chars: int = 400
    max_claim_chars: int = 300
    max_claims_per_field: int = 2
    max_quote_chars: int = 280
    prefer_provider: str = "gemini"
    schema_version: str = SYNTHESIS_SCHEMA_VERSION
    normalizer_version: str = SYNTHESIS_NORMALIZER_VERSION
    clustering_version: str = SYNTHESIS_CLUSTERING_VERSION
    prompt_version: str = SYNTHESIS_PROMPT_VERSION
    validator_version: str = SYNTHESIS_VALIDATOR_VERSION
    card_version: str = SYNTHESIS_CARD_VERSION


@dataclass
class WritingConfig:
    cache_dir: Path = REPO_ROOT / "data" / "writing_cache"
    use_cache: bool = True
    use_llm: bool = False
    dry_run: bool = False
    language: str = "en"
    target_words: int = 1500
    max_generation_batches: int = 3
    max_repair_calls: int = 1
    max_http_attempts_per_run: int = 5
    max_run_tokens: int = 100000
    max_input_tokens: int = 20000
    max_output_tokens: int = 6000
    context_tokens: int = 32768
    request_timeout_s: float = 90.0
    run_deadline_s: float = 600.0
    prefer_provider: str = "gemini"
    run_id: str | None = None
    schema_version: str = "1"
    prompt_version: str = "1"
    validator_version: str = "1"
    render_version: str = "1"

    def __post_init__(self):
        import math
        counts = (self.target_words, self.max_generation_batches, self.max_repair_calls,
                  self.max_http_attempts_per_run, self.max_run_tokens, self.max_input_tokens,
                  self.max_output_tokens, self.context_tokens)
        if any(type(n) is not int for n in counts):
            raise ValueError("Writing budgets must be integers")
        if self.schema_version != "1":
            raise ValueError("Unsupported writing schema")
        if self.language not in {"en", "vi"}:
            raise ValueError("language must be en or vi")
        if not 1 <= self.max_generation_batches <= 3 or not 0 <= self.max_repair_calls <= 1:
            raise ValueError("At most three batches and one repair are supported")
        if not 0 <= self.max_http_attempts_per_run <= 5:
            raise ValueError("HTTP attempt cap must be between zero and five")
        if min(self.target_words, self.max_input_tokens, self.max_output_tokens, self.context_tokens) <= 0 or self.max_run_tokens < 0:
            raise ValueError("Invalid writing token/length budgets")
        if any(not math.isfinite(n) or n <= 0 for n in (self.request_timeout_s, self.run_deadline_s)):
            raise ValueError("Invalid writing deadlines")
