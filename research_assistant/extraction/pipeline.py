"""Candidate-first extraction: local IE, then micro-batch LLM over evidence IDs."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from research_assistant.arxiv_ids import format_version, normalize_arxiv_id
from research_assistant.config import ExtractionConfig
from research_assistant.extraction.candidates import build_candidate_pack
from research_assistant.extraction.fetch import (
    FetchError,
    FetchedSource,
    candidates_path,
    extraction_dir,
    fetch_eprint,
    fetch_pdf,
    parsed_path,
)
from research_assistant.extraction.fingerprint import fingerprint_model, pipeline_fingerprint
from research_assistant.extraction.parse_pdf import PdfParseError, parse_pdf_source
from research_assistant.extraction.parse_tex import ParseError, parse_tex_source
from research_assistant.extraction.prompt import SYSTEM_PROMPT, render_batch, retry_user_message
from research_assistant.extraction.sections import build_abstract_prompt, build_prompt_document
from research_assistant.extraction.types import (
    AttemptOutcome,
    CandidatePack,
    ExtractionAttempt,
    ExtractionBundle,
    ExtractedPaper,
    ExtractionResult,
    PaperDocument,
    PromptDocument,
    SourceKind,
)
from research_assistant.extraction.validate import ground_llm, paper_from_local_candidates, parse_llm_payload
from research_assistant.llm.client import (
    LlmBudgetExceeded,
    RateLimitError,
    complete,
    extraction_provider,
)
from research_assistant.retrieval.types import PaperHit, RetrievalResult

logger = logging.getLogger(__name__)

CompleteFn = Callable[..., str]
SOURCES: tuple[SourceKind, ...] = ("tex", "pdf", "abstract")


@dataclass
class _CallCounter:
    n: int = 0
    last_ts: float | None = None
    max_calls: int = 0
    min_interval_s: float = 0.0

    def remaining(self) -> int:
        if self.max_calls <= 0:
            return 10**9
        return max(0, self.max_calls - self.n)

    def exhausted(self) -> bool:
        return self.max_calls > 0 and self.n >= self.max_calls

    def wrap(self, fn: CompleteFn) -> CompleteFn:
        def _inner(*, system: str, user: str, **kwargs: Any) -> str:
            if self.exhausted():
                raise LlmBudgetExceeded(f"max_llm_calls_per_run={self.max_calls}")
            if self.min_interval_s > 0 and self.last_ts is not None:
                wait = self.min_interval_s - (time.monotonic() - self.last_ts)
                if wait > 0:
                    time.sleep(wait)
            self.n += 1
            try:
                return fn(system=system, user=user, **kwargs)
            finally:
                self.last_ts = time.monotonic()

        return _inner


@dataclass
class _Prepared:
    hit: PaperHit
    attempts: list[ExtractionAttempt] = field(default_factory=list)
    prompt: PromptDocument | None = None
    pack: CandidatePack | None = None
    extracted: ExtractedPaper | None = None
    cached: bool = False


def extract_papers(
    papers: Sequence[PaperHit] | RetrievalResult,
    config: ExtractionConfig | None = None,
    *,
    topic: str | None = None,
    complete_fn: CompleteFn | None = None,
) -> ExtractionResult:
    cfg = config or ExtractionConfig()
    t0 = time.perf_counter()
    if isinstance(papers, RetrievalResult):
        topic = topic or papers.topic
        hits = papers.papers
    else:
        hits = list(papers)

    provider = extraction_provider(cfg.prefer_provider)
    model = fingerprint_model(cfg, provider)
    fingerprint = pipeline_fingerprint(cfg, model)
    counter = _CallCounter(
        max_calls=max(0, int(cfg.max_llm_calls_per_run)),
        min_interval_s=max(0.0, float(cfg.llm_min_interval_s)),
    )
    completer = counter.wrap(complete_fn or _bound_complete(cfg, provider))

    prepared = [_prepare(hit, cfg, fingerprint, completer) for hit in hits]
    pending = [item for item in prepared if item.extracted is None and item.pack is not None]
    batch_size = max(1, int(cfg.llm_batch_size))
    for start in range(0, len(pending), batch_size):
        if counter.exhausted():
            logger.warning("Stopping LLM batches: max_llm_calls_per_run=%s", counter.max_calls)
            break
        _extract_batch(
            pending[start : start + batch_size],
            cfg,
            fingerprint,
            completer,
            counter,
        )

    records = [_finalize(item, fingerprint) for item in prepared]
    cache_hits = sum(item.cached for item in prepared)
    skipped_llm = sum(
        1
        for item in prepared
        if item.extracted is not None
        and not item.cached
        and item.extracted.attempts
        and item.extracted.attempts[-1].outcome == "ok"
        and not any(a.outcome == "ok" and a.attempt_no >= 0 for a in item.extracted.attempts[:-1])
        and item.pack is not None
        and item.pack.local_confidence >= 0.92
    )
    metrics = _metrics(records, cache_hits, prepared)
    metrics["seconds"] = round(time.perf_counter() - t0, 3)
    metrics["llm_calls"] = counter.n
    metrics["llm_skipped_confident"] = skipped_llm
    return ExtractionResult(
        topic=topic,
        records=records,
        fingerprint=fingerprint,
        metrics=metrics,
    )


def extract_one(
    hit: PaperHit,
    config: ExtractionConfig,
    fingerprint: str,
    complete_fn: CompleteFn,
) -> tuple[ExtractedPaper, bool]:
    counter = _CallCounter(
        max_calls=max(0, int(config.max_llm_calls_per_run)),
        min_interval_s=max(0.0, float(config.llm_min_interval_s)),
    )
    completer = counter.wrap(complete_fn)
    prepared = _prepare(hit, config, fingerprint, completer)
    if prepared.extracted is None and prepared.pack is not None:
        _extract_batch([prepared], config, fingerprint, completer, counter)
    return _finalize(prepared, fingerprint), prepared.cached


def _prepare(
    hit: PaperHit,
    config: ExtractionConfig,
    fingerprint: str,
    complete_fn: CompleteFn,
) -> _Prepared:
    arxiv_id = normalize_arxiv_id(hit.arxiv_id)
    version = format_version(hit.latest_version)
    cached = _load_extracted(config, arxiv_id, version, fingerprint)
    if cached is not None:
        return _Prepared(hit=hit, extracted=cached, cached=True)

    prepared = _Prepared(hit=hit)
    paper_doc: PaperDocument | None = None
    parsed_ok: set[str] = set()
    for source_kind in SOURCES:
        if source_kind == "abstract" and not config.allow_abstract_fallback:
            continue
        if (
            source_kind == "pdf"
            and config.pdf_on_parse_fail_only
            and "tex" in parsed_ok
        ):
            continue
        try:
            prompt, parsed = _prompt_for_source(hit, source_kind, config, paper_doc)
        except Exception as exc:
            logger.warning("%s %s %s: %s", arxiv_id, version, source_kind, exc)
            prepared.attempts.append(
                ExtractionAttempt(
                    source_kind=source_kind,
                    attempt_no=0,
                    outcome="parse_fail",
                    validation_errors=[str(exc)],
                )
            )
            continue
        parsed_ok.add(source_kind)
        if parsed is not None:
            paper_doc = parsed
        pack = _load_or_build_pack(parsed or _paper_from_prompt(prompt), prompt, config)
        prepared.prompt = prompt
        prepared.pack = pack
        if config.skip_llm_if_confident:
            local = paper_from_local_candidates(pack)
            if local is not None:
                local.attempts = prepared.attempts + [
                    ExtractionAttempt(source_kind=source_kind, attempt_no=0, outcome="ok")
                ]
                local.fingerprint = fingerprint
                prepared.extracted = local
                _save_extracted(config, fingerprint, prompt, local)
                return prepared
        break
    return prepared


def _extract_batch(
    items: list[_Prepared],
    config: ExtractionConfig,
    fingerprint: str,
    complete_fn: CompleteFn,
    counter: _CallCounter,
    errors: list[str] | None = None,
    *,
    retry_evidence: bool = True,
) -> None:
    items = [item for item in items if item.pack is not None and item.extracted is None]
    if not items:
        return
    if counter.exhausted():
        return
    packs = [item.pack for item in items if item.pack is not None]
    user = render_batch(packs) if not errors else retry_user_message(packs, errors)
    try:
        raw = complete_fn(system=SYSTEM_PROMPT, user=user)
    except RateLimitError as exc:
        _mark_fail(items, "llm_fail", [str(exc)])
        return
    except LlmBudgetExceeded:
        return
    except Exception as exc:
        _mark_fail(items, "llm_fail", [str(exc)])
        return
    try:
        parsed_llm = parse_llm_payload(raw)
    except Exception as exc:
        if len(items) > 1 and counter.remaining() >= 2:
            mid = max(1, len(items) // 2)
            _extract_batch(items[:mid], config, fingerprint, complete_fn, counter)
            _extract_batch(items[mid:], config, fingerprint, complete_fn, counter)
            return
        _mark_fail(items, "struct_fail", [str(exc)])
        return

    by_id: dict[str, Any] = {}
    for llm in parsed_llm:
        if llm.arxiv_id:
            by_id[normalize_arxiv_id(llm.arxiv_id)] = llm
    if len(parsed_llm) == len(items) and not by_id:
        pairs = list(zip(items, parsed_llm))
    else:
        pairs = []
        leftover = list(parsed_llm)
        for item in items:
            aid = normalize_arxiv_id(item.hit.arxiv_id)
            llm = by_id.get(aid)
            if llm is None and leftover:
                llm = leftover.pop(0)
            if llm is None:
                continue
            pairs.append((item, llm))

    for item, llm in pairs:
        pack = item.pack
        assert pack is not None
        grounded, report = ground_llm(llm, pack)
        if grounded.method is None:
            item.attempts.append(
                ExtractionAttempt(
                    source_kind=pack.source_kind,
                    attempt_no=0,
                    outcome="evidence_fail",
                    validation_errors=report.errors or ["method missing"],
                )
            )
            continue
        grounded.attempts = item.attempts + [
            ExtractionAttempt(source_kind=pack.source_kind, attempt_no=0, outcome="ok")
        ]
        grounded.fingerprint = fingerprint
        item.extracted = grounded
        if item.prompt is not None:
            _save_extracted(config, fingerprint, item.prompt, grounded)

    still = [item for item in items if item.extracted is None]
    if not still or not retry_evidence or int(config.max_retries) <= 0:
        return
    for item in still:
        last_errors = item.attempts[-1].validation_errors if item.attempts else ["evidence_fail"]
        for _ in range(int(config.max_retries)):
            if item.extracted is not None or counter.exhausted():
                break
            _extract_batch(
                [item],
                config,
                fingerprint,
                complete_fn,
                counter,
                errors=last_errors,
                retry_evidence=False,
            )


def _mark_fail(items: list[_Prepared], outcome: AttemptOutcome, errors: list[str]) -> None:
    for item in items:
        if item.extracted is not None:
            continue
        source_kind = item.pack.source_kind if item.pack else "tex"
        item.attempts.append(
            ExtractionAttempt(
                source_kind=source_kind,
                attempt_no=0,
                outcome=outcome,
                validation_errors=errors,
            )
        )


def _finalize(item: _Prepared, fingerprint: str) -> ExtractedPaper:
    if item.extracted is not None:
        return item.extracted
    hit = item.hit
    return ExtractedPaper(
        arxiv_id=normalize_arxiv_id(hit.arxiv_id),
        version=format_version(hit.latest_version),
        title=hit.title,
        source_kind="abstract",
        status="skipped",
        attempts=item.attempts,
        error="extraction failed",
        fingerprint=fingerprint,
    )


def _load_or_build_pack(
    paper: PaperDocument, prompt: PromptDocument, config: ExtractionConfig
) -> CandidatePack:
    path = candidates_path(config, paper.arxiv_id, paper.version, paper.source_kind)
    if path.is_file():
        try:
            return CandidatePack.model_validate_json(path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("Ignoring candidate cache %s: %s", path, exc)
    pack = build_candidate_pack(paper, prompt)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(pack.model_dump_json(indent=2), encoding="utf-8")
    return pack


def _paper_from_prompt(prompt: PromptDocument) -> PaperDocument:
    kind = "pdf" if prompt.source_kind == "pdf" else "tex"
    if prompt.source_kind == "abstract":
        kind = "tex"
    return PaperDocument(
        arxiv_id=prompt.arxiv_id,
        version=prompt.version,
        source_kind=kind,  # type: ignore[arg-type]
        title=prompt.title,
        abstract=prompt.abstract,
        sections=list(prompt.selected_sections),
    )


def _prompt_for_source(
    hit: PaperHit,
    source_kind: SourceKind,
    config: ExtractionConfig,
    paper_doc: PaperDocument | None,
) -> tuple[PromptDocument, PaperDocument | None]:
    arxiv_id = normalize_arxiv_id(hit.arxiv_id)
    version = format_version(hit.latest_version)
    if source_kind == "abstract":
        title = (paper_doc.title if paper_doc and paper_doc.title else hit.title) or arxiv_id
        abstract = (paper_doc.abstract if paper_doc and paper_doc.abstract else hit.abstract) or ""
        return build_abstract_prompt(arxiv_id, version, title, abstract, config), paper_doc

    fetched = _fetch_source(hit, source_kind, config)
    parsed = _parse_source(fetched, config)
    if hit.title and not parsed.title:
        parsed.title = hit.title
    if hit.abstract and not parsed.abstract:
        parsed.abstract = hit.abstract
    _save_parsed(config, parsed)
    return build_prompt_document(parsed, config, source_kind=source_kind), parsed


def _fetch_source(hit: PaperHit, source_kind: SourceKind, config: ExtractionConfig) -> FetchedSource:
    arxiv_id = hit.arxiv_id
    version = hit.latest_version
    if source_kind == "tex":
        fetched = fetch_eprint(arxiv_id, version, config)
        if fetched.kind != "tex":
            raise FetchError("e-print is not a TeX archive")
        return fetched
    fetched = fetch_eprint(arxiv_id, version, config)
    if fetched.kind == "pdf":
        return fetched
    return fetch_pdf(arxiv_id, version, config)


def _parse_source(fetched: FetchedSource, config: ExtractionConfig) -> PaperDocument:
    cached = _load_parsed(config, fetched.arxiv_id, fetched.version, fetched.kind)
    if cached is not None:
        return cached
    if fetched.kind == "tex":
        try:
            return parse_tex_source(fetched, config)
        except ParseError:
            raise
        except Exception as exc:
            raise ParseError(str(exc)) from exc
    try:
        return parse_pdf_source(fetched)
    except PdfParseError:
        raise
    except Exception as exc:
        raise PdfParseError(str(exc)) from exc


def _load_parsed(
    config: ExtractionConfig, arxiv_id: str, version: str, source_kind: str
) -> PaperDocument | None:
    path = parsed_path(config, arxiv_id, version, source_kind)
    if not path.is_file():
        return None
    try:
        return PaperDocument.model_validate_json(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Ignoring unreadable parsed cache %s: %s", path, exc)
        return None


def _save_parsed(config: ExtractionConfig, paper: PaperDocument) -> None:
    path = parsed_path(config, paper.arxiv_id, paper.version, paper.source_kind)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(paper.model_dump_json(indent=2), encoding="utf-8")


def _load_extracted(
    config: ExtractionConfig, arxiv_id: str, version: str, fingerprint: str
) -> ExtractedPaper | None:
    path = extraction_dir(config, arxiv_id, version) / f"{fingerprint}.json"
    if not path.is_file():
        return None
    try:
        bundle = ExtractionBundle.model_validate_json(path.read_text(encoding="utf-8"))
        return bundle.extracted
    except Exception as exc:
        logger.warning("Ignoring unreadable extraction cache %s: %s", path, exc)
        return None


def _save_extracted(
    config: ExtractionConfig,
    fingerprint: str,
    prompt: PromptDocument,
    extracted: ExtractedPaper,
) -> None:
    folder = extraction_dir(config, extracted.arxiv_id, extracted.version)
    folder.mkdir(parents=True, exist_ok=True)
    bundle = ExtractionBundle(prompt_document=prompt, extracted=extracted)
    (folder / f"{fingerprint}.json").write_text(bundle.model_dump_json(indent=2), encoding="utf-8")


def _bound_complete(config: ExtractionConfig, provider: str | None) -> CompleteFn:
    def _call(*, system: str, user: str, **kwargs: Any) -> str:
        if not provider:
            raise RuntimeError("No LLM API key set for extraction")
        return complete(
            system=system,
            user=user,
            json_mode=True,
            temperature=0.0,
            timeout_s=config.llm_timeout_s,
            provider=provider,
        )

    return _call


def _metrics(records: list[ExtractedPaper], cache_hits: int, prepared: list[_Prepared]) -> dict:
    n = len(records) or 1
    ok = [r for r in records if r.status == "ok"]
    degraded = [r for r in records if r.status == "degraded"]
    skipped = [r for r in records if r.status == "skipped"]
    tex = sum(1 for r in records if r.status != "skipped" and r.source_kind == "tex")
    pdf = sum(1 for r in records if r.status != "skipped" and r.source_kind == "pdf")
    abstract = sum(1 for r in records if r.status == "degraded")
    proposed = sum(r.units_proposed for r in records)
    kept = sum(r.units_kept for r in records)
    relocated = sum(r.relocated_evidence for r in records)
    packs = [p.pack for p in prepared if p.pack is not None]
    selected = [p.selected_chars for p in packs]
    cand = [p.candidate_chars for p in packs]
    return {
        "n_papers": len(records),
        "ok": len(ok),
        "degraded": len(degraded),
        "skipped": len(skipped),
        "tex_rate": round(tex / n, 3),
        "pdf_fallback_rate": round(pdf / n, 3),
        "abstract_fallback_rate": round(abstract / n, 3),
        "skip_rate": round(len(skipped) / n, 3),
        "evidence_keep_rate": round(kept / proposed, 3) if proposed else 0.0,
        "section_relocate_rate": round(relocated / kept, 3) if kept else 0.0,
        "cache_hits": cache_hits,
        "mean_selected_chars": round(sum(selected) / len(selected), 1) if selected else 0.0,
        "mean_candidate_chars": round(sum(cand) / len(cand), 1) if cand else 0.0,
    }
