"""Offline-first literature reviews with bounded, resumable prose generation."""
from __future__ import annotations

import json
import time
from dataclasses import asdict
from uuid import uuid4

import httpx

from research_assistant.config import WritingConfig
from research_assistant.extraction.validate import extract_json_object
from research_assistant.llm.client import Completion, complete_result, estimate_tokens, extraction_provider, model_for_provider
from research_assistant.llm.governor import Governor, LlmError, Quota, RunBudget, file_lock, current_budget
from research_assistant.synthesis.types import SynthesisResult
from research_assistant.writing.fingerprint import atomic_write, digest, load_json
from research_assistant.writing.plan import SYSTEM_PROMPT, build_plan, user_prompt, output_estimate
from research_assistant.writing.types import WritingInputError, WritingResult, WrittenPayload
from research_assistant.writing.validate import validate_input, validate_written


def write_review(synthesis, config=None, *, complete_fn=None, governor=None, quota=None, parent_budget=None):
    cfg = config or WritingConfig()
    # Validate even if callers mutated a dataclass after construction.
    cfg.__post_init__()
    started = time.perf_counter()
    try:
        snapshot = SynthesisResult.model_validate(synthesis) if not isinstance(synthesis, SynthesisResult) else synthesis
        validate_input(snapshot)
    except ValueError as exc:
        raise WritingInputError(str(exc)) from exc
    quota_reason = None
    if quota is None and (cfg.use_llm or cfg.dry_run):
        try:
            quota = Quota.from_env()
        except LlmError as exc:
            quota_reason = str(exc)
    plan = build_plan(snapshot, cfg, quota)
    source = snapshot.model_dump(exclude={"execution"})
    input_digest = digest(source)
    result = WritingResult(writing_input_digest=input_digest, corpus_digest=snapshot.corpus_digest,
                           topic=snapshot.topic, generation_status="fallback" if snapshot.paper_manifest else "empty", plan=plan,
                           claims=[slot.model_copy(deep=True) for slot in plan.slots])
    ex = result.execution
    ex.quota = asdict(quota) if quota else {}
    if cfg.dry_run:
        result.generation_status = "planned"
        ex.stop_reason = quota_reason
        return _finish(result, snapshot, started)
    if not cfg.use_llm or not plan.slots:
        ex.stop_reason = "offline" if plan.slots else "no_supported_evidence"
        return _finish(result, snapshot, started)
    if quota is None:
        ex.stop_reason = quota_reason or "quota_configuration_required"
        return _finish(result, snapshot, started)
    provider = "injected" if complete_fn else extraction_provider(cfg.prefer_provider)
    if not provider:
        ex.stop_reason = "no_api_key"
        return _finish(result, snapshot, started)
    ex.provider = provider
    ex.model = "injected" if complete_fn else model_for_provider(provider)
    registry = snapshot.evidence_registry.lookup()
    control = governor or Governor()
    ex.run_id = cfg.run_id or str(uuid4())
    budget = RunBudget(ex.run_id, cfg.max_http_attempts_per_run, cfg.max_run_tokens, cfg.run_deadline_s, parent_budget or current_budget())
    deadline = control.start_run(budget)
    stable_config = asdict(cfg)
    for field in ("cache_dir", "use_cache", "dry_run", "run_id", "run_deadline_s", "request_timeout_s",
                  "max_http_attempts_per_run", "max_run_tokens"):
        stable_config.pop(field)
    cache_key = digest([input_digest, plan.model_dump(), stable_config, provider, ex.model])
    cache_path = cfg.cache_dir / "prose" / f"{cache_key}.json"
    checkpoint_path = cfg.cache_dir / "runs" / f"{digest(ex.run_id)}.json"
    accepted = {}
    logical = repair = 0
    attempted_batches = set()
    usage_complete = {"input_tokens": True, "output_tokens": True}
    all_slots = {s.claim_id: s for s in plan.slots}

    def recover(payload):
        if not isinstance(payload, dict) or payload.get("cache_key") != cache_key:
            return {}
        try:
            written = WrittenPayload.model_validate({"claims": payload["claims"]})
            # Missing claims in partial cache are expected, invalid ones are dropped.
            return validate_written(written, plan.slots)[0]
        except (KeyError, ValueError):
            return {}

    def save():
        payload = {"cache_key": cache_key, "claims": [{"claim_id": key, "text": c.text} for key, c in sorted(accepted.items())],
                   "logical_calls": logical, "repair_calls": repair, "attempted_batches": sorted(attempted_batches)}
        atomic_write(checkpoint_path, json.dumps(payload, ensure_ascii=False))
        if cfg.use_cache and accepted:
            atomic_write(cache_path, json.dumps(payload, ensure_ascii=False))

    def call(slots, *, repairing=False):
        nonlocal logical, repair
        prompt = user_prompt(plan, slots, registry)
        if repairing:
            prompt += '\nReturn all requested slots. Correct missing/invalid text; do not emit citations or extra fields.'
        tokens = estimate_tokens(SYSTEM_PROMPT + prompt)
        if tokens > cfg.max_input_tokens or tokens + cfg.max_output_tokens > cfg.context_tokens:
            ex.stop_reason = "repair_prompt_exceeds_budget" if repairing else "prompt_exceeds_budget"
            return
        logical += 1
        repair += int(repairing)
        # Reserve logical repair/generation before transport so crash cannot reset it.
        save()
        if complete_fn:
            def send_once(timeout):
                value = complete_fn(system=SYSTEM_PROMPT, user=prompt)
                completion = value if isinstance(value, Completion) else Completion(value, "injected", "injected")
                return httpx.Response(200, json=asdict(completion))
            response = control.request(send_once, group="injected", quota=quota, budget=budget,
                                       tokens=tokens + cfg.max_output_tokens, timeout_s=cfg.request_timeout_s)
            completion = Completion(**response.json())
        else:
            completion = complete_result(system=SYSTEM_PROMPT, user=prompt, provider=provider,
                                         max_output_tokens=cfg.max_output_tokens, timeout_s=cfg.request_timeout_s,
                                         governor=control, quota=quota, budget=budget)
        for name in ("input_tokens", "output_tokens"):
            value = getattr(completion, name)
            if value is None:
                usage_complete[name] = False
                setattr(ex, name, None)
            elif usage_complete[name]:
                setattr(ex, name, (getattr(ex, name) or 0) + value)
        if completion.truncated:
            result.validation_errors.append("output_truncated")
            return
        try:
            payload = WrittenPayload.model_validate(extract_json_object(completion.text))
            valid, errors = validate_written(payload, slots)
            accepted.update(valid)
            result.validation_errors.extend(errors)
        except ValueError:
            result.validation_errors.append("invalid_writing_json")
        save()

    try:
        # Lock the run as well as the content cache: same-ID resumes and distinct
        # runs with identical input both recheck after single-flight admission.
        with file_lock(checkpoint_path.with_suffix(".lock"), clock=control.clock, sleep=control.sleep, deadline=deadline):
            with file_lock(cache_path.with_suffix(".lock"), clock=control.clock, sleep=control.sleep, deadline=deadline):
                previous = load_json(checkpoint_path)
                if previous is None and checkpoint_path.exists():
                    raise WritingInputError("corrupt_run_checkpoint; use a new explicit run ID")
                if previous is not None:
                    if not isinstance(previous, dict) or previous.get("cache_key") != cache_key:
                        raise WritingInputError("run_id_already_used_for_different_input_or_config")
                    accepted.update(recover(previous))
                    logical, repair = int(previous.get("logical_calls", 0)), int(previous.get("repair_calls", 0))
                    attempted_batches.update(previous.get("attempted_batches", []))
                    if not 0 <= repair <= 1 or not 0 <= logical <= 4 or not attempted_batches <= {b.batch_id for b in plan.batches}:
                        raise WritingInputError("invalid_run_checkpoint")
                if cfg.use_cache:
                    accepted.update(recover(load_json(cache_path)))
                ex.cache_hits = len(accepted)
                for batch in plan.batches:
                    missing = [all_slots[key] for key in batch.claim_ids if key not in accepted]
                    if missing and batch.batch_id not in attempted_batches:
                        attempted_batches.add(batch.batch_id)
                        call(missing)
                # One repair across the run, bounded by the same prompt/token limits.
                missing = [s for s in plan.slots if s.claim_id not in accepted and s.claim_id not in plan.omitted]
                if missing and repair < cfg.max_repair_calls:
                    repair_slots = []
                    for slot in missing:
                        prompt = user_prompt(plan, repair_slots + [slot], registry)
                        cost = estimate_tokens(SYSTEM_PROMPT + prompt) + 128
                        if (cost <= cfg.max_input_tokens and cost + cfg.max_output_tokens <= cfg.context_tokens and
                                cost + cfg.max_output_tokens <= int(quota.tpm * quota.safety) and
                                output_estimate(plan, repair_slots + [slot]) <= cfg.max_output_tokens):
                            repair_slots.append(slot)
                    if repair_slots:
                        call(repair_slots, repairing=True)
                save()
    except LlmError as exc:
        ex.stop_reason = str(exc)
    finally:
        ex.logical_calls, ex.repair_calls = logical, repair
        stats = control.stats(ex.run_id)
        ex.http_attempts, ex.reserved_tokens = stats["http_attempts"], stats["reserved_tokens"]
        for name, value in control.telemetry(ex.run_id).items():
            setattr(ex, name, value)
    result.claims = [accepted.get(s.claim_id, s) for s in plan.slots]
    result.generation_status = "complete" if len(accepted) == len(plan.slots) else ("partial" if accepted else "fallback")
    if plan.omitted:
        ex.stop_reason = ex.stop_reason or "budget_insufficient"
    elif len(accepted) < len(plan.slots):
        ex.stop_reason = ex.stop_reason or "validation_incomplete"
    return _finish(result, snapshot, started)


def _finish(result, snapshot, started):
    supported = {key for c in result.claims for key in c.subject_paper_keys}
    papers = {p.paper_key for p in snapshot.paper_manifest}
    result.bibliography = [{"paper_key": p.paper_key, "title": p.title,
                            "url": f"https://arxiv.org/abs/{p.paper_key}"}
                           for p in sorted(snapshot.paper_manifest, key=lambda p: p.paper_key) if p.paper_key in supported]
    result.coverage = {"total_papers": len(papers), "supported_papers": len(supported),
                       "unsupported_paper_keys": sorted(papers - supported),
                       "llm_claims": sum(c.generation == "llm" for c in result.claims),
                       "template_claims": sum(c.generation == "template" for c in result.claims),
                       "unassigned": [p.model_dump() for p in snapshot.unassigned],
                       "related_work_policy": snapshot.related_work_policy,
                       "unknown_coverage_preserved": True}
    if papers - supported and result.generation_status == "complete":
        result.generation_status = "partial"
        result.execution.stop_reason = result.execution.stop_reason or "missing_evidence"
    result.execution.seconds = round(time.perf_counter() - started, 4)
    result.validation_errors = list(dict.fromkeys(result.validation_errors))
    return result
