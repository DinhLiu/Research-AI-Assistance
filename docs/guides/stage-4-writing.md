# Stage 4 — Writing

Stage 4 consumes `SynthesisResult` JSON and produces a review in Markdown plus a
JSON provenance report. It does not reopen papers, run retrieval, or alter
clusters. Offline templates are the default; `--write` enables prose generation.

## Run

```bash
python -m research_assistant.writing results/pruning.synthesis.json \
  --dry-run --language en --target-words 1500

python -m research_assistant.writing results/pruning.synthesis.json \
  --md-out results/pruning.review.md --json-out results/pruning.writing.json

python -m research_assistant.writing results/pruning.synthesis.json \
  --write --md-out results/pruning.review.md --json-out results/pruning.writing.json
```

`write-review` is also installed as a console entry point after `pip install -e .`.
`--language vi` requests Vietnamese prose; template fallbacks retain the original
evidence language. `--require-complete` exits 1 for partial/fallback output;
invalid input/config exits 2. Dry-run prints packing, omissions, quota and budgets
without making a network request. Use `--write --dry-run` to preview live behavior.

## Live configuration and behavior change for existing stages

Set the API key/model as before, then fill `LLM_RPM`, `LLM_TPM`, `LLM_RPD` from
the actual project quota. See `.env.example`. No universal free-tier numbers are
assumed. **The shared client now requires these values for all live LLM stages**;
without them, the existing stages take their failure/fallback path and writing
returns an offline draft with `quota_configuration_required`.

All `complete()` calls now use the same local persistent governor. The default
state directory is `data/llm_state` at the repository root. Set `LLM_STATE_DIR` to
one shared absolute directory when using multiple clones. `LLM_QUOTA_GROUP` can
name a shared project quota; by default all models for one provider share a
conservative bucket. Never put a secret in that name.

Admission reserves UTF-8-byte input estimates plus max output allowance against
local token limits. This intentionally overestimates most input token counts.
The TPM reservation includes output even for providers that meter only input;
actual response usage is reported separately when available. RPM/TPM use a
sliding minute; RPD uses a conservative rolling 24 hours, independent of provider
reset timezones. Quota state cannot account for applications using the same key
outside this governor or prevent provider overload.

The local POSIX implementation uses a kernel file lock through admission and
transport, plus SQLite reservations. A crash releases the lock but preserves
spent budgets, pacing and cooldown. One in-flight request is allowed across
local callers; no idle permit accumulation. The pacing floor is 15 seconds,
adjusted upward for RPM and token pressure. Requests that exceed TPM are rejected
before sending. 429/5xx retry once through the same admission path; two consecutive
transient failures open the circuit for at least 60 seconds. Retry-After seconds,
HTTP dates and Gemini RetryInfo are honored. Daily/billing/auth failures do not
trigger content repair or key switching.

## Writing budgets, cache and resume

A small review normally uses one generation request. Packing caps generation at
three sequential batches, with one repair for missing/invalid claims across the
whole run. There are at most five HTTP attempts, including transport retries.
The output/context caps default to 6,000/32,768 tokens and input estimate to 20,000;
set the CLI token flags for the actual model. Output sizing is an estimate; a
truncated provider response is rejected and can consume the single repair.

Scope text, headings and bibliography are rendered by code. Packing may split a
section at evidence-slot boundaries. Slots that do not fit three batches retain
template text and explicit omission reasons. Output completeness requires all
selected slots to have valid prose and every manifest paper to have selected
support; it does not mean every evidence unit was narrated. The plan records
unselected evidence IDs and coverage separately.

Successful claims are cached and revalidated on load. Partial caches cannot
claim a complete review; subsequent new runs generate only the missing slots.
Per-input cache locks prevent duplicate concurrent generations. `--no-cache`
disables shared prose reuse; per-run checkpoints still protect resume.

Each live result exposes `execution.run_id`. Pass it through `--run-id` to resume
the same content/config. Attempted batches, repair count and HTTP/token budget
persist. A reused ID with different content/config is rejected. Resuming does not
renew the ten-minute admission deadline; omit the ID to start a new run. HTTPX
request timeouts bound network inactivity, not a strict total wall-clock deadline
for an already active response; no new request is admitted after the run deadline.
There is no cross-machine distributed coordinator.

For a combined Python pipeline, an optional parent budget includes all stages:

```python
from research_assistant.llm.governor import RunBudget, llm_run

with llm_run(RunBudget("my-pipeline-run", max_attempts=12, max_tokens=150000)):
    # Existing retrieval/extraction/synthesis calls use this budget automatically.
    # write_review() creates its own five-attempt child budget as well.
    ...
```

## Evidence guarantees and limits

The model receives fixed claim IDs, source text, subjects and evidence units, and
returns only `{claim_id, text}` objects. It cannot supply citation metadata or a
bibliography. Code validates input ownership/identity and maps accepted text back
to immutable support refs. Unknown IDs, duplicate claims, foreign arXiv versions,
injected citations/URLs and common unsupported performance-ranking phrases fail
validation. Valid claims survive repair failure; templates preserve source text. Clearly recognized
document-formatting boilerplate is excluded from claim selection and recorded in
`plan.evidence_warnings`; this narrow heuristic does not certify other evidence.

These checks establish structural reference integrity, **not semantic entailment**.
An LLM can still misstate evidence in fluent prose. Every claim remains scoped to
the extraction snapshot and marked `verification_needed`. Limitations do not prove
absence of prior research; dataset/metric mentions do not prove evaluation links.
Review prose manually before using it as verified academic writing. Authors/year
are not invented when the upstream manifest lacks them.

## Validation

Run `python -m pytest -q`. Tests use fake transport/clock and local concurrency;
no LLM API calls are made. Coverage includes pacing, retry accounting, TPM/RPM,
parent/run caps, persistent resume, cache invalidation, token packing, structured
citations, truncated responses and CLI JSON/Markdown output. Use
`python -m evaluation.writing_scoring result.writing.json` for an offline execution
and coverage summary. Live quality/latency measurements and human entailment
annotations remain separate evaluation work; no provider benchmark is claimed.
