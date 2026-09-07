# Progress report: Research Assistance Agent — Stage 2

**Date:** 6 September 2026  
**Scope:** **Stage 2 — Extraction** is complete (fetch, parse, section select, grounded LLM extract, evidence gate, cache, CLI, unit tests). Synthesis / Writing are not started. Live gold ablation numbers are **not** in yet (`RUN_LIVE = False`).

---

## 1. Locked-in goal

Stage 1 finds papers. Stage 2 turns **each paper** into grounded knowledge units so Stage 3 can cluster and Stage 4 can cite without opening the PDF again.

Hard boundary: **no cross-paper synthesis in Stage 2**.

```
PaperHit (from Retrieval)
  → fetch TeX (else PDF)
  → PaperDocument (full parse)
  → PromptDocument (SectionPolicy budget)
  → LLM JSON
  → Pydantic structural gate
  → evidence gate vs selected sections
  → retry / PDF / abstract / skip
  → ExtractedPaper
```

Provenance chain for later stages:

```
Synthesis claim
    → ExtractedPaper
    → Claim | Mention
    → Evidence
    → paper / section / quote / char offsets
```

Immediate use case is still Data-Centric AI / dataset pruning.

---

## 2. What was built

Quality-first, not dump-the-whole-paper-into-the-LLM.

### 2.1 Fetch (TeX-first)

- `GET https://arxiv.org/e-print/{id}` with `User-Agent: research-assistant-agent/0.1`
- Detect payload by **magic bytes** (`%PDF`, gzip, tar), not Content-Type — arXiv often returns a PDF even on `/e-print`
- 404 / HTML → `https://arxiv.org/pdf/{id}.pdf`
- Process-wide throttle (~3s) on cache miss only; downloads are sequential
- **Safe tar:** no `extractall`. Reject `..`, absolute paths, symlink / device / fifo / hardlink. Caps on file count, bytes, nesting. Only text-like suffixes (`.tex`, `.bbl`, `.sty`, `.cls`, …)

Version is kept (`2205.09329v2`) so citations point at the parsed source. Retrieval still stores the versionless id.

### 2.2 Parse → `PaperDocument`

- **TeX:** unpack → pick main file (`\documentclass` + `\begin{document}`, tie-break `main`/`paper`/`ms`) → inline `\input`/`\include` (cycle-detect) → strip `%` → split `\section`/`\subsection`/`\abstract` → `pylatexenc` to text
- **PDF:** PyMuPDF; TOC bookmarks if present; hyphen join before normalize
- One unreadable source → `parse_fail` → next source. The pipeline does not crash

`PaperDocument.full_text` is diagnostic only. It is not sent to the LLM and is **not** enough for the evidence gate to pass.

### 2.3 `SectionPolicy` → `PromptDocument`

Parser and selector are separate. Ablation only changes `PaperDocument → PromptDocument`.

Default MVP: keep abstract / intro / method / experiments / limitations / conclusion. Drop related work, appendix, bibliography **in the selector**, not in the parser. Flags:

- `include_related_work` (CLI `--include-related-work`) — off by default; Stage 3 can turn it on
- `include_appendix` / `include_bibliography`
- `abstract_only` (CLI `--abstract-only`) — gold ablation
- `max_input_chars=24000`; `max_input_tokens` is a stub (no tokenizer yet)

### 2.4 LLM extract + two gates

Shared client: `research_assistant/llm/client.py` (query expansion was refactored onto it).

- Extraction provider order: **Gemini → OpenAI → Groq** (Groq 8B is last resort)
- Expansion still: Groq → Gemini → OpenAI
- Temperature 0, JSON mode, timeout 90s
- Prompt may only use `PromptDocument` sections. Missing fields → `null` / `[]`. Quotes ≤ ~40 words, verbatim

**Structural gate:** JSON (markdown fences stripped) → Pydantic v2 `CandidateExtraction`.

**Evidence gate** (implemented and unit-tested before the LLM loop):

1. Locate quote in the **declared** selected section (fuzzy heading, e.g. `3 Experiments` ≈ `Experiments`)
2. Else other **selected** sections → relocate heading, lower confidence
3. `full_text` hit only sets `in_full_text` (diagnostic). `valid=True` requires a selected section

Offsets (`start_char` / `end_char`) are written by the gate on **normalized section text**, not by the LLM.

`status=ok` only if `method` survives the gate. Abstract-only success → `degraded`. Exhausted sources → `skipped`.

### 2.5 Fallback state machine

Explicit loop, not nested `try/except`:

```
for source in [tex, pdf, abstract]:
    parse / select
    for attempt_no in range(max_retries + 1):   # default 2 retries
        LLM → structural gate → evidence gate
        record ExtractionAttempt
```

Every attempt is stored on `ExtractedPaper.attempts` (`ok`, `struct_fail`, `evidence_fail`, `llm_fail`, `parse_fail`) so a later “why is X degraded?” question is answerable.

### 2.6 Cache + fingerprint

```
data/extraction_cache/
└── 2205.09329/
    └── v2/
        ├── source/                      # tar or pdf (reuse if prompt changes)
        ├── parsed_v1_tex.json           # PaperDocument
        └── extraction/
            └── <pipeline_fingerprint>.json   # PromptDocument + ExtractedPaper
```

Fingerprint = SHA-256 of schema / prompt / selector / normalizer versions + **model** + `SectionPolicy` flags. Same fingerprint → no refetch, no re-parse, no re-extract. Changing Gemini→OpenAI or turning on related work is an intentional cache miss.

---

## 3. Data contract (what Stage 3 will consume)

**Semantic claims** (paraphrase + evidence list): `problem`, `method` (required for `ok`), `results`, `limitations`, `contributions`.

**Structured mentions** (name + one evidence span — not bare `list[str]`):

- `datasets`, `metrics`, `method_keywords`

Stage 3 clustering:

```python
[x.value for x in paper.method_keywords]
```

`ExtractedPaper.status`: `ok` | `degraded` | `skipped`.  
`confidence` drops for PDF, abstract fallback, relocated sections, and dropped units.

`ExtractionResult.metrics`: `tex_rate`, `pdf_fallback_rate`, `abstract_fallback_rate`, `skip_rate`, `evidence_keep_rate`, `section_relocate_rate`, `cache_hits`, `seconds`.

---

## 4. How to run

Install (venv):

```bash
pip install -e .
# pymupdf + pylatexenc are now package dependencies
```

Copy `.env.example` → `.env`. Extraction prefers `GEMINI_API_KEY` (default model `gemini-2.5-flash`). Groq is used for query expansion first, but is last-resort for extraction.

### From Stage 1 JSON

```bash
python -m research_assistant.retrieval \
  "dataset pruning and data subset selection for deep neural networks" \
  --top-k 25 \
  --json-out results/pruning.json

python -m research_assistant.extraction \
  results/pruning.json \
  --json-out results/pruning.extracted.json
```

CLI does **not** load FAISS/SPECTER2. Retrieval JSON is enough.

### Single paper (debug)

```bash
python -m research_assistant.extraction --arxiv-id 2205.09329v2 -v
```

Useful flags: `--include-related-work`, `--abstract-only`, `--no-abstract-fallback`, `--concurrency 2`, `--max-retries 2`, `--cache-dir …`.

### Python API

```python
from research_assistant import retrieve, extract_papers
from research_assistant.config import RetrievalConfig, ExtractionConfig

hits = retrieve("proxy maturity in dataset pruning", RetrievalConfig(top_k=25))
extracted = extract_papers(hits, ExtractionConfig())
for paper in extracted.records:
    kws = [m.value for m in paper.method_keywords]
    print(paper.status, paper.arxiv_id, paper.method is not None, kws)
```

Example script: `examples/extract_papers.py` (uses `results/pruning.json` if present, else `2205.09329`).

Entry point: `extract-papers` (same as `python -m research_assistant.extraction`).

---

## 5. Main files

| File | Role |
|---|---|
| `research_assistant/arxiv_ids.py` | Shared id parse; retrieval strips version, extraction keeps it |
| `research_assistant/llm/client.py` | Shared Groq / Gemini / OpenAI HTTP |
| `research_assistant/config.py` | `ExtractionConfig` + version strings |
| `research_assistant/extraction/types.py` | `Evidence`, `Claim`, `Mention`, documents, attempts |
| `research_assistant/extraction/fetch.py` | E-print/PDF, throttle, safe tar |
| `research_assistant/extraction/parse_tex.py` | TeX → `PaperDocument` |
| `research_assistant/extraction/parse_pdf.py` | PDF → `PaperDocument` |
| `research_assistant/extraction/sections.py` | `SectionPolicy` → `PromptDocument` |
| `research_assistant/extraction/normalize.py` | Whitespace / hyphen / heading fuzzy match |
| `research_assistant/extraction/validate.py` | JSON parse + evidence gate + offsets |
| `research_assistant/extraction/prompt.py` | Extractor system/user prompt |
| `research_assistant/extraction/fingerprint.py` | Cache key |
| `research_assistant/extraction/pipeline.py` | State machine + `extract_papers` |
| `research_assistant/extraction/cli.py` | `python -m research_assistant.extraction` |
| `examples/extract_papers.py` | Example API call |

Query expansion now calls the shared LLM client; behavior is unchanged (templates if no key).

---

## 6. Tests

`pytest tests/` — **37 passed** (6 September 2026). No GPU, no live arXiv, no live LLM:

- Safe tar path traversal + symlink skip; magic-byte sniff
- Mini TeX fixture: `\input` inlining, section split, related-work dropped by default, `abstract_only`, char budget
- Synthetic PDF parse (PyMuPDF)
- Evidence gate: declared-section hit, relocate to another selected section, **full_text-only does not pass**, mentions keep provenance, JSON fences
- Pipeline: LLM failure → skip without crash; abstract fallback; fingerprint cache hit does not recall the LLM; one bad paper does not fail the batch
- Scoring helpers (precision/recall, evidence-support rates)
- Existing retrieval tests still pass (Gemini thought-part helper moved to `llm.client`)

---

## 7. Evaluation (harness ready; live numbers not run)

Eval stays **out of** the production package, same pattern as retrieval.

| File | Role |
|---|---|
| `evaluation/extraction_gold.json` | 8 thesis-related papers (incl. `2205.09329`, `2107.07075`, CCS, InfoBatch, DeepCore, …) with keywords / datasets / metrics / limitation + support labels |
| `evaluation/extraction_scoring.py` | Completeness, field P/R, evidence keep rate, evidence precision (`supported` / `partially_supported` / `unsupported`) |
| `evaluation/extraction_eval.ipynb` | Default `RUN_LIVE = False`; scores `extraction_eval.json` if present |

Substring verification only proves the quote **exists**. Gold `evidence_samples` are for **support** (a quote can exist and still not back the claim).

Ablation is selector-only (`abstract_only=True` vs default selected sections), not a second parser. DoD wants abstract-only to lose on completeness **or** field accuracy, without evidence precision dropping. That comparison needs a live run:

1. Set `RUN_LIVE = True` in `evaluation/extraction_eval.ipynb` (needs network + Gemini/OpenAI key), or
2. Extract gold ids and score:

```bash
python -m research_assistant.extraction \
  --arxiv-id 2205.09329 --arxiv-id 2107.07075 \
  --json-out evaluation/extraction_eval.json
```

---

## 8. Design checklist

| Design item | Status |
|---|---|
| TeX-first fetch + PDF fallback + abstract degraded | Done |
| Safe tar extraction | Done |
| `PaperDocument` vs `PromptDocument` | Done |
| Configurable `SectionPolicy` (related work not hard-deleted in parser) | Done |
| Pydantic schema: Claim + Mention + Evidence | Done |
| Evidence gate vs selected sections (not `full_text`) | Done |
| Offsets written by gate | Done |
| Explicit `ExtractionAttempt` fallback loop | Done |
| Cache + `pipeline_fingerprint` (includes model + policy) | Done |
| Shared LLM client; Gemini default for extraction | Done |
| CLI from Retrieval JSON / `--arxiv-id` | Done |
| Unit tests without live LLM | Done (37) |
| Gold file + eval notebook | Done (harness) |
| Live gold ablation numbers | **Not run** |
| Stage 3 method clustering | Not started |
| Stage 4 writing + citation validation | Not started |
| LangGraph | Not started (still plain Python; correct for Stage 2) |

### Definition of Done

1. Top-k Stage 1 JSON → Stage 2 CLI — **yes** (needs network + key for a real run)
2. One paper failure does not crash the pipeline — **yes** (unit-tested)
3. Every `ok` claim and mention has `Evidence` — **yes** (schema + gate)
4. Evidence must match a selected section; full_text-only fails — **yes** (unit-tested)
5. Rerun same fingerprint does not refetch/re-extract — **yes** (unit-tested)
6. Abstract-only vs selected-sections on gold — **harness only**
7. JSON is enough for Stage 3 to cluster without reopening PDF — **yes** (`method` + `method_keywords[].value`)

---

## 9. Suggested next steps

1. **Live extraction pass** on `results/pruning.json` (or gold 8 papers). Log `tex_rate` / skip_rate / evidence_keep_rate. Fix parse failures that show up on real TeX.
2. Run `evaluation/extraction_eval.ipynb` with `RUN_LIVE = True` and fill DoD #6 (selected-sections should beat abstract-only).
3. Optionally compare LLM query expansion vs templates on the Stage 1 qrels (still open from the Stage 1 report) — independent of extraction.
4. **Stage 3 — Synthesis:** embed `method` / `method_keywords`, cluster, per-cluster summaries. Do not re-parse PDFs; only read `ExtractedPaper` JSON. Related-work comparisons can set `include_related_work=True` and re-extract (new fingerprint).
5. Incremental corpus update (Stage 1 leftover) remains optional.
