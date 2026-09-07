# Progress report: Research Assistance Agent — Stage 3

**Date:** 7 September 2026  
**Scope:** **Stage 3 — Synthesis** is complete (identity/inventory, two-channel TF-IDF, distance-cut clustering, evidence registry, descriptive coverage, citation-gated narration, two-layer cache, CLI, unit tests). Writing is not started. Clustering threshold is **provisional**; live ARI on a real extraction snapshot is **not** in yet (`RUN_LIVE = False`).

---

## 1. Locked-in goal

Stage 1 finds papers. Stage 2 turns each paper into grounded `ExtractedPaper` records. Stage 3 **groups that snapshot by method** and optionally narrates with structural citation gates so Stage 4 can write without reopening PDFs or inventing clusters.

Hard boundaries:

- **No source reopen.** Input is `ExtractionResult` JSON only.
- **Clustering is algorithmic.** The LLM never returns or edits membership.
- **Output describes this snapshot.** It cannot establish absence of prior work, retrieval completeness, or extraction completeness.
- Dataset/metric fields are **mentions**, not evaluation relations.

```
ExtractionResult
  → validate snapshot + paper_key identity
  → inventory (disposition + reason for every input row)
  → EvidenceRegistry + PaperCard
  → TF-IDF method-text ⊕ keyword-phrase
  → agglomerative average-linkage (precomputed cosine, distance cut)
  → freeze ClusterAssignment
  → descriptive coverage + GapCandidate
  → optional LLM / strict-evidence templates
  → structural claim gate
  → SynthesisResult
```

Provenance chain for Stage 4:

```
Synthesis claim
  → support_refs (EvidenceRegistry unit ids)
  → ExtractedPaper field + quote/section
  → paper_key (arxiv_id + version)
```

Immediate use case is still Data-Centric AI / dataset pruning.

---

## 2. What was built

Deterministic core first; LLM only for optional narration.

### 2.1 Identity and inventory

- `paper_key = canonical arxiv_id + vN` via `research_assistant/arxiv_ids.py`
- Every input row is retained with a disposition: `accepted`, `skipped`, `missing_method`, `empty_features`, `duplicate`, `superseded_version`, `strict_ok_excluded`
- Identical duplicates keep the earliest row. Same id+version with **different** consumed fields is an input error (`SynthesisInputError`), not a silent overwrite
- Multiple versions: keep the highest numeric version that is eligible
- Cluster only `ok` / `degraded` (or `--strict-ok`) with **nonblank method text**. Missing keywords alone is not exclusion
- Empty input / all skipped → valid empty result with `empty_reason`

### 2.2 Evidence registry and paper cards

Immutable `EvidenceRegistry` built from existing claims/mentions. Each unit id is a digest of `paper_key | field_path | text`. No PDF/TeX fetch.

`PaperCard` is the compact view Stage 3/4 use: field text + evidence-unit ids. Extraction `confidence` is copied as **heuristic provenance**, not a calibrated probability.

### 2.3 Features (TF-IDF baseline)

Two independently normalized channels, then row L2-normalize:

- **Method text:** word unigrams/bigrams, `min_df=1`, `sublinear_tf=True`
- **Keywords:** phrases kept intact (tab-joined), not split into generic words
- Concatenate `sqrt(1-w) · text` with `sqrt(w) · keywords`, `w=0.5` development baseline
- Missing channel uses the available one. Zero rows → `empty_features` (unassigned, not a semantic cluster)
- Narrow alias map only (`data pruning` → `dataset pruning`). Dataset/metric names are **not** feature channels
- Vocabulary fitted in canonical `paper_key` order so reordering input does not change clusters

SPECTER2 / FAISS are **not** loaded. Abstract-level retrieval embeddings would group by topic, not method. sklearn TF-IDF is the declared baseline, not a demonstrated optimum.

### 2.4 Clustering and abstention

Sanitized dense cosine distance (clip, symmetrize, exact zero diagonal). `AgglomerativeClustering(linkage="average", metric="precomputed", n_clusters=None, distance_threshold=…)`.

Default cut `distance_threshold=0.55` is **provisional** (`distance_threshold_status=provisional`). It admits one group, several groups, and singletons. Singleton means “no merge under this representation/threshold”, not scientific novelty.

Silhouette is a diagnostic only when `2 ≤ k ≤ n-1`; otherwise `null` plus a reason. Low silhouette sets `ambiguity_warning`; papers are **not** auto-moved to an `other` cluster.

Cluster ids hash sorted member keys + clustering version (stable for the same snapshot). Labels are descriptive keyword contrast, not verified taxonomy.

### 2.5 Coverage and gaps

Per-paper states: `observed` / `unknown` (MVP does not auto-emit `explicitly_not_evaluated`). Empty mention lists map to `unknown`, not “did not evaluate”.

`GapCandidate` replaces invented research-gap claims:

- `documented_limitation` from extracted limitation text + support refs
- one snapshot-level `relationship_not_observed` stating that method–dataset evaluation relations were **not recorded** in this JSON

`verification_needed=True` on all automatic candidates. Numeric result ranking is rejected (`comparability=unknown`; score-like numbers in comparison text fail the gate).

### 2.6 Optional narration + structural gate

Default `--summarize` is **off**. Paths:

| Mode | Behavior |
|---|---|
| default | clustering only; `summary_status=not_requested` |
| `--strict-evidence` | template wording from extracted method text; no paraphrase |
| `--summarize` | one LLM completion + at most one repair (`max_logical_calls=2`) |

LLM JSON is `extra=forbid` and **must not** contain assignments. Claims carry `text`, `kind` (`shared` / `difference` / `subset`), `subject_paper_keys`, `support_refs`.

The validator checks reference integrity, not entailment:

1. refs exist in the registry and belong to the permitted cluster papers
2. field kinds match the claim kind
3. every named/subject paper has a ref from that paper
4. shared / difference need two supported sides
5. all-members claims need every member

Accepted free-text is `structurally_validated`. `semantically_verified` is reserved for later human/eval support checks. Invalid claims are dropped independently; valid ones are kept. Oversized prompts skip the LLM (`prompt_exceeds_budget`). Missing API key → `unavailable` and keep deterministic clusters. Provider HTTP backoff in `llm/client.py` is still internal (up to four transport attempts); Stage 3 counts **logical** calls only.

### 2.7 Cache

Two keys. **Corpus digest** hashes consumed record fields, not Stage 2’s pipeline fingerprint (that fingerprint is provenance only — same config can produce different corpora).

```
data/synthesis_cache/
  cluster/<cluster_key>.json      # membership, coverage, registry
  narration/<narration_key>.json  # only complete/partial successes
```

- Cluster key: corpus digest + selection/feature/clustering versions
- Narration key: cluster key + evidence digest + prompt/validator/model/truncation
- Offline clustering must not satisfy a requested successful narration entry
- Failed narration is not cached (retryable)
- Corrupt JSON is ignored; writes are temp-file + rename
- `execution.seconds` is always the current run

---

## 3. Data contract (what Stage 4 will consume)

`SynthesisResult`: `schema_version`, `topic`, `corpus_digest`, `upstream_pipeline_fingerprint`, `related_work_policy` (currently `unknown` — Stage 2 does not yet export a policy manifest), `effective_config`, `input_inventory`, `paper_manifest`, `assignments`, `unassigned`, `evidence_registry`, `descriptive_coverage`, `summaries`, `comparisons`, `gap_candidates`, `diagnostics`, `execution`.

`summary_status`: `not_requested` | `unavailable` | `complete` | `partial` | `failed`. Empty summaries must not be read as “no findings”.

Stage 4 must:

- resolve `support_refs` against the registry
- verify `corpus_digest` / schema version
- preserve `unknown` coverage (do not convert to a scientific gap)
- treat `structurally_validated` as reference-integrity, not verified fact

---

## 4. How to run

Install (venv; `scikit-learn>=1.3,<2` is now a package dependency):

```bash
pip install -e .
```

### From Stage 2 JSON

```bash
python -m research_assistant.extraction \
  results/pruning.json \
  --json-out results/pruning.extracted.json

python -m research_assistant.synthesis \
  results/pruning.extracted.json \
  --json-out results/pruning.synthesis.json
```

CLI does **not** load FAISS/SPECTER2. Extraction JSON is enough.

Useful flags: `--summarize`, `--strict-evidence`, `--strict-ok`, `--distance-threshold 0.55`, `--no-cache`, `--require-summary`, `-v`.

`--require-summary` exits `1` when narration is not `complete`/`partial` (for automation). Invalid JSON / conflicting records exit `2`. Provider failure still writes a valid partial result unless `--require-summary`.

### Python API

```python
from research_assistant import synthesize
from research_assistant.config import SynthesisConfig
from research_assistant.extraction.types import ExtractionResult

extracted = ExtractionResult.model_validate_json(Path("results/pruning.extracted.json").read_text())
result = synthesize(extracted, SynthesisConfig(summarize=False))
for cluster in result.assignments:
    print(cluster.label, cluster.paper_keys)
```

Example script: `examples/synthesize_papers.py`.

Entry point: `synthesize-papers` (same as `python -m research_assistant.synthesis`).

---

## 5. Main files

| File | Role |
|---|---|
| `research_assistant/arxiv_ids.py` | Shared `paper_key` / `version_number` |
| `research_assistant/config.py` | `SynthesisConfig` + version strings |
| `research_assistant/synthesis/types.py` | Inventory, cards, claims, coverage, `SynthesisResult` |
| `research_assistant/synthesis/cards.py` | Identity, registry, cards |
| `research_assistant/synthesis/features.py` | Two-channel TF-IDF |
| `research_assistant/synthesis/cluster.py` | Distance matrix, agglomerative cut, labels |
| `research_assistant/synthesis/coverage.py` | Mention coverage + `GapCandidate` |
| `research_assistant/synthesis/validate.py` | Structural claim / comparison gate |
| `research_assistant/synthesis/prompt.py` | Narration prompt + truncation budget |
| `research_assistant/synthesis/fingerprint.py` | Corpus digest, cache keys, atomic IO |
| `research_assistant/synthesis/pipeline.py` | `synthesize()` |
| `research_assistant/synthesis/cli.py` | `python -m research_assistant.synthesis` |
| `examples/synthesize_papers.py` | Example API call |

---

## 6. Tests

`pytest tests/` — **85 passed** (7 September 2026). No GPU, no live arXiv, no live LLM, no FAISS/SPECTER2 init in synthesis modules:

- Empty input, all skipped, blank method, missing keywords, punctuation-only empty vocabulary, identical vectors, n=1/2/3, dissimilar singletons
- Same-id duplicates, conflicting same-version records, version selection, `--strict-ok`, input reorder → stable cluster ids
- Unknown / cross-cluster refs, named unsupported subjects, invented `assignments` field (`extra=forbid`), mixed valid/invalid claims, numeric comparison reject
- Missing dataset stays `unknown`; mentions do not fabricate evaluation pairs
- Same upstream fingerprint + different records → cluster cache miss; evidence change invalidates narration; summarize on/off distinct; corrupt cache recoverable
- Missing key, timeout, malformed JSON + repair, repair exhaustion, oversized prompt; `--strict-evidence` templates
- Offline CLI JSON round-trip + Stage-4-style `support_refs` resolution; synthesis sources do not import `research_assistant.retrieval`

---

## 7. Evaluation (harness ready; live numbers not run)

Eval stays **out of** the production package.

| File | Role |
|---|---|
| `evaluation/synthesis_gold.json` | 8 thesis-related papers as a **smoke fixture** (exclusive cluster, paper role, overlapping tags, uncertain pairs) — **not** independent clustering ground truth |
| `evaluation/synthesis_scoring.py` | ARI, pairwise P/R, singleton rate, citation keep rate with numerator/denominator (`rate=null` if no claims) |
| `evaluation/synthesis_eval.ipynb` | Default `RUN_LIVE = False`; scores `synthesis_eval.json` if present |

Do not report generalization from these eight labels. Calibrate `distance_threshold` / `keyword_weight` on independent development annotations before held-out evaluation. Purity alone rewards singleton partitions; pairwise metrics and ARI on the exclusive subset are the intended clustering scores.

---

## 8. Design checklist

| Design item | Status |
|---|---|
| Consume `ExtractionResult` only; no PDF/TeX reopen | Done |
| `paper_key` identity + full inventory with reasons | Done |
| Duplicate / conflict / version selection | Done |
| Evidence registry from existing claims/mentions | Done |
| Two-channel TF-IDF (text + keyword phrases) | Done |
| Average-linkage + precomputed cosine + distance cut | Done |
| Silhouette diagnostic without forced k / `other` dump | Done |
| Descriptive coverage; no method×dataset Cartesian product | Done |
| `GapCandidate` scoped to snapshot; `verification_needed` | Done |
| Frozen membership; LLM `extra=forbid` | Done |
| Structural citation gate (not entailment) | Done |
| `--strict-evidence` templates | Done |
| Two-layer cache; corpus digest ≠ extraction fingerprint | Done |
| CLI + `complete_fn` injection; no retrieval imports | Done |
| Unit tests without live LLM/GPU | Done (85 total repo tests) |
| Smoke gold + scoring notebook | Done (harness) |
| Threshold calibration / held-out ARI | **Not run** |
| Stage 4 writing + citation validation of prose | Not started |
| LangGraph | Not started (still plain Python; correct for Stage 3) |

### Definition of Done

1. Stage 2 JSON → Stage 3 CLI without opening PDFs — **yes**
2. Conflicting same-version records fail loudly — **yes** (unit-tested)
3. Every clustered/unassigned paper is exclusive; excluded rows have reasons — **yes** (unit-tested)
4. Cluster ids stable under input reorder — **yes** (unit-tested)
5. LLM cannot invent assignments or cite ids outside the cluster — **yes** (unit-tested)
6. Missing dataset ≠ “not evaluated” — **yes** (unit-tested)
7. Same extraction fingerprint, different records → cache miss — **yes** (unit-tested)
8. No API key still returns clustering — **yes** (unit-tested)
9. Calibrated clustering quality vs independent labels — **harness only**

---

## 9. Suggested next steps

1. Produce a real `results/pruning.extracted.json` (Stage 2 live pass still open) and run Stage 3 on it. Inspect cluster labels by hand before trusting the provisional threshold.
2. Create **independent** synthesis annotations (primary method axis, paper role, overlapping tags, uncertain pairs) on a development split; tune `distance_threshold` and `keyword_weight`; then score held-out papers. Do not treat `evaluation/synthesis_gold.json` as that set.
3. Optionally add a small method-text encoder **only if** lexical failures show up on that labeled set. SPECTER2 cost alone is not the decision.
4. Stage 2 leftover: export a related-work policy manifest on `ExtractionResult` so Stage 3 can stop marking `related_work_policy=unknown`.
5. **Stage 4 — Writing:** generate the literature review from `SynthesisResult` + registry; validate every citation against `paper_key` / `support_refs`; keep Research Gap language scoped to the retrieved snapshot.
6. Incremental corpus update (Stage 1 leftover) and LangGraph remain optional.
