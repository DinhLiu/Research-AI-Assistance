# Progress report: Research Assistance Agent

**Date:** 4 September 2026  
**Scope:** **Stage 1 — Retrieval** is complete (corpus preparation + query pipeline). Extraction / Synthesis / Writing are not started.

---

## 1. Locked-in goal

Build a literature-review assistant: take a research topic, find related papers on arXiv, then (in later stages) extract, synthesize, and write a cited review.

The MVP has four stages. So far only the offline data layer and **retrieve-then-rerank** are done.

```
Topic
  → query expansion
  → FAISS (SPECTER2 adhoc-query)
  → rerank with stored proximity embeddings
  → (optional) hybrid RRF with arXiv keyword / citation
  → top-k papers
```

Immediate use case: Data-Centric AI / dataset pruning (thesis topic).

---

## 2. Offline corpus (built once on Kaggle)

Embedded and copied locally to `data/specter2_artifacts`.

| Item | Value |
|---|---|
| Source | arXiv metadata OAI snapshot |
| Filter | `cs.LG`, `cs.AI`, `cs.CL`, `cs.CV`, `stat.ML` · year ≥ 2021 |
| Papers | **449,682** (45 shards, stream exhausted) |
| Paper encoder | `allenai/specter2_base` + **proximity** adapter (`allenai/specter2`) |
| Dim | 768 · stored as `float16` · L2-normalized |
| Index | FAISS `IndexFlatIP` (cosine via inner product on normalized vectors) |
| Embed time (Kaggle) | ~1751 seconds ≈ **257 papers/s** (compute) |

Year distribution (from the Kaggle run): 2021: 48k · 2022: 53k · 2023: 67k · 2024: 87k · 2025: 107k · 2026: 88k.

Artifact layout:

```
data/specter2_artifacts/
  manifest.json
  embeddings/embeddings_part_XXXXX.npy
  metadata/metadata_part_XXXXX.parquet
  index/papers_flatip.faiss
```

Per-paper metadata includes `arxiv_id`, title, abstract, authors, categories, year, DOI, version, and `content_hash` (for later incremental updates). Embeddings are **not** stored in the DataFrame.

Corpus notebook: `notebooks/embedding.ipynb` (optimized, resumable, up to 2 GPUs, FP16, OOM fallback, checkpointing via `manifest.json`).

Kaggle retrieval smoke test with query *dataset pruning and data subset selection for deep neural networks* returned on-topic papers (e.g. `2205.09329` Dataset Pruning, cosine ~0.83).

---

## 3. Notebook: skip the embedding step

After the corpus was finished, the notebook was changed so it **does not re-embed**:

- Flag `SKIP_EMBEDDING = True`
- Auto-finds `manifest.json` in the working dir or `/kaggle/input/*/`
- Does not load the SPECTER2 paper encoder (saves GPU/VRAM)
- Still validates shards, builds/reuses FAISS, and runs the query-encoder smoke test
- The arXiv snapshot is not required when skipping
- Set `SKIP_EMBEDDING = False` to rerun the embed pipeline

---

## 4. Local query pipeline (stage 1.2)

Python package `research_assistant` reads the local corpus and runs whenever a topic is entered.

### Processing flow

1. **Query expansion**  
   An LLM (Groq / Gemini / OpenAI, keys in `.env`) generates ~4 variants. Without a key, templates are used (synonym / methods / survey). The original topic is always the first query.

2. **Broad retrieval**  
   Encode every query with `allenai/specter2_adhoc_query` (do not use proximity for short text). Each query pulls `broad_k=150` FAISS neighbors. Union + dedup by `arxiv_id`, keep the highest FAISS score, cap the pool at 200.

3. **Rerank**  
   Cosine between the **original topic** embedding (adhoc-query) and each candidate's **stored proximity vector**. The 200 papers are not re-embedded — the corpus already has proximity encodings.

4. **Hybrid (on by default)**  
   arXiv keyword API → map into the corpus → Reciprocal Rank Fusion with the rerank order. `--citations` adds Semantic Scholar ranks (off by default because of rate limits).

5. **Top-k = 25**  
   Optional year / category filters. Returns `PaperHit` (arxiv_id, title, abstract, scores, which queries retrieved the paper).

Logged metrics: query count, unique after union, pool after filters, keyword hits in corpus, top-k overlap of broad vs final, latency.

### Main files

| File | Role |
|---|---|
| `research_assistant/config.py` | `RetrievalConfig`, artifact paths, model IDs |
| `research_assistant/retrieval/corpus.py` | Load FAISS + shard metadata/embedding lookup |
| `research_assistant/retrieval/encoder.py` | SPECTER2 query encoder |
| `research_assistant/retrieval/expand.py` | LLM / template query expansion |
| `research_assistant/retrieval/pipeline.py` | Orchestrate retrieve-then-rerank |
| `research_assistant/retrieval/hybrid.py` | RRF, arXiv keyword, Semantic Scholar |
| `research_assistant/retrieval/cli.py` | CLI `python -m research_assistant.retrieval` |
| `examples/retrieve_topic.py` | Example Python API call |

### How to run

```bash
pip install -e .
python -m research_assistant.retrieval \
  "dataset pruning and data subset selection for deep neural networks" \
  --top-k 25 \
  --json-out results/pruning.json
```

Or:

```python
from research_assistant import retrieve
from research_assistant.config import RetrievalConfig

result = retrieve("proxy maturity in dataset pruning", RetrievalConfig(top_k=25))
```

Copy `.env.example` → `.env` and fill in `GROQ_API_KEY` / `GEMINI_API_KEY` for LLM expansion. The first run downloads the query encoder (~440MB).

---

## 5. Tests

`pytest tests/` — unit tests (no GPU / SPECTER2 weights required):

- RRF (including duplicates)
- Parse expansion JSON (including markdown fences)
- Template queries are unique and always include the original topic
- Normalize arXiv IDs (URL, version)
- Year / category filters
- Pipeline wiring with a fake corpus/encoder: rerank prefers the paper closer to the topic; hybrid adds a keyword hit to the pool; `n_query_variants=0` keeps only the original query; `--no-rerank` keeps FAISS order

Retrieval eval **does not live in the production package**. The harness and ablation live in `evaluation/retrieval_ablation.ipynb` (qrels in `evaluation/qrels.json`). Ablation on the local corpus (10 queries, 78 qrels, template expansion, 4 September 2026) — details in `evaluation/ablation.json`:

| Variant | Recall@10 | Recall@25 | nDCG@10 | MRR | Hit@25 |
|---|---|---|---|---|---|
| Original query + FAISS | 0.121 | 0.174 | 0.162 | 0.285 | 0.600 |
| Expanded queries + FAISS | 0.135 | 0.160 | 0.168 | 0.300 | 0.600 |
| Expanded + original-query rerank | 0.121 | 0.174 | 0.162 | 0.285 | 0.600 |
| Expanded + rerank + keyword RRF | **0.121** | **0.212** | **0.173** | **0.367** | **0.700** |

Hybrid is the best variant. Template expansion dropped Recall@25 for `data_selection` (0.286 → 0.143); rerank recovered it — consistent with the retrieve-then-rerank design. Leak@10 on the 4 hard query pairs is 0, partly because many grade-2 papers never reach top-25 and therefore cannot leak into the other query.

Queries still at Recall@25 = 0 even with the full system: `feature_selection`, `knowledge_distillation`, `active_learning`.

Open the notebook (by default it only reads `ablation.json` and does not load models):

```text
evaluation/retrieval_ablation.ipynb
```

Set `RUN_LIVE = True` in the notebook and rerun the ablation cell to measure on the corpus.

---

## 6. Design checklist

| Design item | Status |
|---|---|
| Category + year filter, SPECTER2 embed, local FAISS | Done |
| Resume on the same snapshot via `manifest.json` + `content_hash` | Done (resume; incremental across two snapshots is not) |
| Query expansion 3–5 variants | Done (LLM or template) |
| Broad retrieval top 100–200, union/dedup | Done |
| SPECTER2 rerank vs original topic | Done (uses stored embeddings) |
| Hybrid arXiv keyword + RRF | Done (on by default) |
| Semantic Scholar citation | Present, opt-in `--citations` |
| Log overlap of broad vs rerank | Done (`top_k_overlap_broad_vs_final`) |
| Extraction TeX/PDF + Pydantic schema | Not started |
| Synthesis method clustering | Not started |
| Writing literature review + citation validation | Not started |
| Eval Recall@k / NDCG | Notebook `evaluation/retrieval_ablation.ipynb` + `evaluation/ablation.json` |
| LangGraph orchestration | Not started (plain Python pipeline for MVP) |

---

## 7. Suggested next steps

1. From the ablation table: hybrid helps Recall@25 / MRR / Hit@25; template expansion is unstable. Before Extraction:
   - Compare LLM expansion vs templates (same qrels).
   - Add qrels for the 3 queries still at Recall@25 = 0, or treat them as out of scope.
   - Tune `broad_k` / `candidate_pool_size` on the thesis queries (`dataset_pruning`, `proxy_maturity`).
2. Stage 2 — Extraction: download TeX/PDF, Pydantic schema, retry on invalid JSON.
3. Incremental corpus update when the arXiv snapshot changes (based on `arxiv_id` + `content_hash`), without re-embedding everything.
