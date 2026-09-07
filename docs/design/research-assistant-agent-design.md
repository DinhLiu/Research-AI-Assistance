# Design plan: Research Assistant Agent

**Project goal**: Build an agent that automates a literature-review workflow — take a research topic, find related papers on arXiv, extract content, synthesize, and write a structured literature review with explicit citations.

**Personal context**: Automate the same process previously done by hand when reviewing 60–70 papers for a graduation thesis (Data-Centric AI / proxy maturity in dataset pruning).

---

## 1. Architecture overview

The pipeline has 4 main stages, designed as sequential agents/modules (can be made more agentic after the basic pipeline is stable):

```
[Topic Input]
     │
     ▼
┌─────────────────┐
│ 1. RETRIEVAL     │  → find related papers (candidate pool)
└─────────────────┘
     │
     ▼
┌─────────────────┐
│ 2. EXTRACTION    │  → extract structured content from each paper
└─────────────────┘
     │
     ▼
┌─────────────────┐
│ 3. SYNTHESIS     │  → group papers by approach, summarize per group
└─────────────────┘
     │
     ▼
┌─────────────────┐
│ 4. WRITING       │  → write a complete, cited literature review
└─────────────────┘
     │
     ▼
[Literature Review Document]
```

---

## 2. Stage 1 — Retrieval (semantic search)

A **two-stage retrieve-then-rerank** architecture: broad semantic search over arXiv (high recall) plus SPECTER2 rerank (high precision):

### 2.1 Offline data prep (once, then periodic updates)
- Limit scope so the dataset stays small: categories `cs.LG`, `cs.AI`, `cs.CL`... from roughly the last 5 years (not all of arXiv)
- Use existing metadata + embeddings (public HuggingFace/Kaggle datasets), or embed abstracts with SPECTER2
- Store in a local vector DB: **FAISS** (light, fast enough for a personal project) or **Qdrant** if metadata filters (year, category) are needed
- Design **incremental update**: only embed/add new papers, do not re-embed the whole corpus each run

### 2.2 Query pipeline (each time the user enters a topic)
```
1. Query expansion: LLM generates 3–5 query/keyword variants from the original topic
2. Broad retrieval: embed each query, search the FAISS index
   → take top 100–200 candidates (union + dedup by arxiv_id)
3. Rerank: re-embed the candidate pool with SPECTER2, cosine similarity
   against the original topic embedding, re-sort
4. (optional) Hybrid signal: cross-check with arXiv keyword search /
   Semantic Scholar citation count → prefer papers that are both relevant
   and reputable (reciprocal rank fusion)
5. Take final top-k (e.g. 20–30 papers) → hand off to Extraction
```

### 2.3 Technical notes
- Semantic Scholar API rate limits are tight without a key → request a free API key early
- Log raw recall (step 2) vs after rerank (step 3) so the report has comparison numbers

---

## 3. Stage 2 — Extraction (content extraction)

### 3.1 Fetch paper content
```
1. Try TeX source: arxiv.org/e-print/<id> (fast, clean, keeps structure)
2. Parse with pylatexenc / TexSoup, merge child .tex files (\input, \include)
3. On error or missing source (404) → fall back to PDF (PyMuPDF)
4. Log source vs PDF fallback rate (useful report numbers)
```
- Only the latest version of each paper is needed (old versions can be skipped); store the version number in metadata for accurate citations

### 3.2 Extract structured info
- Force the LLM to return JSON against a fixed schema, validated with **Pydantic**:
```json
{
  "arxiv_id": "...",
  "version": "v2",
  "problem": "...",
  "method": "...",
  "dataset": "...",
  "result": "...",
  "limitation": "..."
}
```
- If the LLM returns the wrong format → automatic retry (capped)
- If one paper fails extraction entirely → skip and log; do not crash the whole pipeline

---

## 4. Stage 3 — Synthesis (group & summarize)

```
1. Embed the "method" field of each extracted paper
2. Cluster (K-Means or HDBSCAN) to group papers by approach
3. Per cluster: LLM summary — how papers in the group attack the problem,
   shared points, differences
4. Compare clusters: surface gaps / underexplored directions
```
- This step applies real data mining (actual clustering, not letting the LLM invent groups)

---

## 5. Stage 4 — Writing (write the literature review)

- Input: structured data already summarized by cluster (not raw paper text) → **avoid citation hallucination**
- Output: structured review text; every claim must trace to a specific `arxiv_id`
- End with a **"Research Gap"** section, matching the part a thesis author would write by hand
- Final validation: check that every citation in the write-up matches the extracted candidate pool (automatic script, not "the LLM says it is correct")

---

## 6. Tech stack

| Component | Tool |
|---|---|
| Orchestration | LangGraph |
| Vector DB | FAISS / Qdrant (local) |
| Retrieval embedding | SPECTER2 (`allenai/specter2_base`) |
| PDF/TeX extraction | pylatexenc, TexSoup, PyMuPDF |
| Structured output | Pydantic |
| Clustering | scikit-learn (K-Means/HDBSCAN) |
| LLM | Groq / Gemini free tier (extraction), GPT-4o-mini (writing) |
| Metadata source | arXiv API, Semantic Scholar API |

---

## 7. Evaluation

- **Retrieval**: build a 10–15 topic test set with known ground-truth papers → measure Recall@k, NDCG. Compare "broad retrieval only" vs "broad + SPECTER2 rerank" to quantify the two-stage architecture.
- **Final literature review**: LLM-as-judge for coherence/coverage (explicitly a proxy metric, not fully objective).
- **Vs the manual process**: hand time (60–70 papers) vs agent runtime — a convincing story for presentations.

---

## 8. Risks & notes to watch throughout the project

1. **Data engineering**: periodic embedding-dataset updates without re-embedding everything each time
2. **Extraction failures**: TeX parse fails on unusual macros → graceful fallback, do not crash the pipeline
3. **Citation hallucination**: the largest risk — every sentence in the final output must trace to a real paper
4. **Cost & rate limits**: request a Semantic Scholar API key early; keep the embedding dataset small (category + year window)
5. **Scope creep**: lock the MVP to search → extract → synthesize → write; extra ideas (code-review agent, slide generation...) stay in "future work"

---

## 9. Suggested timeline

| Week | Work |
|---|---|
| 1 | Retrieval setup: embedding dataset, FAISS index, query-expansion + rerank pipeline |
| 2 | Extraction: TeX/PDF parser, schema + validate, error handling |
| 3 | Synthesis + Writing: clustering, per-cluster summaries, generate literature review |
| 4 | Evaluation: Recall@k/NDCG test set, architecture comparison, write report/demo |

---

## 10. Presentation highlights (CV / interview)

- Two-stage retrieve-then-rerank architecture (serious semantic search, not just an API call)
- Real data mining (cluster papers by approach) instead of leaving everything to an LLM
- Quantitative evaluation numbers (Recall@k, NDCG) — not only a demo
- Personal story: automate the same process previously done by hand for a thesis, with real time/accuracy comparison numbers
