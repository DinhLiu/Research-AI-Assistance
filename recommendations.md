Retrieval is already fairly complete on the **engineering** side, but if the goal is a strong personal project for CV/GitHub and interview depth, I would not jump straight to Extraction. I would add an **evaluation + hardening** pass first.

You already have a 449,682-paper corpus, SPECTER2 proximity embeddings, FAISS, query expansion, hybrid retrieval/RRF, CLI/API, logging, and 9 unit tests. That is a solid foundation. What is missing is not mainly “more features”, but **evidence that Retrieval actually works and an understanding of its failure modes**.

## 1. Highest priority: Retrieval Evaluation before Extraction

The progress report already lists this as unfinished.

I would treat it as a **blocking task before Phase 2**.

Do not only test 10–15 topics like:

```text
query
→ known paper
→ did it retrieve?
```

Build a small benchmark with relevance grades.

For example about **15–20 queries**, split by domain:

```text
Dataset Pruning
Data-Centric AI
Computer Vision
NLP
LLM
Graph ML
Model Compression
Active Learning
Dataset Distillation
Representation Learning
```

For each query, take a union of about 30–50 candidates and label them by hand:

```text
2 = highly relevant
1 = partially relevant
0 = irrelevant
```

Then measure:

* Recall@10 / @25 / @50 / @100
* Precision@10 / @25
* nDCG@10 / @25
* MRR
* HitRate@k

More important is an **ablation**:

| Variant                       | Meaning                      |
| ----------------------------- | ---------------------------- |
| Original query + FAISS        | dense baseline               |
| Expanded queries + FAISS      | value of query expansion     |
| Dense + original-query rerank | value of the current rerank  |
| Keyword only                  | lexical baseline             |
| Dense + keyword RRF           | value of hybrid              |
| Dense + keyword + expansion   | full system                  |

The pipeline already has the components for this experiment.

If the final result looks like:

```text
Dense only            nDCG@10 = 0.71
+ expansion                     0.75
+ RRF                           0.81
```

the project becomes much more convincing.

---

# 2. Add a “hard queries” set

The current smoke test uses dataset pruning and succeeded.

Also deliberately create queries that dense retrieval is likely to confuse.

Examples:

```text
dataset pruning
vs
neural network pruning
```

```text
data selection
vs
feature selection
```

```text
dataset distillation
vs
knowledge distillation
```

```text
sample pruning
vs
weight pruning
```

```text
active learning
vs
coreset selection
```

The goal is not only to show the system finds the right papers, but to check:

> **Can the system tell nearby concepts apart?**

This is the failure mode already seen in an earlier smoke test: SPECTER2 sometimes pulled network-pruning papers into a dataset-pruning query.

I would add a file:

```text
evaluation/
├── queries.json
├── qrels.json
└── hard_queries.json
```

---

# 3. Logging should be richer

You already log:

> query count, unique after union, pool after filters, keyword hits, broad/final overlap, and latency.

Also add:

```text
dense_candidates
keyword_candidates
intersection_dense_keyword

candidate_count_before_dedup
candidate_count_after_dedup

retrieval_time_ms
rerank_time_ms
keyword_api_time_ms
expansion_time_ms
total_time_ms

query_encoder_time_ms
metadata_lookup_time_ms
```

And per paper:

```json
{
  "dense_rank": 3,
  "dense_score": 0.824,
  "keyword_rank": 8,
  "rrf_score": 0.027,
  "matched_queries": [
    "...",
    "..."
  ],
  "final_rank": 2
}
```

This is very useful when debugging:

> “Why is this paper rank #3?”

A good retrieval agent should explain **which path put a paper into the candidate pool**.

---

# 4. Query expansion needs a guardrail

The LLM currently generates about 4 queries and falls back to templates if there is no API key.

That is a good design, but expansion risks **query drift**.

Example:

```text
original:
proxy maturity in dataset pruning

LLM expansion:
neural network compression
efficient deep learning
model pruning
```

If expansion drifts too far, recall can rise while precision drops sharply.

Store:

```text
original_query
expanded_query
similarity(expanded, original)
```

and optionally reject expansions that are too far.

A threshold is not required on day one. Log first, then look at evaluation.

Also ask the LLM to expand by type:

```json
{
  "synonyms": [...],
  "method_terms": [...],
  "broader_terms": [...],
  "narrower_terms": [...]
}
```

instead of 4 query strings with unclear semantics.

---

# 5. Rename the current rerank more accurately

You currently:

> broad retrieval with expanded adhoc queries → candidate union → cosine against the **original topic embedding** to rerank.

That is reasonable.

I would call it:

```text
original-query semantic reranking
```

rather than only:

```text
SPECTER2 rerank
```

because papers are not re-encoded and there is no separate reranker model.

That keeps the README/design accurate.

Later, if evaluation still shows low precision, then benchmark a **cross-encoder reranker**.

Do not add that now.

---

# 6. Tune candidate pool size with numbers

Currently:

```text
150 per expanded query
→ union
→ pool 200
→ top 25
```

These numbers are fairly heuristic.

Evaluation should try:

```text
broad_k:
50
100
150
250

candidate_pool:
100
200
300
500
```

and look at:

```text
Recall@candidate_pool
vs
latency
```

You may find:

```text
pool 100 → recall 0.87
pool 200 → recall 0.96
pool 500 → recall 0.97
```

Then choosing 200 has clear evidence.

---

# 7. Add corpus coverage diagnostics

You already know year and category distributions.

I would add a script that generates:

```text
corpus_report.json
```

including:

```text
paper count
year distribution
category distribution
multi-category overlap
papers with DOI %
papers with journal_ref %
papers with empty authors %
duplicate arxiv_id
duplicate content_hash
average abstract length
p95 abstract length
```

In particular:

```text
duplicate content_hash
```

can catch unusual metadata cases.

---

# 8. Formalize corpus limitation

This should be written plainly in the README.

The corpus currently only includes:

```text
cs.LG
cs.AI
cs.CL
cs.CV
stat.ML
>= 2021
```

So the agent **is not a general scientific literature search engine**.

Describe it as:

> Current MVP targets recent AI/ML literature indexed on arXiv.

And later literature reviews must not write:

> “No prior work has explored X.”

but:

> “Within the retrieved arXiv corpus from 2021 onward, we found limited work on X.”

That will matter a great deal when you reach the Research Gap section.

---

# 9. Real integration tests, not only fake unit tests

The current 9 tests are very good for pipeline logic.

But they:

> do not need GPU / SPECTER2 weights.

I would keep the unit tests and add:

```text
tests/
├── unit/
└── integration/
```

Integration tests can run only when:

```bash
RUN_INTEGRATION=1 pytest tests/integration
```

Test:

```text
load real FAISS
load real query encoder

query:
dataset pruning...

assert:
2205.09329 ∈ top 20
```

3–5 known papers are enough.

This catches bugs a fake encoder will miss:

```text
wrong adapter
wrong normalization
wrong FAISS mapping
wrong model version
artifact mismatch
```

---

# 10. Stronger reproducibility metadata

The manifest is already very good.

I would add:

```json
{
  "transformers_version": "...",
  "adapters_version": "...",
  "torch_version": "...",
  "faiss_version": "...",
  "python_version": "...",

  "base_model_revision": "...",
  "paper_adapter_revision": "...",

  "created_at": "...",
  "git_commit": "..."
}
```

A config fingerprint already checks corpus configuration, but package/model revisions let you actually reproduce results months later.

---

# 11. Treat Semantic Scholar citations as a ranking prior, not as relevance

You currently have:

> Semantic Scholar citation rank opt-in.

I would be careful here.

Citation count has a large bias:

```text
old papers > new papers
popular topics > niche topics
```

For example a 2026 paper that is extremely relevant may have citation ≈0.

If you use a citation signal, use it as a:

```text
small secondary prior
```

do not let it dominate relevance.

And benchmark:

```text
without citations
vs
with citations
```

before turning it on by default.

Keeping `--citations` opt-in is the right call.

---

# 12. LangGraph is not needed yet

The report lists LangGraph as unfinished.

I would **leave that as-is**.

Retrieval today is a deterministic pipeline:

```text
expand
retrieve
rerank
fuse
```

There is no good reason to turn it into a graph agent.

LangGraph starts to pay off at:

```text
Extraction failed?
     ↓
retry TeX
     ↓
fallback PDF

Potential research gap
     ↓
search again
     ↓
counter-evidence?
     ↓
revise gap
```

That is Phase 2–4.

Do not add LangGraph just so the CV can say “LangGraph”.

---

# Suggested order before moving to Phase 2

I would rewrite §7 of the report as:

**Milestone 1A — Retrieval Quality**

1. Create a 15–20 query benchmark + relevance labels.
2. Run Dense / Expansion / Keyword / Hybrid ablation.
3. Compute Recall, Precision, MRR, nDCG.
4. Add hard-query failure analysis.
5. Tune `broad_k`, candidate pool, and RRF parameters.

**Milestone 1B — Retrieval Engineering Hardening**
6. Add integration tests with the real corpus/model.
7. Expand logging + ranking provenance.
8. Add corpus diagnostics.
9. Pin model/package revisions and reproducibility metadata.

**Milestone 2 — Extraction**
10. Only then start TeX/PDF → structured claims → evidence provenance.

If I had to pick **one next task**, I would not do Extraction yet — I would do a **Retrieval Evaluation notebook/script**. Retrieval engineering is already good enough; the biggest gap is quantitative evidence that the architecture you built actually beats the baseline. After that ablation table exists, Phase 1 can be treated as **genuinely complete**, then move to Extraction.
