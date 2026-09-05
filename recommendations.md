Phần Retrieval hiện tại đã khá hoàn chỉnh về mặt **engineering**, nhưng nếu mục tiêu là biến project này thành một project cá nhân có chất lượng tốt để đưa CV/GitHub và có thể nói sâu trong phỏng vấn, tôi chưa chuyển hẳn sang Extraction ngay. Tôi sẽ bổ sung một vòng **evaluation + hardening** trước.

Bạn hiện đã có corpus 449,682 paper, SPECTER2 proximity embeddings, FAISS, query expansion, hybrid retrieval/RRF, CLI/API, logging và 9 unit tests.    Đây là nền tảng tốt. Phần còn thiếu chủ yếu không phải “thêm feature”, mà là **chứng minh Retrieval thực sự tốt và hiểu failure mode của nó**.

## 1. Việc quan trọng nhất: làm Retrieval Evaluation trước Extraction

Trong báo cáo bạn cũng đã ghi đây là phần chưa làm. 

Tôi sẽ xem đây là **blocking task trước Phase 2**.

Không nên chỉ test 10–15 topic kiểu:

```text
query
→ known paper
→ có retrieve ra không
```

Mà nên tạo benchmark nhỏ có relevance grading.

Ví dụ khoảng **15–20 queries**, chia domain:

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

Mỗi query lấy union khoảng 30–50 candidate rồi manually label:

```text
2 = highly relevant
1 = partially relevant
0 = irrelevant
```

Sau đó đo:

* Recall@10 / @25 / @50 / @100
* Precision@10 / @25
* nDCG@10 / @25
* MRR
* HitRate@k

Quan trọng hơn là làm **ablation**:

| Variant                       | Ý nghĩa                 |
| ----------------------------- | ----------------------- |
| Original query + FAISS        | baseline dense          |
| Expanded queries + FAISS      | giá trị query expansion |
| Dense + original-query rerank | giá trị rerank hiện tại |
| Keyword only                  | lexical baseline        |
| Dense + keyword RRF           | giá trị hybrid          |
| Dense + keyword + expansion   | full system             |

Hiện pipeline có đủ component để làm thí nghiệm này. 

Nếu kết quả cuối là:

```text
Dense only            nDCG@10 = 0.71
+ expansion                     0.75
+ RRF                           0.81
```

thì project trở nên thuyết phục hơn rất nhiều.

---

# 2. Thêm một bộ “hard queries”

Smoke test hiện tại dùng dataset pruning và đã thành công. 

Nhưng nên cố tình tạo những query mà dense retrieval dễ nhầm.

Ví dụ:

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

Mục tiêu không phải chỉ chứng minh system tìm được cái đúng mà còn xem:

> **System có phân biệt được các concept gần nhau hay không?**

Đây chính là failure mode đã thấy trong smoke test trước: SPECTER2 đôi lúc kéo network-pruning papers vào query dataset pruning.

Tôi sẽ thêm một file:

```text
evaluation/
├── queries.json
├── qrels.json
└── hard_queries.json
```

---

# 3. Log hiện tại nên phong phú hơn

Bạn đã log:

> số query, unique union, pool sau filter, keyword hits, overlap broad/final và latency. 

Nên bổ sung:

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

Và với từng paper:

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

Điều này rất hữu ích khi debug:

> “Tại sao paper này rank #3?”

Agent retrieval tốt nên có thể giải thích **paper lọt vào candidate pool bằng đường nào**.

---

# 4. Query expansion cần guardrail

Hiện LLM sinh khoảng 4 query và fallback template nếu không có API key. 

Đây là một điểm tốt, nhưng expansion có rủi ro **query drift**.

Ví dụ:

```text
original:
proxy maturity in dataset pruning

LLM expansion:
neural network compression
efficient deep learning
model pruning
```

Nếu expansion trôi quá xa, recall có thể tăng nhưng precision giảm mạnh.

Nên lưu:

```text
original_query
expanded_query
similarity(expanded, original)
```

và có thể reject expansion quá xa.

Không nhất thiết phải dùng một threshold ngay từ đầu. Trước hết hãy log rồi xem evaluation.

Ngoài ra nên yêu cầu LLM expansion theo loại:

```json
{
  "synonyms": [...],
  "method_terms": [...],
  "broader_terms": [...],
  "narrower_terms": [...]
}
```

thay vì 4 câu query không có semantics rõ ràng.

---

# 5. Rerank hiện tại nên đổi tên cho chính xác

Bạn đang:

> broad retrieval bằng expanded adhoc queries → candidate union → cosine với **original topic embedding** để rerank. 

Cách này hợp lý.

Nhưng tôi sẽ gọi nó:

```text
original-query semantic reranking
```

chứ không chỉ:

```text
SPECTER2 rerank
```

vì paper không được encode lại và cũng không có reranker model riêng.

Điều này giúp README/design chính xác hơn.

Về sau, nếu evaluation cho thấy precision vẫn thấp, mới benchmark thêm một **cross-encoder reranker**.

Không nên thêm ngay bây giờ.

---

# 6. Candidate pool size cần được tune bằng số liệu

Hiện:

```text
150 mỗi expanded query
→ union
→ pool 200
→ top 25
```



Các số này hiện tương đối heuristic.

Evaluation nên thử:

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

và xem:

```text
Recall@candidate_pool
vs
latency
```

Có thể bạn sẽ thấy:

```text
pool 100 → recall 0.87
pool 200 → recall 0.96
pool 500 → recall 0.97
```

Khi đó chọn 200 có bằng chứng rõ ràng.

---

# 7. Thêm corpus coverage diagnostics

Hiện bạn biết distribution theo năm và category. 

Tôi sẽ bổ sung một script generate:

```text
corpus_report.json
```

gồm:

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

Đặc biệt:

```text
duplicate content_hash
```

có thể phát hiện các trường hợp metadata bất thường.

---

# 8. Formalize corpus limitation

Đây là điều nên ghi thẳng trong README.

Corpus hiện chỉ gồm:

```text
cs.LG
cs.AI
cs.CL
cs.CV
stat.ML
>= 2021
```



Vì vậy agent **không phải general scientific literature search engine**.

Nên mô tả:

> Current MVP targets recent AI/ML literature indexed on arXiv.

Và Literature Review sau này không được viết:

> “No prior work has explored X.”

mà phải là:

> “Within the retrieved arXiv corpus from 2021 onward, we found limited work on X.”

Điều này sẽ cực kỳ quan trọng khi tới phần Research Gap.

---

# 9. Integration test thật, không chỉ unit test fake

9 test hiện tại rất tốt cho logic pipeline. 

Nhưng chúng:

> không cần GPU / SPECTER2 weights.

Tôi sẽ giữ unit tests và thêm:

```text
tests/
├── unit/
└── integration/
```

Integration test có thể chỉ chạy khi:

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

Thêm 3–5 known paper là đủ.

Điều này bắt được các lỗi mà fake encoder không bắt được:

```text
adapter sai
normalization sai
FAISS mapping sai
model version sai
artifact mismatch
```

---

# 10. Reproducibility metadata nên mạnh hơn

Manifest hiện đã rất tốt.

Tôi sẽ thêm:

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

Hiện config fingerprint giúp kiểm tra cấu hình corpus, nhưng package/model revision giúp bạn thực sự tái tạo kết quả sau vài tháng.

---

# 11. Semantic Scholar citations nên được xem là ranking prior, không phải relevance

Bạn hiện có:

> Semantic Scholar citation rank opt-in. 

Tôi sẽ rất cẩn thận ở đây.

Citation count có bias lớn:

```text
old papers > new papers
popular topics > niche topics
```

Ví dụ paper 2026 mới nhưng cực kỳ relevant có citation ≈0.

Nên nếu dùng citation signal, dùng dạng:

```text
small secondary prior
```

không để nó dominate relevance.

Và benchmark:

```text
without citations
vs
with citations
```

trước khi bật mặc định.

Việc bạn đang để `--citations` opt-in là quyết định đúng.

---

# 12. Chưa cần LangGraph

Báo cáo ghi LangGraph chưa làm. 

Tôi sẽ **giữ nguyên trạng thái này**.

Retrieval hiện tại là deterministic pipeline:

```text
expand
retrieve
rerank
fuse
```

Không có lý do tốt để biến thành graph agent.

LangGraph chỉ bắt đầu có giá trị ở:

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

Tức Phase 2–4.

Không nên thêm LangGraph chỉ để CV có chữ “LangGraph”.

---

# Thứ tự tôi đề xuất trước khi chuyển Phase 2

Tôi sẽ sửa mục §7 trong báo cáo thành:

**Milestone 1A — Retrieval Quality**

1. Tạo 15–20 query benchmark + relevance labels.
2. Chạy ablation Dense / Expansion / Keyword / Hybrid.
3. Tính Recall, Precision, MRR, nDCG.
4. Thêm hard-query failure analysis.
5. Tune `broad_k`, candidate pool và RRF parameters.

**Milestone 1B — Retrieval Engineering Hardening**
6. Thêm integration tests với real corpus/model.
7. Mở rộng logging + provenance của ranking.
8. Thêm corpus diagnostics.
9. Pin model/package revisions và reproducibility metadata.

**Milestone 2 — Extraction**
10. Sau đó mới bắt đầu TeX/PDF → structured claims → evidence provenance.

Nếu phải chọn **một việc duy nhất tiếp theo**, tôi sẽ không làm Extraction ngay mà làm **Retrieval Evaluation notebook/script**. Engineering của giai đoạn Retrieval hiện đã đủ tốt; phần còn thiếu nhất là bằng chứng định lượng rằng kiến trúc bạn vừa xây thực sự tốt hơn baseline. Sau khi có bảng ablation đó, có thể xem **Phase 1 hoàn thành đúng nghĩa** rồi chuyển sang Extraction.
