# Báo cáo tiến độ: Research Assistance Agent

**Ngày:** 4 tháng 9, 2026  
**Phạm vi:** hoàn thành **giai đoạn 1 — Retrieval** (chuẩn bị corpus + pipeline truy vấn). Chưa làm Extraction / Synthesis / Writing.

---

## 1. Mục tiêu đã chốt

Xây dựng agent hỗ trợ literature review: nhận một chủ đề nghiên cứu, tìm paper liên quan trên arXiv, rồi (ở các giai đoạn sau) trích xuất, tổng hợp và viết review có trích dẫn.

MVP gồm bốn giai đoạn. Hiện tại chỉ hoàn thành tầng dữ liệu nền và **retrieve-then-rerank**.

```
Chủ đề
  → query expansion
  → FAISS (SPECTER2 adhoc-query)
  → rerank bằng embedding proximity đã lưu
  → (tuỳ chọn) hybrid RRF với arXiv keyword / citation
  → top-k paper
```

Bối cảnh dùng ngay: Data-Centric AI / dataset pruning (chủ đề khóa luận).

---

## 2. Corpus offline (làm một lần trên Kaggle)

Đã embed và đưa về máy tại `data/specter2_artifacts`.

| Hạng mục | Giá trị |
|---|---|
| Nguồn | arXiv metadata OAI snapshot |
| Filter | `cs.LG`, `cs.AI`, `cs.CL`, `cs.CV`, `stat.ML` · năm ≥ 2021 |
| Số paper | **449,682** (45 shard, stream exhausted) |
| Encoder paper | `allenai/specter2_base` + adapter **proximity** (`allenai/specter2`) |
| Dim | 768 · lưu `float16` · L2-normalize |
| Index | FAISS `IndexFlatIP` (cosine qua inner product trên vector đã chuẩn hóa) |
| Thời gian embed (Kaggle) | ~1751 giây ≈ **257 paper/s** (compute) |

Phân bố năm (từ lần chạy Kaggle): 2021: 48k · 2022: 53k · 2023: 67k · 2024: 87k · 2025: 107k · 2026: 88k.

Cấu trúc artifact:

```
data/specter2_artifacts/
  manifest.json
  embeddings/embeddings_part_XXXXX.npy
  metadata/metadata_part_XXXXX.parquet
  index/papers_flatip.faiss
```

Metadata mỗi paper gồm `arxiv_id`, title, abstract, authors, categories, năm, DOI, version, `content_hash` (để incremental update sau này). Embedding **không** nhét vào DataFrame.

Notebook tạo corpus: `smoke_test.ipynb` (bản tối ưu, resumable, tối đa 2 GPU, FP16, OOM fallback, checkpoint qua `manifest.json`).

Smoke test retrieval trên Kaggle với query *dataset pruning and data subset selection for deep neural networks* trả paper đúng chủ đề (ví dụ `2205.09329` Dataset Pruning, score cosine ~0.83).

---

## 3. Notebook: bỏ qua bước embedding

Sau khi corpus đã xong, notebook được chỉnh để **không embed lại**:

- Cờ `SKIP_EMBEDDING = True`
- Tự tìm `manifest.json` trong working dir hoặc `/kaggle/input/*/`
- Không load SPECTER2 paper encoder (tiết kiệm GPU/VRAM)
- Vẫn validate shard, build/reuse FAISS, chạy smoke test query encoder
- Snapshot arXiv không bắt buộc khi skip
- Đặt `SKIP_EMBEDDING = False` nếu cần chạy lại pipeline embed

---

## 4. Pipeline truy vấn local (giai đoạn 1.2)

Package Python `research_assistant`, đọc corpus local, chạy mỗi khi nhập chủ đề.

### Luồng xử lý

1. **Query expansion**  
   LLM (Groq / Gemini / OpenAI, key trong `.env`) sinh ~4 biến thể. Không có key thì dùng template (synonym / methods / survey). Topic gốc luôn là query đầu tiên.

2. **Broad retrieval**  
   Encode mọi query bằng `allenai/specter2_adhoc_query` (không dùng proximity cho câu ngắn). Mỗi query lấy `broad_k=150` neighbor FAISS. Union + dedup theo `arxiv_id`, giữ điểm FAISS cao nhất, cắt pool 200.

3. **Rerank**  
   Cosine giữa embedding **chủ đề gốc** (adhoc-query) và **vector proximity đã lưu** của candidate. Không embed lại 200 paper — đúng với việc corpus đã encode proximity sẵn.

4. **Hybrid (mặc định bật)**  
   arXiv keyword API → map vào corpus → Reciprocal Rank Fusion với thứ tự rerank. `--citations` thêm rank Semantic Scholar (tắt mặc định vì rate limit).

5. **Top-k = 25**  
   Kèm filter năm / category nếu chỉ định. Trả `PaperHit` (arxiv_id, title, abstract, scores, query nào retrieve được paper).

Metrics log được: số query, unique sau union, pool sau filter, keyword hits trong corpus, overlap top-k broad vs final, thời gian.

### File chính

| File | Vai trò |
|---|---|
| `research_assistant/config.py` | `RetrievalConfig`, đường dẫn artifact, model IDs |
| `research_assistant/retrieval/corpus.py` | Load FAISS + lookup metadata/embedding theo shard |
| `research_assistant/retrieval/encoder.py` | SPECTER2 query encoder |
| `research_assistant/retrieval/expand.py` | LLM / template query expansion |
| `research_assistant/retrieval/pipeline.py` | Orchestrate retrieve-then-rerank |
| `research_assistant/retrieval/hybrid.py` | RRF, arXiv keyword, Semantic Scholar |
| `research_assistant/retrieval/cli.py` | CLI `python -m research_assistant.retrieval` |
| `examples/retrieve_topic.py` | Ví dụ gọi Python API |

### Cách chạy

```bash
pip install -e .
python -m research_assistant.retrieval \
  "dataset pruning and data subset selection for deep neural networks" \
  --top-k 25 \
  --json-out results/pruning.json
```

Hoặc:

```python
from research_assistant import retrieve
from research_assistant.config import RetrievalConfig

result = retrieve("proxy maturity in dataset pruning", RetrievalConfig(top_k=25))
```

Copy `.env.example` → `.env` và điền `GROQ_API_KEY` / `GEMINI_API_KEY` nếu muốn expansion bằng LLM. Lần chạy đầu tải query encoder (~440MB).

---

## 5. Kiểm thử

`pytest tests/` — unit tests (không cần GPU / SPECTER2 weights):

- RRF (kể cả duplicate)
- Parse JSON expansion (kể cả markdown fence)
- Template queries unique, luôn gồm topic gốc
- Chuẩn hóa arXiv ID (URL, version)
- Filter năm / category
- Pipeline wiring với fake corpus/encoder: rerank đúng paper gần topic hơn; hybrid đưa keyword hit vào pool; `n_query_variants=0` chỉ giữ query gốc; `--no-rerank` giữ thứ tự FAISS

Eval retrieval **không nằm trong package production**. Harness + ablation ở notebook `evaluation/retrieval_ablation.ipynb` (qrels `evaluation/qrels.json`). Ablation trên corpus local (10 query, 78 qrels, template expansion, 4 tháng 9 2026) — chi tiết `evaluation/ablation.json`:

| Variant | Recall@10 | Recall@25 | nDCG@10 | MRR | Hit@25 |
|---|---|---|---|---|---|
| Original query + FAISS | 0.121 | 0.174 | 0.162 | 0.285 | 0.600 |
| Expanded queries + FAISS | 0.135 | 0.160 | 0.168 | 0.300 | 0.600 |
| Expanded + original-query rerank | 0.121 | 0.174 | 0.162 | 0.285 | 0.600 |
| Expanded + rerank + keyword RRF | **0.121** | **0.212** | **0.173** | **0.367** | **0.700** |

Hybrid là variant tốt nhất. Template expansion làm tụt Recall@25 của `data_selection` (0.286 → 0.143) rồi rerank kéo lại — đúng giả thuyết kiến trúc retrieve-then-rerank. Leak@10 trên 4 cặp hard query = 0, nhưng một phần vì nhiều paper grade-2 chưa vào top-25 nên cũng không thể “lọt” sang query kia.

Query còn Recall@25 = 0 kể cả full system: `feature_selection`, `knowledge_distillation`, `active_learning`.

Mở notebook (mặc định chỉ đọc `ablation.json`, không load model):

```text
evaluation/retrieval_ablation.ipynb
```

Đặt `RUN_LIVE = True` trong notebook rồi chạy lại cell ablation khi cần đo trên corpus.

---

## 6. Đối chiếu với thiết kế

| Hạng mục thiết kế | Trạng thái |
|---|---|
| Filter category + năm, embed SPECTER2, FAISS local | Xong |
| Resume cùng snapshot qua `manifest.json` + `content_hash` | Xong (resume; incremental giữa hai snapshot thì chưa) |
| Query expansion 3–5 biến thể | Xong (LLM hoặc template) |
| Broad retrieval top 100–200, union/dedup | Xong |
| Rerank SPECTER2 vs chủ đề gốc | Xong (dùng embedding đã lưu) |
| Hybrid arXiv keyword + RRF | Xong (mặc định bật) |
| Semantic Scholar citation | Có, opt-in `--citations` |
| Log overlap broad vs rerank | Xong (`top_k_overlap_broad_vs_final`) |
| Extraction TeX/PDF + Pydantic schema | Chưa |
| Synthesis clustering method | Chưa |
| Writing literature review + validate citation | Chưa |
| Eval Recall@k / NDCG | Notebook `evaluation/retrieval_ablation.ipynb` + `evaluation/ablation.json` |
| LangGraph orchestration | Chưa (pipeline Python thuần cho MVP) |

---

## 7. Việc tiếp theo (gợi ý)

1. Từ bảng ablation: hybrid giúp Recall@25 / MRR / Hit@25; template expansion thì không ổn định. Việc đáng làm trước Extraction:
   - So LLM expansion với template (cùng qrels).
   - Bổ sung qrels cho 3 query đang Recall@25 = 0, hoặc chấp nhận chúng là out-of-scope.
   - Tune `broad_k` / `candidate_pool_size` trên query khóa luận (`dataset_pruning`, `proxy_maturity`).
2. Giai đoạn 2 — Extraction: tải TeX/PDF, schema Pydantic, retry khi JSON lỗi.
3. Incremental update corpus khi snapshot arXiv đổi (dựa trên `arxiv_id` + `content_hash`), không re-embed toàn bộ.
