# Kế Hoạch Thiết Kế: Research Assistant Agent

**Mục tiêu project**: Xây dựng một agent tự động hóa quy trình review literature — nhận một chủ đề nghiên cứu, tự động tìm paper liên quan trên arXiv, trích xuất nội dung, tổng hợp và viết thành một bản literature review có cấu trúc, có trích dẫn rõ ràng.

**Bối cảnh cá nhân**: Tự động hóa lại chính quy trình đã làm thủ công khi review 60-70 paper cho khóa luận tốt nghiệp (Data-Centric AI / proxy maturity trong dataset pruning).

---

## 1. Tổng quan kiến trúc

Pipeline gồm 4 giai đoạn chính, thiết kế dạng agent/module tuần tự (có thể agentic hóa sau khi pipeline cơ bản chạy ổn):

```
[Topic Input]
     │
     ▼
┌─────────────────┐
│ 1. RETRIEVAL     │  → tìm paper liên quan (candidate pool)
└─────────────────┘
     │
     ▼
┌─────────────────┐
│ 2. EXTRACTION    │  → trích xuất nội dung có cấu trúc từ mỗi paper
└─────────────────┘
     │
     ▼
┌─────────────────┐
│ 3. SYNTHESIS     │  → nhóm paper theo hướng tiếp cận, tổng hợp theo nhóm
└─────────────────┘
     │
     ▼
┌─────────────────┐
│ 4. WRITING       │  → viết literature review hoàn chỉnh, có trích dẫn
└─────────────────┘
     │
     ▼
[Literature Review Document]
```

---

## 2. Giai đoạn 1 — Retrieval (Tìm kiếm ngữ nghĩa)

Kiến trúc **retrieve-then-rerank hai tầng**, kết hợp semantic search toàn arXiv (recall rộng) với SPECTER2 rerank (precision cao):

### 2.1 Chuẩn bị dữ liệu nền (làm 1 lần, cập nhật định kỳ)
- Giới hạn phạm vi để dataset gọn: category `cs.LG`, `cs.AI`, `cs.CL`... trong ~5 năm gần nhất (thay vì toàn bộ arXiv)
- Lấy metadata + embedding có sẵn (dataset công khai trên HuggingFace/Kaggle), hoặc tự embed abstract bằng SPECTER2
- Lưu vào vector DB local: **FAISS** (nhẹ, đủ nhanh cho quy mô project cá nhân) hoặc **Qdrant** nếu muốn có thêm filter theo metadata (năm, category)
- Thiết kế cơ chế **incremental update**: chỉ embed/thêm paper mới, không re-embed toàn bộ mỗi lần chạy

### 2.2 Pipeline truy vấn (mỗi khi user nhập chủ đề)
```
1. Query expansion: LLM sinh 3-5 biến thể câu hỏi/từ khóa từ chủ đề gốc
2. Broad retrieval: embed từng query, search trong FAISS index 
   → lấy top 100-200 candidate (union + dedup theo arxiv_id)
3. Rerank: re-embed candidate pool bằng SPECTER2, tính cosine similarity 
   với embedding chủ đề gốc, sort lại
4. (optional) Hybrid signal: cross-check với arXiv keyword search / 
   Semantic Scholar citation count → ưu tiên paper vừa liên quan 
   vừa có uy tín (reciprocal rank fusion)
5. Lấy top-k cuối (VD: 20-30 paper) → chuyển sang giai đoạn Extraction
```

### 2.3 Lưu ý kỹ thuật
- Semantic Scholar API rate limit chặt nếu không có key → xin free API key sớm
- Log lại tỷ lệ recall thô (bước 2) vs sau rerank (bước 3) để có số liệu so sánh cho báo cáo

---

## 3. Giai đoạn 2 — Extraction (Trích xuất nội dung)

### 3.1 Lấy nội dung paper
```
1. Thử tải TeX source: arxiv.org/e-print/<id> (nhanh, sạch, giữ cấu trúc)
2. Parse bằng pylatexenc / TexSoup, merge các file .tex con (\input, \include)
3. Nếu lỗi hoặc không có source (404) → fallback PDF (PyMuPDF)
4. Log tỷ lệ dùng source vs fallback PDF (số liệu hay cho báo cáo)
```
- Chỉ cần bản mới nhất của mỗi paper (không cần xử lý version cũ); lưu số version vào metadata để trích dẫn chính xác

### 3.2 Trích xuất structured info
- Ép LLM trả JSON theo schema cố định, dùng **Pydantic** để validate:
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
- Nếu LLM trả sai format → retry tự động (giới hạn số lần retry)
- Nếu 1 paper lỗi extraction hoàn toàn → skip và log, không để crash cả pipeline

---

## 4. Giai đoạn 3 — Synthesis (Nhóm & tổng hợp)

```
1. Embed phần "method" của mỗi paper đã extract
2. Clustering (K-Means hoặc HDBSCAN) để nhóm paper theo hướng tiếp cận
3. Với mỗi cụm: LLM tổng hợp — các paper trong nhóm giải quyết theo 
   hướng nào, điểm chung, điểm khác biệt
4. So sánh giữa các cụm: chỉ ra gap / hướng chưa được khai thác
```
- Đây là bước áp dụng trực tiếp kỹ năng data mining (clustering thật, không để LLM tự đoán nhóm cảm tính)

---

## 5. Giai đoạn 4 — Writing (Viết literature review)

- Input: dữ liệu structured đã tổng hợp theo cụm (không phải raw text paper) → **tránh hallucination trích dẫn**
- Output: văn bản review có cấu trúc, mỗi nhận định phải trace được về `arxiv_id` cụ thể
- Có phần **"Research Gap"** ở cuối, mô phỏng đúng phần người viết khóa luận phải tự làm tay
- Validate bước cuối: kiểm tra mọi trích dẫn trong bài viết có khớp với candidate pool đã extract hay không (script kiểm tra tự động, không dựa vào "LLM tự nói đúng")

---

## 6. Tech stack tổng hợp

| Thành phần | Công cụ |
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

## 7. Đánh giá (Evaluation)

- **Retrieval**: tự tạo test set 10-15 chủ đề kèm ground-truth paper đã biết trước → đo Recall@k, NDCG. So sánh "chỉ broad retrieval" vs "broad + SPECTER2 rerank" để có con số minh chứng giá trị của kiến trúc 2 tầng.
- **Literature review cuối**: LLM-as-judge chấm coherence/coverage (ghi rõ đây là proxy metric, không hoàn toàn khách quan).
- **So sánh với quy trình thủ công**: thời gian làm tay (60-70 paper) vs thời gian agent chạy — số liệu kể chuyện thuyết phục khi trình bày.

---

## 8. Rủi ro & lưu ý cần theo dõi suốt project

1. **Data engineering**: cần cơ chế cập nhật embedding dataset định kỳ, không re-embed toàn bộ mỗi lần
2. **Extraction lỗi**: TeX parse fail với macro lạ → fallback graceful, không crash pipeline
3. **Hallucination trích dẫn**: rủi ro lớn nhất — mọi câu trong output cuối phải trace được về paper thật
4. **Chi phí & rate limit**: xin Semantic Scholar API key sớm; giới hạn phạm vi dataset embedding (category + khoảng năm) để gọn nhẹ
5. **Scope creep**: chốt MVP là search → extract → synthesize → write; các ý tưởng mở rộng (agent review code, generate slide...) để riêng thành "future work"

---

## 9. Timeline gợi ý

| Tuần | Công việc |
|---|---|
| 1 | Setup retrieval: chuẩn bị embedding dataset, FAISS index, pipeline query expansion + rerank |
| 2 | Extraction: TeX/PDF parser, schema + validate, xử lý lỗi |
| 3 | Synthesis + Writing: clustering, tổng hợp theo cụm, sinh literature review |
| 4 | Evaluation: test set Recall@k/NDCG, so sánh kiến trúc, viết báo cáo/demo |

---

## 10. Điểm nhấn khi trình bày (CV / phỏng vấn)

- Kiến trúc retrieve-then-rerank hai tầng (semantic search nghiêm túc, không chỉ gọi API)
- Áp dụng data mining thật (clustering paper theo hướng tiếp cận) thay vì để LLM tự làm hết
- Có số liệu đánh giá định lượng (Recall@k, NDCG) — không chỉ demo suông
- Câu chuyện cá nhân: tự động hóa chính quy trình đã làm tay cho khóa luận, có số liệu so sánh thời gian/độ chính xác thực tế
