# Stage 4 — Writing: design và kế hoạch triển khai

Ngày: 2026-09-07. Trạng thái: đã triển khai baseline; xem [hướng dẫn và giới hạn thực tế](../guides/stage-4-writing.md). Nội dung bên dưới giữ lại quyết định thiết kế ban đầu.

## 1. Quyết định chính

Stage 4 là pipeline viết có giới hạn: **outline bằng code → evidence pack → LLM viết JSON → validation bằng code → render Markdown**. Không cần vòng lặp nhiều agent hay LLM tự quyết định gọi thêm tool.

- Mặc định một request viết toàn bài khi đủ context/output budget; tối đa một request repair gộp.
- Bài dài: chia trước thành tối đa ba writing batch, chạy tuần tự. Không tạo một call cho mỗi paper/cluster.
- Tất cả HTTP attempt, kể cả retry, phải đi qua một governor chung. Giới hạn logical call riêng không đủ.
- Giữ chế độ offline tạo bản tổng hợp theo template và cache để chạy lại không tốn API.
- Mục tiêu là giảm call mà vẫn giữ evidence coverage. Các giá trị bên dưới là baseline để đo, chưa phải cấu hình tối ưu đã benchmark.

## 2. Những gì repo hiện tại quyết định thiết kế

`research_assistant/synthesis/types.py` đã có `SynthesisResult`, `paper_manifest`, `assignments`, `evidence_registry`, summaries, comparisons, gap candidates và coverage. Stage 4 dùng trực tiếp snapshot này; không mở lại PDF/TeX, không retrieve thêm, không tự đổi membership.

Các vấn đề phải xử lý:

1. `llm/client.py::_post_with_backoff()` có tối đa bốn transport attempt. Stage 3 có hai logical call nên có thể phát sinh tám HTTP request khi lỗi liên tục.
2. Stage 2 đặt khoảng nghỉ bên ngoài `complete()`: retry bên trong không được bộ đếm và khoảng nghỉ đó kiểm soát. Counter này là cục bộ từng run.
3. Client trả `str`, chưa xuất usage/finish reason và chưa truyền giới hạn output token: khó kiểm soát TPM, chi phí và output bị cắt.
4. Stage 3 mặc định `summarize=False`. Stage 4 phải viết được từ cards + registry ngay cả khi không có narration; không tự bật Stage 3 narration làm phát sinh thêm call.
5. `structurally_validated` chỉ chứng minh reference integrity, không chứng minh câu văn đúng về mặt ngữ nghĩa. Coverage `unknown` không phải research gap đã được xác nhận.
6. `PaperCard` chưa có authors/year đầy đủ. MVP render title + arXiv ID/version + link; không yêu cầu LLM bịa bibliography.

Thiết kế dựa trên working tree hiện tại, bao gồm các sửa đổi Stage 3 đang có. File plan này không sửa các thay đổi đó.

## 3. Luồng xử lý

```text
SynthesisResult
  → input/schema/reference validation
  → deterministic outline + claim/evidence selection
  → token-aware batch plan + coverage manifest
  → validated cache / offline templates
  → bounded writing queue → shared LLM governor → single HTTP attempt
  → JSON + claim validation
  → optional one aggregated repair through the same governor
  → deterministic citations + Markdown + provenance JSON
```

### Input gate

- Kiểm tra schema được hỗ trợ, digest không rỗng, uniqueness của paper keys/unit IDs/cluster IDs, ownership của refs và membership nhất quán; không dùng `lookup()` trước khi kiểm tra duplicate IDs.
- Revalidate summaries/comparisons thay vì tin trường status từ JSON. Chỉ đưa claim hợp lệ vào pack.
- Không thể tính lại `corpus_digest` của toàn bộ ExtractionResult chỉ từ SynthesisResult vì snapshot không chứa mọi record ban đầu. Giữ digest như upstream provenance; tính `writing_input_digest` từ toàn bộ nội dung Stage 4 thực sự dùng. Nếu có ExtractionResult đi kèm, mới kiểm tra lại corpus digest đầy đủ.
- Schema không hỗ trợ hoặc reference graph hỏng: lỗi input, zero API call. Snapshot rỗng hợp lệ: kết quả `empty`, zero API call.

### Outline và lựa chọn evidence

Code tạo các phần: phạm vi tài liệu; nhóm phương pháp; so sánh định tính khi đủ support; hạn chế và hướng cần kiểm chứng; kết luận; tài liệu tham khảo.

Giữ nguyên cluster, order ổn định theo ID; singleton có thể chung một phần trình bày nhưng không gộp thành cluster khoa học mới. Unassigned được ghi rõ trong coverage/phụ lục.

Ưu tiên evidence method cho mỗi paper, sau đó contribution/limitation và result phù hợp. Dùng summary đã validate nếu có, luôn gửi kèm evidence text mà summary dẫn tới. Không đưa ID trơ vào prompt; không cắt giữa claim/quote khiến mất qualifier. Deduplicate unit ID và text, giữ origin mapping.

Evidence pack gồm outline, allowed claim/evidence IDs, subjects, nguyên văn evidence được chọn và snapshot caveats. Tất cả paper text là dữ liệu không tin cậy; không được thay đổi policy hoặc điều khiển tool.

### Packing

- Dự toán cả system prompt, JSON schema, evidence, output JSON, citation metadata và safety margin. Dùng tokenizer phù hợp khi có; estimator bảo thủ phải ghi rõ là estimate.
- Một batch nếu vừa cả context lẫn output budget. Nếu không, chia ở ranh giới section/evidence thành tối đa ba batch, mỗi batch có global outline ngắn để giữ nhất quán.
- Phần phạm vi, số lượng, bibliography và các caveat dùng template; không cần call viết intro/conclusion riêng. Không chạy thêm call polish sau khi ghép.
- Nếu ba batch vẫn không đủ: xuất coverage thiếu và template/phụ lục cho phần còn lại, hoặc kết quả `budget_insufficient` khi người dùng yêu cầu full prose coverage. Không âm thầm tăng call hay chỉ viết các cluster đầu.
- `--dry-run` cho biết batch count, token estimate, coverage, quota config và call ceiling trước khi chạy, zero network.

## 4. Data contract đầu ra

Thêm các Pydantic models trong `writing/types.py`, LLM input dùng `extra="forbid"`:

- `ReviewPlan`: section IDs/order, target length, language, selected evidence, omitted evidence với reason, batch assignments.
- `ReviewClaim`: claim ID, text, kind, subject paper keys, support refs, source synthesis claim IDs nếu có, scope, validation status.
- `ReviewSection`: section ID, ordered claim IDs, generation status. Mỗi factual sentence là một claim; tránh đoạn dài chứa nhiều nhận định dùng chung citation.
- `WritingResult`: schema/input digest/upstream corpus digest, plan, sections, claims, bibliography, coverage, validation report, execution.

LLM chỉ trả section/claim objects trong tập ID cho phép. Code tự sinh citation markers và bibliography từ registry → manifest. Không nhận reference list/URL tự do do LLM tạo. Câu chuyển ý và caveat do renderer/template cung cấp để không có đường né validation qua trường free-text.

Execution cần: logical jobs, HTTP attempts, transport retries, repair calls, cache hits, input/output token usage nếu có, reserved tokens, queue wait, API time, total time, provider/model và stop reason. Usage không có là `null`, không ghi 0.

Tách `generation_status` (`complete/partial/fallback/empty/failed`) khỏi `validation_status`. “Complete” nghĩa là đủ phần đã lên kế hoạch và qua structural gate, không có nghĩa nội dung đã được chuyên gia xác minh.

## 5. Governor chống burst — điều kiện bắt buộc trước live Stage 4

### Một nơi sở hữu retry

Tách transport thành `send_once()` chỉ phát đúng một HTTP request. Governor sở hữu admission, counters, timeout và transport retry. Writer chỉ quyết định có repair nội dung hay không; repair cũng nộp vào governor. Không còn nested retry trong client/SDK.

Giữ wrapper `complete(...)->str` cho call sites cũ; bổ sung structured response nội bộ chứa text, usage, finish reason, provider/model và retry metadata. Truyền max output tokens bằng tham số đúng của từng adapter. Regression-test wrapper trước khi nối các stage cũ.

### Admission trước từng HTTP attempt

1. Kiểm tra deadline và hard budget cấp run, đồng thời budget cha nếu chạy toàn pipeline. Giá trị 0 nghĩa là không được gọi, không phải unlimited.
2. Qua cache/single-flight trước để request trùng không bị enqueue nhiều lần.
3. Chờ slot `max_in_flight=1`, khoảng cách thời điểm bắt đầu request, sliding-window RPM/TPM và cooldown.
4. Reserve request/token budget atomically rồi mới gửi. Retry/repair dùng cùng cơ chế, không có đường bypass.
5. Cập nhật usage/response headers khi có; timeout có thể đã tiêu quota nên giữ reservation bảo thủ. Release in-flight slot trong `finally`.

Không tích lũy permit khi idle: pacing capacity một, không “bù” các slot bị bỏ lỡ. Baseline `min_start_interval_s=15`; khi có quota thực, `effective_interval = max(15, 60 / (RPM_limit × 0.8))`. Đồng thời kiểm tra cửa sổ trượt 60 giây, vì khoảng cách RPM không kiểm soát được TPM.

Reserve input estimate + output allowance cho local token budget một cách bảo thủ; quota adapter theo dõi đúng loại token mà provider tính (input/total/khác). Một request lớn hơn quota khả dụng phải repack hoặc dừng trước khi gửi, không chờ vô hạn. Cộng usage chỉ khi phù hợp semantics, không refund quota phỏng đoán.

Gemini áp dụng quota theo project, và có các chiều RPM/TPM/RPD; giá trị thực phụ thuộc tier/model. Cấu hình lấy từ project đang dùng, không hardcode bảng free tier. Nguồn: [Gemini API rate limits](https://ai.google.dev/gemini-api/docs/rate-limits).

### Phạm vi chia sẻ

- Trong một process: governor singleton/injected dùng chung Stage 1–4, keyed theo provider + project/quota group + endpoint; nhóm model chia quota dùng chung bucket.
- Nhiều CLI process trên cùng máy: dùng SQLite transactions lưu reservations, cooldown, request ledger và lease có expiry/heartbeat. Chọn state path chung, không tạo limiter mới trong mỗi output directory. Không ghi API key vào state/log.
- Nếu chưa làm coordinator đa process trong MVP, phải enforce một live run bằng khóa process và vẫn lưu pacing/cooldown qua lần chạy; không chỉ khuyến cáo người dùng tự tránh chạy song song.
- Nhiều máy cần shared coordinator về sau. Limiter local không kiểm soát ứng dụng bên ngoài cùng project; vì vậy không hứa tuyệt đối không có 429.

### Chính sách lỗi

- 429 tạm thời: cooldown chung quota group; parse `Retry-After` dạng seconds hoặc HTTP-date và provider retry metadata nếu có. Chờ tối thiểu server yêu cầu, kết hợp exponential backoff + jitter và pacing. Không switch key/provider tự động.
- Hết daily quota/billing/auth hoặc 400 do request không hợp lệ: dừng, không retry. 503/5xx/network timeout có thể retry một lần nếu còn budget; timeout mơ hồ vẫn tính attempt đã gửi.
- Hai lỗi transient liên tiếp: mở circuit cho quota group; run hiện tại trả partial/fallback. Sau cooldown chỉ một half-open probe được phép, không xả hàng đợi đồng loạt.
- Cooldown vượt deadline: lưu checkpoint rồi trả `deferred` reason trong execution, không sleep vô hạn. Không ngủ sau attempt cuối.
- Parse/schema/citation lỗi: không coi là transport error. Giữ claims tốt, gom lỗi vào tối đa một repair job cho cả run. Repair quá lớn thì sửa phần ưu tiên, phần còn lại fallback; không đệ quy chia thêm call.

## 6. Ngân sách baseline

Các giá trị đề xuất, cần điều chỉnh qua measurement:

```python
WritingConfig(
    use_llm=False,                 # offline mặc định; --write bật prose LLM
    max_generation_batches=3,
    max_repair_calls=1,            # chung cả bài
    max_http_attempts_per_run=5,   # bao gồm generation + repair + retry
    max_transport_retries_per_job=1,
    max_in_flight=1,
    min_start_interval_s=15.0,
    quota_safety_factor=0.8,
    request_timeout_s=90.0,
    run_deadline_s=600.0,
    target_words=1500,
    language="en",
)
```

Input/output/run-token caps và quota RPM/TPM/RPD phải resolve thành giá trị cụ thể theo model/profile trước live call; dry-run hiển thị tất cả. Thiếu quota profile: trả `quota_configuration_required` cùng offline draft; không giả định 15 giây là an toàn cho mọi tài khoản. User có thể chọn ngôn ngữ và độ dài, packing phải tính lại.

- Cache hit hoặc offline: 0 HTTP attempt.
- Review vừa một batch, không lỗi: 1 attempt; có repair: 2 attempts.
- Ba batch + một repair, không transport lỗi: 4 attempts.
- Mọi đường lỗi vẫn tối đa 5 attempts; đây là trần, không phải số call cần dùng hết. Không reserve cứng repair khiến generation thiếu chỗ; hết budget thì fallback.

Ví dụ giả định 60 papers được pack thành 3 batch: generation start sớm nhất tại 0/15/30 giây khi latency dưới 15 giây và quota cho phép. Nếu mỗi request mất 40 giây thì 0/40/80 giây; TPM/cooldown có thể kéo dài thêm. Đây là minh họa scheduler, không phải benchmark.

Để giảm call toàn pipeline, giữ Stage 3 `summarize=False` nếu chỉ cần review cuối; Stage 4 viết trực tiếp từ cards/evidence. Nếu narration đã có trong cache thì tái sử dụng.

## 7. Citation gate và research gap

Code kiểm tra từng claim: refs tồn tại, refs nằm trong pack đã gửi, subjects có evidence đúng paper/version, evidence kind phù hợp, cluster ownership, shared/difference có support đủ các bên. Thiếu/duplicate section, claim ID lạ và output bị cắt đều được phát hiện.

Không tự chèn paper citation cho claim thiếu refs. Không dùng regex arXiv làm kiểm tra duy nhất; bibliography được render từ structured metadata. Numeric ranking và kết luận superiority chưa được hỗ trợ bởi schema hiện tại thì từ chối. Rule/keyword checks bổ sung chỉ là heuristic, không chứng minh semantic support.

Research Gap chia nội dung thành limitation đã ghi nhận có refs và câu hỏi cần kiểm chứng, luôn giữ `scope=extraction_snapshot`, `verification_needed=True`. Quan hệ chưa được ghi nhận là thiếu thông tin trong snapshot, không phải “chưa có nghiên cứu nào”. Không nâng confidence heuristic hoặc clustering threshold provisional thành kết luận chắc chắn.

Đo semantic support bằng review thủ công trên tập giữ riêng; không thêm LLM judge vào đường chạy mặc định. Strict mode dùng evidence nguyên văn + template; prose mode luôn được gắn nhãn structural validation.

## 8. Cache, resume và fallback

- Cache key gồm writing input digest, evidence/assignment/claim content, topic, language/length, section plan, selection/prompt/schema/validator/render versions, provider/model và generation settings. Không chỉ dùng corpus digest vì narration có thể đổi trên cùng corpus.
- Batch cache chỉ nhận phần đã validate; load lại phải validate và đúng key. Partial checkpoint khác successful final cache, không được làm lần sau tưởng bài đã complete.
- Atomic write với unique temp path + rename; single-flight theo cache key. Concurrent callers recheck cache sau khi nhận lock.
- Resume giữ cùng run ID và ledger attempts/tokens đã tiêu; không reset budget khi restart. Run mới phải explicit và vẫn chịu quota state chung.
- No key/provider unavailable/budget hết: giữ phần hợp lệ, phần thiếu dùng template có evidence hoặc ghi thiếu support. Không mô tả offline template là bài viết LLM hoàn chỉnh.
- Output gồm `.review.md` và `.writing.json`; sidecar lưu provenance/coverage/diagnostics. Metadata bibliography thiếu được giữ thiếu, không gọi API bổ sung.

## 9. Thứ tự triển khai

### Bước 1 — Shared LLM control

Thêm `llm/governor.py`, typed errors/response, single-attempt adapters trong `llm/client.py`, persistent quota state và test fake clock/transport. Nối Stage 1–3 vào cùng governor; giữ giới hạn logical hiện có như giới hạn nghiệp vụ. Sửa Stage 3 để không biến transport/budget/auth error thành repair prompt.

Điều kiện qua bước: không call site nào retry HTTP ngoài governor; concurrent/retry requests đều có ledger và pacing.

### Bước 2 — Offline writing core

Thêm `writing/{types,plan,evidence,validate,render,fingerprint,pipeline}.py`. Xây contract, outline, evidence selection, templates, digest/cache và coverage; chưa cần live API.

### Bước 3 — Bounded prose generation

Thêm prompt builder, token packing, generation queue, một repair gộp, checkpoint/resume. Wire `WritingConfig` và metadata từ client. Không thêm LangGraph/Redis khi local coordinator đủ nhu cầu.

### Bước 4 — CLI và evaluation

Thêm `writing/{__init__,__main__,cli}.py`, entry point `write-review`, example và evaluation fixture. CLI dự kiến:

```bash
python -m research_assistant.writing results/pruning.synthesis.json \
  --dry-run --language en --target-words 1500

python -m research_assistant.writing results/pruning.synthesis.json \
  --write --md-out results/pruning.review.md \
  --json-out results/pruning.writing.json
```

Đây là interface dự kiến, chưa chạy được. Có `--require-complete`: exit 1 khi partial/fallback; malformed input/config exit 2. Offline/empty hợp lệ xuất artifact có status rõ ràng.

## 10. Kiểm thử và tiêu chí hoàn thành

Test không dùng API thật, inject clock/sleep/transport:

1. Cache/offline/empty/invalid input đúng zero-call behavior; budget 0 tuyệt đối không request.
2. Mọi attempt kể cả retry và repair giữ min start interval, in-flight ≤ 1, sliding-window limits; idle rồi có nhiều jobs không tạo burst.
3. Concurrent runs/processes không double-reserve; restart giữ cooldown/budget; lease chết được thu hồi mà không gửi trùng ngay.
4. 429 Retry-After cả hai format, 503/timeout, daily exhaustion, invalid auth, circuit/deadline; mọi path attempts ≤ 5 và không nested retry.
5. Token pack vượt context/output/TPM được xử lý trước request, không mất evidence ID hoặc qualifier âm thầm; truncated response không thành complete.
6. Unknown/cross-paper/version refs, refs không trong prompt, unsupported subject, missing section, invalid gap và numeric ranking; valid claims không bị mất khi repair fail.
7. Input narration/evidence/language/model thay đổi gây cache miss; corrupt cache recoverable; partial không che mất phần thiếu.
8. JSON round-trip, Markdown citations/bibliography nhất quán, no PDF/retrieval/model-index init.

Sau unit/regression tests, benchmark có kiểm soát trên snapshot nhỏ/vừa và 60–70 papers: HTTP attempts, input/output tokens, latency và queue wait, 429 rate, paper/cluster evidence coverage, fallback rate, claim support qua người đọc. So sánh one-batch với capped batching trên cùng snapshot/config; báo rõ quality–cost–latency tradeoff. Không tăng concurrency trước khi đo.

Definition of Done: pipeline tạo artifact có provenance; prose bình thường chỉ 1 call khi vừa budget, trần HTTP được chứng minh bằng test; không burst từ request do hệ thống điều phối; claim/reference validity và giới hạn semantic verification được thể hiện rõ. Không cam kết loại bỏ mọi 429 từ quota bị ứng dụng khác dùng hoặc provider overload.
