# Research Assistance Agent

Pipeline nghiên cứu tài liệu gồm 4 stage: **retrieval → extraction → synthesis → writing**.

## Sử dụng giao diện (không cần nhập câu lệnh)

Trên máy Linux hiện tại, mở **Research Assistant.desktop** trong thư mục dự án bằng trình quản lý tệp. Nếu hệ điều hành hỏi, chọn cho phép chạy launcher (Allow Launching). Launcher sử dụng `.venv` của dự án và mở trình duyệt tại **http://127.0.0.1:8765**. Giữ cửa sổ launcher mở trong khi sử dụng.

Giao diện hỗ trợ **Tiếng Việt** và **English** qua nút `VI / EN` ở góc trên bên phải. Lựa chọn được ghi nhớ trên trình duyệt và chỉ thay đổi ngôn ngữ giao diện; trường **Ngôn ngữ bản review / Review language** vẫn điều khiển ngôn ngữ của tài liệu đầu ra riêng biệt.

1. Nhập **chủ đề cần tìm**. Chọn số bài, ngôn ngữ review và số từ mục tiêu.
2. Mở **API & cấu hình .env**, nhập API key của Gemini, OpenAI hoặc Groq; nhập đủ **LLM_RPM**, **LLM_TPM**, **LLM_RPD** theo quota thực tế của nhà cung cấp. Giao diện mặc định bật tổng hợp và viết bằng LLM.
3. Kiểm tra **thư mục dữ liệu SPECTER2**. Mặc định là `data/specter2_artifacts`; thư mục phải có `manifest.json`, các shard embeddings/metadata và FAISS index (index có thể được pipeline dựng lại). UI sử dụng corpus có sẵn, không tự tạo corpus từ Internet. Lần đầu chạy có thể cần tải model SPECTER2.
4. Có thể mở **Cấu hình từng stage** để chỉnh bộ lọc năm, categories, provider, cache, hạn mức gọi LLM và các thông số xử lý khác.
5. Bấm **Lưu cấu hình .env** để dùng lại cấu hình. Bấm **Chạy toàn bộ pipeline** để chạy cả 4 stage bằng giá trị đang hiển thị; nút chạy không tự ghi đè `.env`.
6. Theo dõi trạng thái từng stage, đọc bản review và tải các tệp JSON/Markdown ngay trên giao diện. Có thể **Dừng pipeline**, xem các lượt trước hoặc chạy lại với cấu hình đã lưu.

Ô API key trống giữ nguyên key đã lưu. Muốn xóa key, chọn **Xóa key đã lưu** rồi lưu. Key không được gửi lại từ server về giao diện và không được ghi vào kết quả từng lượt. `.env` vẫn là tệp văn bản chứa key; ứng dụng lưu tệp với quyền chỉ chủ sở hữu đọc/ghi. Cấu hình nâng cao được lưu dưới dạng `RA_<STAGE>_<FIELD>` (ví dụ `RA_RETRIEVAL_TOP_K='25'`). Các biến `RA_*` được UI đọc; các CLI riêng vẫn sử dụng tham số CLI của chúng.

Các lượt chạy được lưu trong `results/ui/<run-id>/`:

- `stage-1-retrieval.json`
- `stage-2-extraction.json`
- `stage-3-synthesis.json`
- `stage-4-writing.json`
- `review.md`
- `config.json` (chủ đề và tham số từng stage, không chứa key) và `status.json`

Nếu một stage lỗi, pipeline dừng, giữ kết quả đã tạo và hiển thị lỗi. Review `partial`, `fallback` hoặc `empty` được hiển thị kèm lưu ý, không được báo thành bản LLM hoàn chỉnh. Đóng server sẽ dừng lượt đang chạy; chạy lại có thể tận dụng cache của các stage, nhưng không tiếp tục tiến trình đã dừng. Mỗi server chỉ chạy một pipeline tại một thời điểm.

Giao diện chỉ lắng nghe trên `127.0.0.1`, dành cho sử dụng cá nhân trên máy. Các API key được gửi đến nhà cung cấp tương ứng khi pipeline gọi LLM; dữ liệu bài báo và chủ đề được xử lý theo hành vi sẵn có của pipeline. Không đưa server này trực tiếp lên Internet.

## Cài đặt lần đầu / dành cho người phát triển

Yêu cầu Python >= 3.10, môi trường Python có dependencies trong `pyproject.toml`, corpus SPECTER2 và kết nối tới các dịch vụ đang bật. Môi trường `.venv` trên máy hiện tại đã có sẵn.

```bash
python -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
.venv/bin/python launch_ui.py
```

Sau khi cài, cũng có thể chạy `research-assistant-ui` hoặc `python -m research_assistant.ui.server`. Có tùy chọn `--port` và `--no-browser`. Nếu chuyển thư mục dự án, cập nhật `Exec` và `Path` trong launcher `.desktop` cho đúng vị trí mới. Trên hệ điều hành khác, chạy `launch_ui.py` bằng Python của môi trường đã cài dependencies.

```bash
.venv/bin/python -m pytest -q
```

Bộ kiểm thử UI dùng pipeline giả lập để kiểm tra chuyển dữ liệu và trạng thái, không tiêu thụ quota thật. Kiểm thử HTTP cần quyền mở socket localhost.

## Cấu trúc dự án

- `research_assistant/`: mã nguồn pipeline và giao diện; mỗi stage có package riêng.
- `tests/`: kiểm thử tự động; `tests/fixtures/` chứa dữ liệu mẫu cho kiểm thử.
- `examples/`: script mẫu để chạy từng stage.
- `evaluation/`: notebook đánh giá, bộ dữ liệu chuẩn và mã chấm điểm.
- `notebooks/`: notebook tạo corpus, hiện có [embedding SPECTER2](notebooks/embedding.ipynb) dành cho Kaggle.
- `docs/design/`: [thiết kế tổng thể](docs/design/research-assistant-agent-design.md) và [đề xuất cải tiến](docs/design/recommendations.md).
- `docs/plans/`: kế hoạch triển khai [synthesis](docs/plans/stage-3-synthesis-plan.md) và [writing](docs/plans/stage-4-writing-plan.md).
- `docs/guides/`: [hướng dẫn stage writing](docs/guides/stage-4-writing.md).
- `docs/reports/`: báo cáo các giai đoạn [1](docs/reports/01-report.md), [2](docs/reports/02-report.md), [3](docs/reports/03-report.md); nội dung phản ánh thời điểm viết báo cáo.
- `data/`: corpus, cache pipeline và trạng thái hạn mức LLM trên máy; không đưa vào Git.
- `results/`: kết quả các lượt chạy trên máy; không đưa vào Git.

Chạy các lệnh từ thư mục gốc dự án. Giữ `launch_ui.py` và `Research Assistant.desktop` tại đây để mở giao diện thuận tiện. Dependency được khai báo duy nhất trong `pyproject.toml`; `requirements.txt` tham chiếu cấu hình đó để tương thích với `pip install -r requirements.txt`.

Các thư mục `__pycache__/`, `.pytest_cache/` và `.ipynb_checkpoints/` là cache có thể xóa và sẽ được tạo lại khi cần. `.venv/` và `*.egg-info/` phục vụ môi trường đã cài đặt; giữ lại khi đang sử dụng dự án.
