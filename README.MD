# AIPR — Trợ lý CTXH quy đổi và đăng ký

**Nhóm:** Lê Quang Thắng — Đặng Tấn Phúc Thịnh.
**Trường:** HCMC UNIVERSITY OF TECHNOLOGY AND ENGINEERING.
Dữ liệu trong project là giả lập, không phải quy định hay hệ thống của trường.

## Bắt đầu trên VS Code
1. Cài Python 3.11 trở lên, Git, VS Code và extension Python của Microsoft.
2. Giải nén toàn bộ ZIP, giữ thư mục `.git`. Mở **thư mục aipr_ctxh** bằng File → Open Folder.
3. Terminal → New Terminal, chạy **một dòng**:

```powershell
python run.py setup-demo
```

Nếu Windows chỉ nhận `py`, dùng `py -3 run.py setup-demo`.
Lần đầu cần mạng tải thư viện; về sau local demo không cần mạng hay API key.
Đã kiểm tra trên Linux/Python 3.12; nhánh đường dẫn Windows có trong run.py nhưng chưa kiểm thử trên máy Windows thật.
Chọn interpreter `.venv\Scripts\python.exe` trong VS Code (Linux/macOS: `.venv/bin/python`).
Không bấm Run trực tiếp từng file trong src vì các file dùng import theo package.

Máy có GNU Make dùng đúng lệnh theo đề:

```bash
make setup && make demo
```

Đây là demo terminal, không cần chạy web server và không có giao diện trình duyệt.

## Các lệnh dùng hằng ngày

| Việc | Windows và các hệ điều hành | Có GNU Make |
|---|---|---|
| Demo 3 câu | `python run.py demo` | `make demo` |
| Tự nhập câu hỏi | `python run.py chat` | `make chat` |
| Đánh giá 3 lần, dense và hybrid | `python run.py eval` | `make eval` |
| Kiểm thử chức năng | `python run.py test` | `make test` |
| Tự kiểm theo đề | `python run.py check` | `make check` |

Ví dụ: `Tính tổng ngày CTXH của A01 và A02.`
Ví dụ có ghi DB: `Đăng ký A01 cho mã sinh viên demo 11110001.`
Ví dụ thiếu dữ liệu: `Quy đổi A04 ra ngày CTXH.`
Gõ `exit` để thoát chat. Mã 11110001 chỉ là mã dùng thử, không phải MSSV của thành viên.

## Bật API để hoàn tất bằng chứng LLM

Copy `.env.example` thành `.env`, nhập key **trong máy của bạn**, không gửi key vào chat:

```dotenv
AIPR_MODE=api
OPENAI_API_KEY=nhap_key_cua_ban
OPENAI_MODEL=gpt-4.1-mini
OPENAI_BASE_URL=https://api.openai.com/v1
```

Nếu dùng Groq như ví dụ trong tài liệu lớp, đặt OPENAI_BASE_URL thành endpoint OpenAI-compatible của tài khoản Groq, đặt OPENAI_MODEL theo model được tài khoản hỗ trợ và dùng key Groq ở biến OPENAI_API_KEY. Tên biến là tên chung của adapter, không bắt buộc provider là OpenAI.
Chạy lại `python run.py demo`, sau đó `make eval` (hoặc `python run.py eval`).
`--engine api` / `--engine local` ghi đè cấu hình. API mode không âm thầm chuyển về local.
Có thể dùng `python run.py eval --engine api --repeats 1` để kiểm tra trước khi chạy đủ 3 lần.
API có thể phát sinh phí. Điền đơn giá hiện hành vào hai biến PRICE trong .env nếu muốn tính cost_usd;
không có đơn giá hoặc usage không đầy đủ thì CSV để trống, không đoán số tiền.

**Bộ được giao hiện tại:** đã kiểm thử local; chưa có key nên chưa có log LLM thật.
Local dùng bộ chọn hành động theo luật, không được gọi kết quả local là chất lượng LLM agent.
API mode dùng model chọn tool qua prompt và catalog MCP, host giữ các guard ghi/verify.
Câu trả lời được tổng hợp từ facts và kết quả tool theo mẫu để dễ kiểm chứng.

## Chunking và lý do (yêu cầu 2.5)
- Chunk theo **câu** (`src/chunking.py`): gom các câu liên tiếp tới tối đa 220 ký tự, không bao giờ cắt giữa câu. Tài liệu ngắn (mọi `ACT-*`, `GUIDE`, `VERIFY`) = 1 chunk; `POLICY` v2 (382 ký tự) tách thành 2 chunk: `POLICY:v2:0` (quy đổi 8 giờ = 1 ngày) và `POLICY:v2:1` (điều kiện đăng ký).
- Lý do: con số luôn đi cùng điều kiện áp dụng (8 giờ/ngày, mở/đóng, số chỗ) nên ranh giới cắt là câu, không phải số ký tự cố định. Hai nhóm quy tắc của POLICY tách riêng để câu hỏi về đăng ký không kéo theo đoạn quy đổi và ngược lại.
- Mỗi chunk có tiền tố tiêu đề (`"title. text"`) để đứng một mình vẫn rõ chủ đề; metadata `doc_id, version, owner, chunk_id = doc_id:vN:i`. Không overlap vì câu đã là ranh giới ý trọn vẹn.
- Chỉ bản `version` cao nhất của mỗi `doc_id` được lập chỉ mục; truy vấn lọc `owner` **trước** khi xếp hạng (mỗi owner một chỉ mục BM25 + embedding).
- Trích dẫn đúng chunk: claim `hours_per_day` phải trích chunk thật sự chứa câu "8 giờ bằng 1 ngày"; `make eval` chấm faithfulness bằng đúng điều kiện này.
- Hạn chế trung thực: corpus chỉ 8 tài liệu nên ngưỡng 220 ký tự chưa được tinh chỉnh; thử 120 / 220 / 400 ký tự và đo lại `first_recall_at_4` nếu có thêm dữ liệu dài.

## Hai chế độ truy vấn (yêu cầu 2.5)
| Chế độ | Hàm (tái dùng `course/ex2_1_hybrid_search.py`, = `ex2_1.py` của nhóm) | Điểm |
|---|---|---|
| `dense` | `dense_search` | cosine(embedding truy vấn, embedding chunk) |
| `hybrid` | `hybrid_search` | 0.3 × BM25 + 0.7 × cosine, sau min-max |

- Embedding: nếu import được `Vector_Store` của lớp (thư mục `../Week_4`, hoặc đặt biến môi trường `W4_PATH`) thì dùng `embed_text` thật; nếu không, dùng `src/embedding.py` (LSA offline) để repo vẫn chạy 1 lệnh. Log `RUN_START` ghi backend đang dùng.
- Chọn chế độ: `python run.py demo --retrieval dense|hybrid` (mặc định hybrid); `make eval` chạy cả hai và so sánh trong `summary.json` và `per_question.csv`.

## Đối chiếu yêu cầu và bằng chứng
Chạy `make check` (hoặc `python run.py check`) để tự kiểm theo bảng yêu cầu của đề.

| Yêu cầu | Bằng chứng |
|---|---|
| 2.3 Typing | `src/models.py` (`@dataclass`, `TypedDict`) |
| 2.4 API retry/timeout/log, prompt tách file | `src/llm.py`, `templates/agent.txt` |
| 2.5 dense + hybrid, chunking, `make eval` | `course/ex2_1_hybrid_search.py`, `src/chunking.py`, `src/retrieval.py`, `results/*/summary.json` |
| 2.6 MAX_STEPS, `Error:`, parse an toàn | `src/agent.py`, `src/config.py`, `tests/test_safety.py` |
| 2.7 ghi rồi verify | log `VERIFY` trong `logs/`, `register_activity` → `get_registration` |
| 2.8 MCP stdio, ≥2 tool, discover động | `src/server.py`, `src/mcp_client.py` |
| Cộng điểm: so sánh dense/hybrid (CSV chi tiết từng câu) | `make eval`, `per_question.csv` |

## Nơi xem kết quả
- `logs/`: JSONL thật, có timestamp, câu hỏi, tool arguments/results, VERIFY và USAGE.
- `results/local-.../` hoặc `results/api-.../`: summary.json, per_question.csv.
- `results/latest_local.txt` / `latest_api.txt`: tên lần đánh giá gần nhất.
- `state/registrations.sqlite3`: DB demo. Không đưa dữ liệu sinh viên thật vào đây.
- Eval dùng DB tạm riêng từng câu và không thay đổi DB demo.
- Xóa riêng file `state/registrations.sqlite3` nếu muốn bắt đầu demo từ DB rỗng (mất đăng ký demo).

## Đọc code theo thứ tự
1. `src/models.py`: dataclass, TypedDict và cấu trúc đáp án.
2. `data/documents.json`: corpus, metadata và facts.
3. `src/chunking.py` → `src/retrieval.py` → `course/ex2_1_hybrid_search.py` (+ `src/embedding.py` dự phòng): chunk, dense, BM25 và fusion.
4. `src/server.py`: bốn MCP tool và SQLite.
5. `src/mcp_client.py`: initialize, list_tools, call_tool.
6. `src/agent.py`: parse JSON, vòng lặp tối đa 8 bước, điều kiện ghi và verify.
7. `src/llm.py` + `templates/agent.txt`: API, retry, timeout, prompt.
8. `src/evaluate.py`: gold, faithfulness quy tắc, từ chối, Recall@4, USAGE.

## Trước khi nộp — chưa được bỏ qua
- [ ] Nếu có `Vector_Store.py` của lớp: đặt vào `../Week_4` (hoặc `W4_PATH`), chạy lại `make eval` và cập nhật số liệu báo cáo; hiện đang dùng embedding LSA dự phòng.
- [ ] Chạy API thật ít nhất 3 lượt, đọc log và cập nhật số liệu cùng trace trong report.
- [ ] Điền giảng viên và MSSV trên bìa; xác nhận tên/đề tài với nhóm.
- [ ] Mỗi người sửa, kiểm thử và commit phần mình bằng tài khoản thật; cập nhật hash trong report.
- [ ] Đọc ai-usage.md; khai báo AI đầy đủ, tập giải thích/sửa code trực tiếp.
- [ ] Thử lệnh cold start trên máy Windows của nhóm, từ một bản giải nén mới.

Không giả commit của sinh viên. Các commit khởi tạo ghi tác giả AI Scaffold, không chứng minh đóng góp cá nhân.
Xem `docs/LIEN_HE_BAI_HOC.md`, `docs/YEU_CAU.md`, `docs/HUONG_DAN_BAO_VE.md`, `docs/PHAN_CONG.md`.

## Chỉnh báo cáo
`report.docx` có khung bìa, mục lục, hình kiến trúc và trace.
Số liệu local đọc từ kết quả thực. Báo cáo là bản có thể chỉnh sửa, chưa thay thế log API và commit của nhóm.
Script dựng báo cáo: `docs/build_report.py`. Sau khi có kết quả API, cập nhật nội dung mô tả local/API,
chọn đúng thư mục kết quả, tạo lại báo cáo và kiểm tra toàn bộ trang trong Word.
Không chỉ thay số mà giữ nguyên lời nhận xét cũ.
