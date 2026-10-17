# AIPR — Trợ lý Công tác xã hội (CTXH)

**Nhóm:** Lê Quang Thắng — Đặng Tấn Phúc Thịnh · HCMC University of Technology and Engineering
**Dữ liệu:** `data/CTXH.md` — *Quy định tổ chức thực hiện Chương trình CTXH đối với SV hệ chính quy* (văn bản thật của trường).
Sổ ghi nhận hoạt động (`state/ctxh.sqlite3`) chỉ là **DEMO**, không phải hệ thống PSC của trường; mã sinh viên `11110001` chỉ để thử.

## Chạy

Cần Python 3.11+.

```bash
make setup && make demo          # Linux/macOS/Git-Bash
python run.py setup-demo         # Windows (tạo .venv, cài thư viện, chạy demo)
```

Mặc định `AIPR_MODE=local` nên demo không cần API key.

| Việc | Lệnh | Windows |
|---|---|---|
| Demo 3 câu (tra mã, RAG, ghi + verify) | `make demo` | `python run.py demo` |
| Hỏi tự do | `make chat` | `python run.py chat` |
| Hỏi một câu tùy ý | — | `python run.py demo --question "..."` |
| So sánh dense và hybrid | `make eval` | `python run.py eval` |
| Kiểm thử (20 test) | `make test` | `python run.py test` |

Tùy chọn thêm cho `demo`/`eval`: `--engine local|api`, `--retrieval dense|hybrid` (chỉ demo/chat), `--repeats N` (chỉ eval).

Ví dụ câu hỏi: `Mã 001202 được bao nhiêu ngày CTXH?` · `Tổng điểm 47 quy đổi mấy ngày?` ·
`Ghi nhận sinh viên 11110001 mã 001202, 5 điểm, ngày 2026-10-10.` · `Không ghi nhận, chỉ tính mã 001202.`

## Dùng LLM thật

Copy `.env.example` thành `.env` rồi điền key trên máy của bạn (`.env` đã nằm trong `.gitignore`):

```dotenv
AIPR_MODE=api
OPENAI_API_KEY=...
OPENAI_MODEL=gpt-4.1-mini
OPENAI_BASE_URL=https://api.openai.com/v1   # hoặc endpoint OpenAI-compatible khác
```

Sau đó chạy `make demo` và `make eval` (mode api mặc định lặp 3 lượt, mỗi lượt có 1 file `logs/*.jsonl` kèm dòng `USAGE`).
API mode **không** tự chuyển về local. **Engine `local` chỉ là bộ chọn hành động theo luật (baseline), không phải LLM.**

Cấu hình gọi API (`src/config.py`, `src/llm.py`): timeout 30 giây, thử lại tối đa 3 lần (lỗi 5xx, 408, 409, 429 và lỗi mạng),
lỗi 4xx khác (sai key/model/base_url) dừng ngay. Mỗi lần gọi chờ thêm 5 giây để tránh vượt hạn mức nên `make eval` ở mode api chạy khá lâu.

## Kiến trúc

```
Người dùng ─▶ src/main.py ─▶ src/agent.py   (vòng lặp tự viết, MAX_STEPS=8, guard ghi, verify bắt buộc)
                              │  chọn hành động: local (src/planner.py) hoặc LLM (src/llm.py + templates/agent.txt)
                              ▼  MCP stdio (list_tools, call_tool)
                          src/server.py     (FastMCP, 6 tool)
     ĐỌC: search_regulations · lookup_activity · points_to_days · get_record · get_student_summary
     GHI: record_activity ──▶ SQLite ──▶ get_record (verify)
                          │ RAG: src/chunking.py → src/retrieval.py → src/ex2_1_hybrid_search.py (w4)
                          └ bảng mã: src/activities.py
```

Văn bản được chia thành 6 tài liệu có metadata `doc_id, title, version, date, owner` (`data/documents.json`):
CTXH-I … CTXH-V theo mục lớn, và CTXH-II-BANG cho bảng mã. Văn bản không ghi ngày ban hành nên `date = null`.

## Chunking

- **Mỗi dòng bảng mã là 1 chunk**, ô "Lĩnh vực" để trống thì lấy của dòng trên. Cắt theo số từ sẽ làm mã `001105` mất chữ "Mùa hè xanh" hoặc tách mã khỏi điểm/ngày.
- **Văn xuôi:** gom các đoạn cùng tiêu đề, tối đa 110 từ; đoạn dài hơn thì cắt ở cuối câu hoặc dấu `;`.
- Chunk văn xuôi bắt đầu bằng đường dẫn tiêu đề trong ngoặc vuông (`[III > 7. Bước 7:] ...`), không overlap.
- Chunk id dạng `doc_id:vN:i` để trích dẫn; `make eval` kiểm tra nguồn trích dẫn có chứa bằng chứng thật không.
- Corpus có 102 chunk (55 dòng bảng + 47 đoạn văn xuôi). Ngưỡng 110 từ chưa được tinh chỉnh bằng thực nghiệm.

## Dense và hybrid

| Chế độ | Hàm (`src/ex2_1_hybrid_search.py`) | Điểm |
|---|---|---|
| `dense` | `dense_search` | cosine giữa embedding câu hỏi và chunk |
| `hybrid` | `hybrid_search` | 0.3 × BM25 + 0.7 × cosine, sau min-max |

- **Embedding:** `src/embedding.py` — TF-IDF + SVD (64 chiều, học từ chính 102 chunk), chạy offline bằng numpy nên repo chạy được bằng 1 lệnh.
  Đây là embedding thống kê, không phải mô hình ngữ nghĩa, nên dense ở đây chủ yếu vẫn dựa trên từ trùng.
- **BM25:** `rank-bm25`, tách từ bằng `words()` trong `src/embedding.py` (chữ thường, theo `\w+`).
- **Metadata filter:** tham số `doc_id` của `search_regulations` lọc chunk *trước* khi xếp hạng. Prompt `templates/agent.txt` hướng dẫn LLM dùng nó.

## Kết quả đánh giá

Bộ test `tests/testset.jsonl`: 19 câu (14 câu có đáp án, 5 câu `no_answer`; trong đó có câu ghi dữ liệu, câu cấm ghi, câu thiếu thông tin).
Số liệu lấy từ `results/*/summary.json`.

**Retrieval** (13 câu có bằng chứng, K = 4): dense và hybrid **giống hệt nhau**: Hit@4 = 0,923, MRR = 0,872.
Câu trượt duy nhất là Q16 ("Không ghi nhận gì cả, chỉ tính mã 001202…"), câu này agent xử lý bằng `lookup_activity` chứ không qua tìm kiếm.
Vì vậy bộ test hiện tại **chưa phân biệt được** dense với hybrid ở tầng retrieval.

**Agent** (độ chính xác theo từng câu):

| Engine | Chế độ | Số lượt | Đạt | Đạt (câu có đáp án) | Từ chối đúng | Trích dẫn đúng | Câu từng sai |
|---|---|---|---|---|---|---|---|
| local (không LLM) | dense | 3 | 1,000 | 1,000 | 1,000 | 1,000 | — |
| local (không LLM) | hybrid | 3 | 1,000 | 1,000 | 1,000 | 1,000 | — |
| api (`gpt-4.1-mini`) | dense | 3 | 0,982 | 1,000 | 0,933 | 1,000 | Q11 |
| api (`gpt-4.1-mini`) | hybrid | 3 | 0,965 | 0,952 | 1,000 | 1,000 | Q17 |

Lượt api (`results/api-20261017-235352`): dense dùng 103 lần gọi, 246.075 token prompt + 5.672 token completion; hybrid dùng 106 lần gọi, 252.081 + 5.873 token.
Trên 57 lượt trả lời (19 câu × 3), dense sai 1 lượt và hybrid sai 2 lượt, nên chênh lệch dense/hybrid ở bảng trên **nằm trong mức nhiễu**, không đủ để kết luận chế độ nào tốt hơn.
Baseline local đạt 100% nhưng luật của nó được chỉnh trực tiếp trên bộ test này (xem phần Giới hạn), nên không nên dùng con số đó để so với LLM.

## Đối chiếu yêu cầu

| Yêu cầu | Bằng chứng |
|---|---|
| 2.1 một lệnh | `make setup && make demo` |
| 2.2 agent tự viết | `src/agent.py` (không dùng framework) |
| 2.3 typing | `src/models.py` (`@dataclass`, `TypedDict`) |
| 2.4 API có retry/timeout/log, prompt tách file | `src/llm.py`, `templates/agent.txt`, `tests/test_core.py::LlmAdapterTests` |
| 2.5 dense + hybrid, chunking, `make eval` | `src/chunking.py`, `src/retrieval.py`, `src/evaluate.py` |
| 2.6 MAX_STEPS, `Error:`, parse an toàn | `src/agent.py::parse_action`, `src/config.py`, `src/mcp_client.py::call` |
| 2.7 ghi dữ liệu + verify | `record_activity` → `get_record`; sự kiện `VERIFY_FORCED`/`VERIFY` trong `logs/` |
| 2.8 MCP server | `src/server.py` (stdio, 6 tool, 1 tool ghi), `src/mcp_client.py` |
| Điểm cộng | metadata filter thật · so sánh dense và hybrid trong `make eval` |

## Xem kết quả

- `logs/*.jsonl`: từng câu có `ACTION`, `TOOL_CALL`, `TOOL_RESULT`, `VERIFY`, `ANSWER`, `USAGE`.
- `results/<engine>-<giờ>/summary.json`: Hit@4, MRR, tỉ lệ đạt, tỉ lệ từ chối đúng, tỉ lệ trích dẫn đúng, token.
  Có nhiều thư mục `results/local-*` vì nhóm chạy nhiều lần trong lúc sửa planner; **thư mục `local-20261017-153735` là bản mới nhất**,
  các bản cũ hơn (`local-20261015-211610`, `local-20261017-153345`) còn câu Q19 sai.
- Eval dùng SQLite tạm, không đụng `state/`.

## Giới hạn

Hệ thống chỉ trả lời dựa trên văn bản quy định CTXH và sổ demo; không biết lịch hoạt động, số chỗ, học phí, điểm chuẩn...
(sẽ trả `no_answer`). Không thay thế phần mềm PSC; không xác minh danh tính sinh viên; chưa xử lý quy tắc trừ ngày (II.2.b)
và chưa phân biệt khóa 2012. Số ngày tối thiểu chỉ có hai mức (4 năm trở lên: 4 ngày; dưới 4 năm: 2 ngày, mục I.3).

Giới hạn của baseline `local`:
- Bộ lọc từ chối dựa vào tỉ lệ từ khóa của câu hỏi có trong bằng chứng (ngưỡng 0,6). Câu "Học phí học kỳ này là bao nhiêu?" (Q19)
  từng bị trả lời thay vì từ chối; hiện đã được xử lý bằng **luật riêng cho cụm "học phí"** và luật "câu hỏi có năm/số cụ thể mà bằng chứng không có"
  trong `coverage()` (`src/planner.py`). Đây là luật viết tay cho những ca đã biết, chưa phải lời giải tổng quát cho mọi câu ngoài phạm vi.
- Engine `api` xử lý việc từ chối chủ yếu bằng prompt.

Giới hạn chung của lớp bảo vệ ghi dữ liệu: việc **cho phép ghi** được quyết định bằng biểu thức chính quy trong `src/planner.py`
(`ghi nhận`, `ghi vào`, `lưu lại`, …) kể cả ở mode api. Cách nói khác ("thêm giúp tôi một bản ghi") sẽ bị chặn dù LLM muốn gọi `record_activity`.
Chặn nhầm an toàn hơn ghi nhầm, nhưng cần biết khi thử câu hỏi mới.
