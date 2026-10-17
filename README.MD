# AIPR — Trợ lý Công tác xã hội (CTXH)

**Nhóm:** Lê Quang Thắng — Đặng Tấn Phúc Thịnh · HCMC University of Technology and Engineering
**Dữ liệu:** `data/CTXH.md` — *Quy định tổ chức thực hiện Chương trình CTXH đối với SV hệ chính quy* (văn bản thật của trường).
Sổ ghi nhận hoạt động (`state/ctxh.sqlite3`) chỉ là **DEMO**, không phải hệ thống PSC của trường; mã sinh viên `11110001` chỉ để thử.

## Chạy

```bash
make setup && make demo          # Linux/macOS/Git-Bash (cần Python 3.11+)
python run.py setup-demo         # Windows
```

Mặc định `AIPR_MODE=local` nên demo không cần API key.

| Việc | Lệnh | Windows |
|---|---|---|
| Demo 3 câu (tra mã, RAG, ghi + verify) | `make demo` | `python run.py demo` |
| Hỏi tự do | `make chat` | `python run.py chat` |
| So sánh dense và hybrid | `make eval` | `python run.py eval` |
| Kiểm thử | `make test` | `python run.py test` |

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

Sau đó chạy `make demo` và `make eval` (mode api mặc định lặp 3 lượt, mỗi lượt có 1 file `logs/*.jsonl` kèm dòng USAGE).
API mode không tự chuyển về local. **Engine `local` chỉ là bộ chọn hành động theo luật (baseline), không phải LLM.**

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

- Embedding: nếu import được `Vector_Store` của lớp (`../Week_4` hoặc biến `W4_PATH`) thì dùng embedding đó; nếu không dùng `src/embedding.py` (TF-IDF + SVD, chạy offline) để repo chạy được bằng 1 lệnh.
- **Metadata filter:** tham số `doc_id` của `search_regulations` lọc chunk *trước* khi xếp hạng. Prompt `templates/agent.txt` hướng dẫn LLM dùng nó.

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
- Eval dùng SQLite tạm, không đụng `state/`.

## Giới hạn

Hệ thống chỉ trả lời dựa trên văn bản quy định CTXH và sổ demo; không biết lịch hoạt động, số chỗ, học phí, điểm chuẩn...
(sẽ trả `no_answer`). Không thay thế phần mềm PSC; không xác minh danh tính sinh viên; chưa xử lý quy tắc trừ ngày (II.2.b)
và chưa phân biệt khóa 2012.

Giới hạn đã biết của baseline `local`: câu "Học phí học kỳ này là bao nhiêu?" (Q19) vẫn bị trả lời thay vì từ chối, vì
bộ lọc từ chối chỉ dựa vào tỉ lệ từ khóa trùng. Engine `api` xử lý việc này bằng prompt.

## Trước khi nộp

- [ ] Chạy API thật ít nhất 3 lượt (`AIPR_MODE=api`), đọc log, cập nhật số liệu trong report.
- [ ] Thu hồi (revoke) API key OpenAI từng nằm trong `.env.example` của bản zip cũ, tạo key mới; không đưa key vào git.
- [ ] Commit thật từng người bằng tài khoản của mình.
- [ ] Viết `report.docx` theo khung của đề.
- [ ] Điền `ai-usage.md` bằng phần từng người đã đọc, sửa, giải thích được.
- [ ] Thử `make setup && make demo` từ một bản giải nén mới.
