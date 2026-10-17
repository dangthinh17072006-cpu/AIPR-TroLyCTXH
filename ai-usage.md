## Phần do AI sinh/sửa (theo file)
| File | Mức độ AI | Ghi chú |
|---|---|---|
| `src/models.py`, `src/config.py`, `src/logging_utils.py` | AI sinh/sửa | dataclass, TypedDict |
| `src/activities.py` | AI sinh/sửa | parse bảng II.2.a, luật làm tròn II.2.c |
| `src/chunking.py` | AI sinh/sửa, thay bản chunk theo dòng của nhóm | dòng bảng + gom đoạn theo tiêu đề |
| `src/embedding.py`, `src/retrieval.py` | AI sinh/sửa | embedding dự phòng; metadata filter trước khi xếp hạng |
| `course/ex2_1_hybrid_search.py` | **Nhóm viết (w4)**, AI chỉnh 3 chỗ | import dự phòng, tham số `tokenizer`, `dense_search` |
| `src/server.py`, `src/mcp_client.py` | AI sinh/sửa | 6 tool, SQLite, stdio |
| `src/agent.py`, `src/planner.py`, `src/llm.py`, `templates/agent.txt` | AI sinh/sửa | vòng lặp, guard ghi, verify, retry/timeout, prompt |
| `src/evaluate.py`, `src/main.py`, `run.py`, `Makefile` | AI sinh/sửa | |
| `tests/test_core.py`, `tests/testset.jsonl` | AI sinh/sửa | câu hỏi và đáp án viết dựa trên `CTXH.md` |
| `README.md`, `ai-usage.md` | AI sinh/sửa | |
| `data/CTXH.md` | Văn bản gốc của trường | không do AI viết |

## Cách kiểm chứng
`make test` (20 test: parser, chunking, retrieval, MCP thật qua stdio, ghi + verify, idempotent, guard ghi, retry HTTP bằng server giả);
`make eval` (dense vs hybrid). Engine `local` **không dùng LLM**; log `logs/eval-local-*` và `logs/demo-local-*` có `calls=0`, token=0.
Log LLM thật phải do nhóm tạo bằng API key riêng (xem README).


