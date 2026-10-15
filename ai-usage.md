# Khai báo sử dụng AI

## Công cụ và phạm vi
ChatGPT / Codex hỗ trợ thiết kế, sinh code Python, tạo corpus giả lập, testset,
viết kiểm thử, chạy kiểm thử, dựng báo cáo Word, README và hướng dẫn bảo vệ.
Phần lớn bản khởi tạo trong src/, tests/, run.py và docs/build_report.py do AI sinh.
Hai file course/original/*.py và đoạn code slide có nguồn từ tài liệu lớp do người dùng cung cấp.
course/hybrid_from_lecture.py là bản chuyển đoạn code slide thành hàm dùng trong project.
Chưa được cung cấp bản gốc ex2_1_hybrid_search.py; không khai là đã tái dùng file đó.

## Prompt chính
“Đọc kỹ yêu cầu đồ án cuối kỳ AIPR, làm project đơn giản dễ hiểu chạy trên VS Code,
nhóm Lê Quang Thắng và Đặng Tấn Phúc Thịnh; có RAG, agent, MCP, logs, testset,
README, ai-usage và report Word 8–9 trang có khung, mục lục, hình minh họa.”

## Cách kiểm chứng
Chạy MCP stdio thật, tạo và đọc lại SQLite; chạy unit/integration test và eval 3 lượt.
Kết quả local không dùng LLM, vì chưa có API key. Test API mô phỏng không được coi là log API thật.
AI không tạo lịch sử commit giả dưới tên sinh viên và không tự nhận phần code là do sinh viên viết.

## Phần nào do AI sinh (theo file)
| File | Mức độ AI | Ghi chú |
|---|---|---|
| src/models.py, config.py, logging_utils.py | AI sinh | Nhóm cần đọc và giải thích được |
| src/chunking.py, embedding.py, textutil.py, retrieval.py, evaluate.py | AI sinh, 14/10/2026 | Nối `ex2_1.py` của nhóm, chunking theo câu, chấm trích dẫn theo chunk |
| course/ex2_1_hybrid_search.py | Nhóm viết (`ex2_1.py`), AI chỉnh 3 chỗ | Import fallback, tham số `tokenizer`, type hint; thêm `dense_search` |
| src/agent.py, llm.py, templates/agent.txt | AI sinh | Vòng lặp, MAX_STEPS, guard ghi/verify |
| src/server.py, mcp_client.py | AI sinh | MCP stdio, SQLite |
| src/check.py, Makefile (target check), README (chunking, bảng đối chiếu) | AI sinh, 14/10/2026 | Thêm theo yêu cầu cập nhật code |
| data/documents.json, tests/testset.jsonl | AI sinh | Dữ liệu giả lập |
| report.docx (docs/build_report.py) | AI sinh | Nhóm phải đọc và sửa lại nội dung |

## Nhóm cần cập nhật trước khi nộp
Từng thành viên ghi rõ phần đã đọc, giải thích được, sửa và kiểm thử; thêm commit thật tương ứng.
Các mục phân công hiện là kế hoạch, chưa phải chứng nhận đóng góp.
Nếu dùng thêm AI để sửa, ghi ngày, prompt chính, file liên quan và cách kiểm chứng ở dưới.

| Ngày | Người thực hiện | Công cụ | Phần thay đổi | Kiểm chứng và commit |
|---|---|---|---|---|
| Chờ nhóm điền | | | | |
