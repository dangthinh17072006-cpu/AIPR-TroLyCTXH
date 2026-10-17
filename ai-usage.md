# AI Usage Disclosure

## 1. Mục đích sử dụng AI

Dự án **AIPR — Trợ lý Công tác xã hội (CTXH)** có sử dụng công cụ AI để hỗ trợ quá trình phát triển phần mềm, chỉnh sửa mã nguồn, xây dựng bộ kiểm thử và soạn thảo tài liệu.
AI được sử dụng như một công cụ hỗ trợ lập trình và rà soát giải pháp. Nhóm chịu trách nhiệm kiểm tra, tích hợp, chạy thử và đánh giá kết quả trước khi đưa vào sản phẩm cuối cùng.

## 2. Phạm vi sử dụng AI

| File/thư mục | Phạm vi hỗ trợ của AI |
|---|---|
| `src/models.py` | Hỗ trợ xây dựng các kiểu dữ liệu bằng `dataclass`, `TypedDict` và type hints. |
| `src/config.py` | Hỗ trợ tổ chức cấu hình, giới hạn số bước thực thi và các tham số vận hành. |
| `src/activities.py` | Hỗ trợ xử lý bảng mã hoạt động, tính điểm và quy đổi điểm thành ngày CTXH theo văn bản nguồn. |
| `src/chunking.py` | Hỗ trợ thiết kế quy tắc chia tài liệu thành các chunk, giữ nguyên quan hệ giữa mã hoạt động và nội dung bảng. |
| `src/embedding.py`, `src/retrieval.py` | Hỗ trợ xây dựng cơ chế embedding dự phòng, truy xuất dense và hybrid, cùng bộ lọc metadata. |
| `course/ex2_1_hybrid_search.py` | Nhóm sử dụng lại mã nguồn bài thực hành tuần 4; AI hỗ trợ chỉnh sửa một số điểm tích hợp. |
| `src/agent.py`, `src/planner.py` | Hỗ trợ phát triển và rà soát vòng lặp agent, phân tích action, giới hạn số bước và xử lý lỗi. |
| `src/llm.py`, `templates/agent.txt` | Hỗ trợ tích hợp API LLM, tách prompt khỏi mã nguồn, xử lý timeout, retry và ghi nhận usage. |
| `src/server.py`, `src/mcp_client.py` | Hỗ trợ triển khai MCP server/client, các công cụ đọc và ghi dữ liệu, cùng quy trình xác minh sau khi ghi. |
| `src/logging_utils.py` | Hỗ trợ tổ chức log theo từng sự kiện để theo dõi quá trình thực thi. |
| `src/evaluate.py`, `src/main.py`, `run.py`, `Makefile` | Hỗ trợ xây dựng lệnh chạy, demo, đánh giá truy xuất và kiểm thử. |
| `tests/`, `test/`, `tests/testset.jsonl` | Hỗ trợ đề xuất tình huống kiểm thử, câu hỏi đánh giá và các trường hợp lỗi. Kết quả phải được chạy kiểm chứng thực tế. |
| `data/documents.json` | Hỗ trợ tổ chức metadata cho tài liệu phục vụ truy xuất. Metadata phải được đối chiếu với tài liệu nguồn. |

## 3. Khai báo sử dụng AI
KHAI BÁO SỬ DỤNG AI – LÊ QUANG THẮNG

Công cụ sử dụng: ChatGPT/Codex.
Mục đích: hỗ trợ viết code, giải thích cách xử lý và sửa lỗi.
Số dòng dưới đây tính theo phiên bản mã nguồn dùng khi khai báo.

1. FILE: src/retrieval.py

Vị trí: hàm Retriever.search(), dòng 23–40.
Phạm vi hỗ trợ: lọc tài liệu theo doc_id, chọn tìm kiếm Dense
hoặc Hybrid và trả về các đoạn tài liệu đã xếp hạng.

Code tiêu biểu, dòng 35–40:

        if mode == "dense":
            ranked = dense_search(question, vectors, top_k)
        else:
            bm25 = build_bm25_index(chunks, tokenizer=words)
            ranked = hybrid_search(question, chunks, bm25, vectors, top_k=top_k, tokenizer=words)
        return [(chunks[i], float(score)) for i, score in ranked]

Giải thích:
Dense tìm kiếm bằng vector. Hybrid kết hợp tìm kiếm từ khóa
BM25 với vector. Các hàm tìm kiếm được gọi từ
src/ex2_1_hybrid_search.py.


2. FILE: src/embedding.py

Vị trí:
- Hàm fit(), dòng 25–33.
- Hàm embed_text(), dòng 36–41.

Phạm vi hỗ trợ: xây dựng embedding dự phòng bằng TF-IDF
và SVD, sau đó chuẩn hóa vector.

Code hàm fit():

def fit(texts: list[str]):
    """Học không gian vector từ các chunk của corpus."""
    global vocab, idf, basis
    all_words = sorted({w for t in texts for w in words(t)})
    vocab = {w: i for i, w in enumerate(all_words)}
    counts = np.array([count_vector(t) for t in texts])
    idf = np.log((1 + len(texts)) / (1 + (counts > 0).sum(axis=0))) + 1
    _, _, vt = np.linalg.svd(counts * idf, full_matrices=False)
    basis = vt[:64]

Code hàm embed_text():

def embed_text(text: str) -> list[float]:
    if basis is None:
        raise RuntimeError("Cần gọi fit() trước")
    vec = (count_vector(text) * idf) @ basis.T
    norm = np.linalg.norm(vec)
    return (vec / norm if norm > 0 else vec).tolist()

Giải thích:
fit() tạo từ điển, tính trọng số từ và học không gian vector.
embed_text() chuyển văn bản sang vector trong không gian đó.
Đây là phương án dự phòng khi không dùng được embedding
từ tài liệu thực hành của lớp.


3. FILE: tests/test_core.py

Các phần kiểm thử được khai báo có AI hỗ trợ:
- DataTests, dòng 24–42: dữ liệu, quy đổi điểm và metadata.
- RetrievalTests, dòng 45–61: Dense/Hybrid và bộ lọc tài liệu.
- ParserTests, dòng 64–76: JSON và ý định ghi dữ liệu.
- McpAgentTests, dòng 79–163: gọi tool, xác minh bản ghi,
  chống ghi trái yêu cầu và xử lý lỗi.
- LlmAdapterTests, dòng 166–210: mô phỏng HTTP để kiểm tra
  retry, token và việc không ghi API key vào log.
- RepoHygieneTests, dòng 213–219: kiểm tra chuỗi bí mật.

Hình thức hỗ trợ:
Đề xuất trường hợp kiểm thử, viết code kiểm thử và hỗ trợ sửa lỗi.

Code tiêu biểu:
Hàm test_model_cannot_force_unauthorized_write(),
dòng 138–142:

    async def test_model_cannot_force_unauthorized_write(self):
        raw = json.dumps({"tool": "record_activity", "arguments": {"student_id": "11110001", "activity_code": "001202", "points": 5, "event_date": "2026-10-10"}})
        with patch("src.agent.call_llm", return_value=raw), patch("src.agent.call") as invoke:
            await run_agent(None, [{"name": "record_activity"}], "Chỉ cho tôi xem mã 001202", "hybrid", "api", lambda *a, **k: None)
            invoke.assert_not_called()

Giải thích:
Giả lập tình huống LLM yêu cầu ghi dữ liệu dù người dùng
chỉ muốn xem. Kiểm thử xác nhận công cụ không được gọi.
Đây là kiểm thử dùng mô phỏng, không gọi API thật.


4. FILE: src/planner.py

Các phần được khai báo có AI hỗ trợ:
- extract(), dòng 40–56: trích xuất mã hoạt động, MSSV,
  điểm, ngày và ý định ghi từ câu hỏi.
- plan(), dòng 69–102: chọn hành động theo luật.
- coverage(), dòng 115–138: kiểm tra mức trùng từ khóa
  giữa câu hỏi và bằng chứng.
- best_sentences(), dòng 141–152: chọn câu phù hợp
  từ đoạn tài liệu.
- compose(), dòng 159–200: dựng câu trả lời, trích dẫn
  hoặc thông báo không đủ bằng chứng.

Hình thức hỗ trợ:
Đề xuất code, giải thích biểu thức chính quy, xử lý điều kiện
và hỗ trợ sửa lỗi.

Code tiêu biểu trong plan(), dòng 72–78:

    if f.write:
        # thiếu thông tin thì từ chối, không đoán
        needed = [("mã hoạt động 6 số", f.codes), ("mã sinh viên 8 số", f.student_id),
                  ("số điểm", f.points), ("ngày YYYY-MM-DD", f.event_date)]
        missing = [name for name, value in needed if value is None or value == []]
        if missing:
            return Action("finish", {"status": "refused", "answer": "Chưa ghi gì. Còn thiếu: " + ", ".join(missing) + "."})

Giải thích:
Khi người dùng yêu cầu ghi nhận hoạt động, chương trình
kiểm tra đủ mã hoạt động, MSSV, điểm và ngày.
Nếu thiếu thông tin thì trả thông báo từ chối.

Lưu ý về chức năng:
plan() chọn hành động theo luật cho engine local.
Trong engine api, LLM chọn công cụ; agent.py thực hiện
kiểm soát quyền ghi và xác minh bản ghi.


TRÁCH NHIỆM SỬ DỤNG

Sinh viên rà soát, chỉnh sửa và chạy kiểm thử mã nguồn.
Khai báo phản ánh các phần thực tế có AI hỗ trợ.
Nếu có phần khác dùng AI, bổ sung vào khai báo.


KHAI BÁO SỬ DỤNG AI – ĐẶNG TẤN PHÚC THỊNH

Công cụ sử dụng: ChatGPT/Codex.


 1. FILE: src/agent.py

Các phần được khai báo có AI hỗ trợ:
- parse_action(), dòng 20–38: đọc JSON do model trả về thành Action.
- same_record(), dòng 45–51: so sánh bản ghi vừa ghi với bản đọc lại.
- answer_from_finish(), dòng 54–68: lấy câu trả lời từ tool finish
  và loại các nguồn không xuất hiện trong kết quả tool.
- run_agent(), dòng 71–136: vòng lặp agent, giới hạn MAX_STEPS,
  chặn ghi trái yêu cầu, bắt buộc xác minh sau khi ghi.
- finish(), dòng 139–end: dựng câu trả lời cuối phụ thuộc kết quả xác minh.
 
Hình thức hỗ trợ:
Đề xuất cấu trúc vòng lặp, viết code, giải thích cơ chế for...else
và hỗ trợ sửa lỗi.
 
Code tiêu biểu trong run_agent(), dòng 83–86:
 
        if to_verify:  # vừa ghi xong: bắt buộc đọc lại để verify
            record_id = to_verify["record"]["record_id"]
            action = Action("get_record", {"record_id": record_id})
            log("VERIFY_FORCED", step=step, record_id=record_id)
 
Giải thích:
Sau khi record_activity thành công, agent tự gọi get_record thay vì để
model quyết định. Chỉ khi bản đọc lại khớp bản vừa ghi thì mới báo
đã ghi thành công. Dòng 110–111 chặn record_activity nếu người dùng
chưa yêu cầu ghi rõ ràng, kể cả khi LLM đề xuất.
 
 
2. FILE: src/config.py
 
Các phần được khai báo có AI hỗ trợ:
- Hằng số cấu hình, dòng 4–8: ROOT, MAX_STEPS, MCP_TIMEOUT_S,
  LLM_TIMEOUT_S, LLM_RETRIES.
- load_env(), dòng 11–21: đọc file .env.
 
Hình thức hỗ trợ:
Đề xuất cấu trúc cấu hình và viết hàm đọc .env bằng thư viện chuẩn.
 
Code tiêu biểu, dòng 16–20:
 
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
 
Giải thích:
Đọc từng dòng .env, bỏ dòng trống và chú thích, tách khóa/giá trị
tại dấu "=" đầu tiên. Biến môi trường đã đặt sẵn được giữ nguyên
(os.environ.setdefault) nên không ghi đè cấu hình của người dùng.
 
 
3. FILE: src/llm.py
 
Phạm vi hỗ trợ: call_llm(), dòng 14–53: gọi API tương thích OpenAI
bằng thư viện chuẩn, có timeout, thử lại, ghi log lỗi HTTP và
che API key khi in lỗi.
 
Hình thức hỗ trợ:
Đề xuất logic phân loại lỗi để thử lại, viết code và hỗ trợ sửa lỗi.
 
Code tiêu biểu, dòng 47–53:
 
            log("HTTP_ERROR", attempt=attempt, status=status, error=type(exc).__name__)
            # lỗi 4xx (sai key, sai model...) thì thử lại cũng vô ích, trừ 408/409/429
            if status is not None and status < 500 and status not in (408, 409, 429):
                raise RuntimeError(f"HTTP {status}: kiểm tra key/model/base_url") from None
            if attempt == LLM_RETRIES:
                raise RuntimeError(f"API không phản hồi sau {LLM_RETRIES} lần thử") from None
            time.sleep(2 ** (attempt - 1))
 
Giải thích:
Lỗi 4xx (sai key, sai model) dừng ngay vì thử lại vô ích; riêng 408,
409, 429 và lỗi 5xx được thử lại tối đa LLM_RETRIES lần với thời gian
chờ tăng dần (1 giây, 2 giây). Dòng 28 chờ 5 giây trước mỗi lần gọi
để tránh vượt hạn mức. Mọi lỗi đều được ghi vào log sự kiện HTTP_ERROR.
Prompt hệ thống nằm ở templates/agent.txt, không viết cứng trong code.
 
 
4. FILE: src/logging_utils.py
 
Phạm vi hỗ trợ: lớp RunLog, dòng 6–18: ghi log JSONL.
 
Hình thức hỗ trợ:
Đề xuất định dạng log và viết code.
 
Code tiêu biểu, dòng 14–18:
 
    def __call__(self, event: str, **fields):
        row = {"timestamp": datetime.now(timezone.utc).isoformat(),
               "question_id": self.question_id, "event": event, **fields}
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
 
Giải thích:
Mỗi sự kiện là một dòng JSON gồm timestamp UTC, mã câu hỏi, tên sự kiện
và các trường kèm theo. Mở file ở chế độ nối thêm nên không mất log
cũ; ensure_ascii=False giữ nguyên tiếng Việt.
 
 
5. FILE: src/main.py
 
Các phần được khai báo có AI hỗ trợ:
- pick_engine(), dòng 19–25: chọn engine local hoặc api.
- demo(), dòng 39–67: chạy demo và chat tương tác.
- main(), dòng 70–85: đọc tham số dòng lệnh và gọi demo hoặc eval.
 
Hình thức hỗ trợ:
Đề xuất cấu trúc lệnh, viết code argparse và vòng lặp chat.
 
Code tiêu biểu trong pick_engine(), dòng 23–24:
 
    if engine == "api" and not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("API mode cần OPENAI_API_KEY trong .env (copy từ .env.example). Không tự chuyển về local.")
 
Giải thích:
Nếu chọn engine api mà thiếu API key thì dừng và báo lỗi rõ ràng,
không âm thầm chuyển sang engine local. Nhờ vậy kết quả chạy api luôn
là kết quả của LLM thật.
 
 
6. FILE: src/mcp_client.py
 
Các phần được khai báo có AI hỗ trợ:
- connect(), dòng 13–26: mở kết nối MCP qua stdio, khởi tạo phiên
  và lấy danh sách tool từ server.
- call(), dòng 29–38: gọi tool, không để lỗi làm dừng chương trình.
 
Hình thức hỗ trợ:
Đề xuất cách dùng thư viện mcp, viết code và hỗ trợ sửa lỗi.
 
Code tiêu biểu, dòng 31–38:
 
    try:
        result = await asyncio.wait_for(session.call_tool(name, arguments), timeout=MCP_TIMEOUT_S)
        text = "\n".join(c.text for c in result.content if c.type == "text")
        if result.isError and not text.startswith("Error:"):
            text = "Error: " + text
        return text
    except Exception as exc:
        return f"Error: MCP {type(exc).__name__}: {exc}"[:300]
 
Giải thích:
Gọi tool có thời hạn MCP_TIMEOUT_S. Mọi lỗi (quá thời gian, tool lỗi,
mất kết nối) đều được đổi thành chuỗi bắt đầu bằng "Error:" để agent
đọc và xử lý, không làm sập chương trình. Danh sách tool lấy động
bằng list_tools nên client không khai báo cứng tên tool.
 
 
7. FILE: src/server.py
 
Các phần được khai báo có AI hỗ trợ:
- open_db(), dòng 32–40: mở SQLite, đường dẫn lấy từ biến AIPR_DB.
- search_regulations(), dòng 47–65: tìm kiếm quy định (RAG).
- lookup_activity(), dòng 68–87: tra mã hoạt động.
- points_to_days(), dòng 90–99: đổi tổng điểm ra ngày.
- record_activity(), dòng 102–141: ghi hoạt động vào SQLite.
- get_record(), dòng 144–153: đọc lại bản ghi để xác minh.
- get_student_summary(), dòng 156–175: tổng hợp tiến độ sinh viên.
 
Hình thức hỗ trợ:
Đề xuất danh sách tool, viết code FastMCP và SQLite, viết mô tả tool
(docstring) và hỗ trợ sửa lỗi.
 
Code tiêu biểu trong record_activity(), dòng 127–139:
 
    con = open_db()
    try:
        row = con.execute("SELECT * FROM records WHERE student_id=? AND activity_code=? AND event_date=?",
                          (student_id, activity_code, event_date)).fetchone()
        if row:  # đã có thì không ghi trùng
            return to_json({"record": dict(row), "created": False})
        record = {"record_id": "REC-" + uuid.uuid4().hex[:10], ...}
        con.execute("INSERT INTO records VALUES (:record_id, ...)", record)
        con.commit()
        return to_json({"record": record, "created": True})
 
Giải thích:
Trước khi ghi, hàm kiểm tra đã có bản ghi cùng (sinh viên, mã, ngày)
chưa; nếu có thì trả lại bản cũ với created=False, không tạo bản trùng.
Trước đoạn này, hàm đã từ chối mã sai, điểm ngoài khoảng, ngày tương lai
và hoạt động đã được trả tiền công. Truy vấn dùng tham số (?) nên không
bị SQL injection. Server không dùng print vì stdout dành cho giao thức MCP.
 
 
8. FILE: src/evaluate.py
 
Các phần được khai báo có AI hỗ trợ:
- retrieval_eval(), dòng 36–46: tính Hit@4 và MRR.
- judge(), dòng 49–64: chấm đạt/không đạt và trích dẫn đúng/sai.
- run_eval(), dòng 67–end: chạy so sánh dense và hybrid, nhiều lượt,
  dùng SQLite tạm, ghi summary.json.
 
Hình thức hỗ trợ:
Đề xuất cách đo, viết code đánh giá và hỗ trợ sửa lỗi.
 
Code tiêu biểu trong judge(), dòng 52–59:
 
    if it["no_answer"]:
        return answer.status == "no_answer", None
    if it.get("expect_refusal"):
        return answer.status == "refused" and not wrote and contains_all(answer.text, it["must_contain"]), None
    if it.get("expect_write") is False and wrote:
        return False, None
    if it.get("expect_write"):
        return wrote and answer.verified is True and contains_all(answer.text, it["must_contain"]), None
 
Giải thích:
Mỗi loại câu hỏi có tiêu chí chấm riêng: câu không có đáp án phải bị
từ chối đúng; câu cấm ghi mà vẫn ghi thì sai; câu cần ghi phải ghi và
xác minh được. Câu thường phải có đủ từ khóa trong must_contain.
Eval dùng DB tạm nên không làm thay đổi dữ liệu demo trong state/.
 
 
9. FILE: templates/agent.txt
 
Phạm vi hỗ trợ: toàn bộ prompt hệ thống (29 dòng) cho engine api:
định dạng đầu ra JSON, quy tắc chọn tool, điều kiện cho phép ghi,
quy tắc trả lời khi thiếu bằng chứng và chống prompt injection.
 
Hình thức hỗ trợ:
Đề xuất nội dung prompt và hỗ trợ chỉnh sửa sau khi đọc log chạy API.
 
 
10. FILE: run.py
 
Phạm vi hỗ trợ: dòng 8–40: tạo môi trường ảo, cài thư viện và chạy
các lệnh setup, demo, chat, eval, test trên Windows, macOS, Linux.
 
Hình thức hỗ trợ:
Viết code và hỗ trợ kiểm tra khác biệt giữa các hệ điều hành.
 
Code tiêu biểu, dòng 9 và 21–24:
 
PY = ROOT / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    ...
    if action in ("setup", "setup-demo"):
        if not PY.exists():
            venv.EnvBuilder(with_pip=True).create(ROOT / ".venv")
        run(PY, "-m", "pip", "install", "-q", "-r", "requirements.txt")
 
Giải thích:
Đường dẫn Python trong môi trường ảo khác nhau giữa Windows (Scripts)
và Linux/macOS (bin) nên được chọn theo os.name. Lệnh setup tạo .venv
nếu chưa có rồi cài thư viện từ requirements.txt; các lệnh còn lại
chạy bằng chính Python của .venv.