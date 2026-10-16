"""Engine 'local': chọn hành động theo luật (không dùng LLM) và dựng câu trả lời từ kết quả tool.

Đây chỉ là baseline để chạy được khi chưa có API key, không phải chất lượng của LLM agent.
Khi dùng engine 'api', LLM chọn tool theo templates/agent.txt; agent.py vẫn giữ guard ghi và bước verify.
"""
import json
import re
from dataclasses import dataclass, field
from datetime import date

from .activities import fmt, fmt_range
from .embedding import words
from .models import Action, Answer, Observation, Source

CODE = re.compile(r"(?<!\d)\d{6}(?!\d)")        # mã hoạt động 6 số
STUDENT = re.compile(r"(?<!\d)\d{8}(?!\d)")     # mã sinh viên 8 số
POINTS_AFTER = re.compile(r"(\d+(?:[.,]\d+)?)\s*điểm", re.I)
POINTS_BEFORE = re.compile(r"điểm[^\d]{0,25}?(\d+(?:[.,]\d+)?)", re.I)
ISO_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")
# "đăng ký" đứng một mình thường là danh từ ("còn bao nhiêu chỗ đăng ký") nên không tính là yêu cầu ghi
WRITE = re.compile(r"\b(ghi nhận|ghi vào|ghi lại|ghi giúp|ghi hộ|ghi cho|lưu lại|lưu giúp|lưu vào|đăng ký (?:cho|giúp|hộ))\b", re.I)
NEGATED = re.compile(r"(không|đừng|chớ|chưa|khỏi)\s+(cần\s+|được\s+)?(ghi|đăng ký|lưu)|chỉ\s+(xem|tính|tra|hỏi)", re.I)
PAID = re.compile(r"(được|có|nhận)\s+(trả|chi)\s+(tiền công|thù lao|tiền|phí)|đã được trả", re.I)
STOP = set("""là có được không bao nhiêu mấy của và các cho khi nào thì bị ở tại trong sau bằng một những này đó gì hay
với để như nếu thế sinh viên sv ctxh công tác xã hội tính theo quy định ạ nhé""".split())
COVERAGE_MIN = 0.6


@dataclass
class Fields:
    codes: list[str] = field(default_factory=list)
    student_id: str | None = None
    points: float | None = None
    event_date: str | None = None
    paid: bool = False
    write: bool = False
    total_points: float | None = None  # câu hỏi đổi tổng điểm ra ngày


def extract(question: str) -> Fields:
    f = Fields()
    f.codes = list(dict.fromkeys(CODE.findall(question)))
    if m := STUDENT.search(question):
        f.student_id = m.group(0)
    if m := ISO_DATE.search(question):
        try:
            f.event_date = date.fromisoformat(m.group(0)).isoformat()
        except ValueError:
            pass
    if m := POINTS_AFTER.search(question) or POINTS_BEFORE.search(question):
        f.points = float(m.group(1).replace(",", "."))
    f.paid = bool(PAID.search(question))
    f.write = bool(WRITE.search(question)) and not NEGATED.search(question)
    if not f.codes and f.points is not None and re.search(r"quy đổi|bao nhiêu ngày|mấy ngày", question, re.I):
        f.total_points = f.points
    return f


def explicit_write_request(question: str) -> bool:
    return extract(question).write


# ---------- chọn hành động ----------

def last(obs: list[Observation], tool: str) -> Observation | None:
    return next((o for o in reversed(obs) if o["tool"] == tool), None)


def plan(question: str, obs: list[Observation], mode: str) -> Action:
    f = extract(question)

    if f.write:
        # thiếu thông tin thì từ chối, không đoán
        needed = [("mã hoạt động 6 số", f.codes), ("mã sinh viên 8 số", f.student_id),
                  ("số điểm", f.points), ("ngày YYYY-MM-DD", f.event_date)]
        missing = [name for name, value in needed if value is None or value == []]
        if missing:
            return Action("finish", {"status": "refused", "answer": "Chưa ghi gì. Còn thiếu: " + ", ".join(missing) + "."})
        lookup = last(obs, "lookup_activity")
        if lookup is None:
            return Action("lookup_activity", {"activity_codes": [f.codes[0]]})
        if lookup["result"].startswith("Error:"):
            return Action("finish")
        write = last(obs, "record_activity")
        if write is None:
            return Action("record_activity", {"student_id": f.student_id, "activity_code": f.codes[0],
                                              "points": f.points, "event_date": f.event_date, "was_paid": f.paid})
        if not write["result"].startswith("Error:") and last(obs, "get_student_summary") is None:
            return Action("get_student_summary", {"student_id": f.student_id})
        return Action("finish")

    if f.codes:
        if last(obs, "lookup_activity"):
            return Action("finish")
        return Action("lookup_activity", {"activity_codes": f.codes})
    if f.total_points is not None:
        if last(obs, "points_to_days"):
            return Action("finish")
        return Action("points_to_days", {"total_points": f.total_points})
    if last(obs, "search_regulations"):
        return Action("finish")
    return Action("search_regulations", {"query": question, "mode": mode})


# ---------- dựng câu trả lời ----------
def parse_json(result: str) -> dict | None:
    try:
        value = json.loads(result)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None



def coverage(question: str, texts: list[str]) -> float:
    """Kiểm tra từ khóa của câu hỏi có trong bằng chứng hay không."""
    question_words = words(question)
    evidence_words = words(" ".join(texts))
    pool = set(evidence_words)

    # Câu hỏi về học phí phải có bằng chứng chứa đúng cụm "học phí".
    question_text = " ".join(question_words)
    evidence_text = " ".join(evidence_words)

    if "học phí" in question_text and "học phí" not in evidence_text:
        return 0.0

    # Không trả lời nếu câu hỏi chứa năm hoặc số cụ thể không có trong bằng chứng.
    for word in question_words:
        if len(word) >= 4 and word.isdigit() and word not in pool:
            return 0.0

    keywords = [word for word in question_words if word not in STOP]
    if not keywords:
        return 0.0

    matched = sum(word in pool for word in keywords)
    return matched / len(keywords)


def best_sentences(text: str, question: str, n: int = 2) -> str:
    """Lấy n câu có nhiều từ trùng với câu hỏi nhất. Câu kết thúc bằng ';' là một vế của liệt kê
    nên lấy thêm câu liền sau nó. Các câu giữ nguyên thứ tự gốc."""
    body = re.sub(r"^\[.*?\]\s*", "", text)  # bỏ đường dẫn tiêu đề ở đầu chunk
    sentences = [s.strip() for s in re.split(r"(?<=[.;])\s+", body) if s.strip()]
    q = {w for w in words(question) if w not in STOP}
    ranked = sorted(range(len(sentences)), key=lambda i: -len(q & set(words(sentences[i]))))
    chosen = set(ranked[:n])
    for i in list(chosen):
        if sentences[i].endswith(";") and i + 1 < len(sentences):
            chosen.add(i + 1)
    return " ".join(sentences[i] for i in sorted(chosen))


def error_text(result: str) -> str:
    return result.removeprefix("Error:").strip()


def compose(question: str, obs: list[Observation]) -> Answer:
    steps = len(obs)

    if write := last(obs, "record_activity"):
        if not parse_json(write["result"]):
            return Answer("Không ghi được: " + error_text(write["result"]), status="refused", steps=steps)
        return Answer("", steps=steps)  # câu chữ cuối do agent dựng sau khi verify

    if lookup := last(obs, "lookup_activity"):
        data = parse_json(lookup["result"])
        if not data:
            return Answer("Không tra được: " + error_text(lookup["result"]), status="no_answer", steps=steps)
        parts = []
        sources = [Source("CTXH-II-BANG", a["chunk_id"], 1) for a in data["activities"] if a.get("chunk_id")]
        for a in data["activities"]:
            parts.append(f"Mã {a['code']} ({a['category']}): {a['description']} — thang điểm "
                         f"{fmt_range(a['points_min'], a['points_max'])}, quy đổi "
                         f"{fmt_range(a['days_min'], a['days_max'])} ngày CTXH.")
        if len(data["activities"]) > 1:
            tp, td = data["total_points"], data["total_days"]
            parts.append(f"Tổng: {fmt_range(tp['min'], tp['max'])} điểm, quy đổi "
                         f"{fmt_range(td['min'], td['max'])} ngày CTXH (làm tròn theo mục II.2.c).")
        return Answer("\n".join(parts), sources, steps=steps)

    if conv := last(obs, "points_to_days"):
        data = parse_json(conv["result"])
        if not data:
            return Answer("Không quy đổi được: " + error_text(conv["result"]), status="no_answer", steps=steps)
        text = (f"{fmt(data['total_points'])} điểm quy đổi thành {fmt(data['days'])} ngày CTXH "
                f"(chia 10; phần dư dưới 5 điểm bỏ, đúng 5 điểm tính 0,5 ngày, trên 5 điểm tính 1 ngày — mục II.2.c).")
        sources = [Source("CTXH-II", data["chunk_id"], 1)] if data.get("chunk_id") else []
        return Answer(text, sources, steps=steps)

    if search := last(obs, "search_regulations"):
        hits = (parse_json(search["result"]) or {}).get("results", [])
        if not hits or coverage(question, [h["text"] for h in hits[:3]]) < COVERAGE_MIN:
            return Answer("Tài liệu quy định CTXH được cung cấp không có thông tin này nên tôi không trả lời.",
                          status="no_answer", steps=steps)
        sources = [Source(h["doc_id"], h["chunk_id"], h["version"]) for h in hits[:2]]
        return Answer("Theo quy định CTXH: " + best_sentences(hits[0]["text"], question), sources, steps=steps)

    return Answer("Không đủ bằng chứng để trả lời.", status="no_answer", steps=steps)
