"""Chia văn bản CTXH.md thành chunk (lý do chọn cách chia: xem README).

- Mỗi dòng của bảng mã hóa là 1 chunk.
- Văn xuôi: gom các đoạn cùng tiêu đề, tối đa MAX_WORDS từ, không cắt giữa đoạn.
- Chunk văn xuôi bắt đầu bằng đường dẫn tiêu đề trong [ngoặc vuông] để đứng riêng vẫn hiểu được.
"""
import json
import re
from pathlib import Path

from .activities import fmt_range, parse_table
from .models import Chunk, Document, Source

DATA = Path(__file__).resolve().parent.parent / "data"
MAX_WORDS = 110
ROMAN = re.compile(r"^(I|II|III|IV|V)\.\s")
NUMBERED = re.compile(r"^\d+\.\s")
LETTERED = re.compile(r"^[a-g]\.\s")


def clean(text: str) -> str:
    text = text.replace("*", "").replace("✓", "").lstrip("#")
    return " ".join(text.split())


def split_long(text: str) -> list[str]:
    """Đoạn dài quá MAX_WORDS thì cắt ở cuối câu hoặc dấu chấm phẩy."""
    if len(text.split()) <= MAX_WORDS:
        return [text]
    pieces, current = [], ""
    for part in re.split(r"(?<=[.;])\s+", text):
        if current and len((current + " " + part).split()) > MAX_WORDS:
            pieces.append(current)
            current = part
        else:
            current = (current + " " + part).strip()
    if current:
        pieces.append(current)
    return pieces


def read_paragraphs(md: str) -> list[tuple[str, str, str]]:
    """Trả về danh sách (mục lớn I-V, đường dẫn tiêu đề, nội dung) của từng đoạn văn xuôi."""
    section, h1, h2, h3 = "I", "", "", ""
    result = []
    for line in md.splitlines():
        line = line.strip()
        if not line or line.startswith("|"):
            continue
        if line.startswith("- "):
            line = line[2:]

        # tách tiêu đề và phần nội dung đi kèm (nếu có)
        bold = re.match(r"^\*\*(.+?)\*\*(.*)$", line)
        if bold:
            head, rest = clean(bold.group(1)), clean(bold.group(2))
        elif line.startswith("#"):
            head, rest = clean(line), ""
        else:
            head, rest = "", clean(line)

        if ROMAN.match(head):
            section, h1, h2, h3 = head.split(".")[0], head, "", ""
        elif LETTERED.match(head):
            h3 = head
        elif NUMBERED.match(head) or line.startswith("#"):
            h2, h3 = head, ""
        elif head:  # in đậm nhưng không phải tiêu đề: coi như nội dung
            rest = clean(line)

        if rest:
            path = " > ".join(x for x in (h1, h2, h3) if x)
            result.append((section, path, rest))
    return result


def chunk_markdown(md: str, docs: list[Document]) -> list[Chunk]:
    doc_of = {d.section: d for d in docs}
    chunks = []
    counter: dict[str, int] = {}

    def add(section: str, text: str, kind: str = "prose", code: str | None = None):
        doc = doc_of[section]
        counter[doc.doc_id] = counter.get(doc.doc_id, 0) + 1
        chunk_id = f"{doc.doc_id}:v{doc.version}:{counter[doc.doc_id]}"
        chunks.append(Chunk(Source(doc.doc_id, chunk_id, doc.version), text, doc, kind, code))

    # 1) văn xuôi: gom các đoạn liên tiếp có cùng mục và cùng tiêu đề
    groups: list[tuple[str, str, list[str]]] = []
    for section, path, content in read_paragraphs(md):
        for piece in split_long(content):
            if groups and groups[-1][:2] == (section, path) \
                    and len(" ".join(groups[-1][2] + [piece]).split()) <= MAX_WORDS:
                groups[-1][2].append(piece)
            else:
                groups.append((section, path, [piece]))
    for section, path, pieces in groups:
        body = " ".join(pieces)
        add(section, f"[{path}] {body}" if path else body)

    # 2) mỗi dòng của bảng là 1 chunk, giữ nguyên mã và điểm/ngày
    for a in parse_table(md):
        text = (f"{a.category} — mã {a.code}: {a.description} "
                f"Thang điểm: {fmt_range(a.points_min, a.points_max)}. "
                f"Quy đổi: {fmt_range(a.days_min, a.days_max)} ngày CTXH.")
        add("II-BANG", text, "table_row", a.code)
    return chunks


def load_documents() -> list[Document]:
    records = json.loads((DATA / "documents.json").read_text(encoding="utf-8"))
    return [Document(**r) for r in records]


def load_chunks() -> list[Chunk]:
    docs = load_documents()
    md = (DATA / docs[0].file).read_text(encoding="utf-8")
    return chunk_markdown(md, docs)
