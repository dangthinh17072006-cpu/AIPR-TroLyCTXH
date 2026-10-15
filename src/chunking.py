import json
from pathlib import Path

from .models import Chunk, Document, Source
def split_into_chunks(text: str, words_per_chunk: int = 100) -> list[str]:
    """Chia văn bản theo dòng để không cắt ngang một dòng bảng."""
    if words_per_chunk <= 0:
        raise ValueError("words_per_chunk phải lớn hơn 0")

    chunks = []
    current_lines = []
    current_words = 0

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue

        line_words = len(line.split())

        if current_lines and current_words + line_words > words_per_chunk:
            chunks.append("\n".join(current_lines))
            current_lines = []
            current_words = 0

        current_lines.append(line)
        current_words += line_words

    if current_lines:
        chunks.append("\n".join(current_lines))

    return chunks
def load_chunks() -> list[Chunk]:
    data_dir = Path(__file__).resolve().parent.parent / "data"
    records = json.loads(
        (data_dir / "documents.json").read_text(encoding="utf-8")
    )

    chunks = []
    for record in records:
        document = Document(**record)
        text = (data_dir / document.file).read_text(encoding="utf-8")

        for number, part in enumerate(split_into_chunks(text), start=1):
            source = Source(document.doc_id, f"chunk-{number}", document.version)
            chunks.append(Chunk(source, part, document))

    return chunks