"""Retriever: dense và hybrid (BM25 + vector), dùng lại hàm của w4 trong src/ex2_1_hybrid_search.py."""
from . import ex2_1_hybrid_search as w4
from .ex2_1_hybrid_search import build_bm25_index, dense_search, hybrid_search

from . import embedding
from .chunking import load_chunks
from .embedding import words
from .models import Chunk


class Retriever:
    def __init__(self):
        self.chunks = load_chunks()
        texts = [c.text for c in self.chunks]
        if w4.EMBEDDING_BACKEND.startswith("LSA"):  # bản dự phòng cần học từ corpus trước
            embedding.fit(texts)
        self.vectors = [w4.embed_text(t) for t in texts]

    @property
    def backend(self) -> str:
        return w4.EMBEDDING_BACKEND

    def search(self, question: str, mode: str = "hybrid", top_k: int = 4,
               doc_id: str | None = None) -> list[tuple[Chunk, float]]:
        if mode not in ("dense", "hybrid"):
            raise ValueError("mode phải là 'dense' hoặc 'hybrid'")

        # metadata filter: lọc theo doc_id TRƯỚC khi xếp hạng
        ids = [i for i, c in enumerate(self.chunks) if doc_id is None or c.source.doc_id == doc_id]
        if not ids:
            raise ValueError(f"doc_id không tồn tại: {doc_id}")
        chunks = [self.chunks[i] for i in ids]
        vectors = [self.vectors[i] for i in ids]

        if mode == "dense":
            ranked = dense_search(question, vectors, top_k)
        else:
            bm25 = build_bm25_index(chunks, tokenizer=words)
            ranked = hybrid_search(question, chunks, bm25, vectors, top_k=top_k, tokenizer=words)
        return [(chunks[i], float(score)) for i, score in ranked]
