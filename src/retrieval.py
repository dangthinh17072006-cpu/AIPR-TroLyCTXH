from course.ex2_1_hybrid_search import (
    build_bm25_index,
    dense_search,
    hybrid_search,
)
from .chunking import load_chunks
from .embedding import embed_text, fit, words
from .models import Chunk


class Retriever:
    def __init__(self):
        self.chunks = load_chunks()
        fit([chunk.text for chunk in self.chunks])
        self.vectors = [embed_text(chunk.text) for chunk in self.chunks]
        self.bm25 = build_bm25_index(self.chunks, tokenizer=words)

    def search(
        self, question: str, mode: str = "hybrid", top_k: int = 3
    ) -> list[tuple[Chunk, float]]:
        if mode == "dense":
            results = dense_search(question, self.vectors, top_k)
        elif mode == "hybrid":
            results = hybrid_search(
                question, self.chunks, self.bm25, self.vectors,
                top_k=top_k, tokenizer=words,
            )
        else:
            raise ValueError("mode phải là 'dense' hoặc 'hybrid'")

        return [(self.chunks[index], score) for index, score in results]