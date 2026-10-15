"""w4/ex2_1 của nhóm (file ex2_1.py), tái dùng nguyên logic hybrid BM25 + vector.

Khác bản gốc đúng 3 chỗ, đều giữ tương thích ngược:
1. Import Vector_Store của lớp (thư mục Week_4, đổi bằng biến môi trường W4_PATH);
   nếu không có thì dùng src/embedding.py (LSA offline) để repo vẫn chạy 1 lệnh.
2. build_bm25_index / hybrid_search nhận thêm tham số `tokenizer` (mặc định vẫn là
   lower().split()) để BM25 xử lý được tiếng Việt có dấu và dấu câu.
3. Sửa type hint trả về cho đúng với thực tế: list[tuple[int, float]].
"""
import os
import sys
from typing import Callable

import numpy as np
from rank_bm25 import BM25Okapi

sys.path.append(os.environ.get('W4_PATH', r'../Week_4'))
try:
    from Vector_Store import Document, DocumentChunk, embed_text
    EMBEDDING_BACKEND = 'Vector_Store (w4)'
except ImportError:
    from src.embedding import embed_text
    EMBEDDING_BACKEND = 'LSA offline (src/embedding.py)'


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Calculate cosine similarity between two embedding vectors."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denominator) if denominator else 0.0


def normalise_scores(scores: list[float]) -> list[float]:
    """Normalise scores to [0,1] using min-max scaling"""
    if not len(scores):
        return []
    min_s = min(scores)
    max_s = max(scores)
    if max_s == min_s:
        return [1.0] * len(scores)
    return [(s - min_s) / (max_s - min_s) for s in scores]


def _whitespace(text: str) -> list[str]:
    return text.lower().split()


def build_bm25_index(chunks: list, tokenizer: Callable[[str], list[str]] = _whitespace) -> BM25Okapi:
    """Build a BM25 index from chunk content."""
    return BM25Okapi([tokenizer(chunk.text) for chunk in chunks])


def dense_search(
    query: str,
    embeddings: list[list[float]],
    top_k: int = 3,
) -> list[tuple[int, float]]:
    """Chế độ 1 - dense: chỉ cosine giữa embedding truy vấn và embedding chunk."""
    query_vector = embed_text(query)
    scores = [cosine_similarity(query_vector, v) for v in embeddings]
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    return ranked[:top_k]


def hybrid_search(
    query: str,
    chunks: list,
    bm25_index: BM25Okapi,
    embeddings: list[list[float]],
    top_k: int = 3,
    w_bm25: float = 0.3,
    w_vector: float = 0.7,
    tokenizer: Callable[[str], list[str]] = _whitespace,
) -> list[tuple[int, float]]:
    """
    Chế độ 2 - hybrid: 0.3 * BM25 (keyword) + 0.7 * vector (semantic), sau min-max.
    Returns: list of (chunk_index, combined_score) tuples sorted by combined_score descending.
    """
    bm25_scores = bm25_index.get_scores(tokenizer(query))
    norm_bm25 = normalise_scores(bm25_scores)

    query_vector = np.asarray(embed_text(query), dtype=float)
    vector_scores = [cosine_similarity(query_vector, v) for v in embeddings]
    norm_vector = normalise_scores(vector_scores)

    hybrid_scores = [
        (i, w_bm25 * norm_bm25[i] + w_vector * norm_vector[i])
        for i in range(len(chunks))
    ]
    hybrid_scores.sort(key=lambda x: x[1], reverse=True)
    return hybrid_scores[:top_k]
