from rank_bm25 import BM25Okapi
import numpy as np
import sys

# [AI chỉnh 1/3] import dự phòng: dùng Vector_Store của lớp (w4) nếu có, nếu không dùng embedding offline
import os
from pathlib import Path

sys.path.append(os.getenv("W4_PATH", str(Path(__file__).resolve().parents[2] / "Week_4")))
sys.path.append(r"../Week_4")
try:
    from Vector_Store import Document, DocumentChunk, embed_text
    EMBEDDING_BACKEND = "Vector_Store (w4)"
except ImportError:
    from src.embedding import embed_text
    Document = DocumentChunk = object
    EMBEDDING_BACKEND = "LSA offline (src/embedding.py)"

def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Calculate cosine similarity between two embedding vectors."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denominator) if denominator else 0.0

def normalise_scores(scores: list[float]) -> list[float]:
    """Normalise scores to [0,1] using min-max scaling"""
    scores = list(scores)  # [AI chỉnh] BM25Okapi.get_scores trả numpy array nên `not scores` bị lỗi
    if not scores:
        return []
    min_s=min(scores)
    max_s=max(scores)
    if max_s == min_s:
        return [1.0] * len(scores)
    return [(s - min_s)/(max_s-min_s) for s in scores]

def build_bm25_index(chunks: list, tokenizer=None) -> BM25Okapi:
    """Build a BM25 index from chunk content (tokenized by whitespace)"""
    # [AI chỉnh 2/3] tokenizer tùy chọn; chunk có thể dùng .content (w4) hoặc .text (repo này)
    tokenize = tokenizer or (lambda t: t.lower().split())
    tokenized = [tokenize(getattr(chunk, "content", None) or chunk.text) for chunk in chunks]
    return BM25Okapi(tokenized)

def hybrid_search(
    query: str,
    chunks: list,
    bm25_index: BM25Okapi,
    embeddings: list[list[float]],
    top_k: int = 3,
    w_bm25: float = 0.3,
    w_vector: float = 0.7,
    tokenizer=None,
) -> list[tuple[int, float]]:
    """
    Hybrid search: combine BM25 (keyword) and vector (semantic) search scores.
    Returns: list of (chunk_index, combined_score) tuples sorted by combined_score descending.
    """
    
    query_tokens = (tokenizer or (lambda t: t.lower().split()))(query)
    bm25_scores = bm25_index.get_scores(query_tokens)
    norm_bm25= normalise_scores(bm25_scores)
    
    query_vector = np.asarray(embed_text(query), dtype=float)
    vector_scores = [
        cosine_similarity(query_vector,chunk_vector) for chunk_vector in embeddings
    ]
    norm_vector= normalise_scores(vector_scores)
    
    hybrid_scores = [
        (i, w_bm25 * norm_bm25[i] + w_vector * norm_vector[i])
        for i in range(len(chunks))
    ]
    
    hybrid_scores.sort(key=lambda x: x[1], reverse=True)
    return hybrid_scores[:top_k]



def dense_search(query: str, embeddings: list[list[float]], top_k: int = 3) -> list[tuple[int, float]]:
    """[AI chỉnh 3/3] Dense-only: xếp hạng theo cosine giữa embedding câu hỏi và từng chunk."""
    query_vector = np.asarray(embed_text(query), dtype=float)
    scores = [(i, cosine_similarity(query_vector, v)) for i, v in enumerate(embeddings)]
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k]
