"""Embedding dự phòng chạy offline (TF-IDF + SVD), dùng khi không import được Vector_Store của lớp (w4)."""
import re
from collections import Counter

import numpy as np

vocab: dict[str, int] = {}
idf = None
basis = None


def words(text: str) -> list[str]:
    """Tách từ dùng chung cho BM25 và embedding."""
    return re.findall(r"\w+", text.lower())


def count_vector(text: str) -> np.ndarray:
    row = np.zeros(len(vocab))
    for w, n in Counter(words(text)).items():
        if w in vocab:
            row[vocab[w]] = n
    return row


def fit(texts: list[str]):
    """Học không gian vector từ các chunk của corpus."""
    global vocab, idf, basis
    all_words = sorted({w for t in texts for w in words(t)})
    vocab = {w: i for i, w in enumerate(all_words)}
    counts = np.array([count_vector(t) for t in texts])
    idf = np.log((1 + len(texts)) / (1 + (counts > 0).sum(axis=0))) + 1
    _, _, vt = np.linalg.svd(counts * idf, full_matrices=False)
    basis = vt[:64]


def embed_text(text: str) -> list[float]:
    if basis is None:
        raise RuntimeError("Cần gọi fit() trước")
    vec = (count_vector(text) * idf) @ basis.T
    norm = np.linalg.norm(vec)
    return (vec / norm if norm > 0 else vec).tolist()
