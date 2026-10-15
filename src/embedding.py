import re
import numpy as np

vocabulary = []
idf = None
basis = None


def words(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def fit(texts: list[str]) -> None:
    """Học cách biểu diễn các đoạn văn thành vector số."""
    global vocabulary, idf, basis

    vocabulary = sorted({word for text in texts for word in words(text)})
    counts = np.array(
        [[words(text).count(word) for word in vocabulary] for text in texts],
        dtype=float,
    )

    idf = np.log((1 + len(texts)) / (1 + (counts > 0).sum(axis=0))) + 1
    _, _, directions = np.linalg.svd(counts * idf, full_matrices=False)
    basis = directions[:32]


def embed_text(text: str) -> list[float]:
    """Biến một câu thành vector để so độ giống nhau."""
    if basis is None:
        raise RuntimeError("Cần gọi fit() trước")

    tokens = words(text)
    counts = np.array([tokens.count(word) for word in vocabulary])
    vector = (counts * idf) @ basis.T
    length = np.linalg.norm(vector)

    if length > 0:
        vector = vector / length

    return vector.tolist()