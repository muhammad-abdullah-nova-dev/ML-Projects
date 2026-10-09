"""
Text embedding module with Transformer support and Hashing/TF-IDF fallback.
Engineered by M. Abdullah.
"""
from typing import List
import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer

_transformer_model = None
_use_transformer = True
_hasher = HashingVectorizer(n_features=384, alternate_sign=False, norm="l2")


def get_transformer_model():
    global _transformer_model, _use_transformer
    if _transformer_model is None and _use_transformer:
        try:
            from sentence_transformers import SentenceTransformer
            _transformer_model = SentenceTransformer("all-MiniLM-L6-v2")
        except Exception:
            _use_transformer = False
            _transformer_model = None
    return _transformer_model


def embed_text(text: str) -> np.ndarray:
    """
    Encode text string into dense vector embedding.
    Uses sentence-transformers if available, else fixed-dimension normalized HashingVectorizer.
    """
    model = get_transformer_model()
    if model is not None:
        emb = model.encode(text, convert_to_tensor=False)
        return np.asarray(emb, dtype=np.float32)

    # Deterministic fixed-dimension fallback (384 dims)
    matrix = _hasher.transform([text])
    return np.asarray(matrix.toarray()[0], dtype=np.float32)


def embed_corpus(texts: List[str]) -> np.ndarray:
    """Encode a collection of documents into a 2D matrix (N, D)."""
    model = get_transformer_model()
    if model is not None:
        return np.asarray(model.encode(texts, convert_to_tensor=False), dtype=np.float32)

    matrix = _hasher.transform(texts)
    return np.asarray(matrix.toarray(), dtype=np.float32)
