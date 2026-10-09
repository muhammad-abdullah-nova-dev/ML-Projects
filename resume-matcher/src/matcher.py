"""
Semantic similarity and ranking engine for resumes.
Engineered by M. Abdullah.
"""
from typing import Dict, List, Tuple
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from src.embedder import embed_corpus


def compute_similarity(resume_embedding: np.ndarray, jd_embedding: np.ndarray) -> float:
    """Compute cosine similarity between two vector representations."""
    u = np.asarray(resume_embedding).reshape(1, -1)
    v = np.asarray(jd_embedding).reshape(1, -1)

    norm_u = np.linalg.norm(u)
    norm_v = np.linalg.norm(v)

    if norm_u == 0 or norm_v == 0:
        return 0.0

    sim = np.dot(u, v.T) / (norm_u * norm_v)
    return float(round(float(sim[0, 0]), 4))


def rank_resumes(resumes_dict: Dict[str, str], job_description: str) -> List[Tuple[str, float]]:
    """
    Given a mapping of {candidate_name: resume_text} and a job description text,
    compute similarity embeddings in batch and return sorted ranking.
    """
    if not resumes_dict:
        return []

    names = list(resumes_dict.keys())
    texts = [resumes_dict[name] for name in names] + [job_description]

    embeddings = embed_corpus(texts)
    resume_embs = embeddings[:-1]
    jd_emb = embeddings[-1:]

    sims = cosine_similarity(resume_embs, jd_emb).flatten()
    results = [(name, float(round(float(sim), 4))) for name, sim in zip(names, sims)]
    results.sort(key=lambda x: x[1], reverse=True)
    return results
