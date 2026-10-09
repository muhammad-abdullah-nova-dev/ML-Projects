"""
Unit tests for Resume Matcher semantic similarity and parsing.
Engineered by M. Abdullah.
"""
import io
import sys
from pathlib import Path
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.parser import extract_text_from_file
from src.embedder import embed_text, embed_corpus
from src.matcher import compute_similarity, rank_resumes


def test_extract_text_from_stream():
    stream = io.StringIO("Machine Learning Engineer with Python experience")
    text = extract_text_from_file(stream)
    assert "Machine Learning" in text


def test_embedding_and_similarity():
    text_a = "Senior Machine Learning Engineer specializing in PyTorch and Transformers"
    text_b = "ML Engineer with deep learning and transformer expertise"
    text_c = "Accountant and certified public tax auditor"

    emb_a = embed_text(text_a)
    emb_b = embed_text(text_b)
    emb_c = embed_text(text_c)

    sim_ab = compute_similarity(emb_a, emb_b)
    sim_ac = compute_similarity(emb_a, emb_c)

    assert 0.0 <= sim_ab <= 1.0
    assert 0.0 <= sim_ac <= 1.0
    # Relevant resumes should have higher semantic similarity than completely unrelated domains
    assert sim_ab > sim_ac


def test_rank_resumes():
    jd = "Seeking a Python developer with FastAPI and Docker experience."
    resumes = {
        "candidate_a.txt": "Python backend software developer proficient in FastAPI, PostgreSQL, and Docker containerization.",
        "candidate_b.txt": "Professional pastry chef with culinary background in French pastry baking."
    }

    ranking = rank_resumes(resumes, jd)
    assert len(ranking) == 2
    # Candidate A must rank first
    assert ranking[0][0] == "candidate_a.txt"
    assert ranking[0][1] > ranking[1][1]
