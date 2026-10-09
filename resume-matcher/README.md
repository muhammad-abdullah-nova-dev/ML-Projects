# Resume Matcher

[![Transformers](https://img.shields.io/badge/Transformers-Sentence--Transformers-FFD21E.svg?style=flat&logo=huggingface&logoColor=white)](https://huggingface.co)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40+-FF4B4B.svg?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Python](https://img.shields.io/badge/Python-3.9%20%7C%203.10%20%7C%203.11-3776AB.svg?style=flat&logo=python&logoColor=white)](https://python.org)
[![Tests](https://img.shields.io/badge/Tests-Passing-brightgreen.svg)]()

A semantic search and resume ranking engine utilizing dense vector embeddings and cosine similarity to match applicant resumes against target job descriptions. Engineered with multi-format document parsing (TXT, PDF, DOCX), deterministic hashing vector fallback for lightweight environments, and an interactive Streamlit UI.

Maintained and enhanced by **M. Abdullah**.

---

## 🔍 Features & Architecture

* **Multi-Format Extraction**: Parses plain text, PDF documents, and DOCX resumes with automatic text decoding and stream buffering.
* **Dense Vector Embeddings**: Uses `sentence-transformers` (`all-MiniLM-L6-v2`) with a graceful L2-normalized `HashingVectorizer` fallback for offline/low-resource execution.
* **Cosine Similarity Ranking**: Computes pairwise cosine similarity between normalized candidate embeddings and the target job description vector.
* **CLI & Streamlit Interfaces**: Automated batch matching via CLI and interactive upload UI via Streamlit.

---

## ⚡ Quickstart

### 1. Installation
```bash
cd resume-matcher
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Run CLI Matcher
```bash
python main.py
```
Outputs ranked candidate table with similarity percentage against `data/job_descriptions/software_engineer.txt`.

### 3. Launch Streamlit Web UI
```bash
streamlit run app.py
```

---

## 🧪 Testing
```bash
pytest tests/ -v
```

---

## 👨‍💻 Maintainer & Attribution
- **Enhanced Implementation Maintainer**: **M. Abdullah**
- **Original Project Origin**: Derivative work based on open-source project by `torresjchristopher`.
- **Copyright**: Copyright © 2026 M. Abdullah for enhancements and additions.
