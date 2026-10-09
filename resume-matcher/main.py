"""
CLI runner for Resume Matcher.
Engineered by M. Abdullah.
Scans data/resumes, matches against target job description, and outputs ranked candidates.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.parser import extract_text_from_file, load_job_description
from src.matcher import rank_resumes


def run_matcher():
    jd_path = PROJECT_ROOT / "data" / "job_descriptions" / "software_engineer.txt"
    if not jd_path.exists():
        jd_path = PROJECT_ROOT / "data" / "job_desc" / "software_engineer.txt"

    print(f"Loading Job Description from {jd_path}...")
    jd_text = load_job_description(str(jd_path))

    resumes_dir = PROJECT_ROOT / "data" / "resumes"
    valid_exts = {".txt", ".pdf", ".docx"}
    resume_files = [f for f in resumes_dir.iterdir() if f.suffix.lower() in valid_exts]

    if not resume_files:
        print(f"No resume documents found in {resumes_dir}.")
        return []

    print(f"Discovered {len(resume_files)} candidate resumes. Computing semantic embeddings...")
    resumes_dict = {}
    for f in resume_files:
        text = extract_text_from_file(f)
        resumes_dict[f.name] = text

    ranked = rank_resumes(resumes_dict, jd_text)

    print("\n" + "=" * 60)
    print("CANDIDATE RANKING (Semantic Similarity Score)")
    print("=" * 60)
    for rank, (name, score) in enumerate(ranked, 1):
        pct = score * 100
        print(f"#{rank} | {name.ljust(30)} | Score: {score:.4f} ({pct:.1f}% match)")
    print("=" * 60 + "\n")

    return ranked


if __name__ == "__main__":
    run_matcher()
