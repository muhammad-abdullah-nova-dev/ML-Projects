"""
Streamlit Web Dashboard for Semantic Resume & Job Matcher.
Engineered by M. Abdullah.
Features dense embedding and vector space cosine similarity scoring and candidate leaderboard ranking.
"""
from pathlib import Path
import streamlit as st
import pandas as pd

from src.parser import extract_text_from_file, load_job_description
from src.embedder import embed_text
from src.matcher import compute_similarity, rank_resumes

st.set_page_config(page_title="Semantic Resume Matcher", page_icon="🧠", layout="wide")

st.title("🧠 Semantic Resume Matcher")
st.markdown(
    "**NLP Document Similarity & Candidate Ranking Engine** | "
    "Uses dense text representations and vector cosine geometry to rank candidates against job requirements."
)

base_dir = Path(__file__).resolve().parent
jd_dir = base_dir / "data" / "job_descriptions"
resumes_dir = base_dir / "data" / "resumes"

available_jds = [f.name for f in jd_dir.glob("*.txt")] if jd_dir.exists() else ["software_engineer.txt"]
available_resumes = [f.name for f in resumes_dir.glob("*.txt") if f.name != ".gitkeep"] if resumes_dir.exists() else []

tab1, tab2 = st.tabs(["🎯 Single Candidate Matcher", "🏆 Batch Candidate Leaderboard"])

with tab1:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("1. Candidate Resume")
        resume_mode = st.radio("Resume Source", ["Pick Sample Resume", "Upload File (PDF/TXT/DOCX)"])
        resume_text = ""
        if resume_mode == "Pick Sample Resume":
            if available_resumes:
                selected_sample = st.selectbox("Sample Resumes", available_resumes)
                resume_text = extract_text_from_file(resumes_dir / selected_sample)
            else:
                st.warning("No sample resumes found.")
        else:
            uploaded_resume = st.file_uploader("Upload Candidate Resume", type=["pdf", "txt", "docx"])
            if uploaded_resume:
                resume_text = extract_text_from_file(uploaded_resume)

        if resume_text:
            with st.expander("Preview Resume Text (First 400 chars)"):
                st.text(resume_text[:400] + ("..." if len(resume_text) > 400 else ""))

    with col2:
        st.subheader("2. Target Job Description")
        jd_choice = st.selectbox("Select Target Job Description", available_jds)
        jd_path = jd_dir / jd_choice
        jd_text = load_job_description(str(jd_path)) if jd_path.exists() else ""

        if jd_text:
            with st.expander("Preview Job Description (First 400 chars)"):
                st.text(jd_text[:400] + ("..." if len(jd_text) > 400 else ""))

    st.markdown("---")
    if st.button("🚀 Calculate Match Score", use_container_width=True):
        if not resume_text:
            st.error("Please provide a resume to match.")
        elif not jd_text:
            st.error("Please provide a job description.")
        else:
            with st.spinner("Embedding text and computing vector cosine similarity..."):
                res_embed = embed_text(resume_text)
                jd_embed = embed_text(jd_text)
                similarity = compute_similarity(res_embed, jd_embed)
                match_pct = round(similarity * 100, 2)

            st.subheader("Semantic Match Result")
            score_col1, score_col2 = st.columns([1, 2])
            score_col1.metric("Match Score", f"{match_pct}%", delta=f"{similarity:.4f} Cosine Score")
            score_col2.progress(min(max(float(similarity), 0.0), 1.0))

            if match_pct >= 70:
                st.success("🌟 Strong Fit: The candidate's background closely aligns with the job profile.")
            elif match_pct >= 45:
                st.info("🟡 Moderate Fit: The candidate meets several qualifications but may have gaps.")
            else:
                st.warning("🔴 Low Fit: Minimal semantic overlap with the required skillset.")

with tab2:
    st.subheader("Batch Candidate Ranking")
    st.markdown("Rank all candidate profiles in the database against the selected job description.")
    
    if st.button("📊 Run Batch Leaderboard", use_container_width=True):
        if not jd_text:
            st.error("Target job description is missing.")
        else:
            with st.spinner("Ranking all candidates..."):
                sample_files = [str(resumes_dir / f) for f in available_resumes]
                ranked_df = rank_resumes(sample_files, jd_text)
                ranked_df["Match %"] = (ranked_df["Score"] * 100).round(2).astype(str) + "%"

            st.dataframe(ranked_df[["Candidate", "Score", "Match %"]], use_container_width=True)
