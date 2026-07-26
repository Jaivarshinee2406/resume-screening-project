"""
app_ats.py
-----------
Upload one or more resumes (PDF, DOCX, or TXT) and paste a job
description. The app tells you, for each resume:
    1. Whether its predicted job category matches the JD
    2. An overall ATS Score (0-100) estimating how well an Applicant
       Tracking System might rate it for this specific role
    3. A breakdown of that score
    4. Which important keywords from the JD are missing

When multiple resumes are uploaded, they're ranked from best to worst
match so you can quickly compare candidates.

Run with:  streamlit run app_ats.py
(Requires models/classifier.pkl and models/vectorizer.pkl - run
src/compare_models.py first if they don't exist yet.)
"""

import sys
import os
import joblib
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from resume_parser import extract_resume_text
from ats_scoring import compute_ats_score

st.set_page_config(page_title="Resume ATS Checker", layout="wide")


@st.cache_resource
def load_model():
    model = joblib.load("models/classifier.pkl")
    vectorizer = joblib.load("models/vectorizer.pkl")
    return model, vectorizer


model, vectorizer = load_model()

st.title("📄 Resume ATS Checker")
st.write(
    "Upload one or more resumes and paste a job description. This estimates how "
    "an Applicant Tracking System might score each resume for that role, "
    "and shows you exactly what's missing."
)

col1, col2 = st.columns(2)

with col1:
    uploaded_files = st.file_uploader(
        "Upload one or more resumes",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
    )

with col2:
    job_description = st.text_area(
        "Job Description", height=220, placeholder="Paste the job description here..."
    )


def score_badge(score):
    if score >= 75:
        return "🟢 Strong match"
    elif score >= 55:
        return "🟡 Moderate match"
    else:
        return "🔴 Weak match"


if st.button("Check ATS Score(s)", type="primary"):
    if not uploaded_files:
        st.warning("Please upload at least one resume file.")
    elif not job_description.strip():
        st.warning("Please paste a job description first.")
    else:
        results = []
        for uploaded_file in uploaded_files:
            try:
                resume_text = extract_resume_text(uploaded_file)
            except ValueError as e:
                st.error(f"**{uploaded_file.name}**: {e}")
                continue

            if len(resume_text.strip()) < 30:
                st.error(
                    f"**{uploaded_file.name}**: Couldn't extract readable text. "
                    "If it's a PDF, it may be a scanned image rather than "
                    "selectable text - try a DOCX or TXT version instead."
                )
                continue

            result = compute_ats_score(resume_text, job_description, vectorizer, model)
            result["filename"] = uploaded_file.name
            result["resume_text"] = resume_text
            results.append(result)

        if results:
            # Rank best to worst
            results.sort(key=lambda r: r["total_score"], reverse=True)

            if len(results) > 1:
                st.subheader(f"Ranked Results ({len(results)} resumes)")
                summary_rows = []
                for rank, r in enumerate(results, start=1):
                    summary_rows.append({
                        "Rank": rank,
                        "File": r["filename"],
                        "ATS Score": r["total_score"],
                        "Match": score_badge(r["total_score"]),
                        "Predicted Category": r["resume_predicted_category"],
                    })
                st.table(summary_rows)

            st.divider()

            for rank, result in enumerate(results, start=1):
                header = f"#{rank} — {result['filename']} — {result['total_score']}/100 {score_badge(result['total_score'])}" \
                    if len(results) > 1 else f"{result['filename']} — {result['total_score']}/100"

                with st.expander(header, expanded=(len(results) == 1)):
                    score = result["total_score"]
                    if score >= 75:
                        st.success(f"### ATS Score: {score}/100 — Strong match")
                    elif score >= 55:
                        st.warning(f"### ATS Score: {score}/100 — Moderate match")
                    else:
                        st.error(f"### ATS Score: {score}/100 — Weak match")

                    cat_match = result["resume_predicted_category"] == result["jd_predicted_category"]
                    if cat_match:
                        st.info(
                            f"✅ Predicted field (**{result['resume_predicted_category']}**) "
                            f"matches this job's predicted field."
                        )
                    else:
                        st.info(
                            f"⚠️ This resume looks like **{result['resume_predicted_category']}**, "
                            f"but the job description looks like **{result['jd_predicted_category']}**."
                        )

                    st.write("**Score Breakdown**")
                    bcol1, bcol2, bcol3, bcol4 = st.columns(4)
                    bcol1.metric("Keyword Match", f"{result['breakdown']['Keyword Match']}%")
                    bcol2.metric("Section Completeness", f"{result['breakdown']['Section Completeness']}%")
                    bcol3.metric("Job Category Fit", f"{result['breakdown']['Job Category Fit']}%")
                    bcol4.metric("Formatting", f"{result['breakdown']['Formatting']}%")

                    st.write("**Resume Sections Detected**")
                    sect_cols = st.columns(4)
                    for i, (section, found) in enumerate(result["sections_found"].items()):
                        sect_cols[i].write(f"{'✅' if found else '❌'} {section}")

                    st.write("**Keywords Missing**")
                    if result["missing_keywords"]:
                        st.write(", ".join(f"`{kw}`" for kw in result["missing_keywords"]))
                    else:
                        st.write("None — this resume covers all top keywords from the JD!")

                    st.write("**Keywords Found**")
                    st.write(", ".join(f"`{kw}`" for kw in result["matched_keywords"]) or "None")

                    with st.expander("View Extracted Resume Text"):
                        st.text(result["resume_text"][:3000])