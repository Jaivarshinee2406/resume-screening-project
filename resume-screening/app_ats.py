"""
app_ats.py
-----------
Upload YOUR resume (PDF, DOCX, or TXT) and paste a job description.
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
    "Upload your resume and paste a job description. This estimates how "
    "an Applicant Tracking System might score your resume for that role, "
    "and shows you exactly what's missing."
)

col1, col2 = st.columns(2)

with col1:
    uploaded_file = st.file_uploader(
        "Upload your resume", type=["pdf", "docx", "txt"]
    )

with col2:
    job_description = st.text_area(
        "Job Description", height=220, placeholder="Paste the job description here..."
    )

if st.button("Check ATS Score", type="primary"):
    if uploaded_file is None:
        st.warning("Please upload a resume file first.")
    elif not job_description.strip():
        st.warning("Please paste a job description first.")
    else:
        try:
            resume_text = extract_resume_text(uploaded_file)
        except ValueError as e:
            st.error(str(e))
            resume_text = None

        if resume_text is not None:
            if len(resume_text.strip()) < 30:
                st.error(
                    "Couldn't extract readable text from this file. If it's a "
                    "PDF, it may be a scanned image rather than selectable text - "
                    "try a DOCX or TXT version instead."
                )
            else:
                result = compute_ats_score(resume_text, job_description, vectorizer, model)

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
                        f"✅ Your resume's predicted field (**{result['resume_predicted_category']}**) "
                        f"matches this job's predicted field."
                    )
                else:
                    st.info(
                        f"⚠️ Your resume looks like **{result['resume_predicted_category']}**, "
                        f"but this job description looks like **{result['jd_predicted_category']}**."
                    )

                st.subheader("Score Breakdown")
                bcol1, bcol2, bcol3, bcol4 = st.columns(4)
                bcol1.metric("Keyword Match", f"{result['breakdown']['Keyword Match']}%")
                bcol2.metric("Section Completeness", f"{result['breakdown']['Section Completeness']}%")
                bcol3.metric("Job Category Fit", f"{result['breakdown']['Job Category Fit']}%")
                bcol4.metric("Formatting", f"{result['breakdown']['Formatting']}%")

                st.subheader("Resume Sections Detected")
                sect_cols = st.columns(4)
                for i, (section, found) in enumerate(result["sections_found"].items()):
                    sect_cols[i].write(f"{'✅' if found else '❌'} {section}")

                st.subheader("Keywords Missing From Your Resume")
                if result["missing_keywords"]:
                    st.write(
                        "Consider adding these terms (if genuinely relevant to your "
                        "experience) to improve keyword match:"
                    )
                    st.write(", ".join(f"`{kw}`" for kw in result["missing_keywords"]))
                else:
                    st.write("None — your resume covers all the top keywords from this JD!")

                with st.expander("Keywords Found In Your Resume"):
                    st.write(", ".join(f"`{kw}`" for kw in result["matched_keywords"]) or "None")

                with st.expander("View Extracted Resume Text"):
                    st.text(resume_text[:3000])