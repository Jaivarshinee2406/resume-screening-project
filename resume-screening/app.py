"""
app.py
------
The web app (Streamlit UI) for the AI Resume Screening & Candidate
Ranking System.

What it does:
1. Recruiter pastes in a Job Description.
2. The app predicts which job Category that description belongs to.
3. The app compares the Job Description against every resume (within
   that category, or across all resumes) using TF-IDF + cosine
   similarity, and ranks candidates from most to least relevant.

Run with:  streamlit run app.py
(Make sure you've already run: python src/train_classifier.py)
"""

import re
import pandas as pd
import joblib
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="AI Resume Screening & Ranking", layout="wide")


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


@st.cache_resource
def load_model():
    model = joblib.load("models/classifier.pkl")
    vectorizer = joblib.load("models/vectorizer.pkl")
    return model, vectorizer


@st.cache_data
def load_resumes(_vectorizer):
    df = pd.read_csv("data/Resume.csv")
    df = df[["ID", "Resume_str", "Category"]].dropna()
    df["clean_text"] = df["Resume_str"].apply(clean_text)
    # Pre-compute TF-IDF vectors for every resume once, so ranking is fast.
    resume_vectors = _vectorizer.transform(df["clean_text"])
    return df, resume_vectors


# ---------- Load everything once ----------
model, vectorizer = load_model()
resumes_df, resume_vectors = load_resumes(vectorizer)

st.title("📄 AI Resume Screening & Candidate Ranking System")
st.write(
    "Paste a job description below. The app will predict its category "
    "and rank the most relevant resumes from the dataset."
)

# ---------- Sidebar controls ----------
st.sidebar.header("Settings")
top_n = st.sidebar.slider("How many top candidates to show?", 3, 20, 10)

category_options = ["Auto-detect from job description"] + sorted(
    resumes_df["Category"].unique().tolist()
)
category_choice = st.sidebar.selectbox(
    "Restrict search to a category:", category_options
)

# ---------- Main input ----------
job_description = st.text_area(
    "Job Description",
    height=200,
    placeholder="Paste the job description here...",
)

if st.button("Find Best Candidates", type="primary"):
    if not job_description.strip():
        st.warning("Please paste a job description first.")
    else:
        cleaned_jd = clean_text(job_description)
        jd_vector = vectorizer.transform([cleaned_jd])

        # Step 1: Predict the category of the job description
        predicted_category = model.predict(jd_vector)[0]
        st.success(f"Predicted job category: **{predicted_category}**")

        # Step 2: Decide which resumes to search over
        if category_choice == "Auto-detect from job description":
            search_category = predicted_category
        else:
            search_category = category_choice

        mask = resumes_df["Category"] == search_category
        filtered_df = resumes_df[mask].reset_index(drop=True)
        filtered_vectors = resume_vectors[mask.values]

        if filtered_df.empty:
            st.warning("No resumes found in this category.")
        else:
            # Step 3: Rank by cosine similarity to the job description
            similarities = cosine_similarity(jd_vector, filtered_vectors).flatten()
            filtered_df["match_score"] = (similarities * 100).round(2)
            ranked = filtered_df.sort_values("match_score", ascending=False).head(top_n)

            st.subheader(f"Top {len(ranked)} candidates in '{search_category}'")

            for _, row in ranked.iterrows():
                with st.expander(
                    f"ID {row['ID']}  |  Match Score: {row['match_score']}%"
                ):
                    st.write(row["Resume_str"][:2000] + "...")
