import re


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_jd_keywords(jd_text: str, vectorizer, top_n: int = 25):
    cleaned = clean_text(jd_text)
    vec = vectorizer.transform([cleaned])
    feature_names = vectorizer.get_feature_names_out()
    scores = vec.toarray()[0]
    top_indices = scores.argsort()[::-1]

    keywords = []
    for idx in top_indices:
        if scores[idx] <= 0:
            break
        keywords.append(feature_names[idx])
        if len(keywords) >= top_n:
            break
    return keywords


def keyword_match_score(resume_text: str, keywords: list):
    cleaned_resume = clean_text(resume_text)
    matched = [kw for kw in keywords if kw in cleaned_resume]
    missing = [kw for kw in keywords if kw not in cleaned_resume]
    pct = (len(matched) / len(keywords) * 100) if keywords else 0
    return pct, matched, missing


def section_completeness_score(resume_text: str):
    text_lower = resume_text.lower()
    has_email = bool(re.search(r'[\w.\-]+@[\w.\-]+', resume_text))
    has_phone = bool(re.search(r'\b\d{10}\b', re.sub(r'[\s\-\(\)]', '', resume_text)))

    sections = {
        "Contact Info": has_email or has_phone,
        "Education": any(w in text_lower for w in
                          ["education", "university", "college", "degree", "bachelor", "master"]),
        "Experience": any(w in text_lower for w in
                           ["experience", "work history", "employment", "professional experience"]),
        "Skills": any(w in text_lower for w in
                      ["skills", "technical skills", "proficien"]),
    }
    found = sum(sections.values())
    pct = found / len(sections) * 100
    return pct, sections


def formatting_score(resume_text: str):
    word_count = len(resume_text.split())

    if 150 <= word_count <= 1200:
        length_score = 100
    elif word_count < 150:
        length_score = max(0, word_count / 150 * 100)
    else:
        length_score = max(0, 100 - (word_count - 1200) / 20)

    has_bullets = bool(re.search(r'[•\-\*]\s', resume_text))
    bullet_score = 100 if has_bullets else 60

    return (length_score + bullet_score) / 2


def compute_ats_score(resume_text: str, jd_text: str, vectorizer, classifier):
    keywords = extract_jd_keywords(jd_text, vectorizer, top_n=25)
    kw_pct, matched, missing = keyword_match_score(resume_text, keywords)

    section_pct, sections = section_completeness_score(resume_text)
    format_pct = formatting_score(resume_text)

    cleaned_resume = clean_text(resume_text)
    cleaned_jd = clean_text(jd_text)
    resume_vec = vectorizer.transform([cleaned_resume])
    jd_vec = vectorizer.transform([cleaned_jd])
    resume_category = classifier.predict(resume_vec)[0]
    jd_category = classifier.predict(jd_vec)[0]
    category_match = (resume_category == jd_category)
    category_pct = 100 if category_match else 40

    total = (
        kw_pct * 0.45
        + section_pct * 0.20
        + category_pct * 0.20
        + format_pct * 0.15
    )

    return {
        "total_score": round(total, 1),
        "breakdown": {
            "Keyword Match": round(kw_pct, 1),
            "Section Completeness": round(section_pct, 1),
            "Job Category Fit": round(category_pct, 1),
            "Formatting": round(format_pct, 1),
        },
        "matched_keywords": matched,
        "missing_keywords": missing,
        "sections_found": sections,
        "resume_predicted_category": resume_category,
        "jd_predicted_category": jd_category,
    }
