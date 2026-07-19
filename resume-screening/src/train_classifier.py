"""
train_classifier.py
--------------------
Trains a machine learning model that reads a resume's text and predicts
which job category it belongs to (e.g. HR, FINANCE, ENGINEERING...).

Run this file once to train the model. It will save two files inside
the models/ folder:
    - vectorizer.pkl   (turns text into numbers)
    - classifier.pkl   (the trained model)

These saved files are then reused by app.py so we don't have to
retrain every time we open the app.
"""

import re
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score


def clean_text(text: str) -> str:
    """
    Basic text cleaning:
    - lowercase everything
    - remove anything that isn't a letter or space
    - collapse multiple spaces into one
    """
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def main():
    print("Loading data...")
    df = pd.read_csv("data/Resume.csv")

    # We only need the resume text and its category label
    df = df[["Resume_str", "Category"]].dropna()

    print("Cleaning text...")
    df["clean_text"] = df["Resume_str"].apply(clean_text)

    # Split into training data (80%) and testing data (20%)
    # "stratify" makes sure each category is represented proportionally
    # in both the train and test sets.
    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_text"],
        df["Category"],
        test_size=0.2,
        random_state=42,
        stratify=df["Category"],
    )

    print("Converting text into numbers (TF-IDF)...")
    # TF-IDF turns each resume into a vector of numbers based on which
    # words appear and how important/rare those words are.
    vectorizer = TfidfVectorizer(
        stop_words="english",   # ignore common filler words like "the", "and"
        max_features=5000,      # keep only the 5000 most useful words
        ngram_range=(1, 2),     # consider single words AND two-word phrases
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    print("Training the classifier...")
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_vec, y_train)

    print("Evaluating on the test set...")
    predictions = model.predict(X_test_vec)
    acc = accuracy_score(y_test, predictions)
    print(f"\nAccuracy: {acc:.2%}\n")
    print(classification_report(y_test, predictions))

    print("Saving model and vectorizer to models/ ...")
    joblib.dump(model, "models/classifier.pkl")
    joblib.dump(vectorizer, "models/vectorizer.pkl")

    print("Done! You can now run the app with: streamlit run app.py")


if __name__ == "__main__":
    main()
