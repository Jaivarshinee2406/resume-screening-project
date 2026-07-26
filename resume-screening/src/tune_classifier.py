import re
import time
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def main():
    print("Loading data...")
    df = pd.read_csv("data/Resume.csv")[["Resume_str", "Category"]].dropna()

    print("Cleaning text...")
    df["clean_text"] = df["Resume_str"].apply(clean_text)

    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_text"],
        df["Category"],
        test_size=0.2,
        random_state=42,
        stratify=df["Category"],
    )

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english")),
        ("clf", LogisticRegression(max_iter=1000)),
    ])

    param_grid = {
        "tfidf__max_features": [3000, 5000, 8000],
        "tfidf__ngram_range": [(1, 1), (1, 2)],
        "clf__C": [0.1, 1, 10],
        "clf__class_weight": [None, "balanced"],
    }

    search = GridSearchCV(
        pipeline,
        param_grid,
        cv=3,
        scoring="f1_macro",
        n_jobs=-1,
        verbose=2,
    )

    print("Starting grid search... this will take a few minutes.")
    t0 = time.time()
    search.fit(X_train, y_train)
    print(f"\nGrid search finished in {time.time() - t0:.1f} seconds")

    print("\nBest settings found:")
    print(search.best_params_)

    print("\nEvaluating best model on the held-out test set...")
    predictions = search.predict(X_test)
    acc = accuracy_score(y_test, predictions)
    print(f"\nTuned model accuracy: {acc:.2%}\n")
    print(classification_report(y_test, predictions))

    best_vectorizer = search.best_estimator_.named_steps["tfidf"]
    best_model = search.best_estimator_.named_steps["clf"]

    print("Saving tuned model and vectorizer to models/ ...")
    joblib.dump(best_model, "models/classifier.pkl")
    joblib.dump(best_vectorizer, "models/vectorizer.pkl")

    print("Done! Restart the Streamlit app to use the improved model.")


if __name__ == "__main__":
    main()
