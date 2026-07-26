import re
import time
import pandas as pd
import joblib
from scipy.stats import randint
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, f1_score


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
        ("tfidf", TfidfVectorizer(stop_words="english", sublinear_tf=True)),
        ("clf", RandomForestClassifier(class_weight="balanced", random_state=42, n_jobs=-1)),
    ])

    param_distributions = {
        "tfidf__max_features": [8000, 10000, 12000, 15000],
        "tfidf__ngram_range": [(1, 2), (1, 3)],
        "clf__n_estimators": randint(300, 900),
        "clf__max_depth": [None, 20, 40, 60],
        "clf__min_samples_split": [2, 4, 6],
        "clf__min_samples_leaf": [1, 2],
    }

    search = RandomizedSearchCV(
        pipeline,
        param_distributions,
        n_iter=25,         
        cv=3,
        scoring="f1_macro",
        n_jobs=-1,
        random_state=42,
        verbose=2,
    )

    print("Starting randomized search... this will take several minutes.")
    t0 = time.time()
    search.fit(X_train, y_train)
    print(f"\nSearch finished in {time.time() - t0:.1f} seconds")

    print("\nBest settings found:")
    print(search.best_params_)

    print("\nEvaluating best model on the held-out test set...")
    predictions = search.predict(X_test)
    acc = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions, average="macro")
    print(f"\nTuned Random Forest: accuracy={acc:.2%}, macro_f1={f1:.3f}\n")
    print(classification_report(y_test, predictions))

    best_vectorizer = search.best_estimator_.named_steps["tfidf"]
    best_model = search.best_estimator_.named_steps["clf"]

    print("Saving tuned model and vectorizer to models/ ...")
    joblib.dump(best_model, "models/classifier.pkl")
    joblib.dump(best_vectorizer, "models/vectorizer.pkl")

    print("Done! Restart the Streamlit app to use this model.")


if __name__ == "__main__":
    main()
