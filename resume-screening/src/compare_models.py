import re
import time
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report


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

    print("Building TF-IDF features...")
    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=8000,
        ngram_range=(1, 3),
        sublinear_tf=True,
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    candidates = {
        "LogisticRegression": LogisticRegression(
            max_iter=1000, C=10, class_weight="balanced"
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=400, class_weight="balanced", random_state=42, n_jobs=-1
        ),
        "LinearSVC": LinearSVC(C=3, class_weight="balanced"),
    }

    results = {}
    for name, model in candidates.items():
        print(f"\nTraining {name}...")
        t0 = time.time()
        model.fit(X_train_vec, y_train)
        preds = model.predict(X_test_vec)
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average="macro")
        elapsed = time.time() - t0
        print(f"{name}: accuracy={acc:.2%}, macro_f1={f1:.3f}, time={elapsed:.1f}s")
        print(classification_report(y_test, preds))
        results[name] = {"model": model, "acc": acc, "f1": f1}

    print("\n=== Comparison Summary ===")
    for name, r in results.items():
        print(f"{name:15s} accuracy={r['acc']:.2%}   macro_f1={r['f1']:.3f}")

    best_name = max(results, key=lambda n: results[n]["f1"])
    best_model = results[best_name]["model"]
    print(f"\nWinner: {best_name} (macro_f1={results[best_name]['f1']:.3f})")

    print("Saving winning model and vectorizer to models/ ...")
    joblib.dump(best_model, "models/classifier.pkl")
    joblib.dump(vectorizer, "models/vectorizer.pkl")

    print("Done! Restart the Streamlit app to use the winning model.")


if __name__ == "__main__":
    main()
