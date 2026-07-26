import time
import pandas as pd
import joblib
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report


def main():
    print("Loading data...")
    df = (
        pd.read_csv("data/Resume.csv")[["ID", "Resume_str", "Category"]]
        .dropna()
        .reset_index(drop=True)
    )

    print("Loading pretrained sentence embedding model (all-MiniLM-L6-v2)...")
    print("(Downloads ~90MB the first time you run this - needs internet)")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    print(f"Encoding {len(df)} resumes into embeddings... this may take a few minutes.")
    t0 = time.time()
    embeddings = embedder.encode(
        df["Resume_str"].tolist(),
        show_progress_bar=True,
        batch_size=32,
    )
    print(f"Encoding finished in {time.time() - t0:.1f} seconds")

    print("Saving resume embeddings to models/resume_embeddings.pkl ...")
    joblib.dump(
        {
            "ids": df["ID"].values,
            "embeddings": embeddings,
            "categories": df["Category"].values,
        },
        "models/resume_embeddings.pkl",
    )

    X_train, X_test, y_train, y_test = train_test_split(
        embeddings,
        df["Category"],
        test_size=0.2,
        random_state=42,
        stratify=df["Category"],
    )

    candidates = {
        "RandomForest": RandomForestClassifier(
            n_estimators=400, class_weight="balanced", random_state=42, n_jobs=-1
        ),
        "LinearSVC": LinearSVC(C=3, class_weight="balanced"),
    }

    results = {}
    for name, model in candidates.items():
        print(f"\nTraining {name}...")
        t0 = time.time()
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average="macro")
        print(f"{name}: accuracy={acc:.2%}, macro_f1={f1:.3f}, time={time.time()-t0:.1f}s")
        print(classification_report(y_test, preds))
        results[name] = {"model": model, "acc": acc, "f1": f1}

    print("\n=== Comparison Summary ===")
    for name, r in results.items():
        print(f"{name:15s} accuracy={r['acc']:.2%}   macro_f1={r['f1']:.3f}")

    best_name = max(results, key=lambda n: results[n]["f1"])
    best_model = results[best_name]["model"]
    print(f"\nWinner: {best_name} (macro_f1={results[best_name]['f1']:.3f})")

    print("Saving winning embeddings-based classifier to models/classifier_embeddings.pkl ...")
    joblib.dump(best_model, "models/classifier_embeddings.pkl")

    print("Done! You can now run: streamlit run app_embeddings.py")


if __name__ == "__main__":
    main()
