import re
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def main():
    print("Loading data...")
    df = pd.read_csv("data/Resume.csv")

    df = df[["Resume_str", "Category"]].dropna()

    print("Cleaning text...")
    df["clean_text"] = df["Resume_str"].apply(clean_text)

    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_text"],
        df["Category"],
        test_size=0.2,
        random_state=42,
        stratify=df["Category"],
    )

    print("Converting text into numbers (TF-IDF)...")
    vectorizer = TfidfVectorizer(
        stop_words="english",   
        max_features=5000,      
        ngram_range=(1, 2),     
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
