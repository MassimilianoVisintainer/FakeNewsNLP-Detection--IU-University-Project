"""
train_tfidf_nb_svm.py
---------------------
Compare multiple classifiers for Fake News Detection using TF-IDF features:
- Logistic Regression
- Multinomial Naive Bayes
- Linear SVM
"""

import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report
import joblib


def evaluate_and_report(model, X_train, y_train, X_test, y_test, name):
    """Train model, evaluate, and print results."""
    print(f"\n=== {name} ===")

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    print(f"Accuracy: {acc:.4f}")
    print(classification_report(y_test, preds, digits=3))

    return model


def main():
    # -------------------
    # Step 1: Load data
    # -------------------
    train = pd.read_csv("data/processed/train.csv")
    test = pd.read_csv("data/processed/test.csv")

    X_train = train["clean_text"].fillna("")
    y_train = train["label"]

    X_test = test["clean_text"].fillna("")
    y_test = test["label"]

    # -------------------
    # Step 2: TF-IDF
    # -------------------
    vectorizer = TfidfVectorizer(
        max_features=20000,
        ngram_range=(1, 2),
        stop_words="english"
    )

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # -------------------
    # Step 3: Models
    # -------------------
    results = {}
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, solver="liblinear"),
        "Naive Bayes": MultinomialNB(),
        "Linear SVM": LinearSVC()
    }

    for name, model in models.items():
        trained_model = evaluate_and_report(model, X_train_tfidf, y_train, X_test_tfidf, y_test, name)
        results[name] = trained_model

    # -------------------
    # Step 4: Save models + vectorizer
    # -------------------
    out_dir = Path("models")
    out_dir.mkdir(exist_ok=True)

    joblib.dump(vectorizer, out_dir / "tfidf_vectorizer.pkl")
    for name, model in results.items():
        filename = name.lower().replace(" ", "_") + ".pkl"
        joblib.dump(model, out_dir / filename)

    print("\n✅ Models and vectorizer saved in 'models/'")


if __name__ == "__main__":
    main()
