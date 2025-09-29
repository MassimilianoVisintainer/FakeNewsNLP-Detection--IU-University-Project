"""
train_tfidf_baseline.py
-----------------------
Baseline model for Fake News Detection using TF-IDF + Logistic Regression.

Steps:
- Load preprocessed train/test splits
- Transform text using TF-IDF (unigrams + bigrams)
- Train Logistic Regression
- Evaluate on test set
- Save model and vectorizer
- Inspect top features
"""

import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import joblib


def show_top_features(vectorizer, model, n=20):
    """Display top words most indicative of fake vs real news."""
    feature_names = vectorizer.get_feature_names_out()
    coefs = model.coef_[0]

    # Fake = positive class (label=1), Real = negative class (label=0)
    top_fake = sorted(zip(coefs, feature_names), reverse=True)[:n]
    top_real = sorted(zip(coefs, feature_names))[:n]

    print("\nTop features for FAKE news:")
    for coef, feat in top_fake:
        print(f"{feat:20s} {coef:.4f}")

    print("\nTop features for REAL news:")
    for coef, feat in top_real:
        print(f"{feat:20s} {coef:.4f}")


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
        max_features=20000,   # limit vocab size
        ngram_range=(1, 2),   # unigrams + bigrams
        stop_words="english"
    )

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # -------------------
    # Step 3: Train model
    # -------------------
    clf = LogisticRegression(max_iter=1000, solver="liblinear")
    clf.fit(X_train_tfidf, y_train)

    # -------------------
    # Step 4: Evaluate
    # -------------------
    preds = clf.predict(X_test_tfidf)

    acc = accuracy_score(y_test, preds)
    print(f"\nAccuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, preds, digits=3))

    # -------------------
    # Step 5: Save model
    # -------------------
    out_dir = Path("models")
    out_dir.mkdir(exist_ok=True)

    joblib.dump(clf, out_dir / "logreg_tfidf.pkl")
    joblib.dump(vectorizer, out_dir / "tfidf_vectorizer.pkl")
    print("\n✅ Model and vectorizer saved in 'models/'")

    # -------------------
    # Step 6: Feature inspection
    # -------------------
    show_top_features(vectorizer, clf, n=20)


if __name__ == "__main__":
    main()
