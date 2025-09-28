import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from pathlib import Path

def main():
    # Step 1: Load train/test
    train = pd.read_csv("data/processed/train.csv")
    test = pd.read_csv("data/processed/test.csv")

    # Step 2: Extract features & labels
    X_train = train["clean_text"].fillna("")
    y_train = train["label"]

    X_test = test["clean_text"].fillna("")
    y_test = test["label"]

    # Step 3: TF-IDF vectorizer
    vectorizer = TfidfVectorizer(
        max_features=20000,   # limit vocabulary size
        ngram_range=(1,2),    # use unigrams + bigrams
        stop_words="english"  # extra filtering
    )

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # Step 4: Logistic Regression
    clf = LogisticRegression(max_iter=1000, solver="liblinear")
    clf.fit(X_train_tfidf, y_train)

    # Step 5: Evaluate
    preds = clf.predict(X_test_tfidf)

    acc = accuracy_score(y_test, preds)
    print(f"Accuracy: {acc:.4f}")
    print(classification_report(y_test, preds, digits=3))

    # Step 6: Save model + vectorizer
    import joblib
    out_dir = Path("models")
    out_dir.mkdir(exist_ok=True)

    joblib.dump(clf, out_dir / "logreg_tfidf.pkl")
    joblib.dump(vectorizer, out_dir / "tfidf_vectorizer.pkl")

    print("Model and vectorizer saved in 'models/'")

if __name__ == "__main__":
    main()
