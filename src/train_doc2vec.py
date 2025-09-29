"""
train_doc2vec.py
----------------
Train a Doc2Vec embedding model for Fake News Detection.
Use learned document embeddings with Logistic Regression.
"""

import pandas as pd
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from pathlib import Path
import joblib


def build_doc2vec(tagged_docs, vector_size=100, window=5, min_count=2, workers=4, epochs=20):
    """Train a Doc2Vec model from tagged documents."""
    model = Doc2Vec(
        documents=tagged_docs,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers,
        epochs=epochs
    )
    return model


def main():
    # -------------------
    # Step 1: Load data
    # -------------------
    train = pd.read_csv("data/processed/train.csv")
    test = pd.read_csv("data/processed/test.csv")

    # Tokenize (split by space since we already preprocessed)
    train_tokens = train["clean_text"].fillna("").apply(str.split).tolist()
    test_tokens = test["clean_text"].fillna("").apply(str.split).tolist()

    y_train = train["label"]
    y_test = test["label"]

    # -------------------
    # Step 2: Create TaggedDocuments
    # -------------------
    tagged_train = [TaggedDocument(words=tokens, tags=[i]) for i, tokens in enumerate(train_tokens)]

    # -------------------
    # Step 3: Train Doc2Vec
    # -------------------
    print("Training Doc2Vec model...")
    d2v_model = build_doc2vec(tagged_train)

    # -------------------
    # Step 4: Vectorize documents
    # -------------------
    X_train = [d2v_model.dv[i] for i in range(len(tagged_train))]
    X_test = [d2v_model.infer_vector(tokens) for tokens in test_tokens]

    # -------------------
    # Step 5: Train classifier
    # -------------------
    clf = LogisticRegression(max_iter=1000, solver="liblinear")
    clf.fit(X_train, y_train)

    # -------------------
    # Step 6: Evaluate
    # -------------------
    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"\nAccuracy: {acc:.4f}")
    print(classification_report(y_test, preds, digits=3))

    # -------------------
    # Step 7: Save models
    # -------------------
    out_dir = Path("models")
    out_dir.mkdir(exist_ok=True)

    d2v_model.save(str(out_dir / "doc2vec.model"))
    joblib.dump(clf, out_dir / "logreg_doc2vec.pkl")

    print("\n✅ Doc2Vec model + classifier saved in 'models/'")


if __name__ == "__main__":
    main()
