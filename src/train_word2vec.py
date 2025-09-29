"""
train_word2vec.py
-----------------
Train a Word2Vec embedding model and use averaged word embeddings
with Logistic Regression for Fake News Detection.
"""

import pandas as pd
from gensim.models import Word2Vec
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from pathlib import Path
import joblib


def build_word2vec(sentences, vector_size=100, window=5, min_count=2, workers=4):
    """Train a Word2Vec model from tokenized sentences."""
    model = Word2Vec(
        sentences=sentences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers
    )
    return model


def vectorize_sentences(sentences, model):
    """Convert list of tokenized sentences into averaged word vectors."""
    vectors = []
    for tokens in sentences:
        vecs = [model.wv[word] for word in tokens if word in model.wv]
        if vecs:
            vectors.append(sum(vecs) / len(vecs))
        else:
            vectors.append([0.0] * model.vector_size)
    return vectors


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
    # Step 2: Train Word2Vec
    # -------------------
    print("Training Word2Vec model...")
    w2v_model = build_word2vec(train_tokens)

    # -------------------
    # Step 3: Vectorize documents
    # -------------------
    X_train = vectorize_sentences(train_tokens, w2v_model)
    X_test = vectorize_sentences(test_tokens, w2v_model)

    # -------------------
    # Step 4: Train classifier
    # -------------------
    clf = LogisticRegression(max_iter=1000, solver="liblinear")
    clf.fit(X_train, y_train)

    # -------------------
    # Step 5: Evaluate
    # -------------------
    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"\nAccuracy: {acc:.4f}")
    print(classification_report(y_test, preds, digits=3))

    # -------------------
    # Step 6: Save models
    # -------------------
    out_dir = Path("models")
    out_dir.mkdir(exist_ok=True)

    w2v_model.save(str(out_dir / "word2vec.model"))
    joblib.dump(clf, out_dir / "logreg_word2vec.pkl")

    print("\n✅ Word2Vec model + classifier saved in 'models/'")


if __name__ == "__main__":
    main()
