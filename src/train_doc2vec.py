"""
train_doc2vec.py
----------------
Trains a Doc2Vec + Logistic Regression classifier for fake news detection.
"""

import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import joblib

# Paths
DATA_PATH = "data/processed/cleaned.csv"
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

def main():
    # Load dataset
    df = pd.read_csv(DATA_PATH)
    df["clean_text"] = df["clean_text"].fillna("")

    texts = df["clean_text"].tolist()
    labels = df["label"].tolist()

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )

    # Prepare data for Doc2Vec
    train_tagged = [TaggedDocument(words=t.split(), tags=[i]) for i, t in enumerate(X_train)]
    test_tagged = [TaggedDocument(words=t.split(), tags=[i]) for i, t in enumerate(X_test)]

    # Train Doc2Vec model
    model = Doc2Vec(vector_size=100, window=5, min_count=2, workers=4, epochs=20)
    model.build_vocab(train_tagged)
    model.train(train_tagged, total_examples=model.corpus_count, epochs=model.epochs)

    # Vectorize train/test sets
    X_train_vectors = [model.infer_vector(doc.words) for doc in train_tagged]
    X_test_vectors = [model.infer_vector(doc.words) for doc in test_tagged]

    # Logistic Regression classifier
    clf = LogisticRegression(max_iter=500)
    clf.fit(X_train_vectors, y_train)

    # Predictions
    y_pred = clf.predict(X_test_vectors)
    y_proba = clf.predict_proba(X_test_vectors)[:, 1]

    # Print metrics
    print(classification_report(y_test, y_pred))

    # Save everything
    joblib.dump(clf, os.path.join(MODELS_DIR, "doc2vec_logreg.pkl"))
    model.save(os.path.join(MODELS_DIR, "doc2vec_gensim.model"))  # save gensim model

    np.save(os.path.join(MODELS_DIR, "doc2vec_y_test.npy"), y_test)
    np.save(os.path.join(MODELS_DIR, "doc2vec_y_pred.npy"), y_pred)
    np.save(os.path.join(MODELS_DIR, "doc2vec_y_proba.npy"), y_proba)

    print("✅ Doc2Vec model and predictions saved to /models")

if __name__ == "__main__":
    main()
