"""
train_bert_embeddings.py
------------------------
Use DistilBERT embeddings + Logistic Regression for Fake News Detection.

Steps:
- Load preprocessed dataset
- Convert texts to BERT embeddings
- Train a classifier (LogReg)
- Evaluate performance
"""

import pandas as pd
import torch
from transformers import DistilBertTokenizer, DistilBertModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from pathlib import Path
import joblib
import numpy as np


def get_bert_embeddings(texts, tokenizer, model, batch_size=16, max_length=128):
    """Convert a list of texts into BERT embeddings."""
    embeddings = []
    model.eval()

    # Process in batches
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]

        # Tokenize
        encodings = tokenizer(
            batch,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt"
        )

        with torch.no_grad():
            outputs = model(**encodings)
            # CLS token embedding (first token) → representation of whole sequence
            cls_embeddings = outputs.last_hidden_state[:, 0, :]

        embeddings.append(cls_embeddings.cpu().numpy())

    return np.vstack(embeddings)


def main():
    # -------------------
    # Step 1: Load data
    # -------------------
    train = pd.read_csv("data/processed/train.csv")
    test = pd.read_csv("data/processed/test.csv")

    X_train = train["clean_text"].fillna("").tolist()
    y_train = train["label"].tolist()

    X_test = test["clean_text"].fillna("").tolist()
    y_test = test["label"].tolist()

    # -------------------
    # Step 2: Load DistilBERT
    # -------------------
    print("Loading DistilBERT model...")
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    bert_model = DistilBertModel.from_pretrained("distilbert-base-uncased")

    # -------------------
    # Step 3: Generate embeddings
    # -------------------
    print("Encoding training set...")
    X_train_emb = get_bert_embeddings(X_train, tokenizer, bert_model)

    print("Encoding test set...")
    X_test_emb = get_bert_embeddings(X_test, tokenizer, bert_model)

    # -------------------
    # Step 4: Train classifier
    # -------------------
    clf = LogisticRegression(max_iter=1000, solver="liblinear")
    clf.fit(X_train_emb, y_train)

    # -------------------
    # Step 5: Evaluate
    # -------------------
    preds = clf.predict(X_test_emb)
    acc = accuracy_score(y_test, preds)
    print(f"\nAccuracy: {acc:.4f}")
    print(classification_report(y_test, preds, digits=3))

    # -------------------
    # Step 6: Save model + classifier
    # -------------------
    out_dir = Path("models")
    out_dir.mkdir(exist_ok=True)

    joblib.dump(clf, out_dir / "logreg_bert.pkl")
    tokenizer.save_pretrained(out_dir / "distilbert_tokenizer")
    bert_model.save_pretrained(out_dir / "distilbert_model")

    print("\n✅ Saved classifier + BERT model in 'models/'")


if __name__ == "__main__":
    main()
