"""
train_random_forest_fast.py
---------------------------
Random Forest classifier with TF-IDF features (fast training)
"""

import os
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np

DATA_PATH = "data/processed/cleaned.csv"
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

def main():
    # Load dataset
    df = pd.read_csv(DATA_PATH)
    df["clean_text"] = df["clean_text"].fillna("")

    texts = df["clean_text"]
    labels = df["label"]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )

    # TF-IDF Vectorizer
    vectorizer = TfidfVectorizer(max_features=20000, ngram_range=(1, 2))
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # Random Forest (faster with limited estimators and n_jobs=-1)
    clf = RandomForestClassifier(
        n_estimators=200,  # fewer trees → faster
        max_depth=20,      # limit depth
        random_state=42,
        n_jobs=-1
    )
    clf.fit(X_train_tfidf, y_train)

    # Predictions
    y_pred = clf.predict(X_test_tfidf)
    y_proba = clf.predict_proba(X_test_tfidf)[:, 1]

    # Print report
    print(classification_report(y_test, y_pred))

    # Save
    joblib.dump(clf, os.path.join(MODELS_DIR, "random_forest.pkl"))
    joblib.dump(vectorizer, os.path.join(MODELS_DIR, "rf_vectorizer.pkl"))
    joblib.dump((X_test_tfidf, y_test), os.path.join(MODELS_DIR, "random_forest_test.pkl"))

    np.save(os.path.join(MODELS_DIR, "random_forest_y_test.npy"), y_test)
    np.save(os.path.join(MODELS_DIR, "random_forest_y_pred.npy"), y_pred)
    np.save(os.path.join(MODELS_DIR, "random_forest_y_proba.npy"), y_proba)

    print("Fast Random Forest model saved to /models")

if __name__ == "__main__":
    main()
