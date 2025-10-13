import os
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from xgboost import XGBClassifier

# Paths
DATA_PATH = "data/processed/cleaned.csv"
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

def load_data():
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["clean_text", "label"])
    return df["clean_text"].astype(str).tolist(), df["label"].astype(int).tolist()

def main():
    print("Loading dataset...")
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Extracting TF-IDF features...")
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # Save vectorizer
    joblib.dump(vectorizer, os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl"))

    models = {
        "svm": LinearSVC(random_state=42),
        "random_forest": RandomForestClassifier(n_estimators=200, random_state=42),
        "naive_bayes": MultinomialNB(),
        "xgboost": XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42
        ),
    }

    for name, model in models.items():
        print(f"\n Training {name}...")
        model.fit(X_train_tfidf, y_train)

        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test_tfidf)[:, 1]
        else:
            # Some models like LinearSVC don't support predict_proba
            y_proba = np.zeros(len(y_test))

        y_pred = model.predict(X_test_tfidf)

        # Save model + predictions
        joblib.dump(model, os.path.join(MODELS_DIR, f"{name}.pkl"))
        np.save(os.path.join(MODELS_DIR, f"{name}_y_test.npy"), y_test)
        np.save(os.path.join(MODELS_DIR, f"{name}_y_pred.npy"), y_pred)
        np.save(os.path.join(MODELS_DIR, f"{name}_y_proba.npy"), y_proba)

        print(f" {name} model and predictions saved to /models")

if __name__ == "__main__":
    main()
