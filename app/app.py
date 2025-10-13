import os
import sys
import joblib
import numpy as np
from flask import Flask, render_template, request


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.preprocessing import TextPreprocessor, PreprocessConfig
app = Flask(__name__)

# --- PATHS ---
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

# --- INITIALIZE PREPROCESSOR ---
config = PreprocessConfig(
    lowercase=True,
    remove_urls=True,
    remove_emails=True,
    remove_numbers=True,
    expand_contractions=True,
    remove_stopwords=True,
    lemmatize=True,
    keep_negations=True,
    remove_bylines=True,
    remove_html=True
)
preprocessor = TextPreprocessor(config)

# --- LOAD MODELS ---
models = {}

def load_model(name, model_file, vectorizer_file):
    model_path = os.path.join(MODELS_DIR, model_file)
    vect_path = os.path.join(MODELS_DIR, vectorizer_file)
    if os.path.exists(model_path) and os.path.exists(vect_path):
        models[name] = {
            "model": joblib.load(model_path),
            "vectorizer": joblib.load(vect_path)
        }

# Load available models
load_model("TF-IDF + Logistic Regression", "tfidf_model.pkl", "tfidf_vectorizer.pkl")
load_model("SVM (LinearSVC)", "svm.pkl", "svm_vectorizer.pkl")
load_model("Random Forest", "random_forest.pkl", "rf_vectorizer.pkl")
load_model("Naive Bayes", "naive_bayes.pkl", "nb_vectorizer.pkl")
load_model("XGBoost", "xgboost.pkl", "xgb_vectorizer.pkl")


# --- ROUTES ---
@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    text = ""
    selected_model = None
    label_color = None

    if request.method == "POST":
        text = request.form["text"]
        selected_model = request.form.get("model")

        if not selected_model or selected_model not in models:
            return render_template("index.html", models=models, error="Please select a model.", text=text)

        model_config = models[selected_model]
        model = model_config["model"]
        vectorizer = model_config["vectorizer"]

        # Preprocessing using TextPreprocessor
        clean_text = preprocessor.transform(text)
        X = vectorizer.transform([clean_text])
        pred = model.predict(X)[0]

        prediction = "Fake News 🟥" if pred == 1 else "Real News 🟩"
        label_color = "red" if pred == 1 else "green"

    return render_template(
        "index.html",
        models=models,
        prediction=prediction,
        selected_model=selected_model,
        text=text,
        label_color=label_color
    )


if __name__ == "__main__":
    app.run(debug=True)
