from flask import Flask, render_template, request
import joblib
import os

app = Flask(__name__)

MODELS_DIR = "../models"

# Define available models
models = {
    "tfidf": {"name": "TF-IDF + Logistic Regression", "file": "tfidf_logreg.pkl", "vectorizer": "tfidf_vectorizer.pkl"},
    "svm": {"name": "SVM (Linear)", "file": "svm.pkl", "vectorizer": "svm_vectorizer.pkl"},
    "random_forest": {"name": "Random Forest", "file": "random_forest.pkl", "vectorizer": "rf_vectorizer.pkl"},
    "naive_bayes": {"name": "Naive Bayes", "file": "naive_bayes.pkl", "vectorizer": "nb_vectorizer.pkl"},
    "xgboost": {"name": "XGBoost", "file": "xgboost.pkl", "vectorizer": "xgb_vectorizer.pkl"},
}

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    text = ""
    selected_model = "tfidf"  # default model

    if request.method == "POST":
        text = request.form.get("news_text", "")
        selected_model = request.form.get("model", "tfidf")  # ✅ safe fallback

        if text.strip():
            try:
                cfg = models[selected_model]
                clf = joblib.load(os.path.join(MODELS_DIR, cfg["file"]))
                vectorizer = joblib.load(os.path.join(MODELS_DIR, cfg["vectorizer"]))

                X_input = vectorizer.transform([text])
                pred = clf.predict(X_input)[0]

                prediction = "Fake News ❌" if pred == 1 else "Real News ✅"
            except Exception as e:
                prediction = f"⚠️ Error: {str(e)}"

    return render_template(
        "index.html",
        models=models,
        selected_model=selected_model,
        prediction=prediction,
        text=text
    )

if __name__ == "__main__":
    app.run(debug=True)
