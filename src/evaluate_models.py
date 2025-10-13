import os
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

MODELS_DIR = "models"
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

def evaluate_model(name):
    """Load predictions and evaluate metrics."""
    print(f"Evaluating {name}...")

    y_test = np.load(os.path.join(MODELS_DIR, f"{name}_y_test.npy"))
    y_pred = np.load(os.path.join(MODELS_DIR, f"{name}_y_pred.npy"))
    y_proba_path = os.path.join(MODELS_DIR, f"{name}_y_proba.npy")

    if os.path.exists(y_proba_path):
        y_proba = np.load(y_proba_path)
    else:
        y_proba = None

    acc = accuracy_score(y_test, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="binary")
    auc = roc_auc_score(y_test, y_proba) if y_proba is not None else None

    metrics = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": auc,
    }

    # Save metrics
    np.save(os.path.join(RESULTS_DIR, f"{name}_metrics.npy"), metrics)
    print(f"✅ Saved metrics for {name}: {metrics}")
    return metrics

def main():
    models = [
        "tfidf",          # TF-IDF + Logistic Regression baseline
        "doc2vec",        # Doc2Vec + Logistic Regression
        "bert",           # BERT fine-tuned
        "svm",            # Linear SVM
        "random_forest",  # Random Forest
        "naive_bayes",    # Multinomial Naïve Bayes
        "xgboost",        # XGBoost
    ]

    all_metrics = {}
    for model in models:
        try:
            all_metrics[model] = evaluate_model(model)
        except Exception as e:
            print(f"Skipping {model}, error: {e}")

    # Save all results together
    np.save(os.path.join(RESULTS_DIR, "all_metrics.npy"), all_metrics)
    print("\n All metrics saved in results/all_metrics.npy")

if __name__ == "__main__":
    main()
