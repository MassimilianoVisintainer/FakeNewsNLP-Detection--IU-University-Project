import os
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

MODELS_DIR = "models"
DATA_DIR = "data/processed"
os.makedirs("outputs", exist_ok=True)


def load_predictions(model_name):
    """Load test labels, predictions, and probabilities for a given model."""
    y_test = np.load(os.path.join(MODELS_DIR, f"{model_name}_y_test.npy"))
    y_pred = np.load(os.path.join(MODELS_DIR, f"{model_name}_y_pred.npy"))
    y_proba = np.load(os.path.join(MODELS_DIR, f"{model_name}_y_proba.npy"))
    return y_test, y_pred, y_proba


def analyze_model(model_name, test_texts):
    print(f"\n🔍 Error Analysis for {model_name.upper()}")

    # Load predictions
    y_test, y_pred, y_proba = load_predictions(model_name)

    # Classification report
    print("\n📊 Classification Report:")
    print(classification_report(y_test, y_pred, digits=4))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Real", "Fake"],
                yticklabels=["Real", "Fake"])
    plt.title(f"{model_name.upper()} - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(f"outputs/{model_name}_confusion_matrix.png")
    plt.close()

    print(f"✅ Confusion matrix saved: outputs/{model_name}_confusion_matrix.png")

    # Save misclassified examples
    misclassified = []
    for i, (true, pred) in enumerate(zip(y_test, y_pred)):
        if true != pred:
            misclassified.append({
                "text": test_texts[i],
                "true_label": true,
                "pred_label": pred,
                "proba_fake": y_proba[i]
            })

    df_errors = pd.DataFrame(misclassified)
    csv_path = f"outputs/{model_name}_misclassified.csv"
    df_errors.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"⚠️ Misclassified examples saved: {csv_path}")


if __name__ == "__main__":
    # Load test texts for qualitative error analysis
    test_data = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    test_texts = test_data["clean_text"].fillna("").tolist()

    for model in ["tfidf", "doc2vec", "bert"]:
        try:
            analyze_model(model, test_texts)
        except FileNotFoundError:
            print(f"⚠️ Skipping {model} (predictions not found)")
