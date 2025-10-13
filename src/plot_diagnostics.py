import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix, ConfusionMatrixDisplay

MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

MODELS = ["tfidf", "doc2vec", "bert", "naive_bayes", "random_forest", "svm", "xgboost"]  

RESULTS_DIR = "results"


def load_predictions(model_name):
    """Load saved predictions and labels for a given model if available."""
    try:
        y_test = np.load(os.path.join(MODELS_DIR, f"{model_name}_y_test.npy"))
        y_pred = np.load(os.path.join(MODELS_DIR, f"{model_name}_y_pred.npy"))
        y_proba = np.load(os.path.join(MODELS_DIR, f"{model_name}_y_proba.npy"))
        return y_test, y_pred, y_proba
    except FileNotFoundError:
        print(f"Skipping {model_name.upper()} – no saved predictions found.")
        return None, None, None


def plot_roc_curves():
    print("Plotting ROC curves...")
    plt.figure(figsize=(8, 6))

    for model in MODELS:
        y_test, _, y_proba = load_predictions(model)
        if y_test is None:
            continue

        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f"{model.upper()} (AUC = {roc_auc:.2f})")

    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")
    plt.legend()
    plt.savefig(os.path.join(RESULTS_DIR, "roc_curves.png"))
    plt.close()


def plot_pr_curves():
    print("Plotting Precision-Recall curves...")
    plt.figure(figsize=(8, 6))

    for model in MODELS:
        y_test, _, y_proba = load_predictions(model)
        if y_test is None:
            continue

        precision, recall, _ = precision_recall_curve(y_test, y_proba)
        plt.plot(recall, precision, lw=2, label=f"{model.upper()}")

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curves")
    plt.legend()
    plt.savefig(os.path.join(RESULTS_DIR, "pr_curves.png"))
    plt.close()


def plot_confusion_matrices():
    print(" Plotting confusion matrices...")
    for model in MODELS:
        y_test, y_pred, _ = load_predictions(model)
        if y_test is None:
            continue

        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap="Blues", values_format="d")
        plt.title(f"Confusion Matrix - {model.upper()}")
        plt.savefig(os.path.join(RESULTS_DIR, f"cm_{model}.png"))
        plt.close()


if __name__ == "__main__":
    plot_roc_curves()
    plot_pr_curves()
    plot_confusion_matrices()
    print(" All plots saved in /models")
