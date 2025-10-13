import os
import json
import matplotlib.pyplot as plt
import pandas as pd

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

def load_metrics():
    metrics = {}
    for file in os.listdir(RESULTS_DIR):
        if file.endswith(".json"):
            model_name = file.replace("_metrics.json", "")
            with open(os.path.join(RESULTS_DIR, file), "r") as f:
                metrics[model_name] = json.load(f)
    return metrics

def summarize_metrics(metrics):
    # Convert dict to DataFrame
    df = pd.DataFrame(metrics).T
    df = df[["accuracy", "precision", "recall", "f1"]]  # ensure consistent order
    print("\n Model Performance Summary:\n")
    print(df.round(4))
    return df

def plot_metrics(df):
    ax = df.plot(kind="bar", figsize=(10, 6), rot=0)
    plt.title("Model Comparison (Accuracy, Precision, Recall, F1)")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.legend(title="Metric")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "model_comparison.png"))
    plt.show()
    print(f"Saved comparison plot to {os.path.join(RESULTS_DIR, 'model_comparison.png')}")

if __name__ == "__main__":
    metrics = load_metrics()
    if not metrics:
        print("No metrics found in results/. Run evaluate_models.py first.")
    else:
        df = summarize_metrics(metrics)
        plot_metrics(df)
