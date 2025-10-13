import os
import numpy as np
import matplotlib.pyplot as plt

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

def plot_metrics():
    metrics_path = os.path.join(RESULTS_DIR, "all_metrics.npy")
    if not os.path.exists(metrics_path):
        raise FileNotFoundError("Run evaluate_models.py first to generate all_metrics.npy")

    all_metrics = np.load(metrics_path, allow_pickle=True).item()

    models = list(all_metrics.keys())
    metrics_names = ["accuracy", "precision", "recall", "f1", "roc_auc"]

    # Individual plots (per metric)
    for metric in metrics_names:
        values = [all_metrics[m].get(metric, None) for m in models if all_metrics[m] is not None]

        plt.figure(figsize=(8, 5))
        plt.bar(models, values, color="skyblue")
        plt.title(f"Model Comparison - {metric.upper()}")
        plt.ylabel(metric)
        plt.xticks(rotation=30, ha="right")
        plt.ylim(0, 1)  # all metrics are between 0 and 1
        plt.tight_layout()

        out_path = os.path.join(RESULTS_DIR, f"{metric}_comparison.png")
        plt.savefig(out_path)
        plt.close()
        print(f"Saved {metric} plot -> {out_path}")

    # Combined grouped bar chart
    x = np.arange(len(models))
    width = 0.15  # width of each bar

    plt.figure(figsize=(10, 6))
    for i, metric in enumerate(metrics_names):
        values = [all_metrics[m].get(metric, None) for m in models if all_metrics[m] is not None]
        plt.bar(x + i * width, values, width, label=metric)

    plt.xticks(x + width * (len(metrics_names) - 1) / 2, models, rotation=30, ha="right")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.title("Model Comparison - All Metrics")
    plt.legend()
    plt.tight_layout()

    out_path = os.path.join(RESULTS_DIR, "all_metrics_comparison.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Saved combined metrics plot -> {out_path}")

if __name__ == "__main__":
    plot_metrics()
