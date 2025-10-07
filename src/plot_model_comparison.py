# src/plot_model_comparison.py
import numpy as np
import matplotlib.pyplot as plt

RESULTS_DIR = "results"
all_metrics = np.load(f"{RESULTS_DIR}/all_metrics.npy", allow_pickle=True).item()

# Focus on key metrics: Accuracy, F1, ROC-AUC
metrics_to_plot = ["accuracy", "f1", "roc_auc"]

models = list(all_metrics.keys())
values = {metric: [all_metrics[m].get(metric, 0) for m in models] for metric in metrics_to_plot}

# Plotting
x = range(len(models))
plt.figure(figsize=(10,6))
for metric in metrics_to_plot:
    plt.plot(x, values[metric], marker='o', label=metric)

plt.xticks(x, models, rotation=45)
plt.ylim(0, 1.05)
plt.ylabel("Score")
plt.title("Model Comparison Metrics")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/model_comparison.png")
plt.show()
