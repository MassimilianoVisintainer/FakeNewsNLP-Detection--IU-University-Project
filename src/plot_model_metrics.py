import numpy as np
import matplotlib.pyplot as plt

RESULTS_DIR = "results"
METRICS_FILE = f"{RESULTS_DIR}/all_metrics.npy"

# Load all metrics
all_metrics = np.load(METRICS_FILE, allow_pickle=True).item()

# Extract models and metrics
models = list(all_metrics.keys())
accuracy = [all_metrics[m]["accuracy"] for m in models]
f1 = [all_metrics[m]["f1"] for m in models]
roc_auc = [all_metrics[m]["roc_auc"] if all_metrics[m]["roc_auc"] is not None else 0 for m in models]

x = range(len(models))

# Plot metrics
plt.figure(figsize=(12, 6))
plt.plot(x, accuracy, marker='o', label="Accuracy")
plt.plot(x, f1, marker='s', label="F1 Score")
plt.plot(x, roc_auc, marker='^', label="ROC-AUC")
plt.xticks(x, models, rotation=45)
plt.ylim(0, 1.05)
plt.ylabel("Score")
plt.title("Model Comparison Metrics")
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()
plt.tight_layout()

# Save and show plot
plt.savefig(f"{RESULTS_DIR}/model_comparison_plot.png")
plt.show()

print(f"✅ Model comparison plot saved to {RESULTS_DIR}/model_comparison_plot.png")
