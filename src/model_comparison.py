# src/model_comparison.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Load all metrics
all_metrics_path = os.path.join(RESULTS_DIR, "all_metrics.npy")
all_metrics = np.load(all_metrics_path, allow_pickle=True).item()

# Convert to DataFrame
df = pd.DataFrame(all_metrics).T  # transpose so models are rows

# Save CSV
csv_path = os.path.join(RESULTS_DIR, "model_comparison.csv")
df.to_csv(csv_path)
print(f"✅ Model comparison saved to {csv_path}\n")
print(df)

# --- Plotting ---
metrics_to_plot = ["accuracy", "f1", "roc_auc"]
for metric in metrics_to_plot:
    plt.figure(figsize=(8,5))
    df[metric].plot(kind="bar", color="skyblue", edgecolor="black")
    plt.title(f"Model Comparison: {metric.capitalize()}")
    plt.ylabel(metric.capitalize())
    plt.ylim(0,1.05)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plot_path = os.path.join(RESULTS_DIR, f"{metric}_comparison.png")
    plt.savefig(plot_path)
    print(f"✅ Plot saved: {plot_path}")
    plt.close()
