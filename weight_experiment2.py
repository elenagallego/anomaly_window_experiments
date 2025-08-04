import os
import time
import tracemalloc

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from statsmodels.tsa.stattools import acf
import stumpy

from dtaianomaly.preprocessing import MovingAverage
from dtaianomaly.anomaly_detection import LocalOutlierFactor
from dtaianomaly.thresholding import FixedCutoff
from dtaianomaly.evaluation import Precision, Recall, FBeta, AreaUnderROC, AreaUnderPR, ThresholdMetric
from dtaianomaly.data import UCRLoader
from eventwise_metrics import EventWisePrecision, EventWiseRecall, EventWiseFBeta

dataset_base_path = r"C:\Users\34622\Desktop\1MAI\thesiscode\datasets\UCR_Anomaly_FullData"
output_dir = r"C:\Users\34622\Desktop\1MAI\thesiscode\results\Weights\median_method\definite2\ExperimentSlope\ltstdbs"
os.makedirs(output_dir, exist_ok=True)

dataset_info = {  "180_UCR_Anomaly_ltstdbs30791ES_20000_52600_52800.txt": (52600,52800),
"179_UCR_Anomaly_ltstdbs30791AS_23000_52600_52800.txt": (52600,52800),
"178_UCR_Anomaly_ltstdbs30791AI_17555_52600_52800.txt": (52600, 52800)}

dataset_paths = {os.path.join(dataset_base_path, fname): rng for fname, rng in dataset_info.items()}
best_ws_dict = {p: 102 for p in dataset_paths.keys()}

colors = ["b", "g", "r", "c", "m", "orange", "limegreen", "darkred"]
colors_by_method = {"fft": "blue", "acf": "red", "mp": "green", "suss": "yellow"}


class WeightedLOFDetector:
    def __init__(self, slope: float, window_size: int, n_neighbors: int = 20):
        self.slope = slope
        self.window_size = window_size
        self.n_neighbors = n_neighbors
        self.thresholding = FixedCutoff(cutoff=0.85)

    def _generate_linear_weights(self):
        w = np.logspace(0, np.log10(self.slope), self.window_size)
        return w / w.sum()

    def _apply_weighting(self, ts):
        weights = self._generate_linear_weights()
        return np.convolve(ts, weights, mode='same')

    def fit_predict(self, X_train, X_test):
        Xtr_w = self._apply_weighting(X_train.ravel())
        Xte_w = self._apply_weighting(X_test.ravel())

        lof = LocalOutlierFactor(
            window_size=self.window_size,
            n_neighbors=self.n_neighbors
        )
        lof.fit(Xtr_w)
        scores = lof.decision_function(Xte_w)
        probs = (scores - scores.min()) / (scores.max() - scores.min())
        binary = self.thresholding.threshold(probs)
        return probs, binary


def run_experiments(dataset_path, anomaly_range, experiment_name):
    ws = best_ws_dict[dataset_path]
    dataset_name = os.path.basename(dataset_path)
    print(f"\n▶ Ejecutando {experiment_name} en {dataset_name} con ws={ws} puntos")

    data = UCRLoader(dataset_path).load()
    X_train, y_train = data.X_train, data.y_train
    X_test,  y_test  = data.X_test,  data.y_test

    pre = MovingAverage(window_size=10)
    Xtr_p, _ = pre.fit_transform(X_train)
    Xte_p, _ = pre.transform(X_test)

    slope_values = np.linspace(1.0, 20.0, 20)
    results = []

    for slope in slope_values:
        model = WeightedLOFDetector(slope=slope, window_size=ws, n_neighbors=10)


        tracemalloc.clear_traces()
        tracemalloc.start()
        start_time = time.time()

        y_pred, y_bin = model.fit_predict(Xtr_p, Xte_p)


        elapsed = time.time() - start_time
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        mem_mb = peak / (1024 * 1024)


        precision = Precision().compute(y_test, y_bin)
        recall    = Recall().compute(y_test, y_bin)
        f1        = ThresholdMetric(FixedCutoff(0.85), FBeta(1.0)).compute(y_test, y_pred)
        auc_roc   = AreaUnderROC().compute(y_test, y_pred)
        auc_pr    = AreaUnderPR().compute(y_test, y_pred)
        eprec     = EventWisePrecision().compute(y_test, y_bin)
        erec      = EventWiseRecall().compute(y_test, y_bin)
        ef05      = EventWiseFBeta(beta=0.5).compute(y_test, y_bin)

        results.append({
            "Dataset":        dataset_name,
            "Slope":          slope,
            "Precision":      precision,
            "Recall":         recall,
            "F1 Score":       f1,
            "AUC-ROC":        auc_roc,
            "AUC-PR":         auc_pr,
            "Event Precision": eprec,
            "Event Recall":    erec,
            "Event F1 Score":  ef05,
            "Time (s)":        round(elapsed, 4),
            "Memory (MB)":     round(mem_mb, 2),
        })

    return pd.DataFrame(results)


all_dfs = []
for path, rng in dataset_paths.items():
    df = run_experiments(path, rng, "Slope Sweep con best_ws")
    df["Dataset"] = os.path.basename(path)
    all_dfs.append(df)

combined = pd.concat(all_dfs, ignore_index=True)
output_csv = os.path.join(output_dir, "slope_sweep_results.csv")
combined.to_csv(output_csv, index=False)
print(f" Resultados guardados en {output_csv}")

import numpy as np

numeric_cols = combined.select_dtypes(include=[np.number]).columns.tolist()

numeric_cols.remove('Slope')

grouped = (
    combined
    .groupby("Slope")[numeric_cols]
    .agg(["mean", "std"])
    .reset_index()
)

metrics = [
    "Precision", "Recall", "F1 Score", "AUC-ROC",
    "AUC-PR", "Event Precision", "Event Recall", "Event F1 Score"
]

for (metric, color) in zip(metrics, colors):
    plt.figure(figsize=(10, 6))
    x = grouped["Slope"]                   # 1-D
    y = grouped[(metric, "mean")]
    yerr = grouped[(metric, "std")]
    plt.plot(x, y, marker="o", color=color, label=metric)
    plt.fill_between(x, y - yerr, y + yerr, color=color, alpha=0.2)
    plt.title(f"{metric} vs Slope (Mean ± Std)")
    plt.xlabel("Slope")
    plt.ylabel(metric)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

plt.figure(figsize=(12, 8))
for (metric, color) in zip(metrics, colors):
    x = grouped["Slope"]
    y = grouped[(metric, "mean")]
    plt.plot(x, y, marker="o", label=metric, color=color)
plt.title("All Metrics vs Slope (Mean)")
plt.xlabel("Slope")
plt.ylabel("Value")
plt.grid(True)
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize=(10, 4))
metrics_grp1 = ["Precision", "Recall", "F1 Score"]
for metric, color in zip(metrics_grp1, ["b", "g", "r"]):
    m = grouped[(metric, "mean")]
    s = grouped[(metric, "std")]
    ax.plot(grouped["Slope"], m, "o-", color=color, label=f"{metric} (mean)")
    ax.fill_between(grouped["Slope"], m - s, m + s, alpha=0.2, color=color)
ax.set(title="Precision, Recall & F1 Score vs Slope", xlabel="Slope", ylabel="Score")
ax.legend(); ax.grid(); fig.tight_layout()
fig.savefig(os.path.join(output_dir, "precision_recall_f1_vs_slope.png"))
plt.show()


fig, ax = plt.subplots(figsize=(10, 4))
metrics_grp2 = ["AUC-ROC", "AUC-PR"]
for metric, color in zip(metrics_grp2, ["c", "m"]):
    m = grouped[(metric, "mean")]
    s = grouped[(metric, "std")]
    ax.plot(grouped["Slope"], m, "o-", color=color, label=f"{metric} (mean)")
    ax.fill_between(grouped["Slope"], m - s, m + s, alpha=0.2, color=color)
ax.set(title="AUC-ROC & AUC-PR vs Slope", xlabel="Slope", ylabel="AUC Score")
ax.legend(); ax.grid(); fig.tight_layout()
fig.savefig(os.path.join(output_dir, "auc_roc_pr_vs_slope.png"))
plt.show()


fig, ax = plt.subplots(figsize=(10, 4))
metrics_grp3 = ["Event Precision", "Event Recall", "Event F1 Score"]
for metric, color in zip(metrics_grp3, ["orange", "limegreen", "darkred"]):
    m = grouped[(metric, "mean")]
    s = grouped[(metric, "std")]
    ax.plot(grouped["Slope"], m, "o-", color=color, label=f"{metric} (mean)")
    ax.fill_between(grouped["Slope"], m - s, m + s, alpha=0.2, color=color)
ax.set(title="Event-wise Precision, Recall & F1 vs Slope", xlabel="Slope", ylabel="Score")
ax.legend(); ax.grid(); fig.tight_layout()
fig.savefig(os.path.join(output_dir, "eventwise_metrics_vs_slope.png"))
plt.show()