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
from dtaianomaly.anomaly_detection import MedianMethod
from dtaianomaly.anomaly_detection import MatrixProfileDetector
from dtaianomaly.anomaly_detection import LocalOutlierFactor
from dtaianomaly.pipeline import Pipeline
from dtaianomaly.thresholding import FixedCutoff
from dtaianomaly.evaluation import Precision, Recall, FBeta, AreaUnderROC, AreaUnderPR, ThresholdMetric
from dtaianomaly.data import UCRLoader
from eventwise_metrics import EventWisePrecision, EventWiseRecall, EventWiseFBeta

dataset_base_path = r"C:\Users\34622\Desktop\1MAI\thesiscode\datasets\UCR_Anomaly_FullData"
output_dir = r"C:\Users\34622\Desktop\1MAI\thesiscode\results\Weights\median_method\definite2\Experiment1\ltstdbs"
os.makedirs(output_dir, exist_ok=True)

dataset_info = {  "180_UCR_Anomaly_ltstdbs30791ES_20000_52600_52800.txt": (52600,52800),
"179_UCR_Anomaly_ltstdbs30791AS_23000_52600_52800.txt": (52600,52800),
"178_UCR_Anomaly_ltstdbs30791AI_17555_52600_52800.txt": (52600, 52800)}

"""dataset_info = {
    "044_UCR_Anomaly_DISTORTEDPowerDemand1_9000_18485_18821.txt": (18485, 18821),
    "045_UCR_Anomaly_DISTORTEDPowerDemand2_14000_23357_23717.txt": (23357, 23717),
    "046_UCR_Anomaly_DISTORTEDPowerDemand3_16000_23405_23477.txt": (23405, 23477),
    "047_UCR_Anomaly_DISTORTEDPowerDemand4_18000_24005_24077.txt": (24005, 24077)
}"""
"""dataset_info = {
    "119_UCR_Anomaly_ECG1_10000_11800_12100.txt": (11800,12100),
    "120_UCR_Anomaly_ECG2_15000_16000_16100.txt": (16000,16100),
    "122_UCR_Anomaly_ECG3_8000_17000_17100.txt":(17000,17100),
    "123_UCR_Anomaly_ECG4_5000_16800_17100.txt": (16800,17100)
}"""

"""dataset_info = {
    "157_UCR_Anomaly_TkeepFirstMARS_3500_5365_5380.txt": (5365, 5380),
    "160_UCR_Anomaly_TkeepThirdMARS_3500_4711_4809.txt": (4711, 4809),
    "158_UCR_Anomaly_TkeepForthMARS_3500_5988_6085.txt": (5988, 6085),
    "159_UCR_Anomaly_TkeepSecondMARS_3500_9330_9340.txt": (9330, 9340),
    "156_UCR_Anomaly_TkeepFifthMARS_3500_5988_6085.txt": (5988, 6085)
}"""
"""dataset_info = {  "180_UCR_Anomaly_ltstdbs30791ES_20000_52600_52800.txt": (52600,52800),
"179_UCR_Anomaly_ltstdbs30791AS_23000_52600_52800.txt": (52600,52800),
"178_UCR_Anomaly_ltstdbs30791AI_17555_52600_52800.txt": (52600, 52800)}"""

"""dataset_info = { 
"135_UCR_Anomaly_InternalBleeding16_1200_4187_4199.txt": (4187,4199),
"136_UCR_Anomaly_InternalBleeding17_1600_3198_3309.txt": (3198, 3309),
"137_UCR_Anomaly_InternalBleeding18_2300_4485_4587.txt": (4485, 4587),
"138_UCR_Anomaly_InternalBleeding19_3000_4187_4197.txt": (4187, 4197),
"139_UCR_Anomaly_InternalBleeding20_2700_5759_5919.txt": (5759, 5919),
"""

dataset_paths = {os.path.join(dataset_base_path, file): info for file, info in dataset_info.items()}

normalized_ws_global = np.logspace(np.log10(1/4), np.log10(3), num=15)

def estimate_window_acf(ts, max_lag=300):
    autocorr = acf(ts, nlags=max_lag)
    skip = 5
    lag = np.argmax(autocorr[skip:]) + skip
    return lag

def estimate_window_fft(ts):
    ts_detrended = ts - np.mean(ts)
    fft_vals = np.fft.fft(ts_detrended)
    freqs = np.fft.fftfreq(len(ts))
    pos_mask = (freqs > 0)
    pos_freqs = freqs[pos_mask]
    fft_mags = np.abs(fft_vals)[pos_mask]
    dominant_freq = pos_freqs[np.argmax(fft_mags)]
    if dominant_freq == 0:
        return None
    return int(round(1 / dominant_freq))

def estimate_window_matrix_profile(ts, min_w=20, max_w=200):
    best_w = min_w
    best_score = np.inf
    for w in range(min_w, max_w + 1, 10):
        profile = stumpy.stump(ts, m=w)
        mean_dist = np.mean(profile[:, 0])
        score = mean_dist / np.sqrt(w)
        if score < best_score:
            best_score = score
            best_w = w
    return best_w

def estimate_window_suss(ts, min_w=20, max_w=200):
    ts_mean = np.mean(ts)
    ts_std = np.std(ts)
    best_w = min_w
    best_score = np.inf
    for w in range(min_w, max_w, 10):
        stats = []
        for i in range(0, len(ts) - w + 1, w):
            window = ts[i:i+w]
            stats.append([np.mean(window), np.std(window)])
        stats = np.array(stats)
        dist = np.mean((stats[:, 0] - ts_mean) ** 2 + (stats[:, 1] - ts_std) ** 2)
        if dist < best_score:
            best_score = dist
            best_w = w
    return best_w

def apply_weighted_convolution(data, weight_type="none", window_size=10):
    if weight_type == "none":
        return data
    if weight_type == "mild":
        weights = np.logspace(0, 0.3, window_size)
    elif weight_type == "moderate":
        weights = np.logspace(0, 0.48, window_size)
    elif weight_type == "strong":
        weights = np.logspace(0, 0.7, window_size)
    elif weight_type == 'very strong':
        weights = np.logspace(0, np.log10(7.5), window_size)
    else:
        raise ValueError(f"Unknown weighting type: {weight_type}")
    weights /= weights.sum()
    return np.convolve(data, weights, mode='same')

def run_experiments(dataset_path, anomaly_range, normalized_ws_set, experiment_name):
    dataset_name = os.path.basename(dataset_path)
    print(f"\nRunning {experiment_name} experiments on {dataset_name}...\n")

    if not os.path.exists(dataset_path):
        print(f" ERROR: Dataset file not found: {dataset_path}")
        return None

    data = UCRLoader(dataset_path).load()
    X_train, y_train = data.X_train, data.y_train
    X_test, y_test = data.X_test, data.y_test

    preprocessor = MovingAverage(window_size=10)
    X_train_, y_train_ = preprocessor.fit_transform(X_train)
    X_test_, y_test_ = preprocessor.transform(X_test)

    anomaly_length = anomaly_range[1] - anomaly_range[0]
    all_weighting_schemes = ['none', 'mild', 'moderate', 'strong', 'very strong']
    results = []

    for weight_type in all_weighting_schemes:
        print(f"▶ Weighting scheme: {weight_type}")
        for norm_ws in normalized_ws_set:
            ws = int(round(norm_ws * anomaly_length))
            if ws < 3:
                continue
            print(f"  → Running with window size: {ws} (normalized: {norm_ws:.4f})")
            X_train_weighted = apply_weighted_convolution(X_train_.ravel(), weight_type, ws)
            X_test_weighted = apply_weighted_convolution(X_test_.ravel(), weight_type, ws)

            tracemalloc.stop()
            tracemalloc.clear_traces()
            tracemalloc.start()

            start_time = time.time()

            detector = MedianMethod(ws)
            y_pred = detector.fit(X_train_weighted).predict_proba(X_test_weighted)

            prediction_time = time.time() - start_time
            current, peak = tracemalloc.get_traced_memory()
            memory_usage_mb = peak / (1024 * 1024)

            tracemalloc.stop()
            tracemalloc.clear_traces()
            thresholding = FixedCutoff(cutoff=0.85)
            y_pred_binary = thresholding.threshold(y_pred)

            precision = Precision().compute(y_test, y_pred_binary)
            recall = Recall().compute(y_test, y_pred_binary)
            f1 = ThresholdMetric(thresholding, FBeta(1.0)).compute(y_test, y_pred)
            auc_roc = AreaUnderROC().compute(y_test, y_pred)
            auc_pr = AreaUnderPR().compute(y_test, y_pred)
            event_precision = EventWisePrecision().compute(y_test, y_pred_binary)
            event_recall = EventWiseRecall().compute(y_test, y_pred_binary)
            event_f05 = EventWiseFBeta(beta=0.5).compute(y_test, y_pred_binary)

            results.append({
                "Weighting": weight_type,
                "Window Size": ws,
                "Normalized Window Size": norm_ws,
                "Precision": precision,
                "Recall": recall,
                "F1 Score": f1,
                "AUC-ROC": auc_roc,
                "AUC-PR": auc_pr,
                "Event Precision": event_precision,
                "Event Recall": event_recall,
                "Event F0.5 Score": event_f05,
                "Prediction Time (s)": round(prediction_time, 4),
                "Memory Usage (MB)": round(memory_usage_mb, 2)
            })

    return pd.DataFrame(results)

precomputed_ws = {'fft': [], 'acf': [], 'mp': [], 'suss': []}
for dataset_path, _ in dataset_paths.items():
    ts = UCRLoader(dataset_path).load().X_train.ravel()
    precomputed_ws['fft'].append(estimate_window_fft(ts))
    precomputed_ws['acf'].append(estimate_window_acf(ts))
    precomputed_ws['mp'].append(estimate_window_matrix_profile(ts))
    precomputed_ws['suss'].append(estimate_window_suss(ts))

window_stats = {method: (np.mean(ws), np.std(ws)) for method, ws in precomputed_ws.items()}

all_results = []
for dataset_path, anomaly_range in dataset_paths.items():
    df_results = run_experiments(dataset_path, anomaly_range, normalized_ws_global, "Normalized Context Windows")
    all_results.append(df_results)

combined_results = pd.concat(all_results, ignore_index=True)
grouped_results = combined_results.groupby(["Normalized Window Size", "Weighting"]).agg(['mean', 'std']).reset_index()
output_file = os.path.join(output_dir, "aggregated_results_weighted.csv")
grouped_results.to_csv(output_file, index=False)

avg_f1_df = grouped_results.groupby("Normalized Window Size")[[("F1 Score", "mean")]].mean()

avg_f1_by_norm = avg_f1_df[("F1 Score", "mean")]

best_norm_ws = avg_f1_by_norm.idxmax()
best_avg_f1 = avg_f1_by_norm.max()

print(f" Best normalized window size: {best_norm_ws:.4f}  with Avg F1={best_avg_f1:.4f}")

anomaly_lengths = [end - start for _, (start, end) in dataset_paths.items()]
mean_anomaly_length = np.mean(anomaly_lengths)

best_abs_ws = int(round(best_norm_ws * mean_anomaly_length))
print(f"   → Approx. absolute window size (mean over datasets): {best_abs_ws} points")

metrics = ["Precision", "Recall", "F1 Score", "AUC-ROC", "AUC-PR","Event Precision", "Event Recall", "Event F0.5 Score"]
weighting_styles = ['none', 'mild', 'moderate', 'strong', 'very strong']
colors = ["b", "g", "r", "c", "m","orange", "limegreen", "darkred"]
method_colors = {'fft': 'blue', 'acf': 'red', 'mp': 'green', 'suss': 'yellow'}

normalized_window_means = {
    method: mean / 100.0
    for method, (mean, std) in window_stats.items()
}

for metric, color in zip(metrics, colors):
    plt.figure(figsize=(10, 6))
    for weight in weighting_styles:
        subset = grouped_results[grouped_results["Weighting"] == weight]
        x = subset["Normalized Window Size"]
        y = subset[(metric, 'mean')]
        yerr = subset[(metric, 'std')]
        plt.plot(x, y, marker='o', label=weight)
        plt.fill_between(x, y - yerr, y + yerr, alpha=0.2)

    for method, norm_ws in normalized_window_means.items():
        plt.axvline(x=norm_ws, color=method_colors[method], label=f'{method.upper()} estimate')

    plt.title(f"{metric} (Mean ± Std)")
    plt.xlabel("Normalized Window Size")
    plt.ylabel(metric)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{metric}_plot_with_std.png"))
    plt.show()

for metric, color in zip(metrics, colors):
    plt.figure(figsize=(10, 6))
    for weight in weighting_styles:
        subset = grouped_results[grouped_results["Weighting"] == weight]
        x = subset["Normalized Window Size"]
        y = subset[(metric, 'mean')]
        plt.plot(x, y, marker='o', label=weight)

    for method, norm_ws in normalized_window_means.items():
        plt.axvline(x=norm_ws, color=method_colors[method], label=f'{method.upper()} estimate')

    plt.title(f"{metric} (Mean Only)")
    plt.xlabel("Normalized Window Size")
    plt.ylabel(metric)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{metric}_plot_mean_only.png"))
    plt.show()


fig, axs = plt.subplots(len(metrics), 1, figsize=(12, 25))
for i, (metric, color) in enumerate(zip(metrics, colors)):
    ax = axs[i]
    for weight in weighting_styles:
        subset = grouped_results[grouped_results["Weighting"] == weight]
        x = subset["Normalized Window Size"]
        y = subset[(metric, 'mean')]
        yerr = subset[(metric, 'std')]
        ax.plot(x, y, marker='o', label=weight)
        ax.fill_between(x, y - yerr, y + yerr, alpha=0.2)
    for method, norm_ws in normalized_window_means.items():
        ax.axvline(x=norm_ws, color=method_colors[method], label=f'{method.upper()} estimate' if i == 0 else None)
    ax.set_title(f"{metric} (Mean ± Std)")
    ax.set_xlabel("Normalized Window Size")
    ax.set_ylabel(metric)
    ax.grid(True)
    if i == 0:
        ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "Aggregated_Plot_All_Metrics_with_std.png"))
plt.show()


fig, axs = plt.subplots(len(metrics), 1, figsize=(12, 25))
for i, (metric, color) in enumerate(zip(metrics, colors)):
    ax = axs[i]
    for weight in weighting_styles:
        subset = grouped_results[grouped_results["Weighting"] == weight]
        x = subset["Normalized Window Size"]
        y = subset[(metric, 'mean')]
        ax.plot(x, y, marker='o', label=weight)
    for method, norm_ws in normalized_window_means.items():
        ax.axvline(x=norm_ws, color=method_colors[method], label=f'{method.upper()} estimate' if i == 0 else None)
    ax.set_title(f"{metric} (Mean Only)")
    ax.set_xlabel("Normalized Window Size")
    ax.set_ylabel(metric)
    ax.grid(True)
    if i == 0:
        ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "Aggregated_Plot_All_Metrics_mean_only.png"))
plt.show()


runtime_metrics = ["Prediction Time (s)", "Memory Usage (MB)"]

for metric in runtime_metrics:
    plt.figure(figsize=(10, 6))
    for weight in weighting_styles:
        subset = combined_results[combined_results["Weighting"] == weight]
        grouped_mean = subset.groupby("Normalized Window Size")[metric].mean()
        grouped_std = subset.groupby("Normalized Window Size")[metric].std()

        common_index = grouped_mean.index.intersection(grouped_std.index)
        grouped_mean = grouped_mean.loc[common_index]
        grouped_std = grouped_std.loc[common_index]

        plt.plot(common_index, grouped_mean, marker='o', label=weight)
        plt.fill_between(common_index,
                         grouped_mean - grouped_std,
                         grouped_mean + grouped_std,
                         alpha=0.2)
    plt.title(f"{metric} by Normalized Window Size and Weighting")
    plt.xlabel("Normalized Window Size")
    plt.ylabel(metric)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{metric.replace(' ', '_')}_plot.png"))
    plt.show()


summary_metrics = [
    "Normalized Window Size",
    "Precision", "Recall", "F1 Score",
    "AUC-ROC", "AUC-PR", "Event Precision", "Event Recall", "Event F0.5 Score",
    "Prediction Time (s)", "Memory Usage (MB)"
]

summary_table = combined_results.groupby(["Normalized Window Size"])[
    ["Precision", "Recall", "F1 Score", "AUC-ROC", "AUC-PR", "Prediction Time (s)", "Memory Usage (MB)"]
].agg(['mean', 'std']).reset_index()

summary_table.columns = [' '.join(col).strip() if col[1] else col[0] for col in summary_table.columns.values]

summary_csv_path = os.path.join(output_dir, "summary_table_weighted_experiments.csv")
summary_table.to_csv(summary_csv_path, index=False, float_format="%.4f")
print(f" Summary table saved to: {summary_csv_path}")


grouped_all = combined_results.groupby("Normalized Window Size").agg(['mean','std']).reset_index()
wsizes_all = grouped_all["Normalized Window Size"]

fig, ax = plt.subplots(figsize=(10, 6))
metrics_group1 = ["Precision", "Recall", "F1 Score"]
for metric, color in zip(metrics_group1, ["b", "g", "r"]):
    mean_vals = grouped_all[(metric, 'mean')]
    std_vals  = grouped_all[(metric, 'std')]
    ax.plot(wsizes_all, mean_vals, 'o-', color=color, label=f"{metric} (mean)")
    ax.fill_between(wsizes_all, mean_vals - std_vals, mean_vals + std_vals, alpha=0.2, color=color)
ax.set(
    title="Weighted Experiments: Precision, Recall & F1 vs. Normalized WS",
    xlabel="Normalized WS",
    ylabel="Score"
)
ax.legend(); ax.grid(); fig.tight_layout()
fig.savefig(os.path.join(output_dir, "weighted_precision_recall_f1_vs_ws.png"), dpi=150)
plt.show()


fig, ax = plt.subplots(figsize=(10, 6))
metrics_group2 = ["AUC-ROC", "AUC-PR"]
for metric, color in zip(metrics_group2, ["c", "m"]):
    mean_vals = grouped_all[(metric, 'mean')]
    std_vals  = grouped_all[(metric, 'std')]
    ax.plot(wsizes_all, mean_vals, 'o-', color=color, label=f"{metric} (mean)")
    ax.fill_between(wsizes_all, mean_vals - std_vals, mean_vals + std_vals, alpha=0.2, color=color)
ax.set(
    title="Weighted Experiments: AUC-ROC & AUC-PR vs. Normalized WS",
    xlabel="Normalized WS",
    ylabel="AUC Score"
)
ax.legend(); ax.grid(); fig.tight_layout()
fig.savefig(os.path.join(output_dir, "weighted_auc_roc_pr_vs_ws.png"), dpi=150)
plt.show()


fig, ax = plt.subplots(figsize=(10, 6))
metrics_group3 = ["Event Precision", "Event Recall", "Event F0.5 Score"]
for metric, color in zip(metrics_group3, ["orange", "limegreen", "darkred"]):
    mean_vals = grouped_all[(metric, 'mean')]
    std_vals  = grouped_all[(metric, 'std')]
    ax.plot(wsizes_all, mean_vals, 'o-', color=color, label=f"{metric} (mean)")
    ax.fill_between(wsizes_all, mean_vals - std_vals, mean_vals + std_vals, alpha=0.2, color=color)
ax.set(
    title="Weighted Experiments: Event-wise Precision, Recall & F0.5 vs. Normalized WS",
    xlabel="Normalized WS",
    ylabel="Event-wise Score"
)
ax.legend(); ax.grid(); fig.tight_layout()
fig.savefig(os.path.join(output_dir, "weighted_eventwise_precision_recall_f0.5_vs_ws.png"), dpi=150)
plt.show()