import sys
import os
import time
import tracemalloc

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import LinearSegmentedColormap
from statsmodels.tsa.stattools import acf
import stumpy

from dtaianomaly.preprocessing import MovingAverage
from dtaianomaly.anomaly_detection import MatrixProfileDetector
from dtaianomaly.anomaly_detection import LocalOutlierFactor
from dtaianomaly.anomaly_detection import IsolationForest
from dtaianomaly.anomaly_detection import KShapeAnomalyDetector
from dtaianomaly.anomaly_detection import MedianMethod
from dtaianomaly.anomaly_detection import KNearestNeighbors
from dtaianomaly.thresholding import FixedCutoff
from dtaianomaly.evaluation import (
    Precision, Recall, FBeta, AreaUnderROC, AreaUnderPR, ThresholdMetric
)
from dtaianomaly.data import UCRLoader
from eventwise_metrics import EventWisePrecision, EventWiseRecall, EventWiseFBeta

DATASET_BASE = r"C:\Users\34622\Desktop\1MAI\thesiscode\datasets\UCR_Anomaly_FullData"
OUTPUT_DIR   = r"C:\Users\34622\Desktop\1MAI\thesiscode\results\Size\matrix_profile\definite\ECG"
os.makedirs(OUTPUT_DIR, exist_ok=True)

dataset_info = {
    "119_UCR_Anomaly_ECG1_10000_11800_12100.txt": (11800,12100),
    "120_UCR_Anomaly_ECG2_15000_16000_16100.txt": (16000,16100),
    "122_UCR_Anomaly_ECG3_8000_17000_17100.txt":(17000,17100),
    "123_UCR_Anomaly_ECG4_5000_16800_17100.txt": (16800,17100)
}


DATASET_PATHS = {
     os.path.join(DATASET_BASE, fname): rng
     for fname, rng in dataset_info.items()
}

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
"139_UCR_Anomaly_InternalBleeding20_2700_5759_5919.txt": (5759, 5919),}
"""
NORMALIZED_WS = np.logspace(np.log10(0.25), np.log10(4), num=15)

def estimate_window_acf(ts, max_lag=300):
    ac = acf(ts, nlags=max_lag)
    return np.argmax(ac[5:]) + 5

def estimate_window_fft(ts, min_w=50, max_w=200):
    ts0 = ts - ts.mean()
    freqs = np.fft.fftfreq(len(ts0))
    mags  = np.abs(np.fft.fft(ts0))
    pos   = freqs > 0
    if not np.any(pos): return None
    dom_f = freqs[pos][np.argmax(mags[pos])]
    return int(round(1/dom_f)) if dom_f != 0 else None

def estimate_window_mp(ts, min_w=50, max_w=200, step=10):
    best_w, best_s = min_w, np.inf
    for w in range(min_w, max_w+1, step):
        prof = stumpy.stump(ts, m=w)[:,0]
        score = prof.mean()/np.sqrt(w)
        if score < best_s:
            best_s, best_w = score, w
    return best_w

def estimate_window_suss(ts, min_w=20, max_w=200, step=10):
    μ, σ = ts.mean(), ts.std()
    best_w, best_s = min_w, np.inf
    for w in range(min_w, max_w+1, step):
        segs = np.lib.stride_tricks.sliding_window_view(ts, w)[::w]
        stats = np.array([[s.mean(), s.std()] for s in segs])
        score = ((stats[:,0]-μ)**2 + (stats[:,1]-σ)**2).mean()
        if score < best_s:
            best_s, best_w = score, w
    return best_w

def run_experiments(path, anomaly_range, norm_ws_set, label):
    print(f"\n→ Running {label} on {os.path.basename(path)}")
    data = UCRLoader(path).load()
    Xtr, ytr = data.X_train, data.y_train
    Xte, yte = data.X_test,  data.y_test

    ma = MovingAverage(window_size=10)
    Xtr_p, ytr_p = ma.fit_transform(Xtr, ytr)
    Xte_p, yte_p = ma.transform(Xte, yte)

    a_len = anomaly_range[1] - anomaly_range[0]
    rows = []
    for norm_ws in norm_ws_set:
        ws = int(round(norm_ws * a_len))
        if ws < 3: continue

        print(f"  • ws={ws}")
        tracemalloc.start()
        t0 = time.time()

        n_windows = len(Xtr_p) - ws + 1
        n_neighbors = int(np.clip(0.05 * n_windows, 5, 50))
        det = MatrixProfileDetector(
                   window_size = ws)
        prob = det.fit(Xtr_p).predict_proba(Xte_p)
        dt  = time.time() - t0
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        mem = peak/1024**2
        bin_pred = FixedCutoff(0.85).threshold(prob)

        p   = Precision().compute(yte_p, bin_pred)
        r   = Recall().compute(yte_p, bin_pred)
        f1  = ThresholdMetric(FixedCutoff(0.85), FBeta(1.0)).compute(yte_p, prob)
        roc = AreaUnderROC().compute(yte_p, prob)
        pr  = AreaUnderPR().compute(yte_p, prob)
        ewp = EventWisePrecision().compute(yte_p, bin_pred)
        ewr = EventWiseRecall().compute(yte_p, bin_pred)
        ewf = EventWiseFBeta(1.0).compute(yte_p, bin_pred)

        rows.append({
            "Window Size": ws,
            "Normalized WS": norm_ws,
            "Precision": p, "Recall": r, "F1 Score": f1,
            "AUC-ROC": roc, "AUC-PR": pr,
            "Event Precision": ewp, "Event Recall": ewr, "Event F1 Score": ewf,
            "Time(s)": round(dt,4), "Memory(MB)": round(mem,2)
        })
    return pd.DataFrame(rows)

def plot_ts_with_windows(dataset_paths, normalized_ws_set, output_dir=None):
    n = len(normalized_ws_set)
    cmap = LinearSegmentedColormap.from_list(
        "gtl_green", ["#004d00", "#ccffcc"], N=n)

    for path, (a0,a1) in dataset_paths.items():
        name = os.path.basename(path)
        data = UCRLoader(path).load()
        Xtr, Xte = data.X_train.ravel(), data.X_test.ravel()
        ts = np.concatenate([Xtr, Xte])
        Ltr = len(Xtr)
        al  = a1 - a0

        fig, ax = plt.subplots(figsize=(14,4))
        ax.plot(ts, color="black", label="Series")
        for i,nw in enumerate(normalized_ws_set):
            w = int(round(nw*al))
            if w<3: continue
            c = cmap(i/(n-1))
            α = 0.9 - 0.7*(i/(n-1))
            ax.axvspan(Ltr-w, Ltr, facecolor=c, edgecolor=c, alpha=α, label=f"ws={w}")
        ax.set(title=f"{name} — Context Windows Only", xlabel="Time", ylabel="Value")
        ax.legend(fontsize=6, ncol=4, loc="upper right")
        fig.tight_layout()
        if output_dir:
            fig.savefig(os.path.join(output_dir, f"{name}_ws_only.png"), dpi=200)
        plt.show()

        fig, ax = plt.subplots(figsize=(14,4))
        ax.plot(ts, color="black", label="Series")
        for i,nw in enumerate(normalized_ws_set):
            w = int(round(nw*al))
            if w<3: continue
            c = cmap(i/(n-1))
            α = 0.9 - 0.7*(i/(n-1))
            ax.axvspan(Ltr-w, Ltr, facecolor=c, edgecolor=c, alpha=α, label=f"ws={w}")

        ax.axvspan(Ltr+(a0-Ltr), Ltr+(a1-Ltr),
                   facecolor="#D62728", edgecolor="#D62728", alpha=0.4, label="Anomaly")
        ax.set(title=f"{name} — Windows + Anomaly", xlabel="Time", ylabel="Value")
        ax.legend(fontsize=6, ncol=4, loc="upper right")
        fig.tight_layout()
        if output_dir:
            fig.savefig(os.path.join(output_dir, f"{name}_ws_and_anomaly.png"), dpi=200)
        plt.show()

def main():
    pre = {"fft":[], "acf":[], "mp":[], "suss":[]}
    for p in DATASET_PATHS:
        ts = UCRLoader(p).load().X_train.ravel()
        pre["fft"].append(estimate_window_fft(ts))
        pre["acf"].append(estimate_window_acf(ts))
        pre["mp"].append(estimate_window_mp(ts))
        pre["suss"].append(estimate_window_suss(ts))
    stats = {m:(np.mean(v),np.std(v)) for m,v in pre.items()}

    dfs = []
    for p,rng in DATASET_PATHS.items():
        df = run_experiments(p, rng, NORMALIZED_WS, "Normalized WS")
        dfs.append(df)
    combined = pd.concat(dfs, ignore_index=True)

    grouped = combined.groupby("Normalized WS").agg(['mean','std']).reset_index()
    csv_path = os.path.join(OUTPUT_DIR, "aggregated_results.csv")
    grouped.to_csv(csv_path, index=False)
    print(f" Saved {csv_path}")

    plot_ts_with_windows(DATASET_PATHS, NORMALIZED_WS, OUTPUT_DIR)

    metrics = [
        "Precision", "Recall", "F1 Score", "AUC-ROC", "AUC-PR",
        "Event Precision", "Event Recall", "Event F1 Score"
    ]
    colors = ["b", "g", "r", "c", "m", "orange", "limegreen", "darkred"]
    colors_by_method = {"fft":"blue","acf":"red","mp":"green","suss":"yellow"}
    avg_an = np.mean([e0 - s0 for s0,e0 in DATASET_PATHS.values()])

    for metric,color in zip(metrics,colors):
        fig, ax = plt.subplots(figsize=(10,6))
        wsizes = grouped["Normalized WS"]
        means  = grouped[(metric,'mean')]
        stds   = grouped[(metric,'std')]
        ax.plot(wsizes, means, 'o-', color=color, label=f"{metric} (mean)")
        ax.fill_between(wsizes, means-stds, means+stds, alpha=0.2, color=color)
        for method,(m,s) in stats.items():
            nm, ns = m/avg_an, s/avg_an
            mc = colors_by_method[method]
            ax.axvline(nm, color=mc, linestyle='--', label=f"{method.upper()} mean")
            ax.axvspan(nm-ns, nm+ns, alpha=0.2, color=mc)
        ax.set(title=f"{metric} vs. Normalized WS", xlabel="Normalized WS", ylabel=metric)
        ax.legend(); ax.grid(); fig.tight_layout()

        fig.savefig(os.path.join(OUTPUT_DIR, f"{metric.replace('/', '-')}_vs_ws.png"), dpi=150)
        plt.show()

    fig, ax = plt.subplots(figsize=(12,7))
    for metric,color in zip(metrics,colors):
        means  = grouped[(metric,'mean')]
        stds   = grouped[(metric,'std')]
        ax.plot(wsizes, means, 'o-', label=metric, color=color)
        ax.fill_between(wsizes, means-stds, means+stds, alpha=0.2, color=color)
    for method,(m,s) in stats.items():
        nm, ns = m/avg_an, s/avg_an; mc = colors_by_method[method]
        ax.axvline(nm, color=mc, linestyle='--', label=f"{method.upper()} mean")
        ax.axvspan(nm-ns, nm+ns, alpha=0.2, color=mc)
    ax.set(title="All Metrics vs. Normalized WS", xlabel="Normalized WS", ylabel="Score")
    ax.legend(); ax.grid(); fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "all_metrics_with_std.png"), dpi=150)
    plt.show()

    fig, ax = plt.subplots(figsize=(12,7))
    for metric,color in zip(metrics,colors):
        ax.plot(wsizes, grouped[(metric,'mean')], 'o-', label=metric, color=color)
    for method,(m,_) in stats.items():
        nm = m/avg_an; mc = colors_by_method[method]
        ax.axvline(nm, color=mc, linestyle='--', label=method.upper())
    ax.set(title="Metrics (Mean Only) vs. Normalized WS", xlabel="Normalized WS", ylabel="Score")
    ax.legend(); ax.grid(); fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "all_metrics_mean_only.png"), dpi=150)
    plt.show()

    runtime = ["Time(s)","Memory(MB)"]
    rc = ["purple","orange"]
    for metric,color in zip(runtime,rc):
        fig, ax = plt.subplots(figsize=(10,6))
        means = grouped[(metric,'mean')]
        stds  = grouped[(metric,'std')]
        ax.plot(wsizes, means, 'o-', color=color, label=f"{metric} mean")
        ax.fill_between(wsizes, means-stds, means+stds, alpha=0.2, color=color)
        ax.set(title=f"{metric} vs. Normalized WS", xlabel="Normalized WS", ylabel=metric)
        ax.legend(); ax.grid(); fig.tight_layout()
        fig.savefig(os.path.join(OUTPUT_DIR, f"{metric.replace(' ','_')}_vs_ws.png"), dpi=150)
        plt.show()


    fig, ax = plt.subplots(figsize=(10, 4))
    metrics_group1 = ["Precision", "Recall", "F1 Score"]
    for metric, color in zip(metrics_group1, ["b", "g", "r"]):
        mean_vals = grouped[(metric, 'mean')]
        std_vals = grouped[(metric, 'std')]
        ax.plot(wsizes, mean_vals, 'o-', color=color, label=f"{metric} (mean)")
        ax.fill_between(wsizes, mean_vals - std_vals, mean_vals + std_vals, alpha=0.2, color=color)
    for method, (m, s) in stats.items():
        nm = m / avg_an
        mc = colors_by_method[method]
        ax.axvline(nm, color=mc, linestyle='--', label=f"{method.upper()} mean")
    ax.set(
        title="Precision, Recall, and F1 Score vs. Normalized WS",
        xlabel="Normalized WS",
        ylabel="Score"
    )
    ax.legend();
    ax.grid();
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "precision_recall_f1_vs_ws.png"), dpi=150)
    plt.show()

    fig, ax = plt.subplots(figsize=(10, 4))
    metrics_group2 = ["AUC-ROC", "AUC-PR"]
    for metric, color in zip(metrics_group2, ["c", "m"]):
        mean_vals = grouped[(metric, 'mean')]
        std_vals = grouped[(metric, 'std')]
        ax.plot(wsizes, mean_vals, 'o-', color=color, label=f"{metric} (mean)")
        ax.fill_between(wsizes, mean_vals - std_vals, mean_vals + std_vals, alpha=0.2, color=color)

    for method, (m, s) in stats.items():
        nm = m / avg_an
        mc = colors_by_method[method]
        ax.axvline(nm, color=mc, linestyle='--', label=f"{method.upper()} mean")
    ax.set(
        title="AUC-ROC and AUC-PR vs. Normalized WS",
        xlabel="Normalized WS",
        ylabel="AUC Score"
    )
    ax.legend();
    ax.grid();
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "auc_roc_pr_vs_ws.png"), dpi=150)
    plt.show()

    fig, ax = plt.subplots(figsize=(10, 4))
    metrics_group3 = ["Event Precision", "Event Recall", "Event F1 Score"]
    for metric, color in zip(metrics_group3, ["orange", "limegreen", "darkred"]):
        mean_vals = grouped[(metric, 'mean')]
        std_vals = grouped[(metric, 'std')]
        ax.plot(wsizes, mean_vals, 'o-', color=color, label=f"{metric} (mean)")
        ax.fill_between(wsizes, mean_vals - std_vals, mean_vals + std_vals, alpha=0.2, color=color)

    for method, (m, s) in stats.items():
        nm = m / avg_an
        mc = colors_by_method[method]
        ax.axvline(nm, color=mc, linestyle='--', label=f"{method.upper()} mean")
    ax.set(
        title="Event-wise Precision, Recall, and F1 Score vs. Normalized WS",
        xlabel="Normalized WS",
        ylabel="Event-wise Score"
    )
    ax.legend();
    ax.grid();
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "eventwise_precision_recall_f1_vs_ws.png"), dpi=150)
    plt.show()


if __name__ == "__main__":
    main()
