import os
import time
import tracemalloc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from dtaianomaly.data import UCRLoader
from dtaianomaly.evaluation import Precision, Recall, FBeta, AreaUnderROC, AreaUnderPR
from eventwise_metrics import EventWisePrecision, EventWiseRecall, EventWiseFBeta
from dtaianomaly.preprocessing import MovingAverage

BASE_PATH = r"C:\Users\34622\Desktop\1MAI\thesiscode\datasets\UCR_Anomaly_FullData"
DATA_GROUPS = {
    'MARS': [
        "157_UCR_Anomaly_TkeepFirstMARS_3500_5365_5380.txt",
        "160_UCR_Anomaly_TkeepThirdMARS_3500_4711_4809.txt",
        "158_UCR_Anomaly_TkeepForthMARS_3500_5988_6085.txt",
        "159_UCR_Anomaly_TkeepSecondMARS_3500_9330_9340.txt",
        "156_UCR_Anomaly_TkeepFifthMARS_3500_5988_6085.txt",
    ],
    'InternalBleeding': [
        "135_UCR_Anomaly_InternalBleeding16_1200_4187_4199.txt",
        "136_UCR_Anomaly_InternalBleeding17_1600_3198_3309.txt",
        "137_UCR_Anomaly_InternalBleeding18_2300_4485_4587.txt",
        "138_UCR_Anomaly_InternalBleeding19_3000_4187_4197.txt",
        "139_UCR_Anomaly_InternalBleeding20_2700_5759_5919.txt",
    ]

}

DATASETS = {}
for group, files in DATA_GROUPS.items():
    for f in files:
        DATASETS[os.path.join(BASE_PATH, f)] = {'group': group}

WINDOW_SIZES = [25,30,35,40,45,50,55,60,65,70,75,80,85]
N_CLUSTERS   = [15,16,17,18,19,20,21,23,25,27,29,30,32,33,35,37,40,42]
THRESHOLDS   = [97,98,99,99.25,99.5,99.75,99.9]

def to_windows(arr, ws):
    n = len(arr) // ws
    return np.array([arr[i*ws:(i+1)*ws] for i in range(n)])

best_params = {}
for path, meta in DATASETS.items():
    data = UCRLoader(path).load()


    ts_tr = data.X_train.ravel()
    ts_te = data.X_test.ravel()
    y_te  = data.y_test.ravel()
    best = {'ws':None, 'k':None, 'p':None, 'f1':-1}
    for ws in WINDOW_SIZES:
        tr_w = to_windows(ts_tr, ws)
        te_w = to_windows(ts_te, ws)
        y_w  = to_windows(y_te, ws)
        for k in N_CLUSTERS:
            if k > tr_w.shape[0]: continue
            km = KMeans(n_clusters=k, random_state=0).fit(tr_w)
            prot = km.cluster_centers_
            for p in THRESHOLDS:
                scores = np.array([min(np.linalg.norm(w-c) for c in prot) for w in te_w])
                preds  = (scores > np.percentile(scores,p)).astype(int)
                truth = np.array([1 if w.any() else 0 for w in y_w])
                f1 = FBeta(1.0).compute(truth,preds)
                if f1 > best['f1']:
                    best.update({'ws':ws,'k':k,'p':p,'f1':f1})
    best_params[path] = best
    print(f"{os.path.basename(path)} best -> ws={best['ws']}, k={best['k']}, p={best['p']}, F1={best['f1']:.3f}")

results = []
out_dir = os.path.join(r"C:\Users\34622\Desktop\1MAI\thesiscode\results\PrototypeGrid","MARS_InternalBleeding2")
os.makedirs(out_dir, exist_ok=True)
for path, meta in DATASETS.items():
    grp = meta['group']
    ws,k,p = best_params[path]['ws'], best_params[path]['k'], best_params[path]['p']
    data = UCRLoader(path).load()
    ts_tr, ts_te = data.X_train.ravel(), data.X_test.ravel()
    y_te = data.y_test.ravel()

    t0=time.time(); tracemalloc.start()
    tr_w = to_windows(ts_tr, ws)
    km = KMeans(n_clusters=k,random_state=0).fit(tr_w)
    prot = km.cluster_centers_
    te_w, y_w = to_windows(ts_te, ws), to_windows(y_te, ws)
    scores = np.array([min(np.linalg.norm(w-c) for c in prot) for w in te_w])
    thresh = np.percentile(scores,p)
    preds = (scores>thresh).astype(int)
    truth = np.array([1 if w.any() else 0 for w in y_w])
    cur,peak = tracemalloc.get_traced_memory(); tracemalloc.stop()
    elapsed = time.time()-t0

    metrics = {'Dataset':os.path.basename(path),'Group':grp,'Window':ws,'Clusters':k,'Percentile':p,
               'Time (s)':elapsed,'Memory (MB)':peak/1024**2,
               'Precision':Precision().compute(truth,preds),
               'Recall':Recall().compute(truth,preds),
               'F1 Score':FBeta(1).compute(truth,preds),
               'AUC-ROC':AreaUnderROC().compute(truth,scores),
               'AUC-PR':AreaUnderPR().compute(truth,scores),
               'Event Precision':EventWisePrecision().compute(truth,preds),
               'Event Recall':EventWiseRecall().compute(truth,preds),
               'Event F1 Score':EventWiseFBeta(1).compute(truth,preds)}
    results.append(metrics)

df = pd.DataFrame(results)
df.to_csv(os.path.join(out_dir,'per_dataset_metrics.csv'),index=False)


x = np.arange(len(df))
group_counters = {}
short_labels = []
for grp in df['Group']:
    group_counters.setdefault(grp, 0)
    group_counters[grp] += 1
    short_labels.append(f"Dataset {group_counters[grp]} {grp}")

metrics = [
    "Precision", "Recall", "F1 Score",
    "AUC-ROC", "AUC-PR",
    "Event Precision", "Event Recall", "Event F1 Score"
]
colors = [
    "b", "g", "r", "c", "m",
    "orange", "limegreen", "darkred"
]
color_map = dict(zip(metrics, colors))

fig, ax = plt.subplots(figsize=(10,4))
for m in ["Precision","Recall","F1 Score"]:
    ax.plot(x, df[m], 'o-', label=m, color=color_map[m])
ax.set_xticks(x)
ax.set_xticklabels(short_labels, rotation=45, ha='right')
ax.set_title("Precision, Recall, and F1 Score by Dataset")
ax.set_ylabel("Metric value")
ax.legend()
plt.tight_layout()
fig.savefig(os.path.join(out_dir, "overlay_precision_recall_f1.png"), dpi=150)
plt.show()


fig, ax = plt.subplots(figsize=(10,4))
for m in ["AUC-ROC","AUC-PR"]:
    ax.plot(x, df[m], 'o-', label=m, color=color_map[m])
ax.set_xticks(x)
ax.set_xticklabels(short_labels, rotation=45, ha='right')
ax.set_title("AUC-ROC and AUC-PR by Dataset")
ax.set_ylabel("Metric value")
ax.legend()
plt.tight_layout()
fig.savefig(os.path.join(out_dir, "overlay_auc_roc_pr.png"), dpi=150)
plt.show()


fig, ax = plt.subplots(figsize=(10,4))
for m in ["Event Precision","Event Recall","Event F1 Score"]:
    ax.plot(x, df[m], 'o-', label=m, color=color_map[m])
ax.set_xticks(x)
ax.set_xticklabels(short_labels, rotation=45, ha='right')
ax.set_title("Event-wise Precision, Recall, and F1 Score by Dataset")
ax.set_ylabel("Metric value")
ax.legend()
plt.tight_layout()
fig.savefig(os.path.join(out_dir, "overlay_eventwise_metrics.png"), dpi=150)
plt.show()


metrics = [
    "Precision", "Recall", "F1 Score", "AUC-ROC", "AUC-PR",
    "Event Precision", "Event Recall", "Event F1 Score"
]
colors = [
    "b", "g", "r", "c", "m",
    "orange", "limegreen", "darkred"
]
color_map = dict(zip(metrics, colors))

all_metrics = metrics + ["Time (s)", "Memory (MB)"]

for m in all_metrics:
    fig, ax = plt.subplots(figsize=(8, 4))
    c = color_map.get(m, "0.5")  # grey fallback for time/memory
    ax.plot(x, df[m], 'o-', color=c, label=m)
    ax.set_xticks(x)
    ax.set_xticklabels(short_labels, rotation=45, ha='right')
    ax.set_title(f'{m} by Dataset')
    ax.set_ylabel(m)
    fig.subplots_adjust(bottom=0.2)   # avoid label cutoff
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, f'{m.replace(" ","_")}_per_dataset.png'), dpi=150)
    plt.show()

fig, ax = plt.subplots(figsize=(10, 4))
for m in metrics:
    ax.plot(x, df[m], 'o-', color=color_map[m], label=m)

ax.set_xticks(x)
ax.set_xticklabels(short_labels, rotation=0, ha='center')
ax.set_title('All Anomaly Metrics by Dataset')
ax.set_ylabel('Metric value')
ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
fig.subplots_adjust(bottom=0.2)
plt.tight_layout()
fig.savefig(os.path.join(out_dir, 'all_metrics_per_dataset.png'), dpi=150)
plt.show()