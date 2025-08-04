import os
import numpy as np
import pandas as pd
import tracemalloc
import time
import matplotlib.pyplot as plt
from dtaianomaly.anomaly_detection import LocalOutlierFactor
from dtaianomaly.anomaly_detection import KNearestNeighbors
from dtaianomaly.preprocessing import MovingAverage
from dtaianomaly.data import UCRLoader
from dtaianomaly.evaluation import Precision, Recall, FBeta, AreaUnderROC, AreaUnderPR
from eventwise_metrics import EventWisePrecision, EventWiseRecall, EventWiseFBeta

data_dir = os.path.join("datasets", "UCR_Anomaly_FullData")
out_dir  = os.path.join("results",  "position","LocalOutlierFactor",  "PowerDemand")
os.makedirs(out_dir, exist_ok=True)

datasets = {
    "044_UCR_Anomaly_DISTORTEDPowerDemand1_9000_18485_18821.txt": (18485, 18821),
    "045_UCR_Anomaly_DISTORTEDPowerDemand2_14000_23357_23717.txt": (23357, 23717),
    "046_UCR_Anomaly_DISTORTEDPowerDemand3_16000_23405_23477.txt": (23405, 23477),
    "047_UCR_Anomaly_DISTORTEDPowerDemand4_18000_24005_24077.txt": (24005, 24077)
}
data_paths = {os.path.join(data_dir, f): r for f, r in datasets.items()}

N_PARTS        = 15
CUTOFF_PERCENT = 85
START_DATE     = pd.Timestamp("1995-01-01 00:00")

anomaly_ts = {
    f: pd.Timestamp(ts) for f, ts in {
        "044_UCR_Anomaly_DISTORTEDPowerDemand1_9000_18485_18821.txt": "1997-02-09 03:00",
        "045_UCR_Anomaly_DISTORTEDPowerDemand2_14000_23357_23717.txt": "1997-08-31 00:00",
        "046_UCR_Anomaly_DISTORTEDPowerDemand3_16000_23405_23477.txt": "1997-09-03 05:00",
        "047_UCR_Anomaly_DISTORTEDPowerDemand4_18000_24005_24077.txt": "1997-09-27 05:00"
    }.items()
}

metrics_defs = [
    ("Precision", lambda y, yp, sc: Precision().compute(y, yp)),
    ("Recall",    lambda y, yp, sc: Recall().compute(y, yp)),
    ("F1 Score",        lambda y, yp, sc: FBeta(1).compute(y, yp)),
    ("AUC-ROC",   lambda y, yp, sc: AreaUnderROC().compute(y, sc)),
    ("AUC-PR",    lambda y, yp, sc: AreaUnderPR().compute(y, sc)),
    ("Event Precision",   lambda y, yp, sc: EventWisePrecision().compute(y, yp)),
    ("Event Recall",    lambda y, yp, sc: EventWiseRecall().compute(y, yp)),
    ("Event F1 Score",    lambda y, yp, sc: EventWiseFBeta(1).compute(y, yp))
]
colors = ["b","g","r","c","m","orange","limegreen","darkred"]

metrics_defs = [
    ("Precision", lambda y, yp, sc: Precision().compute(y, yp)),
    ("Recall",    lambda y, yp, sc: Recall().compute(y, yp)),
    ("F1 Score",        lambda y, yp, sc: FBeta(1).compute(y, yp)),
    ("AUC-ROC",   lambda y, yp, sc: AreaUnderROC().compute(y, sc)),
    ("AUC-PR",    lambda y, yp, sc: AreaUnderPR().compute(y, sc)),
    ("Event Precision",   lambda y, yp, sc: EventWisePrecision().compute(y, yp)),
    ("Event Recall",    lambda y, yp, sc: EventWiseRecall().compute(y, yp)),
    ("Event F1 Score",    lambda y, yp, sc: EventWiseFBeta(1).compute(y, yp))
]
colors = ["b","g","r","c","m","orange","limegreen","darkred"]

def estimate_window_suss(ts, min_w=20, max_w=200):
    m, s = ts.mean(), ts.std()
    best_w, best_sc = min_w, np.inf
    for w in range(min_w, max_w, 10):
        segs = np.array([[ts[i:i+w].mean(), ts[i:i+w].std()] for i in range(0, len(ts)-w+1, w)])
        sc = np.mean((segs[:,0]-m)**2 + (segs[:,1]-s)**2)
        if sc < best_sc:
            best_sc, best_w = sc, w
    return best_w

year_part_positions = {}  # {k: [part_indices]}
all_records = []

for path, (a0, a1) in data_paths.items():
    fname = os.path.basename(path)
    print("Processing", fname)

    dl = UCRLoader(path).load()
    Xtr_raw, Xte_raw = dl.X_train, dl.X_test
    ytr_raw = dl.y_train if dl.y_train is not None else np.zeros_like(Xtr_raw)
    yte_raw = dl.y_test

    ma = MovingAverage(window_size=10)
    Xtr_ma, _ = ma.fit_transform(Xtr_raw)
    Xte_ma, _ = ma.transform(Xte_raw)
    Xtr = Xtr_ma.ravel(); Xte = Xte_ma.ravel()
    ytr = ytr_raw.ravel(); yte = yte_raw.ravel()

    full_x = np.concatenate([Xtr, Xte])
    full_y = np.concatenate([ytr, yte])
    full_t = START_DATE + pd.to_timedelta(np.arange(full_x.size), unit='h')

    ws = estimate_window_suss(Xtr)

    anomaly_dt = anomaly_ts[fname]
    yrs_back = int((anomaly_dt - START_DATE).days // 365)
    marks = [anomaly_dt - pd.DateOffset(years=k) for k in range(1, yrs_back+1) if anomaly_dt - pd.DateOffset(years=k) >= START_DATE]

    year_idx = np.searchsorted(full_t, marks[0]) if marks else len(Xtr)
    base_len = max(1, len(Xtr)//N_PARTS)
    new_Ltr = min(max(len(Xtr), year_idx + 3*base_len), full_x.size)

    Xtr_ext, ytr_ext = full_x[:new_Ltr], full_y[:new_Ltr]
    Xte_ext, yte_ext = full_x[new_Ltr:], full_y[new_Ltr:]
    tr_t = full_t[:new_Ltr]

    part_len = new_Ltr // N_PARTS
    fig, ax = plt.subplots(figsize=(14,4))
    ax.plot(full_t, full_x, color='black', lw=0.6)
    for p in range(N_PARTS):
        t0, t1 = tr_t[p*part_len], tr_t[(p+1)*part_len] if p<N_PARTS-1 else tr_t[-1]
        ax.axvspan(t0, t1, color='#add8e6', alpha=0.3)
    for k, dt in enumerate(marks,1): ax.axvline(dt, color='orange', ls='--', label=f'-{k}yr' if k==1 else None)
    ax.axvspan(full_t[a0], full_t[a1], color='red', alpha=0.3, label='Anomaly')
    ax.set(title=f"{fname} (ws={ws}, train_end={new_Ltr})", xlabel='Date', ylabel='Value')
    ax.legend(fontsize=6); plt.tight_layout()
    plt.show()
    fig.savefig(os.path.join(out_dir, f"{fname}_series.png"), dpi=150)

    cont = max(1e-3, (a1 - a0) / max(1, len(Xte_ext)))
    part_vals = {name: [] for name, _ in metrics_defs}
    part_times = []
    part_mems = []

    for p in range(N_PARTS):
        s, e = p * part_len, (p + 1) * part_len if p < N_PARTS - 1 else new_Ltr
        ctx = Xtr_ext[s:e]

        if ctx.size < ws:
            for n in part_vals: part_vals[n].append(np.nan)
            part_times.append(np.nan)
            part_mems.append(np.nan)
        else:
            tracemalloc.clear_traces()
            tracemalloc.start()
            t0 = time.time()

            lof = LocalOutlierFactor(window_size=ws, contamination=cont)
            lof.fit(ctx.reshape(-1, 1))
            scores = lof.decision_function(Xte_ext.reshape(-1, 1))

            elapsed = time.time() - t0
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            part_times.append(elapsed)
            part_mems.append(peak / 1024 ** 2)

            thr = np.percentile(scores, CUTOFF_PERCENT)
            yp = (scores > thr).astype(int)
            for n, f in metrics_defs:
                part_vals[n].append(f(yte_ext, yp, scores))

    for idx,(n,_) in enumerate(metrics_defs):
        parts_idx = np.arange(1,len(part_vals[n])+1)
        fig,ax=plt.subplots(figsize=(9,5))
        ax.plot(parts_idx, part_vals[n], 'o-', color=colors[idx], label=n)
        for k,dt in enumerate(marks,1):
            pm = int(np.searchsorted(tr_t,dt)//part_len)+1
            ax.axvline(pm, color='orange', ls='--', label=f'-{k}yr' if k==1 else None)
        ax.set(title=f"{fname}: {n} per Part", xlabel='Part',ylabel=n)
        ax.grid();ax.legend(fontsize=6);plt.tight_layout()
        plt.show(); fig.savefig(os.path.join(out_dir, f"{fname}_{n}.png"), dpi=150)

    for p in range(1, len(part_vals[n]) + 1):
        rec = {
            'Dataset': fname,
            'Part': p,
            'Time (s)': round(part_times[p - 1], 4),
            'Memory (MB)': round(part_mems[p - 1], 2)
        }
        for n in part_vals:
            rec[n] = part_vals[n][p - 1]
        all_records.append(rec)
    for k,dt in enumerate(marks,1):
        part_k = int(np.searchsorted(tr_t,dt)//part_len)+1
        year_part_positions.setdefault(k,[]).append(part_k)

agg_df = pd.DataFrame(all_records)
agg_df.to_csv(os.path.join(out_dir,'aggregated_parts_metrics.csv'),index=False)

agg_dict = {n: ['mean','std'] for n,_ in metrics_defs}
agg_dict.update({'Time (s)': ['mean'], 'Memory (MB)': ['mean']})

summary = agg_df.groupby('Part').agg(agg_dict).reset_index()
parts=summary['Part']
avg_marks={k:int(round(np.mean(v))) for k,v in year_part_positions.items()}
for idx,(n,_) in enumerate(metrics_defs):
    mean=summary[(n,'mean')]; std=summary[(n,'std')]
    fig,ax=plt.subplots(figsize=(10,6))
    ax.plot(parts,mean,'o-',color=colors[idx],label=f'{n} mean')
    ax.fill_between(parts,mean-std,mean+std,alpha=0.2,color=colors[idx])
    for k,pm in avg_marks.items(): ax.axvline(pm,color='orange',ls='--',label=f'-{k}yr avg' if k==1 else None)
    ax.set(title=f"Aggregated {n}",xlabel='Part',ylabel=n);ax.grid();ax.legend(fontsize=6);plt.tight_layout();plt.show()
    fig.savefig(os.path.join(out_dir,f'agg_{n}.png'),dpi=150)

fig,ax=plt.subplots(figsize=(12,8))
for idx,(n,_) in enumerate(metrics_defs): ax.plot(parts,summary[(n,'mean')],'o-',color=colors[idx],label=n)
for k,pm in avg_marks.items(): ax.axvline(pm,color='orange',ls='--',label=f'-{k}yr avg' if k==1 else None)
ax.set(title='Aggregated all metrics',xlabel='Part',ylabel='Value');ax.grid();ax.legend(fontsize=6);plt.tight_layout();plt.show();fig.savefig(os.path.join(out_dir,'agg_all.png'),dpi=150)

fig, ax = plt.subplots(figsize=(10, 4))
for metric, color in zip(["Precision","Recall","F1 Score"], ["b","g","r"]):
    m = summary[(metric,"mean")]
    s = summary[(metric,"std")]
    ax.plot(parts, m,    "o-", color=color, label=f"{metric}")
    ax.fill_between(parts, m-s, m+s, alpha=0.2, color=color)

for k, style in zip((1,2), ("--", ":")):
    pm = avg_marks.get(k)
    if pm is not None:
        ax.axvline(pm, color="orange", linestyle=style, label=f"-{k}yr avg")

ax.set(title="Precision, Recall & F1 Score vs Part",
       xlabel="Part", ylabel="Score")
ax.legend(fontsize=6); ax.grid(); plt.tight_layout()
fig.savefig(os.path.join(out_dir, "agg_precision_recall_f1_with_years.png"), dpi=150)
plt.show()

fig, ax = plt.subplots(figsize=(10, 4))
for metric, color in zip(["AUC-ROC","AUC-PR"], ["c","m"]):
    m = summary[(metric,"mean")]
    s = summary[(metric,"std")]
    ax.plot(parts, m,    "o-", color=color, label=f"{metric}")
    ax.fill_between(parts, m-s, m+s, alpha=0.2, color=color)

for k, style in zip((1,2), ("--", ":")):
    pm = avg_marks.get(k)
    if pm is not None:
        ax.axvline(pm, color="orange", linestyle=style, label=f"-{k}yr avg")

ax.set(title="AUC-ROC & AUC-PR vs Part",
       xlabel="Part", ylabel="AUC Score")
ax.legend(fontsize=6); ax.grid(); plt.tight_layout()
fig.savefig(os.path.join(out_dir, "agg_auc_roc_pr_with_years.png"), dpi=150)
plt.show()

fig, ax = plt.subplots(figsize=(10, 4))
for metric, color in zip(["Event Precision","Event Recall","Event F1 Score"], ["orange","limegreen","darkred"]):
    m = summary[(metric,"mean")]
    s = summary[(metric,"std")]
    ax.plot(parts, m,    "o-", color=color, label=f"{metric}")
    ax.fill_between(parts, m-s, m+s, alpha=0.2, color=color)

for k, style in zip((1,2), ("--", ":")):
    pm = avg_marks.get(k)
    if pm is not None:
        ax.axvline(pm, color="orange", linestyle=style, label=f"-{k}yr avg")

ax.set(title="Event-wise Precision, Recall & F1 Score vs Part",
       xlabel="Part", ylabel="Score")
ax.legend(fontsize=6); ax.grid(); plt.tight_layout()
fig.savefig(os.path.join(out_dir, "agg_eventwise_with_years.png"), dpi=150)

plt.show()
