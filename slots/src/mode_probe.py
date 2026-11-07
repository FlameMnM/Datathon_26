"""Can the hidden payout-size mode (low/high) at the query tick be predicted with >60% accuracy?
Labels: validation true size distribution at the query tick (exactly one of two per type)."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
from v2 import load, SIZE_MODES, SIZE_IDX

qv, sv = load("validation")
M = pd.read_csv("size_modes.csv", index_col=0)
qv["high"] = (M["mode"] == "high").astype(int).values
y = qv.high.to_numpy()
print(f"validation queries {len(y)}; P(high)={y.mean():.3f}; always-low accuracy={1 - y.mean():.3f}")


# ------------------------------------------------------------------ 1. unsupervised: how fast does the mode switch?
def hmm_ll(seq_list, eps, p_high=0.46):
    """Total log-likelihood of win-size emissions under a 2-state chain with per-tick switch prob eps."""
    tot = 0.0
    for mtype, a, cutoff in seq_list:
        lo, hi = SIZE_MODES[mtype]; lo, hi = lo / lo.sum(), hi / hi.sum()
        f = np.array([1 - p_high, p_high])
        for t in range(cutoff):
            f = f * (1 - eps) + f[::-1] * eps
            v = a[t]
            if not np.isnan(v) and v > 0:
                e = np.array([lo[SIZE_IDX[int(v)]], hi[SIZE_IDX[int(v)]]])
                s = f @ e
                tot += np.log(s)
                f = f * e / s
    return tot


seqs = []
for split in ("train", "validation"):
    q, s = load(split)
    q = q[~q.condition.isin(["short_history"])]
    seqs += [(r.machine_type, s[r.machine_id], r.cutoff) for r in q.sample(200, random_state=0).itertuples()]
print("\n[1] Unsupervised HMM fit on 400 full-history machines: log-lik of size emissions by switch prob eps")
lls = {}
for eps in (0.0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.2, 0.3, 0.5):
    lls[eps] = hmm_ll(seqs, eps)
best = max(lls, key=lls.get)
for e, l in lls.items():
    print(f"   eps={e:<6} LL={l:10.1f}  {'<-- best' if e == best else ''}  (gain vs memoryless eps=.5: {l - lls[0.5]:+.1f})")


# ------------------------------------------------------------------ 2. label-based methods
def filt(a, cutoff, mtype, eps, window=None):
    lo, hi = SIZE_MODES[mtype]; lo, hi = lo / lo.sum(), hi / hi.sum()
    f = np.array([0.54, 0.46])
    start = 0 if window is None else max(0, cutoff - window)
    for t in range(start, cutoff):
        if eps:
            f = f * (1 - eps) + f[::-1] * eps
        v = a[t]
        if not np.isnan(v) and v > 0:
            f = f * np.array([lo[SIZE_IDX[int(v)]], hi[SIZE_IDX[int(v)]]]); f /= f.sum()
    if eps:
        f = f * (1 - eps) + f[::-1] * eps
    return f[1]


def report(name, p):
    acc = np.mean((p > 0.5) == y)
    # also best-threshold accuracy (optimistic, reported separately)
    print(f"   {name:48s} acc={acc:.3f}  AUC={roc_auc_score(y, p):.3f}  Brier={np.mean((p - y) ** 2):.4f}")
    return acc


print("\n[2] Filtering the size mode from the machine's own history (no fitting to labels):")
for eps in (0.0, 0.01, 0.03, 0.1, best):
    report(f"HMM posterior eps={eps}", np.array([filt(sv[r.machine_id], r.cutoff, r.machine_type, eps) for r in qv.itertuples()]))
for w in (1, 3, 5, 10, 20, 40):
    report(f"likelihood ratio of last {w} ticks' wins (eps=0)",
           np.array([filt(sv[r.machine_id], r.cutoff, r.machine_type, 0.0, window=w) for r in qv.itertuples()]))

print("\n[3] Supervised models (5-fold CV grouped by episode) on observable features:")
rows = []
for r in qv.itertuples():
    a = sv[r.machine_id]; c = r.cutoff
    d = {f"llr_last{w}": np.log(filt(a, c, r.machine_type, 0, w) / (1 - filt(a, c, r.machine_type, 0, w))) for w in (1, 5, 20)}
    d["llr_all"] = np.log(filt(a, c, r.machine_type, 0) / (1 - filt(a, c, r.machine_type, 0)))
    for k in range(1, 9):
        v = a[c - k]
        for p in (0, 1, 2, 5, 20):
            d[f"l{k}_is{p}"] = float(v == p)
    d["wins_last8"] = np.nansum(a[c - 8:c] > 0); d["wins_last20"] = np.nansum(a[c - 20:c] > 0)
    d["signal"] = r.signal; d["cutoff"] = c; d["cut_parity"] = c % 2
    for t in ("steady", "balanced", "jackpot"):
        d[f"type_{t}"] = float(r.machine_type == t)
    d["midx"] = int(r.machine_id[-3:])
    rows.append(d)
X = pd.DataFrame(rows).fillna(0).to_numpy()
gkf = GroupKFold(5)
for name, model in [("logistic L2 (C=0.1)", LogisticRegression(C=0.1, max_iter=2000)),
                    ("logistic L2 (C=1)", LogisticRegression(C=1.0, max_iter=2000))]:
    p = np.zeros(len(y))
    for tr, te in gkf.split(X, y, qv.episode_id):
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
        p[te] = model.fit((X[tr] - mu) / sd, y[tr]).predict_proba((X[te] - mu) / sd)[:, 1]
    report(name, p)
try:
    import lightgbm as lgb
    p = np.zeros(len(y))
    for tr, te in gkf.split(X, y, qv.episode_id):
        m = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.03, num_leaves=7, min_child_samples=20, verbose=-1)
        p[te] = m.fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    report("LightGBM (shallow)", p)
except ImportError:
    pass

print("\n[4] Deterministic-rule checks: P(high) by simple observables")
F = pd.DataFrame(rows); F["high"] = y
for col in ("cut_parity", "signal", "midx", "wins_last8"):
    print(f"   by {col}:", F.groupby(col).high.agg(["mean", "count"]).round(2).T.to_dict())
for k in (1, 2, 7):
    print(f"   by payout at lag {k}:", F.assign(v=[sv[r.machine_id][r.cutoff - k] for r in qv.itertuples()]).groupby("v").high.mean().round(2).to_dict())
