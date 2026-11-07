"""Deeper dive: which lags / lag pairs drive outcomes, per machine, and how signal relates to direction."""
from pathlib import Path
import itertools
import numpy as np
import pandas as pd

pd.set_option("display.width", 200)
DATA = Path(__file__).resolve().parents[1] / "kit" / "public"
PAY = np.array([0, 1, 2, 5, 20])
PCOLS = [f"p_{p}" for p in PAY]


def wide(split):
    q = pd.read_csv(DATA / split / "queries.csv")
    h = pd.read_csv(DATA / split / "history.csv")
    w = h.pivot(index="machine_id", columns="tick", values="payout")
    w = w.reindex(columns=range(int(q.cutoff.max())))
    return q.set_index("machine_id"), w


def lag_table(q, w, maxlag=8):
    """Self-supervised rows: for each machine and tick t, outcome at t and payouts at t-1..t-maxlag."""
    arr = w.to_numpy()
    out = []
    for i, mid in enumerate(w.index):
        a = arr[i]
        T = len(a)
        t = np.arange(maxlag, T)
        df = pd.DataFrame({"machine_id": mid, "t": t, "y": a[t]})
        for k in range(1, maxlag + 1):
            df[f"l{k}"] = a[t - k]
        out.append(df)
    d = pd.concat(out)
    d = d.join(q[["episode_id", "machine_type", "signal", "cutoff"]], on="machine_id")
    d = d[d.t < d.cutoff].dropna()
    return d


q, w = wide("train")
cond = pd.read_csv(DATA / "train/conditions.csv")
q = q.join(cond.set_index(q.reset_index().set_index("query_id").index.name or "query_id"), on="query_id")
d = lag_table(q, w)
d["win"] = (d.y > 0).astype(int)
for k in range(1, 9):
    d[f"w{k}"] = (d[f"l{k}"] > 0).astype(int)
print("self-supervised rows (train, fully observed 8 lags):", len(d))

print("\n=== A. Per-machine effect of each lag on win (P(win|lag win) - P(win|lag loss)) ===")
eff = {}
for k in range(1, 9):
    g = d.groupby(["machine_id", f"w{k}"]).win.mean().unstack()
    eff[k] = g[1] - g[0]
eff = pd.DataFrame(eff).dropna(how="all")
print("mean |effect| by lag:", eff.abs().mean().round(3).to_dict())
print("mean signed effect by lag:", eff.mean().round(3).to_dict())
meta = q[["machine_type", "signal", "condition"]]
e7 = eff.join(meta)
print("\nlag-7 effect by machine_type x signal (mean, and share positive):")
print(e7.groupby(["machine_type", "signal"])[7].agg(["mean", lambda s: (s > 0).mean(), "count"]).round(3))

print("\n=== B. Strongest single lag per machine & its |effect| distribution ===")
best = eff.abs().fillna(0).idxmax(axis=1)
print(best.value_counts().sort_index().to_dict())
print("max |effect| quantiles:", eff.abs().max(axis=1).quantile([.1, .25, .5, .75, .9]).round(3).to_dict())

print("\n=== C. Pair interactions: XOR/AND of two lags (pooled MI, top 10) ===")


def mi(x, y):
    ct = pd.crosstab(x, y).to_numpy().astype(float)
    pj = ct / ct.sum(); o = pj.sum(1, keepdims=True) * pj.sum(0, keepdims=True); nz = pj > 0
    return float((pj[nz] * np.log2(pj[nz] / o[nz])).sum())


# per machine: MI(win ; (wi,wj)) averaged across machines — catches machine-specific directions
pairs = []
for i, j in itertools.combinations(range(1, 9), 2):
    vals = []
    for mid, g in d.groupby("machine_id"):
        vals.append(mi(g[f"w{i}"] * 2 + g[f"w{j}"], g.win) - mi(g[f"w{i}"], g.win) - mi(g[f"w{j}"], g.win))
    pairs.append((i, j, np.mean(vals)))
pairs = pd.DataFrame(pairs, columns=["i", "j", "mean_extra_MI_from_pair"]).sort_values("mean_extra_MI_from_pair", ascending=False)
print(pairs.head(10).round(4).to_string(index=False))
print("...median pair:", pairs.mean_extra_MI_from_pair.median().round(4))

print("\n=== D. Does payout SIZE at a lag matter beyond win/loss? (lag 7, pooled by type) ===")
print(d.pivot_table(index="machine_type", columns="l7", values="win", aggfunc="mean").round(3))
print("\nmean next payout given lag-7 payout:")
print(d.pivot_table(index="machine_type", columns="l7", values="y", aggfunc="mean").round(3))

print("\n=== E. Validation: how sharp are true probs vs observed lag context? ===")
qv, wv = wide("validation")
tv = pd.read_csv(DATA / "validation/targets.csv")
cv = pd.read_csv(DATA / "validation/conditions.csv")
qv = qv.reset_index().merge(tv, on=["query_id", "episode_id"]).merge(cv, on="query_id").set_index("machine_id")
arr = wv.to_numpy()
for k in range(1, 9):
    qv[f"l{k}"] = [arr[wv.index.get_loc(m), c - k] for m, c in zip(qv.index, qv.cutoff)]
qv["pwin"] = 1 - qv.p_0
print("missing lag share (short history should have 0 since last 40 consecutive):",
      qv[[f"l{k}" for k in range(1, 9)]].isna().mean().round(3).to_dict())
for k in (1, 2, 4, 7):
    print(f"true P(win) by lag{k} win:", qv.groupby(qv[f"l{k}"] > 0).pwin.mean().round(3).to_dict())
print("\ntrue P(win) quantiles by machine_type:")
print(qv.groupby("machine_type").pwin.quantile([.1, .5, .9]).unstack().round(3))
print("\nconditional on winning, true size distribution (normalized) by type:")
sz = qv[PCOLS[1:]].div(qv[PCOLS[1:]].sum(1), axis=0)
print(sz.groupby(qv.machine_type).mean().round(3))
print("std of size-dist across queries:", sz.std().round(3).to_dict())
