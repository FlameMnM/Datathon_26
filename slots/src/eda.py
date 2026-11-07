"""Exploratory analysis of the House Edge slot-machine data (pandas + numpy)."""
from pathlib import Path
import numpy as np
import pandas as pd

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)
DATA = Path(__file__).resolve().parents[1] / "kit" / "public"
PAY = [0, 1, 2, 5, 20]
PCOLS = [f"p_{p}" for p in PAY]


def load(split):
    q = pd.read_csv(DATA / split / "queries.csv")
    h = pd.read_csv(DATA / split / "history.csv")
    for name in ("labels", "conditions"):
        f = DATA / split / f"{name}.csv"
        if f.exists():
            q = q.merge(pd.read_csv(f), on="query_id")
    f = DATA / split / "targets.csv"
    if f.exists():
        q = q.merge(pd.read_csv(f).drop(columns=["episode_id", "outcome"]), on="query_id")
    return q, h


def mutual_info(a, b):
    """Mutual information (bits) between two discrete arrays."""
    ct = pd.crosstab(a, b).to_numpy().astype(float)
    pj = ct / ct.sum()
    outer = pj.sum(1, keepdims=True) * pj.sum(0, keepdims=True)
    nz = pj > 0
    return float((pj[nz] * np.log2(pj[nz] / outer[nz])).sum())


def section(t):
    print("\n" + "=" * 8, t, "=" * 8)


splits = {s: load(s) for s in ("train", "validation", "test")}

section("1. Sizes")
for s, (q, h) in splits.items():
    print(f"{s:10s} queries={len(q):5d} episodes={q.episode_id.nunique():4d} "
          f"machines/ep={q.groupby('episode_id').size().unique()} history_rows={len(h):7d}")

section("2. Query features (all splits)")
allq = pd.concat([q.assign(split=s) for s, (q, _) in splits.items()])
print(pd.crosstab(allq.machine_type, allq.split))
print(pd.crosstab(allq.signal, allq.split))
print("cutoff range by split:\n", allq.groupby("split").cutoff.describe()[["min", "50%", "max"]])
print("cutoffs constant within episode:", (allq.groupby("episode_id").cutoff.nunique() == 1).all())
print("machine types per episode (train):\n",
      splits["train"][0].groupby("episode_id").machine_type.value_counts().unstack(fill_value=0).describe().loc[["mean", "min", "max"]])

section("3. History coverage")
for s, (q, h) in splits.items():
    g = h.groupby("machine_id").tick.agg(["count", "min", "max"])
    g = g.join(q.set_index("machine_id")[["cutoff"] + (["condition"] if "condition" in q else [])])
    g["max_lt_cutoff"] = g["max"] < g["cutoff"]
    g["full"] = g["count"] == g["cutoff"]
    print(f"{s}: obs/machine {g['count'].value_counts().head(4).to_dict()} | all ticks<cutoff: {g.max_lt_cutoff.all()} "
          f"| full-history share {g.full.mean():.2f}")
    if "condition" in g:
        print(g.groupby("condition")[["count", "full"]].mean().round(2))

section("4. Payout distribution")
for s, (q, h) in splits.items():
    hh = h.merge(q[["machine_id", "machine_type", "signal"]], on="machine_id")
    print(s, "overall:", hh.payout.value_counts(normalize=True).reindex(PAY).round(3).to_dict())
    if s == "train":
        print(pd.crosstab(hh.machine_type, hh.payout, normalize="index").round(3))
        print(pd.crosstab([hh.machine_type, hh.signal], hh.payout, normalize="index").round(3))
        rtp = hh.groupby("machine_type").payout.mean()
        print("mean return per chip (RTP) by type:", rtp.round(3).to_dict())

section("5. Validation TRUE probabilities")
v, vh = splits["validation"]
T = v[PCOLS].to_numpy()
print("mean true probs:", dict(zip(PCOLS, T.mean(0).round(3))))
print("std  true probs:", dict(zip(PCOLS, T.std(0).round(3))))
print("max prob per query quantiles:", np.quantile(T.max(1), [.1, .25, .5, .75, .9]).round(3))
print("mean true probs by type:\n", v.groupby("machine_type")[PCOLS].mean().round(3))
print("mean true probs by condition:\n", v.groupby("condition")[PCOLS].mean().round(3))

ref = pd.read_csv(DATA / "validation" / "reference.csv").set_index("query_id").loc[v.query_id, PCOLS].to_numpy()


def cond_loss(pred):
    """Contest-style: per-query SSE -> mean per episode -> mean per condition."""
    d = v.assign(sse=((pred - T) ** 2).sum(1))
    return d.groupby(["condition", "episode_id"]).sse.mean().groupby("condition").mean()


B = cond_loss(ref)
print("pooled reference loss by condition:", B.round(4).to_dict())


def points(pred):
    return 60 / (1 + (cond_loss(pred) / B).mean())


print(f"reference points {points(ref):.2f}")
const = np.tile(T.mean(0), (len(v), 1))
print(f"oracle constant (mean truth) points {points(const):.2f}")
bytype = v.groupby("machine_type")[PCOLS].transform("mean").to_numpy()
print(f"oracle per-type mean points {points(bytype):.2f}")
bymach_avg = v.groupby(["machine_type", "signal"])[PCOLS].transform("mean").to_numpy()
print(f"oracle per-type×signal mean points {points(bymach_avg):.2f}")

section("6. Lag structure: does payout at lag k predict next payout? (train, full ticks only)")
q, h = splits["train"]
h = h.sort_values(["machine_id", "tick"])
wide = h.pivot(index="machine_id", columns="tick", values="payout")
# build (lag-k payout, current payout) pairs where both observed
rows = []
for k in range(1, 13):
    cur = wide.iloc[:, k:].to_numpy().ravel()
    prev = wide.iloc[:, :-k].to_numpy().ravel()
    m = ~np.isnan(cur) & ~np.isnan(prev)
    mi = mutual_info(prev[m], cur[m])
    rows.append((k, round(mi, 4), round(np.mean(cur[m] == 0), 3),
                 round(np.mean(cur[m][prev[m] == 0] == 0), 3), round(np.mean(cur[m][prev[m] > 0] == 0), 3)))
print(pd.DataFrame(rows, columns=["lag", "MI_bits_pooled", "P(0)", "P(0|lag=0)", "P(0|lag>0)"]).to_string(index=False))

print("\nPer-machine MI by lag (median and 90th pct across machines; machine-specific effects can cancel when pooled):")
mi_rows = []
for mid, row in wide.iterrows():
    arr = row.to_numpy()
    out = {"machine_id": mid}
    for k in range(1, 11):
        cur, prev = arr[k:], arr[:-k]
        m = ~np.isnan(cur) & ~np.isnan(prev)
        if m.sum() < 100:
            continue
        out[k] = mutual_info(prev[m], cur[m])
    mi_rows.append(out)
mi = pd.DataFrame(mi_rows).set_index("machine_id").dropna(how="all")
print(mi.quantile([.5, .9]).round(3))
# shuffled-null for comparison (MI is biased upward with finite samples)
rng = np.random.default_rng(0)
null = []
for mid, row in wide.head(100).iterrows():
    arr = row.dropna().to_numpy()
    sh = rng.permutation(arr)
    null.append(mutual_info(sh[1:], sh[:-1]))
print(f"shuffled-null MI median {np.median(null):.3f}, 90th pct {np.quantile(null, .9):.3f}")
print("share of machines with MI above null-90th, by lag:", (mi > np.quantile(null, .9)).mean().round(2).to_dict())
print("best lag per machine (counts):", mi.idxmax(axis=1).value_counts().sort_index().to_dict())

section("7. Drift: payout rate early vs late in history (train, full histories)")
qq = q.set_index("machine_id")
h2 = h.join(qq[["cutoff", "condition", "machine_type"]], on="machine_id")
h2["phase"] = pd.cut(h2.tick / h2.cutoff, [0, .33, .66, 1.0], labels=["early", "mid", "late"], include_lowest=True)
h2["win"] = h2.payout > 0
print(h2.pivot_table(index="condition", columns="phase", values="win", aggfunc="mean", observed=True).round(3))
print(h2.pivot_table(index="condition", columns="phase", values="payout", aggfunc="mean", observed=True).round(3))

section("8. Cross-machine: does other machines' last payout predict mine? (train)")
eps = h.merge(q[["machine_id", "episode_id"]], on=["machine_id", "episode_id"])
ep_mean = eps.assign(win=eps.payout > 0).groupby(["episode_id", "tick"]).win.mean().rename("ep_win")
x = eps.assign(win=(eps.payout > 0).astype(int)).join(ep_mean, on=["episode_id", "tick"])
x["ep_win_others"] = (x.ep_win * 8 - x.win) / 7  # approx (assumes 8 obs at tick)
x = x.sort_values(["machine_id", "tick"])
x["others_prev"] = x.groupby("machine_id").ep_win_others.shift(1)
x["self_prev"] = x.groupby("machine_id").win.shift(1)
print(x[["win", "others_prev", "self_prev"]].corr().round(3))

section("9. Episode-level co-movement in validation targets (shared change)")
v["pwin"] = 1 - v.p_0
print(v.groupby("condition").apply(lambda d: d.groupby("episode_id").pwin.std().mean(), include_groups=False).round(3)
      .rename("mean within-episode std of true P(win)"))

section("10. Does the train LABEL agree with history-only frequency? (sanity)")
q, h = splits["train"]
last = h.sort_values("tick").groupby("machine_id").tail(20).groupby("machine_id").payout.apply(lambda s: (s > 0).mean())
qq = q.set_index("machine_id").join(last.rename("recent_win"))
print("corr(recent 20 win-rate, label>0):", round(np.corrcoef(qq.recent_win, qq.outcome > 0)[0, 1], 3))
