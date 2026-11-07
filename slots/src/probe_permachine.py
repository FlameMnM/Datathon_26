"""Feasibility probe: per-machine logistic model of win on lag-win indicators + key pairs,
size distribution from type-level priors. Scored on validation true probabilities."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

DATA = Path(__file__).resolve().parents[1] / "kit" / "public"
PCOLS = ["p_0", "p_1", "p_2", "p_5", "p_20"]
PAIRS = [(2, 5), (1, 3), (4, 7), (4, 5), (2, 6), (3, 8)]

q = pd.read_csv(DATA / "validation/queries.csv")
h = pd.read_csv(DATA / "validation/history.csv")
t = pd.read_csv(DATA / "validation/targets.csv").set_index("query_id")
c = pd.read_csv(DATA / "validation/conditions.csv").set_index("query_id")
ref = pd.read_csv(DATA / "validation/reference.csv").set_index("query_id")
q = q.join(t[PCOLS]).join(c)  # q index default; join on query_id below
q = pd.read_csv(DATA / "validation/queries.csv").join(t[PCOLS], on="query_id").join(c, on="query_id")
W = h.pivot(index="machine_id", columns="tick", values="payout").reindex(columns=range(260))

# type-level size distribution given win, from history
hh = h.merge(q[["machine_id", "machine_type"]], on="machine_id")
size_prior = (hh[hh.payout > 0].groupby("machine_type").payout.value_counts(normalize=True)
              .unstack().reindex(columns=[1, 2, 5, 20]).fillna(0))


def feats(a, ts):
    X = np.stack([(a[ts - k] > 0).astype(float) for k in range(1, 9)], 1)
    P = np.stack([X[:, i - 1] * X[:, j - 1] for i, j in PAIRS], 1)
    return np.hstack([X, P])


def predict(C, use_pairs=True):
    preds = []
    for r in q.itertuples():
        a = W.loc[r.machine_id].to_numpy()
        ts = np.arange(8, r.cutoff)
        ok = ~np.isnan(a[ts]) & np.all([~np.isnan(a[ts - k]) for k in range(1, 9)], 0)
        ts = ts[ok]
        X, y = feats(a, ts), (a[ts] > 0).astype(int)
        xq = feats(a, np.array([r.cutoff]))
        if not use_pairs:
            X, xq = X[:, :8], xq[:, :8]
        if len(np.unique(y)) < 2:
            pw = y.mean() * 0.9 + 0.05
        else:
            m = LogisticRegression(C=C, max_iter=1000).fit(X, y)
            pw = m.predict_proba(xq)[0, 1]
        sz = size_prior.loc[r.machine_type].to_numpy()
        # per-machine size dist shrunk to type prior
        wins = a[~np.isnan(a) & (a > 0)]
        cnt = np.array([(wins == v).sum() for v in (1, 2, 5, 20)])
        sz = (cnt + 20 * sz) / (cnt.sum() + 20)
        preds.append(np.concatenate([[1 - pw], pw * sz]))
    return np.array(preds)


T = q[PCOLS].to_numpy()


def cond_loss(P):
    d = q.assign(sse=((P - T) ** 2).sum(1))
    return d.groupby(["condition", "episode_id"]).sse.mean().groupby("condition").mean()


B = cond_loss(ref.loc[q.query_id, PCOLS].to_numpy())
for C in (0.1, 0.3, 1.0, 3.0):
    for up in (False, True):
        L = cond_loss(predict(C, up))
        print(f"C={C:<4} pairs={up!s:5}  points={60 / (1 + (L / B).mean()):.2f}  ratios={(L / B).round(3).to_dict()}")
