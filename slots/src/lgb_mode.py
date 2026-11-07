"""Rigorous check: does a LightGBM size-mode classifier beat v3's calibrated LLR, in contest points?
Repeated grouped CV (different episode shuffles), out-of-fold probabilities only."""
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from v2 import load, SIZE_MODES, SIZE_IDX
from hm import contest_points, PCOLS

qv, sv = load("validation")
y = (pd.read_csv("size_modes.csv", index_col=0)["mode"] == "high").astype(int).values
V3 = pd.read_csv("../runs/main-v3/validation_predictions.csv").set_index("query_id").loc[qv.query_id, PCOLS].to_numpy()
pw = 1 - V3[:, 0]
ref = pd.read_csv("../kit/public/validation/reference.csv").set_index("query_id").loc[qv.query_id, PCOLS].to_numpy()


def llr(a, c, mtype, start=0):
    lo, hi = SIZE_MODES[mtype]; lo, hi = lo / lo.sum(), hi / hi.sum()
    v = a[max(start, 0):c]; v = v[~np.isnan(v) & (v > 0)]
    return float(sum(np.log(hi[SIZE_IDX[int(x)]] / lo[SIZE_IDX[int(x)]]) for x in v))


def features(q, s, extra=False):
    rows = []
    for r in q.itertuples():
        a = s[r.machine_id]; c = r.cutoff
        d = {"llr_all": llr(a, c, r.machine_type)}
        for w in (1, 5, 20, 40):
            d[f"llr_last{w}"] = llr(a, c, r.machine_type, c - w)
        for k in range(1, 9):
            for p in (0, 1, 2, 5, 20):
                d[f"l{k}_is{p}"] = float(a[c - k] == p)
        d["wins_last8"] = float(np.nansum(a[c - 8:c] > 0)); d["wins_last20"] = float(np.nansum(a[c - 20:c] > 0))
        d["n_obs"] = float(np.sum(~np.isnan(a[:c])))
        d["signal"] = float(r.signal)
        for t in ("steady", "balanced", "jackpot"):
            d[f"type_{t}"] = float(r.machine_type == t)
        if extra:
            d["cutoff"] = float(c); d["midx"] = float(int(r.machine_id[-3:]))
        rows.append(d)
    return pd.DataFrame(rows)


def size_from(ph):
    S = np.array([(1 - p) * SIZE_MODES[t][0] + p * SIZE_MODES[t][1] for t, p in zip(qv.machine_type, ph)])
    return S / S.sum(1, keepdims=True)


def pts(ph, mask=None):
    P = np.c_[1 - pw, pw[:, None] * size_from(ph)]
    if mask is None:
        return contest_points(qv, P)[0]
    q = qv[mask].reset_index(drop=True)
    return contest_points(q, P[mask], ref[mask])[0]


v3_ph = 1 / (1 + np.exp(-(-0.252 + 0.094 * features(qv, sv)["llr_all"].to_numpy())))
print(f"v3 (calibrated LLR, fitted on all val)  acc={np.mean((v3_ph > .5) == y):.3f}  points={pts(v3_ph):.3f}")

PARAMS = {"n_estimators": 200, "learning_rate": 0.03, "num_leaves": 7, "min_child_samples": 20, "verbose": -1, "n_jobs": 1}
eps = qv.episode_id.to_numpy(); ueps = np.unique(eps)
results = {}
for name, extra in (("lgb clean feats", False), ("lgb + cutoff/midx", True)):
    X = features(qv, sv, extra).to_numpy()
    accs, gains, ps = [], [], []
    for rep in range(5):
        rng = np.random.default_rng(rep); perm = dict(zip(ueps, rng.permutation(len(ueps))))
        groups = np.array([perm[e] % 5 for e in eps])
        ph = np.zeros(len(y)); ph_cal = np.zeros(len(y))
        for f in range(5):
            tr, te = groups != f, groups == f
            ph[te] = lgb.LGBMClassifier(**PARAMS, random_state=rep).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
            # v3-style calibrated LLR, also out-of-fold, for a fair comparison
            from sklearn.linear_model import LogisticRegression
            ph_cal[te] = LogisticRegression().fit(X[tr][:, :1], y[tr]).predict_proba(X[te][:, :1])[:, 1]
        accs.append((np.mean((ph > .5) == y), np.mean((ph_cal > .5) == y)))
        gains.append(pts(ph) - pts(ph_cal))
        ps.append(ph)
    accs = np.array(accs)
    print(f"{name:20s} acc lgb={accs[:, 0].mean():.3f}±{accs[:, 0].std():.3f}  acc calibrated-LLR={accs[:, 1].mean():.3f}  "
          f"points gain vs calibrated-LLR (OOF, 5 reps): mean {np.mean(gains):+.3f}  min {np.min(gains):+.3f}")
    results[name] = np.mean(ps, 0)

# half-split stability of the clean model's OOF probs vs v3
ph = results["lgb clean feats"]
rng = np.random.default_rng(7); d = []
for s in range(10):
    m = np.isin(eps, rng.choice(ueps, len(ueps) // 2, replace=False))
    for mm in (m, ~m):
        d.append(pts(ph, mm) - pts(v3_ph, mm))
d = np.array(d)
print(f"clean lgb (OOF) vs shipped v3 on 20 val halves: mean {d.mean():+.3f}, min {d.min():+.3f}, positive {np.mean(d > 0) * 100:.0f}%")
print(f"full-val points: shipped v3 {pts(v3_ph):.3f}  clean lgb OOF {pts(ph):.3f}  blend 50/50 OOF {pts((ph + v3_ph) / 2):.3f}")
