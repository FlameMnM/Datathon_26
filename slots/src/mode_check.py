import numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold
from v2 import load, predict, BASE, SIZE_MODES
from hm import contest_points, brier_labels
from mode_calib import llr, feats, Fv, y, qv, sv, pw, Pv
A, B = -0.252, 0.094   # calibrated P(high) = sigmoid(A + B * llr_all), fit on validation
def sized(q, s, P):
    pw = 1 - P[:, 0]
    S = []
    for r in q.itertuples():
        ph = 1 / (1 + np.exp(-(A + B * llr(s[r.machine_id], r.cutoff, r.machine_type))))
        lo, hi = SIZE_MODES[r.machine_type]; d = (1 - ph) * lo + ph * hi; S.append(d / d.sum())
    return np.c_[1 - pw, pw[:, None] * np.array(S)]
# train-label Brier (independent episodes)
qt, st = load("train")
Pt, _ = predict(qt, st, BASE)
print("train Brier current:", round(brier_labels(qt, Pt).mean(), 5), " with calibrated mode:", round(brier_labels(qt, sized(qt, st, Pt)).mean(), 5))
# split-half consistency on validation
P2 = sized(qv, sv, Pv)
rng = np.random.default_rng(1); eps = sorted(qv.episode_id.unique())
ref = pd.read_csv("../kit/public/validation/reference.csv").set_index("query_id").loc[qv.query_id, ["p_0","p_1","p_2","p_5","p_20"]].to_numpy()
d = []
for s in range(10):
    m = qv.episode_id.isin(set(rng.choice(eps, 48, replace=False))).values
    for mm in (m, ~m):
        q = qv[mm].reset_index(drop=True)
        d.append(contest_points(q, P2[mm], ref[mm])[0] - contest_points(q, Pv[mm], ref[mm])[0])
d = np.array(d); print(f"validation half-splits: gain mean {d.mean():+.3f}, min {d.min():+.3f}, positive in {np.mean(d>0)*100:.0f}% of 20 halves")
# LightGBM out-of-fold mode probs -> contest points (reference only, not shippable stdlib)
import lightgbm as lgb
from mode_probe import X
p = np.zeros(len(y))
for tr, te in GroupKFold(5).split(X, y, qv.episode_id):
    p[te] = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.03, num_leaves=7, min_child_samples=20, verbose=-1).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
S = np.array([((1 - ph) * SIZE_MODES[r.machine_type][0] + ph * SIZE_MODES[r.machine_type][1]) for r, ph in zip(qv.itertuples(), p)]); S /= S.sum(1, keepdims=True)
print("LightGBM OOF mode -> points", round(contest_points(qv, np.c_[1 - pw, pw[:, None] * S])[0], 3), " vs current", round(contest_points(qv, Pv)[0], 3))
