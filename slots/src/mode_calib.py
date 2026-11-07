"""Calibrated size-mode classifier -> size distribution; measure accuracy AND effect on contest score.
All probabilities used for scoring are out-of-fold (5-fold, grouped by episode)."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
from v2 import load, predict, BASE, SIZE_MODES, SIZE_IDX
from hm import contest_points, brier_labels, PCOLS

qv, sv = load("validation")
y = (pd.read_csv("size_modes.csv", index_col=0)["mode"] == "high").astype(int).values


def llr(a, cutoff, mtype, start=0):
    lo, hi = SIZE_MODES[mtype]; lo, hi = lo / lo.sum(), hi / hi.sum()
    s = 0.0
    for t in range(start, cutoff):
        v = a[t]
        if not np.isnan(v) and v > 0:
            s += np.log(hi[SIZE_IDX[int(v)]] / lo[SIZE_IDX[int(v)]])
    return s


def feats(q, s):
    out = []
    for r in q.itertuples():
        a = s[r.machine_id]; c = r.cutoff
        nw = np.nansum(a[:c] > 0)
        out.append([llr(a, c, r.machine_type), llr(a, c, r.machine_type, c - 40), nw])
    return np.array(out)


Fv = feats(qv, sv)
SETS = {"llr_all": [0], "llr_all+llr_last40": [0, 1], "llr_all+last40+n_wins": [0, 1, 2]}

# baseline: current v2 size distribution -> implied P(high) by least squares
Pv, _ = predict(qv, sv, BASE)
pw = 1 - Pv[:, 0]; size_now = Pv[:, 1:] / pw[:, None]
imp = []
for r, sz in zip(qv.itertuples(), size_now):
    lo, hi = SIZE_MODES[r.machine_type]; d = hi - lo
    imp.append(np.clip((sz - lo) @ d / (d @ d), 0, 1))
imp = np.array(imp)
print(f"current v2 (shrink k=20) implied P(high): acc={np.mean((imp > .5) == y):.3f} AUC={roc_auc_score(y, imp):.3f} Brier={np.mean((imp - y) ** 2):.4f}")
base_pts, base_R = contest_points(qv, Pv)
print(f"current v2 points {base_pts:.3f}  {base_R.round(3).to_dict()}\n")

gkf = GroupKFold(5)
for name, cols in SETS.items():
    for C in (0.1, 1.0):
        p = np.zeros(len(y)); coefs = []
        for tr, te in gkf.split(Fv, y, qv.episode_id):
            m = LogisticRegression(C=C, max_iter=1000).fit(Fv[tr][:, cols], y[tr])
            p[te] = m.predict_proba(Fv[te][:, cols])[:, 1]; coefs.append(np.r_[m.intercept_, m.coef_[0]])
        # plug out-of-fold P(high) into the size distribution
        S = np.array([(1 - ph) * SIZE_MODES[r.machine_type][0] + ph * SIZE_MODES[r.machine_type][1]
                      for r, ph in zip(qv.itertuples(), p)])
        S /= S.sum(1, keepdims=True)
        pts, R = contest_points(qv, np.c_[1 - pw, pw[:, None] * S])
        print(f"{name:26s} C={C:<4} acc={np.mean((p > .5) == y):.3f} AUC={roc_auc_score(y, p):.3f} Brier={np.mean((p - y) ** 2):.4f}"
              f"  -> points {pts:.3f} ({pts - base_pts:+.3f})  coefs(mean over folds)={np.round(np.mean(coefs, 0), 3)}")
