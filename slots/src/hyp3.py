"""Drift: is the reversal a linear ramp, when does it start, is start aligned in-episode?
Also: drift/shared change classification signals."""
import numpy as np, pandas as pd
from hypotheses import machines, xor, loglik
P3 = [(2, 5), (1, 3), (4, 7)]
def base(M):
    W = (M > 0).astype(float); return np.column_stack([np.ones(len(M))] + [xor(W, i, j) for i, j in P3])
res = []
for cond in ("drift", "shared_change", "stable"):
    for r, ts, M, y in [m for m in machines if m[0].condition == cond]:
        X, yb = base(M), (y > 0).astype(float)
        l0, _ = loglik(X, yb, 0.1)
        best_r = (-np.inf, None); best_c = (-np.inf, None)
        for tau in range(20, r.cutoff - 25, 4):
            rr = np.maximum(ts - tau, 0) / 100.0
            l, _ = loglik(np.column_stack([X, X * rr[:, None]]), yb, 0.1)
            if l > best_r[0]: best_r = (l, tau)
            k = np.searchsorted(ts, tau)
            if 25 < k < len(ts) - 25:
                l1, _ = loglik(X[:k], yb[:k], 0.1); l2, _ = loglik(X[k:], yb[k:], 0.1)
                if l1 + l2 > best_c[0]: best_c = (l1 + l2, tau)
        res.append((cond, r.episode_id, r.cutoff, best_r[1], best_r[0] - l0, best_c[1], best_c[0] - l0))
D = pd.DataFrame(res, columns=["cond", "ep", "cutoff", "ramp_tau", "ramp_gain", "cp_tau", "cp_gain"])
D["ramp_minus_cp"] = D.ramp_gain - D.cp_gain
print(D.groupby("cond")[["ramp_gain", "cp_gain", "ramp_minus_cp"]].describe().round(1).T.to_string())
print("\nwithin-episode std of best tau:")
print(D.groupby(["cond", "ep"])[["ramp_tau", "cp_tau"]].std().groupby("cond").median().round(1))
print("\ncutoff - tau median by cond:")
print(D.assign(r=D.cutoff - D.ramp_tau, c=D.cutoff - D.cp_tau).groupby("cond")[["r", "c"]].median())
E = D.groupby(["cond", "ep"])[["ramp_gain", "cp_gain"]].sum().reset_index()
print("\nepisode-summed gains (8 machines):")
print(E.groupby("cond")[["ramp_gain", "cp_gain"]].describe().round(0).T.to_string())
D.to_csv("regime_scan.csv", index=False)
