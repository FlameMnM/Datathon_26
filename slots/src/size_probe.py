import numpy as np, pandas as pd
from v2 import load, SIZE_PRIOR
from hm import PCOLS
qv, sv = load("validation")
T = qv[PCOLS].to_numpy(); tw = 1 - T[:, 0]
ts = T[:, 1:] / np.maximum(tw[:, None], 1e-12)
D = pd.DataFrame(ts, columns=["s1", "s2", "s5", "s20"]); D["type"] = qv.machine_type; D["signal"] = qv.signal; D["cond"] = qv.condition
# expected payout given win (summary of size dist)
D["Ewin"] = ts @ np.array([1, 2, 5, 20])
feats = {}
for r in qv.itertuples():
    a = sv[r.machine_id][:r.cutoff]; c = r.cutoff
    v = a[~np.isnan(a)]; w = v[v > 0]
    rec = a[-40:]; rec = rec[~np.isnan(rec)]; rw = rec[rec > 0]
    f = {"emp_E_all": w.mean() if len(w) else np.nan, "emp_E_recent40": rw.mean() if len(rw) else np.nan,
         "win_rate_recent": (rec > 0).mean()}
    for k in range(1, 9):
        f[f"pay_l{k}"] = a[c - k]
        f[f"win_l{k}"] = float(a[c - k] > 0)
    feats[r.Index] = f
F = pd.DataFrame.from_dict(feats, orient="index")
D = D.join(F)
D["prior_E"] = D.type.map({t: SIZE_PRIOR[t] @ np.array([1, 2, 5, 20]) for t in SIZE_PRIOR})
D["resid"] = D.Ewin - D.prior_E
print("true E[payout|win] by type:\n", D.groupby("type").Ewin.describe().round(2))
print("\ncorr of residual E[payout|win] (true - type prior) with features, by type:")
cols = ["signal", "emp_E_all", "emp_E_recent40", "win_rate_recent"] + [f"win_l{k}" for k in range(1, 9)] + [f"pay_l{k}" for k in range(1, 9)]
print(D.groupby("type")[cols + ["resid"]].corr()["resid"].unstack().drop(columns="resid").round(2).T.to_string())
print("\nby condition resid std:", D.groupby("cond").resid.std().round(3).to_dict())
D.to_csv("size_probe.csv")
