import numpy as np, pandas as pd
from v2 import *
qv, sv = load("validation")
cfg = BASE
rows = []
for r in qv[qv.condition == "stable"].itertuples():
    a = sv[r.machine_id]; ts, X, y = xrows(a, r.cutoff); xq = xq_of(a, r.cutoff)
    pb, _ = bma(X, y, np.ones(len(y)), xq, TYPE_B0[r.machine_type], cfg)
    pl, _ = l2fit(X, y, np.ones(len(y)), xq)
    Xd = np.column_stack([np.ones(len(y)), X]); b, _, _ = map_fit(Xd, y, np.ones(len(y)), np.zeros(4), np.array([1e-4, .1, .1, .1]))
    rows.append((r.machine_id, r.machine_type, 1 - r.p_0, pb, pl, *np.round(b, 1), *xq))
D = pd.DataFrame(rows, columns=["m", "type", "true", "bma", "l2", "b0", "c25", "c13", "c47", "x25", "x13", "x47"])
D["eb"], D["el"] = (D.bma - D.true).abs(), (D.l2 - D.true).abs()
print(D.sort_values("eb", ascending=False).head(15).round(3).to_string(index=False))
