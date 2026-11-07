import numpy as np
import v2
from v2 import *
qv, sv = load("validation")
r = qv[qv.machine_id == "main-validation-e00064-m003"].iloc[0]
a = sv[r.machine_id]; ts, X, y = xrows(a, r.cutoff); xq = xq_of(a, r.cutoff)
cfg = BASE; one = np.ones((len(y), 1)); w = np.ones(len(y))
print("n", len(y), "xq", xq)
for k in range(3):
    for s in (1, -1):
        Xk = np.column_stack([one, X[:, k]]); m = np.array([1.0, s * 4.85]); pr = np.array([1/4, 1.0])
        b, ll, H = map_fit(Xk, y, w, m, pr)
        print(k, s, "b", b.round(2), "ll", round(ll, 1), "ev", round(log_evidence(b, ll, H, m, pr), 1), "pred", round(float(sig(b[0] + b[1]*xq[k])), 3))
