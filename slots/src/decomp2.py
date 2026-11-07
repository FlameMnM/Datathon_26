import numpy as np, pandas as pd
from v2 import predict, BASE, load
from hm import contest_points, PCOLS
qv, sv = load("validation")
P, _ = predict(qv, sv, BASE); T = qv[PCOLS].to_numpy()
pw, tw = 1 - P[:, 0], 1 - T[:, 0]
size, tsize = P[:, 1:] / pw[:, None], T[:, 1:] / np.maximum(tw[:, None], 1e-12)
def show(n, X):
    pts, R = contest_points(qv, X); print(f"{n:34s} {pts:6.2f}", R.round(3).to_dict())
show("v2 model", P)
show("true win + model size", np.c_[1 - tw, tw[:, None] * size])
show("model win + true size", np.c_[1 - pw, pw[:, None] * tsize])
# calibration: temperature on win logit
lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
for t in (0.8, 0.9, 1.1, 1.25):
    pw2 = 1 / (1 + np.exp(-lg(pw) * t)); show(f"win-logit temperature x{t}", np.c_[1 - pw2, pw2[:, None] * size])
# size: does true size dist vary with condition / is our prior strength right?
for k in (5, 50, 200):
    from v2 import size_dist, SIZE_PRIOR
    S2 = np.array([size_dist(sv[r.machine_id], r.cutoff, SIZE_PRIOR[r.machine_type], k) for r in qv.itertuples()])
    show(f"size prior strength k={k}", np.c_[1 - pw, pw[:, None] * S2])
