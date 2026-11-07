"""Error decomposition: swap in the TRUE win prob or TRUE size distribution to see which part loses points."""
import numpy as np
from trials import SPLITS, predict, TOP_PAIRS
from hm import contest_points, PCOLS
qv, sv = SPLITS["validation"]
cfg = {"pairs": TOP_PAIRS, "lam": 0.1, "hl": None, "adaptive": [None, 80, 40, 20]}
P = predict(qv, sv, cfg); T = qv[PCOLS].to_numpy()
pw, tw = 1 - P[:, 0], 1 - T[:, 0]
size, tsize = P[:, 1:] / pw[:, None], T[:, 1:] / np.maximum(tw[:, None], 1e-12)
def show(n, X):
    pts, R = contest_points(qv, X); print(f"{n:38s} {pts:6.2f}", R.round(3).to_dict())
show("model", P)
show("true win prob + model size", np.c_[1 - tw, tw[:, None] * size])
show("model win prob + true size", np.c_[1 - pw, pw[:, None] * tsize])
# does true size dist vary within a machine type? (only one query per machine, so look at spread)
import pandas as pd
sz = pd.DataFrame(tsize, columns=PCOLS[1:]); sz["type"] = qv.machine_type
print("within-type std of TRUE size dist:\n", sz.groupby("type").std().round(3))
print("within-type std of MODEL size dist:\n", pd.DataFrame(size, columns=PCOLS[1:]).assign(type=qv.machine_type).groupby("type").std().round(3))
