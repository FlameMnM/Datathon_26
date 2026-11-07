import sys
from v2 import run, BASE
T = {
 "V3 episode tau": {**BASE, "tau_mode": "episode"},
 "V4 own tau": {**BASE, "tau_mode": "own"},
 "V5 size from post-change": {**BASE, "size_post": True},
 "V6 post-change hl=40": {**BASE, "post_hl": 40},
 "H1 sd=0.7": {**BASE, "sd": 0.7},
 "H2 sd=1.5": {**BASE, "sd": 1.5},
 "H3 p_none=0.15": {**BASE, "p_none": 0.15},
 "H4 b0sd=3": {**BASE, "b0sd": 3.0},
 "H5 min_seg=15": {**BASE, "min_seg": 15},
 "H6 tau_step=1": {**BASE, "tau_step": 1},
}
sel = sys.argv[1:]
for n, c in T.items():
    if not sel or n.split()[0] in sel: run(n, c)
