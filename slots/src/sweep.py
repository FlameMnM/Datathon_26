import sys
from trials import run, TOP_PAIRS, ALL_PAIRS
P10 = TOP_PAIRS + [(1, 2), (2, 3), (3, 4), (1, 4)]
ad = [None, 80, 40, 20]
base = {"pairs": TOP_PAIRS, "lam": 0.1, "hl": None, "adaptive": ad}
T = [
 ("D1 lam=0.03", {**base, "lam": 0.03}),
 ("D2 lam=0.01", {**base, "lam": 0.01}),
 ("D3 lam=0.03 cands+10", {**base, "lam": 0.03, "adaptive": [None, 80, 40, 20, 10]}),
 ("D4 lam=0.03 temp=100", {**base, "lam": 0.03, "temp": 100}),
 ("D5 lam=0.03 temp=10", {**base, "lam": 0.03, "temp": 10}),
 ("D6 lam=0.03 V=50", {**base, "lam": 0.03, "V": 50}),
 ("D7 lam=0.03 V=20", {**base, "lam": 0.03, "V": 20}),
 ("D8 lam=0.03 10 pairs", {**base, "lam": 0.03, "pairs": P10}),
 ("D9 lam=0.03 3 pairs", {**base, "lam": 0.03, "pairs": TOP_PAIRS[:3]}),
]
log = []
for n, c in T: run(n, c, log)
