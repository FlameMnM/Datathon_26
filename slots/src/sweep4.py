from trials import run, TOP_PAIRS
base = {"pairs": TOP_PAIRS[:3], "lam": 0.03, "hl": None, "adaptive": [None, 40, 20, 10], "temp": 100, "singles": [], "xor": True}
T = [
 ("G1 temp=300", {**base, "temp": 300}),
 ("G2 temp=30", {**base, "temp": 30}),
 ("G3 V=40", {**base, "V": 40}),
 ("G4 V=60", {**base, "V": 60}),
 ("G5 V=40 cands {inf,80,40,20,10}", {**base, "V": 40, "adaptive": [None, 80, 40, 20, 10]}),
 ("G6 lam=0.1 V=40", {**base, "V": 40, "lam": 0.1}),
]
log = []
for n, c in T: run(n, c, log)
