from trials import run, TOP_PAIRS, ALL_PAIRS
P3 = TOP_PAIRS[:3]
base = {"pairs": P3, "lam": 0.03, "hl": None, "adaptive": [None, 80, 40, 20], "temp": 100, "singles": [], "xor": True}
T = [
 ("F1 3 XOR + single lag7", {**base, "singles": [7]}),
 ("F2 6 XOR", {**base, "pairs": TOP_PAIRS}),
 ("F3 28 XOR lam=1", {**base, "pairs": ALL_PAIRS, "lam": 1.0}),
 ("F4 28 XOR lam=3", {**base, "pairs": ALL_PAIRS, "lam": 3.0}),
 ("F5 3 XOR lam=0.3", {**base, "lam": 0.3}),
 ("F6 3 XOR cands {inf,60,30,15,8}", {**base, "adaptive": [None, 60, 30, 15, 8]}),
 ("F7 3 XOR cands {inf,40,20,10}", {**base, "adaptive": [None, 40, 20, 10]}),
 ("F8 3 XOR fixed hl=inf", {**base, "adaptive": None}),
]
log = []
for n, c in T: run(n, c, log)
