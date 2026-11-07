from trials import run, TOP_PAIRS
P3 = TOP_PAIRS[:3]
base = {"pairs": P3, "lam": 0.03, "hl": None, "adaptive": [None, 80, 40, 20], "temp": 100}
T = [
 ("E1 3 pairs (D9+temp100)", base),
 ("E2 3 pairs, singles {1,2,3,4,5,7}", {**base, "singles": [1, 2, 3, 4, 5, 7]}),
 ("E3 3 XOR features only", {**base, "singles": [], "xor": True}),
 ("E4 3 XOR + 8 singles", {**base, "xor": True}),
 ("E5 2 pairs (2,5),(1,3)", {**base, "pairs": TOP_PAIRS[:2]}),
 ("E6 4 pairs +(4,5)", {**base, "pairs": TOP_PAIRS[:4]}),
 ("E7 3 pairs lam=0.01", {**base, "lam": 0.01}),
 ("E8 3 pairs lam=0.1", {**base, "lam": 0.1}),
 ("E9 3 XOR only lam=0.01", {**base, "singles": [], "xor": True, "lam": 0.01}),
]
log = []
for n, c in T: run(n, c, log)
