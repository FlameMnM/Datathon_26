import itertools, numpy as np, pandas as pd
from hypotheses import machines, stable, xor, loglik
ALLP = list(itertools.combinations(range(1, 9), 2))
print("\n=== A. Which XOR pairs are active per stable machine? (fit all 28 XOR, L2=0.3) ===")
act = []
for r, ts, M, y in stable:
    W = (M > 0).astype(float)
    X = np.column_stack([np.ones(len(y))] + [xor(W, i, j) for i, j in ALLP])
    _, b = loglik(X, (y > 0).astype(float), lam=0.3)
    act.append(b[1:])
A = pd.DataFrame(act, columns=[f"{i}{j}" for i, j in ALLP])
big = A.abs() > 2.0
print("share of machines where pair is active (|b|>2):")
print(big.mean().sort_values(ascending=False).round(3).head(12).to_dict())
print("number of active pairs per machine:", big.sum(1).value_counts().sort_index().to_dict())
print("active-set patterns (top 12):")
print(big.apply(lambda r: ",".join(c for c in big.columns if r[c]), axis=1).value_counts().head(12).to_string())
mag = A.where(big).stack()
print("active |b| quantiles:", mag.abs().quantile([.1, .25, .5, .75, .9]).round(2).to_dict())

print("\n=== B. Drift: rolling-window coefficient trajectories (first 4 drift machines) ===")
drift = [m for m in machines if m[0].condition == "drift"]
P3 = [(2, 5), (1, 3), (4, 7)]
for r, ts, M, y in drift[:4]:
    W = (M > 0).astype(float)
    X = np.column_stack([np.ones(len(y))] + [xor(W, i, j) for i, j in P3])
    yb = (y > 0).astype(float)
    print(f"{r.machine_id} ({r.machine_type}, cutoff {r.cutoff})")
    for s in range(0, len(ts) - 50, 25):
        sl = slice(s, s + 50)
        _, b = loglik(X[sl], yb[sl], lam=0.1)
        print(f"   ticks {ts[sl][0]:3d}-{ts[sl][-1]:3d}  b0={b[0]:6.2f}  x25={b[1]:6.2f}  x13={b[2]:6.2f}  x47={b[3]:6.2f}")

print("\n=== C. Shared change: best single change point per machine; aligned within episode? ===")
sc = [m for m in machines if m[0].condition == "shared_change"]
rows = []
for r, ts, M, y in sc:
    W = (M > 0).astype(float)
    X = np.column_stack([np.ones(len(y))] + [xor(W, i, j) for i, j in P3])
    yb = (y > 0).astype(float)
    best = (-np.inf, None)
    l_all, _ = loglik(X, yb, lam=0.1)
    for k in range(30, len(ts) - 30, 3):
        l1, b1 = loglik(X[:k], yb[:k], lam=0.1); l2, b2 = loglik(X[k:], yb[k:], lam=0.1)
        if l1 + l2 > best[0]:
            best = (l1 + l2, ts[k], b1, b2)
    rows.append((r.episode_id, r.machine_id, r.cutoff, best[1], best[0] - l_all, *np.round(best[2], 1), *np.round(best[3], 1)))
D = pd.DataFrame(rows, columns=["ep", "m", "cutoff", "cp", "llgain", "b0a", "x25a", "x13a", "x47a", "b0b", "x25b", "x13b", "x47b"])
print(D.head(16).to_string(index=False))
print("\nwithin-episode std of change point:", D.groupby("ep").cp.std().describe().round(1).to_dict())
print("change point relative to cutoff (cutoff - cp):", (D.cutoff - D.cp).describe().round(1).to_dict())
