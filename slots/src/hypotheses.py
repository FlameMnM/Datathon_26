"""Rigorous hypothesis tests on the slot-machine generating process.

Null model H0 (current): logit P(win_t) = b0 + sum_p b_p * XOR(win[t-i], win[t-j]),  pairs (2,5),(1,3),(4,7)
Each test adds terms and runs a per-machine likelihood-ratio test (LRT ~ chi2(df) under H0).
If H0 is complete, p-values are Uniform(0,1): the share with p<0.01 should be ~1%.
Uses STABLE machines from train+validation (no regime change, full history) unless stated.
"""
import itertools
import numpy as np
import pandas as pd
from scipy import stats
from hm import load, fit_logreg, sigmoid, PCOLS

np.set_printoptions(suppress=True, linewidth=160)
PAIRS = [(2, 5), (1, 3), (4, 7)]
L = 12

frames = []
for s in ("train", "validation"):
    q, series = load(s)
    q["split"] = s
    frames.append((q, series))


def rows(a, cutoff):
    ts = np.arange(L, cutoff)
    M = np.stack([a[ts - k] for k in range(1, L + 1)], 1)
    ok = ~np.isnan(M).any(1) & ~np.isnan(a[ts])
    return ts[ok], M[ok], a[ts[ok]]


def xor(W, i, j):
    return np.abs(W[:, i - 1] - W[:, j - 1])


def loglik(X, y, lam=1e-3):
    b = fit_logreg(X, y, np.ones(len(y)), lam)
    p = np.clip(sigmoid(X @ b), 1e-12, 1 - 1e-12)
    return float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))), b


machines = []  # (q row, ts, payouts-lag matrix, y payouts)
for q, series in frames:
    for r in q.itertuples():
        ts, M, y = rows(series[r.machine_id], r.cutoff)
        if len(ts) >= 150:
            machines.append((r, ts, M, y))
stable = [m for m in machines if m[0].condition == "stable"]
print(f"machines with >=150 usable rows: {len(machines)}; stable: {len(stable)}")


def lrt_battery(name, extra_fn, df, pool=stable, binar=lambda M: (M > 0).astype(float)):
    ps, gains = [], []
    for r, ts, M, y in pool:
        W = binar(M)
        yb = (y > 0).astype(float)
        X0 = np.column_stack([np.ones(len(yb))] + [xor(W, i, j) for i, j in PAIRS])
        X1 = np.column_stack([X0, extra_fn(W, M, r, ts)])
        l0, _ = loglik(X0, yb)
        l1, _ = loglik(X1, yb)
        stat = max(2 * (l1 - l0), 0)
        ps.append(stats.chi2.sf(stat, df))
        gains.append((l1 - l0) / len(yb))
    ps = np.array(ps)
    print(f"{name:52s} df={df:<3d} share p<.01={np.mean(ps < .01):.3f} (null .010)  "
          f"p<.05={np.mean(ps < .05):.3f} (null .050)  KS-uniform p={stats.kstest(ps, 'uniform').pvalue:.3g}  "
          f"mean LL gain/row={np.mean(gains):.4f}")
    return ps


print("\n=== T1. Is the 3-XOR win model complete? (stable machines) ===")
lrt_battery("+ 8 single lag wins", lambda W, M, r, ts: W[:, :8], 8)
lrt_battery("+ single lags 9..12", lambda W, M, r, ts: W[:, 8:12], 4)
others = [p for p in itertools.combinations(range(1, 9), 2) if p not in PAIRS]
lrt_battery(f"+ all {len(others)} other XOR pairs (lags 1-8)",
            lambda W, M, r, ts: np.column_stack([xor(W, i, j) for i, j in others]), len(others))
lrt_battery("+ XOR pairs reaching lags 9-12 (sample of 8)",
            lambda W, M, r, ts: np.column_stack([xor(W, i, j) for i, j in [(1, 9), (2, 10), (3, 11), (4, 12), (5, 9), (6, 10), (7, 11), (8, 12)]]), 8)
lrt_battery("+ interactions among the 3 XOR terms (3 two-way + 1 three-way)",
            lambda W, M, r, ts: np.column_stack([xor(W, 2, 5) * xor(W, 1, 3), xor(W, 2, 5) * xor(W, 4, 7),
                                                 xor(W, 1, 3) * xor(W, 4, 7), xor(W, 2, 5) * xor(W, 1, 3) * xor(W, 4, 7)]), 4)
lrt_battery("+ AND terms of the 3 pairs (XOR vs full 2x2 table)",
            lambda W, M, r, ts: np.column_stack([W[:, i - 1] * W[:, j - 1] for i, j in PAIRS]
                                                + [W[:, i - 1] for i, j in PAIRS] ), 6)
lrt_battery("+ log(payout+1) at the 6 pair lags (size matters?)",
            lambda W, M, r, ts: np.log1p(M[:, [1, 4, 0, 2, 3, 6]]), 6)
lrt_battery("+ linear time trend (drift inside stable?)", lambda W, M, r, ts: ((ts - ts.mean()) / 100.0)[:, None], 1)

print("\n=== T2. Alternative definitions of 'win' in the XOR (compare total log-lik, stable) ===")
for nm, fn in [("payout>0 (current)", lambda M: (M > 0)), ("payout>1", lambda M: (M > 1)),
               ("payout>=5", lambda M: (M >= 5)), ("payout==1", lambda M: (M == 1))]:
    tot = 0.0
    for r, ts, M, y in stable:
        W = fn(M).astype(float)
        X0 = np.column_stack([np.ones(len(y))] + [xor(W, i, j) for i, j in PAIRS])
        tot += loglik(X0, (y > 0).astype(float))[0]
    print(f"  {nm:20s} total LL = {tot:10.1f}")

print("\n=== T3. Does payout SIZE (given win) depend on the XOR state or lags? (stable, chi2 independence) ===")
for nm, feat in [("XOR(2,5)", lambda W: xor(W, 2, 5)), ("XOR(1,3)", lambda W: xor(W, 1, 3)),
                 ("XOR(4,7)", lambda W: xor(W, 4, 7)), ("win lag1", lambda W: W[:, 0]), ("win lag7", lambda W: W[:, 6])]:
    ps = []
    for r, ts, M, y in stable:
        W = (M > 0).astype(float)
        m = y > 0
        ct = pd.crosstab(feat(W)[m], y[m])
        if ct.shape[0] == 2 and ct.shape[1] >= 2:
            ps.append(stats.chi2_contingency(ct)[1])
    ps = np.array(ps)
    print(f"  size ~ {nm:10s} share p<.01={np.mean(ps < .01):.3f}  p<.05={np.mean(ps < .05):.3f}  n={len(ps)}")

print("\n=== T4. Coefficient structure: are effect sizes discrete / tied to type or signal? (stable) ===")
coefs = []
for r, ts, M, y in stable:
    W = (M > 0).astype(float)
    X0 = np.column_stack([np.ones(len(y))] + [xor(W, i, j) for i, j in PAIRS])
    _, b = loglik(X0, (y > 0).astype(float), lam=0.03)
    coefs.append([r.machine_type, r.signal, *b])
C = pd.DataFrame(coefs, columns=["type", "signal", "b0", "x25", "x13", "x47"])
print(C.describe().round(2))
print("mean coefs by type:\n", C.groupby("type")[["b0", "x25", "x13", "x47"]].mean().round(2))
print("mean coefs by signal:\n", C.groupby("signal")[["b0", "x25", "x13", "x47"]].mean().round(2))
print("share positive by type:\n", C.groupby("type")[["x25", "x13", "x47"]].agg(lambda s: (s > 0).mean()).round(2))
print("histogram of |x25|,|x13|,|x47| (look for discrete magnitudes):")
for c in ("x25", "x13", "x47"):
    print(f"  {c}", np.histogram(C[c], bins=np.arange(-8, 8.1, 1))[0])
print("corr of coefs:\n", C[["b0", "x25", "x13", "x47"]].corr().round(2))
C.to_csv("stable_coefs.csv", index=False)

print("\n=== T5. Cross-machine effects within an episode (stable): other machines' recent wins ===")
ep_series = {}
for q, series in frames:
    for ep, g in q.groupby("episode_id"):
        ep_series[ep] = np.vstack([series[m] for m in g.machine_id])


def others_feat(W, M, r, ts):
    E = ep_series[r.episode_id]
    feats = []
    for k in (1, 2, 7):
        win_k = np.nanmean(E[:, ts - k] > 0, axis=0)  # includes self; fine as approx
        feats.append(win_k)
    return np.column_stack(feats)


lrt_battery("+ episode mean win at lags 1,2,7", others_feat, 3)

print("\n=== T6. Drift machines: linear-in-time coefficients vs constant (train+val drift) ===")
drift = [m for m in machines if m[0].condition == "drift"]


def tv(W, M, r, ts):
    tt = (ts - ts.mean()) / 100.0
    return np.column_stack([tt] + [tt * xor(W, i, j) for i, j in PAIRS])


lrt_battery("drift: + time x (intercept, 3 XOR)", tv, 4, pool=drift)
lrt_battery("stable: + time x (intercept, 3 XOR)  [control]", tv, 4, pool=stable)
sc = [m for m in machines if m[0].condition == "shared_change"]
lrt_battery("shared_change: + time x (intercept, 3 XOR)", tv, 4, pool=sc)
