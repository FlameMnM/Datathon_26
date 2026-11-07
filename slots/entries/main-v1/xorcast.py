"""XOR-pair per-machine logistic forecaster for House Edge. Python standard library only.

P(win) for each machine comes from a weighted L2 logistic regression fitted on that machine's own
history, with features XOR(win[t-i], win[t-j]) for the lag pairs (2,5), (1,3), (4,7). Several
recency half-lives are fitted and blended, weighted by predictive log-loss on the machine's last V
observed ticks. Payout size given a win = the machine's own win-size counts shrunk toward a
per-type prior estimated on the training split.
"""
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

PAYOUTS = (0, 1, 2, 5, 20)
SIZES = (1, 2, 5, 20)

DEFAULT_CONFIG = {
    "pairs": [[2, 5], [1, 3], [4, 7]],
    "max_lag": 8,
    "lam": 0.03,              # L2 penalty on XOR coefficients
    "intercept_lam": 1e-4,
    "half_lives": [None, 40, 20, 10],  # None = no recency weighting
    "holdout_ticks": 30,      # V: last ticks used to score half-lives
    "temp": 100.0,            # softmax temperature on mean holdout log-likelihood
    "min_train": 20, "min_holdout": 5,
    "size_strength": 20.0,    # pseudo-count weight of the type size prior
    "newton_iters": 25, "newton_tol": 1e-7,
}


# ------------------------------------------------------------------ IO
def read_queries(path):
    with open(path, newline="", encoding="utf-8") as f:
        return [dict(r, signal=int(r["signal"]), cutoff=int(r["cutoff"])) for r in csv.DictReader(f)]


def read_history(path):
    """{(episode_id, machine_id): {tick: payout}}"""
    hist = defaultdict(dict)
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            hist[(r["episode_id"], r["machine_id"])][int(r["tick"])] = int(float(r["payout"]))
    return hist


# ------------------------------------------------------------------ math
def sigmoid(z):
    z = max(-30.0, min(30.0, z))
    return 1.0 / (1.0 + math.exp(-z))


def solve(A, b):
    """Gaussian elimination with partial pivoting (small dense systems)."""
    n = len(b)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(M[r][c]))
        M[c], M[p] = M[p], M[c]
        piv = M[c][c]
        for r in range(c + 1, n):
            f = M[r][c] / piv
            if f:
                for k in range(c, n + 1):
                    M[r][k] -= f * M[c][k]
    x = [0.0] * n
    for r in range(n - 1, -1, -1):
        x[r] = (M[r][n] - sum(M[r][k] * x[k] for k in range(r + 1, n))) / M[r][r]
    return x


def fit_logreg(X, y, w, cfg):
    d = len(X[0])
    pen = [cfg["intercept_lam"]] + [cfg["lam"]] * (d - 1)
    beta = [0.0] * d
    for _ in range(cfg["newton_iters"]):
        g = [pen[j] * beta[j] for j in range(d)]
        H = [[0.0] * d for _ in range(d)]
        for j in range(d):
            H[j][j] = pen[j]
        for xi, yi, wi in zip(X, y, w):
            p = sigmoid(sum(a * b for a, b in zip(xi, beta)))
            r, s = wi * (p - yi), wi * p * (1 - p)
            for j in range(d):
                g[j] += r * xi[j]
                sj = s * xi[j]
                Hj = H[j]
                for k in range(d):
                    Hj[k] += sj * xi[k]
        step = solve(H, g)
        beta = [b - s for b, s in zip(beta, step)]
        if max(abs(s) for s in step) < cfg["newton_tol"]:
            break
    return beta


# ------------------------------------------------------------------ model
def features(ticks, t, cfg):
    """Return feature row for predicting tick t, or None if any needed lag is unobserved."""
    wins = []
    for k in range(1, cfg["max_lag"] + 1):
        v = ticks.get(t - k)
        if v is None:
            return None
        wins.append(1.0 if v > 0 else 0.0)
    return [1.0] + [abs(wins[i - 1] - wins[j - 1]) for i, j in cfg["pairs"]]


def recency(ts, ref, hl):
    return [1.0] * len(ts) if hl is None else [2.0 ** (-(ref - 1 - t) / hl) for t in ts]


def win_probability(ticks, cutoff, cfg):
    rows = []
    for t in sorted(ticks):
        if t >= cutoff:
            continue
        x = features(ticks, t, cfg)
        if x is not None:
            rows.append((t, x, 1.0 if ticks[t] > 0 else 0.0))
    xq = features(ticks, cutoff, cfg)
    if xq is None or not rows:  # fallback: smoothed raw win rate
        obs = [v for t, v in ticks.items() if t < cutoff]
        return (sum(v > 0 for v in obs) + 1.0) / (len(obs) + 2.0)
    ts = [r[0] for r in rows]; X = [r[1] for r in rows]; y = [r[2] for r in rows]
    V, hls = cfg["holdout_ticks"], cfg["half_lives"]
    tr = [i for i, t in enumerate(ts) if t < cutoff - V]
    va = [i for i, t in enumerate(ts) if t >= cutoff - V]
    scores = []
    for hl in hls:
        if len(tr) < cfg["min_train"] or len(va) < cfg["min_holdout"]:
            scores.append(0.0)
            continue
        b = fit_logreg([X[i] for i in tr], [y[i] for i in tr], recency([ts[i] for i in tr], cutoff - V, hl), cfg)
        ll = 0.0
        for i in va:
            p = min(max(sigmoid(sum(a * c for a, c in zip(X[i], b))), 1e-4), 1 - 1e-4)
            ll += y[i] * math.log(p) + (1 - y[i]) * math.log(1 - p)
        scores.append(ll / len(va))
    top = max(scores)
    wts = [math.exp((s - top) * cfg["temp"]) for s in scores]
    tot = sum(wts)
    pw = 0.0
    for hl, wt in zip(hls, wts):
        b = fit_logreg(X, y, recency(ts, cutoff, hl), cfg)
        pw += wt / tot * sigmoid(sum(a * c for a, c in zip(xq, b)))
    return pw


def size_distribution(ticks, cutoff, prior, cfg):
    cnt = [0.0] * 4
    for t, v in ticks.items():
        if t < cutoff and v > 0:
            cnt[SIZES.index(v)] += 1
    k = cfg["size_strength"]
    tot = sum(cnt) + k
    return [(c + k * p) / tot for c, p in zip(cnt, prior)]


def fit_size_priors(queries, history):
    """Per-type distribution of payout size given a win, from training histories."""
    counts = defaultdict(lambda: [0.0] * 4)
    for q in queries:
        for v in history.get((q["episode_id"], q["machine_id"]), {}).values():
            if v > 0:
                counts[q["machine_type"]][SIZES.index(v)] += 1
    return {t: [c / sum(cs) for c in cs] for t, cs in sorted(counts.items())}


def predict(queries, history, model):
    cfg, priors = model["config"], model["size_priors"]
    fallback = [sum(p[i] for p in priors.values()) / len(priors) for i in range(4)]
    out = {}
    for q in sorted(queries, key=lambda r: r["query_id"]):
        ticks = history.get((q["episode_id"], q["machine_id"]), {})
        pw = win_probability(ticks, q["cutoff"], cfg)
        sz = size_distribution(ticks, q["cutoff"], priors.get(q["machine_type"], fallback), cfg)
        probs = [1.0 - pw] + [pw * s for s in sz]
        total = sum(probs)
        out[q["query_id"]] = [p / total for p in probs]
    return out


def write_submission(path, preds):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["query_id", "p_0", "p_1", "p_2", "p_5", "p_20"])
        for qid in sorted(preds):
            w.writerow([qid] + [repr(float(p)) for p in preds[qid]])


def load_model(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))
