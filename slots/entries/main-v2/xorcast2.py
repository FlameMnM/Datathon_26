"""House Edge forecaster v2 — Python standard library only.

Generative structure found in the data (see README):
  * P(win_t) = sigmoid(b0 + c * XOR(win[t-i], win[t-j])) with exactly one active lag pair (i,j) in
    {(2,5), (1,3), (4,7)} per machine (|c| ~ 4.85, sign machine-specific), or no active pair.
  * drift / shared_change episodes contain a regime change; shared changes hit all machines of the
    episode at the same tick.
  * payout size given a win follows a per-type distribution (two hidden modes; we use their mixture).

Algorithm per episode:
  1. Change-point scan: for each machine and candidate tick tau, log-likelihood gain of fitting the
     3-XOR logistic model separately before/after tau. Sum gains over the episode's machines.
     If the best summed gain > threshold, the episode changed; each machine uses the episode tau unless
     its own best tau is clearly better.
  2. Win probability: Bayesian model averaging (Laplace evidence) over 7 models: no active pair, or one
     of 3 pairs with coefficient prior N(+-4.85, 1). Fitted on post-change ticks if changed.
  3. Size: machine's win-size counts shrunk to the per-type prior (pseudo-count 20).
Because features are 3 binary XORs, every fit uses sufficient statistics (counts per pattern).
"""
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

PAIRS = ((2, 5), (1, 3), (4, 7))
SIZES = (1, 2, 5, 20)

DEFAULT_CONFIG = {
    "mu": 4.85, "sd": 1.0, "b0sd": 2.0, "p_none": 0.07,
    "type_b0": {"steady": 0.87, "balanced": -0.67, "jackpot": -2.57},
    "short_obs": 100, "tau_lo": 30, "tau_step": 3, "min_seg": 25, "ep_thresh": 150.0,
    "own_margin": 3.0, "cp_lam": 0.1, "size_strength": 20.0,
    "newton_iters": 60, "newton_tol": 1e-8, "max_step": 2.0,
}


# ------------------------------------------------------------------ IO
def read_queries(path):
    with open(path, newline="", encoding="utf-8") as f:
        return [dict(r, signal=int(r["signal"]), cutoff=int(r["cutoff"])) for r in csv.DictReader(f)]


def read_history(path):
    hist = defaultdict(dict)
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            hist[(r["episode_id"], r["machine_id"])][int(r["tick"])] = int(float(r["payout"]))
    return hist


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


# ------------------------------------------------------------------ math
def sigmoid(z):
    z = max(-30.0, min(30.0, z))
    return 1.0 / (1.0 + math.exp(-z))


def solve(A, b):
    n = len(b)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(M[r][c]))
        M[c], M[p] = M[p], M[c]
        for r in range(c + 1, n):
            f = M[r][c] / M[c][c]
            if f:
                for k in range(c, n + 1):
                    M[r][k] -= f * M[c][k]
    x = [0.0] * n
    for r in range(n - 1, -1, -1):
        x[r] = (M[r][n] - sum(M[r][k] * x[k] for k in range(r + 1, n))) / M[r][r]
    return x


def logdet(A):
    """log|det A| via Gaussian elimination (A symmetric positive definite here)."""
    n = len(A)
    M = [row[:] for row in A]
    s = 0.0
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(M[r][c]))
        M[c], M[p] = M[p], M[c]
        s += math.log(abs(M[c][c]))
        for r in range(c + 1, n):
            f = M[r][c] / M[c][c]
            for k in range(c, n):
                M[r][k] -= f * M[c][k]
    return s


def map_fit(rows, mu, prec, cfg):
    """MAP logistic regression on grouped data. rows: list of (x_vector, n_trials, n_wins).
    Gaussian prior N(mu, diag(1/prec)). Damped Newton from zero. Returns (beta, loglik, Hessian)."""
    d = len(mu)
    b = [0.0] * d

    def grad_hess(b):
        g = [prec[j] * (b[j] - mu[j]) for j in range(d)]
        H = [[0.0] * d for _ in range(d)]
        for j in range(d):
            H[j][j] = prec[j]
        for x, n, k in rows:
            p = sigmoid(sum(a * c for a, c in zip(x, b)))
            r, s = n * p - k, n * p * (1 - p)
            for j in range(d):
                if x[j]:
                    g[j] += r * x[j]
                    sj = s * x[j]
                    for l in range(d):
                        H[j][l] += sj * x[l]
        return g, H

    for _ in range(cfg["newton_iters"]):
        g, H = grad_hess(b)
        st = solve(H, g)
        mx = max(abs(v) for v in st)
        if mx > cfg["max_step"]:
            st = [v * cfg["max_step"] / mx for v in st]
        b = [bj - sj for bj, sj in zip(b, st)]
        if mx < cfg["newton_tol"]:
            break
    _, H = grad_hess(b)
    ll = 0.0
    for x, n, k in rows:
        p = min(max(sigmoid(sum(a * c for a, c in zip(x, b))), 1e-12), 1 - 1e-12)
        ll += k * math.log(p) + (n - k) * math.log(1 - p)
    return b, ll, H


def log_evidence(b, ll, H, mu, prec):
    return (ll + 0.5 * sum(math.log(p) for p in prec)
            - 0.5 * sum(p * (bj - m) ** 2 for p, bj, m in zip(prec, b, mu)) - 0.5 * logdet(H))


# ------------------------------------------------------------------ features / sufficient stats
def machine_series(ticks, cutoff):
    """Observed training rows (t, pattern, win) for t < cutoff with lags 1..8 observed."""
    out = []
    for t in sorted(ticks):
        if t < 8 or t >= cutoff:
            continue
        pat = pattern(ticks, t)
        if pat is not None:
            out.append((t, pat, 1 if ticks[t] > 0 else 0))
    return out


def pattern(ticks, t):
    w = []
    for k in range(1, 9):
        v = ticks.get(t - k)
        if v is None:
            return None
        w.append(1 if v > 0 else 0)
    return sum((w[i - 1] ^ w[j - 1]) << n for n, (i, j) in enumerate(PAIRS))


def bits(pat):
    return [(pat >> n) & 1 for n in range(3)]


class Prefix:
    """Prefix sums of (trials, wins) per pattern over a machine's rows, for O(8) segment stats."""

    def __init__(self, rows):
        self.ts = [r[0] for r in rows]
        self.n = [[0] * 8]
        self.k = [[0] * 8]
        for _, pat, y in rows:
            n, k = self.n[-1][:], self.k[-1][:]
            n[pat] += 1
            k[pat] += y
            self.n.append(n)
            self.k.append(k)

    def index(self, tau):  # number of rows with t < tau
        lo, hi = 0, len(self.ts)
        while lo < hi:
            m = (lo + hi) // 2
            if self.ts[m] < tau:
                lo = m + 1
            else:
                hi = m
        return lo

    def seg(self, a, b):
        return [self.n[b][p] - self.n[a][p] for p in range(8)], [self.k[b][p] - self.k[a][p] for p in range(8)]


def seg_ll_l2(n, k, cfg):
    if sum(n) < 5:
        return 0.0
    rows = [([1.0] + [float(v) for v in bits(p)], n[p], k[p]) for p in range(8) if n[p]]
    lam = cfg["cp_lam"]
    return map_fit(rows, [0.0] * 4, [1e-4, lam, lam, lam], cfg)[1]


def bma_win(n, k, xq_bits, b0m, cfg):
    mu, sd, b0sd, p_none = cfg["mu"], cfg["sd"], cfg["b0sd"], cfg["p_none"]
    evs, preds = [], []
    rows = [([1.0], sum(n), sum(k))]
    m, pr = [b0m], [1 / b0sd ** 2]
    b, ll, H = map_fit(rows, m, pr, cfg)
    evs.append(log_evidence(b, ll, H, m, pr) + math.log(p_none))
    preds.append(sigmoid(b[0]))
    for j in range(3):
        n2, k2 = [0, 0], [0, 0]
        for p in range(8):
            v = (p >> j) & 1
            n2[v] += n[p]
            k2[v] += k[p]
        rows = [([1.0, float(v)], n2[v], k2[v]) for v in (0, 1) if n2[v]]
        for s in (1, -1):
            m, pr = [b0m, s * mu], [1 / b0sd ** 2, 1 / sd ** 2]
            b, ll, H = map_fit(rows, m, pr, cfg)
            evs.append(log_evidence(b, ll, H, m, pr) + math.log((1 - p_none) / 6))
            preds.append(sigmoid(b[0] + b[1] * xq_bits[j]))
    top = max(evs)
    wts = [math.exp(e - top) for e in evs]
    tot = sum(wts)
    return sum(w * p for w, p in zip(wts, preds)) / tot


# ------------------------------------------------------------------ prediction
def fit_size_priors(queries, history):
    counts = defaultdict(lambda: [0.0] * 4)
    for q in queries:
        for v in history.get((q["episode_id"], q["machine_id"]), {}).values():
            if v > 0:
                counts[q["machine_type"]][SIZES.index(v)] += 1
    return {t: [c / sum(cs) for c in cs] for t, cs in sorted(counts.items())}


def size_distribution(ticks, cutoff, prior, k):
    cnt = [0.0] * 4
    for t, v in ticks.items():
        if t < cutoff and v > 0:
            cnt[SIZES.index(v)] += 1
    tot = sum(cnt) + k
    return [(c + k * p) / tot for c, p in zip(cnt, prior)]


def predict(queries, history, model):
    cfg, priors = model["config"], model["size_priors"]
    fallback = [sum(p[i] for p in priors.values()) / len(priors) for i in range(4)]
    episodes = defaultdict(list)
    for q in queries:
        episodes[q["episode_id"]].append(q)
    out = {}
    for ep in sorted(episodes):
        qs = sorted(episodes[ep], key=lambda r: r["query_id"])
        cutoff = qs[0]["cutoff"]
        info = []
        for q in qs:
            ticks = history.get((q["episode_id"], q["machine_id"]), {})
            rows = machine_series(ticks, cutoff)
            info.append((q, ticks, Prefix(rows)))
        n_obs = sum(sum(1 for t in tk if t < cutoff) for _, tk, _ in info) / len(info)
        short = n_obs < cfg["short_obs"]
        taus = list(range(cfg["tau_lo"], cutoff - cfg["min_seg"] + 1, cfg["tau_step"]))
        changed, profiles, ep_i = False, [], None
        if not short and taus:
            for q, ticks, pre in info:
                N = len(pre.ts)
                full = seg_ll_l2(*pre.seg(0, N), cfg)
                prof = []
                for tau in taus:
                    i = pre.index(tau)
                    if i < cfg["min_seg"] or N - i < cfg["min_seg"]:
                        prof.append(-math.inf)
                    else:
                        prof.append(seg_ll_l2(*pre.seg(0, i), cfg) + seg_ll_l2(*pre.seg(i, N), cfg) - full)
                profiles.append(prof)
            S = [sum(max(p[t], -50.0) for p in profiles) for t in range(len(taus))]
            ep_i = max(range(len(taus)), key=lambda t: S[t])  # first maximum
            changed = S[ep_i] > cfg["ep_thresh"]
        for m_idx, (q, ticks, pre) in enumerate(info):
            N = len(pre.ts)
            start = 0
            if changed:
                prof = profiles[m_idx]
                own_i = max(range(len(taus)), key=lambda t: prof[t])
                use_i = ep_i if prof[own_i] - prof[ep_i] < cfg["own_margin"] else own_i
                start = pre.index(taus[use_i])
            n, k = pre.seg(start, N)
            pq = pattern(ticks, cutoff)
            b0m = cfg["type_b0"].get(q["machine_type"], 0.0)
            if pq is None or sum(n) == 0:
                obs = [v for t, v in ticks.items() if t < cutoff]
                pw = (sum(v > 0 for v in obs) + 1.0) / (len(obs) + 2.0)
            else:
                pw = bma_win(n, k, bits(pq), b0m, cfg)
            sz = size_distribution(ticks, cutoff, priors.get(q["machine_type"], fallback), cfg["size_strength"])
            probs = [1.0 - pw] + [pw * s for s in sz]
            tot = sum(probs)
            out[q["query_id"]] = [p / tot for p in probs]
    return out
