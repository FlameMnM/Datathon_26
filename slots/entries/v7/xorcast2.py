"""House Edge forecaster v7 (structure learned from the round's training split; based on v2-v6) — Python standard library only.

Generative structure found in the data (see README):
  * P(win_t) = sigmoid(b0 + c * XOR(win[t-i], win[t-j])) with exactly one active lag pair (i,j) in
    {(2,5), (1,3), (4,7)} per machine (|c| ~ 4.85, sign machine-specific), or no active pair.
  * drift / shared_change episodes contain a regime change; shared changes hit all machines of the
    episode at the same tick.
  * payout size given a win is one of two per-type distributions (hidden low/high mode); we use a
    calibrated posterior mixture of the two.

Algorithm per episode:
  1. Change-point scan: for each machine and candidate tick tau, log-likelihood gain of fitting the
     3-XOR logistic model separately before/after tau. Sum gains over the episode's machines.
     If the best summed gain > threshold, the episode changed; each machine uses the episode tau unless
     its own best tau is clearly better.
  2. Win probability: Bayesian model averaging (Laplace evidence) over 7 models: no active pair, or one
     of 3 pairs with coefficient prior N(+-4.85, 1). Fitted on post-change ticks if changed.
  3. Size: P(high mode) = sigmoid(a + b * log-likelihood-ratio of the machine's observed win sizes),
     mixture of the two exact per-type size distributions.
Because features are 3 binary XORs, every fit uses sufficient statistics (counts per pattern).
"""
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

PAIRS = ((2, 5), (1, 3), (4, 7))
SIZES = (1, 2, 5, 20)
FLIP_ON = True
FL_EPS = 0.06
FL_SD = 0.75
FL_SHRINK = 0.87
MID = {'balanced': (-0.78, 0.33), 'jackpot': (-2.03, 0.30), 'steady': (0.54, 0.62)}
SIG = 0.35
MIDSD = 0.6
CSD = 0.7
MU = 4.85

DEFAULT_CONFIG = {
    "mu": 4.85, "sd": 1.0, "b0sd": 2.0, "p_none": 0.07,
    "type_b0": {"steady": 0.87, "balanced": -0.67, "jackpot": -2.57},
    "short_obs": 100, "tau_lo": 30, "tau_step": 3, "min_seg": 25, "ep_thresh": 150.0,
    "own_margin": 3.0, "cp_lam": 0.1, "size_strength": 20.0,
    # payout size | win is exactly one of two per-type distributions (hidden low/high mode).
    # P(high) = sigmoid(a + b * LLR), LLR = sum over the machine's observed wins of log(p_high(size)/p_low(size)).
    # a, b calibrated by logistic regression on validation targets (5-fold episode CV: 68% mode accuracy).
    "size_mode": {"a": -0.252, "b": 0.094, "modes": {
        "balanced": [[0.49985, 0.34793, 0.13047, 0.02175], [0.21407, 0.54674, 0.20503, 0.03417]],
        "jackpot": [[0.33515, 0.24495, 0.31493, 0.10498], [0.12079, 0.32392, 0.41647, 0.13882]],
        "steady": [[0.75399, 0.19681, 0.04822, 0.00098], [0.45512, 0.43590, 0.10680, 0.00218]]}},
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
    return [(pat >> n) & 1 for n in range(len(PAIRS))]


class Prefix:
    """Prefix sums of (trials, wins) per pattern over a machine's rows, for O(8) segment stats."""

    def __init__(self, rows):
        self.ts = [r[0] for r in rows]
        self.n = [[0] * (1 << len(PAIRS))]
        self.k = [[0] * (1 << len(PAIRS))]
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
        P = 1 << len(PAIRS)
        return [self.n[b][p] - self.n[a][p] for p in range(P)], [self.k[b][p] - self.k[a][p] for p in range(P)]


def seg_ll_l2(n, k, cfg):
    if sum(n) < 5:
        return 0.0
    K = len(PAIRS)
    rows = [([1.0] + [float(v) for v in bits(p)], n[p], k[p]) for p in range(1 << K) if n[p]]
    lam = cfg["cp_lam"]
    return map_fit(rows, [0.0] * (K + 1), [1e-4] + [lam] * K, cfg)[1]


def bma_core(n, k, xq_bits, b0m, cfg, mid=0.0, logpri=None, mids=None, midsd=None):
    """7-model BMA. Model 0 = no active pair; model 1+2j+(0 if s=+1 else 1) = pair j with sign s.
    Returns (posterior weights, predictions, fitted midpoints/intercepts)."""
    b0sd, p_none = cfg["b0sd"], cfg["p_none"]
    if logpri is None:
        logpri = [math.log(p_none)] + [math.log((1 - p_none) / (2 * len(PAIRS)))] * (2 * len(PAIRS))
    evs, preds, fits = [], [], []
    rows = [([1.0], sum(n), sum(k))]
    m, pr = [b0m], [1 / b0sd ** 2]
    b, ll, H = map_fit(rows, m, pr, cfg)
    evs.append(log_evidence(b, ll, H, m, pr) + logpri[0])
    preds.append(sigmoid(b[0])); fits.append(b[0])
    for j in range(len(PAIRS)):
        n2, k2 = [0, 0], [0, 0]
        for p in range(1 << len(PAIRS)):
            v = (p >> j) & 1
            n2[v] += n[p]
            k2[v] += k[p]
        rows = [([1.0, float(v) - 0.5], n2[v], k2[v]) for v in (0, 1) if n2[v]]
        for si, s in enumerate((1, -1)):
            i = 1 + 2 * j + si
            mm = mids[i] if mids is not None else mid
            msd = midsd if mids is not None else MIDSD
            m, pr = [mm, s * MU], [1 / msd ** 2, 1 / CSD ** 2]
            b, ll, H = map_fit(rows, m, pr, cfg) if rows else ([mm, s * MU], 0.0, [[pr[0], 0.0], [0.0, pr[1]]])
            evs.append(log_evidence(b, ll, H, m, pr) + logpri[i])
            preds.append(sigmoid(b[0] + b[1] * (xq_bits[j] - 0.5))); fits.append(b[0])
    top = max(evs)
    wts = [math.exp(e - top) for e in evs]
    tot = sum(wts)
    return [w / tot for w in wts], preds, fits


def flip_map():
    """Model index -> same pair, opposite sign. Model 0 = no active pair; pair k has models 2k+1 (+) and 2k+2 (-)."""
    out = [0]
    for k in range(len(PAIRS)):
        out += [2 * k + 2, 2 * k + 1]
    return out


def bma_win(n, k, xq_bits, b0m, cfg, mid=0.0, pre=None):
    if pre is None:
        w, p, _ = bma_core(n, k, xq_bits, b0m, cfg, mid)
    else:  # post-change fit with the pre-change posterior (pair kept, sign flipped) as prior
        n0, k0 = pre
        w0, _, f0 = bma_core(n0, k0, xq_bits, b0m, cfg, mid)
        K2, FLIP = 2 * len(PAIRS), flip_map()
        base = [cfg["p_none"]] + [(1 - cfg["p_none"]) / K2] * K2
        logpri = [math.log(FL_EPS * base[i] + (1 - FL_EPS) * w0[FLIP[i]]) for i in range(K2 + 1)]
        mids = [None] + [FL_SHRINK * f0[FLIP[i]] + (1 - FL_SHRINK) * mid for i in range(1, K2 + 1)]
        w, p, _ = bma_core(n, k, xq_bits, b0m, cfg, mid, logpri, mids, FL_SD)
    return sum(a * b for a, b in zip(w, p))


# ------------------------------------------------------------------ structure learned from the round's training data
def _binom_ll(k, n):
    if n == 0 or k == 0 or k == n:
        return 0.0
    p = k / n
    return k * math.log(p) + (n - k) * math.log(1 - p)


def learn_structure(train_dir, cfg, min_gain=7.5, min_share=0.10, log=print):
    """Estimate the round-specific structure from its *training* split (inputs and condition labels only):
      * which XOR lag pairs are active: every pair that is the best-fitting pair (likelihood-ratio gain > min_gain,
        i.e. chi2 p < 1e-4) for at least min_share of the stable training machines;
      * p_none: share of stable training machines with no significant pair;
      * prior for active machines, logit = m + c*(XOR - 0.5): m ~ N(a_type + b_type*signal, MIDSD^2),
        |c| ~ N(MU, CSD^2). Spreads use the empirical-Bayes method of moments (observed variance minus the
        mean sampling variance), so per-machine noise does not inflate the prior.
    Sets the module globals PAIRS, MID, MU, CSD, MIDSD and cfg['p_none']; returns a summary dict."""
    global PAIRS, MID, MU, CSD, MIDSD
    train_dir = Path(train_dir)
    queries = read_queries(train_dir / "queries.csv")
    history = read_history(train_dir / "history.csv")
    stable = None
    if (train_dir / "conditions.csv").exists():
        with open(train_dir / "conditions.csv", newline="", encoding="utf-8") as f:
            stable = {r["query_id"] for r in csv.DictReader(f) if r["condition"] == "stable"}
    all_pairs = [(i, j) for i in range(1, 9) for j in range(i + 1, 9)]
    machines = []
    for q in queries:
        if stable is not None and q["query_id"] not in stable:
            continue
        ticks = history.get((q["episode_id"], q["machine_id"]), {})
        if stable is None and sum(1 for t in ticks if t < q["cutoff"]) < 150:
            continue
        rows = []
        for t in sorted(ticks):
            if 8 <= t < q["cutoff"] and all((t - d) in ticks for d in range(1, 9)):
                rows.append(([1 if ticks[t - d] > 0 else 0 for d in range(1, 9)], 1 if ticks[t] > 0 else 0))
        if len(rows) < 60:
            continue
        n_all, k_all = len(rows), sum(y for _, y in rows)
        best = None
        for (i, j) in all_pairs:
            n, k = [0, 0], [0, 0]
            for w, y in rows:
                v = w[i - 1] ^ w[j - 1]
                n[v] += 1
                k[v] += y
            gain = _binom_ll(k[0], n[0]) + _binom_ll(k[1], n[1]) - _binom_ll(k_all, n_all)
            if best is None or gain > best[0]:
                best = (gain, (i, j), n, k)
        machines.append((q, best))
    strong = [(q, b) for q, b in machines if b[0] > min_gain]
    counts = {}
    for _, b in strong:
        counts[b[1]] = counts.get(b[1], 0) + 1
    pairs = sorted([pr for pr, c in counts.items() if c >= min_share * len(strong)], key=lambda pr: -counts[pr])
    if not pairs:
        log("learn_structure: no clear pairs found; keeping defaults")
        return None
    PAIRS = tuple(pairs)
    cfg["p_none"] = max(0.02, 1 - len(strong) / max(1, len(machines)))
    # per-machine estimates for machines whose best pair is one of the selected pairs
    est = {}
    for q, (gain, pr, n, k) in strong:
        if pr not in PAIRS:
            continue
        p0, p1 = (k[0] + 0.5) / (n[0] + 1), (k[1] + 0.5) / (n[1] + 1)
        v0, v1 = 1 / ((n[0] + 1) * p0 * (1 - p0)), 1 / ((n[1] + 1) * p1 * (1 - p1))
        l0, l1 = math.log(p0 / (1 - p0)), math.log(p1 / (1 - p1))
        est.setdefault(q["machine_type"], []).append(((l0 + l1) / 2, abs(l1 - l0), q["signal"], (v0 + v1) / 4, v0 + v1))
    allc = [e for es in est.values() for e in es]
    mean = lambda xs: sum(xs) / len(xs)
    MU = mean([e[1] for e in allc])
    var_c = mean([(e[1] - MU) ** 2 for e in allc]) - mean([e[4] for e in allc])
    CSD = math.sqrt(max(var_c, 0.3 ** 2))
    MID, resid, svar = {}, [], []
    for t, es in est.items():   # least squares: midpoint = a + b * signal, per machine type
        xs, ys = [e[2] for e in es], [e[0] for e in es]
        mx, my = mean(xs), mean(ys)
        sxx = sum((x - mx) ** 2 for x in xs)
        b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx if sxx else 0.0
        a = my - b * mx
        MID[t] = (a, b)
        resid += [y - a - b * x for x, y in zip(xs, ys)]
        svar += [e[3] for e in es]
    var_m = mean([r * r for r in resid]) - mean(svar)
    MIDSD = math.sqrt(max(var_m, 0.2 ** 2))
    summary = {"pairs": PAIRS, "pair_counts": {str(k): v for k, v in sorted(counts.items(), key=lambda kv: -kv[1])[:6]},
               "machines": len(machines), "strong": len(strong), "p_none": round(cfg["p_none"], 3),
               "MU": round(MU, 3), "CSD": round(CSD, 3), "MIDSD": round(MIDSD, 3),
               "MID": {t: (round(a, 3), round(b, 3)) for t, (a, b) in MID.items()}}
    log(f"learn_structure: {summary}")
    return summary


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


MODE_FEATURES = (["llr_all", "llr_last1", "llr_last5", "llr_last20", "llr_last40"]
                 + [f"l{k}_is{p}" for k in range(1, 9) for p in (0, 1, 2, 5, 20)]
                 + ["wins_last8", "wins_last20", "n_obs", "signal", "type_steady", "type_balanced", "type_jackpot"])


def _llr(ticks, start, cutoff, lo, hi):
    s = 0.0
    for t in sorted(ticks):
        if start <= t < cutoff and ticks[t] > 0:
            j = SIZES.index(ticks[t])
            s += math.log(hi[j] / lo[j])
    return s


def mode_features(ticks, cutoff, mtype, signal, modes):
    lo, hi = modes[mtype]
    lo = [v / sum(lo) for v in lo]
    hi = [v / sum(hi) for v in hi]
    f = [_llr(ticks, 0, cutoff, lo, hi)] + [_llr(ticks, cutoff - w, cutoff, lo, hi) for w in (1, 5, 20, 40)]
    for k in range(1, 9):
        v = ticks.get(cutoff - k)
        f += [1.0 if v == p else 0.0 for p in (0, 1, 2, 5, 20)]
    f.append(float(sum(1 for t in range(cutoff - 8, cutoff) if ticks.get(t, 0) > 0)))
    f.append(float(sum(1 for t in range(cutoff - 20, cutoff) if ticks.get(t, 0) > 0)))
    f.append(float(sum(1 for t in ticks if t < cutoff)))
    f.append(float(signal))
    f += [1.0 if mtype == t else 0.0 for t in ("steady", "balanced", "jackpot")]
    return f


def mode_lgb_prob(x, lgbm):
    """Evaluate exported LightGBM binary trees (numeric '<=' splits) with the standard library."""
    raw = 0.0
    for node in lgbm["trees"]:
        while isinstance(node, list):
            node = node[2] if x[node[0]] <= node[1] else node[3]
        raw += node
    return 1.0 / (1.0 + math.exp(-raw))


def size_mode_distribution(ticks, cutoff, mtype, sm, signal=0, lgbm=None):
    """Calibrated mixture of the two hidden size modes; None if the type is unknown."""
    if mtype not in sm["modes"]:
        return None
    lo, hi = sm["modes"][mtype]
    lo = [v / sum(lo) for v in lo]
    hi = [v / sum(hi) for v in hi]
    llr = 0.0
    for t, v in ticks.items():
        if t < cutoff and v > 0:
            j = SIZES.index(v)
            llr += math.log(hi[j] / lo[j])
    if lgbm is not None:
        ph = mode_lgb_prob(mode_features(ticks, cutoff, mtype, signal, sm["modes"]), lgbm)
    else:
        ph = sigmoid(sm["a"] + sm["b"] * llr)
    return [(1 - ph) * l + ph * h for l, h in zip(lo, hi)]


def predict(queries, history, model, lgbm=None):
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
                pw = bma_win(n, k, bits(pq), b0m, cfg, MID[q['machine_type']][0] + MID[q['machine_type']][1] * q['signal'],
                             pre.seg(0, start) if (FLIP_ON and start > 0) else None)
            sz = size_mode_distribution(ticks, cutoff, q["machine_type"], cfg["size_mode"], q["signal"], lgbm) if "size_mode" in cfg else None
            if sz is None:
                sz = size_distribution(ticks, cutoff, priors.get(q["machine_type"], fallback), cfg["size_strength"])
            probs = [1.0 - pw] + [pw * s for s in sz]
            tot = sum(probs)
            out[q["query_id"]] = [p / tot for p in probs]
    return out
