"""v2 research model: regime detection (episode change point) + structured one-active-pair Bayesian model averaging.
Evaluated against v1 (adaptive half-life 3-XOR) on validation points and train Brier."""
import sys, json, time
import numpy as np
import pandas as pd
from hm import load, contest_points, brier_labels, PAY, PCOLS

P3 = [(2, 5), (1, 3), (4, 7)]
TYPE_B0 = {"steady": 0.87, "balanced": -0.67, "jackpot": -2.57}


def sig(z):
    return 1 / (1 + np.exp(-np.clip(z, -30, 30)))


def xrows(a, cutoff, start=8):
    ts = np.arange(max(start, 8), cutoff)
    M = np.stack([a[ts - k] for k in range(1, 9)], 1)
    ok = ~np.isnan(M).any(1) & ~np.isnan(a[ts])
    ts, M = ts[ok], M[ok]
    W = (M > 0).astype(float)
    X = np.column_stack([np.abs(W[:, i - 1] - W[:, j - 1]) for i, j in P3])
    return ts, X, (a[ts] > 0).astype(float)


def xq_of(a, cutoff):
    W = (np.array([a[cutoff - k] for k in range(1, 9)]) > 0).astype(float)
    return np.array([abs(W[i - 1] - W[j - 1]) for i, j in P3])


def map_fit(X, y, w, mu, prec, iters=60):
    b = np.zeros_like(mu)
    for _ in range(iters):
        p = sig(X @ b)
        g = X.T @ (w * (p - y)) + prec * (b - mu)
        H = (X * (w * p * (1 - p))[:, None]).T @ X + np.diag(prec)
        st = np.linalg.solve(H, g)
        mx = np.abs(st).max()
        if mx > 2.0:
            st *= 2.0 / mx  # damped Newton: avoid divergence from far starts
        b -= st
        if np.abs(st).max() < 1e-8:
            break
    p = np.clip(sig(X @ b), 1e-12, 1 - 1e-12)
    H = (X * (w * p * (1 - p))[:, None]).T @ X + np.diag(prec)
    ll = np.sum(w * (y * np.log(p) + (1 - y) * np.log(1 - p)))
    return b, ll, H


def log_evidence(b, ll, H, mu, prec):
    d = b - mu
    return ll + 0.5 * np.sum(np.log(prec)) - 0.5 * np.sum(prec * d * d) - 0.5 * np.linalg.slogdet(H)[1]


def bma(X, y, w, xq, b0_mean, cfg):
    """Models: none, or exactly one XOR pair active with coefficient ~ N(+-mu, sd^2). Returns (p, log evidence)."""
    mu, sd, b0sd, p_none = cfg["mu"], cfg["sd"], cfg["b0sd"], cfg["p_none"]
    evs, preds = [], []
    one = np.ones((len(y), 1))
    b, ll, H = map_fit(one, y, w, np.array([b0_mean]), np.array([1 / b0sd**2]))
    evs.append(log_evidence(b, ll, H, np.array([b0_mean]), np.array([1 / b0sd**2])) + np.log(p_none))
    preds.append(sig(b[0]))
    for k in range(3):
        Xk = np.column_stack([one, X[:, k]])
        for s in (1, -1):
            m = np.array([b0_mean, s * mu]); pr = np.array([1 / b0sd**2, 1 / sd**2])
            b, ll, H = map_fit(Xk, y, w, m, pr)
            evs.append(log_evidence(b, ll, H, m, pr) + np.log((1 - p_none) / 6))
            preds.append(sig(b[0] + b[1] * xq[k]))
    evs = np.array(evs)
    wts = np.exp(evs - evs.max()); wts /= wts.sum()
    lse = evs.max() + np.log(np.exp(evs - evs.max()).sum())
    return float(wts @ np.array(preds)), lse


def l2fit(X, y, w, xq, lam=0.03):
    Xd = np.column_stack([np.ones(len(y)), X])
    prec = np.array([1e-4, lam, lam, lam])
    b, ll, _ = map_fit(Xd, y, w, np.zeros(4), prec)
    return float(sig(b[0] + b[1:] @ xq)), ll


def seg_ll(X, y, b0m, cfg):
    """Fast segment log-likelihood (3-XOR L2 fit) for change-point scoring."""
    if len(y) < 5:
        return 0.0
    Xd = np.column_stack([np.ones(len(y)), X])
    return map_fit(Xd, y, np.ones(len(y)), np.zeros(4), np.array([1e-4, .1, .1, .1]))[1]


def cp_profile(ts, X, y, taus, b0m, cfg):
    """log evidence(two segments split at tau) - log evidence(one segment), for each tau."""
    full = seg_ll(X, y, b0m, cfg)
    out = []
    for tau in taus:
        k = np.searchsorted(ts, tau)
        if k < cfg["min_seg"] or len(ts) - k < cfg["min_seg"]:
            out.append(-np.inf); continue
        out.append(seg_ll(X[:k], y[:k], b0m, cfg) + seg_ll(X[k:], y[k:], b0m, cfg) - full)
    return np.array(out)


def size_dist(a, cutoff, prior, k=20, start=0):
    v = a[start:cutoff]; v = v[~np.isnan(v) & (v > 0)]
    cnt = np.array([(v == s).sum() for s in (1, 2, 5, 20)], float)
    return (cnt + k * prior) / (cnt.sum() + k)


SIZE_MODES = {  # exact per-type size distributions given a win (low mode, high mode), from validation targets
    "balanced": (np.array([0.49985, 0.34793, 0.13047, 0.02175]), np.array([0.21407, 0.54674, 0.20503, 0.03417])),
    "jackpot": (np.array([0.33515, 0.24495, 0.31493, 0.10498]), np.array([0.12079, 0.32392, 0.41647, 0.13882])),
    "steady": (np.array([0.75399, 0.19681, 0.04822, 0.00098]), np.array([0.45512, 0.43590, 0.10680, 0.00218])),
}
SIZE_IDX = {1: 0, 2: 1, 5: 2, 20: 3}


def size_hmm(a, cutoff, mtype, eps, p_high=0.45, start=0):
    """Forward filter over a 2-state hidden size mode with per-tick switch prob eps."""
    lo, hi = SIZE_MODES[mtype]
    lo, hi = lo / lo.sum(), hi / hi.sum()
    f = np.array([1 - p_high, p_high])
    for t in range(start, cutoff):
        if eps:
            f = f * (1 - eps) + f[::-1] * eps
        v = a[t]
        if not np.isnan(v) and v > 0:
            j = SIZE_IDX[int(v)]
            f = f * np.array([lo[j], hi[j]])
            f /= f.sum()
    if eps:
        f = f * (1 - eps) + f[::-1] * eps
    return f[0] * lo + f[1] * hi


def type_size_prior():
    q, s = load("train")
    out = {}
    for t, g in q.groupby("machine_type"):
        v = np.concatenate([s[m][~np.isnan(s[m])] for m in g.machine_id]); v = v[v > 0]
        out[t] = np.array([(v == x).mean() for x in (1, 2, 5, 20)])
    return out


SIZE_PRIOR = type_size_prior()


def predict(q, series, cfg):
    P = np.zeros((len(q), 5))
    diag = []
    for ep, g in q.groupby("episode_id", sort=False):
        cutoff = int(g.cutoff.iloc[0])
        data = {}
        for r in g.itertuples():
            a = series[r.machine_id]
            data[r.Index] = (r, a, *xrows(a, cutoff))
        n_obs = np.mean([np.sum(~np.isnan(d[1][:cutoff])) for d in data.values()])
        short = n_obs < cfg["short_obs"]
        taus = np.arange(cfg["tau_lo"], cutoff - cfg["min_seg"] + 1, cfg["tau_step"])
        prof = {}
        if not short and cfg["regime"]:
            for i, (r, a, ts, X, y) in data.items():
                prof[i] = cp_profile(ts, X, y, taus, TYPE_B0[r.machine_type], cfg)
            S = np.sum([np.maximum(p, -50) for p in prof.values()], 0)
            ep_tau = taus[int(np.argmax(S))]
            changed = S.max() > cfg["ep_thresh"]
        else:
            changed, ep_tau, S = False, None, np.array([0.0])
        for i, (r, a, ts, X, y) in data.items():
            xq = xq_of(a, cutoff)
            b0m = TYPE_B0[r.machine_type]
            start = 0
            if changed:
                own = taus[int(np.argmax(prof[i]))]
                # per-machine tau vs episode tau: compare their evidence gains
                gi_own, gi_ep = prof[i].max(), prof[i][list(taus).index(ep_tau)]
                tau = ep_tau if (cfg["tau_mode"] == "episode" or
                                 (cfg["tau_mode"] == "auto" and gi_own - gi_ep < cfg["own_margin"])) else own
                start = tau
                m = ts >= tau
                ts_, X_, y_ = ts[m], X[m], y[m]
            else:
                ts_, X_, y_ = ts, X, y
            w = np.ones(len(y_))
            if cfg["post_hl"] and changed:
                w = 2.0 ** (-(cutoff - 1 - ts_) / cfg["post_hl"])
            if cfg["win_model"] == "bma":
                pw, _ = bma(X_, y_, w, xq, b0m, cfg)
            else:
                pw, _ = l2fit(X_, y_, w, xq)
            if cfg.get("size_hmm") is not None:
                sz = size_hmm(a, cutoff, r.machine_type, cfg["size_hmm"], start=start if cfg["size_post"] else 0)
            else:
                sz = size_dist(a, cutoff, SIZE_PRIOR[r.machine_type], cfg["size_k"], start if cfg["size_post"] else 0)
            P[i] = np.concatenate([[1 - pw], pw * sz])
            diag.append((r.query_id, ep, short, changed, ep_tau, float(S.max())))
    return P, pd.DataFrame(diag, columns=["query_id", "ep", "short", "changed", "ep_tau", "S"])


BASE = {"mu": 4.85, "sd": 1.0, "b0sd": 2.0, "p_none": 0.07, "short_obs": 100, "regime": True,
        "tau_lo": 30, "tau_step": 3, "min_seg": 25, "ep_thresh": 150.0, "tau_mode": "auto", "own_margin": 3.0,
        "post_hl": None, "win_model": "bma", "size_k": 20, "size_post": False}


def run(name, cfg, log=None):
    t0 = time.time()
    qv, sv = load("validation")
    Pv, dv = predict(qv, sv, cfg)
    pts, R = contest_points(qv, Pv)
    qt, st = load("train")
    Pt, dt = predict(qt, st, cfg)
    br = brier_labels(qt, Pt)
    # regime detector accuracy on train (labels known): changed should == condition in {drift, shared_change}
    dt = dt.merge(qt[["query_id", "condition"]], on="query_id")
    det = dt[~dt.short].assign(truth=lambda d: d.condition.isin(["drift", "shared_change"]))
    acc = (det.changed == det.truth).mean()
    row = {"name": name, "val_points": round(pts, 2), **{f"R_{k}": round(v, 3) for k, v in R.items()},
           "train_brier": round(br.mean(), 4), "train_regime_acc": round(acc, 3), "secs": round(time.time() - t0, 1)}
    print(json.dumps(row), flush=True)
    if log is not None:
        log.append(row)
    return Pv, qv


if __name__ == "__main__":
    T = {
        "V0 bma+regime(auto tau)": BASE,
        "V1 l2 win model + regime": {**BASE, "win_model": "l2"},
        "V2 bma, no regime": {**BASE, "regime": False},
        "V3 bma+regime, episode tau": {**BASE, "tau_mode": "episode"},
        "V4 bma+regime, own tau": {**BASE, "tau_mode": "own"},
        "V5 V0 + size from post-change": {**BASE, "size_post": True},
        "V6 V0 + post-change hl=40": {**BASE, "post_hl": 40},
    }
    sel = sys.argv[1:]
    for n, c in T.items():
        if not sel or n.split()[0] in sel:
            run(n, c)
