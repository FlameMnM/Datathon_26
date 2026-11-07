"""Regression-model trials for House Edge. Scores each config on validation (true odds) and train (Brier vs outcomes)."""
import sys, time, json
import numpy as np
import pandas as pd
from hm import (load, machine_rows, design, lag_wins, fit_logreg, sigmoid, contest_points, brier_labels,
                TOP_PAIRS, ALL_PAIRS, PAY, PCOLS)

SPLITS = {s: load(s) for s in ("validation", "train")}


def type_size_prior(q, series):
    out = {}
    for t, g in q.groupby("machine_type"):
        vals = np.concatenate([series[m][~np.isnan(series[m])] for m in g.machine_id])
        wins = vals[vals > 0]
        out[t] = np.array([(wins == v).mean() for v in (1, 2, 5, 20)])
    return out


def recency_w(ts, ref_tick, hl):
    return np.ones(len(ts)) if hl is None else 2.0 ** (-(ref_tick - 1 - ts) / hl)


def win_prob(a, cutoff, cfg):
    pairs = cfg["pairs"]
    ts, Wn, y = machine_rows(a, cutoff)
    sg, xr = cfg.get("singles"), cfg.get("xor", False)
    X, yb = design(Wn, pairs, sg, xr), (y > 0).astype(float)
    xq = design(lag_wins(a, np.array([cutoff]))[0], pairs, sg, xr)
    lam = cfg["lam"]
    hl = cfg["hl"]
    if cfg.get("adaptive"):
        # choose half-life per machine by predictive log-loss on its own last V ticks
        V = cfg.get("V", 30)
        cands = cfg["adaptive"]
        tr, va = ts < cutoff - V, ts >= cutoff - V
        scores = []
        for c in cands:
            if tr.sum() < 20 or va.sum() < 5:
                scores.append(0.0); continue
            b = fit_logreg(X[tr], yb[tr], recency_w(ts[tr], cutoff - V, c), lam)
            p = np.clip(sigmoid(X[va] @ b), 1e-4, 1 - 1e-4)
            scores.append(np.mean(yb[va] * np.log(p) + (1 - yb[va]) * np.log(1 - p)))
        s = np.array(scores)
        wts = np.exp((s - s.max()) * cfg.get("temp", 30))
        wts /= wts.sum()
        pw = 0.0
        for c, wt in zip(cands, wts):
            b = fit_logreg(X, yb, recency_w(ts, cutoff, c), lam)
            pw += wt * sigmoid(xq @ b)[0]
        return pw, ts, y
    b = fit_logreg(X, yb, recency_w(ts, cutoff, hl), lam)
    return sigmoid(xq @ b)[0], ts, y


def size_dist(a, cutoff, prior, cfg):
    k, shl = cfg.get("size_k", 20), cfg.get("size_hl")
    ticks = np.arange(cutoff)
    m = ~np.isnan(a[:cutoff]) & (a[:cutoff] > 0)
    w = recency_w(ticks[m], cutoff, shl)
    vals = a[:cutoff][m]
    cnt = np.array([w[vals == v].sum() for v in (1, 2, 5, 20)])
    return (cnt + k * prior) / (cnt.sum() + k)


def predict(q, series, cfg):
    prior = type_size_prior(q, series)
    P = np.zeros((len(q), 5))
    for i, r in enumerate(q.itertuples()):
        a = series[r.machine_id]
        pw, _, _ = win_prob(a, r.cutoff, cfg)
        if cfg.get("shrink_win"):  # blend with machine's raw recent win rate
            pw = (1 - cfg["shrink_win"]) * pw + cfg["shrink_win"] * np.nanmean(a[:r.cutoff] > 0)
        eps = cfg.get("eps", 0.0)
        pw = eps + (1 - 2 * eps) * pw
        P[i] = np.concatenate([[1 - pw], pw * size_dist(a, r.cutoff, prior[r.machine_type], cfg)])
    return P


def run(name, cfg, log):
    t0 = time.time()
    qv, sv = SPLITS["validation"]
    pts, R = contest_points(qv, predict(qv, sv, cfg))
    qt, st = SPLITS["train"]
    br = brier_labels(qt, predict(qt, st, cfg))
    row = {"name": name, "val_points": round(pts, 2), **{f"R_{k}": round(v, 3) for k, v in R.items()},
           "train_brier": round(br.mean(), 4), "secs": round(time.time() - t0, 1)}
    log.append(row)
    print(json.dumps(row), flush=True)


if __name__ == "__main__":
    base = {"pairs": TOP_PAIRS, "lam": 1 / 3, "hl": None}
    trials = [
        ("A0 previous probe (8 lags+6 pairs, no recency)", base),
        ("A1 all 28 pairs", {**base, "pairs": ALL_PAIRS}),
        ("A2 all 28 pairs, lam=1", {**base, "pairs": ALL_PAIRS, "lam": 1.0}),
        ("B1 recency hl=80", {**base, "hl": 80}),
        ("B2 recency hl=40", {**base, "hl": 40}),
        ("B3 recency hl=20", {**base, "hl": 20}),
        ("C1 adaptive hl per machine {inf,80,40,20}", {**base, "adaptive": [None, 80, 40, 20]}),
        ("C2 adaptive + size recency hl=60", {**base, "adaptive": [None, 80, 40, 20], "size_hl": 60}),
        ("C3 adaptive + lower lam (0.1)", {**base, "adaptive": [None, 80, 40, 20], "lam": 0.1}),
    ]
    sel = sys.argv[1:] or None
    log = []
    for n, c in trials:
        if sel is None or n.split()[0] in sel:
            run(n, c, log)
    pd.DataFrame(log).to_csv(sys.stdout if False else "trials_log.csv", mode="a", index=False,
                             header=not __import__("os").path.exists("trials_log.csv"))
