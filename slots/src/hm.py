"""House Edge modelling library: data loading, per-machine regression models, scoring.
numpy + pandas only (pandas only for loading/scoring), so the final predictor can be stdlib/numpy."""
from pathlib import Path
import itertools
import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parents[1] / "kit" / "public"
PAY = np.array([0, 1, 2, 5, 20])
PCOLS = ["p_0", "p_1", "p_2", "p_5", "p_20"]
TOP_PAIRS = [(2, 5), (1, 3), (4, 7), (4, 5), (2, 6), (3, 8)]
ALL_PAIRS = list(itertools.combinations(range(1, 9), 2))


# ---------------------------------------------------------------- data
def load(split):
    q = pd.read_csv(DATA / split / "queries.csv")
    h = pd.read_csv(DATA / split / "history.csv")
    for name in ("conditions", "labels"):
        f = DATA / split / f"{name}.csv"
        if f.exists():
            q = q.merge(pd.read_csv(f), on="query_id")
    f = DATA / split / "targets.csv"
    if f.exists():
        q = q.merge(pd.read_csv(f)[["query_id"] + PCOLS], on="query_id")
    W = h.pivot(index="machine_id", columns="tick", values="payout").reindex(columns=range(int(q.cutoff.max()) + 1))
    series = {m: W.loc[m].to_numpy() for m in q.machine_id}
    return q.reset_index(drop=True), series


# ---------------------------------------------------------------- features
def lag_wins(a, ts, L=8):
    """Win indicators at lags 1..L for target ticks ts -> (n, L); NaN-safe mask returned too."""
    M = np.stack([a[ts - k] for k in range(1, L + 1)], 1)
    ok = ~np.isnan(M).any(1)
    return (M > 0).astype(float), ok


def design(Wn, pairs, singles=None, xor=False):
    S = Wn if singles is None else Wn[:, [k - 1 for k in singles]]
    cols = [np.ones(len(Wn))[:, None]]
    if S.shape[1]:
        cols.append(S)
    if pairs:
        f = (lambda a, b: np.abs(a - b)) if xor else (lambda a, b: a * b)
        cols.append(np.stack([f(Wn[:, i - 1], Wn[:, j - 1]) for i, j in pairs], 1))
    return np.hstack(cols)


def machine_rows(a, cutoff, L=8):
    """Training rows from a machine's own history (strictly before cutoff)."""
    ts = np.arange(L, cutoff)
    Wn, ok = lag_wins(a, ts, L)
    ok &= ~np.isnan(a[ts])
    return ts[ok], Wn[ok], a[ts[ok]]


# ---------------------------------------------------------------- weighted L2 logistic regression (Newton)
def fit_logreg(X, y, w, lam, prior_mean=None, iters=25):
    """Minimise sum w*logloss + lam/2 ||beta - prior_mean||^2 (intercept unpenalised)."""
    d = X.shape[1]
    mu = np.zeros(d) if prior_mean is None else prior_mean
    beta = mu.copy()
    P = np.full(d, lam); P[0] = 1e-4
    for _ in range(iters):
        z = np.clip(X @ beta, -30, 30)
        p = 1 / (1 + np.exp(-z))
        g = X.T @ (w * (p - y)) + P * (beta - mu)
        H = (X * (w * p * (1 - p))[:, None]).T @ X + np.diag(P)
        step = np.linalg.solve(H, g)
        beta -= step
        if np.abs(step).max() < 1e-7:
            break
    return beta


def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -30, 30)))


# ---------------------------------------------------------------- scoring
def contest_points(q, P, ref=None, max_points=60):
    T = q[PCOLS].to_numpy()
    d = q.assign(sse=((P - T) ** 2).sum(1))
    E = d.groupby(["condition", "episode_id"]).sse.mean().groupby("condition").mean()
    if ref is None:
        ref = pd.read_csv(DATA / "validation/reference.csv").set_index("query_id").loc[q.query_id, PCOLS].to_numpy()
    dr = q.assign(sse=((ref - T) ** 2).sum(1))
    B = dr.groupby(["condition", "episode_id"]).sse.mean().groupby("condition").mean()
    R = E / B
    return max_points / (1 + R.mean()), R


def brier_labels(q, P):
    """Multiclass Brier vs observed outcome (train split check), averaged per condition then overall."""
    Y = (q.outcome.to_numpy()[:, None] == PAY[None, :]).astype(float)
    d = q.assign(b=((P - Y) ** 2).sum(1))
    return d.groupby("condition").b.mean()
