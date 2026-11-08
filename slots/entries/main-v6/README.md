# House Edge — main round entry (xorcast v6)

- Team ID: penberthy-pushers
- Round: main
- Season ID: 5fffdc4d0c700bbf61c3f911
- Environment: `python` image (see `reproduce.json`). **Python standard library only**, no extra packages.
- Tested with Python 3.12.3 on Linux (WSL2), CPU only. Deterministic: no randomness, no seeds needed.

## Recreate the forecast

Run from this folder:

```sh
python predict.py --data /data --out /output/submission.csv
```

- Loads `model.json` (config + per-type payout-size priors) and `mode_lgb.json` (exported size-mode trees) and
  predicts every query in `/data/test/`. **Prediction needs no third-party packages**: the LightGBM trees are
  evaluated by plain Python in `xorcast2.mode_lgb_prob`.
- Runtime: ~15 s on one CPU core, <100 MB RAM.
- Local scoring: `python predict.py --data path/to/public --out val.csv --split validation`

## Model (`xorcast2.py`)

Structure found by per-machine likelihood-ratio tests on train/validation histories:
P(win_t) = sigmoid(b0 + c * XOR(win[t-i], win[t-j])) with one active lag pair per machine from
{(2,5), (1,3), (4,7)} (|c| ≈ 4.85, sign machine-specific). Drift and shared-change episodes contain a regime change.

1. **Regime detection.** For each machine and candidate tick tau (every 3 ticks, ≥25 rows per side), compute the
   log-likelihood gain of fitting the 3-XOR logistic model separately before and after tau. Sum the gains over
   the episode's 8 machines. If the best sum exceeds 150, the episode changed: each machine uses the episode tau,
   unless its own best tau gains ≥3 more log-likelihood. Short-history episodes (<100 observations) skip this step.
2. **Win probability.** Bayesian model averaging (Laplace evidence) over 7 logistic models (no pair, or one pair
   with coefficient prior N(±4.85, 1); intercept prior N(type mean, 2²)), fitted on post-change ticks.
3. **Payout size given win.** Each type has exactly two size distributions (a hidden low/high mode). P(high) comes
   from a LightGBM classifier with 200 trees and 7 leaves, trained on validation targets. Its inputs are the
   machine's size log-likelihood ratios (all history and the last 1/5/20/40 ticks), one-hot payouts at lags 1-8,
   recent win counts, observation count, signal and type. The trees are exported to `mode_lgb.json` and evaluated
   in pure Python, matching LightGBM to 0.0. The result is the mixture of the two size distributions.
   Fallback when `mode_lgb.json` is absent: sigmoid(-0.252 + 0.094 · LLR).

All fits use sufficient statistics (counts per XOR pattern) and damped Newton's method. Settings are in `model.json["config"]`.

## Training

```sh
python train.py --data path/to/public          # model.json (stdlib only)
python train_mode.py --data path/to/public     # mode_lgb.json (needs numpy==2.5.4, lightgbm==4.7.0; Python 3.12)
```

`train_mode.py` is deterministic (n_jobs=1, random_state=0, deterministic=True).

`train.py` writes `model.json`. The structural constants (lag pairs, |c| prior, type intercept means) come from
train/validation history analysis; the hyperparameters were checked on validation and on train-label Brier.

v6 adds a regime-change prior: after a detected change the machine almost always keeps its active XOR pair and flips its sign (train: 93.7%), so the pre-change posterior (pair kept, sign flipped, midpoint 0.87x carried over, sd 0.75) is the prior for the post-change fit; all three constants from train histories only. Rolling next-tick test on changed machines: Brier -28%/-30%/-23% (train/val/test histories) 3 ticks after a change, fading to 0 by ~45 ticks; placebo (sign kept) is worse everywhere. v6 validation: 56.32 in-sample (+0.02 vs v5 at query time, where changes are 53-104 ticks old).

v5 added a tied-coefficient prior for active machines: logit = m + c*(XOR - 0.5) with m ~ N(mid[type] + slope[type]*signal, 0.6^2), c ~ N(+-4.85, 0.7^2) (constants fitted on train-split histories only; see MID in xorcast2.py). v5 validation: 56.30 in-sample, ~55.4 out-of-fold (+0.34 over v4 with the same size model).

v4 validation: 55.95 / 60 in-sample. The mode trees were trained on validation, so this is optimistic; the honest
5-fold out-of-fold estimate is ~55.05. Without the trees (v3): 54.84. v2: 54.58.
