# House Edge — main round entry (xorcast v3)

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

- Loads `model.json` (config + per-type payout-size priors) and predicts every query in `/data/test/`.
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
3. **Payout size given win.** Each type has exactly two size distributions (hidden low/high mode). P(high) =
   sigmoid(-0.252 + 0.094 · LLR), where LLR is the sum of log(p_high/p_low) over the machine's observed win sizes.
   The result is the mixture of the two distributions. Both coefficients were calibrated by logistic regression on
   validation targets (5-fold episode CV: 68% mode accuracy).

All fits use sufficient statistics (counts per XOR pattern) and damped Newton's method. Settings are in `model.json["config"]`.

## Training

```sh
python train.py --data path/to/public
```

This writes `model.json`. The structural constants (lag pairs, |c| prior, type intercept means) come from
train/validation history analysis; the hyperparameters were checked on validation and on train-label Brier.

Local validation: 54.84 / 60 technical points (loss ratios: stable 0.078, short_history 0.122, drift 0.090, shared_change 0.086). v2 without the size-mode step: 54.58.
