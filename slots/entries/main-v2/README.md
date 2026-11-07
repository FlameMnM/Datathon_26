# House Edge — main round entry (xorcast v2)

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
3. **Payout size given win.** The machine's win-size counts, shrunk toward per-type priors from `public/train`
   (pseudo-count 20).

All fits use sufficient statistics (counts per XOR pattern) and damped Newton's method. Settings are in `model.json["config"]`.

## Training

```sh
python train.py --data path/to/public
```

This writes `model.json`. The structural constants (lag pairs, |c| prior, type intercept means) come from
train/validation history analysis; the hyperparameters were checked on validation and on train-label Brier.

Local validation: 54.58 / 60 technical points (loss ratios: stable 0.079, short_history 0.132,
drift 0.094, shared_change 0.092).
