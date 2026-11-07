# House Edge — main round entry (xorcast v1)

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

- Loads the saved model `model.json` and predicts every query in `/data/test/`.
- Runtime: ~15 s on one CPU core, <100 MB RAM.
- Local scoring: `python predict.py --data path/to/public --out val.csv --split validation`

## Model

`xorcast.py`, for each query machine, using only that machine's own history before the cutoff:

1. **Win probability**: weighted L2 logistic regression with an intercept and three features
   `XOR(win[t-i], win[t-j])` for lag pairs (2,5), (1,3), (4,7). Fitted with Newton's method.
   Four recency half-lives (none, 40, 20, 10 ticks) are each scored by log-loss on the machine's last
   30 observed ticks (fitted on earlier ticks), then refitted on all ticks and blended with softmax
   weights (temperature 100).
2. **Payout size given a win**: the machine's own counts of 1/2/5/20 wins, shrunk toward a per-type
   prior (pseudo-count 20).
3. Output `[1-p_win, p_win * size_dist]`.

All settings are in `model.json["config"]`.

## Training

```sh
python train.py --data path/to/public
```

This writes `model.json`. The only learned parameters are the per-type size priors, taken from
`public/train` histories. The lag pairs were chosen from training-history analysis; the
hyperparameters were chosen on validation and cross-checked against multiclass Brier on train labels.

Local validation: 51.16 / 60 technical points (loss ratios: stable 0.122, short_history 0.237,
drift 0.154, shared_change 0.179).
