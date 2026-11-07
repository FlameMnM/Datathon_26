# House Edge — working notes

Kit: `slots/kit/` (unzipped from `data/raw/main-participants-60-40.zip`). Run kit scripts from `slots/kit/`.

## Deadlines (Central)
- Main submissions close **Sat Nov 7, 9 PM**
- Round two (adaptation) kit opens Sat 9 PM, closes **Sun Nov 8, 11 AM**

## Conflicting info to confirm with organizers
- PDF guide: main 80 / round two 20, uploads via "House Edge submission page".
- Kit RULES.md + metadata.json + checker: main **60** / adaptation **40**, uploads via **Google Form**. Kit is newer → assume 60/40.
- Runtime: `"image": "python"` = Python 3.13 stdlib. RULES.md: extra packages allowed if pinned in root `requirements.txt`. Safest: final predict.py uses stdlib (or numpy only).

## Data facts
- Train / val / test: 96 / 96 / 192 episodes × 8 machines. 4 conditions equally weighted (24/24/48 episodes each).
- Cutoff 192–256, constant within an episode. Long histories = every tick < cutoff. short_history = 64 obs (40 recent consecutive + 24 older sampled). Last 8 ticks always observed.
- Types: steady (P0≈.42, mostly 1×), balanced (P0≈.59), jackpot (P0≈.78, big 5×/20×). signal +1 → more wins, -1 → fewer (small effect).
- **True probabilities are very sharp**: median max-prob .73; 25% of queries > .965. Outcomes are nearly determined by recent history.
- Win/loss at a lag matters; payout *size* at a lag matters little.
- **Lag 7** is the strongest single lag (best lag for ~36% of machines, mostly positive effect).
- **Pair interactions** (XOR-like) dominate: (2,5), (1,3), (4,7), then (4,5), (2,6), (3,8). Machine-specific signs.
- Cross-machine effect is weak (corr .03 vs self .14) — but shared_change is about joint regime changes, not direct influence.
- Static priors cap out ≈30 pts (per-type×signal oracle mean = 29.97) — must use sequence structure.

## Scores on validation (points /60, ratios lower=better)
| Model | pts | stable | short | drift | shared |
|---|---|---|---|---|---|
| pooled reference | 30.00 | 1 | 1 | 1 | 1 |
| pooled_recent (starter) | 30.83 | | | | |
| per-machine logistic, 8 lags + 6 pairs, C=3 | 32.14 | **.138** | .691 | 1.468 | 1.171 |

Takeaway: per-machine pair model nails stable machines; drift/shared_change need recency weighting / change detection.

## Regression trials (Sat 3:40 PM) — code: src/hm.py, src/trials.py, src/sweep*.py; log: src/trials_log.csv
Model = per-machine weighted L2 logistic regression for P(win) (custom numpy Newton solver) × payout-size dist (machine counts shrunk to type prior, k=20).
Second check: train-split multiclass Brier vs real outcomes (lower = better) — agreed with validation every time.

| Trial | val pts | drift | shared | short | stable | train Brier |
|---|---|---|---|---|---|---|
| A0 8 lags + 6 AND-pairs, no recency | 32.14 | 1.469 | 1.171 | .689 | .138 | .553 |
| B2 + recency half-life 40 | 38.00 | .732 | .556 | .729 | .297 | .474 |
| C3 + per-machine adaptive half-life, lam .1 | 41.76 | .521 | .418 | .658 | .151 | .440 |
| D9 only 3 pairs (2,5),(1,3),(4,7) | 43.29 | .419 | .337 | .642 | .147 | .421 |
| **E3 only 3 XOR features (no single lags)** | 50.49 | .209 | .198 | .241 | .105 | .382 |
| **F7 E3 + half-life cands {inf,40,20,10}** | **51.15** | .154 | .179 | .237 | .122 | .375 |
| G6 F7 + lam .1, V=40 | 51.28 | .157 | .182 | .218 | .124 | .380 |

Findings:
- Simulator's win logit ≈ intercept + Σ c·XOR(win[t-i], win[t-j]) for pairs (2,5),(1,3),(4,7), machine-specific signs. Extra pairs / single lags HURT.
- Error decomposition: win prob is ~90% of remaining error; perfect size dist only adds ~1.5 pts.
- No recency → stable best (.090) but drift collapses (1.76). Adaptive half-life per machine (pick by log-loss on own last 30 ticks) is the key for drift/shared.
- Plateau ≈51.2 on these knobs. Next ideas: change-point detection (episode-wide for shared_change), better selection so stable keeps hl=inf.

## Submission-ready entries
| Entry | val pts | files | status |
|---|---|---|---|
| main-v1 (xorcast, stdlib only) | 51.16 | `runs/main-v1/submission.csv` (SHA dfb55fcc…) + `runs/main-v1/entry.zip`; source `entries/main-v1/` | checker VALID; fresh read-only reproduction byte-identical; ~15 s runtime |
| **main-v2 (xorcast2, stdlib only)** | **54.58** | `runs/main-v2/submission.csv` (SHA d9edffdf…) + `runs/main-v2/entry.zip`; source `entries/main-v2/` | checker VALID; fresh read-only reproduction byte-identical; ~16 s; beats v1 on 10/10 val half-splits (+3.1..+3.8), train Brier better/equal in every condition |

## Rigorous analysis findings (src/hypotheses.py, hyp2.py, hyp3.py, size_probe.py)
- Stable machines: exactly ONE active XOR pair (352/384), from {(2,5),(1,3),(4,7)}; |coef| ≈ 4.85 (IQR 4.3–5.3); 7% none. Sign ±, ~50/50, not tied to type/signal.
- Other pairs / single lags / lags 9-12 / interactions / cross-machine: tiny or null once single-active-pair is used. 'win' = payout>0 is right definition.
- Drift & shared_change: abrupt change point fits better than linear ramp. Shared change: tau aligned within episode (sd ~2 ticks); drift: per-machine tau. Change ~53-104 ticks before cutoff. Episode-summed LL gain: stable ≤53, changed ≥287 (test: ≤35 vs ≥230) → perfect detection (train acc 1.0; test 48/96/48 split).
- Payout size | win: EXACTLY two discrete distributions per type (low/high mode, hidden), mode independent across machines, weakly related to lags (|r|≤.1). HMM posterior did not beat shrinkage → treat as unobservable. Temperature calibration: none needed (T=1 optimal).
- Error decomposition v2: perfect win → 56.7; perfect size → 57.5.
| **main-v3 (v2 + calibrated size mode)** | **54.84** | `runs/main-v3/submission.csv` + `runs/main-v3/entry.zip`; source `entries/main-v3/` | checker VALID; fresh reproduction byte-identical; all 4 conditions improved vs v2 |

## Size-mode study (src/mode_probe.py, mode_calib.py, mode_check.py)
- Base rate: low 54%. Mode accuracy at query tick (5-fold episode CV where fitted):
  last-k-wins LR ≤ 64%; HMM eps=0 full history 69% but overconfident (Brier .267); v2 shrink implied 66.8%;
  calibrated sigmoid(-.252 + .094·LLR_all) 68.0% (Brier .213); + LLR_last40 69.1%; LightGBM ~50 feats 72.5%.
- Unsupervised HMM on emissions prefers switch prob ~0.1/tick, yet the full-history LLR still predicts the mode → mode has a persistent machine-level component plus noise.
- Shipped calibrated LLR_all (2 params): +0.26 pts avg over 20 val half-splits (all positive), train Brier .3625→.3613.
- LightGBM would give ~55.03 (+0.45) but needs lightgbm at runtime → not shipped.
| **main-v4 (v3 + LightGBM size-mode trees, exported → stdlib)** | 55.95 in-sample / **≈55.05 OOF** | `runs/main-v4/submission.csv` (SHA 7d103a1b…) + `runs/main-v4/entry.zip`; source `entries/main-v4/` | checker VALID; fresh reproduction byte-identical; no runtime deps; mode acc 71.2% OOF; +0.17..0.21 pts vs v3 OOF (all 5 reps & 20 halves positive); train Brier .36129→.36080 |
