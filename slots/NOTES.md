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

## v5 — tied-coefficient prior (Sat 8:30 PM; scratch tests in session scratchpad)
Tried (web-sourced ideas): MacKay moderated posterior predictive (probit approx.) → +0.004 max, rejected; partial-pattern rows for short_history → only +3-5 rows/machine, skipped.
Found: for active machines, intercept b0 and XOR coef c are tied (corr -.89). Midpoint m = b0 + c/2 has sd ~.58 (vs b0 sd ~2.4) and is linear in signal per type.
Reparametrized: logit = m + c*(XOR-0.5), m ~ N(mid[type]+slope[type]*signal, .6²), c ~ N(±4.85, .7²); constants fit on TRAIN histories only
(balanced -.78/.33, jackpot -2.03/.30, steady .54/.62). SDs = measured train spread, not tuned on val.
| **main-v5** | 56.30 in-sample / **≈55.4 OOF** (+0.34 vs v4, 20/20 val halves positive, min +.009) | `runs/main-v5/submission.csv` (SHA 80f225a0…) + `runs/main-v5/entry.zip` (fd7cde49…); source `entries/main-v5/` | checker VALID; fresh read-only reproduction byte-identical; stdlib only |
All 4 condition ratios improved (LLR-mode base): drift .090→.083, shared .087→.081, short .122→.110, stable .078→.076.
Untried ideas: soft change-point (BOCPD run-length averaging) for drift/shared; tighter SDs (.4/.5 gave +0.08 more on val but would be tuning on val).
Generalization check (v5): midpoint structure re-found independently in train/val/TEST histories (last-60-tick fits; mid sd ~.8 vs b0 sd ~2.25 in all three;
per-type lines agree within ~±.15). Constants fit from train / val / test histories → val +.34 / +.26 / +.29 and train-label Brier -.0021/-.0026/-.0025
(every condition improves in every variant). Crude constants (one slope, wide sds) still +.19..+.22. Not val-specific.
Risk: round two may change the generator → idea: re-estimate MID from the round's own histories at predict time.

## Round of deeper search after v5 (Sat night) — nothing reached +0.20, v5 stays
- Regime-change structure: at a change the active pair is KEPT (90% drift / 97% shared) and the sign FLIPS (99-100%); midpoint corr .78 pre/post.
  Pre-change posterior as prior for post-change fit (same pair, flipped sign): +0.014..+0.055 → rejected (post segments already identify the model).
- True win prob is NOT constant within an XOR state on stable machines (z sd 2.0 vs 1.0). Cause found: hidden size mode also shifts win logit (+~0.5 high vs low).
  Adjusting win logit by gamma*(P(high)-.46): LLR mode +0.02, LGB mode +0.18 (in-sample, train Brier unchanged) → rejected.
  Ceiling with PERFECT mode knowledge: win-adjust only +0.36; perfect mode for SIZES +2.7 → the mode predictor is the bottleneck.
- Mode search: no periodicity in cutoff, no episode clustering (sd .172 vs binomial .176), no signal effect, slot effect = noise.
  Lag-1 payout evidence is NEGATIVELY related to current mode (-.08 vs ~+.11 at lags 2-40, flat, no decay) → mode flips after showing itself, and
  otherwise = persistent machine trait + iid noise. LGB already uses last payouts; full-history LLR is near-optimal.
- Moderated (MacKay) predictive: +0.004. Residual correlation of v5 errors with any lag/XOR/size feature ≤ .11.

## main-v6 — regime-change (flip) prior, rigorously tested (Sat night)
Settings from TRAIN histories only: FL_EPS .06 (P(pair kept & sign flipped)=.937), FL_SHRINK .87, FL_SD .75. No val tuning.
| Check | v5 | v6 (flip) | placebo (sign kept) |
|---|---|---|---|
| val points (true probs) | 56.296 | 56.320 (+.024; 72% of 50 halves +) | 56.239 (-.057; 6% +) |
| train next-tick Brier | .35867 | .35859 | .35853 (noise level at this scale) |
| rolling next-tick Brier, 3 ticks after change: train/val/TEST | .222/.198/.208 | .160/.138/.160 | .326/.309/.297 |
| rolling all lags 3-45 (n=2670/2672/5351) | .132/.126/.132 | .111/.108/.112 | .154/.150/.152 |
Gain fades to ~0 by 45 ticks after the change → small at query time (changes 53-104 ticks old). Real & general, but < +0.2 on score.
| main-v6 | 56.32 in-sample | `runs/main-v6/submission.csv` (SHA 1e90bcd3…) + `runs/main-v6/entry.zip` (7ab9bc41…) | checker VALID; read-only reproduction byte-identical |
Revalidation of v6 (rigor2.py in session scratchpad):
- Settings re-derived per split: train .937/.87/.75, val .947/.83/.80, TEST .925/.90/.81 → structure is a property of the generator.
- Val points with settings from train/val/test: +.024/+.015/+.013; eps .03-.15 and shrink .5-1.0 all +.024 (flat). sd .5 gives +.079 but NOT adopted (would be val tuning).
- Stable/short_history predictions identical to v5 (max diff 0). 4 disjoint episode folds: +.006 +.103 -.021 +.058.
- Rolling next-tick Brier change vs v5, episode-bootstrap 95% CI: train -.021 [-.026,-.015], val -.018 [-.024,-.013], test -.020 [-.023,-.016]; all perturbed variants within ±.002.
- Test calibration OK (bins within ~.03; slight overconfidence at extremes).
- Failure mode: FAKE change injected into unchanged machines → v6 +.018 worse. Mitigated by detector separation (stable ≤53 vs changed ≥287, thr 150; test flags exactly 96/192 episodes). Round two: rerun splitstats.py on new histories before trusting the flip prior.
