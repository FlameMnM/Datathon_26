## Inspiration

Slot machines feel random, but the House Edge simulator isn't purely random: each machine's next payout depends on its own recent history in ways that differ from machine to machine, and sometimes the rules change mid-session. The challenge was to predict the *true* odds of the next spin (0×, 1×, 2×, 5× or 20×), scored against the simulator's actual probabilities rather than against one lucky or unlucky spin. We treated it as a detective problem: find the generating rules hidden in the logs, then build the smallest model that captures them.

## What it does

For every query (one machine, one cutoff tick), our forecaster outputs five calibrated probabilities for the next payout. It handles all four conditions the competition scores equally:

- **Stable:** the machine's rules never change.
- **Short history:** only 64 observations, 40 recent and 24 older scattered ticks.
- **Drift:** a machine's rules change partway through.
- **Shared change:** several machines in the same episode change at the same moment.

On the organizers' validation set, our final model scores **about 55.4 out of 60** in held-out estimates (56.32 on the checker; part of the size model was trained on validation, so the held-out figure is the honest one). The frozen reference model scores 30 and the provided starter 30.8.

## How we built it

We followed a strict loop: form a hypothesis from the data, test it on held-out episodes and an independent check, and ship only changes that clearly win. Each version built on the last:

| Version | Key idea | Validation points (of 60) |
|---|---|---|
| Starter (organizers) | Pooled recent frequencies | 30.8 |
| Logistic regression, 8 lags + 6 lag pairs | Order of past wins matters | 32.1 |
| + recency weighting, adaptive per machine | Handle changing machines | 41.8 |
| **Only 3 XOR features** | The real signal is XOR of wins at lag pairs (2,5), (1,3), (4,7) | 50.5 |
| **v1–v2:** change-point detection + Bayesian model averaging | Detect regime changes across the whole episode | 54.6 |
| **v3–v4:** hidden payout-size mode | Each machine type has exactly two payout-size distributions | 54.8 → ~55.05 |
| **v5:** tied-coefficient prior | The two numbers describing each machine are linked | ~55.4 |
| **v6:** regime-change prior | After a change, the machine keeps its lag pair and flips the sign | ~55.4 (+0.02) |

**What we discovered about the simulator:**

- **One active lag pair per machine.** A machine's win probability follows

  $$P(\text{win}_t) = \sigma\big(b_0 + c \cdot (\text{win}_{t-i} \oplus \text{win}_{t-j})\big), \qquad |c| \approx 4.85,$$

  for exactly one pair $(i, j)$ out of (2,5), (1,3), (4,7), or none. Adding more lags or pairs made predictions *worse*.
- **Regime changes are abrupt and shared.** A change-point scan (log-likelihood gain summed across an episode's machines) separates changed from stable episodes perfectly on train and validation. At a change, a machine keeps its lag pair 90–97% of the time and its effect flips sign almost always.
- **Payout size has two hidden modes.** Given a win, sizes come from exactly one of two discrete distributions per machine type. Which mode is active is hidden, so we predict it (about 71% accurately) and mix the two distributions.
- **The two machine parameters are tied.** The intercept and the XOR coefficient are strongly linked (correlation −0.89). Their midpoint varies three times less than the intercept alone and shifts predictably with machine type and signal. Building that link into the prior gave a large gain on short histories.

**The model (pure Python at prediction time):**

1. **Change detection:** per-episode change-point scan, with each machine allowed its own change point if clearly better.
2. **Win probability:** Bayesian model averaging over 7 candidate models (no pair, or one of 3 pairs with either sign), using Laplace-approximated evidence and the tied-coefficient prior. After a detected change, the pre-change fit becomes the prior for the post-change fit.
3. **Payout size:** a mixture of the two hidden size distributions, weighted by a LightGBM classifier's estimate of the current mode. We exported the trees to JSON and evaluate them in plain Python, so the submission needs no third-party packages.
4. **Fast math:** because every feature is one of 8 binary patterns, all fits run on sufficient statistics. The full test set takes about 15 seconds on one CPU core.

**Validation discipline.** Every shipped change had to beat the previous version:
- on validation points, and consistently across random half-splits of the episodes;
- on next-spin outcomes in the training split (an independent check);
- with constants estimated from **training data only**.

For v6 we also ran a placebo control (the same code with the wrong structure, which lost everywhere) and a rolling forecast test on 10,000+ real outcomes inside the train, validation *and test* histories. Every entry was checked with the official format checker and reproduced byte-for-byte from a fresh, read-only unzip of the code package.

**Built with:** Python 3 (standard library only for the submitted predictor), NumPy, pandas, scikit-learn, LightGBM (training only; trees exported to JSON), SciPy, WSL / Linux.

## Challenges we ran into

- **More features made things worse.** A logistic regression with every lag and pair looked reasonable but scored badly. Stripping it down to the three XOR features was the single biggest jump (+7 points).
- **Stable vs changing machines pull in opposite directions.** Recency weighting rescues drifting machines but throws away data on stable ones. Explicit change-point detection let us use full history when nothing changed and only post-change data when something did.
- **The hidden payout mode.** We tried hidden Markov models, which were confident but wrong, before landing on a calibrated classifier.
- **Overfitting risk.** With validation answers public, it's easy to tune to them. We fixed constants from training data, required gains across random half-splits, and rejected ideas that looked good only on validation, including a variant scoring +0.08 that would have meant tuning on validation.
- **Packaging constraints.** The runner offers only Python's standard library, so we wrote our own Newton solver, linear algebra and LightGBM tree evaluator in plain Python.

## Accomplishments that we're proud of

- Reverse-engineering most of the simulator's structure from logs alone: one active XOR pair per machine, two hidden size modes, and sign flips at regime changes.
- Going from the 30-point reference to about **55.4 / 60** with a model small enough to explain on one page.
- A fully reproducible, dependency-free entry that the official checker accepts and that rebuilds byte-identically.
- Saying no to gains we couldn't justify: several "improvements" were tested and rejected because they didn't hold up out of sample.

## What we learned

- On simulated data, finding the generating mechanism beats throwing a bigger model at it.
- Ask "what does the error look like?" before "what model do I add?" Most of our gains came from studying where predictions were wrong.
- Independent checks (held-out halves, a different split, placebo controls) are what separate a real gain from an overfit.

## Adaptation round: when the rules changed

Round two released fresh data where "some patterns have changed," with 90 minutes to respond. We first ran our main-round model unchanged, and it dropped to **reference level: 19.8 / 40** on validation. Our diagnostics showed why within minutes. The simulator kept exactly the same structure (one active XOR lag pair per machine, the same hidden payout-size modes), but the lag pairs moved from (2,5), (1,3), (4,7) to **(1,4), (3,8), (2,6)**.

Instead of hard-coding the new pairs, we built **v7**, which learns the round-specific structure from each round's own training split every time it runs:

- **Active lag pairs:** every pair that best explains at least 10% of the stable training machines, with a significant likelihood-ratio gain.
- **Prior settings:** the midpoint line per machine type, the typical effect size, and the share of machines with no pattern. The spreads use an empirical-Bayes method-of-moments correction (observed variance minus sampling variance), so noise doesn't inflate the priors.

The same code, with no round-specific constants, rediscovers the main-round pairs on main data (56.30 / 60, matching v6) and the new pairs on adaptation data. It scores **37.2 / 40 on adaptation validation**, fully out-of-sample, since nothing was tuned on that validation set.

| Round | Model | Validation points |
|---|---|---|
| Main (60) | v6 | 56.32 |
| Adaptation (40) | v6 unchanged | 19.82 |
| Adaptation (40) | **v7 (learns structure from training data)** | **37.21** |
| **Total (100)** | **v6 main + v7 adaptation** | **≈ 93.5** |

We also tested two ideas suggested by the new round's notes: a neighboring-machine effect and gradual-drift tracking. Neither held up on held-out data or cleared our minimum gain, so we left them out. Our main-round files are unchanged; the adaptation files are added alongside them and clearly labelled.

## What's next for House Edge

- Better hidden-mode prediction: perfect knowledge of the payout mode would be worth about +2.7 points, the largest remaining gap.
- Re-estimating the prior constants from each round's own histories at prediction time, so the model adapts automatically when the simulator's rules shift.
- Soft change-point averaging instead of a single best change point.
