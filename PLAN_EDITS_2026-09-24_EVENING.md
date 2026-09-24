# PLAN EDITS, September 24 2026 evening

I have read roughly 60 percent of `OPALSCALES_PLAN.md` in this session, so
regenerating it whole would silently drop content I never saw. These are
targeted edits instead. Apply them and delete this file, or keep it as the
record of what changed.

---

## EDIT 1. New block at the top of "WHAT IS ESTABLISHED (do not redo)"

Insert before the existing **Markets** subsection:

```
**Deployment (NEW, September 24 evening)**
- **A CHANGE IS NOT SHIPPED UNTIL IT IS RUNNING.** Phase 4.4 was committed as
  `767d3ac` and never pushed. The deployed app served pre-4.4 code while
  every local verification script passed. No script in this project can see
  the gap between committed and deployed. Check `git status -sb` for `ahead`,
  and put a version string on screen. Plan item 6.14.
- **THERE IS NO MORE HISTORICAL PLAYER-PROP DATA TO BUY.** The Odds API's
  historical player props begin 2023-05-03. The general archive reaches 2020
  but covers featured markets only. The 2023 backfill start is the vendor
  floor, not a budget choice. Retire plan item 1.10's "restrict to books
  present in all four seasons" as cheap-and-untested only in the sense that
  it cannot be extended backward. Every further cut of the existing four
  seasons spends power on searched data.
```

---

## EDIT 2. Replace the "Betting" subsection's candidate-rule bullets

Find the bullets beginning "The candidate rule (receptions, lines 2.5 and
3.5 only..." and "**Its weakness is the threshold-selection step, not season
concentration.**" Replace both with:

```
- The candidate rule (receptions, lines 2.5 and 3.5 only, sqrt sigma,
  FanDuel settlement, threshold about 0.09) gives holdout +0.1475, t +1.74,
  176 bets. Reproduced exactly four times on September 24. Decomposition
  also reproduces: +0.0524 book only, +0.0742 model only, neither
  distinguishable from zero alone.
- **SIX explanations were ruled out and a seventh was found.** Ruled out: the
  model beating the price rather than the line (dev +0.0056, t 0.54 with
  price controlled); other books' lines (book term -0.01, placebo centred);
  longshot winners (matches a planted genuine edge); ROI leverage (win-rate
  gap +8.83 pp at t +2.30, a HIGHER t than the ROI); payout dispersion
  (median-price ROI +0.1937, above actual, so dispersion hurts it);
  threshold selection alone (0 of 400 placebo draws reach +0.1475); and a
  level bias toward overs (price rule holdout -0.0401).
- **THE SEVENTH: A DISCONTINUITY WITH NO MECHANISM.** Non-cumulative edge
  bins, no selection: -0.04, -4.97, -2.07, then +5.02, +13.10, +8.93 across
  the 0.09 threshold. Nothing below it is positive. Sigma and the blend are
  smooth in the projection, so nothing in the pricing layer switches on at
  0.09. The dose response across all 4,234 rows is NEGATIVE (-0.3765,
  t -1.70) and the ten edge deciles are flat.
- **Resolution requires about 248 bets in the small arm at 80 percent power.
  The rule has 176.** That is roughly 29 more weeks of live firing, and no
  further analysis of the existing seasons substitutes for it.
```

---

## EDIT 3. New subsection in "WHAT IS ESTABLISHED", after Betting

```
**Side selection (NEW, September 24 evening)**
- **THE MODEL-SELECTION QUESTION DEPENDS ON THE TARGET.** Every model here is
  fitted by least squares on the statistic. Squared error in yards is
  dominated by the 350-yard games; a side decision only cares which side of
  the line the outcome falls on. A model can be the best forecaster and the
  wrong object for picking a side. The original bake-off tested FORECAST
  ACCURACY and has never been run on the classification target.
- **NO BAYESIAN MODEL HAS EVER BEEN TESTED.** Not in any document. The
  record says linear beat boosting on accuracy.
- **QB_PASSING SHOWS A SIDE-PICKING SIGNAL.** A logistic on `book_p` +
  `line_fanduel` + the existing five features: disagreement slope +0.832,
  SE 0.312, t +2.67; per-season +0.947 and +0.734; placebo centred; log-loss
  difference positive; holdout ROI +0.1278, SE 0.0611, t +2.09 on 248 bets.
  Boosting on the same inputs gave t +1.02, so the target and not the
  nonlinearity is what helped. Features checked for leakage and clean.
- **IT SURVIVES THE TEST THAT KILLED THE RECEPTIONS RULE.** Non-cumulative
  bins hold at +8.86, +10.87, +10.00 across three consecutive bins covering
  252 rows, a plateau rather than a spike, and the cutoff takes 23 percent
  of rows rather than 4. With no threshold selected, the game-clustered gap
  CI is [+2.61, +14.81], 99.7 percent positive.
- **NOT ESTABLISHED, and for four stated reasons.** The fixed-cutoff gap is
  concentrated in 2025 (+14.90 at t 3.08 against +5.87 at t 1.39 for 2024)
  even though the slopes are stable, and those two metrics disagreeing is
  unresolved. 217 of 265 bets are overs. The ROI CI touches zero. And
  qb_passing is priced at flat odds, so the baseline is a base rate of
  0.4982 rather than a real forecast, making this "features predict
  over/under nonlinearly" rather than "we beat the book's price".
- **THIS CONTRADICTS A SHIPPED DECISION.** qb_passing beta was +0.085 at
  t +1.5, clipped to zero, market displayed as reference-only. The clip may
  have been right for a linear yards model and wrong as a verdict on the
  market.
- **RECEIVING also showed a slope (+0.570, t +2.72) and should be set
  aside.** Per-season +0.290, +1.314, +0.239, so almost entirely 2025, and
  ROI -0.0076.
- **Detection limit about 10 to 12 points; profitability limit about 6,
  because the hold is 5.7.** Pockets in that band are real and invisible to
  this data.
```

---

## EDIT 4. Replace the "Pattern 1" remaining-work line

In "THE TWO BUG PATTERNS", the line reading "Remaining: the
`targets_roll >= 3` TRAINING gate in `receiving` and `receptions`..."
becomes:

```
Fixed: `anytime_td` (`touches >= 3`), `rushing` (`carries >= 5`),
`qb_passing` (`attempts >= 10`). Remaining: `qb_rushing.actual_result`.

**NEW instance, September 24 evening: the TRAINING POPULATION conditioned on
a prop existing.** A side-picker script fitted its projection model on the
props-joined frame rather than the module's full dataset, which selects for
starters and volume. Two of four market betas failed their gate (rushing
-0.043 to +0.014, qb_passing +0.085 to +0.368) and the drift scaled with how
much the join discarded. **Rule: fit on the module's full dataset, then join
to props for scoring, exactly as `eval_harness.score_market` does.**
```

---

## EDIT 5. Jump the queue

- **J3**: append "**Now entangled with a new bug.** The Line Movement save
  path writes ZERO rows: `app.py:889` filters on `latest_edge.notna()`, and
  after the September 24 fix every surviving row is a captured row, which
  carries no edge by definition. The button reports "Saved 0 pick(s)"
  silently and the caption at `app.py:770` still tells users to use it.
  Decide between computing projection and edge live in `get_line_movement`,
  or removing the Save button and Bet? column and fixing the caption."
- **J5**: mark **DONE**, deleted from `app.py`, local only.
- **J6**: mark **DONE**, and add "and it was how the unpushed commit was
  found: a Line Captures count of 90 where the true figure is 15."
- **NEW J7. THE PRIORITY IS PICKING A SIDE.** Of six live markets, zero
  currently support a defensible side recommendation. This is the app's
  stated purpose and it outranks cleanup, documentation and further cuts of
  the existing four seasons. First step: resolve qb_passing's season
  concentration, because the slope and the fixed-cutoff gap disagree about
  stability on the same data.

---

## EDIT 6. New Phase 3 item

```
**3.8 (NEW) THE CLASSIFICATION BAKE-OFF.** The original model selection
tested forecast accuracy on the statistic and has never been run on the
side-picking target. Compare per market, walk-forward, on log loss AND
realised ROI, with multiplicity counted ACROSS markets rather than within
them: logistic on features plus line plus book probability; gradient
boosting on the same; the current linear projection through `mc_pricing`;
and a Bayesian model. Prerequisite: calibrate whatever metric is used
against a planted pocket first. Average log loss was demonstrated blind to a
6-point pocket in 5 percent of rows at this sample size.
```

---

## EDIT 7. Phase 4, reopen sigma on a new basis

Under Phase 4.2's "closed as not an improvement", append:

```
**REOPENED on a different basis, September 24 evening.** 4.2 compared
hand-specified functions of the MEAN and that comparison stands. It did not
test a coherent predictive distribution, and it could not, because sigma here
is a function of the mean only. Two players with the same projection get
identical spread whether it rests on twelve games or two. See plan item 7.6.
```

---

## EDIT 8. New Phase 6 items

```
**6.14 (NEW) DEPLOYED-VERSION INDICATOR.** A version or commit string in
`app.py`, printed in the footer. The unpushed Phase 4.4 commit was invisible
to every verification script in the project.

**6.15 (NEW) PRE-REGISTRATION LOGGING.** 2026 is the only source of new rows
for the rest of this project's life. Whatever rule survives must be recorded
before outcomes exist: the rule written as a constant rather than a sweepable
parameter, every firing logged at capture time with projection, line, odds,
edge, side and timestamp and nothing about the outcome, and grading done
afterward in a separate write. Manual save-to-log will lose bets and lost
bets are not random.
```

---

## EDIT 9. New Phase 7 item

```
**7.6 (NEW) BAYESIAN HIERARCHICAL MODEL. Promoted from untried to agenda.**
Never tested in this project; the record's "linear won" was about forecast
accuracy against boosting.

Why it is different from anything tried. Every model here produces a point
projection and `mc_pricing` then imposes a SEPARATE sigma fitted
independently. A posterior predictive gives mean and spread from one object.

Why the distribution reaches the SIDE decision, not just the magnitude:
- **Skew flips sides.** `db._model_side` already documents it: under the
  gamma, when the line sits between median and mean, projection-versus-line
  says OVER while the probability says UNDER. A posterior predictive has
  different skew from a moment-matched gamma.
- **Spread decides which props clear the threshold**, and the threshold
  decision is entirely about magnitude.
- **It can express something no current sigma can:** uncertainty that depends
  on how much history backs a projection, not just on its level.

First version: hierarchical negative binomial for receptions, player-level
random effects with partial pooling, which shrinks thin-history players
toward the population instead of dropping them below the projection floor.
PyMC or Stan. Real work, awkward to cache in Streamlit, scope deliberately.
```

---

## EDIT 10. "What the app becomes", final section

Replace the receptions bullet and add one:

```
- **receptions at 2.5 and 3.5**: the candidate rule is now weaker and
  precisely weaker. Its +0.1475 sits on a discontinuity at the selected
  threshold with no mechanism in the pricing layer to produce one. Needs
  about 70 more bets to resolve, which is 29 weeks. Do not ship.
- **qb_passing**: the strongest side-picking candidate in the project, and
  currently displayed as reference-only. Survives the plateau test and a
  selection-free bootstrap. Blocked on the 2025 concentration question.
```
