# OpalScales Handoff
### Thursday, September 24, 2026, EVENING. Week 3 of the 2026 NFL season.

This **supersedes nothing**. It is the second handoff of September 24 and it
**adds to `HANDOFF_2026-09-24.md`**, which remains correct. Read that one
first for the Phase 4.4 ship, the retractions of the sigma floor, and the
harness architecture decision.

The short version. Phase 4.4 was never actually deployed when that document
was written: the commit sat local and unpushed, and the app had been serving
pre-4.4 code the whole time. Fixed. Then the session spent itself on one
question, how to pick a side, and produced the first result in three sessions
that both shows signal and pays: **qb_passing, a logistic on the existing
features, holdout ROI +0.1278 at t +2.09, and it survives the fixed-cutoff
test that killed the receptions candidate rule.**

Along the way the receptions candidate rule survived six attempts to explain
it away and then failed a seventh.

---

## THE OPERATIONAL FINDING, AND IT IS THE MOST IMPORTANT ONE

### PHASE 4.4 WAS COMMITTED BUT NEVER PUSHED

`git status -sb` read `## main...origin/main [ahead 1]`. Commit `767d3ac`,
containing the whole Phase 4.4 ship across `mc_pricing.py`, `mc.py`, `db.py`
and `app.py`, existed only locally. The deployed app had been running the
pre-4.4 code: raw projection as the distribution mean, unfiltered
`get_line_movement`, edge columns on all five markets.

**Nothing in the project could see this.** Every verification script runs
locally against local files. `python mc_pricing.py`, `smoke_ship.py` and
`blend_sigma_grid.py` all passed against code the users were not running.

**What caught it was a count that should have been 15 and was 90.** Christian
Watson, receptions, week 3, showed 90 Line Captures. The true figure is 15
FanDuel snapshots. 90 is exactly 15 snapshots times 6 books, which is the
un-collapsed row count that `_select_book` exists to prevent. After
`git push origin main` (`1f53513..767d3ac`) and a hard reboot, the same cell
reads **15**.

**Add to the working principles: a change is not shipped until it is running.
"Committed" and "deployed" are different states and the project had no check
for the gap.** The cheapest fix is a version string in `app.py` that the
Streamlit footer prints, so the running commit is visible on screen.

### THE RECEIVING GATE, REVERTED

`git status` also showed `M models/receiving.py`: an uncommitted
`targets_roll >= 1.5` edit, the Phase 3.4 change, still live in that file.
The Phase 3.4 instruction had specified BOTH `receptions.py:77` and
`receiving.py:87`, but every revert instruction afterward named
`receptions.py` only. One file was walked back and the other was forgotten.

Phase 3.4's no-op measurement (max 0.137 receptions, mean 0.058) was
**receptions only**. Separate module, separate `load_model`, different
target, so the verdict does not transfer. Reverted with
`git checkout -- models/receiving.py`; both files now read `>= 3` at
receiving:87 and receptions:77, confirmed by `findstr /n /c:">= 3"`.

### SMALL ITEMS

- **J5 DONE.** The temporary `whoami` `st.write` is deleted from `app.py`.
  Local, not yet pushed.
- **J3 saved, not pushed.** The `p_over` one-liner in the Line Movement save
  block is edited. See the next item before deciding whether it matters.
- **NEW BUG: the Line Movement save path writes ZERO rows.** `app.py:889`
  reads `savable = mv[mv["latest_edge"].notna()]`. After the September 24
  fix, `_drop_saved_rows` keeps only captured rows, and captured rows carry
  no `projection` and no `edge` by definition. So `latest_edge` is null for
  every surviving row, `savable` is always empty, and the button reports
  "Saved 0 pick(s)" silently. The caption at `app.py:770` still tells users
  to "Check **Bet?** and hit Save to log picks."
  **Two options, and the choice depends on the side-picking work below:**
  either compute projection and edge live inside `get_line_movement` (more
  correct, more work), or remove the Save button and the Bet? column and fix
  the caption (honest, ten minutes).

---

## THE CANDIDATE RULE: SIX EXPLANATIONS RULED OUT, THEN ONE FOUND

The rule is receptions at FanDuel lines 2.5 and 3.5 only, sqrt-law sigma,
threshold near 0.09 chosen leave-season-out. It reproduced **exactly** four
separate times today: +0.1475, SE 0.0849, t +1.74, 176 bets, win rate 0.5341.

The correct command, which took two attempts to get right:

```
python edge_threshold_v5.py --all-seasons --cache lines_cache.parquet \
  --markets receptions --line-value 2.5,3.5 --sigma-sqrt
```

The decomposition also reproduced: **+0.0524 book only** (`--no-model`),
**+0.0742 model only** (`--same-line`). They sum to 0.1266 against a full
0.1475, and neither arm is distinguishable from zero alone.

### WHAT WAS RULED OUT

| explanation | test | result |
|---|---|---|
| the model beats the price, not just the line | odds_space v3 Test E | dev +0.0056, t 0.54, with price controlled |
| other books' lines carry it | odds_space v4 Test H | book term -0.0101 / -0.0159, placebo centred |
| a few longshot winners | anatomy section 2 | matches a planted genuine edge almost exactly |
| ROI leverage inflating a small gap | anatomy section 1 | win-rate gap +8.83 pp at t +2.30, HIGHER t than the ROI |
| payout dispersion helping it | anatomy section 2 | median-price ROI +0.1937, above actual; dispersion HURTS it |
| threshold selection alone | anatomy section 4 | 0 of 400 placebo draws reach +0.1475 |
| a level bias toward overs | model_free section 2 | price rule holdout -0.0401 |

The composition is 150 of 176 bets on the UNDER at mean decimal odds 2.1626.

### WHAT FINALLY EXPLAINED IT: A CLIFF WITH NO MECHANISM

`model_free_test_v3.py`, non-cumulative bins, no selection anywhere:

```
edge bin         rows   gap pp
0.050 to 0.065    395    -0.04
0.065 to 0.080    261    -4.97
0.080 to 0.090    110    -2.07
0.090 to 0.100     61    +5.02     <- the selected point
0.100 to 0.115     70   +13.10
0.115 to 0.140     38    +8.93
```

**Nothing below 0.09 is positive. The bin immediately below the threshold is
-2.07 and the two immediately above are +5.02 and +13.10.** That is a 7 to
15 point discontinuity across a 0.01 boundary with no gradient approaching
it.

**A real distributional edge cannot do that.** Sigma and the blend are smooth
functions of the projection, so P(over) is smooth, so the edge is smooth.
Nothing in the pricing layer switches on at 0.09.

Supporting evidence for the same reading:
- The dose response across all 4,234 rows is NEGATIVE, -0.3765 at t -1.70.
- The ten edge deciles are flat: +0.06, +1.05, +2.69, -0.20, -2.89, -0.88,
  +1.01, +0.46, -1.32, +0.70.
- The top decile spans 0.0720 to 0.2301 with an overall gap of +0.70 pp. The
  rule takes 177 of its 424 rows at +8.53, so the other 247 run about -4.9.
  A thirteen-point swing inside one decile.
- At a FIXED 0.09 the per-season gaps are +2.73, +10.04, **+19.69**, more
  monotone than the selected version, largest effect in the smallest cell.

**Status change.** From "not established, weakness is threshold selection" to
**"the threshold-selection weakness now has direct evidence behind it, in the
form of a discontinuity with no mechanism."** Against a generously counted
Bonferroni threshold of 0.00057 over about 88 cuts of this one population, a
t of 2.42 is not close.

### THE POWER ACCOUNTING, WHICH SETTLES WHAT WOULD RESOLVE IT

At a base win probability of 0.4458 and an observed gap of +8.83 pp, 80
percent power needs about **248 bets in the small arm. The rule has 176.**
That is roughly 70 more bets, or **29 weeks at 2.5 firings a week.**

Nothing further on the existing four seasons will help. See the next section.

---

## THERE IS NO MORE HISTORICAL DATA TO BUY. AT ANY PRICE.

**The Odds API's historical player props begin 2023-05-03.** Confirmed from
their documentation and their historical-odds page. The general archive goes
back to 2020 but covers featured markets only: moneyline, spreads, totals.

So the 2023 start of the backfill was **not a budget decision, it is the
floor.** I proposed extending to 2021 and 2022 for about 6,000 credits before
checking, and the cost arithmetic was correct and irrelevant.

**Consequences.**
1. The four seasons in hand are all there will ever be for the backtest.
2. Every additional cut of them spends power on searched data with no
   possibility of topping it up. The multiplicity accounting gets sharper,
   not softer, over time.
3. 2026 is the only source of genuinely new rows, which makes
   pre-registration plumbing the highest-value operational work rather than
   an optional extra.

Record this so nobody prices it again.

---

## THE SIDE-PICKER SEARCH, AND THE FIRST POSITIVE RESULT IN THREE SESSIONS

### WHY THE EARLIER NEGATIVES DID NOT CLOSE THE QUESTION

Every test in odds_space v1 through v4 asked whether `(projection - line)`
predicts `(actual - line)` **linearly**. Beta is one scalar. That
specification assumes the market's error is proportional to our disagreement
with it, so it cannot see a market that is efficient on average and wrong in
identifiable pockets. Pockets are interactions and a scalar has no way to
represent one.

**And there is a second, larger gap: the TARGET.** Every model in the project
is fitted by least squares on the statistic. Squared error in yards is
dominated by getting the 350-yard games roughly right, and a side decision
only cares which side of the line the outcome falls on, which is exactly
where squared error pays least attention. **A model can be the best yards
forecaster and the wrong object for picking a side.**

**Note for the record: the original model bake-off tested FORECAST ACCURACY,
and it never included a Bayesian model at all.** Nothing in either document
mentions one. "Linear beat Bayesian and XGBoost" is not in the project
record; what is recorded is linear beating boosting on accuracy. Nobody has
ever compared model families on the CLASSIFICATION target, because until the
historical line data arrived there was no line to define a binary outcome
around.

### TWO BUGS IN MY FIRST ATTEMPT, BOTH WORTH KEEPING

**Bug A: average log loss cannot detect a pocket.** Planted a 6-point market
error in a 5 percent pocket defined by a feature interaction. Boosting
returned t -5.49 with no pocket and t -5.45 with the pocket. Identical. The
model's own variance costs about 0.012 of log loss, which swamps the signal.
**A script using average log loss would have returned a confident negative
that meant nothing.**

The working metric is the **directional slope**: regress the outcome on the
model's disagreement with the book, controlling for the book. Noise
uncorrelated with the outcome contributes zero slope. Planted calibration:
no pocket t +0.27, 6-point pocket t -0.07, 12-point pocket t +2.17.

**Detection limit about 10 to 12 points. Profitability limit about 6 points,
because the hold is 5.7. There is a band of profitable-but-invisible
pockets and no amount of method fixes it.**

**Bug B: the training population.** `eval_harness.score_market` fits its
projection on the module's FULL dataset then joins to props. My first version
fitted on the JOINED frame, which conditions the training population on a
FanDuel prop existing, which selects for starters and volume. The drift
scaled with how much the join discards:

| market | joined / full | published | got |
|---|---|---|---|
| receptions | 7850 / 22540 | +0.226 | +0.241 |
| receiving | 8676 / 22540 | +0.066 | +0.073 |
| rushing | 2671 / 6635 | -0.043 | **+0.014 FAIL** |
| qb_passing | 1670 / 2817 | +0.085 | **+0.368 FAIL** |

Fixed, and all four then reproduced to three decimals.

**Bug C: nested walk-forward.** The challengers were fitted on the gate's
OUTPUT, which already excludes the first season, so the second season trained
on nothing and was never scored. Receiving reported 3,148 out-of-sample rows
instead of 5,939. Fixed.

**Also recorded: seed averaging is a no-op for
`HistGradientBoostingClassifier`.** Five seeds gave slopes identical to three
decimals, because it is deterministic unless rows are subsampled;
`random_state` only affects binning. Caught by the project's own "identical
when it should have moved" rule.

### THE RESULT: QB_PASSING

`side_picker_search_v2.py`, all gates exact. Challenger 3 is a **logistic on
`book_p` + `line_fanduel` + the existing five features**.

```
disagreement slope  +0.832   SE 0.312   t +2.67   p 0.0076
2024                +0.947   t 1.85
2025                +0.734   t 1.72
placebo             centred, t +0.26
dLogLoss            +0.00034  (positive)
HOLDOUT ROI         +0.1278   SE 0.0611   t +2.09   248 bets
```

Boosting on the same inputs gave t +1.02. **So it is not the nonlinearity
that helped, it is the target.** Feature leakage checked and absent:
`attempts_roll` and `def_pass_roll` are both `shift(1).rolling(6)`,
`def_pass_roll`'s fillna uses `expanding().mean().shift(1)`, and `wind_eff`
is game conditions rather than a player result.

**Receiving also showed signal and should be set aside.** Slope shrank from
+1.191 to +0.570 once four seasons were scored, per-season it is +0.290,
+1.314, +0.239, so it lives almost entirely in 2025, and ROI is -0.0076.

### THE VERIFICATION: IT SURVIVES THE TEST THAT KILLED RECEPTIONS

`qb_passing_verify.py`, four gates exact (beta 0.0851, slope 0.8324, ROI
0.1278, 248 bets). Non-cumulative bins, no selection:

```
edge bin       rows   gap pp   SE    t      ROI
0.00 to 0.02    282    +2.48  3.02  0.82  -0.0029
0.02 to 0.04    215    -0.23  3.39 -0.07  -0.0545
0.04 to 0.06    158    +8.86  4.09  2.17  +0.1190
0.06 to 0.08     69   +10.87  5.93  1.83  +0.1572
0.08 to 0.10     25   +10.00 10.54  0.95  +0.1409
```

**Compare the two markets above their selected cutoffs:**

```
receptions   -0.04  -4.97  -2.07 | +5.02  +13.10  +8.93
qb_passing   +2.48  -0.23        | +8.86  +10.87  +10.00
```

Receptions wanders. **qb_passing holds at +9 to +11 across three consecutive
bins covering 252 rows.** The cutoff at 0.04 takes 265 of 1,136 rows, 23
percent, not a 4 percent tail. A shelf, not a cliff.

**With NO threshold selected at all**, game-clustered bootstrap at a fixed
0.040: gap CI **[+2.61, +14.81]**, 99.7 percent positive. ROI CI
[-0.0000, +0.2328], 97.5 percent positive.

### WHAT STOPS IT BEING ESTABLISHED

1. **Season concentration, the same pattern that killed receiving.** At a
   fixed cutoff, 2024 gives +5.87 at t 1.39 and 2025 gives **+14.90 at
   t 3.08**. 2026 has 17 rows. The slopes were stable (0.947, 0.734) while
   the fixed-cutoff gaps are not, so two metrics on the same data disagree
   about stability. That tension is unresolved and is the first thing to
   look at.
2. **217 of 265 bets are overs, 82 percent.** The structural over-lean in
   this market is real but small: zero-deviation actual over rate 0.5060
   against a priced 0.4530. Nowhere near +8.88 points, so the model is
   choosing WHICH overs rather than just betting overs, but the skew is the
   filtered-sample signature.
3. **The ROI CI touches zero exactly** and p 0.0076 does not clear the
   38-comparison Bonferroni threshold of 0.00132.
4. **The baseline is weak.** qb_passing is priced at flat odds both sides, so
   `book_p` is near-constant and the challenger only beats a base rate of
   0.4982. This reads as "features predict over/under nonlinearly" rather
   than "we beat the book's price". Receptions faced a real forecast, price
   slope 1.008.

**Still: qb_passing now has a stronger case than the receptions candidate
rule ever had.** A plateau rather than a spike, a selection-free CI excluding
zero, stable per-season slopes, a positive log-loss difference. Four partly
independent facts.

**And it contradicts a shipped decision.** qb_passing beta was +0.085 at
t +1.5, clipped to zero, and the market is displayed as reference-only. The
clip may have been right for a linear yards model and wrong as a verdict on
the market.

---

## NEXT SESSION: THE PRIORITY IS PICKING A SIDE

**This is the project's stated goal and it is not yet solved. Treat it as the
priority over cleanup, over documentation, and over further cuts of the
existing four seasons.** Of six live markets, zero currently support a
defensible side recommendation. That is the gap to close.

### 1. Resolve qb_passing's season concentration. First.

The slopes are stable across 2024 and 2025 while the fixed-cutoff gaps are
not. Same data, two metrics, different answers. Find out which is right
before anything else, because everything below depends on it.

### 2. Run the CLASSIFICATION bake-off across all four markets.

The original model selection tested forecast accuracy on the statistic. It
has never been run on the side-picking target. Compare, per market, on log
loss and on realised ROI, walk-forward, with multiplicity counted across
markets and not within them:

- logistic on features plus line plus book probability
- gradient boosting, same inputs
- the current linear projection passed through `mc_pricing`
- **and a Bayesian model, which has never been tried at all**

### 3. BAYESIAN IS NOW A REAL AGENDA ITEM, NOT A CURIOSITY.

**Why it is different from anything tested.** Every model in the project
produces a point projection, and `mc_pricing` then imposes a SEPARATE sigma
fitted independently. Phase 4.2 closed sigma as "not an improvement", but it
only ever compared hand-specified functions of the MEAN. A posterior
predictive gives mean and spread from one coherent object.

**Why the distribution reaches the side decision, which I initially got
wrong:**
- **Skew flips sides.** `db._model_side` already documents this: under the
  gamma, when the line sits between median and mean, projection-versus-line
  says OVER while the probability says UNDER. A posterior predictive has
  different skew from a moment-matched gamma, so it flips individual props.
- **Spread decides which props clear the threshold.** Sigma does not change
  the favoured side when the mean is clearly off the line, but the threshold
  decision is entirely about magnitude.
- **A capability nothing in the project can express.** Sigma is a function of
  the mean only, so two players with the same projection get identical spread
  whether that projection rests on twelve games or two. A hierarchical
  posterior separates them automatically.

The natural first version is a hierarchical negative binomial for receptions:
player-level random effects with partial pooling, which shrinks thin-history
players toward the population instead of dropping them below a projection
floor. PyMC or Stan. It is real work and awkward to cache in Streamlit, so
scope it deliberately.

### 4. Pre-registration plumbing, because 2026 is the only new data.

Whatever rule survives, it has to be recorded BEFORE outcomes exist:
1. Write the rule down as a constant, not a sweepable parameter. If it
   changes later, the count restarts.
2. Log every firing at capture time with projection, line, odds, edge, side
   and timestamp, and nothing about the outcome. Manual save-to-log will lose
   bets, and lost bets are not random.
3. Grade afterward, separately, never in the same write.

### 5. Add a deployed-version indicator.

A version string in `app.py` printed in the footer. The unpushed-commit
failure was invisible to every script in the project and cost a day of
believing a change was live.

### 6. Then the small items.

J3 or the Line Movement save decision, 6.10's source column on `lines`,
6.11's hardcoded `GAMES_PLAYED = 0`, 6.12's dead fourth sigma, 6.13's
integer reception lines.

### Do not bother with

- More cuts of the receptions 2.5 and 3.5 stratum. About 88 already, and
  there is no more data to buy.
- Extending the backfill. Player props start 2023-05-03.
- Average log loss as a model-comparison metric at this sample size.
  Demonstrated blind to a planted pocket.
- Seed-averaging `HistGradientBoostingClassifier`. Deterministic.
- Boosting for side decisions. Lost to logistic on the same inputs in every
  market, and the log-loss differences were negative throughout.

---

## MY ERRORS TODAY, RECORDED

Eleven. The pattern differs from both earlier sessions: most were caught by
tests I had written for the purpose, and two would have produced confident
false negatives if I had not calibrated the instrument against planted data.

### Would have produced a wrong conclusion

1. **Test G's ROI sweep was ONE-SIDED.** Every cell backed the expensive
   side. The candidate rule backs the CHEAP side at plus money, so I never
   tested the direction the rule actually bets, then read the negatives as
   evidence against it.
2. **Test D pooled all line strata with no fixed effects.** Over rate falls
   monotonically with the line, so between-stratum covariance loaded onto
   `dev`. The placebo caught it: mean +0.00850, about 17 SEs above zero.
3. **Average log loss for pocket detection.** Planted test showed it blind.
   Would have returned a confident negative on the side-picker question.
4. **Trained the side-picker's projection on the joined frame**, conditioning
   the population on a prop existing. Two of four gates failed.
5. **Nested the walk-forward**, so the second season trained on nothing and
   receiving was scored on two seasons rather than four.

### Wrong predictions, stated in advance and refuted

6. "The book term carries the candidate rule." Zero in both specs.
7. "The book term is largest in 2023." No season pattern.
8. "No plateau for qb_passing." There is a plateau.
9. "The receiving slope will not survive." It shrank but cleared its
   per-market threshold, though it failed on season stability.

### Sloppiness

10. **Gave the wrong v5 commands three times**, running all markets and all
    lines instead of the candidate rule's population, then read the mismatch
    as a failure to reproduce.
11. **A gate that passed by 0.0001 on a nonsense comparison.** Passed
    `et5.GRID` (0 to 0.25, an EDGE grid) as the threshold grid for a DECIMAL
    ODDS column, so every row cleared it and the "price rule" was the whole
    population. It reported -0.0302 against a published -0.0401 and passed a
    0.010 tolerance by one ten-thousandth.

### And one claim I accepted without checking

**"Linear beat Bayesian and XGBoost."** I repeated this back from the
conversation. Nothing in either document mentions a Bayesian model. Ted
caught it. The record says linear beat boosting on FORECAST ACCURACY, which
is a different question from side selection and does not include Bayesian at
all.

---

## WORKING PRINCIPLES

Unchanged: hard reboot before concluding a change did not work; always
`encoding='utf-8'`; never use em dashes; test, do not assume; gate before
sweeping; a t-statistic of +6 is a bug until proven otherwise; plain-English
reasoning before design decisions; one command at a time; import, do not
restate; a script that reimplements an existing procedure must reproduce that
procedure's published output before producing a new number; record the sign
convention in every calibration table; check which parameters a result was
computed under before comparing two results; the mirrored filter rule; verify
that a change registered before asking whether it helped; never measure a
small effect through a rounded output; measure alpha directly at zero
deviation; print a multiplicity threshold beside every p-value; report the
minimum detectable effect beside every null result.

Added this evening:

- **A CHANGE IS NOT SHIPPED UNTIL IT IS RUNNING.** Committed and deployed are
  different states. Every verification script in this project runs against
  local files and cannot see the difference. Check `git status -sb` for
  `ahead`, and put a version string on screen.
- **CALIBRATE THE INSTRUMENT AGAINST PLANTED DATA BEFORE TRUSTING A NULL.**
  Average log loss was demonstrably blind to the exact object it was built to
  find. A negative from an uncalibrated metric is worth nothing. Plant the
  effect, check the metric sees it, then run it for real.
- **REPORT THE DETECTION LIMIT ALONGSIDE THE PROFITABILITY LIMIT.** This data
  detects pockets above roughly 10 points and pockets above 6 points would
  pay. The band between is real and invisible, and saying so is part of the
  result.
- **A PLATEAU IS EVIDENCE, A SPIKE IS SELECTION.** Fixed-cutoff
  neighbourhood tests with no argmax anywhere separate the two. This is what
  distinguished qb_passing from the receptions candidate rule.
- **NON-CUMULATIVE BINS, NOT CUMULATIVE ONES.** A cumulative curve is
  autocorrelated by construction and makes almost any pattern look smooth.
- **COUNT MULTIPLICITY ACROSS THE SESSION, NOT WITHIN THE TABLE.** Printing a
  per-market Bonferroni threshold while running twenty comparisons across
  four markets understates the look-elsewhere effect by a factor of four.
- **THE MODEL-SELECTION QUESTION DEPENDS ON THE TARGET.** Least squares on
  the statistic and a side decision at the line are different objects. A
  result established for one does not transfer.
- **CHECK A REMEMBERED PREMISE BEFORE REASONING FROM IT.** I accepted
  "Bayesian lost" from conversation and it is nowhere in the record.

---

## THE HONEST FRAME

The morning's handoff described Phase 4.4 as shipped. It was committed and
not pushed, so the users were on pre-4.4 code and no script in the project
could tell. That is the most important thing in this document, and the thing
that caught it was one number that should have been 15.

On the substance, the day went the opposite way from the last two sessions.
Those retracted headline findings. This one spent itself failing to kill the
receptions candidate rule, ruling out six explanations in a row, and then
found the seventh: a discontinuity at the selected threshold with no
mechanism in the pricing layer to produce it. That rule's status is now
weaker, and precisely weaker, which is better than weak for unstated reasons.

And the side-picking question, which the project had effectively closed by
concluding that four of five markets carry no signal, turns out to have been
closed on the wrong test. Beta is a scalar fitted by least squares on the
statistic. A logistic on the same features, aimed at the side rather than the
mean, finds something in qb_passing that survives a plateau test, a
selection-free bootstrap, and a placebo. It is not established. Its case
rests on 1,136 out-of-sample rows across two scorable seasons, one of which
carries most of the effect.

**The priority for the next session is explicitly the side-picking problem.**
Not cleanup, not documentation, not more slices of the four seasons that are
all there will ever be. If that means a new model family, including a
Bayesian hierarchical model that this project has never tried, then that is
what it means. Of six live markets, zero currently support a defensible side
recommendation, and closing that gap is the whole point of the app.

Eleven errors, five of which would have produced wrong conclusions and were
caught by instruments built to catch them. The apparatus is still the asset.
