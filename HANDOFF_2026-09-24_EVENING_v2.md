# OpalScales Handoff
### Thursday, September 24, 2026, EVENING. Week 3 of the 2026 NFL season.

**This replaces the first version of `HANDOFF_2026-09-24_EVENING.md`, which
was pushed earlier this evening and contained an error in its caveat 1
against the qb_passing signal. Delete that file.** The error and its
correction are recorded below, because the error is instructive.

This **adds to `HANDOFF_2026-09-24.md`**, which remains correct. Read that
one first for the Phase 4.4 ship, the retraction of the sigma floor, and the
harness architecture decision.

The short version, three findings:

1. **Phase 4.4 was never deployed** when the morning handoff called it
   shipped. The commit sat local and unpushed and the app served pre-4.4
   code. Fixed. No script in this project could see the gap.
2. **The receptions candidate rule survived six attempts to explain it away
   and then failed a seventh.** Its +0.1475 sits on a discontinuity at the
   selected threshold with no mechanism in the pricing layer to produce one.
3. **qb_passing has a side-picking signal and it is the best-supported
   result in the project.** A logistic on the existing features, aimed at the
   side rather than the mean. Holdout ROI +0.1278 at t +2.09, a plateau
   rather than a spike, a selection-free CI excluding zero, no season
   heterogeneity, and a monotone dose response with a clean sign flip at
   zero disagreement. Not established.

---

## PART 1. THE OPERATIONAL FINDING

### PHASE 4.4 WAS COMMITTED BUT NEVER PUSHED

`git status -sb` read `## main...origin/main [ahead 1]`. Commit `767d3ac`,
containing the whole Phase 4.4 ship across `mc_pricing.py`, `mc.py`, `db.py`
and `app.py`, existed only locally. The deployed app had been running
pre-4.4 code: raw projection as the distribution mean, unfiltered
`get_line_movement`, edge columns on all five markets.

**Nothing in the project could see this.** Every verification script runs
locally against local files. `python mc_pricing.py`, `smoke_ship.py` and
`blend_sigma_grid.py` all passed against code the users were not running.

**What caught it was a count that should have been 15 and was 90.** Christian
Watson, receptions, week 3, showed 90 Line Captures in the Line Movement tab.
The true figure is 15 FanDuel snapshots. 90 is exactly 15 snapshots times 6
books, which is the un-collapsed row count `_select_book` exists to prevent.

The diagnostic path is worth recording because it was not obvious:
- Raw query returned 98 rows for that player across all weeks and books.
- 8 of those were early single-book captures under `week=1` and `week=2`
  labels, legitimately excluded from a week-3 view.
- That left 90 rows in week 3 across 6 books, so 15 snapshots.
- The app showed 90, which is the week-3 population with NO book filter.
- `_select_book` has no fallback path back to unfiltered data, so it could
  not produce 90 if it were running. Therefore it was not running.

After `git push origin main` (`1f53513..767d3ac`) and a hard reboot, the same
cell reads **15**. Items J6 verified: receptions keeps its edge column, the
four reference-only markets lose edge, side and tier entirely with the
explanatory note, and the prefill populates.

**Plan item 6.14: a version or commit string in `app.py`, printed in the
footer.** "Committed" and "deployed" are different states and the project had
no check for the gap.

### THE RECEIVING GATE, REVERTED

`git status` also showed `M models/receiving.py`: an uncommitted
`targets_roll >= 1.5` edit, the Phase 3.4 change, still live in that file.

The Phase 3.4 instruction had specified BOTH `receptions.py:77` and
`receiving.py:87` and a verification grep had confirmed both landed. Then the
harness came back byte-identical, the work moved to direct measurement, and
every revert instruction afterward named `receptions.py` only. One file was
walked back and the other forgotten.

**Phase 3.4's no-op measurement (max 0.137 receptions, mean absolute 0.058)
was receptions only.** Separate module, separate `load_model`, different
target, so the verdict does not transfer. `gate_1p5.json`, `gate_3.json` and
the unrounded diff all came from `models.receptions`. Receiving was never
measured at either gate.

Reverted with `git checkout -- models/receiving.py`. Both files now read
`>= 3`, confirmed positively at receiving:87 and receptions:77 by
`findstr /n /c:">= 3" models\receiving.py models\receptions.py`. Note the
first verification attempt searched for `">= 1.5"` and returned zero hits,
which was the right answer for the wrong reason: the serving gates are
written `min_targets=1.5`, not `>= 1.5`, so that pattern could never have
matched. **A zero-hit `findstr` needs a positive confirmation beside it.**

### SMALL ITEMS

- **J5 DONE.** The temporary `whoami` `st.write` at `app.py:766` is deleted.
- **J3 saved.** The `p_over` one-liner in the Line Movement save block is
  edited. Variables resolve: `savable` at 889, `is_td` at 810, and
  `get_line_movement` does return a real `p_over` column. It will not crash.
  It is also currently a no-op. See the next item.
- **NEW BUG: THE LINE MOVEMENT SAVE PATH WRITES ZERO ROWS.** `app.py:889`
  reads `savable = mv[mv["latest_edge"].notna()]`. After the September 24
  fix, `_drop_saved_rows` keeps only captured rows, and captured rows carry
  no `projection` and no `edge` by definition, which is the very
  discriminator the fix uses. So `latest_edge` is null for every surviving
  row, `savable` is always empty, and the button reports "Saved 0 pick(s)"
  silently. The caption at `app.py:770` still tells users to "Check
  **Bet?** and hit Save to log picks."

  This also corrects a misreading from earlier in the session. The blank
  Tier, Proj and Edge columns in the Line Movement table are NOT the
  reference-only feature working. They are blank because captured rows never
  carried a projection or an edge, and the rows that did carry them were the
  hand-saved ones the fix just dropped. That is true for receptions too.
  Reference-only gating is real and works, but on the Board, not here.

  **Two options.** Compute projection, edge and p_over live inside
  `get_line_movement` from each market's model, which restores the tab and
  makes it consistent with the Board. Or remove the Save button and the Bet?
  column and fix the caption. The first is more correct and more work; the
  second is honest and takes ten minutes.

---

## PART 2. THE CANDIDATE RULE: SIX EXPLANATIONS OUT, THEN A SEVENTH IN

The rule is receptions at FanDuel lines 2.5 and 3.5 only, sqrt-law sigma,
threshold near 0.09 chosen leave-season-out. It reproduced **exactly** four
separate times: +0.1475, SE 0.0849, t +1.74, 176 bets, win rate 0.5341.

The correct command, which took three attempts to get right:

```
python edge_threshold_v5.py --all-seasons --cache lines_cache.parquet \
  --markets receptions --line-value 2.5,3.5 --sigma-sqrt
```

Running v5 with default flags prices all four markets, all lines and flat
sigma, which is a different object entirely. The decomposition also
reproduced: **+0.0524 book only** (`--no-model`), **+0.0742 model only**
(`--same-line`). They sum to 0.1266 against a full 0.1475, and neither arm is
distinguishable from zero alone.

### WHAT WAS RULED OUT, AND BY WHAT

| explanation | test | result |
|---|---|---|
| the model beats the price, not just the line | odds_space v3 Test E | dev +0.0056, t 0.54, price controlled; book_p 0.9901 at t 10.21 |
| other books' lines carry it | odds_space v4 Test H | book term -0.0101 and -0.0159 in two coordinate systems, placebo centred |
| a few longshot winners | anatomy section 2 | ROI without top 5 +0.1024 against a planted-genuine-edge benchmark of +0.0993; without top 10 +0.0584 against +0.0577 |
| ROI leverage inflating a small gap | anatomy section 1 | win-rate gap +8.83 pp at t +2.30, a HIGHER t than the ROI's +1.74 |
| payout dispersion helping it | anatomy section 2 | ROI if every win paid the MEDIAN price is +0.1937, above the actual, so dispersion HURTS it |
| threshold selection alone | anatomy section 4 | full-procedure placebo mean -0.0437, sd 0.0454, 0 of 400 draws reach +0.1475 |
| a level bias toward overs | model_free section 2 | the price rule under the same holdout discipline returns -0.0401 |

Composition: **150 of 176 bets are UNDERS at mean decimal odds 2.1626**, mean
devigged fair probability of the chosen side 0.4458.

### WHAT FINALLY EXPLAINED IT: A DISCONTINUITY WITH NO MECHANISM

`model_free_test_v3.py`, non-cumulative bins, no selection anywhere:

```
edge bin         rows   gap pp   SE pp     t      ROI
0.050 to 0.065    395    -0.04    2.51  -0.02  -0.0583
0.065 to 0.080    261    -4.97    2.99  -1.66  -0.1683
0.080 to 0.090    110    -2.07    4.63  -0.45  -0.1248
0.090 to 0.100     61    +5.02    6.35   0.79  +0.0573   <- selected point
0.100 to 0.115     70   +13.10    5.86   2.23  +0.2290
0.115 to 0.140     38    +8.93    7.99   1.12  +0.1776
```

**Nothing below 0.09 is positive. The bin immediately below the threshold is
-2.07 and the two immediately above are +5.02 and +13.10.** A 7 to 15 point
discontinuity across a 0.01 boundary with no gradient approaching it.

**A real distributional edge cannot do that.** Sigma and the blend are smooth
functions of the projection, so P(over) is smooth, so the edge is smooth.
Nothing in the pricing layer switches on at 0.09.

Corroborating:
- The dose response across all 4,234 rows is **negative**, -0.3765 at t -1.70.
- The ten edge deciles are flat: +0.06, +1.05, +2.69, -0.20, -2.89, -0.88,
  +1.01, +0.46, -1.32, +0.70.
- The top decile spans 0.0720 to 0.2301 with an overall gap of +0.70 pp. The
  rule takes 177 of its 424 rows at +8.53 pp, so the other 247 must run about
  -4.9. A thirteen-point swing inside one decile.
- At a FIXED 0.09, per-season gaps are +2.73, +10.04, **+19.69**, MORE
  monotone than the selected version, largest effect in the smallest cell
  (42 rows).
- D1 cumulative peaks at 0.100 (+11.20) and falls to +6.56 at 0.110. No
  plateau.

**One thing still stands in the rule's favour and should not be dropped.**
With the threshold held FIXED at 0.09, the game-clustered bootstrap gives a
gap CI of [+1.91, +16.47] with 99.3 percent positive. That removes the
argmax. It does not remove the fact that 0.09 was originally chosen by an
argmax on this same data in an earlier session, so fixing it now is not
pre-specification.

**Status change: from "not established, weakness is threshold selection" to
"the threshold-selection weakness now has direct evidence behind it, in the
form of a discontinuity with no mechanism."** Against a generously counted
Bonferroni threshold of 0.00057 over about 88 cuts of this one population, a
t of 2.42 is not close.

### THE POWER ACCOUNTING, WHICH SETTLES WHAT WOULD RESOLVE IT

At a base win probability of 0.4458 and an observed gap of +8.83 pp, 80
percent power needs about **248 bets in the small arm. The rule has 176.**
Roughly 70 more bets, which is **29 weeks at 2.5 firings a week**. Analytic
SE of a win-rate difference at 176 against 4,057 is 3.82 pp, MDE 10.69 pp
against an effect of 8.83, so the binary tail test is underpowered by
construction.

Nothing further on the existing four seasons helps. See Part 3.

---

## PART 3. THERE IS NO MORE HISTORICAL DATA TO BUY. AT ANY PRICE.

**The Odds API's historical player props begin 2023-05-03.** Confirmed from
their API documentation and their historical-odds page. The general archive
reaches back to 2020 but covers featured markets only: moneyline, spreads,
totals.

So the 2023 start of the backfill was **not a budget decision, it is the
vendor floor.** I proposed extending to 2021 and 2022 for about 6,000 credits
before checking availability. The cost arithmetic was correct and irrelevant.

**Consequences, all three of which change how future sessions should work.**
1. The four seasons in hand are all there will ever be for the backtest.
2. Every additional cut of them spends power on searched data with no
   possibility of topping it up. The multiplicity accounting gets sharper
   over time, not softer.
3. 2026 is the only source of genuinely new rows for the rest of this
   project's life, which makes pre-registration plumbing the highest-value
   operational work rather than an optional extra. Plan item 6.15.

Record this so nobody prices it again.

---

## PART 4. THE SIDE-PICKER SEARCH

### WHY THE EARLIER NEGATIVES DID NOT CLOSE THE QUESTION

Every test in odds_space v1 through v4 asked whether `(projection - line)`
predicts `(actual - line)` **linearly**. Beta is one scalar. That assumes the
market's error is proportional to our disagreement with it, so it cannot see
a market that is efficient on average and wrong in identifiable pockets.
Pockets are interactions and a scalar cannot represent one.

**And there is a second, larger gap: the TARGET.** Every model in this
project is fitted by least squares on the statistic. Squared error in yards
is dominated by getting the 350-yard games roughly right, and a side decision
only cares which side of the line the outcome falls on, which is exactly
where squared error pays least attention. **A model can be the best yards
forecaster and the wrong object for picking a side.**

**For the record: the original model bake-off tested FORECAST ACCURACY, and
it never included a Bayesian model at all.** Nothing in either document
mentions one. "Linear beat Bayesian and XGBoost" is not in the project
record; what is recorded is linear beating boosting on accuracy. Nobody has
compared model families on the CLASSIFICATION target, because until the
historical line data arrived there was no line to define a binary outcome
around.

### WHAT ODDS-SPACE v1 TO v4 DID ESTABLISH, AND IT IS SUBSTANTIAL

These remain correct and should not be redone.

**FanDuel uses ONE channel per market and never both.**

| market | median stratum price SD | testable |
|---|---|---|
| receptions | 0.0531 | yes |
| receiving | 0.0019 | no |
| rushing | 0.0000 | no |
| qb_passing | 0.0000 | no |

Receptions gets a coarse half-integer line and all the information goes into
the odds. The yardage markets get a fine half-yard line and flat -113 both
sides. **The yardage slopes are UNDEFINED rather than zero**, which is what
v1's MDE of 4.096 for receiving and 27.068 for rushing means.

**The receptions price is an efficient forecast.** Within-line-stratum slope
1.008, SE 0.090, t 11.18. Per line: 0.838, 0.927, 1.017, 1.158, 1.636,
1.594, rising with the line. The pooled interaction of price with line is
+0.1465 at t 2.10, which clears a raw 0.05 but not its Bonferroni threshold
of 0.0063 and has an MDE of 0.1958. Suggestive, unresolved.

**FanDuel's receptions price is calibrated within about 2 points**, and every
band is negative, meaning overs hit less often than priced. Converting back
through the 5.66 percent hold, the under side lands 0.05 to 1.65 points
below break-even in every band. Real bias, dies to the vig.

**Realised ROI of price-extremity rules on receptions, all negative.** Every
over -0.0974, every under -0.0212 against a coin-flip baseline of about
-0.028. So the over bias is real and the under is merely average.

### THREE BUGS IN MY FIRST ATTEMPT, ALL WORTH KEEPING

**Bug A: average log loss cannot detect a pocket.** Planted a 6-point market
error in a 5 percent pocket defined by a feature interaction, which is
exactly the object the test existed to find. Boosting returned t -5.49 with
no pocket and t -5.45 with the pocket. Identical. The model's own variance
costs about 0.012 of log loss, which swamps the signal. **A script using
average log loss would have returned a confident negative that meant
nothing.**

The working metric is the **directional slope**: regress the outcome on the
model's disagreement with the book, controlling for the book. Noise
uncorrelated with the outcome contributes zero slope. Planted calibration:

```
no pocket        t +0.27
6-point pocket   t -0.07
12-point pocket  t +2.17
```

**Detection limit about 10 to 12 points. Profitability limit about 6 points,
because the hold is 5.7. There is a band of profitable-but-invisible pockets
and no method fixes it on this data.**

**Bug B: the training population.** `eval_harness.score_market` fits its
projection on the module's FULL dataset then joins to props for scoring. My
first version fitted on the JOINED frame, which conditions the training
population on a FanDuel prop existing, which selects for starters and volume.
The drift scaled with how much the join discards:

| market | joined / full | published | got |
|---|---|---|---|
| receptions | 7850 / 22540 | +0.226 | +0.241 |
| receiving | 8676 / 20884 | +0.066 | +0.073 |
| rushing | 2671 / 6356 | -0.043 | **+0.014 FAIL** |
| qb_passing | 1670 / 2688 | +0.085 | **+0.368 FAIL** |

Fixed, and all four then reproduced to three or four decimals. **This is a
new instance of bug pattern 1: a filter applied before something that must
not see it, here the training population conditioned on the book's decision
to post a line.**

**Bug C: nested walk-forward.** The challengers were fitted on the gate's
OUTPUT, which already excludes the first season, so the second season trained
on nothing and was never scored. Receiving reported 3,148 out-of-sample rows
instead of 5,939 and its 2025 model trained on 2024 alone. Fixed. **Print the
walk-forward plan, season by season with train and test counts. That would
have exposed this at a glance.**

**Also recorded: seed averaging is a no-op for
`HistGradientBoostingClassifier`.** Five seeds gave slopes identical to three
decimals, because it is deterministic unless rows are subsampled;
`random_state` only affects binning. Caught by the project's own "identical
when it should have moved" rule.

---

## PART 5. QB_PASSING, THE BEST-SUPPORTED RESULT IN THE PROJECT

`side_picker_search_v2.py`, all four gates exact. Challenger 3 is a
**logistic regression on `book_p` + `line_fanduel` + the existing five
features** (`attempts_roll`, `team_spread`, `total_line`, `wind_eff`,
`def_pass_roll`).

```
disagreement slope  +0.832   SE 0.312   t +2.67   p 0.0076
2024                +0.947   SE 0.513   t 1.85
2025                +0.734   SE 0.427   t 1.72
placebo             centred, mean +0.0056, t +0.26
dLogLoss            +0.00034  (positive)
HOLDOUT ROI         +0.1278   SE 0.0611   t +2.09   248 bets
```

Boosting on the same inputs gave t +1.02, and its log-loss difference was
negative. **So it is not the nonlinearity that helped, it is the target.**

**Feature leakage checked and absent.** `attempts_roll` and `def_pass_roll`
are both `shift(1).rolling(6)`. `def_pass_roll`'s fillna uses
`expanding().mean().shift(1)` rather than a full-sample mean, which the
module docstring records as a previously fixed leak. `wind_eff` is game
conditions, not a player result. I suspected receiving's `ypt_roll` of the
same thing and was wrong: it is `shift(1).ewm(...)` on `ypt_game` and only
ever sees prior games.

### THE VERIFICATION: IT SURVIVES THE TEST THAT KILLED RECEPTIONS

`qb_passing_verify.py`, four gates exact. Non-cumulative bins, no selection:

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

**With NO threshold selected**, game-clustered bootstrap at a fixed 0.040:
gap CI **[+2.61, +14.81]**, 99.7 percent positive. ROI CI
[-0.0000, +0.2328], 97.5 percent positive.

### CORRECTION: MY CAVEAT 1 WAS WRONG

**The first version of this handoff listed as caveat 1: "Season
concentration, the same pattern that killed receiving. At a fixed cutoff,
2024 gives +5.87 at t 1.39 and 2025 gives +14.90 at t 3.08."**

**I read two point estimates and called it concentration without testing
whether they differ.** `season_stability.py` tested it formally, with
receiving as a positive control.

```
qb_passing slope interaction, 2025 minus 2024
    -0.0312   SE 0.6453   t -0.05   p 0.9614   MDE 1.8068

qb_passing fixed-cutoff gap interaction, 2025 minus 2024
    +9.03 pp  SE 6.41     t +1.41   p 0.1589   MDE 17.96 pp

receiving slope interaction, 2025 minus 2024   [POSITIVE CONTROL]
    +0.9635   SE 0.4484   t +2.15   p 0.0316   MDE 1.2554
```

**The positive control works.** Receiving's heterogeneity is detected at
t +2.15, so the test is not simply underpowered everywhere.

**qb_passing shows NO detectable season heterogeneity.** The slope
interaction is -0.03, pointing the opposite way from the story I told. The
gap interaction is +9.03 with an MDE of 17.96, so that null is weak, but the
slope null is not: the point estimate is essentially zero with an MDE of 1.81
against slopes of 0.947 and 0.734.

**So receiving and qb_passing are genuinely different objects.** Receiving's
slope moves between seasons. qb_passing's does not. Setting receiving aside
was right, for a reason I had not actually established when I did it.

**This is the same failure mode the project's rules exist to catch.** Report
the minimum detectable effect beside every null applies to differences
between subgroups too, not only to single nulls. The estimator's own
calibration makes the point: at n 2,200 a planted interaction of +0.8 came
back at +0.595 with t +1.52, not significant, so a difference below roughly
1.1 is undetectable at that size and qb_passing has about half that n for
the seasons in question.

### H4 REFUTED MY MECHANICAL EXPLANATION, AND IN THE OPPOSITE DIRECTION

I predicted 2025 would show wider disagreement, which would explain a larger
tail gap through composition rather than through any change in the
relationship. The reverse is true:

```
qb_passing   season   rows   sd dis   p90 |dis|   rows over cut   share
               2024    532   0.0594      0.0870             154   28.9%
               2025    540   0.0490      0.0790              94   17.4%
               2026     64   0.0552      0.0898              17   26.6%
```

**2025 achieved a LARGER gap from a NARROWER and SMALLER tail.** That is not
composition. It is 94 rows landing well. Combined with the null interaction,
the honest reading is that the season difference is noise.

For contrast, receiving's disagreement collapses between seasons: sd 0.0361
to 0.0252, and rows over a 0.04 cut fall from 248 (8.9 percent) to 24 (0.9
percent). That market's model changed behaviour materially between seasons,
which is consistent with its detected heterogeneity.

### H5 IS THE FINDING WORTH CARRYING FORWARD, AND IT BEATS THE SLOPE

Octiles of disagreement against the realised outcome, all 1,136 rows:

```
octile   rows   mean dis   over rate   book_p   gap pp   SE pp
     0    142    -0.0679      0.4437   0.5002    -5.65    4.26
     1    142    -0.0248      0.4225   0.5000    -7.75    4.34
     2    142    -0.0010      0.4789   0.5000    -2.11    4.03
     3    142    +0.0172      0.5493   0.5000    +4.93    4.10
     4    142    +0.0333      0.5000   0.5000    +0.00    4.21
     5    142    +0.0492      0.5141   0.5000    +1.41    4.24
     6    142    +0.0662      0.5211   0.5000    +2.11    4.21
     7    142    +0.0975      0.5563   0.4999    +5.65    4.13
```

**The sign flips cleanly at zero disagreement.** Every negative-disagreement
octile is negative, every positive one is positive or zero. The extremes are
roughly symmetric, -7.75 and +5.65, which is what a directional signal looks
like rather than a one-sided artifact. And it is measured on the whole
population rather than on a 265-row tail.

Receiving's H5 is flatter and only the top octile separates (+3.66), which is
consistent with its heterogeneity and with setting it aside.

**Why this matters for what comes next.** The signal is in the DIRECTION of
disagreement across the whole distribution, not in a tail. That is an
argument for a well-calibrated probability rather than a threshold rule, and
it is exactly what a posterior predictive provides.

### WHAT STOPS QB_PASSING BEING ESTABLISHED. FOUR THINGS, ALL REAL.

1. **217 of 265 bets in the selected tail are OVERS, 82 percent.** The
   structural over-lean in this market is real but small: the zero-deviation
   check found an actual over rate of 0.5060 against a priced 0.4530, so
   about 5 points, not 8.88. So the model is choosing WHICH overs rather than
   just betting overs, but the skew is the filtered-sample signature.
2. **The ROI CI touches zero exactly**, [-0.0000, +0.2328].
3. **p 0.0076 does not clear the multiplicity threshold.** 5 challengers
   across 4 markets is 20 comparisons, giving 0.0025; counting this session's
   verification cuts as well gives 38 comparisons and 0.00132.
4. **The baseline is weak.** qb_passing is priced at flat odds both sides, so
   `book_p` is near-constant and the challenger only has to beat a base rate
   of 0.4982. This reads as "features predict over/under nonlinearly" rather
   than "we beat the book's price". Receptions faced a real forecast with a
   measured price slope of 1.008.

**And it contradicts a shipped decision.** qb_passing beta was +0.085 at
t +1.5, clipped to zero, and the market is displayed as reference-only. The
clip may have been right for a linear yards model and wrong as a verdict on
the market. Do NOT un-clip it on the strength of this: the beta clip governs
the mean, and this result is about a probability. They are different
parameters.

**Sample size: 1,136 out-of-sample rows across two scorable seasons.** 2026
adds roughly 550 rows this season.

---

## PART 6. NEXT SESSION: THE PRIORITY IS PICKING A SIDE

**This is the project's stated goal and it is not solved. Treat it as the
priority over cleanup, over documentation, and over further cuts of the
existing four seasons.** Of six live markets, zero currently support a
defensible side recommendation. Closing that gap is the whole point of the
app.

### 1. THE BAYESIAN MODEL. Promoted from untried to the top of the queue.

Never tested in this project. The record's "linear won" was about forecast
accuracy against boosting.

**Why it is different from anything tried.** Every model here produces a
point projection and `mc_pricing` then imposes a SEPARATE sigma fitted
independently. Phase 4.2 closed sigma as "not an improvement", but it only
ever compared hand-specified functions of the MEAN. A posterior predictive
gives mean and spread from one coherent object.

**Why the distribution reaches the SIDE decision, not just the magnitude.**
- **Skew flips sides.** `db._model_side` already documents it: under the
  gamma, when the line sits between the median and the mean,
  projection-versus-line says OVER while the probability says UNDER. A
  posterior predictive has different skew from a moment-matched gamma, so it
  flips individual props. This already caused one real bug (Juwan Johnson,
  proj 43.9, line 43.5, displayed OVER +8.5 when the model favoured the
  UNDER).
- **Spread decides which props clear the threshold.** Sigma does not change
  the favoured side when the mean is clearly off the line, but the threshold
  decision is entirely about magnitude.
- **It expresses something no current sigma can.** Sigma is a function of the
  mean only, so two players with the same projection get identical spread
  whether it rests on twelve games or two. A hierarchical posterior separates
  them automatically, and those thin-history players are exactly the obscure
  targets the strategy depends on.
- **And H5 above points the same way.** The qb_passing signal lives in the
  direction of disagreement across the whole distribution rather than in a
  tail, which argues for a calibrated probability rather than a cutoff.

**First version:** hierarchical negative binomial for receptions,
player-level random effects with partial pooling, shrinking thin-history
players toward the population instead of dropping them below the projection
floor. PyMC or Stan. Real work, slow fits, awkward to cache in Streamlit.
Scope deliberately and gate it on reproducing the existing calibration before
comparing.

### 2. THE CLASSIFICATION BAKE-OFF.

The original selection tested forecast accuracy on the statistic and has
never been run on the side-picking target. Per market, walk-forward, on log
loss AND realised ROI, with multiplicity counted ACROSS markets rather than
within them: logistic on features plus line plus book probability; gradient
boosting on the same; the current linear projection through `mc_pricing`; and
the Bayesian model.

**Prerequisite: calibrate whatever metric is used against a planted pocket
first.** Average log loss was demonstrated blind to a 6-point pocket in 5
percent of rows at this sample size.

### 3. Pre-registration plumbing, because 2026 is the only new data.

Whatever rule survives has to be recorded BEFORE outcomes exist.
1. Write the rule down as a constant, not a sweepable parameter. If it
   changes later, the count restarts.
2. Log every firing at capture time with projection, line, odds, edge, side
   and timestamp, and nothing about the outcome. Manual save-to-log will lose
   bets and lost bets are not random. Note the Line Movement save path
   currently writes zero rows, so this needs building rather than fixing.
3. Grade afterward, separately, never in the same write.

The concrete target for receptions is 70 more bets, about 29 weeks. For
qb_passing it is roughly 550 rows this season.

### 4. A deployed-version indicator. Plan 6.14.

A version or commit string in `app.py`, printed in the footer. The unpushed
commit was invisible to every verification script and cost a day of believing
a change was live.

### 5. The Line Movement save decision.

Live projection and edge inside `get_line_movement`, or remove the Save
button and Bet? column and fix the caption. Do not leave a button that
reports success while writing nothing.

### 6. Then the small items.

6.10's source column on `lines`, 6.11's hardcoded `GAMES_PLAYED = 0`,
6.12's dead fourth sigma in `mc.SIGMA_RULES`, 6.13's integer reception lines,
6.8's five `edge_threshold` versions, and a `.gitignore` for the loose CSVs
and JSON diagnostics. That last one matters more than it looks: 30 lines of
untracked noise in `git status` is what made the unpushed commit easy to
miss.

### Do not bother with

- **More cuts of the receptions 2.5 and 3.5 stratum.** About 88 already and
  no more data to buy.
- **Extending the backfill.** Player props start 2023-05-03.
- **Average log loss as a model-comparison metric at this sample size.**
  Demonstrated blind to a planted pocket.
- **Seed-averaging `HistGradientBoostingClassifier`.** Deterministic unless
  rows are subsampled.
- **Boosting for side decisions.** Lost to logistic on the same inputs in
  every market, with negative log-loss differences throughout.
- **Un-clipping qb_passing's beta on the strength of the logistic result.**
  Different parameter, different question.
- The sigma constant, the sigma floor, Phase 3.4, Phase 4.3, rewiring the
  harness, caliber or game-context tables, the alt ladder, line shopping,
  consensus settlement, BetRivers. All previously closed and none reopened.

---

## PART 7. MY ERRORS TODAY, RECORDED

Thirteen. Most were caught by tests written for the purpose, and three would
have produced confident wrong conclusions if the instruments had not been
calibrated against planted data first.

### Would have produced a wrong conclusion

1. **Test G's ROI sweep was ONE-SIDED.** Every cell backed the expensive
   side. The candidate rule backs the CHEAP side at plus money, so I never
   tested the direction the rule actually bets, then read the negatives as
   evidence against it.
2. **Test D pooled all line strata with no fixed effects.** Over rate falls
   monotonically with the line, so between-stratum covariance loaded onto
   `dev`. The placebo caught it at +0.00850, about 17 SEs above zero.
3. **Average log loss for pocket detection.** The planted test showed it
   blind. Would have returned a confident negative on the whole side-picker
   question.
4. **Trained the side-picker's projection on the joined frame**, conditioning
   the training population on a prop existing. Two of four gates failed.
5. **Nested the walk-forward**, so the second season trained on nothing and
   receiving was scored on two seasons rather than four.
6. **Claimed qb_passing's signal was concentrated in 2025** from two point
   estimates without testing the difference. The interaction is t -0.05.
   This one reached a pushed document.

### Wrong predictions, stated in advance and refuted

7. "The book term carries the candidate rule." Zero in both coordinate
   systems.
8. "The book term is largest in 2023." No season pattern at all.
9. "No plateau for qb_passing." There is a plateau.
10. "2025 has wider disagreement, explaining its larger gap." It is
    narrower, 0.0490 against 0.0594.
11. "Removing the top 5 winners cuts the candidate rule's ROI by more than
    half." It cuts it 31 percent, and a planted genuine edge of the same
    shape loses 31 percent too, so the test was not diagnostic as designed.

### Sloppiness

12. **Gave the wrong v5 commands three times**, running all markets and all
    lines instead of the candidate rule's population, then read the mismatch
    as a failure to reproduce.
13. **A gate that passed by 0.0001 on a nonsense comparison.** Passed
    `et5.GRID` (0 to 0.25, an EDGE grid) as the threshold grid for a DECIMAL
    ODDS column, so every row cleared it and the "price rule" was the entire
    population. It reported -0.0302 against a published -0.0401 and passed a
    0.010 tolerance by one ten-thousandth.

### And one claim accepted without checking

**"Linear beat Bayesian and XGBoost."** I repeated this back from the
conversation. Nothing in either document mentions a Bayesian model. Ted
caught it. The record says linear beat boosting on FORECAST ACCURACY, which
is a different question and does not include Bayesian at all.

---

## PART 8. WORKING PRINCIPLES

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
  different states. Every verification script here runs against local files
  and cannot see the difference. Check `git status -sb` for `ahead`, and put
  a version string on screen.
- **CALIBRATE THE INSTRUMENT AGAINST PLANTED DATA BEFORE TRUSTING A NULL.**
  Average log loss was demonstrably blind to the exact object it was built to
  find. A negative from an uncalibrated metric is worth nothing. Plant the
  effect, check the metric sees it, then run it for real. This applies to
  estimators too: the interaction estimator was calibrated at n 2,200 and
  found to need a difference above about 1.1.
- **A DIFFERENCE BETWEEN TWO SUBGROUPS NEEDS ITS OWN TEST AND ITS OWN MDE.**
  Comparing point estimates with intervals by eye is the error that reached a
  pushed document today. +5.87 against +14.90 is t +1.41.
- **USE A POSITIVE CONTROL WHEN TESTING FOR ABSENCE.** Receiving's known
  heterogeneity is what made qb_passing's null readable. Without it the null
  could have been an underpowered test rather than a real absence.
- **REPORT THE DETECTION LIMIT ALONGSIDE THE PROFITABILITY LIMIT.** This data
  detects pockets above roughly 10 points; pockets above 6 points would pay.
  The band between is real and invisible, and saying so is part of the result.
- **A PLATEAU IS EVIDENCE, A SPIKE IS SELECTION.** Fixed-cutoff neighbourhood
  tests with no argmax anywhere separate the two. This is what distinguished
  qb_passing from the receptions candidate rule.
- **NON-CUMULATIVE BINS, NOT CUMULATIVE ONES.** A cumulative curve is
  autocorrelated by construction and makes almost any pattern look smooth.
- **COUNT MULTIPLICITY ACROSS THE SESSION, NOT WITHIN THE TABLE.** Printing a
  per-market Bonferroni threshold while running twenty comparisons across
  four markets understates the look-elsewhere effect fourfold.
- **THE MODEL-SELECTION QUESTION DEPENDS ON THE TARGET.** Least squares on
  the statistic and a side decision at the line are different objects. A
  result established for one does not transfer.
- **CHECK A REMEMBERED PREMISE BEFORE REASONING FROM IT.** I accepted
  "Bayesian lost" from conversation and it is nowhere in the record.
- **PRINT THE WALK-FORWARD PLAN, season by season with train and test
  counts.** The nested-walk-forward bug would have been visible at a glance.
- **A ZERO-HIT `findstr` NEEDS A POSITIVE CONFIRMATION BESIDE IT.** Searching
  for `">= 1.5"` returned nothing because the serving gates read
  `min_targets=1.5`. Right answer, wrong reason.

---

## PART 9. THE HONEST FRAME

The morning's handoff described Phase 4.4 as shipped. It was committed and
not pushed, so the users were on pre-4.4 code, and no script in this project
could tell. That is the most important thing in this document, and what
caught it was one number that should have been 15.

On the substance, the day went the opposite way from the last two sessions.
Those retracted headline findings. This one spent itself failing to kill the
receptions candidate rule, ruling out six explanations in a row, and then
found the seventh: a discontinuity at the selected threshold with no
mechanism in the pricing layer to produce one. That rule is now weaker, and
precisely weaker, which is better than weak for unstated reasons.

And the side-picking question, which the project had effectively closed by
concluding that four of five markets carry no signal, turns out to have been
closed on the wrong test. Beta is a scalar fitted by least squares on the
statistic. A logistic on the same features, aimed at the side rather than the
mean, finds something in qb_passing that survives a plateau test, a
selection-free bootstrap, a formal season-heterogeneity test against a
working positive control, and a placebo, and that shows a monotone dose
response with a clean sign flip at zero disagreement. It is not established.
Its case rests on 1,136 out-of-sample rows, an 82 percent over skew in the
selected tail, and a baseline that was only a base rate.

**The priority for the next session is explicitly the side-picking problem,
and the Bayesian hierarchical model is now at the top of that queue.** Not
cleanup, not documentation, not more slices of the four seasons that are all
there will ever be. H5 says the qb_passing signal lives in the direction of
disagreement across the whole distribution rather than in a tail, which is an
argument for a calibrated predictive distribution rather than a threshold
rule. That is the one thing this project has never built.

Thirteen errors, six of which would have produced wrong conclusions, five
caught by instruments built to catch them and one caught by Ted after it
reached a pushed document. The apparatus is still the asset, and it is now
measurably better at catching its own operator than it was this morning.
