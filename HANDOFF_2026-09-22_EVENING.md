# OpalScales Handoff
### Tuesday, September 22, 2026, evening. Week 3 of the 2026 NFL season.

This **supersedes `HANDOFF_2026-09-22.md`** (the midday version). That document
is now wrong in about twenty places and should be deleted rather than kept
alongside this one. Read `OPALSCALES_PLAN.md` for the phased plan; this
document covers state, findings, and what to do next.

The short version: two survivorship filters were removing the edge-determining
rows from the EVALUATION sample, not just from the model. Fixing that took the
measured beta in two of four markets to zero. The project now knows which
market has signal, and it is only one of them.

---

## THE HEADLINE

Consensus closing lines, walk-forward, 25,482 joined rows, 863 game clusters,
four seasons:

| market | beta | SE | 95% CI | t(b=0) | t(b=1) | alpha | residSD |
|---|---|---|---|---|---|---|---|
| **receptions** | **0.275** | 0.029 | [+0.218, +0.332] | **+9.45** | -24.97 | +0.12 | 2.0 |
| receiving | 0.093 | 0.040 | [+0.014, +0.171] | +2.32 | -22.71 | +4.13 | 27.1 |
| qb_passing | 0.075 | 0.068 | [-0.058, +0.209] | +1.11 | -13.59 | +1.53 | 72.6 |
| rushing | **-0.032** | 0.044 | [-0.119, +0.056] | -0.71 | -23.21 | +3.15 | 29.7 |
| **pooled (sigma units)** | **0.172** | 0.023 | [+0.128, +0.217] | +7.56 | -36.34 | +0.10 | 1.0 |

Per season, pooled in sigma units: 2023 **0.184**, 2024 **0.143**,
2025 **0.213**, 2026 weeks 1-2 **0.015** (SE 0.103, too thin to read).

**THE POOLED NUMBER IS NOW MISLEADING AND SHOULD NOT BE USED IN PRICING.**
Three of four markets sit at zero. Receptions carries the pooled figure almost
single-handed. Phase 5.1 must apply **market-specific** betas, which means
rushing and qb_passing display no edge at all and receiving almost none.

At FanDuel rather than consensus, pooled is 0.141 (SE 0.025).

### What changed from the midday handoff, and why

Midday said pooled 0.273 with every market positive: receiving 0.160,
receptions 0.277, rushing 0.337, qb_passing 0.402. Three of those four numbers
were artifacts of the evaluation sample.

`eval_harness.score_market` called each module's `build_dataset()`.
`models.rushing.build_dataset` filtered `carries >= 5` and
`models.qb_passing.build_dataset` filtered `attempts >= 10`. Both are IN-GAME
results. FanDuel posts a line on every active back and every active QB before
kickoff, so the population the line was set for is every active player-week.
Scoring only the players who later got volume conditions the evaluation on the
single fact that most determines the outcome.

| market | beta before | beta after | filter removed |
|---|---|---|---|
| rushing | 0.337 | **-0.032** | `carries >= 5` |
| qb_passing | 0.402 | **0.075** | `attempts >= 10` |
| receiving | 0.160 | 0.093 | none (moved for other reasons) |
| receptions | 0.277 | 0.275 | none |

The two markets with a volume filter lost their beta entirely. The two without
one barely moved. That is as clean a natural experiment as this project is
going to get.

---

## ESTABLISHED FINDINGS (do not re-litigate)

### About the model

- **Receptions is the only market with signal.** Beta 0.275, SE 0.029,
  t +9.45. It is also the best-calibrated market (see the zero-deviation table
  below) and the one where FanDuel prices per-player odds rather than flat
  vig. Everything worth doing next runs through receptions.
- **Rushing carries no information.** Beta -0.032, and the leave-season-out
  estimates in `edge_threshold` came back at exactly 0.000 in all four seasons
  because the raw value was negative and the code clips at zero. Its previous
  0.337 was the `carries >= 5` filter. Rushing was already the weakest model
  (corr 0.341, within-tier as low as 0.015); the honest reading is zero.
- **qb_passing carries no information and blending actively hurts.** Beta
  0.075 (CI includes zero). `RMSE proj` is **78.73** against `RMSE line`
  **72.59**, so the projection is six yards worse than ignoring the model, and
  the blend at 72.67 is worse than the line too. This is the first market
  where the blend loses.
- **Receiving is near zero.** Beta 0.093, CI [+0.014, +0.171]. Technically
  excludes zero, practically worthless: a beta of 0.09 moves a blended mean
  about 1.6 yards on a sigma of 27.
- **The pricing assumption is rejected** at t(b=1) = -36.34 pooled. Displayed
  edges remain far too large.
- **The raw projection is worse than the line** as a mean estimate in every
  market, out of sample. Unchanged from the first day of this work.
- **More training data does not help.** `--fixed-train 2022` trained one model
  on 4,608 receiving rows and scored all seasons: betas 0.255 / 0.186 / 0.278,
  pooled 0.230, against walk-forward 0.229 with up to 20,313 rows. Receiving
  was marginally BETTER fixed. Sample size is not the binding constraint, so
  feature engineering and more seasons are both the wrong lever.
- **No overfitting.** In-sample pooled 0.241 against walk-forward 0.229, a gap
  of 0.012. The models are not fitting noise. (Measured before the filter
  fixes; worth one re-run but the conclusion is unlikely to move.)
- **Beta is stable across seasons**, 0.184 / 0.143 / 0.213, and the training
  counts confirm it is not a training-size effect (beta is not monotone in
  `n_train`; 2023 trains on one season and beats 2024 on two).
- **Early-season beta is HIGHER than full-season, not lower.** Weeks 1 and 2
  of 2023, 2024, 2025: 0.341, 0.229, 0.552, pooled **0.365** (SE 0.074)
  against a full-season 0.229. So there is no early-season weakness to
  correct, and 2026's 0.015 is 32 games of noise rather than a calendar
  effect. Speculative mechanism worth recording but not acting on: books have
  less in-season information to price in September, so nflverse priors are
  relatively more valuable then. Measured before the filter fixes, so worth a
  re-run.

### About the alpha finding (it is dead)

- **The alpha finding is closed.** Alpha is `mean(actual - line)`. Books charge
  flat -113 both sides on yardage, which asserts P(over) is about 0.5, so the
  line sits near the MEDIAN. For a right-skewed outcome the mean sits above
  the median, so a positive alpha is expected BY CONSTRUCTION and carries no
  betting information. Converting alpha to an over rate by dividing by sigma
  assumes symmetry, the one assumption that is false.
- **The empirical over rates settle it** (consensus lines, breakeven at -113
  is 0.5305):

| market | over rate | 95% CI | alpha |
|---|---|---|---|
| qb_passing | 0.4986 | [0.4735, 0.5236] | +0.23 |
| receiving | 0.4912 | [0.4813, 0.5012] | +4.52 |
| receptions | 0.4799 | [0.4699, 0.4899] | +0.13 |
| rushing | 0.4721 | [0.4549, 0.4893] | +3.11 |

  Receiving carries a +4.52 yard alpha and wins the over 49.1 percent of the
  time. A blind over on receiving loses about 6 percent of turnover. Plan item
  4.8 is closed.
- **Alpha remains a VALID PRICING parameter** for Phase 5.3. Setting the
  distribution mean to `line + alpha` is what puts the median near the line,
  which is what makes P(over) come out near 0.5. Valid for pricing, invalid as
  a signal. Do not conflate these again.
- Supporting evidence that it is skew: before the filter fixes, the four alphas
  in sigma units were rushing 0.131, receiving 0.118, receptions 0.021,
  qb_passing -0.022, which is monotone in how right-skewed each outcome is,
  with the near-symmetric market at zero.

### About shopping

- **The "best line" beta of 0.345 was mostly artifact.** `line_best` is
  selected USING the projection and then appears in both `dev` and `out`, so
  `Var(line_best)` enters the covariance. Placebo with 200 shuffles: observed
  0.3446, placebo mean 0.1882, **artifact-corrected 0.1564**, against a
  consensus beta of 0.273 on the same rows. Shopping adds NO forecast value.
  Strike the 0.345 rather than correcting it.
- **Shopping does add real PRICE value**, measured properly by holding the
  side fixed and counting realized win rate at each source. On 2025 alone
  (8 books, mean spread 2.09): receiving +3.2 pp, rushing +3.1 pp, receptions
  +1.8 pp, qb_passing +1.7 pp.
- **Do not use the all-seasons shopping numbers.** That run reports **11
  books** and a mean spread of **4.67**, against 2025's 8 books and 2.09. The
  extra books in 2023 and 2024 widen max-minus-min, so the pooled +0.04 to
  +0.057 figures are not capturable today. Use 2025.
- Caveat on all shopping numbers: `best` spans every book including stale and
  limited ones (bovada at 46.5 against a 50.5 consensus in the worked Kelce
  example), so part of the gain is not available to you. Restrict to books you
  hold accounts with to size it honestly.

### About the pricing layer

- **The gamma is OVER-skewed; the negative binomial is exact.** At zero
  deviation, where the model contributes nothing and only the skew correction
  is active:

| market | priced P(over) | actual over | gap | z |
|---|---|---|---|---|
| qb_passing | 0.4645 | 0.5060 | +4.1 pp | +3.25 |
| receiving | 0.4541 | 0.4897 | +3.6 pp | +6.55 |
| rushing | 0.4566 | 0.4702 | +1.4 pp | +1.53 |
| **receptions** | **0.4724** | **0.4718** | **-0.1 pp** | **-0.10** |

  Every gamma market under-prices the over before the model speaks. The count
  market is exact to one tenth of a point. The fix is a one-parameter median
  calibration: choose alpha so priced P(over) at zero deviation matches the
  measured over rate, rather than reading alpha off the regression's mean
  offset.
- **Sigma must be LEVEL-DEPENDENT. The shipped functional form was right.**
  The midday recommendation to replace it with the harness residual SD was
  wrong. A flat sigma per market applied to a heteroskedastic outcome produced
  monotonically worsening overconfidence, -17.4 pp at the top band. Fitted
  exponents against the shipped ones:

| market | fitted b | shipped |
|---|---|---|
| receiving | 0.54 to 0.57 | 0.6172 |
| rushing | 0.48 to 0.53 | 0.7149 |
| receptions | 0.47 to 0.53 | linear in proj |
| qb_passing | unidentifiable | 0.3531 * proj |

  What should change is the ANCHOR, from the projection to the blended mean,
  not the shape.
- **qb_passing cannot support a power-law sigma.** Every QB projects into a
  narrow 180 to 300 band, so `log(mean)` barely varies and the fit is
  unidentified: it returned a = 55.3, 127.7, 27.4, 66.7 with b near zero
  across four seasons, and in 2024 assigned sigma 127.7 against a flat 70.
  `edge_threshold.fit_sigma_level` now gates on the top bucket being at least
  3x the bottom and falls back to flat, which prints a reason.
- **The pricing layer is over-dispersed, not biased.** Unconditional
  calibration (no side selection) is nearly unbiased overall at +1.1 pp, but
  low probabilities are too low and high probabilities too high:
  0.00-0.35 **+7.3 pp**, 0.35-0.45 +2.6, 0.45-0.50 +1.1, 0.50-0.55 -0.3,
  0.55-0.65 -4.7, 0.65-1.00 **-11.6**. Every predicted probability needs
  pulling toward 0.5.
- **The winner's curse from side selection is measured at about 4.3 points of
  win probability, which is larger than the vig.** Conditional calibration
  (picking the better of two edges) shows an overall gap near -3.2 pp against
  the unconditional +1.1 pp. At -113 each point is worth roughly 1.9 percent
  of turnover, so side selection costs about 8 percent against a house edge
  of about 6. This is the quantitative explanation for "tier size is
  anti-predictive as displayed": the larger the displayed edge, the more of it
  is estimation error, and choosing the favoured side is choosing the noisier
  estimate.
- **65.3 percent of props still show a positive edge after shrinkage.**
  Against a 6 percent hold a calibrated layer should find positive edge on a
  small minority. This figure is itself proof the layer is not yet calibrated.

### About the edge threshold (item 2, and the answer is no)

- **No edge threshold is established.** Pooled holdout ROI **+0.0183**, SE
  0.0741, t +0.25, on 235 bets, where the threshold is chosen on other seasons
  and spent on the held-out one. Per market:

| market | bets | win rate | ROI | t |
|---|---|---|---|---|
| receiving | 239 | 0.5732 | +0.0940 | +1.47 |
| rushing | 291 | 0.5533 | +0.0561 | +1.04 |
| receptions | 261 | 0.4751 | +0.0489 | +0.68 |
| qb_passing | 329 | 0.5380 | +0.0247 | +0.45 |

- **The receiving +0.0940 is weaker than it looks.** The threshold chosen from
  other seasons left only 4 bets in 2025 and 2 in 2026, so the whole figure
  rests on 2023 and 2024. The edge distribution is not stable enough across
  seasons for a fixed cutoff to fire consistently.
- The in-sample "highest ROI" rows in every threshold curve are selected
  across fifty cutoffs and biased upward. Only the holdout counts.
- **J1 stands, now on measurement rather than on a calibration curiosity.**
  Do not bet on displayed edges as probabilities.

### About the data

- **Lead-time contamination is real but irrelevant. DO NOT spend the 1,900
  credits re-pulling.** Filtering to a 20 to 90 minute window keeps 96 percent
  of props and changes pooled beta from 0.229 to 0.229, with per-season shifts
  of 0.004 or less and over rates moving under 0.002. Gate says no.
- **The mis-timed snapshots are fully explained.** The closing timestamp was
  computed as placeholder kickoff minus 45 minutes using the SEPTEMBER
  enumeration, while `commence_time` stores the real kickoff. Convert the
  offending `captured_at` values to Eastern and they collapse onto two
  numbers: eleven games at lead 249.4 all kick at 4:25pm ET and were all
  captured at **12:15pm ET**; four games at 484.4 kick at 8:20pm ET Sunday and
  were also captured at 12:15pm ET. 12:15pm ET is 45 minutes before a 1:00pm
  kickoff. The negatives are the mirror image: a game that kicked at 1:02pm
  was captured at 3:40pm ET, which is 45 minutes before a 4:25pm slot. Affected
  weeks are exactly the ones whose kickoff times are unknown in September:
  week 18, the playoff weeks, and the flex-eligible weeks 14 through 17.
- Counts: **5 events captured AFTER kickoff** (719 rows, in-play prices) and
  **33 events over 90 minutes** (12,047 rows, genuine pre-game prices, just
  not closes). Out of 833 events.
- **Mid-season enumeration is week-scoped for free.** The September 1 call
  returned all 272 regular-season games; a September 15 call returned only 16.
  The full-season overshoot comes entirely from starting the date walk in
  September. Starting later gives correct kickoff times without a fix.
- **Events come back chronologically.** Confirmed by running
  `--limit-events 2` on 2026, which returned New England at Seattle and San
  Francisco at Los Angeles, both week 1, both with full 8-book closing data.
  An unplayed game would have returned nothing.
- **`backfill_ledger.json` records completed WRITES, not enumeration.** The
  30-event run correctly skipped the 2 already recorded.
- **Credits are charged per market that RETURNS data, not per market
  requested.** A Week 3 live capture estimated 80 and was charged 24, because
  only about 1.6 markets per game had posted. Do not budget from the estimate.

### About the join (and what the missing rows are)

Overall join rate is now **88.1 percent**, up from 86.0.

| market | props | joined | rate |
|---|---|---|---|
| qb_passing | 1791 | 1732 | 96.7% |
| receptions | 10584 | 10020 | 94.7% |
| receiving | 11333 | 10636 | 93.8% |
| rushing | 5211 | 3089 | **59.3%** |

- **Rushing's 59.3 percent is the qb_rushing label bug, confirmed outright.**
  Every one of the top ten unjoined rushing names is a quarterback: Josh Allen
  57, Baker Mayfield 55, Jalen Hurts 54, Patrick Mahomes 54, Jordan Love 52,
  Lamar Jackson 50, Jared Goff 50, Geno Smith 49, C.J. Stroud 48, Brock Purdy
  46. `player_rush_yds` covers both, those rows store as `market = 'rushing'`,
  and `models.rushing` filters to RBs. That is Phase 3.7, not absent stat rows.
- **Concrete name-normalisation gaps**, worth 1 to 2 points of join rate:
  parenthetical team tags (`Trey McBride (Ari)`, `Lamar Jackson (BAL)`,
  `Michael Thomas (NO)`, `Michael (Saints) Thomas`), initial forms
  (`A. Trautman`, `J. Hill`, `M. Jones Jr.`), nickname aliases
  (`Joshua Palmer` against Josh Palmer, `Cameron Ward` against Cam Ward,
  `Hollywood Brown` against Marquise Brown), and dotted variants
  (`C.J. Uzomah` against CJ Uzomah, `CJ Stroud` against C.J. Stroud).
- **Kyle Juszczyk (32 unjoined) is a fullback**, so he is outside the
  `WR / TE / RB` position filter in receiving and receptions. Structural, not a
  bug, unless you want FB receiving props.
- Join rate is now the primary survivorship tell. A market whose join rate is
  well below the others is a market whose evaluation sample is conditioned on
  something.

### Rejected and closed (cumulative)

XGBoost; the qb_passing feature hunt (two rounds); early-season hierarchical
shrinkage (all 56 variants worse); postseason exclusion from rolling windows;
the qb_rushing quintile-4 defect; home/road splits; a standing parallel
Bayesian system; the weeks-1-to-4 correction; anytime TD variant C (training
on every active game overcorrected because the training base rate was lower
than the served one).

**Closed this session:** the alpha betting edge (plan 4.8); the best-line beta
as a forecast gain; the lead-time re-pull; "use the harness residual SD as
sigma" (wrong functional form); the RMSE comparison as independent
corroboration of beta (it is circular and algebraically implied by beta and
R-squared); rushing as a signal-bearing market; qb_passing as a
signal-bearing market.

---

## WHAT WAS BUILT AND FIXED THIS SESSION

### Data added

2026 weeks 1 and 2 backfilled at closing, 8 books, about 1,700 credits.
Verified on the same instrument as the earlier seasons: lead times 44.4 to
49.4 minutes, mean 46.8, against 2025's mean of 54.8. `fill_weeks.py` matched
32 of 32 events, weeks 1 and 2, 16 events each, 6,306 and 6,369 rows.

| season | events | rows | instrument |
|---|---|---|---|
| 2023 | 281 | 123,543 | closing, 8 books |
| 2024 | 273 | 102,682 | closing, 8 books |
| 2025 | 279 | 115,654 | closing, 8 books |
| 2026 wk 1-2 | 32 | 12,675 | closing, 8 books |
| **total** | **865** | **354,554** | |

Credits remaining: approximately **56,700**.

Why it was worth it: 2026 weeks 1 and 2 are the same games, the same outcomes
and the same calendar position as the 596 self-selected FanDuel rows that
produced Monday's beta of -0.079, but measured on the unselected instrument.
That is the cleanest available test of the selection hypothesis. Still
underpowered at 32 events, and it resolves on its own as 2026 accumulates.

### `models/rushing.py` rewritten (Phase 3.1, plus two more instances)

1. `carries_roll` was computed AFTER the `carries >= 5` filter. Rolling
   features now come from a new `build_all_rows()` with no volume filter.
   Measured inflation, which reproduces the midday figures exactly:
   **+6.31** carries in the 0-2 bucket, +3.81 at 2-5, +1.81 at 5-10, +0.29 at
   10-15, **+0.06** at 15+.
2. `load_model` trained on `carries_roll >= 5` while `project_week` served at
   **1.5**. Both now read `MIN_CARRIES_ROLL = 1.5`. Training population mean
   rushing yards: old gate **57.2**, unconditional **32.5**, new gate
   **39.1**. Deliberately between, matching the served population rather than
   going fully unconditional.
3. `build_upcoming_week` bridged from `build_dataset()`, so every Week 1
   projection inherited the inflation. It now bridges from `build_all_rows()`.
4. Null `rushing_yards` for an active RB is a genuine zero and was being
   dropped by `dropna`. Now filled.
5. `load_model` hardcoded `season <= 2024`, which blocked walk-forward
   scoring. Now `load_model(train_max_season, train_min_carries_roll)`.

`build_dataset` still applies `carries >= 5`, so the board, `all_players` and
`player_history` are unchanged. Served players went from 102 to **127**,
gained 25, **lost 0**, so the volume threshold did not cut off the obscure
low-line backs the strategy targets. New model: intercept **-0.14**, coefs
`carries_roll` 3.794, `team_spread` 0.422, `total_line` 0.098. A back with no
expected carries now projects near zero rather than inheriting an inflated
constant.

Unfiltered RB rows 6,635 across 279 players, against 3,758 and 219 filtered.
The filter was removing **43 percent** of RB player-weeks, and 60 players
existed only in the removed portion.

### `models/qb_passing.py` rewritten (Phase 3.3, plus a leak)

Same four fixes as rushing, with `MIN_ATTEMPTS_ROLL = 1.5`, plus:

5. **A full-sample statistic leak.**
   `qb["def_pass_roll"].fillna(qb["def_pass_roll"].mean())` took the mean over
   all five seasons, so a 2023 row was filled with a number computed partly
   from 2025. `eval_harness` walk-forwards the model COEFFICIENTS but cannot
   undo a leak baked into a feature. Now an expanding mean in chronological
   order, verified to use only prior rows. Residual caveat: the earliest 2022
   rows have no prior and fall back to the first available expanding value,
   which cannot affect 2023 onward.
6. `def_pass_roll` is now computed from ALL QB rows rather than only 10-plus
   attempt rows, so a defence that faced a QB who left after 8 attempts is
   credited with those yards.
7. `actual_result` read `build_dataset()`, so a QB who threw 9 attempts had no
   row and the bet silently never graded. Those are benched, injured and
   blowout games, which are disproportionately UNDERS, so the missing grades
   flattered the record. Now reads unfiltered stats with a normalised name
   fallback, and treats a null passing line for an active QB as zero.

Variant C check passed: new train gate mean **191.5** passing yards against an
unconditional **185.5** and the old gate's **221.0**. Six yards above
unconditional, same shape as rushing. `MIN_ATTEMPTS_ROLL = 1.5` stands.

### `eval_harness.py` rewritten (now 1,112 lines)

Nine changes, each with a stated reason in the module docstring:

1. **Empirical over rate**, counted with a game-clustered SE, replacing the
   inference from alpha. This is what killed the alpha finding.
2. **Out-of-sample blend RMSE**, beta fitted by 5-fold CV grouped on
   `event_id`. The old version fitted beta on the same rows it scored, and
   beta = 0 is nested in that fit, so the blend essentially could not lose.
   The reported margin was algebraically implied by beta and R-squared, not
   independent confirmation.
3. **Per-season pooling in sigma units.** The old per-season rows pooled
   markets in raw units, so passing yards (sd 70) swamped receptions (sd 2)
   and the coefficient was meaningless. Scales are computed once per market
   across all seasons so season differences are not driven by changing scales.
4. **Training counts** (`n_train`, `train_seasons`) carried through the join
   and printed, plus `--fixed-train YEAR` to hold the model constant across
   seasons. Separates instability from training-size effects.
5. **Honest in-sample comparison.** The old version hardcoded
   `season <= 2024`, so on `--all-seasons` it trained on three seasons for
   2023 against one for walk-forward and the gap mixed the leak with a 3x size
   difference. Now `<= s` against `< s`.
6. **`--lead-window min,max`** to gate the mis-timed snapshots.
7. **`--placebo N`** for the best-line beta. **Important implementation note:
   the permutation is of the DEVIATION, not the projection.** Permuting the
   projection breaks the pairing between a projection and its own line, so
   placebo deviations come out several times wider than real ones and the
   placebo UNDER-states the artifact badly. Verified on synthetic data with a
   known-zero signal: permuting the projection recovered only +0.014 of a
   +0.103 artifact, permuting the deviation recovered +0.093.
8. **Shopping as a realized win rate** with the side held fixed, replacing the
   `line_best` beta, which has a shared-term problem and measures the wrong
   thing anyway. Shopping improves the price, not the forecast.
9. **`--cache path.parquet`**, which appends only missing seasons. 354,554
   rows at 1,000 a page is 355 round trips and Phase 3 runs this constantly.

Plus: `--score-population {all,board}` (see below), a join diagnostic with the
most frequent unjoined names per market, 2026 added to `ALL_SEASONS`, and
`qb_rushing` removed from the default market list because there are zero rows
to fetch under that label.

**`--score-population` is the fix that mattered most.** Default `all` scores
`build_all_rows()` where a module exposes it, otherwise `build_dataset()`.
This is the EVALUATION population, and it matters as much as the training one.
Fixing the model's features and training gate did NOT fix the measurement,
because `build_dataset` still decided which rows got scored and joined. Pass
`board` to reproduce the old numbers.

### `edge_threshold.py` written (new, 896 lines)

Prices every prop through the corrected Phase 5.1 to 5.3 pipeline and asks
where a shrunk edge clears the vig.

- Blended mean `line_consensus + alpha + beta * (projection - line_consensus)`
- Gamma for yardage, negative binomial (or Poisson where underdispersed) for
  receptions, moment-matched
- P(over) evaluated at the line you would actually bet
- Edge is `P(model) - P(breakeven)` using the REAL FanDuel American odds, which
  matters most in receptions where breakevens run 0.417 to 0.610
- Realized profit per unit staked

Three defences against lying to yourself, each of which caught something:

- **Leave-season-out shrinkage.** Fitting beta and alpha on the rows being
  priced makes the tail look good by construction.
- **Holdout threshold selection.** The threshold is chosen on other seasons
  and spent on the held-out one. The in-sample best is biased upward.
- **Tail profile.** Survivorship does not show up as a calibration failure,
  it shows up as profit. What detects it is the composition of the tail.

Plus the unconditional calibration table (which isolated the winner's curse),
the zero-deviation pricing check (which found the gamma over-skew), the sigma
multiplier scan, and a per-market holdout.

### `.github/workflows/capture-lines.yml` written (NOT YET COMMITTED)

Runs `live_capture.py` on a clock: every 4 hours daily, plus Sunday 12:23pm
and 3:23pm ET, plus 7:23pm ET Thursday, Sunday and Monday for the prime-time
closes. About 47 runs a week, under 15,000 credits for the rest of the
subscription window. Five repo secrets required: `ODDS_API_KEY`,
`SUPABASE_URL`, `SUPABASE_KEY`, `OPAL_EMAIL`, `OPAL_PASSWORD`.

---

## THE BUG PATTERN, STATED ONCE SO IT IS NOT REDISCOVERED

Every significant bug found today is the same shape: **a filter on an in-game
result applied before something that should not see it.** Four places it can
bite, and all four have now been seen:

1. **Before a rolling feature.** The feature is measured only over the games
   the player qualified, so a backup's workload reads like a starter's.
2. **On the training population.** The model is fitted on survivors and then
   asked to score everyone, which is extrapolation below the training range.
3. **On the grading path.** The bet never grades, and the missing grades are
   disproportionately losses, which flatters the record.
4. **On the EVALUATION population.** This is the one that was missed all day.
   The measured beta, over rate, ROI and calibration are computed on a sample
   conditioned on the outcome. Fixing 1 through 3 does not fix 4.

The tell for 4 is the **join rate**, not the calibration table. Survivorship
leaves calibration looking fine, because alpha absorbs the shift: a filtered
market prices P(over) at 0.61 and the filtered sample delivers 0.609. It is
calibrated to a population that is not the one the line was set for.

Demonstrated on synthetic data: a market with **zero signal by construction**,
with a filter removing only 5.4 percent of rows non-randomly, produced a
threshold curve from +0.073 to +0.163 with t-statistics between +9.8 and
+12.9. Rushing lost 52 percent of its props and showed +0.082 rising to +0.65.

**Audit rule going forward: any `build_dataset` that filters on an in-game
statistic needs a `build_all_rows` sibling, and the harness must score the
sibling.**

---

## CURRENT MODEL STATUS, PLAINLY

| market | beta | filter in build_dataset | instrument | verdict |
|---|---|---|---|---|
| receptions | **0.275** | none | always honest | **the only real market** |
| receiving | 0.093 | none | always honest | near zero, blend adds 0.4 yds on 27.5 |
| qb_passing | 0.075 | `attempts >= 10` | **fixed today** | zero, and the blend hurts |
| rushing | -0.032 | `carries >= 5` | **fixed today** | zero |
| qb_rushing | untested | none | blocked by label bug | corr 0.510, never graded |
| anytime_td | n/a | `touches >= 3` | fixed previously | log loss 0.5452, AUC 0.654 |

- **receptions** corr 0.459, MAE 1.67. Beta 0.275, the most precise estimate
  in the project. Alpha +0.12, essentially zero, consistent with the negative
  binomial being correctly skewed. FanDuel uses per-player odds here
  (breakeven 0.417 to 0.610, about 6.9 percent hold) rather than flat vig.
  Unfixed: WR receptions bias +0.30 (z +5.39) on 1,456 rows, the largest
  measured effect in the project, plus V4 high-volume +0.32 (z +3.15).
  **Unfixed and now higher priority: trains on `targets_roll >= 3`, serves at
  1.5.**
- **receiving** corr 0.487, MAE 22.79. Beta 0.093. Alpha +4.13 and an over
  rate of 0.4912, so the alpha is skew. Unfixed: RB receiving bias +3.86
  (z +3.59), which decomposes entirely into yards per reception rather than
  volume. **Same train/serve gate mismatch as receptions.**
- **qb_passing** Beta 0.075, `RMSE proj` 78.73 against `RMSE line` 72.59. The
  +8.81 bias that appeared in all sixteen subgroup cells was very likely the
  `attempts >= 10` filter, which is now out of the features, the training
  population and the grading path.
- **rushing** Beta -0.032. Weakest model in the project and now measured at
  zero. Sigma `1.922*proj^0.7149` and floor 34.5 were fitted against the OLD
  projections and are **invalid**. That is Phase 3.2 and it is not optional.
- **qb_rushing** corr 0.510, highest of any market, essentially unbiased, and
  **zero graded rows ever**. Two independent causes: the `MARKET_MAP` label
  bug, and `actual_result` returning `None` when `rushing_yards` is null where
  zero is correct for an active QB.

### Week 1 and 2 betting record (context, not evidence)

Week 1: 7-5, +0.69 units, +6.02 percent ROI. By market qb_passing +4.43,
rushing +0.88, qb_rushing +0.44, receiving -4.06, receptions -1.00. A boosted
Maye bet at +133 returned +2.66 units, most of the profit. **The profit is not
evidence of skill**, and given what is now known about qb_passing and rushing
carrying zero signal, the two markets that produced the profit were the two
whose apparent edge was an artifact.

---

## NEXT SESSION: START HERE

### 0. Fire the GitHub Actions workflow. Time-sensitive.

Still not done, and Thursday Night Football is now a day away. Commit
`.github/workflows/capture-lines.yml`, add the five repo secrets, then fire it
manually from the Actions tab before trusting the cron. Confirm the job goes
green and a new `captured_at` appears for week 3. Opening lines are the only
perishable thing in this project.

### 1. The `targets_roll >= 3` versus 1.5 gate in receiving and receptions

Highest-value remaining model fix, because it is the same population mismatch
that erased rushing's and qb_passing's betas, and it is sitting in **the one
market that has signal**. `load_model` trains on `targets_roll >= 3` while
`project_week` serves at 1.5.

This one needs more care than the previous two. Receptions is the only market
worth anything, so a careless change could destroy the single real finding.
Both modules already have honest evaluation populations (no volume filter in
`build_dataset`, confirmed at 22,540 rows each), so ONLY the training gate
moves. Run `eval_harness` before and after and keep the change only if beta
holds or improves.

### 2. Phase 3.2, refit the rushing sigma and floor

Non-optional consequence of 3.1. `1.922*proj^0.7149` and floor 34.5 were
fitted against projections that no longer exist. Until this lands, rushing is
mispriced in the app. Given beta is now zero, the correct display for rushing
may simply be no edge at all.

### 3. Phase 3.7, the qb_rushing label bug, plus its grading bug

Unlocks a market with the highest correlation in the project (0.510) that has
never graded a row. Two fixes: `MARKET_MAP` in `import_lines` has no
`qb_rushing` key so QB rushing props store as `market = 'rushing'`, and
`qb_rushing.actual_result` returns `None` where zero is correct. Solve the
label once at analysis time by splitting on position from nflverse after the
join, rather than at fetch time, since the API data has the same issue.

This is also worth 40 percent of rushing's join rate.

### 4. Name normalisation

The specific gaps are listed under "About the join" above. Worth 1 to 2 points
of join rate across three markets, and join rate is now the survivorship tell,
so it is worth more than it looks. Consolidating `db._norm_name` and
`data_utils.norm_join_name` is Phase 6.1 and should happen in the same pass so
the rules cannot drift.

### 5. Median-calibrated alpha

One parameter, measurable, and it removes a 3.6 to 4.1 point structural lean
from every gamma prop. Choose alpha so the priced P(over) at zero deviation
matches the measured over rate, instead of reading it off the regression mean
offset. `edge_threshold.zero_dev_check` already reports the gap to target.

### 6. Sigma as a function of level AND deviation

The over-dispersion finding. Sigma is currently a function of the projection
level only. The conditional variance almost certainly also rises with
`|projection - line|`, because large disagreements happen when something
unusual is going on: injuries, role changes, weather. Rows with large `|dev|`
are getting distributions that are too narrow, and those rows populate the top
and bottom calibration bands. Fit sigma on both and re-check the
unconditional table.

### 7. Then Phase 5, pricing off the blended mean, with market-specific betas

`line + alpha + beta * (projection - line)` as the distribution mean, with
**per-market** beta, not pooled. Pooled 0.172 would display phantom edges in
three markets. At the measured values, rushing and qb_passing should show
nothing.

### 8. The line-as-feature model (plan 4.3) is now the main strategic lever

`actual = alpha + gamma*line + sum(delta_i * feature_i)`. The reason this
moved up: `--fixed-train` showed more data does not help, so the constraint is
which features carry information the market misses, not how much data is
behind them. Delta tells them apart by construction. Fit it AFTER items 1
through 4 so the features being tested are the honest versions.

### Do not bother with

- Re-pulling the 38 mis-timed snapshots. Gated and irrelevant.
- More training seasons or more feature engineering on public nflverse data
  before item 8. Measured not to help.
- Monte Carlo. Still premature for marginal single-stat distributions.

---

## OPERATIONAL NOTES

### Environment

- **Python 3.14.6 in `.venv`** at the repo root. `.venv/` is gitignored.
- pandas reported as 2.3.3 in one run and 3.0.6 in the install log. Worth
  pinning. Deprecation warnings on bare-integer timedelta comparisons at
  `live_capture.py:224` and `fill_weeks.py:194`. Cosmetic today, but the
  Actions runner installs fresh pandas on every run, so a version bump turns
  a warning into a scheduled job that fails at 3am on a Sunday.
- Ted works in VS Code on Windows, **cmd only, never PowerShell**. One command
  per line, no multi-line `python -c`.
- **`models/` has no `__init__.py`.** Imports work because Python treats it as
  a namespace package. Nothing is broken, but an empty `__init__.py` removes a
  class of surprise, since namespace packages resolve differently.
- **`nflreadpy` has no `get_config()`**, so the cache mode cannot be read that
  way.
- Two passwords exist and are easy to confuse. The Supabase dashboard uses
  GitHub OAuth and has no API password. The scripts need the **OpalScales app
  password** for `avoliotg@gmail.com`, which lives in `auth.users`.
  `odds_backfill.py` prompts for "Supabase password" but means the app
  password, and reads `ADMIN_EMAIL` where other scripts read `OPAL_EMAIL`.
- `.streamlit/secrets.toml` holds `ODDS_API_KEY`, `OPAL_EMAIL`,
  `OPAL_PASSWORD` and the Supabase values. Gitignored, verified.
- `live_capture.py` reads env vars with precedence over `secrets.toml`
  (line 84), and `sys.exit(1)` on a missing secret (line 88). Both are what
  make GitHub Actions viable and safe.

### Gotchas encountered this session

- **A 15-minute apparent hang was a cold nflreadpy fetch, not a bug.** Once
  warm the same call ran in 2.2 seconds. Ctrl+C returned to the prompt with no
  traceback, which is the signature of a block in a C-level socket call where
  the interrupt cannot be delivered to Python. When something appears hung,
  time each step separately with `flush=True` before theorising.
- **`ast.parse` proves a file is valid Python, not that it is the file you
  meant.** A stale `models/rushing.py` passed the parse check and then failed
  on a missing attribute. **Grep for a symbol you know is new** when swapping
  a whole file.
- **Browser downloads append a suffix, and the numbers do not track
  recency across files.** `eval_harness (2).py` was current while
  `edge_threshold (2).py` was three versions stale. **Sort by date, not by
  number:** `dir /o-d "%USERPROFILE%\Downloads\name*.py"`. Then verify with a
  line count and a grep before running.
- **Nested `st.cache_data` works fine outside a Streamlit runtime.** The
  `MemoryCacheStorageManager` and `missing ScriptRunContext` warnings are
  noise in bare mode.
- Argparse will raise on a duplicate option, so a patch that adds the same
  flag twice fails at startup rather than at the point of use.
- `getpass` shows nothing as you type; right-click pastes in cmd, and a new
  paste appends rather than overwrites, so Ctrl+C and restart is cleaner.
- Historical endpoints are paid-plan only, including the historical EVENTS
  endpoint, so a free dry run of the backfill is impossible.
- `fill_weeks.py` correctly skips already-filled events, so the 833 existing
  ones cost nothing on a 2026 run.

### Working principles (Ted's, earned)

- **Hard reboot before concluding a change did not work.** Streamlit caches
  `build_dataset` and `load_model`.
- **Always `encoding='utf-8'`** in parse checks, and parse-check before
  pushing.
- **Never use em dashes** in any writing. Firm.
- **Test, do not assume.** Silent failures have consistently been more
  dangerous than noisy ones.
- **Gate before sweeping.** Measure whether an effect exists before tuning its
  parameter. This session it killed the lead-time re-pull, the alpha edge, and
  the best-line shopping beta, and it caught a +0.46 ROI that was entirely an
  artifact within ten minutes of its appearing.
- **A t-statistic of +6 is a bug until proven otherwise.** Twice today a
  result appeared that would have been the best finding in the project, and
  both times it was survivorship. The instinct to check the composition of the
  winning bets rather than celebrate the number is what saved it.
- Ted wants plain-English reasoning before design decisions, pushes back when
  framing feels off, and is frequently right. He values being corrected over
  validated.
- Writes handoff docs at context limits. On mobile, prefers whole-file
  replacements over surgical edits.

### Claude's mistakes this session, recorded so they are not repeated

- Recommended replacing the shipped level-dependent sigma with a flat residual
  SD. Wrong: the functional form was right, only the anchor needed moving.
- Fixed the rushing model and claimed the harness numbers would move, without
  noticing that `score_market` still read the filtered `build_dataset`. The
  predictions failed for that reason. Model population and evaluation
  population are different things.
- Designed the best-line placebo to permute the projection, which understated
  the artifact sevenfold. Caught only because it was smoke-tested against a
  known-zero signal.
- Hypothesised that receiving's tail came from alpha divided by a small sigma
  on low-line props. Disproved by direct computation: at zero deviation the
  gamma's skew more than offsets alpha at every line level, so receiving's
  over edge is negative everywhere.
- Overwrote the working rewritten harness with the original upload via a
  careless `cp`. Recovered from `outputs/`. Check what a copy is about to
  clobber.

### Files and modules

- `models/`: `receiving.py`, `receptions.py`, `rushing.py` (rewritten),
  `qb_passing.py` (rewritten), `qb_rushing.py`, `anytime_td.py`,
  `data_utils.py`. No `__init__.py`.
- Root: `db.py`, `mc.py`, `mc_pricing.py`, `betlog.py`, `game_export.py`,
  `lm_export.py`, `app.py`
- Diagnostics and scripts: `eval_harness.py` (rewritten),
  `edge_threshold.py` (new), `odds_api_check.py`, `odds_backfill.py`,
  `live_capture.py`, `fill_weeks.py`, `beta_check.py`, `clv_check.py`,
  `td_reversion_check.py`, `rushing_filter_diag.py`, `td_backtest.py`,
  `data_fetch_check.py`, `calibration_check.py`, `signal_check.py`,
  `subgroup_strength.py`, `atm_check.py`, `probe_rushing.py`,
  `probe_rushing2.py`
- Local cache: `lines_cache.parquet` (184,782 rows for the four default
  markets). Delete it or pass `--refresh` after any new backfill.
- Tables: `lines` (live captures, with `book`), `historical_lines` (backfill),
  `bets` (the log; the outcome column is **`result`**)
- Repo: `avoliotg/nfl-props-app`, public

### Pricing parameters currently shipped (and their status)

| market | family | sigma | status |
|---|---|---|---|
| receiving | gamma | 3.272*proj^0.6172 | exponent roughly right (fitted 0.54-0.57), anchor wrong |
| rushing | gamma | 1.922*proj^0.7149 | **INVALID**, projections changed |
| qb_rushing | gamma | 4.624 + 0.7246*proj | untested, market never graded |
| qb_passing | gamma | 0.3531*proj | **INVALID**, projections changed |
| receptions | neg binomial | 1.143 + 0.2992*proj | best-calibrated market, exact at zero deviation |

Floors: receiving 11.5, rushing 34.5, qb_rushing 1.5, qb_passing 159.0,
receptions 1.7. PI_CAP = 0.35. `USE_STAGE_MULTIPLIER = False`.

---

## THE HONEST FRAME

Yesterday this project looked dead. This morning it looked like a measured edge
with a standard error of 0.040. Tonight it is one real market and three that
were measuring their own filters.

That is progress, not a setback. The three zeroes were always zero; the only
thing that changed is that they are now visible. A project that reported beta
0.337 for rushing would have shipped a rushing edge column and lost money on
it with complete confidence. Twice today a t-statistic above +6 appeared, and
twice the measurement infrastructure identified it as an artifact within
minutes, because the instrument was built to be suspicious of its own output
before it was pointed at anything.

What remains is narrower and more honest. Receptions has real signal at
beta 0.275 with t +9.45, and it does not yet clear a 6 percent hold. The gap
between those two facts is the whole project now. Closing it means better
pricing (median-calibrated alpha, sigma on level and deviation), better
population hygiene (the train/serve gate, the label bug, the name joins), and
then the line-as-feature model, which is the one idea on the list that could
find information the market misses rather than re-fitting information it
already has.

The measurement apparatus is the asset. There are 354,554 rows of multi-book
closing lines, a harness that answers the central question in under a minute
off a local cache, an edge-threshold script that prices the whole board through
the corrected pipeline, and a documented bug pattern that has now been caught
four different ways. Use it on every change, and believe the holdout table
rather than the curve.
