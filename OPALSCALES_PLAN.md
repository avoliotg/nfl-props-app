# OpalScales Master Plan
### Measurement first, then the pricing mean, then new modeling

Rewritten September 24, 2026. Replaces the September 23 evening version, which
was built partly on a retracted finding. Read `HANDOFF_2026-09-24.md` alongside
it for numbers and evidence; this document is the project overview and the
phased plan.

**What changed from the September 23 version.** The sigma floor does not exist,
so every item justified by it is gone. The evaluation harness turns out never
to have measured the app's model, which made three handoffs of model-change
instructions unexecutable. Phase 3.4 is a measured no-op. And Phase 4.4, the
blended pricing mean, is SHIPPED and was the actual defect: it moved the worst
calibration band from 18-32 points to 1.5-5.3 points in all four measurable
markets, with sigma untouched.

---

## WHAT THIS PROJECT IS

OpalScales is a personal NFL player-prop modeling and betting research app.
It projects individual player statistics from public nflverse data, compares
those projections to sportsbook lines, prices the disagreement as a
probability, and displays an edge. It is used by Ted and a small friend group.

**Stack:** Streamlit front end, Supabase Postgres, nflreadpy/nflverse for
player stats and schedules, The Odds API for lines. Six live markets:
receiving yards, receptions, rushing yards, QB passing yards, QB rushing
yards, anytime TD.

**The stated product vision:** loud when there is something to say, silent
otherwise.

**Where it stands.** Of four measurable markets, exactly one (receptions)
contains information the closing line does not already price. **The pricing
layer is now correct in the mean for the first time** (Phase 4.4, shipped
September 24), and four of five markets display no edge at all because their
beta is statistically indistinguishable from zero. One candidate betting rule
reproduces and survives a proper bootstrap at 91 percent of replicates
positive, but is not established and the app stays silent.

---

## THE REASONING BEHIND THE ORDERING

### Why measurement came first (and this was right)

In September the project was open-loop: a model change could not be evaluated
against the market for weeks. Three seasons of historical multi-book prop
lines for $59 converted "we will know by Week 8" into "we will know in a
minute". It has paid for itself several times over, mostly by killing ideas.

### Why population hygiene comes before new modeling (learned September 22)

Two of the four measured betas were entirely artifacts of volume filters in
the evaluation sample. Rushing went 0.337 to -0.032, qb_passing 0.402 to
0.075. Population hygiene is not cleanup, it is the difference between
measuring the model and measuring the filter.

### Why the PRICING MEAN moved up, and why SIGMA did not (learned September 24)

The September 23 plan promoted sigma on the strength of a floor that does not
exist. Retracted in full; see the handoff.

The real defect was in the pricing MEAN. This module was handed the raw
projection as the distribution mean. Pricing the blended mean instead,
`line + alpha + beta*(projection - line)`, collapses the worst calibration
band in every market:

| market | raw projection | blended mean |
|---|---|---|
| receptions | +21.1 | **+2.4** |
| receiving | -18.4 | **-3.8** |
| rushing | +31.7 | **-1.5** |
| qb_passing | -20.6 | **-5.3** |

**Sigma unchanged.** The over-dispersion tilt that two handoffs treated as an
open problem was mean error, which is exactly what the September 23
`dP/dsigma` finding predicted: sigma shifts every band the same direction and
can never move the two ends in opposite directions.

Three sigma forms were then tested AGAINST the corrected mean and the shipped
`PARAMS` forms won in every market. Phase 4.2 is closed as not an improvement.

### Why the HARNESS did not move (learned September 24)

`eval_harness` does not call `load_model`. It fits its own regression and
borrows only `build_dataset` and the feature list, which its own docstring
lists as future work at line 92. So beta 0.275 is a fair PROXY for the app's
model (served projections agree to about 0.1 receptions) and not a description
of it.

Rewiring it was considered and rejected: `load_model` is a single fit pinned to
`season <= 2024`, and calling it would score 2023 and 2024 with a model
trained on them, destroying the leave-season-out discipline that makes every
number in the project trustworthy. That is a real cost to fix a 0.1-reception
proxy error.

### Why more data and more features are NOT next

Unchanged and still settled. `--fixed-train 2022` gave pooled beta 0.230
against walk-forward 0.229 with four times the rows. The in-sample gap is
0.012, so no overfitting either. The constraint is that the features carry
information the market already prices. That points at exactly one idea: the
**line-as-feature model**.

---

## THE TWO BUG PATTERNS (read both before touching any code)

### Pattern 1: a filter on an in-game result applied before something that must not see it

Four places it bites: before a rolling feature; on the training population; on
the grading path; on the **evaluation** population. Fixing 1 through 3 does
not fix 4. The tell is the join rate, not the calibration table.

**Rule: any `build_dataset` that filters on an in-game statistic needs a
`build_all_rows` sibling with rolling features computed over the unfiltered
frame, the training gate matched to the serving gate, grading read from
unfiltered stats, and the harness scoring the sibling.**

Fixed: `anytime_td` (`touches >= 3`), `rushing` (`carries >= 5`),
`qb_passing` (`attempts >= 10`). Remaining: the `targets_roll >= 3` TRAINING
gate in `receiving` and `receptions` (their `build_dataset` has no in-game
filter, so they legitimately need no sibling), and
`qb_rushing.actual_result`.

### Pattern 2 (NEW, September 23): a reader of `lines` that does not collapse on book

`lines` holds one row per BOOK per snapshot. `db.get_line_movement` treats
each ROW as a snapshot, so its capture counts are inflated about 8x and its
`Move` and `toward_away` columns are contaminated by book spread rather than
measuring movement over time. It was correct when the capture was FanDuel
only; the 8-book capture broke it silently.

**Rule: any function that reads `lines` must collapse or select on `book`
before counting anything, and should carry `n_books` as an explicit column.**
`eval_harness` could never hide this bug because it does exactly that.

**FIXED September 24, and it was worse than described.** Two functions read
`lines`, not one. `get_lines`, which populates the editable Line column and
therefore feeds the PRICED line, also never filtered on book: the prefill was
whichever of eight books happened to write last at the latest `captured_at`.
Both now select FanDuel, which is the app's pricing anchor because the
`mc_pricing.BLEND` constants were fitted on FanDuel closing lines.

Consensus is the **median**, and it is the right anchor for the HARNESS, where
it estimates truth. It is the wrong anchor for the APP, because a median across
eight books is not a price any book offered. Select, do not average.

### Pattern 3 (NEW, September 24): a reader of `lines` that does not exclude hand-saved rows

The Save-to-Log path writes into the SAME table as the capture. As of
September 24 there are **5,071 hand-saved rows** in a 43,465-row table, all
labelled `book='fanduel'`, with `captured_at` spread across whenever a board
was saved. They interleaved with real captures and corrupted every first vs
latest comparison.

**There is no column marking a row as saved.** The discriminator is that a
captured row has no `projection` and no `edge`, because the capture has no
model attached. That is an accident rather than a design: if a future writer
populates `projection` on capture, `_drop_saved_rows` breaks silently. **Plan
item 6.10: add an explicit source column.**

**Rule: any function that reads `lines` must exclude rows with a non-null
`projection` before differencing or counting.**

Note the historical data is NOT in this table. `lines` holds 2026 only; the
354,554 historical rows live in `lines_cache.parquet` and come from The Odds
API directly. That is why patterns 2 and 3 never reached an analysis.

### Pattern 4 (NEW, September 24): the MIRRORED filter

Pattern 1 is an evaluation population inheriting a training filter. The mirror
also bites: **the APP filters and the INSTRUMENT does not.**

`mc_pricing.p_over` returns `None` for any projection below the market's
projection floor, so those rows never display. `sigma_refit` scored 1,402 such
rows, 14 percent of the receptions population, and its entire
dispersion-below-1 finding came from them: of 402 rows with dispersion below 1,
352 are sub-floor and the other 50 are an artifact of pricing around a blend
the app did not compute. **Zero of them can occur in the app as it runs.**

**Rule: any script that scores the pricing layer must apply the app's own
projection floor**, or its calibration table describes a population wider than
the board.

---

## WHAT IS ESTABLISHED (do not redo)

The handoff carries numbers and intervals. Summarised:

**Markets**
- Receptions is the only market with signal: beta 0.275, SE 0.029, t +9.45.
- Rushing (-0.032) and qb_passing (0.075) carry zero. Receiving 0.093.
- Pooled beta 0.172 must not be used in pricing. Per-market only.
- The raw projection is worse than the line as a mean estimate everywhere.
- More training data does not help. No overfitting.

**Pricing (rewritten September 24)**
- **THE SIGMA FLOOR DOES NOT EXIST.** `PARAMS[market]["floor"]` is a
  PROJECTION floor: `p_over` returns `None` below it. The only sigma floor is
  `SIGMA_MIN = 0.05`, which never binds. `sigma_refit` line 92 hardcoded a
  fictional `SIGMA_FLOOR = 1.7` and labelled the REAL shipped form
  `shipped-nofloor`, as though production behaviour were the counterfactual.
  Retract the 22.47 pp figure and every floored row.
- The cross-market version was impossible, not merely wrong: rushing and
  qb_passing are `family="gamma"`, which has no dispersion parameter and no
  zero atom. qb_passing's single-calibration-band symptom is real but comes
  from beta 0, not from sigma.
- **THE DEFECT WAS THE MEAN.** Pricing the blend instead of the raw projection
  moves the worst band from +21.1 to +2.4 (receptions), -18.4 to -3.8
  (receiving), +31.7 to -1.5 (rushing), -20.6 to -5.3 (qb_passing). Sigma
  unchanged. **Shipped, Phase 4.4.**
- **SIGMA IS CORRECT AS SHIPPED.** Tested against the corrected mean, the
  `PARAMS` forms beat a flat residual SD in every market (receiving -3.8
  against -20.2) and beat `sqrt(1.22*mean)` for receptions (+2.4 against
  +7.4). The level is nearly identical; the FORM is what matters, because
  sub-proportional scaling tracks the player and a scalar does not.
  **Phase 4.2 closed as not an improvement. Do not touch sigma without
  re-running `blend_sigma_grid.py`.**
- **Sigma does not depend on `abs(dev)`.** C within line +0.0000, CI [-0.048,
  +0.053]. Phase 4.3 closed.
- **dP/dsigma is negative in every band and family.** Confirmed independently:
  correcting the mean removed the tilt, which is what this predicted.
- The gamma is over-skewed at zero deviation; the negative binomial is exact.
- **BETA IS CLIPPED TO ZERO wherever its game-clustered CI includes zero.**
  Only receptions survives (0.226, SE 0.035, t +6.5). Four of five markets are
  REFERENCE ONLY: projection beside line, a probability, no edge and no tier.
- **ALPHA IS MEASURED DIRECTLY AT ZERO DEVIATION, not as a regression
  intercept.** With beta clipped, the intercept of a free-slope fit carries a
  correction for a slope the model no longer has (rushing: intercept +3.46
  against direct +4.01). The intercept also manufactured position effects in
  two markets (p=0.0033 and p=0.0391) that vanished on direct measurement
  (p=0.55, p=0.81).
- **Dropping alpha is much worse than keeping it.** Pricing at the line
  understates P(over) systematically and put 7,858 of 7,861 receiving props in
  the top tier.
- **Alpha and beta are FLAT across caliber and game context.** About twenty
  cuts: target-share quartile, position, exact line value, n_books, home/away,
  favourite/underdog, game total, roof, weekday, divisional, rest days. All
  consistent against a Bonferroni threshold. Minimum detectable beta spread
  ran 0.10 to 0.42, so this means "no effect we could see at this n".
- **Receptions alpha is NOT flat across seasons.** Direct p=0.0259: 2023
  +0.369 against 2024 +0.054, 2025 +0.081, 2026 +0.067. Same season every arm
  of the candidate rule loses in. Rushing alpha IS stable (p=0.735). If alpha
  is improved, recency weighting is the lever, not a caliber table.
- **BETA IS HIGHLY SENSITIVE TO THE LINE ANCHOR**, spanning 0.24 to 0.37
  across min / median / FanDuel / max, which is 7 to 10 SEs. `line_min` and
  `line_max` are order statistics over eight books and inflate beta
  mechanically, the same artifact family as `line_best` and BetRivers. FanDuel
  and consensus agree (0.226 vs 0.275, z -1.07) and are the only real anchors.
- **The typed line is verified for receptions.** The app prices a hand-entered
  line; typed reception lines match a captured FanDuel line on 92.4 percent of
  comparable props. Yardage markets match at 44 to 47 percent, which is line
  movement on a fine grid plus the `get_lines` bug, and does not matter there
  because beta is zero.

**Betting**
- No edge threshold is established. J1 stands: do not bet displayed edges as
  probabilities.
- The candidate rule (receptions, lines 2.5 and 3.5 only, sqrt sigma, FanDuel
  settlement, threshold about 0.09) gives holdout +0.1475, t +1.74, 176 bets,
  bootstrap CI [-0.0709, +0.3133] with 91.2 percent of replicates positive
  against a placebo at 24.4 percent.
- **Its weakness is the threshold-selection step, not season concentration.**
  The argmax needs three seasons to land near 0.090; with two it loses.
- The threshold search carries no upward bias (the 100-bet floor keeps it off
  thin cells).
- Neither component of the rule survives alone: model only +0.0742, book only
  +0.0524.
- **Every arm loses in 2023 and wins in 2024 and 2025.** Beta is stable across
  seasons, so this is not forecast quality. Leading candidate is book
  composition. **Corroborated from a new direction September 24:** receptions
  alpha measured directly at zero deviation is +0.369 in 2023 against +0.054,
  +0.081 and +0.067 in the later seasons (p=0.0259). 2023 is anomalous in the
  pricing parameter as well as in the rule's returns, which is what a changing
  consensus book set would produce. Restrict to books present in all four
  seasons and re-run both.
- Line shopping adds no forecast value and is negative as a strategy.
- The winner's curse from side selection costs about 4.3 points of win
  probability, larger than the vig. Measured from conditional against
  unconditional calibration, so it stands independently of the band pattern.

**Alt ladders (new, and a closed door)**
- FanDuel's alt ladders are **overs only** and priced off a single negative
  binomial: with the hold measured, one distribution fits every rung of every
  ladder to within 1.3 pp.
- At the overlapping rung (a half-integer main line makes "over 4.5" and "5+"
  the same event) the alt price is worse on 11 of 12 players and never better.
  Mean alt hold 8.4 percent against 7.0 on the main line.
- The far rungs are worse than a single NB, not better.
- **What survives: template pricing at the bottom of the board.** Players on
  different teams with different roles get identical prices to the dollar.

**Data**
- Lead-time contamination is real and irrelevant. No re-pull.
- Early-season beta is higher than full-season, not lower.
- `line_consensus` is a median over a **changing** book set: eleven books in
  2023 and 2024 including three defunct brands, eight in 2025. Mean book
  spread 4.67 all-seasons against 2.09 in 2025.
- Receptions is 93 percent flat on the line because FanDuel moves reception
  prices through the ODDS, so line-space movement and line-space CLV are blind
  for that market.
- Books post anytime TD early and yardage late. Tuesday afternoon is when they
  begin; counts climbed 32 percent overnight Tuesday into Wednesday.

---

## JUMP THE QUEUE

**J0. Fire the GitHub Actions capture workflow. DONE, and proven.** Five
secrets added, manual run green at 1,783 rows, and four scheduled runs fired
overnight with no intervention. Runs land 9 to 11 minutes after the cron
minute from best-effort queueing, which is inside the historical lead-time
window. **Remaining: verify the Sunday-specific entries fire.** Those are how
2026 becomes measurable on the same instrument as the 354,554 historical rows.

**J1. Do not bet displayed edges as probabilities.** Stands. 67 percent of
props show a positive edge after shrinkage, arithmetically impossible against
a 6 percent hold unless the layer is miscalibrated, and three of four markets
have zero signal.

**J2. RLS on `lines`. CLOSED. It was already correct.**
`admin_write_lines` (cmd ALL) is hardcoded to Ted's UUID, and
`authenticated_read_lines` is SELECT-only. Read is **intentionally** open to
authenticated users because Line Movement is meant to be shared with the
friend group, and the 17 registered users are all people Ted knows. Hardcoding
the UUID is correct here because `lines` has no `user_id` column. `auth.uid()`
confirmed resolving in a real app session. Revisit only if the app is opened to
strangers, at which point the signup gate is the lever, not the policy.

**J3. The `p_over` one-liner in `app.py`.** In the Line Movement save block,
change `"p_over": None,` to `"p_over": savable["p_over"] if not is_td else
None,`. Still open.

**J4. Add an empty `models/__init__.py`.** Now earned rather than cosmetic:
namespace resolution caused two surprises on September 23. `import
models.data_utils` works from the repo root, `import data_utils` does not.

**J5. Remove the temporary `whoami` `st.write` from the Line Movement tab in
`app.py`.** It was added to verify `auth.uid()` resolves. Still open.

**J6 (NEW, September 24). OPEN THE APP AND LOOK AT IT.** Four markets lost
their edge column, more rows price than before, and Line Movement lost 5,071
contaminated rows. No script exercises the Streamlit render path. Verify: the
receptions board shows an edge column; receiving shows the reference-only note
with no edge column; the prefill populates (an empty prefill plus a warning
means FanDuel is absent from that week's capture, which is the new intended
behaviour and is worth investigating in its own right); Line Movement shows
plausible capture counts rather than 8x inflation.

---

## PHASE 1: DATA (complete)

**1.1 to 1.5 complete.** The Odds API $59 tier. 2023, 2024, 2025 closing
snapshots plus 2026 weeks 1 and 2. 865 events, 354,554 rows, 8 books, zero
null weeks.

**Credit arithmetic (confirmed, sport-agnostic):**

```
live:       markets x regions        credits per event
historical: markets x regions x 10   credits per event
events endpoint (live)               free
events endpoint (historical)         1 credit per call, PAID PLANS ONLY
```

Credits are charged per market that RETURNS data, not per market requested.

**Credits remaining: approximately 56,659.** Monthly allocation, not a
balance. Unspent credits do not survive renewal.

**1.6 The subscription continues at a lower tier, not cancelled.** The big
tier was sized for the one-off backfill, which is done.

**1.7 Ongoing capture budget.** About 47 runs a week, roughly 16,000 credits a
month in season. Measured: a week 3 capture cost 32 credits against an 85
estimate. **Check the lesser tier's monthly allocation against 16,000 before
downgrading.**

**1.8 2026 as a complete season: the live captures ARE the closing
snapshots**, landing 20 to 60 minutes before kickoff, which is the same
instrument as the historical rows. **Do NOT disable the workflow at the
downgrade.**

**1.9 The known backfill defect, documented and NOT worth fixing.** 38 events
outside a 20 to 90 minute window, 5 captured after kickoff. Filtering changes
pooled beta by 0.000.

**1.10 (NEW) Book composition is inconsistent across seasons.** Eleven book
names in 2023 and 2024 against eight in 2025, three of them defunct brands
(`pointsbetus`, `barstool`, `unibet_us`). `line_consensus` is therefore a
median over a changing set, and this is the leading explanation for every arm
of the candidate rule losing in 2023. **Cheap and untested: restrict to books
present in all four seasons and re-run the per-season betas and the holdout.**

---

## PHASE 2: THE EVALUATION HARNESS (complete, and it is the asset)

`eval_harness.py`, `edge_threshold_v5.py`, plus the September 23 diagnostics
(`alpha_by_line_v4.py`, `sigma_by_dev.py`, `sigma_refit.py`,
`calib_reconcile.py`, `holdout_bootstrap_v2.py`). Run the relevant ones after
every change.

**The habits that make it trustworthy:**

1. **Nothing is fitted on the rows it scores.** Beta, alpha and every sigma
   parameter come leave-season-out. Thresholds are chosen out of sample. Every
   in-sample number is labelled as biased upward.
2. **Every claim has a placebo or a null.** The synthetic rigs plant known
   answers, including known-zero ones, so a tool that reports an edge on a
   null is broken. Three tool defects were caught this way on September 23.
3. **Composition is inspected, not just the coefficient.** The tail profile is
   what detects survivorship, and on September 23 it revealed that an extreme
   under-heavy tail was a flat-sigma artifact.
4. **(NEW) A script that reimplements an existing procedure must reproduce
   that procedure's published output before it is allowed to produce a new
   number, and should ABORT otherwise.** `holdout_bootstrap_v2` is the
   template. v1 of that script did not, and produced a confidently wrong
   result.

**Implementation notes worth preserving:**

- `--score-population {all,board}` decides the EVALUATION population. Default
  `all`. Note receptions and receiving have no `build_all_rows` sibling and
  legitimately need none, so the flag is a no-op for them.
- The best-line placebo permutes the **deviation**, not the projection.
- Per-season pooling must be in sigma units.
- `--cache lines_cache.parquet` appends only missing seasons.
- **`edge_threshold_v5` defaults to `--sigma-mode flat`**, which is the
  regression residual SD and the wrong functional form. The candidate rule
  requires `--sigma-sqrt` explicitly.
- v5's holdout: GRID 0 to 0.25 step 0.005; training floor 100 bets; a
  held-out season with fewer than 20 bets is DROPPED entirely; selection is a
  plain argmax of mean profit. `main()` is guarded so v5 can be imported.
- **Log loss is nearly blind to sigma** at half-integer lines, because P(over)
  is driven by the mean. Judge sigma forms on calibration, not on log loss.
- **Record the sign convention** in every calibration table. `actual -
  predicted` and `predicted - actual` both appear in this project's history and
  the mismatch has already caused one wrong conclusion.

---

## PHASE 3: POPULATION HYGIENE

**3.1 Rushing `carries_roll` survivorship. DONE.**

**3.2 Refit the rushing sigma and floor. CLOSED, the premise was false.**
Rushing is `family="gamma"`, which has no dispersion parameter and no zero
atom, so the reported zero-atom mechanism could not have applied. Its bottom
calibration band DID run +20.0 pp, and Phase 4.4 moved it to -1.5 by
correcting the mean. Sigma was never the problem.

**3.3 qb_passing `attempts >= 10`. DONE.**

**3.4 The `targets_roll >= 3` training gate. CLOSED. MEASURED NO-OP.**

Called the highest-value remaining model fix in three consecutive handoffs.
Measured September 24, directly on the app's own model because the harness
cannot see `load_model` at all.

Moving the gate from 3 to 1.5 adds 3,960 rows to a 7,280-row fit and moves the
coefficients materially (intercept -1.6706 to -1.0288). **It moves the served
projections by a maximum of 0.137 receptions across 296 players, mean absolute
0.058.** There is a faint pattern (low volume up, high volume down, crossing
between 4 and 6 targets) but within-band spread is five times the between-band
means.

Against lines spaced 1.0 apart, on a projection `project_week` rounds to one
decimal, this is irrelevant. **Gate left at `>= 3`. The choice is free.**

Two traps this exposed, both now working principles. The harness produced
BYTE-IDENTICAL output before and after, which is arithmetically impossible and
is what revealed the harness gap. And the first measurement used
`project_week`'s ROUNDED output, producing a fake "105 of 296 players moved
over 0.10" that was entirely quantization.

Also in the same pass: `actual_result` in both modules reads `build_dataset()`
and returns `None` on a null stat. nflverse appears to store 0.0 for active
WR/TE/RB rows (confirmed: 0 nulls in 22,540 receptions rows), so the practical
impact is nil, but matching the pattern removes the possibility.

**3.5 The qb_rushing label bug, plus its grading bug.** `MARKET_MAP` in
`import_lines` has no `qb_rushing` key, so QB rushing props store as
`market = 'rushing'`. The API data has the same issue. Solve it once at
analysis time by splitting on position from nflverse after the join.
Separately, `qb_rushing.actual_result` returns `None` where zero is correct.
Payoff: unlocks the highest-correlation market in the project (0.510) which
has never graded a row, and recovers about 40 percent of rushing's join rate.

**3.6 Name normalisation, and consolidate the two normalizers.** Parenthetical
team tags, initial forms, nickname aliases, dotted variants. `db._norm_name`
is scalar and `data_utils.norm_join_name` is vectorised with different rules;
one should call the other. Worth 1 to 2 points of join rate.

**3.7 WR receptions bias +0.30 (z +5.39) on 1,456 rows.** Largest measured
effect in the project, still unfixed, in the one market with signal.

**3.8 RB receiving as a yards-per-target problem.** RB receptions bias -0.16
while RB receiving yards is +3.86 (z +3.59), so the entire error lives in
yards per catch, not volume.

**3.9 Audit every remaining reader of `lines` for bug pattern 2**, and every
remaining `build_dataset` for pattern 1. `anytime_td` and `qb_rushing` are the
two not yet verified end to end against pattern 1.

**3.10 Fix `db.get_line_movement`. DONE September 24, and there were TWO
bugs.** Eight books treated as eight snapshots, AND 5,071 hand-saved rows
interleaved into the time series. Both fixed: select FanDuel, exclude rows
with a non-null `projection`, carry `n_books` and `book` as explicit columns.

**3.11 (NEW) Fix `db.get_lines`. DONE September 24.** This is the one that
mattered more, because it feeds the PRICED line. It never filtered on book, so
the prefilled line was whichever of eight books wrote last at the latest
`captured_at`. Now FanDuel only, saved rows excluded, and it returns empty with
a warning rather than silently substituting another book. That matters because
beta spans 0.24 to 0.37 across anchors.

**3.12 (NEW) The display REDESIGN for Line Movement is still parked**, pending
decisions about how lines feed betting choices. The bugs are not.

---

## PHASE 4: THE PRICING LAYER (4.4 SHIPPED September 24)

**4.1 Median-calibrated alpha. NOW A SMALL, WELL-POSED PROBLEM.** Choose
alpha so the priced P(over) at zero deviation matches the measured over rate.

After 4.4 the residual offsets are small and uniform rather than tilted:
receptions +2.4 in its worst band, receiving -3.8, rushing -1.5, qb_passing
-5.3. qb_passing's actual over rate is 0.5060 against a priced 0.4530, so it
is the one with room.

Two constraints learned September 24. **Use the DIRECT measurement at zero
deviation, not a regression intercept**; the intercept is an extrapolation
whose SE depends on where the data sits, and it manufactured position effects
in two markets that vanished on direct measurement. And **a per-line or
per-caliber alpha table is not supported**: alpha is flat across target-share
quartile, position, exact line value and every game-context cut tested. The
only real instability is ACROSS SEASONS for receptions (p=0.0259, 2023 four
times the others), which argues for recency weighting if anything.

**4.2 Replace the sigma form. CLOSED AS NOT AN IMPROVEMENT.**

There is no floor to remove, and the shipped sigma forms are correct. Tested
against the corrected mean:

| market | PARAMS (shipped) | flat residSD | sqrt(1.22*mean) |
|---|---|---|---|
| receptions | **+2.4** | +5.5 | +7.4 |
| receiving | **-3.8** | -20.2 | |
| rushing | **-1.5** | +3.3 | |
| qb_passing | -5.3 | **-5.1** | |

`eval_harness`'s legend claims its `residSD` is the correct sigma for pricing
around a blended mean. That is wrong as applied. The LEVEL is nearly identical
(receiving `PARAMS` gives 29.16 at the median mean against residSD 28.4), so
the damage is the FORM: `3.272 * proj^0.6172` scales with the player, while a
flat 28.4 gives a 10-yard receiver the same spread as a 90-yard one. The
sqrt constant of 1.22 was raw variance around the blend, which is not residual
spread conditional on the mean.

**Do not touch sigma without re-running `blend_sigma_grid.py` first.**

Two September 23 cautions are cleared rather than outstanding. `PI_CAP` never
binds for receptions, rushing or qb_passing because all three have
`gate=None`, which collapses `_components` to a plain moment match. And
`USE_STAGE_MULTIPLIER` is `False` in `mc.py:19` with 1.0 hard-substituted at
line 56, so nothing in production passes a multiplier.

**4.3 Sigma as a function of deviation. CLOSED.** Measured at zero within
line value, CI [-0.048, +0.053]. Do not add the term.

**4.4 Price off the blended mean, with MARKET-SPECIFIC beta. SHIPPED
September 24. This was the actual defect.**

```
market       alpha      beta     basis
receptions   +0.1491    0.2261   direct alpha SE 0.042; beta t +6.5
receiving    +3.4623    0.0      beta clipped (t +1.5)
rushing      +4.0067    0.0      beta clipped (t -0.9)
qb_passing    0.0       0.0      alpha unresolvable (SE 2.55)
qb_rushing    0.0       0.0      no data (MARKET_MAP label bug)
```

Beta is clipped to zero wherever the game-clustered CI includes zero. Shipping
rushing's -0.043 would price AWAY from the projection; shipping receiving's
0.066 would display phantom edges in a market with no signal. The harness's
own out-of-sample procedure already clips rushing to 0.000.

qb_passing's alpha is zero because it is UNRESOLVABLE, not absent: direct
+0.55 with SE 2.55, per-season +1.37 / +4.68 / -3.94. Sigma there is about 73
yards, so a few yards of skew is one fortieth of a standard deviation.

**Four of five markets are now REFERENCE ONLY**: projection, line and P(over)
with no edge, side or tier column at all. This was necessary rather than
cosmetic, because the unsuppressed board would put 6,956 receiving props in
Strong and 1,928 rushing props in Lean. J1 in code.

**Side effect:** more rows price than before, because the blend sits near the
line and the projection floor bites less. receptions 7,338 to 7,451, rushing
1,928 to 1,942, qb_passing 1,514 to 1,662. A receptions projection of 1.0 used
to render blank and now prices at 0.39.

**4.5 Shrink for the winner's curse. LARGELY ABSORBED BY 4.4. Re-measure
before doing anything.**

The over-dispersion tilt was mean error, and 4.4 is a mean correction, so most
of what 4.5 was aiming at is gone: the tilt no longer exists in any market.
Beta of 0.226 IS a shrinkage of the projection toward the line, so 4.4 does
the job structurally.

The instability at slopes 0.995, 0.849, 0.688, 0.852 is now most plausibly
explained: the correction was being fitted against a mean error it could only
partly absorb.

**Re-measure the residual tilt under 4.4 before fitting anything.** What
remains is a small uniform offset, which is Phase 4.1's job, not a shrinkage's.
The 4.3-point side-selection cost is measured independently and stands.

**4.6 The 281-row bottom band. RESOLVED.** They ARE the 0.5 and 1.5 lines, and
the projection floor already removes 78 percent of the 0.5 line and 37 percent
of the 1.5 line from the app's board. The band was largely measuring rows the
app does not display, which is bug pattern 4. Under the blend the bottom band
is empty for receptions.

**4.7 Then, and only then, recompute the edge threshold.**

---

## PHASE 5: THE BETTING RULE

**5.1 Fix the threshold-selection step. This is the candidate rule's real
weakness.** The argmax over 51 cutoffs needs three seasons to land near 0.090;
with a two-season training set it picks something that loses on 769 bets.
Candidates: a one-standard-error rule, a smoothed curve, or choosing the
cutoff on calibration rather than on realized ROI. Worth more than another
backtest of the same rule, because this is the part that will not generalise
to 2026.

**5.2 Paper-trade the candidate rule.** 2 to 3 bets a week, tracked by hand.
It answers the limit question that no dataset can, and it is the only way to
find out whether the prices are actually available at size.

**The 4.2 prerequisite is discharged** (sigma unchanged) and 4.4 has shipped,
so the pricing path is now stable. Note the rule STILL cannot be reproduced by
the app: `edge_threshold_v5` prices with `sqrt(1.25*mean)` sigma and its own
threshold logic, while the app uses `PARAMS` sigma and displays no threshold.
Paper-trading must be driven from v5, not from the app's board.

**5.3 CLV as the primary feedback metric.** Two hundred graded bets say almost
nothing about ROI and a great deal about closing line value. **Prerequisite:
reception movement must be computed in PROBABILITY space**, because receptions
is 93 percent flat on the line and FanDuel moves reception prices through the
odds. Line-space CLV is blind on the one market that matters.

**5.4 The line-as-feature model.** `actual = alpha + gamma*line + sum(delta_i
* feature_i)`. The one route to finding information the market misses rather
than re-fitting what it already has. Fit it after Phase 3 and 4 so the
features being tested are the honest versions.

**5.5 Compound distribution: receptions x yards per reception.** Receiving
yards equals receptions times yards per catch. Receptions is a usage quantity
with real signal; yards per catch is close to noise. Modeling yards directly
dilutes the one signal with the one unpredictable thing, which is why
receptions is 0.275 and receiving is 0.093. Strengthened by the reception
distribution now being validated at every rung of the ladder. Test offline on
four seasons: does it predict the realized yards distribution better than the
plain gamma, and does it get the rate of zero-yard games right. **Check the
correlation between reception count and yards per catch first**, because a
negative correlation would mean the compound model overstates variance.

**5.6 Role-change detection from snap counts and depth charts.** Both sources
are already loaded and nothing tests whether the market underweights them.
This is information the market UNDERWEIGHTS rather than lacks, which is the
only kind realistically exploitable with public data. Supported indirectly by
beta being HIGHER on frequently quoted players (spearman +1.00), which is the
population where a role change is most detectable against a long baseline.

**5.7 Injury and inactive filter at import, plus the large-move warning.**
Kills stale rows and a whole class of fake edge. The operational edge is
speed: a starter ruled out at 11am Sunday with a backup's line that has not
moved is a real prop edge requiring no forecasting.

**5.8 Proper two-sided devigging.** Raw implied probability double-counts the
hold. On receptions, where FanDuel uses per-player odds, multiplicative,
additive and Shin devigging differ by 1 to 2 points. Affects every edge
computed anywhere.

**5.9 Reception movement in odds space.** 254 of 273 reception props showed a
flat line because FanDuel moves reception prices through the ODDS. Now a
prerequisite for 5.3.

**5.10 Record which book a bet was placed at in `bets`.**

**5.11 Pinnacle via the `eu` region.** Downgraded: FanDuel came out second
sharpest of eleven books with the tightest hold at 0.0608, so you are already
anchored on one of the best available prices. One line of code, about 80
credits a slate, modest expectations.

---

## PHASE 6: STRUCTURAL CLEANUP

**6.1** Consolidate the two name normalizers (folded into 3.6).
**6.2** Move `_all_player_stats` into `data_utils`. Duplicated in
`rushing.py`, `anytime_td.py` and `qb_passing.py`.
**6.3** Verify the TSV export column order matches the tracker workbook.
**6.4** UTC to Eastern for import stamps, display-time conversion only.
**6.5** Pin pandas, and clean the bare-integer timedelta deprecations at
`live_capture.py:224` and `fill_weeks.py:194`. **Lower urgency than it
looked:** the Actions runner installed pandas 3.0.6 and `live_capture.py` ran
clean, so this is not currently a live failure mode. Still worth pinning
because the runner installs fresh every run.
**6.6** Add `models/__init__.py` (folded into J4, now earned).
**6.7** Make `odds_backfill.py` stop on the first 401, and rename its
"Supabase password" prompt to "app password".
**6.8 (NEW)** Consolidate the `edge_threshold` versions. There are five files
(`edge_threshold.py` plus v2 through v5) and only v5 can reproduce the
candidate rule. Retire the rest or move them to an `archive/` folder so nobody
runs v1 by accident. Note v1's default `--sigma-mode flat` silently uses the
wrong sigma form.
**6.9** Add a row-count assertion to `live_capture.py` so a scheduled run that
writes zero rows while events exist exits nonzero. A green check that wrote
nothing is the failure mode to watch for, and nothing currently distinguishes
it.
**6.10 (NEW, September 24) Add an explicit source column to `lines`.** The
saved-versus-captured discriminator is `projection IS NULL`, which works by
accident. If any future writer populates `projection` on capture,
`_drop_saved_rows` breaks silently and both Line Movement and the prefill
recontaminate. One column fixes it permanently.
**6.11 (NEW) `GAMES_PLAYED = 0` is hardcoded in `app.py`** with a TODO saying
generalize before Week 2. It is Week 3. Currently harmless because
`USE_STAGE_MULTIPLIER` is `False`, but if that flag is ever flipped on, every
projection gets a 1.5x sigma. Either generalize it or delete the parameter.
**6.12 (NEW) `mc.SIGMA_RULES` is a FOURTH sigma**, reachable via
`legacy_base_sigma`. Dead in the live path since `base_sigma` delegates to
`mc_pricing`, but it holds a flat 2.1 for receptions and 71.0 for qb_passing
and it makes the "three sigmas" count wrong. Delete or clearly mark it.
**6.13 (NEW) Integer reception lines exist and are mispriced.** 452 rows at
0.5 but only 5 at 1.0; 2,724 at 2.5 but 24 at 3.0; 36 at 2.0. About 187 rows
sit on integer lines where a push is possible, and `p_over` uses
`floor(line)`, so a line of 2.0 is priced as `P(Y >= 3)` while the actual bet
pushes at exactly 2. Low volume, real defect.

---

## PHASE 7: LONG HORIZON

**7.1 NHL, shots on goal.** Deliberately deferred. On a continuously paid tier
this is a scheduling question rather than a race. Note the volume asymmetry:
1,312 NHL regular season games against the NFL's 272, so one market is about
13,000 credits per historical season. Shots on goal chosen because event
counts are high enough not to be luck-dominated, shots are a role and usage
statistic which is stable and autocorrelated, and lines sit on a coarse
integer grid where the line cannot sit at the median. The negative binomial
work from receptions transfers directly. Open question first: check the
quality of the free hockey stat data against nflverse.

**7.2 Copula or Monte Carlo for correlated props.** Where Monte Carlo
genuinely becomes necessary. Commercially interesting because SGP pricing is
where books are weakest. Gate on a demonstrated single-prop edge, which does
not yet exist.

**7.3 Play-by-play engine.** The only item that could produce genuinely
differentiated projections rather than better-fitted versions of public data.
Offseason.

**7.4 Re-run `td_reversion_check.py` at Week 5 and Week 8.**

**7.5 Traffic dashboard or wider distribution.** Not before there is something
worth distributing. Note that if the app is opened to strangers, the signup
gate becomes the lever; the RLS policy already handles writes.

---

## THE HONEST RISK, RESTATED

The September 22 version said: if beta comes back near zero across three
seasons after every Phase 3 fix, that is a definitive answer. For rushing and
qb_passing that has happened. Receiving at 0.093 is on the same path.
Receptions survived at 0.275.

September 24 settles the pricing axis, and not the way September 23 thought.
The sigma floor was fiction. The real defect was that the layer priced the raw
projection as the distribution mean, and correcting it moved the worst
calibration band from 18-32 points to 1.5-5.3 points in all four measurable
markets with no change to sigma, no new data and no new features.

That is a real improvement to the app independent of whether any bet is ever
placed, and it is the first production change in three sessions.

It came with a loss of apparent functionality that is the most honest thing in
the change: four of five markets now display no edge, because their beta is
statistically indistinguishable from zero and an edge column there was a
verdict the evidence never supported.

The third axis, added the same day, is that the **instrument** has a known
limit. `eval_harness` has never measured the app's model. Beta 0.275 is a fair
proxy and not a description, and any future model-change plan has to say which
of the two it is testing.

The candidate rule is the one open question that could still make this a
betting project rather than a measurement project. It reproduces, its search
carries no upward bias, and 91 percent of bootstrap replicates are positive.
It is not established. Its weakness is now identified precisely enough to
attack: the threshold-selection step needs more seasons than it has.

The strategy list that does not require beating the market's mean has shrunk
by one. Line shopping is dead (tested, negative). The alt ladder is dead
(FanDuel prices it off one distribution and charges more). What remains:

- **Distributional edge** where line increments are too coarse to sit at the
  median, which is exactly where receptions lives, and which is now
  well-calibrated for the first time (worst band +2.4 points, down from
  +21.1)
- **Template pricing at the bottom of the board**, visible in a screenshot,
  requiring no model
- **Speed on news** where a line has not moved yet
- **Correlated props** where books price correlation crudely
- **The line-as-feature model**

---

## WHAT THE APP BECOMES

Not a general-purpose edge column across six markets. Tier size is measured
anti-predictive, and the reason is structural: side selection costs about 4.3
points of win probability, which exceeds the vig, and it worsens as the
displayed edge grows.

The shape the evidence supports:

- **receptions at 2.5 and 3.5 only**, with a shrunk edge and a stated
  threshold near 0.09, expected to fire 2 to 3 times a week and stay silent
  otherwise, and not until the selection step is fixed
- **the other four markets shown as reference only**, projection beside line,
  no verdict. **DONE September 24**, enforced in code by
  `mc_pricing.has_model_signal`.
- **correctly priced probabilities everywhere**, which after 4.4 they are for
  the first time. Worst calibration band by market: receptions +2.4,
  rushing -1.5, receiving -3.8, qb_passing -5.3.
- per-book lines with FanDuel marked as the sharpest available anchor
- a flag for large recent moves, since those are usually news the model cannot
  see
- an inactive filter so dead props never appear
- Line Movement rebuilt on a book-collapsed series, with `n_books` visible

**Suggestions come from one market and one line band. The rest of the app is a
research tool that reports honestly, which is most of its value.**
