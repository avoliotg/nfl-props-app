# OpalScales Master Plan
### Measurement first, then population hygiene, then new modeling

Rewritten September 22, 2026, evening. Replaces the version written
September 21. Read `HANDOFF_2026-09-22_EVENING.md` alongside it for state and
findings; this document is the project overview and the phased plan.

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
otherwise. An edge column that fires on some weeks and stays quiet on others,
rather than manufacturing a number for every prop on the board.

**Where it stands:** the projection layer works as a projection layer. The
pricing layer overstates edges. Of four measurable markets, exactly one
(receptions) contains information the closing line does not already price, and
even that one does not yet clear the vig. See the handoff for numbers.

---

## THE REASONING BEHIND THE ORDERING

### Why measurement came first (and this was right)

In September the project was **open-loop**: a change to a model could not be
evaluated against the market for weeks, because there were 596 graded rows and
no historical lines. Every fix in the backlog was unfalsifiable.

Three seasons of historical multi-book prop lines for $59 converted "we will
know by Week 8" into "we will know in a minute". That was the right call and
it has already paid for itself several times over, mostly by killing ideas.

### Why population hygiene comes before new modeling (learned September 22)

The backlog was framed as bug fixes with incidental effects on accuracy. It is
not. **Two of the four measured betas were entirely artifacts of volume
filters in the evaluation sample.** Rushing went from 0.337 to -0.032 and
qb_passing from 0.402 to 0.075 when the filters came out of the instrument.

That means population hygiene is not cleanup, it is the difference between
measuring the model and measuring the filter. Any new modeling done before the
populations are clean is modeling against corrupted feedback.

### Why more data and more features are NOT next

Two measurements settled this. `--fixed-train 2022` trained one model on 4,608
rows and scored four seasons, giving pooled beta 0.230 against walk-forward
0.229 with up to 20,313 rows. More training data does not improve the edge
over the line at all. And the in-sample gap is 0.012, so the models are not
overfitting either. The constraint is not sample size and not variance. It is
that the features carry information the market already prices.

That points at exactly one idea on the list: the **line-as-feature model**,
which searches for the residual the market omits by construction rather than
competing with the market's own information.

---

## THE BUG PATTERN (read this before touching any model)

Every significant bug found in this project has the same shape: **a filter on
an in-game result applied before something that must not see it.** Four places
it bites:

1. **Before a rolling feature.** The feature is measured only over games the
   player qualified, so a backup's workload reads like a starter's. Measured
   inflation in rushing `carries_roll`: +6.31 carries in the 0-2 bucket,
   falling monotonically to +0.06 at 15 or more.
2. **On the training population.** The model is fitted on survivors and then
   scores everyone, which is extrapolation below the training range. Rushing
   trained on rows averaging 57.2 yards and served a population averaging
   32.5.
3. **On the grading path.** The bet never grades, and the missing grades are
   disproportionately losses, which flatters the record.
4. **On the EVALUATION population.** The measured beta, over rate, ROI and
   calibration are computed on a sample conditioned on the outcome. **Fixing
   1 through 3 does not fix 4.**

**The tell is the join rate, not the calibration table.** Survivorship leaves
calibration looking fine, because alpha absorbs the shift: a filtered market
prices P(over) at 0.61 and the filtered sample delivers 0.609. It is
calibrated to a population that is not the one the line was set for.

Demonstrated on synthetic data with zero signal by construction: a filter
removing 5.4 percent of rows non-randomly produced ROI from +0.073 to +0.163
with t-statistics from +9.8 to +12.9.

**Rule: any `build_dataset` that filters on an in-game statistic needs a
`build_all_rows` sibling with the rolling features computed over the
unfiltered frame, the training gate matched to the serving gate, grading read
from unfiltered stats, and the harness scoring the sibling.**

Instances found and fixed: `anytime_td` (`touches >= 3`), `rushing`
(`carries >= 5`), `qb_passing` (`attempts >= 10`). Instances remaining: the
`targets_roll >= 3` training gate in `receiving` and `receptions`, and
`qb_rushing.actual_result`.

---

## WHAT IS ESTABLISHED (do not redo)

Summarised here; the handoff carries the numbers and confidence intervals.

- Receptions is the only market with signal: beta 0.275, SE 0.029, t +9.45.
- Rushing and qb_passing carry zero signal. Their earlier betas were filters.
- Receiving is near zero at beta 0.093.
- Pooled beta is 0.172 but must not be used in pricing; use per-market betas.
- The pricing assumption `beta = 1` is rejected at t = -36.34.
- The raw projection is worse than the line as a mean estimate in every market.
- No edge threshold is established: pooled holdout ROI +0.0183, SE 0.0741.
- The alpha betting edge is dead. Receiving over rate 0.4912 against a 0.5305
  breakeven. Alpha remains a valid PRICING parameter.
- Shopping adds no forecast value (best-line beta artifact-corrected from
  0.345 to 0.156) but does add real price value, about 1.7 to 3.2 points of
  win probability on 2025's 8-book data.
- Sigma must be level-dependent; the shipped functional form was right.
- The gamma is over-skewed by 3.6 to 4.1 points at zero deviation; the
  negative binomial is exact to 0.1.
- The pricing layer is over-dispersed rather than biased: unconditional
  calibration is +1.1 pp overall but +7.3 at the bottom band and -11.6 at the
  top.
- The winner's curse from side selection costs about 4.3 points of win
  probability, which is larger than the vig.
- Lead-time contamination in the backfill is real and irrelevant. No re-pull.
- Early-season beta is higher than full-season, not lower.
- More training data does not help. No overfitting.
- Monte Carlo is premature for marginal single-stat distributions. It becomes
  necessary only for joint distributions or a play-by-play engine.
- Receptions is 93 percent flat on the line because FanDuel moves reception
  prices through the ODDS, so Line Movement measures the wrong quantity there.
- Books post anytime TD early and yardage late. A Tuesday Week 3 capture
  returned 87 percent anytime TD rows and yardage only one or two games deep.

---

## JUMP THE QUEUE

**J0. Fire the GitHub Actions capture workflow.** Time-sensitive and still not
done. `.github/workflows/capture-lines.yml` is written. Commit it, add the
five repo secrets, fire it manually before trusting the cron. Opening lines
are the only perishable thing in this plan: historical data is not going
anywhere, but Thursday's pre-movement numbers will be gone permanently.

**J1. Do not bet on displayed edges as probabilities.** Now supported by
measurement rather than by a calibration curiosity. 65.3 percent of props show
a positive edge after shrinkage, which is arithmetically impossible against a
6 percent hold unless the layer is miscalibrated. Three of four markets have
zero signal.

**J2. RLS on `lines`.** `authenticated_read_lines` is still `USING (true)`, so
any authenticated user can read the whole table. **There are 17 registered
users**, several signing in as recently as September 20. The working pattern is
proven on `historical_lines`:

```sql
create policy "owner full access" on historical_lines
  for all to authenticated
  using (auth.uid() = user_id) with check (auth.uid() = user_id);
```

Ten minutes. `db.py` now threads real user sessions through
`get_authed_client`, so `auth.uid()` resolves at the database.

**J3. The `p_over` one-liner in `app.py`.** In the Line Movement save block,
change `"p_over": None,` to `"p_over": savable["p_over"] if not is_td else
None,`. `get_line_movement` already computes it.

**J4. Add an empty `models/__init__.py`.** Imports currently work via
namespace-package resolution. Nothing is broken; this removes a class of
surprise.

---

## PHASE 1: DATA (complete, with one decision outstanding)

**1.1 to 1.5 complete.** The Odds API $59 tier, 100,000 credits. Backfill of
2023, 2024, 2025 closing snapshots plus 2026 weeks 1 and 2. 865 events,
354,554 rows, 8 books, zero null weeks.

**Credit arithmetic (confirmed, sport-agnostic):**

```
live:       markets x regions        credits per event
historical: markets x regions x 10   credits per event
events endpoint (live)               free
events endpoint (historical)         1 credit per call, PAID PLANS ONLY
```

No sport multiplier. Empty responses are not charged. **Credits are charged
per market that RETURNS data, not per market requested**, which is why a Week
3 live capture estimated 80 and was charged 24.

Five NFL market keys cover all six app markets because `player_rush_yds`
contains both RB and QB rushing: `player_pass_yds`, `player_rush_yds`,
`player_reception_yds`, `player_receptions`, `player_anytime_td`.

**Credits remaining: approximately 56,700, expiring mid-October.**

**1.6 The outstanding decision: 2026 as a complete season.** Credits are a
monthly allocation and do not survive cancellation. A full 2026 closing
backfill cannot be done now because the games have not been played. Getting
2026 on the same instrument as 2023 through 2025 means one more month of
subscription in February. Cheap, but decide deliberately rather than
discovering it in January.

**1.7 Budget for the remaining window:**

| item | credits |
|---|---|
| live capture through mid-October, 47 runs a week | ~15,000 |
| NHL historical, two past seasons at 10 credits a game | ~26,000 |
| reserve | ~15,000 |

**1.8 Cancel a few days before the renewal date**, not as soon as the backfill
verified. Cancelling ends the dense capture window, and the free tier supports
only about one NFL capture a week. Disable or thin the Actions workflow in the
same sitting or it will blow through the free 500 credits a month.

**1.9 The known backfill defect, documented and NOT worth fixing.** The
closing timestamp was computed as placeholder kickoff minus 45 minutes using
the September enumeration, while `commence_time` stores the real kickoff. 38
events are outside a 20 to 90 minute window, 5 of them captured after kickoff.
Gated as irrelevant: filtering changes pooled beta by 0.000. If it is ever
worth re-pulling, drive the request timestamp from the stored actual
`commence_time`, and note that **mid-season enumeration is week-scoped for
free** (a September 1 call returns all 272 games; a September 15 call returns
16).

---

## PHASE 2: THE EVALUATION HARNESS (complete, and it is the asset)

`eval_harness.py` and `edge_threshold.py`. Run both after every change.

What the harness reports: beta with game-clustered standard errors per market,
per season in sigma units and pooled; alpha; the **empirical over rate** with
a clustered SE; out-of-sample blend RMSE against line and projection;
residual SD; training counts; a join diagnostic naming the most frequent
unjoined players per market; and a shopping table measuring realized win rate
at each line source with the side held fixed.

What `edge_threshold.py` reports: leave-season-out shrinkage parameters;
level-dependent sigma with a span gate; the zero-deviation pricing check;
unconditional and conditional calibration (the difference is the winner's
curse); threshold curves with clustered SEs; a tail profile showing what the
winning bets consist of; and a holdout where the threshold is chosen on other
seasons and spent on the held-out one.

**The three habits that make it trustworthy, all of which caught something
real:**

1. **Nothing is fitted on the rows it scores.** Beta and alpha come
   leave-season-out; blend RMSE uses 5-fold CV grouped on `event_id`;
   thresholds are chosen out of sample. Every in-sample number in the output
   is labelled as biased upward.
2. **Every claim has a placebo or a null.** The best-line beta has a
   permutation test. The synthetic smoke tests use known-zero signals so a
   tool that reports an edge on them is broken.
3. **Composition is inspected, not just the coefficient.** The tail profile
   is what detects survivorship, because survivorship shows up as profit
   rather than as a calibration failure.

**Implementation notes worth preserving:**

- `--score-population {all,board}` decides whether the EVALUATION population
  is `build_all_rows()` or `build_dataset()`. Default `all`. This flag is the
  single most consequential line in the harness.
- The best-line placebo permutes the **deviation**, not the projection.
  Permuting the projection breaks the pairing between a projection and its own
  line and understates the artifact sevenfold.
- Per-season pooling must be in **sigma units** with scales computed once per
  market across all seasons, or passing yards swamps receptions and the
  coefficient is meaningless.
- `--cache lines_cache.parquet` appends only missing seasons. 354,554 rows at
  1,000 a page is 355 round trips.
- `--fixed-train YEAR` holds the model constant across seasons, which
  separates instability from training-size effects.

---

## PHASE 3: POPULATION HYGIENE (the critical path)

Every item gets the harness run before and after. A fix that does not move the
numbers gets reverted or reconsidered rather than kept on faith.

**3.1 Rushing `carries_roll` survivorship. DONE.** Rolling features moved to
`build_all_rows`, training gate matched to the serving gate at
`MIN_CARRIES_ROLL = 1.5`, Week 1 bridge moved off the filtered frame, null
`rushing_yards` treated as zero, `load_model` parameterised on the training
cutoff. Result: beta 0.337 to -0.032, join rate 48.2 to 59.3 percent, over
rate 0.5380 to 0.4721, served players 102 to 127 with none lost.

**3.2 Refit the rushing sigma and floor. NEXT, and non-optional.**
`1.922*proj^0.7149` and floor 34.5 were fitted against projections that no
longer exist. Given beta is now zero, the correct display for rushing may be
no edge at all.

**3.3 qb_passing `attempts >= 10`. DONE.** Same four fixes plus two more: a
**full-sample statistic leak** in the `def_pass_roll` fallback (the league
mean was taken over all five seasons, so a 2023 row was filled partly from
2025; now an expanding mean in chronological order), and `def_pass_roll` now
computed from all QB rows so a defence that faced a QB who left after 8
attempts is credited with those yards. Result: beta 0.402 to 0.075, residual
SD 70.0 to 72.6 as the left tail was restored, join rate 95.0 to 96.7 percent.

**3.4 The `targets_roll >= 3` training gate in receiving and receptions. THE
HIGHEST-VALUE REMAINING FIX.** `load_model` trains on `targets_roll >= 3`
while `project_week` serves at 1.5. Same population mismatch that erased two
betas, now sitting in the one market that has signal.

Handle this more carefully than the previous two. Both modules already have
honest evaluation populations (no volume filter in `build_dataset`, confirmed
at 22,540 rows each), so **only the training gate moves**. Receptions is the
only market worth anything, so a careless change could destroy the single real
finding. Run the harness before and after and keep the change only if beta
holds or improves.

Also in the same pass: `actual_result` in both modules reads `build_dataset()`
and returns `None` on a null stat. nflverse appears to store 0.0 rather than
null for active WR/TE/RB rows, so the practical impact is small, but matching
the rushing and anytime_td pattern removes the possibility.

**3.5 The qb_rushing label bug, plus its grading bug.** `MARKET_MAP` in
`import_lines` has no `qb_rushing` key, so QB rushing props store as
`market = 'rushing'`. The `is_qb_model` flag routes the edge calculation
correctly in memory but never reaches the database. **The API data has the same
issue**, since `player_rush_yds` contains both. Solve it once at analysis time
by splitting on position from nflverse after the join, rather than at fetch
time.

Second, independent cause of the same symptom: `qb_rushing.actual_result`
returns `None` when `rushing_yards` is null, where zero is correct for an
active QB. Both must be fixed or the market still never grades.

Payoff: unlocks the market with the highest correlation in the project (0.510,
essentially unbiased) which has **never graded a single row**, and recovers
about 40 percent of rushing's join rate.

**3.6 Name normalisation, and consolidate the two normalizers.** Specific gaps
identified from the join diagnostic: parenthetical team tags
(`Trey McBride (Ari)`, `Lamar Jackson (BAL)`, `Michael Thomas (NO)`,
`Michael (Saints) Thomas`), initial forms (`A. Trautman`, `J. Hill`,
`M. Jones Jr.`), nickname aliases (`Joshua Palmer`, `Cameron Ward`,
`Hollywood Brown`), and dotted variants (`C.J. Uzomah`, `CJ Stroud`).
`db._norm_name` is scalar and `data_utils.norm_join_name` is vectorised and
they apply different rules; one should call the other so they cannot drift.

Worth 1 to 2 points of join rate across three markets, and join rate is now
the primary survivorship tell, so it is worth more than it looks.

**3.7 WR receptions bias +0.30 (z +5.39) on 1,456 rows.** The largest measured
effect in the project and still unfixed, in the one market with signal. At a
4.5 line with sigma about 2.49 that is roughly 0.12 sigma, overstating P(over)
by about 5 points. High-volume receptions is worse at +0.32 (z +3.15).

**3.8 RB receiving as a yards-per-target problem.** RB receptions bias is
-0.16 while RB receiving yards is +3.86 (z +3.59). At about 7 yards a catch
the reception shortfall predicts -1 yard, so the entire error lives in ypt,
not volume. Fix the RB ypt feature, not the RB level.

**3.9 Audit every remaining `build_dataset` for the four-place bug pattern.**
Three instances found so far by reading six files. `anytime_td` and
`qb_rushing` are the two not yet re-verified against the pattern end to end.

---

## PHASE 4: FIX THE PRICING LAYER

**4.1 Median-calibrated alpha.** One parameter, and it removes a 3.6 to 4.1
point structural lean from every gamma prop before the model says anything.
Choose alpha so the priced P(over) at zero deviation matches the measured over
rate, rather than reading it off the regression's mean offset.
`edge_threshold.zero_dev_check` already reports the gap to target. Receptions
needs no change: its negative binomial is exact to 0.1 of a point.

**4.2 Keep sigma level-dependent, move the anchor.** The shipped
`a * proj^b` shape is right (fitted exponents 0.54 to 0.57 for receiving
against a shipped 0.6172, 0.48 to 0.53 for rushing against 0.7149). What
changes is that the spread should be measured around the **blended mean**
rather than the model's own projection, because the shipped version includes
the projection error. Note that qb_passing cannot support a power law at all
(every QB projects into a narrow 180 to 300 band, so the fit is unidentified
and must fall back to flat).

**4.3 Sigma as a function of level AND deviation.** The over-dispersion
finding. Sigma currently depends on the projection level only, but the
conditional variance almost certainly also rises with
`|projection - line|`, because large disagreements happen when something
unusual is going on: injuries, role changes, weather. Rows with large `|dev|`
get distributions that are too narrow, and those rows populate the top and
bottom calibration bands where the gaps are +7.3 and -11.6 points.

**4.4 Price off the blended mean, with MARKET-SPECIFIC beta.**
`line + alpha + beta * (projection - line)` as the distribution mean. Pooled
0.172 would display phantom edges in three markets. At the measured values,
rushing and qb_passing should show nothing and receiving almost nothing.

**4.5 Shrink for the winner's curse, or stop conditioning on the side.**
Side selection costs about 4.3 points of win probability, which exceeds the
vig. Two routes: shrink the displayed probability toward 0.5 by the measured
curse, or display both sides with their own probabilities and let the user see
that neither clears. The second is more honest and fits the product vision
better.

**4.6 Then, and only then, recompute the edge threshold.** The infrastructure
exists. It currently says no threshold clears on four seasons.

---

## PHASE 5: NEW MODELING

**5.1 The line-as-feature model. This is now the main strategic lever.**

```
actual = alpha + gamma*line + beta*(projection - line) + sum(delta_i * feature_i)
```

Any feature with delta significantly nonzero is information the market missed,
**by construction**. No longer competing with the market's information, only
searching for the residual it omits. Strictly more informative than a scalar
beta because it says WHERE the signal is.

Why this moved to the top: `--fixed-train` proved more data does not help and
the in-sample gap proved there is no overfitting, so the binding constraint is
which features carry information the market misses. A scalar beta of 0.275 in
receptions says "27 percent of the deviation is real, always", which is almost
certainly wrong in an interesting way. Delta tells the informative features
from the already-priced ones.

Must be a second-stage model trained only on rows with captured lines, since
the existing modules train on 2022-2024 where no line data exists. Rule of
thumb 50 to 100 rows per feature; four seasons supports 15 or more. Fit AFTER
Phase 3 so the features being tested are the honest versions.

**5.2 Feature snapshotting.** `bets` stores `projection` and `line` but not
`target_share_roll`, `snap_roll` and the rest. Either snapshot features at
import or reconstruct them from `build_dataset` at analysis time. 5.1 needs
this.

**5.3 Variance model for low-integer markets.** FanDuel charges flat -113 both
sides on yardage, which asserts P(over) is about 0.5 for every yardage prop
and adjusts the LINE to make it true. If the line sits at the median, knowing
the variance buys nothing.

That breaks down when the line CANNOT sit at the median. Receptions lines are
2.5, 3.5, 4.5; QB rushing lines are 0.5 to 5.5. Increments are coarse relative
to the distribution, so true P(over) is genuinely 0.42 or 0.58. FanDuel
already knows this: receptions is the one market where they use per-player
odds (breakeven 0.417 to 0.610) instead of flat vig. The question is whether
they price it well.

This is now more attractive than it was, because receptions is the only market
with signal and the only one where the distributional route is open. Current
sigma is a function of the projection alone, so a 10-target slot receiver and
a 4-target deep threat projected at the same yardage get identical spreads.
**Testable offline on four seasons of outcomes with no lines at all.**

**5.4 Multi-book consensus and disagreement as a display.** Comes free in
credit terms. Instead of model-versus-FanDuel, show per-book lines, consensus
(median, robust to one stale book), and disagreement (spread between books).
Disagreement fires on some props and stays quiet on others, which is the
behaviour originally wanted from the edge column. Requirements: consensus only
where 2 or more books have a line; match each book's nearest capture within an
hour or two and flag when it cannot; `PRIMARY_BOOK = "fanduel"` so existing
behaviour is unchanged by default; record which book a bet was placed at in
`bets` so shopping value becomes measurable.

**5.5 Line shopping as a first-class product.** The highest-confidence edge
available and it requires no forecasting skill. Worth 1.7 to 3.2 points of win
probability on 2025's 8-book data, which is 3 to 6 percent of turnover. Note
it adds NO forecast value (the best-line beta was mostly artifact), so it is
purely a price improvement. Restrict to books you actually hold accounts with,
because `best` across all books includes stale and limited prices.

**5.6 Receptions movement in odds space.** 254 of 273 reception props showed a
flat line because FanDuel moves reception prices through the ODDS. Line
Movement measures the wrong quantity for that market and has been blind to its
price action. Compute reception movement in implied probability, the way the
TD section does. More important now that receptions is the one real market.

**5.7 Injury and inactive filter at import, plus the large-move warning.**
nflverse publishes injury reports. Kills stale rows and removes a whole class
of fake edge. The large-move warning matters because the biggest line moves
are usually news the model cannot see. Also the operational edge: a starter
ruled out at 11am Sunday with a backup's line that has not moved is a real
prop edge, and it is about speed rather than modeling.

**5.8 Tier cutoffs.** Re-bucketable from stored edges. Note two pricing
regimes: pre-Sep-9 rows used the old normal pricing with double-counted vig,
but projection and line are both stored so edges are recomputable.

---

## PHASE 6: STRUCTURAL CLEANUP

**6.1** Consolidate the two name normalizers (folded into 3.6).
**6.2** Move `_all_player_stats` into `data_utils`. Now duplicated in
`rushing.py`, `anytime_td.py` and `qb_passing.py`, and the same pattern is
needed for the remaining Phase 3 fixes.
**6.3** Verify the TSV export column order matches the tracker workbook.
**6.4** UTC to Eastern for import stamps, display-time conversion only.
**6.5** Pin pandas, and clean the two bare-integer timedelta deprecations at
`live_capture.py:224` and `fill_weeks.py:194`. Cosmetic today, but the Actions
runner installs fresh pandas every run, so a version bump turns a warning into
a scheduled job that fails silently at 3am.
**6.6** Add `models/__init__.py` (folded into J4).
**6.7** Make `odds_backfill.py` stop on the first 401 rather than grinding
through 24, and rename its "Supabase password" prompt to "app password".

---

## PHASE 7: LONG HORIZON

**7.1 NHL, shots on goal.** Deliberately deferred until the NFL work is
settled. The season opens in early October, so re-running the NHL section of
`odds_api_check.py` once a game is inside 48 hours becomes possible then, and
costs nothing if it returns nothing (Phase 0 returned zero books on all six
prop keys in September, which is probably just "props not posted a week out").

Shots on goal chosen over points and goals for the same reasons receptions
outperforms anytime TD here: event counts are high enough that the outcome is
not luck-dominated (3 to 4 shots a game, mean 2 to 3.5); assists are the least
projectable component of any hockey stat; shots are a role and usage statistic
(ice time, power play deployment, shot rate per minute) which is stable and
autocorrelated, exactly what the EWMA work exploits; and lines sit at 1.5,
2.5, 3.5, a coarse integer grid where the line cannot sit at the median, which
is precisely the 5.3 condition. The negative binomial work from receptions
transfers directly, and receptions is the one market that works here.

Open question before committing: shots on goal counts only shots reaching the
net, not blocked or missed attempts. Check the quality of the free hockey data
ecosystem against nflverse first. A good line dataset with poor stat data is
only half useful.

**7.2 Copula or Monte Carlo for correlated props.** This is where Monte Carlo
genuinely becomes necessary. A same-game parlay of Mahomes over passing and
Kelce over receiving is not the product of two independent probabilities.
Commercially interesting because SGP pricing is where books are weakest:
correlation modeling is hard and they apply crude haircuts. Gate this on a
demonstrated single-prop edge existing first, which it does not yet.

**7.3 Play-by-play engine.** Simulating drive by drive naturally produces
correlated outcomes, game-script effects, and garbage-time dynamics a marginal
model cannot represent. The only item on this list that could produce
genuinely DIFFERENTIATED projections rather than better-fitted versions of
public data. Offseason.

**7.4 Re-run `td_reversion_check.py` at Week 5 and Week 8.** The +0.031
anytime TD movement effect is a lead, not an edge. Direction-only count is
null at 0.472, so the effect may be about move SIZE rather than direction.

**7.5 Traffic dashboard, Patreon or free-tier distribution.** Not before J2
(RLS) and not before there is something worth distributing.

---

## THE HONEST RISK, RESTATED

The September version of this document said: if beta comes back near zero
across three seasons after every Phase 3 fix, that is a definitive answer.

**For rushing and qb_passing, that has now happened.** Beta -0.032 and 0.075,
after the fixes, on four seasons. Those two markets do not beat sharp NFL prop
lines and further feature engineering on public nflverse data will not change
it. Receiving at 0.093 is on the same path.

Receptions survived at 0.275 with t +9.45, and it still does not clear a 6
percent hold on the holdout. So the risk did not fully materialise, but it
came close, and the project is now one market rather than four.

That does not kill it. It redirects it at the things that do not require
beating the market's mean:

- **Line shopping** (5.4, 5.5), the highest-confidence edge, worth 3 to 6
  percent of turnover and requiring no forecasting skill at all
- **Distributional edge** where line increments are too coarse to sit at the
  median (5.3), which is exactly where receptions lives
- **Correlated props** where books price correlation crudely (7.2)
- **Speed on news** where a line has not moved yet (5.7)
- **The line-as-feature model** (5.1), the one route to finding information
  the market misses rather than re-fitting what it already has

All five are supported by the data already purchased.

---

## ONE COST WORTH ACKNOWLEDGING

The "proprietary historical line dataset that compounds in value" framing took
a hit when a version of it turned out to be purchasable for $59. The 6,807
original snapshots are FanDuel-specific and timestamped to a personal capture
schedule, so they are not a moat.

That was a good trade anyway. The moat only mattered if the model worked, and
the data is what told us that three of four models do not. Learning that in a
day for $59 beats two seasons of manual transcription reaching the same
conclusion.

The screenshot-to-CSV transcription workflow is retired. No batching, no
`UNCERTAIN` flags, no cumulative team tracker, no 20-image sessions.

What did turn out to be the asset is the **measurement apparatus**: a harness
that fits nothing on the rows it scores, placebos every claim, inspects the
composition of its own winning bets, and has now caught four survivorship
artifacts including two that presented as t-statistics above +6. That is
harder to buy for $59.
