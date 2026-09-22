# OpalScales Master Plan
### Measurement first, then fixes, then new modeling

Written September 21, 2026, after Week 2 of the 2026 season.

---

## The reasoning behind the ordering

The project is currently **open-loop**. A change to a model cannot be evaluated
against the market for weeks or months, because there are only 596 graded
yardage rows and no historical lines. Every fix in the backlog is unfalsifiable
today.

That is the real bottleneck. Not the bugs, not the pricing layer, not the
model. Three seasons of historical multi-book prop lines converts "we will know
by Week 8" into "we will know in twenty minutes", across roughly 40,000 to
80,000 prop rows instead of 596.

So the data purchase comes before the fixes, and an evaluation harness comes
before the fixes, because unmeasurable work on a model is how people spend
years being wrong.

**Budget: $59, one month of The Odds API, then cancel.**

---

## What is already established (do not redo)

Findings that are settled, so they do not get re-litigated:

- **The pricing assumption is rejected.** `(actual - line) = a + b(projection - line)`
  gives pooled beta -0.079, SE 0.122, CI [-0.319, +0.161], t(beta=1) = **-8.81**.
  All four markets agree. R-squared 0.0008.
- **Out of sample, the projection is worse than the line alone.** RMSE proj
  1.0917 vs RMSE line 1.0253 on Week 1, and 1.0430 vs 0.9784 on Week 2.
- **Yardage line movement is unpredictable by the model.** Pooled b +0.006,
  CI [-0.012, +0.023], toward share 0.505 (p 0.90), flat across all deviation
  buckets.
- **Anytime TD movement is partly real.** Baseline b +0.0512, placebo mean
  +0.0198, so the honest effect is about +0.031. Test 3 with the shared term
  removed gives +0.0253 (t +2.06). Odds series mean-revert (b -0.1947,
  t -2.58), which is the artifact mechanism. Direction-only count is null at
  0.472, so the effect may be about move size rather than direction. **A lead,
  not an edge. Re-run at Week 5 and Week 8.**
- **Range restriction, not structural failure**, explains the within-tier
  correlation collapse. Observed within-tier correlations are roughly what the
  Thorndike correction predicts.
- **Monte Carlo is premature.** Closed-form gamma and negative binomial are
  exact for marginal single-stat distributions and validated to 1 to 2 pp. MC
  becomes necessary only for **joint** distributions (correlated props, SGP) or
  a play-by-play engine.
- **Rejected and closed:** XGBoost, qb_passing feature hunt (two rounds),
  early-season hierarchical shrinkage, postseason exclusion from rolling
  windows, the qb_rushing quintile-4 defect, home/road splits, a standing
  parallel Bayesian system, and the weeks-1-to-4 correction.

Fixes already shipped today:

- `rushing.actual_result` and `anytime_td.actual_result` no longer read
  filtered frames. Surfaced 282 previously invisible graded rows.
- `anytime_td` rolls features before the volume filter and trains on the
  serving population (`touches_roll >= 3`). Backtested: log loss 0.5452 vs
  0.5537, AUC 0.654 vs 0.639, calibration gap +0.3 pp (z +0.39) vs -2.1 (z -2.81).
- `data_utils._try_seasons` raises on a failed season fetch instead of
  silently returning a short frame. nflreadpy filesystem caching enabled.
  Baseline: 78,041 player-stat rows across 2022-2026.
- `db.py` rewritten: one authenticated client per session (fixes the
  single-use refresh token exhaustion that blanked Line Movement),
  pagination past the 1000-row PostgREST cap, non-silent query failures,
  `compute_p_over` added.

---

## JUMP THE QUEUE (do these regardless, they are cheap)

**J1. Do not bet Week 3 on displayed edges.**
Thirty seconds, no code. The measured relationship between tier size and
outcome is inverted at |z| near 4 in both tails. The 0.20-0.40 band came in
+20.1 pp (z +4.84), the 0.60-0.80 band -22.4 pp (z -3.85). Max and Strong are
currently the worst-calibrated bets on the board.

**J2. RLS.** (old item 8)
`authenticated_read_lines` is still `USING (true)`, so any authenticated user
can read the whole lines table. Ten minutes. Note the historical fix: the app
now threads real user sessions through `get_authed_client`, so `auth.uid()`
resolves at the DB and a proper policy will work.

**J3. The `p_over` one-liner in `app.py`.**
In the Line Movement save block, change `"p_over": None,` to
`"p_over": savable["p_over"] if not is_td else None,`. `get_line_movement`
already computes it.

---

## PHASE 0: Validate the API for free (today, $0)

Goal: confirm the data is usable before spending anything.

**0.1** Get a free API key at `the-odds-api.com`. Note there is a lookalike
site at a similar domain; the real one carries an impersonation warning in its
own footer.

**0.2** Write `odds_api_check.py`:
- Hit the `events` endpoint for `americanfootball_nfl` (free, 1 credit)
- Hit `event-odds` for one game with all five market keys
- Confirm `fanduel` and `draftkings` both appear as bookmaker keys
- Confirm the five markets return data: `player_reception_yds`,
  `player_receptions`, `player_rush_yds`, `player_pass_yds`,
  `player_anytime_td`
- Match returned player names against nflverse via `data_utils.norm_join_name`
  and report the unmatched rate
- Compare one line against what FanDuel shows in the app right now
- Log the `x-requests-used` and `x-requests-last` response headers
- **Also confirm NHL**: hit `icehockey_nhl` events and check that
  `player_shots_on_goal` returns data. Other NHL keys seen in the wild are
  `player_points`, `player_assists`, `player_power_play_points`,
  `player_blocked_shots` and `player_goals`, but coverage changes over time so
  verify the exact set rather than assuming. NHL games may not be listed until
  closer to the season opener, in which case check a historical event instead.

**Gate:** if the unmatched name rate is above roughly 5%, or FanDuel props are
missing, stop and reassess before paying. Budget about 30 credits.

**Note:** `player_rush_yds` is one market covering both RB and QB rushing, so
five market keys cover all six app markets.

### Phase 0 RESULT (run September 21, 2026, 5 credits used)

**NFL: PASS.**
- Five books returned: fanduel, draftkings, betmgm, betrivers, betonlineag
- All five market keys returned data. Samples looked correct: Jaxson Dart pass
  yards 213.5 at -113, Theo Johnson receiving 8.5 at -113
- `player_anytime_td` returns `name: "Yes"` with `point: NA`, which matches how
  the app already stores it

**Name matching: PASS, despite the script flagging 21%.** The seven unmatched
names were four team defenses (excluded anyway), two rookies with no stat rows
yet (CJ Daniels, Max Klare), and Odell Beckham Jr., who normalised correctly to
`odell beckham` but has no recent stat row. Real skill-player match rate is 26
of 27. `norm_join_name` is working; the headline rate was an artifact of a
33-name sample where a third are structurally unmatchable.

**NHL: FAIL for now.** All six prop keys returned zero books. See the NHL
question under Phase 1.

---

## PHASE 1: Buy and backfill (this week, $59)

**Credit arithmetic, confirmed from the docs. The formula is sport-agnostic:**

```
live:       markets x regions        credits per event
historical: markets x regions x 10   credits per event
historical events endpoint           1 credit per call
```

There is no sport multiplier. An NHL game with one market costs 10 credits
historically, an NFL game with five costs 50. NHL is only expensive because it
has 1,312 regular season games against the NFL's 272.

`regions=us` returns **every US book in one call**, so multi-book costs nothing
extra. That means all of Phase 4.1 comes free with the ingest. Player prop
history is available after 2023-05-03, and snapshots exist at 5-minute
intervals from September 2022.

Five NFL market keys cover all six app markets, because `player_rush_yds` is
one market containing both RB and QB rushing.

### Budget: 100,000 credits

**Revised after Phase 0.** NFL coverage confirmed excellent: five books
(fanduel, draftkings, betmgm, betrivers, betonlineag) and all five market keys
returning data. NHL returned **zero books on all six prop keys**, so NHL is
held out of the purchase until that resolves. See the NHL note below.

Five books per prop at no extra credit cost also makes the consensus and
disagreement work in Phase 4.1 much stronger than the two books originally
planned for.

| pull | games | snaps | per call | credits |
|---|---|---|---|---|
| NFL 2025 historical, opening + closing | 272 | 2 | 50 | 27,200 |
| NFL 2024 historical, opening + closing | 272 | 2 | 50 | 27,200 |
| NFL 2023 historical, closing | 272 | 1 | 50 | 13,600 |
| NFL playoffs, 3 seasons, closing | 39 | 1 | 50 | 1,950 |
| historical events endpoint, event IDs | | | 1 | ~60 |
| retries, mistakes, live capture | | | | ~5,000 |
| **subtotal** | | | | **~75,000** |
| **reserve, for NHL if it materialises** | | | | **~25,000** |

Two full seasons of historical line **movement** instead of one, which
strengthens the CLV work, plus a 25,000 credit reserve.

Note: prop history starts 2023-05-03, so the 2022 season is **not** available.
2023, 2024 and 2025 is the complete set.

Live capture during the subscription month is nearly free by comparison:

```
NFL live:  5 markets x 1 region = 5 credits per game = 85 per 17-game slate
NHL live:  1 market  x 1 region = 1 credit per game  = ~8 per day
```

| live pattern | credits |
|---|---|
| NFL, 3 captures per week, 5 weeks | ~1,300 |
| NFL, 6 captures per week, 5 weeks | ~2,600 |
| NFL, 12 captures per week, 5 weeks | ~5,100 |

Even twice-daily NFL capture for the whole month fits inside the retry
allowance. During the subscription month, capture as often as convenient.

### The NHL question (unresolved)

Phase 0 probed six NHL prop keys against a September 29 game and every one
returned zero books. The probes cost nothing, since the API does not charge
when there is nothing to return.

Two explanations with very different consequences:

1. **Books have not posted props yet.** Props typically appear a day or two
   before puck drop, not a week out. If this is it, live props appear in
   October and historical props probably exist for past seasons.
2. **The Odds API does not carry NHL player props.** The market lists found
   were from a third-party R wrapper and secondary sources, not the official
   docs, so they may have been over-trusted.

**Resolution, cheap:** re-run the NHL section of `odds_api_check.py` once a
game is within 48 hours. If props appear live, historical almost certainly
exists and the 25,000 reserve buys two seasons of shots on goal. If they still
do not, NHL is out and the reserve goes to more NFL depth or simply unspent.

Do not include NHL in the purchase decision until this resolves.

### Why shots on goal for NHL, if it materialises

Chosen over points and goals because it is the most projectable NHL market, for
the same reasons receptions outperforms anytime TD in this project:

- **Event counts are high enough.** A top forward takes 3 to 4 shots a game, so
  the mean is 2 to 3.5 and the outcome is not luck-dominated. Points average
  0.6 to 1.0, and goals require a shot AND a conversion, which is a rare event
  stacked on a common one.
- **Assists are the least projectable component of any hockey stat.** A point is
  either your own goal or someone else's goal you touched, and the second half
  is close to a coin flip conditional on linemates scoring.
- **Shots are a role and usage statistic**: ice time, power play deployment,
  shot rate per minute. Stable and autocorrelated, which is exactly what the
  EWMA work exploits.
- **Lines sit at 1.5, 2.5, 3.5**, a coarse integer grid relative to the
  distribution, which is precisely the condition in Phase 4.5 where the line
  cannot sit at the median and distributional edge can exist. The negative
  binomial work from receptions transfers directly.

Two seasons of the market that would actually be modelled beats shallow
coverage of two markets.

**Open question to resolve before modelling NHL:** shots on goal counts only
shots reaching the net, not blocked or missed attempts. Check the quality of
the free hockey data ecosystem against nflverse before committing to a model. A
good line dataset with poor stat data is only half useful.

### Steps

**1.1** Subscribe to the $59 / 100,000 credit tier.

**1.2** Schema. Add `book` to the existing `lines` table:
```sql
alter table lines add column book text not null default 'fanduel';
```
Existing 6,807 snapshots become FanDuel rows retroactively. No migration.
Create a separate `historical_lines` table for the backfill so the live table
stays clean, with columns: `sport, book, season, week, market, player, line,
over_odds, under_odds, captured_at, event_id, commence_time, snapshot_label`
where `snapshot_label` is `opening` or `closing`. Include `sport` from the
start so NHL rows coexist rather than needing a later migration.

**1.3** Write `odds_backfill.py`. Requirements:
- **Resumable.** Track completed `(event_id, snapshot_label)` pairs in a table
  or local file. At 50 credits per game a restart is expensive.
- **Credit-aware.** Read `x-requests-used` and `x-requests-last` after every
  call, log burn rate, and hard-stop at a configurable ceiling.
- **Verify before proceeding.** Pull NFL 2025 first and inspect it. A gappy
  2023 is then a cheap discovery rather than an expensive one.
- Closing snapshot: roughly 30 to 60 minutes before `commence_time`.
- Opening snapshot: earliest available, typically Tuesday or Wednesday.
- Handle the empty-response case: a call returning no events is not charged,
  but it should be logged, not silently skipped. Same principle as the
  `_try_seasons` fix.

**1.4** Order: NFL 2025, verify, then NFL 2024, NFL 2023, then NHL 2025-26,
then NHL 2024-25. NFL first because it unblocks Phase 2 and 3, which are the
critical path. NHL is banked data with no immediate dependency.

**1.5** Cancel the subscription **after** the backfill completes and the data
is verified in Supabase, not before. Cancellation is via a form or the
accounts portal and takes effect before the next billing cycle. Do not risk
losing access with unused credits.

**1.6** After cancelling, note that live capture continues to be viable on the
free 500-credit tier, but only for one sport at a time: NFL at 5 markets is
about 320 credits a month for one capture per week, and NHL shots at 1 market
is about 440 a month for one capture per game day. Both together exceed 500.

---

## PHASE 2: The evaluation harness (the centerpiece)

This is the thing that makes every later phase measurable. Build it once,
then run it after every change.

**2.1** `eval_harness.py`. Takes the models as they are, scores three seasons
of games, joins to historical closing lines, and reports per market and pooled:
- **beta** with HC1 and game-clustered standard errors
- **alpha**, the mean-versus-median offset per market
- **calibration** against real lines by probability band
- **RMSE** of three predictors: line alone, projection alone, blend
- **AUC** and log loss
- **residual SD**, which is the correct sigma for pricing around a blended mean

Out of sample by construction: models train on 2022-2024, so score 2025
separately from 2023-2024 and report both.

**2.2** Cluster standard errors by game. (previously noted as a caveat)
Props within one game share information. The historical data includes
`event_id`, so this is now straightforward and makes every CI honest.

**2.3** Establish the baseline. Run the harness before touching any model, and
record the numbers. Everything in Phase 3 is measured against this.

---

## PHASE 3: Bug fixes, each one measured

Every item here gets the harness run before and after. A fix that does not move
the numbers gets reverted or reconsidered rather than kept on faith.

**3.1** Rushing `carries_roll` survivorship. (old item 5)
`build_dataset` filters `carries >= 5` before the rolling mean, so a backup's
workload feature is measured only from weeks the starter was hurt. Inflation is
+6.31 carries in the 0-2 bucket, falling monotonically to +0.06 at 15+. The
training population is also wrong: surviving rows average 54.0 rushing yards
against a true unconditional 32.5, so the intercept is inflated by roughly 66%
for backups. This is the largest known distortion in the codebase and produced
projections of 39.9 for Justice Hill against a 13.5 line (he ran for 6).
Apply the same shape as the `anytime_td` fix: roll over all active games, then
filter, and train on the serving population rather than the in-game-volume one.

**3.2** Refit the rushing sigma and floor. (part of old item 5)
Sigma `1.922 * proj^0.7149` and floor 34.5 were fitted against the OLD
projections. Changing the projection distribution invalidates them. Non-optional
consequence of 3.1.

**3.3** qb_passing `attempts >= 10`. (old item 3)
Same survivorship shape. Benched, injured and blowout games are deleted from
training, removing the left tail, and those bets also silently fail to grade.
Most likely mechanism behind the +8.81 bias that appears in all sixteen
subgroup cells. Note the bias is slope-shaped (rising +1.06 to +10.48 across
quintiles), so a level correction is the wrong fix.

**3.4** `receiving` and `receptions` `actual_result`. (old item 4)
Both return `None` when the stat is null for an active player, where 0 is
correct. Neither `build_dataset` applies a volume filter, so this is smaller
than the rushing and TD versions, but it is the same bug family.

**3.5** WR receptions bias. (old item 7)
+0.30 at **z = +5.39** on 1,456 rows, the largest single measured effect in the
subgroup output and still unfixed. At a 4.5 line with sigma about 2.49 that is
roughly 0.12 sigma, overstating P(over) by about 5 pp. High-volume receptions
is worse at +0.32 (z +3.15) with the worst calibration gap in the file (-4.6).

**3.6** RB receiving as a yards-per-reception problem.
RB receptions bias is -0.16 while RB receiving yards is +3.86 (z +3.59). At
about 7 yards per catch the reception shortfall predicts -1 yard, so the entire
error lives in **ypt, not volume**. Fix the RB ypt feature, not the RB level.

**3.7** The `qb_rushing` market label. (old item 6)
`MARKET_MAP` in `import_lines` has no `qb_rushing` key, so QB rushing props
store as `market = 'rushing'`. The `is_qb_model` flag routes the edge
calculation correctly in memory but never reaches the database, so anything
repricing from `(market, projection, line)` uses the RB sigma and the 34.5
floor instead of `4.624 + 0.7246*proj` and 1.5. Currently zero graded
qb_rushing rows exist.

**3.8** Audit every `build_dataset` filter for survivorship. (old item 17)
Three instances of the same bug turned up by looking at six files. The pattern
is always: a filter applied before a rolling feature, or a filtered frame used
for grading. Worth a systematic pass rather than waiting to trip over the next
one.

**3.9** Investigate the 7-row receiving anomaly.
Predicted 0.218, actual 0.857, z +4.83 in the 0.00-0.25 band. Small n, but
these appeared with the newly graded rows and may be a fourth instance of the
survivorship pattern.

**3.10** Re-estimate beta on the full historical data. (old item 9)
The current beta was measured on projections containing all of the above
errors. This is the honest measurement of where the project stands, and it is
the number that decides how much of Phase 5 is worth doing.

---

## PHASE 4: New modeling, unblocked by the data

**4.1** Multi-book consensus and disagreement. (old item 10)
Comes free in credit terms since `regions=us` returns all books per call.
Display changes shape: instead of model-versus-FanDuel edge, show per-book
lines, **consensus** (median across books, robust to one stale outlier), and
**disagreement** (spread between books). Disagreement becomes the primary
signal, and it naturally fires on some weeks and stays quiet on others, which
is the behaviour originally wanted from the edge column.

Requirements:
- Consensus only where 2+ books have a line; show "1 book" rather than a
  misleading zero
- Match each book's **nearest capture within an hour or two**, and flag when it
  cannot. Comparing a Thursday FanDuel line to a Sunday DraftKings line
  measures the capture schedule, not disagreement.
- `PRIMARY_BOOK = "fanduel"` constant so all existing behaviour is unchanged
  by default and multi-book is additive
- Record which book a bet was placed at in `bets`, so shopping value becomes
  measurable

**4.2** Line shopping as a first-class product.
The highest-confidence edge available, requiring no forecasting skill. Maye at
234.5 on FanDuel against 232.5 and 230.5 elsewhere is a 4-yard spread on one
player. Worth roughly 2 to 3% of turnover. Both uses matter: consensus as a
better mean estimate, and best-available price as directly capturable value.

**4.3** Line-as-feature model. (old item 12)
This is the conceptual unlock. Rather than building a projection from features
and comparing it to the line, put the line in as a regressor:

```
actual = alpha + gamma*line + beta*(projection - line) + sum(delta_i * feature_i)
```

Any feature with delta significantly nonzero is information the market missed,
**by construction**. No longer competing with the market's information, only
searching for the residual it omits. Strictly more informative than the scalar
beta, because it says *where* the signal is.

Must be a **second-stage model** trained only on rows with captured lines,
because the existing modules train on 2022-2024 where no line data exists.
Rule of thumb: 50 to 100 rows per feature. Three seasons supports 15+ features.

**4.4** Feature snapshotting. (old item 13)
`bets` stores `projection` and `line` but not `target_share_roll`, `snap_roll`
and the rest. Either snapshot features at import or reconstruct them from
`build_dataset` at analysis time (deterministic given the data, so both work).
4.3 needs this.

**4.5** Variance model for low-integer markets. (old item 11)
Structural insight worth acting on: FanDuel charges flat -113 both sides on
yardage, which asserts P(over) is about 0.5 for every yardage prop and adjusts
the **line** to make it true. If the line sits at the median, knowing the
variance buys nothing.

That breaks down when the line **cannot** sit at the median. Receptions lines
are 2.5, 3.5, 4.5. QB rushing lines are 0.5 to 5.5. Increments are coarse
relative to the distribution, so true P(over) is genuinely 0.42 or 0.58.
FanDuel already knows this: receptions is the one market where they use
per-player odds (breakeven 0.417 to 0.610) instead of flat vig. The question is
whether they price it well.

Current sigma is a function of the projection alone, so a 10-target slot
receiver and a 4-target deep threat projected at the same yardage get identical
spreads. Model player-specific variance from target depth, target share and
role volatility.

**Testable offline right now** on four seasons of outcomes with no lines at
all: fit a variance model, check whether residual spread is predictable from
pre-game features.

**4.6** Receptions movement in odds space. (old item 2)
254 of 273 reception props showed a flat line. FanDuel moves reception prices
through the **odds**, not the line, so Line Movement measures the wrong
quantity for that market and has been blind to its price action. Compute
reception movement in implied probability, the way the TD section does.

**4.7** Injury and inactive filter at import, plus the large-move warning.
(old item 14)
nflverse publishes injury reports. Kills stale rows like Bowers and removes a
whole class of fake edge. The large-move warning matters because the biggest
line moves are usually news the model cannot see. Also the operational edge:
a starter ruled out at 11am Sunday with a backup's line that has not moved is
a real prop edge, and it is about speed rather than modelling.

**4.8** Receiving alpha.
Receiving outcomes landed **+5.40 yards above the line** on average across two
weeks, independent of anything the model said. Receptions alpha is +0.06 by
contrast. If this survives three seasons, a blind over on receiving beats the
model. Now directly testable rather than a two-week curiosity.

**4.9** Tier cutoffs.
Distribution check and results check, re-bucketable from stored edges. Note the
two pricing regimes: pre-Sep-9 rows used the old normal pricing with
double-counted vig, but projection and line are both stored so edges are
recomputable.

---

## PHASE 5: Fix the pricing layer

**5.1** Price off the blended mean. (old item 1)
In `mc_pricing`, use `line + beta*(projection - line)` as the distribution mean
instead of the projection, with beta configurable and **defaulting to the
measured value**. At beta = 0 every edge collapses to the vig-only baseline,
which is the honest current state. One parameter carries the entire question.

With three seasons of data, beta can be **market-specific** rather than pooled,
which 596 rows could not support.

**5.2** Refit sigma around the blended mean.
The current sigma was fitted as spread of actuals around the model's own
projection, so it includes the projection error. Around a blended mean the
correct spread is different, and the harness reports it directly as the
regression residual SD.

**5.3** Apply alpha per market.
The mean-versus-median offset, estimated from real lines rather than guessed.
This is the rigorous version of the "gamma is over-skewed" hypothesis that the
Week 1 data appeared to support and Week 2 dissolved.

---

## PHASE 6: Structural cleanup

**6.1** Consolidate the two name normalizers. (old item 15)
`db._norm_name` is scalar, `data_utils.norm_join_name` is vectorised, and they
apply different rules. One should call the other so the rules cannot drift.

**6.2** Move `_all_player_stats` into `data_utils`. (old item 16)
Currently duplicated in `rushing.py` and `anytime_td.py`, and the same pattern
is needed for the Phase 3 fixes in other modules.

**6.3** Verify the TSV export column order matches the tracker workbook.

**6.4** UTC to Eastern for import stamps, display-time conversion only.

---

## PHASE 7: Long horizon

**7.1** Copula or Monte Carlo for correlated props. (old item 18)
This is where Monte Carlo genuinely becomes necessary. A same-game parlay of
Mahomes over passing and Kelce over receiving is not the product of two
independent probabilities, because both depend on the same game script, and
there is no closed form for the joint distribution.

Commercially interesting because SGP pricing is where books are **weakest**:
correlation modelling is hard and they apply crude haircuts. A soft corner of
the market that hobby limits can attack. Gate this on a demonstrated
single-prop edge existing first.

**7.2** Play-by-play engine. (old item 19)
Simulating drive by drive and aggregating naturally produces correlated
outcomes, game-script effects, and the garbage-time and blowout dynamics a
marginal model cannot represent. The only item on this list that could produce
genuinely **differentiated** projections rather than better-fitted versions of
public data. Offseason.

**7.3** Re-run `td_reversion_check.py` at Week 5 and Week 8.
The +0.031 TD movement effect is the only surviving positive finding. If it
holds and the direction count turns positive with more rows, it becomes worth
pursuing.

**7.4** Traffic dashboard, Patreon or free-tier distribution.

---

## The honest risk

If beta comes back near zero across three seasons **after** every Phase 3 fix,
that is a definitive answer: these projections do not beat sharp NFL prop
lines, and further feature engineering on public nflverse data will not change
it. Better to learn that in a week for $59 than across two seasons of manual
transcription.

It would not kill the project. It would redirect it at the things that do not
require beating the market's mean:

- **Line shopping** across books (4.1, 4.2), the highest-confidence edge
- **Distributional edge** where line increments are too coarse to sit at the
  median (4.5)
- **Correlated props** where books price correlation crudely (7.1)
- **Speed on news** where a line has not moved yet (4.7)

All four are supported by the data purchase.

---

## One cost worth acknowledging

The "proprietary historical line dataset that compounds in value" framing takes
a hit. A version of it is purchasable for $59. The 6,807 existing snapshots are
still FanDuel-specific and timestamped to a personal capture schedule, but they
are not a moat.

That is a good trade anyway. The moat only mattered if the model worked, and
the data is what tells you whether it can.

Also worth noting: the screenshot-to-CSV transcription workflow can be retired
once the API ingest works. No batching, no `UNCERTAIN` flags, no cumulative
team tracker, no 20-image sessions.
