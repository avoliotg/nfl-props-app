# OpalScales Handoff
### Wednesday, September 23, 2026, evening. Week 3 of the 2026 NFL season.

This **supersedes `HANDOFF_2026-09-22_EVENING.md`**. That document is still
correct about the filter findings and the four betas. It is wrong or stale
about the pricing layer, the calibration table, J2, and the file inventory.
Read `OPALSCALES_PLAN.md` for the phased plan and
`NEW_DIRECTIONS_2026-09-23_EVENING.md` for the strategic list.

The short version: the automated capture is live and proven. The receptions
pricing layer has a one-parameter fix with four independent routes to the same
number, and the sigma FLOOR turns out to be the worst single defect in the
project. The candidate rule reproduces exactly and survives a proper
bootstrap at 91 percent of replicates positive, but its weakness is not the
season concentration everyone assumed.

---

## THE HEADLINE

### 1. Receptions sigma should be `sqrt(1.25 * mean)`, with no floor

One parameter. Four independent measurements landed on it:

| source | implied var/mean |
|---|---|
| conditional residuals, 5 consecutive line values | 1.20 to 1.23 |
| `sigma_refit` leave-season-out fit | 1.223, 1.246, 1.274, 1.247 |
| `edge_threshold.py --sigma-mode level`, fitted a and b | 1.17 to 1.30 |
| FanDuel's own alt reception ladders, devigged | 1.38 |

Against a shipped `max(1.143 + 0.2992*mean, 1.7)` which implies 1.93 at a
mean of 1.5 and 1.37 at 3.5.

`prop var` beats the shipped form on every criterion at once: worst
calibration band **1.15 pp against 22.47**, zero-deviation gap -0.47 against
-2.17, log loss better in all four seasons with a paired cluster bootstrap of
[-0.0094, -0.0058]. It also beats both two-parameter forms (`lin var` 1.54 pp,
`power` 1.39 pp), so the imposed exponent of 0.5 costs nothing.

### 2. THE SIGMA FLOOR IS THE WORST THING IN THE PRICING LAYER

| form | worst band gap | share with NB dispersion < 1 |
|---|---|---|
| shipped + floor | **22.47 pp** | 4.5% |
| shipped, no floor | 9.95 pp | 4.0% |
| shipped x c + floor | 22.88 pp | 4.5% |
| **prop var, no floor** | **1.15 pp** | **0.0%** |

Every floored form sits near 22.5. Every unfloored form sits between 1.15 and
2.86. Correcting the LEVEL while keeping the floor does nothing (22.88).

The mechanism, verified analytically. At a low projected mean the floor forces
sigma so wide that the moment match sends the negative binomial dispersion
parameter below 1, which piles mass on the zero atom and collapses P(over):

| mean | line | sigma | dispersion r | P(0) | P(over) |
|---|---|---|---|---|---|
| 0.8 | 0.5 | 1.70 (floored) | 0.31 | 0.675 | **0.325** |
| 0.8 | 0.5 | 1.00 (prop var) | 3.20 | 0.490 | **0.510** |

`prop var` with k = 1.25 can never produce dispersion below about 4, so the
pathology the floor was invented to guard against becomes impossible. **The
floor is not a guard, it is the cause.**

Consequence in realized outcomes: the 0.00 to 0.35 band is 281 props priced
at 0.309 that came in at **0.534**. `prop var` moves that to 0.463.

**This is cross-market, not a receptions problem.** Rushing shows the same
+20.4 pp on 637 rows. In that band, mean dispersion is 0.91 with **82.2
percent of rows below 1**. qb_passing's floor of 159.0 is the extreme case: it
produces a SINGLE calibration band (0.35 to 0.45) across all 1,731 rows, so
that model cannot output a probability outside that range for any player.

### 3. Sigma does NOT depend on deviation. Phase 4.3 is dead

`sigma_by_dev`, C estimated within line value with per-line level constants so
level is fully absorbed:

```
C within line   +0.0000   cluster bootstrap 95% CI [-0.0476, +0.0525]
C pooled        -0.0000   cluster bootstrap 95% CI [-0.0418, +0.0400]
```

No monotone trend at any line value in the descriptive table either. Do not
add a deviation term.

### 4. Sigma is a SHIFT, not a spread

`dP(over)/dsigma` is **negative in every calibration band and in both
distribution families**, pooled -0.23, -0.14, -0.19, -0.25, -0.37. Never
positive.

That means a sigma change moves every band in the SAME direction. The
handoff's over-dispersion pattern (low too low AND high too high) needs the
ends moved in OPPOSITE directions, and only error in the MEAN does that. So
the two findings are about different errors and both can be true:

- too-wide sigma biases every predicted probability down by roughly the same
  amount, a uniform offset
- a monotone tilt across bands is mean error, which is what Phase 4.5's
  shrinkage targets

**Phase 4.5 therefore stands** (see the correction below) but must be fitted
AFTER sigma, or it is fitted partly against a shift it cannot correct.

### 5. The candidate rule reproduces exactly and survives a real bootstrap

`edge_threshold_v5.py --markets receptions --line-value 2.5,3.5 --sigma-sqrt`
reproduces the doc to every digit: **+0.1475, SE 0.0849, t +1.74, 176 bets**,
thresholds 0.090 / 0.090 / 0.095.

Cluster bootstrap that re-runs the ENTIRE protocol per replicate, so
threshold-selection variance is included:

```
point         +0.1475 on 176 bets
boot median   +0.1482
95% CI        [-0.0709, +0.3133]
share > 0     91.2%
placebo       -0.0701, share > 0 24.4%
```

Not established by the project's own standard, so J1 holds and the app stays
silent. But 91 percent positive against a placebo at 24 percent is not
nothing. The interval straddles zero asymmetrically.

### 6. THE RULE'S WEAKNESS IS THE SELECTION STEP, NOT SEASON CONCENTRATION

This inverts what everyone (including me, loudly) assumed from the 38-bet 2025
row at +0.3509.

| dropped | bets | pooled ROI | change |
|---|---|---|---|
| none | 176 | +0.1475 | |
| 2023 | 769 | **-0.0432** | -0.1906 |
| 2024 | 787 | **-0.0795** | -0.2269 |
| **2025** | 207 | **+0.0530** | -0.0944 |

2025 is the LEAST load-bearing of the three. What actually happens when a
season is removed is that a two-season training set picks a different
threshold and the bet count explodes from 176 to 769. **The
threshold-selection argmax needs three seasons to land near 0.090; with two it
lands somewhere that loses money.**

That is a far more specific and more fixable problem than "one lucky season".
It points at replacing the raw argmax with a smoothed curve, a
one-standard-error rule, or a cutoff chosen on calibration rather than on
realized ROI.

Note the jackknife is **sensitivity, not attribution**: dropping a season also
removes it from the TRAINING set for every other season, so the threshold
moves too.

---

## CORRECTIONS TO THE SEPTEMBER 22 HANDOFF

### The calibration table DOES reproduce. There is no sign flip.

The September 22 handoff reports `actual - predicted`. The scripts written
today report `predicted - actual`. That is a units mismatch, not a
contradiction. **Record the convention explicitly in any new table.**

Under `edge_threshold.py --sigma-mode level` the table reproduces almost
exactly:

| band | Sept 22 (act-pred) | today, level mode |
|---|---|---|
| 0.00-0.35 | +7.3 | +7.2 |
| 0.35-0.45 | +2.6 | +1.1 |
| 0.45-0.50 | +1.1 | -0.4 |
| 0.50-0.55 | -0.3 | -1.5 |
| 0.55-0.65 | -4.7 | -5.0 |
| 0.65-1.00 | -11.6 | **-12.7** |

So the over-dispersion finding is real and Phase 4.5 keeps its basis.

### There are THREE sigmas in play, and none of the betting results used the app's

1. the app's `max(1.143 + 0.2992*mean, 1.7)` in `mc_pricing`
2. `edge_threshold`'s **default `--sigma-mode flat`**, the regression residual
   SD, one scalar per market
3. `prop var` / `--sigma-sqrt`

Every ROI figure in the project came from 2 or 3. **The app's sigma has never
been evaluated against money.** The Sept 22 handoff's own note that a flat
residual SD is the wrong functional form applies to `edge_threshold`'s default.

### `--sigma-mode flat` versus `level`, same rows

| | flat | level |
|---|---|---|
| pooled holdout | +0.1056 (197 bets) | **+0.0489 (261 bets)** |
| zero-dev gap | +0.8 pp | **-0.1 pp** |
| worst band | -18.7 | -12.7 |

The `+0.0489` with win rate 0.4751 and t +0.68 is character-for-character the
receptions row in the Sept 22 edge-threshold table, so that table was the
level-mode run.

### The extreme under-heavy tail is a FLAT-SIGMA artifact

| sigma | min edge | bets | % over |
|---|---|---|---|
| flat | 0.10 | 510 | **9%** |
| flat | 0.15 | 111 | **5%** |
| level | 0.10 | 294 | 33% |
| level | 0.15 | 26 | 50% |

A correct sigma produces a balanced tail. This is the tail-composition check
earning its keep.

It does NOT explain the candidate rule's 82 percent unders, because that run
already imposes sqrt. The base population at line values 2.5 and 3.5 is
already **61 percent unders** (39 percent overs at min edge 0.00), and the
edge cutoff sharpens 61 to 82. **The line restriction does most of the work,
not the edge selection.**

---

## THE NEW BUG: `db.get_line_movement` IS 8x WRONG

`lines` holds one row per BOOK per snapshot. `get_line_movement` treats each
ROW as a snapshot. Symptoms seen in the app today:

- Matthew Golden displayed **55 "Line Captures"** when there were 8 distinct
  `captured_at` values for week 3
- counts track book coverage TIMES captures. ATL at GB read 46 to 55 (8 books
  deep); Adonai Mitchell read 4 (1 book)
- the sparkline is 8 books interleaved at each timestamp, not a time series
- `First Line` to `Latest Line` is first ROW to last ROW, and within a
  `captured_at` the book order is arbitrary, so **`Move` and `toward_away` are
  contaminated by book spread rather than measuring movement**

It predates the Actions workflow. The original 6,807 snapshots were FanDuel
only, so counting rows WAS counting captures. The 8-book capture broke the
assumption silently.

### Triage result: the research instrument is CLEAN

`eval_harness.py` collapses books before anything else touches the data:

```
296: keys = ["season", "week", "market", "player"]
297: grp = lines.groupby(keys, sort=False)
299:   "n_books": ("book", "nunique"),
300:   "line_consensus": ("line", "median"),
937: props = collapse_books(lines)
```

All four grouping dimensions, median across books, and `n_books` carried as an
explicit column. `edge_threshold` consumes `line_consensus` from the same
frame. Nothing in either script counts raw line rows. The defect is confined
to one display function.

**NEW AUDIT RULE, a cousin of the filter rule: any function that reads
`lines` must collapse or select on `book` before counting anything.** The
harness could never hide this bug because it tracks `n_books` as a
first-class column; `get_line_movement` has no equivalent.

**Consensus is the MEDIAN, not the mean.** Use median in any display work so
the app and the instrument cannot diverge.

---

## THE FANDUEL ALT LADDER: TIER 1 IS REFUTED

Read off FanDuel for ATL at GB, Thursday week 3, captured 1:39 to 1:40pm ET.
Alt ladders are **OVERS ONLY**, 4 to 10 rungs per player, scrollable.

### The method that works, and the one that does not

A first attempt fitting a distribution plus an overround to three one-sided
rungs is **degenerate**: three equations, three unknowns, exactly identified.
Proof that it is worthless: Skyy Moore and Olamide Zaccheaus have ladders
within two cents of each other and the fits returned mean 1.18 against 0.50
and hold 43 percent against 267 percent.

What rescues it: **a receptions main line is a half integer, so "over 4.5" and
the alt rung "5+" are the SAME EVENT.** The main line is two-sided, so its vig
strips out exactly and pins FanDuel's true probability at one rung.

### The alt ladder is a strictly worse way to place the same bet

| player | line | main over | same rung, alt | fair P | alt hold |
|---|---|---|---|---|---|
| Watson | 4.5 | -130 | -136 | 0.5282 | 9.1% |
| Golden | 4.5 | +102 | -102 | 0.4621 | 9.3% |
| London | 5.5 | +106 | +104 | 0.4542 | 7.9% |
| Pitts | 3.5 | -140 | -144 | 0.5458 | 8.1% |
| Kraft | 3.5 | -154 | -158 | 0.5670 | 8.0% |
| Dotson | 1.5 | -162 | -166 | 0.5785 | 7.9% |

Twelve players. **Alt price worse on eleven, equal on one, never better.**
Mean alt hold 8.4 percent (range 6.9 to 9.8) against 7.0 on the main line.

### FanDuel prices the whole ladder off one distribution

With the hold MEASURED rather than fitted, a single negative binomial fits
every ladder to within **1.3 pp at every rung**, on 4 to 9 rungs per player.
Watson's eight residuals: -0.5, +0.4, +0.8, -0.6, -0.6, -0.8, +0.7, +1.3.

No drift to exploit. And the far rungs are **worse** than a single NB, not
better: pooled residuals +0.64, +1.05, +1.25 pp at four, five and six rungs
past the main line. FanDuel's devigged probability sits ABOVE the fitted
distribution there, meaning shorter odds than a pure count model justifies on
top of a higher hold.

**The Tier 1 premise that thin volume leaves the upper rungs undefended is
refuted on this game. FanDuel defends them by charging more.**

### What survived, with no math at all

**Template pricing at the bottom of the board.** Skyy Moore is a Packer,
Olamide Zaccheaus is a Falcon, different offenses and different roles:

| market | Skyy Moore | Zaccheaus |
|---|---|---|
| receiving yards 15+/20+/25+ | +130 / +194 / +280 | +132 / +194 / +280 |
| receptions 2+/3+/4+ | +126 / +370 / +920 | +134 / +400 / +920 |

Jonnu Smith and MarShawn Lloyd both sit at +880 for 4 or more receptions.
FanDuel is not pricing the bottom of the board player by player, it stamps a
template below some threshold. That is the least defended surface visible, and
it was found by taking a screenshot rather than by fitting anything.

---

## ALPHA BY LINE VALUE: THE HYPOTHESIS IS REFUTED

Driven by the ladder read implying mean-minus-line of +0.52 for Pitts and
+0.61 for Kraft at a 3.5 line, against a pooled receptions alpha of +0.12.
Twelve players on one game, so a lead.

`alpha_by_line_v4.py`, 7,850 rows on FanDuel lines, 845 game clusters.

**Sanity passed**: alpha +0.1171 against the handoff's +0.12, over rate
0.4718 against 0.4799.

**Verdict REFUTE**: alpha at 2.5 is **+0.141**, at 3.5 is **+0.097**. Both
within 0.05 of the pooled +0.12. The two-player read did not generalise, and
the candidate rule's under tail is NOT explained by an alpha error in its own
line band.

### What the run found anyway

**Alpha is not a stable parameter at half the line values.** Chi-square
homogeneity across seasons:

| line | 2023 | 2024 | 2025 | 2026 | p | verdict |
|---|---|---|---|---|---|---|
| 1.5 | +0.337 | +0.369 | +0.298 | +0.273 | 0.877 | stable |
| 2.5 | +0.288 | +0.165 | -0.036 | +0.359 | **0.003** | DISAGREES |
| 3.5 | +0.064 | +0.107 | +0.145 | -0.067 | 0.789 | stable |
| 4.5 | +0.317 | +0.015 | -0.164 | -0.231 | **0.010** | DISAGREES |
| 5.5 | +0.175 | +0.316 | -0.613 | . | **0.000** | DISAGREES |
| 6.5 | +0.286 | -0.176 | +0.244 | . | 0.406 | stable |

So a fixed per-line alpha is defensible at **two of six** values. That
constrains Phase 4.1 more than expected.

**The 1.5 line's apparent pricing error is beta leaking in, not a pricing
error.** All rows: alpha +0.326, gap -7.73 pp at z -5.47. Zero deviation:
alpha +0.191, gap -1.76 pp at z -0.63. The zero-dev panel has the power to
see a -7.7 gap (its gap SE is about 2.8 pp) and does not see one. That
corroborates the low-line beta finding from a new direction.

**Untested lead, and my omission.** When FanDuel's line differs from
consensus, alpha is higher at five of six line values, and at 1.5 it is
**+0.804 against +0.297**. The mechanical effect of FanDuel sitting 0.045
below consensus accounts for 0.045 of that, not 0.5. But step 5 prints no
standard errors and the differs cell at 1.5 is about 69 rows. Needs SEs before
anyone reads it.

**receptions legitimately has no `build_all_rows` sibling** because
`build_dataset` applies no in-game filter (confirmed at 22,540 rows, 0 nulls,
6,060 genuine zeros). So `--score-population all` and `board` are identical
for this market. Do not spend a run comparing them.

---

## MY ERRORS TODAY, RECORDED SO THEY ARE NOT REPEATED

Seven wrong claims and three tool defects. The tool defects were caught by
synthetic tests before output was shown. Error 7 was not, and it is the one
that mattered.

1. **Said the sigma floor binds on "most of the board."** It binds on 15.6
   percent of FanDuel-line rows.
2. **Said reception lines cluster hard at 1.5 in the FanDuel set.** 2.5 is the
   biggest bucket at 2,434 rows.
3. **Claimed a sign flip between my scripts and the handoff.** It was a units
   mismatch, act-pred against pred-act. Checkable in one minute and I raised
   it as a substantive conflict instead.
4. **Claimed the handoff's calibration table does not reproduce.** It
   reproduces almost exactly under level mode. `calib_reconcile` priced with
   the APP's floored sigma, which is a third sigma, and I read the difference
   as the handoff being wrong.
5. **Claimed the +0.1475 rests on a broken pricing layer.** The candidate rule
   already imposes sqrt. Today's sigma work CONFIRMS a choice already made
   rather than invalidating anything.
6. **Pre-registered three options for `calib_reconcile`, none of which
   contained the truth**, because I wrote them before working out the sign of
   dP/dsigma. The point of pre-registering is to constrain interpretation
   afterward, and mine could not.
7. **Shipped `holdout_bootstrap` v1 with a protocol that did not reproduce
   v5, and presented the output with numbers attached.** Three mismatches:
   TRAIN_MIN 30 against v5's 100, no TEST_MIN 20 drop rule, GRID top 0.20
   against 0.25. The smoke test verified the bootstrap machinery detects a
   planted edge and rejects a null, which it does. It never checked that the
   protocol reproduces v5 on the same input, and that check is one line.

Also: recommended shipping the sigma change and THEN re-running
`edge_threshold`, which is testing after changing production code. And said
"writing it now" and then did not.

### Tool defects found by synthetic tests, before any output was shown

- **`alpha_by_line`**: a fixed 0.30 spread threshold flagged 2 of 4 line
  values as unstable on data whose truth was flat by construction. At 300 rows
  per season-line the SE is about 0.14, so a 0.30 spread is two standard
  errors. Replaced with a chi-square homogeneity test against the per-season
  SEs.
- **`sigma_by_dev`**: the within-line test counted signs, and reported 4 of 5
  line values positive on data with C = 0 planted. With five lines that
  happens about 19 percent of the time. Replaced with a profile-likelihood C
  estimated with per-line level constants.
- **`sigma_refit`**: the decision rule required beating shipped on log loss in
  EVERY season, and that **rejected the planted truth**. Log loss is nearly
  blind to sigma here: all eight forms landed within 0.0005 of each other on
  data where the truth was one of the candidates, because P(over) at a
  half-integer line is driven by the mean. Calibration is now primary. The
  log-loss guard was ALSO backwards, requiring the bootstrap upper bound at or
  below zero, which demands a significant improvement rather than the absence
  of a significant loss.

### The lesson worth keeping

**A synthetic test of the machinery is not a test of the measurement.** Three
defects were caught because the rig planted a known answer. The one that got
through was the one where the right check was to reproduce a number already
printed on screen. When a script reimplements an existing procedure, the first
test is that it reproduces that procedure's published output, and it should
refuse to run otherwise. `holdout_bootstrap_v2` now does exactly that.

---

## OPERATIONAL: THE CAPTURE IS LIVE AND PROVEN

### GitHub Actions

Five repo secrets added (`ODDS_API_KEY`, `SUPABASE_URL`, `SUPABASE_KEY`,
`OPAL_EMAIL`, `OPAL_PASSWORD`). Manual run green:

```
wrote 1783 rows at 2026-09-22T20:09:03Z  (4:09pm ET)
credits used 43341, remaining 56659, this run 32, calls 18
```

Cron then proven on four scheduled runs with no intervention:

| captured_at (UTC) | rows |
|---|---|
| 2026-09-23 09:34:10 | 2414 |
| 2026-09-23 05:33:21 | 2366 |
| 2026-09-23 01:32:46 | 2202 |
| 2026-09-22 21:32:05 | 1826 |

**Two things to know.** Runs land at :32 to :34 rather than :23, a consistent
9 to 11 minutes of best-effort queueing drift. The Sunday 12:23pm ET capture
therefore lands about 26 minutes before a 1pm kickoff rather than 37. Still
inside the 20 to 90 minute window the 354,554 historical rows sit in, so the
instrument comparison is intact. Worth knowing, not worth fixing.

And row counts climbed 1,826 to 2,414 overnight, up 32 percent from the
1,783 manual run. Books post Tuesday afternoon through Wednesday. That is now
measured rather than assumed.

### J2 IS CLOSED. It was already done.

`lines` already carries the right pair of policies:

| policyname | cmd | qual |
|---|---|---|
| `authenticated_read_lines` | SELECT | `true` |
| `admin_write_lines` | ALL | `auth.uid() = '814531de-...'` |

Read is open to authenticated users, which Ted confirmed is **intentional**:
Line Movement is meant to be visible to the friend group, and the 17
registered users are all people he knows. Writes are owner-only by hardcoded
UUID, which is the correct pattern here precisely because `lines` has **no
`user_id` column** to compare against. Permissive policies OR together, but
the read policy is SELECT-only so it cannot grant writes.

This also explains why the Actions job writes fine: it signs in as Ted's
account and `auth.uid()` matches.

`auth.uid()` confirmed resolving in a real app session via a temporary
`whoami()` RPC, returning `814531de-0354-4f6d-856f-959543c7fa74`. **Remove
the temporary `st.write` line from the Line Movement tab in `app.py`.**

Revisit only if the app is opened to people Ted does not know, at which point
the signup gate is the lever, not the policy.

### Small things

- `tabulate` was in `requirements.txt` but missing from the local `.venv`.
  Installed. It only fires when opening the Copy-as-text expander in Markdown
  mode.
- Event 17 in the week 3 capture (Pittsburgh at Cleveland) returned no week
  label where the other sixteen showed wk3. It wrote zero rows so nothing
  landed badly, but worth a look if it returns data with a null week later.
- `models/` still has no `__init__.py`, and namespace resolution caused a
  surprise twice today. `import models.data_utils` works from the repo root;
  `import data_utils` does not. That is plan item J4 and it is now earned.

---

## FILE INVENTORY (the Sept 22 handoff is badly out of date here)

### Written today, in the repo

| file | purpose |
|---|---|
| `alpha_by_line_v4.py` | alpha, over rate, pricing gap and dispersion by EXACT line value, with a chi-square season-homogeneity test. Reports `a_match`, the Phase 4.1 parameter per line. |
| `sigma_by_dev.py` | tests whether sigma depends on `abs(dev)`. Returned LEVEL ONLY. |
| `sigma_refit.py` | eight candidate sigma forms, leave-season-out, scored on calibration and proper scoring rules. Selected `prop var`. |
| `calib_reconcile.py` | reconciles the calibration table across sigmas and families; computes the sign of dP/dsigma per band. |
| `holdout_bootstrap_v2.py` | cluster bootstrap of v5's holdout with selection variance. **Refuses to run unless it reproduces v5 first.** |
| `smoke_alpha.py`, `smoke_sigma.py`, `smoke_refit.py`, `smoke_calib.py` | synthetic rigs with planted answers. No database, no repo needed. Run after any edit. |
| `priced_2p5_3p5.csv` | v5's priced rows for the candidate rule, input to the bootstrap |
| `holdout_src.txt` | extract of v5's `holdout()` for reference |

### Already existed but NOT in the Sept 22 handoff

`edge_threshold_v2.py` through **`edge_threshold_v5.py`**, all written the
morning of September 23. **v5 is the one that matters.** It has
`--line-value`, `--sigma-sqrt`, `--curse-correct`, `--no-model`,
`--same-line`, `--line-quartile`, `--freq-quartile`, `--line-min/max`. The
September 22 `edge_threshold.py` cannot reproduce the candidate rule.

### Not in the repo (analysis run in the chat container only)

`ladder_fit.py`, `ladder_overlap.py`, `ladder_shape.py`. The FanDuel ladder
arithmetic. Worth re-creating in the repo if the ladder work resumes, because
the overlap-rung method is the reusable part.

### v5 internals worth knowing without re-reading 1,235 lines

```
GRID      = np.round(np.arange(0.0, 0.2501, 0.005), 4)    # 51 cutoffs
holdout() training floor:  len(g) < 100  -> skip that cutoff
holdout() test floor:      len(g) < 20   -> DROP that season entirely
selection: plain argmax of mean profit on the other seasons
weeks_by_season: display only, does not affect selection
main() is guarded, so importing v5 is safe
```

---

## ESTABLISHED FINDINGS, CUMULATIVE

Unchanged from September 22 and NOT re-litigated: receptions is the only
market with signal (beta 0.275); rushing -0.032 and qb_passing 0.075 carry
none; receiving 0.093; pooled beta must not be used in pricing; the raw
projection is worse than the line in every market; more training data does not
help and there is no overfitting; the alpha BETTING edge is dead while alpha
remains a valid pricing parameter; line shopping adds no forecast value; the
lead-time re-pull is gated off; early-season beta is higher not lower; Monte
Carlo is premature.

Added or revised today:

- **Sigma for receptions is `sqrt(1.25 * mean)`, no floor.** Four independent
  routes.
- **The sigma floor is the largest single defect in the pricing layer**, worth
  20+ points of calibration in the bottom band, across three markets.
- **Sigma does not depend on deviation.** Phase 4.3 closed.
- **dP/dsigma is negative everywhere**, so sigma shifts rather than spreads,
  and the over-dispersion tilt is mean error.
- **The Sept 22 calibration table reproduces** under level mode. No sign flip.
- **Three sigmas exist** and no betting result has ever used the app's.
- **The extreme under-heavy tail was a flat-sigma artifact**; the 61 percent
  base rate at 2.5 and 3.5 explains most of the rest.
- **The candidate rule's weakness is the threshold-selection step needing
  three seasons**, not season concentration.
- **The threshold search carries no upward bias** (placebo -0.0701, 24.4
  percent positive), because the 100-bet floor keeps it off thin cells.
- **FanDuel's alt ladders are internally consistent and priced worse than the
  main line.** Tier 1 refuted.
- **Alpha by line value is not stable at 4 of 6 values.**
- **`db.get_line_movement` is 8x wrong** on capture counts and its Move column
  is contaminated. The research instrument is clean.

### Every arm of the candidate rule loses in 2023 and wins in 2024 and 2025

| arm | 2023 | 2024 | 2025 | pooled |
|---|---|---|---|---|
| full | +0.0373 (71) | +0.1488 (67) | +0.3509 (38) | +0.1475 |
| `--no-model` | **-0.0222** (129) | +0.2004 (65) | (16, dropped) | +0.0524 |
| `--same-line` | **-0.0774** (84) | +0.1705 (51) | +0.2605 (42) | +0.0742 |

Beta is stable across seasons (0.184, 0.143, 0.213), so this is not forecast
quality. **Leading candidate: book composition.** 2023 and 2024 had eleven
books including three defunct brands (`pointsbetus`, `barstool`, `unibet_us`),
so `line_consensus` in 2023 is a different quantity than in 2025, and mean
book spread was 4.67 all-seasons against 2.09 in 2025. If the BOOK component
of the edge is driven by consensus composition, the 2023 loss is mechanical.
**Cheap test nobody has run: restrict to books present in all four seasons and
re-run the per-season betas and the holdout.**

### A red flag in the pricing code

With `--no-model` (beta forced to zero, so the model contributes nothing), the
unconditional calibration collapses to three bands and **147 rows are priced
above 0.65**, delivering 0.5714. With no model input, P(over) should sit near
0.47 for every prop. The only thing that can push it to 0.68 is the bet line
differing from consensus. Worth understanding before trusting the book
component of any edge.

### Edge decomposition at 2.5 and 3.5

```
alpha                        0.106
MODEL  beta*(proj-consensus) 0.108
BOOK   consensus-bet line    0.071
props where bet line differs from consensus: 298 of 4234 (7.0%)
```

Neither component survives alone: model only +0.0742 (t +0.84), book only
+0.0524 (t +0.66), together +0.1475 (t +1.74).

---

## NEXT SESSION: START HERE

### 0. Verify the Sunday pre-kickoff captures. Time-sensitive, nothing to build.

The 12:23, 15:23 and 19:23 UTC-offset Sunday runs are how 2026 becomes
measurable on the same instrument as the 354,554 historical rows. The cron is
proven on the 4-hour grid but the Sunday-specific entries have not fired yet.
Check on Monday:

```sql
select captured_at, count(*) from lines
where captured_at > '2026-09-27' group by captured_at order by captured_at;
```

Expect three additional timestamps on Sunday beyond the 4-hourly ones.

### 1. Fix `db.get_line_movement`. The display redesign is parked; the bug is not.

Collapse or select on `book` before counting. Median across books to match the
instrument. Add an `n_books` column so the defect cannot recur silently. The
`Move` and `toward_away` columns have been wrong for as long as the capture
has been 8 books deep, and they are columns Ted has looked at while choosing
bets.

Also remove the temporary `whoami` `st.write` from the Line Movement tab.

### 2. Ship the sigma fix, and handle the floor deliberately.

`sqrt(1.25 * mean)` for receptions in `mc_pricing`. Three cautions:

- do NOT paste `sigma_refit`'s A and B in directly. They come from a normal
  likelihood on a count residual, which is fine for estimating a variance
  function and wrong for setting a floor.
- the floor needs its own decision, informed by the `size<1` share. Under
  `prop var` that share is 0.0 percent, which is the argument for removing it
  rather than retuning it.
- `mc_pricing` has `PI_CAP = 0.35` and `USE_STAGE_MULTIPLIER` which none of
  today's scripts exercised. `sigma_refit` REIMPLEMENTED the moment match
  rather than calling `mc_pricing`, so agreement between them is **assumed,
  not verified**. That is exactly how `get_line_movement` drifted. Verify
  before trusting.

Then re-run `edge_threshold_v5` and confirm nothing moved unexpectedly.

### 3. The same floor problem in rushing and qb_passing.

Rushing's floor of 34.5 produces +20.4 pp in the bottom band on 637 rows.
qb_passing's 159.0 collapses the entire market into one calibration band.
Neither will become profitable (betas are zero) but the app currently displays
confidently wrong probabilities on the low end of every board. The fix is the
same shape: a variance form that cannot drive the dispersion parameter below
1, rather than a floor.

### 4. The threshold-selection step. This is the candidate rule's real weakness.

The argmax over 51 cutoffs needs three seasons to land near 0.090. With two it
picks something that loses on 769 bets. Candidates: a one-standard-error rule,
a smoothed curve, or choosing the cutoff on calibration rather than on
realized ROI. This is worth more than another backtest of the same rule,
because it is the part that will not generalise to 2026.

### 5. Phase 3.4, the `targets_roll >= 3` gate in receiving and receptions.

**STILL NOT DONE.** It was the headline item in the September 22 handoff and
the whole day went elsewhere. Unchanged in importance: it is the same
population mismatch that erased two betas, sitting in the one market with
signal. Both modules have honest evaluation populations, so only the training
gate moves. Harness before and after, paired on matched rows, revert if beta
falls materially below 0.275 (not at exactly 0.275, which is inside one SE).

Note the reason a revert rule is legitimate here and was not for rushing: the
evaluation population does not change, so a beta drop is a genuine loss of
skill rather than a revealed artifact.

### 6. The 281-row bottom band, still +7.0 pp under `prop var`.

Down from +22.5 but now the largest remaining receptions miscalibration.
Find out whether those rows are specifically the 0.5 and 1.5 lines before
guessing at a fix.

### 7. Book composition across seasons.

`line_consensus` is a median over a changing book set: eleven books in 2023
and 2024 including three defunct brands, eight in 2025. Restrict to books
present in all four seasons and re-run the per-season betas and the candidate
rule's holdout. This is the leading explanation for every arm losing in 2023.

### 8. Standard errors in `alpha_by_line` step 5.

My omission. The FanDuel-differs column shows alpha higher at five of six line
values and +0.804 against +0.297 at 1.5, but with no SEs and roughly 69 rows
in that cell it cannot be read. It bears on the 7 percent concentration
problem, so it is worth doing properly.

### 9. Then Phase 4.5, and only then.

The over-dispersion tilt is real and reproduces. It is mean error, not sigma.
Fit the shrinkage AFTER the sigma fix, or it absorbs a shift it cannot
correct. That is also the most likely mechanical explanation for the curse
correction coming back unstable at slopes 0.995, 0.849, 0.688, 0.852.

### Do not bother with

- adding the alternate market key to the capture. The ladders are internally
  consistent, priced worse than the main line at the overlapping rung, and
  worse than a single negative binomial at the far rungs.
- a deviation term in sigma. Measured at zero.
- re-pulling the 38 mis-timed snapshots. Still gated.
- more feature engineering on public nflverse data before the line-as-feature
  model.

---

## WORKING PRINCIPLES (Ted's, earned)

Unchanged: hard reboot before concluding a change did not work; always
`encoding='utf-8'` in parse checks; never use em dashes; test, do not assume;
gate before sweeping; a t-statistic of +6 is a bug until proven otherwise;
plain-English reasoning before design decisions; one command at a time.

Added today:

- **Import, do not restate.** Both of today's worst errors came from
  reimplementing something that already existed: `get_line_movement`
  paraphrasing the harness's book collapse, and `holdout_bootstrap` v1
  paraphrasing v5's selection rule. Import the constant and call the function.
- **A new script that reimplements an existing procedure must reproduce that
  procedure's published output before it is allowed to produce a new number.**
  Make it abort, not warn.
- **Record the sign convention in every calibration table.** One units
  mismatch cost an hour and produced a confident wrong conclusion.
- **Check which parameters a result was computed under before comparing two
  results.** Three sigmas were in play all afternoon and I compared across
  them twice.
- **`ast.parse` proves a file is valid Python, not that it is the file you
  meant.** Grep for a symbol you know is new. Sort downloads by date, not by
  the number in the filename.

---

## THE HONEST FRAME

Yesterday the project knew which single market has signal. Today it knows what
its pricing layer was doing wrong, and the answer is a single parameter with
four independent confirmations plus a floor that was actively causing the
pathology it was meant to prevent.

The candidate rule came through better than expected. It reproduces exactly,
its threshold search carries no upward bias, and 91 percent of bootstrap
replicates come back positive against a placebo at 24 percent. It is still not
established, and the app still stays silent. But the reason it is not
established turned out to be a fixable property of the selection step rather
than one lucky season, which is a better problem than the one we thought we
had.

Two leads died cleanly. The alt ladder is not soft: FanDuel prices it off one
distribution and charges more for the privilege. And the alpha-by-line
hypothesis, which was mine and which I argued for, was refuted by the data in
under an hour.

What is left that is genuinely untested and cheap: the book composition across
seasons, which would explain the 2023 pattern in all three arms, and the
template pricing at the bottom of FanDuel's board, which was visible in a
screenshot and required no model at all.

The apparatus is still the asset, and today it caught three of its own defects
before they reached a conclusion. The fourth reached a conclusion with numbers
attached, and the fix for that class of error is now written into
`holdout_bootstrap_v2` as a hard gate. Use it as the template: a measurement
script should refuse to run when it cannot reproduce what it claims to
measure.
