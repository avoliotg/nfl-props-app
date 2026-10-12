# PRE-REGISTRATION: NHL shots-on-goal price calibration

**Written 2026-10-09, before the data exists.** 40 of 1,321 events are bought
(the pilot). The remaining 1,281 are not. Every number in sections 2 and 3
comes from the 40-event pilot and is fixed here so that nothing in the
analysis can be chosen after seeing the test sample.

Primary hypothesis is monotone and directional. Decided jointly with Ted on
the grounds that a single test with a direction fixed in advance means what it
says, that a monotone miss has a mechanism where a single bad bin does not,
and that it implies a bet. The localised version is demoted to exploratory
and reported against the null-maximum bar, not a bare p-value.

---

## 1. WHY THIS MARKET, AND WHY CALIBRATION

The granularity framework holds that a book expresses its view through the
LINE when the line grid is fine relative to the outcome distribution, and
through the ODDS when it is coarse. Shots on goal is the coarsest grid in the
project: a half-shot step on a 1.5 line is 33 percent of the number, against
15 for receptions and 1.4 for rushing yards.

The framework predicted in advance of any purchase that this market would
price almost entirely through the odds. The pilot confirms it at 2.1 percent
symmetric, against receptions at 5.3 and rushing at 91.4. That is the sixth
correct advance prediction.

Calibration is the primary question rather than encompassing because
calibration requires no model. Every model route in this project has failed:
`BLEND` beta is 0.0 across four markets and four seasons. Rule 1 beats the
line with no projection at all and Rule 2 uses one only as a veto. Starting a
new sport with the encompassing test would mean starting with the single
approach that has never worked here, against a model that does not yet exist.
Calibration needs only the price and the result.

---

## 2. MEASURED CONSTANTS FROM THE PILOT, FIXED HERE

From 523 FanDuel rows across 39 events, spread evenly across 2024-25.

| quantity | value |
|---|---|
| hold (mean raw implied-probability sum) | **1.0651, so 6.11%** |
| hold range across all 523 props | 1.0592 to 1.0718 |
| hold by line value | 6.11 / 6.11 / 6.09 / 6.17 |
| devigged p: mean | 0.5036 |
| devigged p: sd | **0.0736** |
| devigged p: range | 0.3694 to 0.6361 |
| symmetric prices | 2.1% (11 of 523) |
| distinct lines | **1.5 (224), 2.5 (255), 3.5 (42), 4.5 (2)** |
| FanDuel rows per event | 13.4 |

Two notes on these.

**The hold is near-constant.** 6.09 to 6.17 across line values and 1.0592 to
1.0718 across individual props. FanDuel applies a fixed margin and puts all
its variation into the split. Devigging is therefore a well-defined operation
here rather than an approximation, which the whole test depends on.

**Four distinct lines, not three.** The handover of 2026-10-08 recorded three,
from 516 observed lines. A fourth appears at 4.5 on 2 of 523 rows. This is a
footnote, 0.4 percent, but "three distinct lines" was about to become a
repeated claim and is now corrected.

---

## 3. THE PRIMARY TEST, SPECIFIED IN FULL

### Population

- `historical_lines`, `sport = 'NHL'`, **`snapshot_label = 'closing_nhl'`**

  CORRECTED 2026-10-12. This document originally said `'closing'`, which is
  what `odds_backfill.py` writes, and a grouped query on 2026-10-09 confirmed
  3,097 NHL rows under that label. By 2026-10-12 there were none: all NHL
  rows read `'closing_nhl'`. A repo-wide grep returns nothing, so no script
  produces it and it was a manual UPDATE, unattributed. The reason is sound:
  `eval_harness.fetch_lines` filters on `snapshot_label = 'closing'` and has
  **no sport filter**, so the suffix is the only thing keeping NHL rows out
  of every NFL research script. Filtering on the old value returned zero rows
  and cost a run.
- `book = 'fanduel'` only. Cross-book and best-price settlement is a closed
  route, negative at -10.0% against +14.8% FanDuel-only.
- **The 40 pilot events are EXCLUDED.** Bin edges below were chosen by
  looking at pilot data, so the pilot cannot also be the test sample.
- Expected n: **about 17,165 props across 1,281 events.**

**No outcome has been joined to the pilot, deliberately.** The pilot's 523
props could be matched to the outcome cache today for zero credits, giving a
preliminary read on slope and intercept. That was declined. The pilot
established properties of the PRICE only (dispersion, symmetry, hold, line
values), which is what makes the bin edges in section 5 defensible as
price-side design rather than outcome-informed choices. Looking at pilot
outcomes would not contaminate the test sample, which is disjoint, but it
would mean this document was written with outcome data in hand, and the
credibility of the fixed bins rests on it not having been.

### Specification

Linear probability model, outcome on price:

    over_hit = a + b * devigged_p + e

where `over_hit` is 1 if realised shots on goal exceeded the line.
Standard errors clustered on `event_id`, matching project practice. A
logistic fit is reported alongside as a robustness check, not as the primary.

Devigging uses the standard two-sided normalisation already applied in the
pilot queries. Outcomes come from `nhl_sog_20242025.parquet`, cached from
`api-web.nhle.com`, which is an entirely independent source from the Odds API.
**There is therefore no circularity risk here**, unlike the defensive-quality
work where yards allowed came from the same `actual` values that graded the
bet.

### Direction, stated in advance

**Predicted: b < 1.0.** The classic favourite-longshot shape, the book
overstating its own certainty at both extremes. The mechanism is that a book
pricing a coarse grid entirely through the odds has to express its whole view
in a single number per side, with no ability to move the line to the player's
median.

**A slope above 1.0 is a REFUTATION and must not be reinterpreted as success.**
It would mean the book is underconfident at the extremes, which is the
opposite claim and has no mechanism offered here.

### Chronological split, fixed here

Events are split on game date at the median, giving two halves of about 640
events and 8,582 props each. The exact boundary date is whatever the median
turns out to be and is reported.

The reason is this project's own record. `def_quality_test` found a lagged
rank correlation of -0.9000 that replicated at -0.8000 in BOTH the development
and holdout periods, and that replication is the main reason the ordering is
believed at all. A split costs nothing here at 1,321 events and converts "the
slope is 0.84" into "the slope is 0.84 in both halves."

### Pass / fail

Both conditions must hold.

1. **Pooled:** b significantly below 1.0 at two-sided 0.05, clustered on
   `event_id`.
2. **Both halves:** the point estimate of b is below 1.0 in each half
   separately. Point estimate only, not significance, since each half has
   SE(b) about 0.0933 and is underpowered alone.

**FAIL** if the pooled interval includes 1.0, if pooled b is above 1.0, or if
either half's point estimate is above 1.0.

Report b, its clustered standard error, the interval, n, the number of
clusters, the measured intra-game correlation, and b for each half with its
own interval.

**What the conjunction costs and why it is not a substitute.** At a true b of
1.0, the probability that both halves land below 1.0 by chance is 0.25, so the
both-halves condition alone would be a 25 percent false-positive rate. It is
meaningful only conjoined with the pooled test. Joint power at 0.05 pooled and
both halves below 1.0:

| true b | P(both halves < 1) | P(pooled significant) | joint power |
|---|---|---|---|
| 1.00 | 0.250 | 0.025 | 0.006 |
| 0.95 | 0.495 | 0.115 | 0.057 |
| 0.90 | 0.736 | 0.328 | 0.242 |
| 0.85 | 0.895 | 0.623 | 0.557 |
| 0.80 | 0.968 | 0.858 | 0.830 |
| 0.75 | 0.993 | 0.966 | 0.959 |

So the replication condition costs roughly 3 to 10 percent of power in the
range that matters, which is cheap for what it buys.

### The fitted intercept, reported but not a hypothesis

The fitted intercept `a` is reported as a descriptive number, because the
slope's meaning depends on it: b below 1.0 with the intercept near its
calibrated value means something different from b below 1.0 with a large
offset. The first is a confidence miscalibration, the second a confidence
miscalibration sitting on top of a flat bias, and they imply different bets.

**It is NOT a second primary hypothesis** and carries no threshold of its own,
because two primaries make the test a best-of-two and the correction would
have to scale accordingly.

The reason no prior is offered on it is simply that **no outcome has ever been
observed on this market**, so there is no basis for expecting a flat miss in
either direction. Note specifically that the pilot's mean devigged p of 0.5036
does NOT make a flat miss improbable: that figure says the book's prices are
centred, not that realised outcomes match them. A realised over-rate of 0.45
against a mean price of 0.5036 would be a 5-point flat miss and would leave
the price distribution looking exactly as it does now. The centring constrains
the book's stated view, not its accuracy, and accuracy is what is under test.

### Minimum detectable effect, stated before the result

Linear probability model, sigma 0.5, sd(p) 0.0736, n 17,165, inflated for
intra-game clustering at 13.4 props per event:

| assumed intra-game rho | SE(b) | MDE at 80% power | detects b below |
|---|---|---|---|
| 0.00 | 0.0519 | 0.145 | 0.855 |
| 0.02 | 0.0579 | 0.162 | 0.838 |
| **0.05** | **0.0660** | **0.185** | **0.815** |
| 0.10 | 0.0776 | 0.217 | 0.783 |

The realised rho will be measured and reported. A null is not a null without
the MDE printed beside it.

---

## 4. SIGNIFICANCE AND PROFITABILITY ARE SEPARATE QUESTIONS

The project's own lesson is that an ordering is not an edge: a rank
correlation of -0.90 coexisted with four of five bins losing money. The same
discipline applies here, so the profitability arithmetic is fixed in advance.

At a hold of 1.0651, betting the under needs the realised rate below
`1.0651p - 0.0651`, and the over needs it above `1.0651p`. The dead zone
between them is 6.51 points wide at every price.

Under `r = 0.5 + b(p - 0.5)`, the props a given slope actually makes bettable:

| slope b | bet over if p < | bet under if p > | share of props | n per season |
|---|---|---|---|---|
| 0.70 | 0.4108 | 0.5892 | 22.6% | 3,884 |
| 0.80 | 0.3772 | 0.6228 | 9.6% | 1,642 |
| 0.85 | 0.3487 | 0.6513 | 4.0% | 686 |
| 0.90 | 0.3028 | 0.6972 | 0.8% | 128 |
| 0.95 | 0.2172 | 0.7828 | 0.0% | 2 |

Shares use a normal approximation at the pilot's mean and sd, so they are
estimates and will be recomputed on the real distribution.

**The consequence, fixed here:** the detectable range (b below about 0.815)
and the actionable range overlap, but only just. A result at b = 0.85 would be
statistically marginal and would generate roughly 686 bets a season. A result
at b = 0.90 would be real, publishable within the project, and **worth
essentially nothing to bet**, at about 128 props. This must be stated in the
write-up whatever the outcome. A significant slope is not permission to bet.

---

## 5. PRE-SPECIFIED BINS

Edges fixed on the probability scale, not by quantile, so the test cannot
become a search over bin placement. Seven bins. Expected counts from the
pilot distribution, stated before looking:

| bin | expected n |
|---|---|
| [0.00, 0.40) | 1,367 |
| [0.40, 0.44) | 1,959 |
| [0.44, 0.48) | 3,098 |
| [0.48, 0.52) | 3,672 |
| [0.52, 0.56) | 3,263 |
| [0.56, 0.60) | 2,173 |
| [0.60, 1.00) | 1,633 |

Bins are for display and for the exploratory test in section 6. **The primary
test does not use bins at all**, which is the point of making it a slope.

---

## 6. SECONDARY, EXPLICITLY EXPLORATORY

Whether any single bin shows a calibration miss, as a best-of-seven.

Reported against a max-statistic permutation null over the seven bins, not a
bare p-value. The null-maximum schedule gives the relevant context: over 52
broad cells, chance alone manufactures +0.0542 ROI, and over families of 22 to
40 conditioned cells it reaches +0.1176 to +0.1591. Over seven pre-enumerated
bins the bar is lower but must be computed, not assumed.

**This is exploratory and will be labelled as such regardless of outcome.** A
single bin clearing is the shape of the Cleveland cell, which cleared at
p 0.0230, survived holdout, and then dissolved into a defensive-quality
gradient that lost money in four of five bins.

---

## 7. THINGS THAT WOULD INVALIDATE THE TEST

Checked and reported before the primary result is read.

1. **Name-match rate.** The probe matched 56 of 60 distinct book names, 93.3%,
   with all four misses rookies. Unmatched props are excluded, and the
   exclusion rate is reported. **Critically: the distribution of devigged p
   among EXCLUDED props must be compared to the included set.** If rookies sit
   systematically at low p, excluding them truncates the very tail the slope
   test depends on.
2. **Non-regular-season events.** Enumeration found 1,321 events against
   1,312 in the outcome cache, and the final capture is 2025-04-22, past the
   April 17 season end. The surplus is likely playoff games. They will not
   join and are excluded; the count is reported.
3. **Hold stability on the full sample.** If the hold is not near-constant
   across the full season the way it is in the pilot, devigging is no longer
   clean and the specification needs revisiting before the slope is read.
4. **Range truncation.** If the full sample's devigged range is materially
   narrower than the pilot's 0.3694 to 0.6361, the power table above is
   optimistic and the revised MDE is reported.
5. **Reproduction gate.** The analysis script must reproduce the pilot's
   published figures before producing any new number. Hard gate. The
   equivalent caught the `split_qb_rushing` omission in `gap_test.py`.

   **AMENDED 2026-10-12: an EXACT reproduction is no longer possible.**
   The pilot was published as 523 FanDuel rows across 39 events. It now holds
   535 across 40. The difference is one game: the event that had no FanDuel
   rows when the pilot ran gained them afterwards. The tolerances were NOT
   widened until the old numbers passed, because that would defeat the gate.
   Instead the gate now checks what a 12-row growth cannot move, and fails on
   anything structural:
   - hold within 0.0010 of 1.0651
   - sd of devigged p within 0.0040 of 0.0736
   - symmetric share within 1.5 points of 2.1
   - every line value inside the known set {0.5, 1.5, 2.5, 3.5, 4.5}, with
     counts printed but not gated
   - the lowest-id block must SPAN the season. `--spread-events 40` produces
     an even stride, so a block confined to October would be the first 40
     games of the full run and the pilot identification would be wrong.
   The row and event counts are printed against published either way, so the
   discrepancy stays visible rather than absorbed.

---

## 8. PURCHASE PLAN

- Remaining: **1,281 events at 10 credits = 12,810**, plus **200** for
  re-enumeration, since enumeration is not cached between runs. **13,010
  total.**
- Credits at writing: 82,002. After purchase: about 68,992.
- The ceiling drops to 30,000 on 22 October. 13,010 fits either side, so this
  purchase is **not time-pressured**. The NFL historical purchases at 22,800
  are, since they do not fit in 30,000.
- The `sport` labelling condition is **already discharged**: the 40-event
  pilot wrote 3,097 rows, all labelled `NHL`, with NFL's four labels unchanged
  at 354,554 / 5,199 / 1,821 / 40,487. No further limited run is needed.

Command:

    python odds_backfill.py --run --sport nhl --season 2024

The ledger skips the 40 completed events, so this costs 12,810 in odds calls
rather than 13,210.

---

## 9. REPORTING COMMITMENT

Primary and secondary are both reported regardless of outcome, with the MDE
beside any null and the null-maximum bar beside the exploratory result. If the
slope is significant but lands above 0.85, the write-up must say plainly that
the finding is real and not worth betting.

---

## 10. RESOLVED: THE TWO NFL ROW COUNTS

Recorded here because both numbers are correct for different things and a
future session will hit this again.

`historical_lines` with `sport = 'NFL'` and `snapshot_label = 'closing'` holds
**354,554 rows**. The handover of 2026-10-08 records **184,782**. Checked
2026-10-09: there are **zero duplicates**, 354,554 rows distinct on
(season, week, market, player, book, event_id). The difference is entirely
`anytime_td`:

| market | rows |
|---|---|
| anytime_td | 169,772 |
| receiving | 72,826 |
| receptions | 65,474 |
| rushing | 33,885 |
| qb_passing | 12,597 |
| **all five** | **354,554** |
| **four two-sided markets** | **184,782** |

So 184,782 was never the table's row count. It is the count across the four
markets that carry a two-sided price, which is the research population.
`anytime_td` is 47.9 percent of the table on its own, because it has no line
to vary and every book posts one price per player per game.

**Nothing downstream is affected.** No interval widens or narrows. Rule 1 runs
on rushing, one of the four, and stands at +0.0589 on 1,425 bets with
[+0.0115, +0.1075].
