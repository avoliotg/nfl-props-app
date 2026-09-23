# OpalScales: New Directions
### Written the evening of Tuesday, September 22, 2026

Companion to `HANDOFF_2026-09-22_EVENING.md`, which covers measured state.
This document is about where to go next. It exists because the handoff ends on
a narrow finding (one market with signal, no established threshold) and that
finding was treated as a conclusion when it was only a measurement.

---

## THE REFRAME

Every test so far asked the same question: **does my model beat the closing
line?** Three of four markets answered no.

That was the wrong primary question. It assumes the model has to be the source
of the edge. Almost nobody who profits from props works that way. The standard
approach is to find the sharpest available price, treat it as the best estimate
of truth, and bet the books that disagree with it. The model becomes a filter
on top of a price signal rather than the signal itself.

**You have 354,554 rows across 8 books and have never once asked which book is
best.** That is the gap. Everything below either exploits price structure the
model was never involved in, or fixes a measurement that was asking the model
to do too much on its own.

Three things also worth carrying forward from last night:

- The app may already be useful as a research surface even where the automated
  edge number is not trustworthy. A projection at corr 0.487 sitting next to a
  line is a legitimate reference point for someone doing their own thinking.
  Those are compatible facts and the measurements only speak to one of them.
- "Success stories" from the friend group are encouraging but not evidence.
  Week 1 was 7-5 and +6 percent ROI with most of the profit from one boosted
  Maye bet at +133, and the two markets that produced it, qb_passing and
  rushing, are the two now measured at zero. That is what good luck looks
  like. It is worth finding out what they were actually using: following high
  tiers was luck, but using the app to notice a player or sanity-check an odd
  line is a use case nothing measured so far touches.
- The instinct that kept being right last night was the one that refused the
  framing. The capture schedule should be clock-driven, which the Tuesday
  posting curve confirmed. The soft-market hypothesis had never been tested,
  which turned out to be true. Both came from pushing back, not from the
  analysis.

---

## THE TWELVE

### 1. Rank the 8 books by predictive power, then bet the others against the winner

**The highest-value test available, and it needs nothing you do not have.**

Two measurements. First, compute beta using each book's line as the reference
instead of FanDuel. The book against which your model has the LOWEST beta is
the sharpest book, because it is the one leaving least information on the
table. Second, and more important: does book A's deviation from book B predict
the outcome? If a weak book sitting 3 points off the sharp book reliably
resolves toward the sharp book, that is a bet with no model in it at all.

The sharpest book then becomes your reference line, permanently. Your model
stops being the thing that decides and becomes a filter.

Free. Runs on `lines_cache.parquet`. No credits.

Expected shape of the answer: one or two books price sharply and the rest are
followers, which is how every market of this kind looks.

### 2. Add the `eu` region and get Pinnacle

`live_capture.py` and `odds_backfill.py` pull `regions=us`. **Pinnacle is
available under `eu`.** Pinnacle is the sharpest book in the world: low margin,
high limits, and they welcome sharp action rather than limiting it, which is
precisely why their price is the best public probability estimate that exists.

A devigged Pinnacle line against a FanDuel line is a professional-grade signal
requiring no model whatsoever.

Cost: credits scale with regions, so roughly 160 per slate instead of 80.
Against ~16,000 a month of ongoing capture that is nothing. Add `us2` in the
same change for ESPN Bet, Fanatics and Hard Rock, which widens the reference
set and the shopping surface together.

One-line change in two files. Do this early: the reference-book work in item 1
gets much stronger with Pinnacle in the set.

### 3. Openers versus closes, and switch the metric to CLV

The automated capture makes this measurable for the first time. The question:
do lines move toward the truth between posting and kickoff? If they do, **the
opener is the soft number and the correct play is to bet early.**

That inverts the old rule. "Wait for the line to move toward the model as
confirmation" meant the confirmation only existed once the price had already
deteriorated, so every bet was entered late at a worse number. The signal may
have been real; the rule built on it could not capture it.

**Change the feedback metric from ROI to closing line value.** Did the number
you took beat the closing consensus? Two hundred graded bets tell you almost
nothing about ROI because outcome variance swamps everything, and they tell you
a great deal about CLV because the noise is tiny by comparison. This is why
sharps track CLV: it converges in weeks rather than seasons.

The multi-snapshot capture now running is exactly the dataset CLV needs. That
capability did not exist before yesterday.

### 4. Stale line detection

FanDuel at 50.5 against a seven-book median of 47.5 is a bet on the market
being right and FanDuel being slow. Bucket every prop by how far each book sits
from consensus, then count realized win rates by bucket.

No model involved. Free, existing rows, runs today. If props where FanDuel is
2 or more points off consensus win at 54 percent, that is a complete selection
rule with an obvious mechanism.

### 5. Attention stratification

`attention_strata.py` is written and needs committing. Splits every market
three ways: how many books quoted the prop (1-2 through 7-8), how much the
books disagreed, and line level within market. Bottom-quartile receiving lines
are the WR3s and backup backs.

Requires a monotone trend, not a single high cell, because 48 cells means the
best one is high by construction. Includes a holdout that picks the stratum on
2023-2024 and measures it on 2025-2026.

Rushing is the most interesting case: `carries >= 5` deleted exactly the
backup-running-back rows the hypothesis is about, so its -0.032 has never been
measured on a clean stratified sample.

### 6. Alternate lines

FanDuel posts alt lines: a receiver at over 25.5, 35.5, 45.5 yards. Books
generate these from a base distribution with a crude vig ladder, and **they are
systematically less carefully priced than the main line** because far less
money flows through them.

If your variance model beats their implied one, alt lines are exactly where a
distributional edge cashes, because you are betting the SHAPE of the
distribution rather than its mean. That sidesteps the entire problem that the
mean is already priced.

The Odds API has `player_reception_yds_alternate` and siblings. This is a whole
market surface never examined. Start by capturing one week of alt lines for one
market and checking whether the implied distribution is internally consistent.

### 7. Compound distribution: receptions x yards per reception

Ask why receptions has signal at 0.275 and receiving has 0.093 when they are
built from nearly identical features.

**Receiving yards = receptions x yards per reception.** Receptions is a usage
quantity: targets, target share, snap rate. Stable, autocorrelated, exactly
what the EWMA work exploits. Yards per reception is close to noise week to
week. So the model has a real edge on the volume term, no edge on the
multiplier, and modeling yards directly dilutes the one signal with the one
unpredictable thing.

Build it as a compound distribution instead:

```
receptions   ~ negative binomial   (your model, beta 0.275)
yards/catch  ~ gamma               (player and league prior, no edge claimed)
yards        = sum of that many draws
```

Four payoffs at once, each independently testable:

1. Transfers the receptions edge into receiving yards rather than refitting
   from scratch and getting 0.093.
2. Fixes the zero-deviation gamma over-skew. A plain gamma has zero density at
   zero; real receiving yards has a genuine point mass there because a player
   can catch nothing. That misspecification is likely part of the 3.6 point
   (z +6.55) error.
3. Fixes the over-dispersion. Compound variance grows correctly with the
   volume term, which is the "sigma must depend on more than level" finding.
4. Same machinery for qb_passing (completions x yards per completion) and
   rushing (carries x yards per carry). Rushing at -0.032 may have exactly
   this structure, since `carries_roll` is a usage signal and yards per carry
   is famously unpredictable.

Standard actuarial math (compound negative binomial), not exotic. Testable
offline on four seasons with no lines and no credits: does the compound
distribution describe realized yards better than the plain gamma? Check the
predicted rate of zero-yard games against the actual rate, which for
low-volume players is probably 10 to 20 percent while a plain gamma says
nearly zero.

### 8. Proper two-sided devigging

Raw implied probability from American odds double-counts the hold.
Multiplicative, additive and Shin devigging give materially different answers,
and on receptions where FanDuel uses per-player odds (breakevens from 0.417 to
0.610) the gap is 1 to 2 points.

**If the vig is measured wrong, every edge computed is wrong by the same
amount.** Cheap to fix and it affects everything downstream, including every
item above that compares a model probability to a book probability.

### 9. Role-change detection from snap counts and depth charts

Both data sources are already loaded. The market is demonstrably slow on role
changes: a WR2 whose snap share jumped from 45 to 78 percent last week, a
backup RB promoted after an injury, a player in a new offense after a trade.

The EWMA features partly capture this, but nothing in the app flags it
explicitly and nothing tests whether the market underweights it. This is
information the market **underweights** rather than information it lacks, which
is the only kind realistically exploitable with public data.

### 10. Injury and inactive speed

A starter ruled out at 11am Sunday whose backup's line has not moved yet is a
real, repeatable edge that requires no forecasting at all. It is purely about
being faster than the book.

nflverse publishes injury reports and the capture pipeline already runs on a
schedule that could catch the gap. Also kills a whole class of fake edge:
props on players who will not play.

### 11. Team plays-per-game as a separate model

Vegas totals encode expected **scoring**, not expected **plays**. Pace,
no-huddle rate and time of possession drive opportunity volume, and a high-pace
matchup inflates every yardage prop in the game.

If the player models take `total_line` as a feature, they are using a proxy for
the thing that actually matters. A direct plays-per-game model feeding the
volume term of the compound distribution in item 7 is the natural pairing.

### 12. Limits-aware strategy, which turns a weakness into the thesis

If soft props exist, books manage them with **limits** rather than with sharp
prices. A beatable WR3 line capped at $100 is worthless to a syndicate and
completely fine for you.

**That is not a caveat, it is the mechanism.** It is the reason such a pocket
could persist indefinitely without being arbitraged away, because the people
with the capital to close it cannot get enough money down to bother.

It also implies the strategy should be many small bets on obscure props rather
than few large bets on marquee ones, which is the opposite of what a tier
system encourages. And it means the only real test is a bet slip: no dataset
can show you a limit.

---

## ORDER OF OPERATIONS

**Tomorrow, free, existing data, no credits.** Items 1, 4, 5. Three scripts off
`lines_cache.parquet`. Any one of them could produce a selection rule, and item
1 is the most likely to pay.

**This week, cheap.** Items 2 and 8. Adding `eu` for Pinnacle and fixing the
devig are both small changes with outsized effects on everything else. Item 2
should ideally land before item 1 is finalised, since Pinnacle in the reference
set makes that analysis much stronger.

**As capture data accumulates.** Item 3, which needs several weeks of
multi-snapshot data before openers and closes can be compared.

**Then the modeling work.** Items 6 and 7, both of which attack the
distribution rather than the mean, which is where the remaining room is.

**Ongoing.** Items 9, 10, 11 as feature and workflow improvements. Item 12 is
a posture rather than a task.

---

## WHAT THE APP BECOMES

Not an edge column with tiers. Tier size is measured anti-predictive, and the
reason is structural: side selection costs about 4.3 points of win probability,
which is larger than the vig, and it gets worse as the displayed edge grows.
Betting the biggest edges selected for the rows where the estimate was most
wrong.

The replacement is a decision surface:

- per-book lines side by side, with the sharpest book marked
- the devigged sharp-book probability against FanDuel's implied probability
- a flag where FanDuel is an outlier against the reference
- the model's projection shown as a reference point, not a verdict
- a flag on large recent moves, because those are usually news the model
  cannot see
- a role-change flag from snap and depth-chart deltas
- an inactive filter so dead props never appear

**Suggestions come from price disagreement, filtered by the model, rather than
from the model alone.** That is the change, and it is what the data supports.

---

## ONE THING TO REMEMBER

The measurement apparatus is the asset, not any individual number it produced.
It caught two artifacts at t above +6 in a single afternoon because it was
built to be suspicious of its own output before being pointed at anything.

Most people who build a betting model never find out theirs does not work.
Finding out in September, for $59, is the cheapest outcome available. Everything
on this list is now being tested against an instrument that will tell the truth
about it, which is a much better position than having twelve untested ideas and
no way to sort them.
