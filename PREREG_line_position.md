# Pre-registration: is a receptions line misplaced relative to discrete mass?

Written 2026-10-01, BEFORE any query that joins a receptions line to an
outcome by line value. The descriptive counts below were run first and are
reported so it is clear exactly what was known when the predictions were
fixed.

## What is already known, and what closed the obvious version

**Integer lines do not exist.** FanDuel posts `.5` lines universally: 4 of
23,037 closing rows are integers, all in receiving and rushing. A
half-point line cannot push, so the book never refunds. The
integer-versus-half-integer contrast is therefore untestable and is closed
without further work.

**Receptions has only nine distinct lines.** 0.5 through 8.5, across 8,117
closing rows, so the market is almost fully enumerable at roughly 900 rows
per line.

**A flat under bias on receptions is ALREADY CLOSED.** `banded_under_test.py`
measured receptions unders at ROI **-0.0225** with permutation p **0.9502**,
well powered at over 7,500 bets with an MDE of 0.0306. So this test is only
different from that one if the effect is **concentrated at specific lines**
and the pooled figure averaged it away.

    That is a real possibility and it is also exactly the shape of story
    that manufactures artifacts. Hence the decay prediction below, fixed in
    advance, and judged as a monotone pattern rather than as a best cell.

**Prices are symmetric on only 5.3 percent of receptions rows**, so unlike
the yardage markets the price here is informative and varies: 311 distinct
pairs, average absolute gap 204.5 points. Any ROI must therefore be computed
at the ACTUAL price, not at an assumed -110. This is the market where that
distinction bites hardest.

## The mechanism, stated so it can be wrong

Reception counts are non-negative integers with a floor at zero and a right
tail. For a player whose expected catches is near 3, the distribution is
right-skewed, so the median sits below the mean.

A book that places its line near the **mean** would therefore put the line
above the median, and the under would be favoured at a symmetric price. That
is precisely Rule 1's mechanism, which is live on rushing and has never been
tested on receptions in this form.

The skew is a function of the mean: it is largest when the mean is small,
because the zero floor binds, and it shrinks as the mean rises and the
distribution approaches symmetry. Since the line tracks the mean, **the
effect must decay as the line rises.**

## PREDICTIONS, fixed before the test

**P1. MONOTONE DECAY, NOT A BEST CELL.** Under ROI by line, ordered from 0.5
to 8.5, shows a decreasing pattern. Specifically: the Spearman rank
correlation between line value and under ROI is **negative**, and the test
of record is that correlation, not the best-performing line.

    This is the prediction that distinguishes a mechanism from a fished
    cell. A single profitable line among nine proves nothing; nine lines
    ordered by a quantity the mechanism predicts is a different claim.

**P2. THE LOW LINES ARE POSITIVE AND THE HIGH ONES ARE NOT.** Under ROI at
lines 0.5 to 2.5 is above zero; at 5.5 and above it is below zero. Stated as
a sign pattern rather than a magnitude, because magnitudes at 900 rows per
cell are noisy.

**P3. THE POOLED FIGURE STAYS NEGATIVE.** Pooled under ROI across all lines
remains close to the already-measured -0.0225. If pooling suddenly turns
positive, the population differs from `banded_under_test.py`'s and the
discrepancy must be explained before anything else is read.

**P4. IT WILL NOT CLEAR AFTER CORRECTION.** My honest expectation is that P1
fails or is too weak to survive a permutation over nine cells. Reason:
receptions' price is NOT symmetric on 94.7 percent of rows, so the book has
already expressed a directional opinion through the odds, and the mechanism
requires the book to be placing a symmetric line at the mean. On this market
it mostly is not placing a symmetric line at all.

    P4 is the prediction I expect to be right, and stating it first is the
    point: if the test comes back positive it will have beaten a
    pre-registered expectation of failure rather than confirming a hope.

## Method, fixed before the test

**Population.** FanDuel, `snapshot_label = 'closing'`, receptions, lines 0.5
to 8.5, joined to the realised reception count. Rows with no outcome
excluded. No alternate-line filter is needed since receptions has nine
distinct lines and all are plausible main lines, but any row whose price
implies a devigged probability outside 0.15 to 0.85 is excluded as a likely
alt, with that threshold fixed here.

**Outcome.** Under ROI at the ACTUAL under price, in units staked. Reported
with bets, win rate, average price, and a bootstrap interval clustered on
game. Win rate alone is not the endpoint, because prices vary by 204 points
on average across this market.

**Grouping.** All nine lines. Exhaustive and fixed, so there is no band to
choose. No sub-grouping by role, opponent, season or book: each would
multiply the cells and each is available to be interesting.

**Test of record.** Spearman rank correlation between line and under ROI,
with a permutation null that shuffles the win/loss outcome **globally**
while each row keeps its own line and its own price. 2,000 permutations.

> **Corrected 2026-10-01, before the first real run.** This originally
> specified shuffling the outcome WITHIN each line cell, to preserve the
> per-line base rates. That null is degenerate: a within-cell shuffle
> cannot change that cell's own win rate, so every cell ROI is identical to
> the observed one and the statistic never moves. Measured on planted data
> it had sd 0.0000 and returned p 1.0000 against a true correlation of
> -0.88.
>
> The error was in the reasoning, not the code. The per-line base rates
> ARE the thing under test, so preserving them preserves the effect. The
> global shuffle keeps the rows per line, every row's real price and the
> overall win rate, and destroys only the link between line value and
> winning, which is H0 stated properly.

**Power, measured on the corrected null.** With nine cells the null sd is
0.354, so the smallest correlation detectable at p<0.05 one-sided is about
**-0.55**. That requires roughly seven or eight of the nine lines to be
correctly ordered. A weaker observed correlation is UNTESTED rather than
cleared, and the script prints this beside the result.

**Correction.** The permutation is over the correlation statistic, so the
nine cells are handled jointly rather than by Bonferroni over nine separate
tests. Any per-cell figure reported afterwards is descriptive only and is
labelled as such.

**Holdout.** Correlation estimated on 2023 to 2024, direction and grouping
then frozen, evaluated on 2025 to week 2 of 2026. Chronological, never
random, because pricing behaviour changes across seasons.

**Decision rule, fixed now.** Alive only if ALL of: the permutation p on the
full-sample correlation is below 0.05; the holdout correlation has the same
sign; and the holdout's low-line cells are positive at the actual price. Any
other outcome is a null and the route closes.

**MDE must be printed beside the result**, per project standard, and a null
without it is not a null.

## What a positive result would and would not mean

It would mean the book's line placement on low-count markets is
systematically above the median, which is Rule 1's mechanism generalising to
a second market. That would be the first time a mechanism in this project
replicated across markets, which is worth considerably more than a second
independent rule.

It would not mean a bet yet. Receptions prices are asymmetric, so a cell
with positive ROI at the actual price needs its average price reported
before anyone sizes anything, and a 2.38-point hurdle at -110 becomes much
larger at -175.
