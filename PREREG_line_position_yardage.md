# Pre-registration: does under ROI decay with the line on rushing and receiving?

Written 2026-10-01, after the receptions version returned a null and BEFORE
any query binning a rushing or receiving line against an outcome.

Two markets, two different questions, one family.

## Why these two markets and not receptions

The receptions test failed for a structural reason, not a statistical one.
Rule 1's mechanism needs a book that expresses its opinion through LINE
PLACEMENT alone: a symmetric price asserts the line is at the median, and on
a right-skewed outcome a line at the mean sits above the median, so the
under is underpriced. Receptions prices asymmetrically on 94.7 percent of
rows, so the book is not doing that there, and the measured result confirmed
it: at the 1.5 line the under wins exactly 50.00 percent at an average price
of -20. The skew is real and already in the odds.

Rushing is **91.4 percent symmetric** and receiving **95.3 percent**. The
precondition exists in both.

Line distributions are comparable, so the mechanism has no obvious reason
not to transfer:

| market | n | avg line | sd | % <= 20 | % <= 40 |
|---|---|---|---|---|---|
| receiving | 9,015 | 32.3 | 19.5 | 34.1 | 69.7 |
| rushing | 4,221 | 36.1 | 23.2 | 32.8 | 57.9 |

Rule 1's `line <= 46.5` therefore covers roughly two thirds of rushing
rather than a narrow tail.

## The two questions are not the same

**RUSHING tests the mechanism behind a LIVE rule.** Rule 1 is deployed at
ROI +0.0589, 95 percent interval [+0.0115, +0.1075], about 438 bets a
season. Its strength currently rests on cross-book consistency, 54.8 to 56.1
percent under rates at six of seven books, and on per-season sign
consistency. It does NOT rest on its pooled permutation result:
`banded_under_test.py` returned p **0.2587** for rushing on a nine-cell
grid.

    So this is not a search for a new rule. It asks whether the skew story
    we attribute to an existing rule is visible as the monotone pattern
    that story predicts.

**RECEIVING tests for a new rule in a market already closed once.**
`banded_under_test.py` measured receiving at ROI -0.0350 with permutation p
**0.9801**, well powered. A flat under bias there is dead. This can only
differ if the effect is concentrated and best-of-grid averaged it away.

    The prior is poor and is stated as poor.

## Why a rank correlation rather than another banded test

`banded_under_test.py` asked "is any band good" and corrected for
best-of-grid. That has many implicit trials: every band is a candidate.

A rank correlation across fixed bins asks a different and narrower
question: are the bins ORDERED as the mechanism predicts. One statistic, no
band to choose, and a permutation null that handles all bins jointly.

    It is legitimately more powerful on the same data, so rushing could
    clear here having not cleared there. That is a better instrument, not
    a second attempt, PROVIDED the statistic and the bins are fixed before
    any result is seen. They are fixed below.

## BINS, fixed now, chosen for being round

Receptions had nine enumerable lines. These markets have 107 and 110
distinct lines, so binning is unavoidable and is a degree of freedom that
the receptions test did not have.

    Bins: [0, 10), [10, 20), [20, 30), [30, 40), [40, 50), [50, 60),
          [60, 70), [70, +)

Eight bins, ten yards wide, chosen because they are round numbers. No bin
edge is placed near 46.5, deliberately: aligning a bin to Rule 1's
threshold would build the rule's own answer into the test.

Bins with fewer than 100 bets are reported but EXCLUDED from the
correlation, with that floor fixed here. A bin of four bets contributed
+0.7758 in the receptions run and is noise.

## PREDICTIONS, fixed before the test

**R1 (rushing). The correlation is NEGATIVE.** Under ROI falls as the line
rises. This is Rule 1's mechanism stated as an ordering rather than as a
band.

**R2 (rushing). The [40, 50) bin is near the boundary.** Rule 1 fires at
46.5 and below, so if the mechanism is real the decay should already be
underway by that bin rather than switching sign abruptly at one value. A
sharp cliff instead of a gradient would suggest the 46.5 threshold is an
argmax artifact, which is the defect three findings in this project have
had.

**V1 (receiving). The correlation is NEGATIVE**, by the same skew argument.
Confidence is LOW given p 0.9801 from the banded test.

**V2 (receiving). It will not clear.** My honest expectation. The banded
test was well powered and the pooled figure is -0.0350, which is worse than
receptions' -0.0225, so there is more to overcome.

**E1 (both). The author expects RUSHING to show the pattern and RECEIVING
not to.** If receiving clears and rushing does not, that is evidence against
the mechanism rather than for a receiving rule, because the rule that
exists lives in rushing.

## Method

**Population.** FanDuel, `snapshot_label = 'closing'`, one market at a
time, joined to the realised stat. Rows needing a line, both prices and an
outcome. Devigged probability outside [0.15, 0.85] excluded as a likely
alternate line, the same threshold as the receptions test.

**Outcome.** Under ROI at the ACTUAL under price, per unit staked. Reported
with bets, win rate, average price. Both markets are mostly symmetric so
the price varies less than receptions, but it is still computed per row.

**Statistic of record.** Spearman rank correlation between bin midpoint and
bin under ROI, over bins clearing the 100-bet floor.

**Null.** Shuffle the win/loss outcome GLOBALLY while each row keeps its own
line and its own price. 2,000 permutations.

    NOT within-bin. The receptions pre-registration specified a within-cell
    shuffle and it was degenerate: preserving each cell's win rate
    preserves the effect, so the statistic never moved, sd 0.0000 against a
    true rho of -0.88. The per-bin rates ARE the thing under test.

**Correction for TWO markets.** Threshold is **0.025 per market**, not 0.05.
Running two and reporting whichever clears at 0.05 is the exact defect this
project keeps catching. Both results are reported regardless of outcome.

**Holdout.** Correlation on 2023 to 2024, sign and bins then frozen,
evaluated on 2025 to week 2 of 2026. Chronological.

**MDE printed beside every result.** A null without it is not a null. On
eight bins the null sd will be near 0.38, so detection needs |rho| above
roughly 0.6.

**Decision rule, fixed now.** Alive only if: permutation p below 0.025 on
the full sample; holdout correlation the same sign; and the low-line bins
positive at the actual price. Anything else is a null and that market
closes.

## What a rushing positive would mean, and what it would not

It would corroborate Rule 1's mechanism independently of the band that
defines the rule, which is worth more than the rule's pooled ROI because
mechanism is what makes an edge likely to persist.

It would NOT raise the rule's expected return, change its threshold, or
license widening it. The rule's ROI is already measured on its own band, and
a monotone pattern says nothing about where to cut.
