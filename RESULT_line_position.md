# Receptions line-position test: NULL, and refuted in the predicted direction

Run 2026-10-01 against `PREREG_line_position.md`, written before any query
joining a receptions line to an outcome by line value.

## The result

| prediction | outcome |
|---|---|
| **P1** Spearman(line, under ROI) is NEGATIVE | **FAILED on sign. +0.7500**, p 0.9985 |
| **P2** under ROI positive at 0.5-2.5, negative at 5.5+ | **FAILED.** Low lines are the worst cells |
| **P3** pooled ROI near the published -0.0225 | **HELD.** -0.0219 on 7,932 bets, difference +0.0006 |
| **P4** it will not clear after correction | **HELD** |

Population reconciles with `banded_under_test.py` to within 0.0006, so this
is the same row set and the comparison is valid. 7,932 of 8,117 closing rows
joined, 97.7 percent, and the devig filter dropped nothing.

## Under ROI by line, at the actual under price

| line | ROI | bets | win rate | avg odds |
|---|---|---|---|---|
| 0.5 | -0.3487 | 30 | 0.2667 | +146.0 |
| 1.5 | -0.0379 | 1,248 | 0.5000 | -20.0 |
| 2.5 | -0.0307 | 2,463 | 0.5246 | -55.6 |
| 3.5 | -0.0246 | 1,880 | 0.5330 | -69.0 |
| 4.5 | -0.0028 | 1,310 | 0.5405 | -66.2 |
| 5.5 | +0.0166 | 626 | 0.5607 | -77.2 |
| 6.5 | -0.0599 | 292 | 0.5103 | -66.1 |
| 7.5 | +0.1673 | 79 | 0.6456 | -105.0 |
| 8.5 | +0.7758 | 4 | 1.0000 | -131.0 |

The two positive cells with any ROI worth looking at are 79 bets and 4 bets.
Neither is a candidate and neither is treated as one.

## Why the mechanism failed, which is the useful part

The pre-registered argument was: reception counts have a floor at zero and a
right tail, so skew is largest when the mean is small; a book placing a line
near the mean puts it above the median; therefore the under should be most
favoured at low lines and the advantage should decay as the line rises.

**The skew argument is correct. The pricing assumption was not.**

Look at the `avg odds` column against the win rate. At the 1.5 line the
under wins exactly 50.00 percent and is priced at an average of **-20**,
which in a two-way market means the under is a heavy favourite. At 2.5 it
wins 52.46 percent at -55.6. The win rate rises with the line and so does
the price the book charges for it, and the price rises faster.

    So the book is not placing a symmetric line near the mean and leaving
    the skew on the table. It is pricing the skew explicitly, through the
    odds, and taking slightly more than the skew is worth.

That is exactly what the symmetry measurement of the same morning said it
would do: receptions is symmetric on only 5.3 percent of rows, with 311
distinct price pairs and an average absolute gap of 204.5 points. The
condition Rule 1 exploits on rushing, a symmetric price asserting the line
is at the median, is absent here on 94.7 percent of props.

**Rule 1's mechanism therefore does not generalise to receptions, and the
reason is structural rather than empirical.** Rule 1 needs a book that
expresses itself through line placement alone. Receptions is the one market
where the book cannot do that, because a half-reception is too large a step
to sit at the median.

## The correlation came out POSITIVE, which is its own finding

+0.75 is not merely "not negative". Under ROI improves monotonically from
1.5 to 5.5, which says the book's pricing of the skew is **least accurate at
high lines**, the opposite end from where the mechanism predicted.

That is not actionable as it stands: 5.5 at +0.0166 on 626 bets is inside
noise, and the permutation null has sd 0.3567, so detecting structure at all
needs |rho| above roughly 0.55 in the predicted direction.

It is worth recording as a direction for any future look at this market: if
there is anything here, it is at the top of the line range on thin samples,
not at the bottom where the theory pointed.

## Process notes

**The permutation null was degenerate as pre-registered and was corrected
before the first real run.** The document specified shuffling the outcome
within each line cell, to preserve per-line base rates. Those base rates are
the thing under test, so preserving them preserves the effect: measured on
planted data the null had sd 0.0000 and returned p 1.0000 against a true
correlation of -0.88. The corrected null shuffles win/loss globally while
each row keeps its own line and price, and has sd 0.3567.

**A format string split across two print calls shipped a TypeError** that
only fired after the statistic had printed. The per-string format checker
passed it, because each string is individually valid and the mismatch exists
only across the pair. The checker now also flags a bare `print` string
carrying format specs with no `%` operator attached.

## Status

Closed. The line-position route on receptions is a null with the mechanism
refuted in the direction it predicted, on a population that reconciles with
the earlier banded test. No further work is warranted here without a new
mechanism rather than a new slice.
