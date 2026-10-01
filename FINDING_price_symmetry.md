# FanDuel prices symmetrically, and that decides which instrument works

Measured 2026-10-01 on `historical_lines`, FanDuel closing lines, seasons
2023 through week 2 of 2026.

```sql
select market, count(*) as n,
       sum(case when over_odds = under_odds then 1 else 0 end) as symmetric,
       round(100.0 * sum(case when over_odds = under_odds then 1 else 0 end)
             / count(*), 1) as pct_symmetric,
       count(distinct (over_odds, under_odds)) as distinct_pairs,
       round(avg(abs(over_odds - under_odds)), 1) as avg_abs_gap
from historical_lines
where book = 'fanduel' and snapshot_label = 'closing'
  and over_odds is not null and under_odds is not null
group by market;
```

| market | n | symmetric | distinct pairs | avg abs gap |
|---|---|---|---|---|
| qb_passing | 1,684 | **99.6%** | 8 | 0.0 |
| receiving | 9,015 | **95.3%** | 55 | 2.7 |
| rushing | 4,221 | **91.4%** | 81 | 9.1 |
| receptions | 8,117 | **5.3%** | 311 | 204.5 |

A symmetric two-sided price devigs to exactly 0.5000. So on 99.6 percent of
qb_passing props, 95.3 percent of receiving and 91.4 percent of rushing,
**FanDuel states no player-specific probability at all.** It places the line
where it believes the median sits and charges the same toll on both sides.
Receptions is the opposite: 94.7 percent asymmetric, 311 distinct price
pairs, average gap 204.5 points.

## The mechanism is line granularity

A receiving line moves in half-yard steps on a 40-yard number. Half a yard
is a tiny fraction of that distribution, so the book can place the line
essentially at the median and price both sides at -110. A reception line can
only be 2.5 or 3.5. Half a reception is a large fraction of the
distribution, so the book physically cannot sit at the median and has to
express the remainder in the odds.

The ordering is monotone in granularity-relative-to-dispersion, and the
separation is two orders of magnitude on the average gap.

## This explains four earlier observations with one fact

1. `receptions_shipped_vs_price.py` aborted on receiving with "book_p barely
   varies". Correct: `book_p` sd is 0.0032 there against 0.0634 on
   receptions, because it is the constant 0.5 on 95 percent of rows.
2. Six books on one Dak Prescott snapshot sat at 260.5 to 270.5 all priced
   near -113. Ten yards of line spread at zero price spread. Each book
   parks at the middle of its own number.
3. receptions moves its price on 82.5 percent of book-paths while its line
   moves on 8.7 percent; the yardage markets are the reverse.
4. `side_picker_search_v2.py`'s model 1 ("logit book_p only") returned
   slope 3.351 with SE 3.255 on receiving and -1.843 with SE 1.464 on
   qb_passing. Those are the signature of fitting a logistic regression on
   a regressor with no variance. Model 2 ("book_p + line") reduces to the
   line alone for the same reason.

Point 4 has a consequence worth acting on: two of five challenger rungs are
near-degenerate on three of four markets, and they still widen the
Bonferroni correction. qb_passing's surviving result cleared by 0.0019 of
p-value against a threshold of 0.0100 computed over five challengers. A
ladder of three real challengers would have given it a threshold of 0.0167.

## What it reframes

**The 25 September conclusion was correct about receptions and has been
over-generalised.** That conclusion was: the line is the stale quantity, the
price is the sharp one, and a model that improves on the line adds nothing
to the price you bet against. It is a claim about a market that does not
move its line. For qb_passing, receiving and rushing the line IS the
market's entire opinion, stated as precisely as the half-point grid allows,
and there is no price channel to be sharper than it.

**So the encompassing test is a receptions-only method in this project.** It
has been run there and receptions failed it. It cannot be run on the other
three, and the script is right to refuse.

**And beta against the line is the honest instrument on those three.** Not
the discredited one. This restores a research capability that had been
discouraged for a week.

## What it does not do

It does not produce an edge. The over rate sits near 0.50 in all three
symmetric markets, against a breakeven of 0.5305 at -113:

| market | over rate | vs breakeven |
|---|---|---|
| receiving | 0.4898 | -0.0407 |
| rushing | 0.4705 | -0.0600 |
| qb_passing | 0.5060 | -0.0245 |

A book that always prices symmetrically is a book that has to get the line
right, and these figures say it does. Every edge must therefore come from
the line being in the wrong place, not from the price being wrong.

## It supports Rule 1 rather than threatening it

Rule 1 bets the under on rushing below 46.5. Its mechanism is distributional:
alpha measured at zero deviation is +4.0067, so the mean sits about four
yards above the line while the under wins 55.86 percent, which places the
median below it. Mean above, median below, is right skew straddling the
line.

A book pricing symmetrically is asserting the line is at the median. On a
right-skewed outcome with a floor at zero, a line placed at the mean is
above the median, and the under is then underpriced at -110. Symmetric
pricing is the condition under which that error is possible, so this finding
is the mechanism's prerequisite rather than a problem for it.

## Consequence for the vig

The honest hurdle is now explicit. A symmetric -110 implies 0.5000 with a
breakeven of 52.38 percent. The market offering 0.5000 is not an invitation;
it is 0.5000 plus a 2.38-point toll. Any side-picker on these markets starts
2.38 points behind, which is why qb_passing's holdout ROI of +0.0996 on 194
bets matters more than its slope of +0.831.
