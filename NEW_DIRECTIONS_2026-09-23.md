# OpalScales: New Directions, v2
### Rewritten Wednesday, September 23, 2026

Supersedes `NEW_DIRECTIONS_2026-09-22.md`. That document listed twelve ideas
with no evidence behind any of them. This one is ordered by what a full day of
testing actually found, and **three of the original twelve now have evidence
against them.**

Read with `HANDOFF_2026-09-22_EVENING.md` for measured state and
`OPALSCALES_PLAN.md` for the phased plan. Both need a light update; the
material below is the current truth where they disagree.

---

## THE HEADLINE: THERE IS A CANDIDATE RULE, AND A BETTER LEAD BEHIND IT

**The candidate rule**, best-supported configuration after a day of trying to
break it:

```
market          receptions
lines           2.5 and 3.5 ONLY (exact values, not a quartile)
sigma           a * sqrt(mean), exponent imposed at 0.5
settle at       FanDuel
minimum edge    about 0.09
volume          2 to 3 bets a week
holdout ROI     +0.1475, SE 0.0849, t +1.74
by season       2023 +0.037, 2024 +0.149, 2025 +0.351
thresholds      stable at 0.090 to 0.095 across holdout seasons
side mix        82 percent UNDERS
```

Three seasons all positive, a stable chosen threshold, and an in-sample curve
that climbs monotonically rather than spiking in the tail. **At t +1.74 the
confidence interval still includes zero.** 176 bets cannot distinguish +0.15
from +0.03. Promising, consistent, underpowered.

**The better lead**, and it is the strongest single piece of evidence the
project has produced: the reception distribution is calibrated across the
ENTIRE alternate ladder, not just at the posted line. See Tier 1 below. An alt
ladder edge does not require beating the closing line, which is the thing that
has been failing all day, and it has 10,022 observations behind it with the
failure mode explicitly tested for and absent.

---

## TIER 1: THE ALT LADDER. Do this next.

### 1. Alternate reception lines

Books post a ladder: over 1.5, 2.5, 3.5, 4.5 and up, each priced separately.
The ladder comes from one base distribution plus a vig schedule, and the upper
rungs take a fraction of the money the main line does, so that is where a
book's shape assumptions are least defended.

**Why this is now the top item.** Every other idea on this list asks the model
to beat a sharp closing line. The alt ladder asks something different and
easier: is your distribution's SHAPE better than the book's vig ladder. Those
are separate claims, and the evidence now separates cleanly.

**The prerequisite has been tested and passed.** `ladder_calibration.py`
measured predicted P(X >= k) against realized frequency at every rung, out of
sample, on 10,022 props across four seasons:

| rung k | predicted | realized | gap | z |
|---|---|---|---|---|
| 2 | 0.7187 | 0.7219 | +0.3 pp | +0.70 |
| 4 | 0.3743 | 0.3715 | -0.3 | -0.56 |
| 6 | 0.1585 | 0.1557 | -0.3 | -0.74 |
| 8 | 0.0575 | 0.0565 | -0.1 | -0.45 |
| 10 | 0.0183 | 0.0166 | -0.2 | -1.37 |

Ten rungs, every gap inside half a point, no drift with k. And split by
distance from the posted line, which is the split that matters because the alt
ladder lives at d of +2 and above:

| d | n | predicted | realized | gap | z |
|---|---|---|---|---|---|
| +0 | 19,454 | 0.5836 | 0.5881 | +0.4 | +1.01 |
| **+2** | **19,901** | **0.2213** | **0.2207** | **-0.1** | **-0.13** |
| **+4** | **19,290** | **0.0604** | **0.0585** | **-0.2** | **-0.90** |
| +6 | 11,731 | 0.0149 | 0.0138 | -0.1 | -0.87 |

**The failure mode was tested for and is absent.** A negative binomial fixes
the mean and variance and constrains nothing else, so it can have the right
moments and the wrong tail. A synthetic control with a deliberately fattened
tail produced a gap climbing to +4.9 points at z +12.8. The real data shows
nothing like that.

The small NEGATIVE bias at d of +2 through +4 is the favourable direction: the
model is marginally conservative about upside, so its fair odds are marginally
too short, and a book more generous than the model is genuinely more generous.

Two caveats worth carrying. d = +5 shows -1.9 points at z -2.37 on only 122
observations, so the far tail is thin. And k = 1 runs +1.1 points at z +3.62,
meaning the model slightly understates "at least one catch", which matters for
unders at the bottom rung rather than overs.

**Next step costs nothing.** `ladder_calibration.py --export-ladder 2026,3`
writes the model's implied fair American odds at every rung for the current
week. Open FanDuel, read a dozen players' alternate reception ladders, and
compare by hand. Two things to look for:

  - rungs where FanDuel's price implies a probability materially below the
    model's fair number
  - whether FanDuel's rungs are internally consistent with a single
    distribution, or drift as you climb the ladder. Drift is the mispricing.

**Then, and only then**, add the alternate market key to the capture. Roughly
one extra credit per event, about 3,000 credits a month, trivial against the
budget. Run `odds_api_check.py` against the alternate keys first to confirm
naming and availability rather than assuming.

**One direction problem to respect.** The current profitable tail is 82 percent
UNDERS, and books offer rich alt-over ladders and thin alt-under ones. So the
alt ladder cannot serve the bets the main-line rule is finding. It is a
separate surface requiring its own selection logic, most likely over-side bets
on players the model projects well above their posted line.

---

## TIER 2: STILL LIVE, WORTH TESTING

### 2. Compound distribution: receptions x yards per reception

Unchanged from v1 and still the best modeling idea. Receiving yards equals
receptions times yards per catch. Receptions is a usage quantity and the model
has real signal there. Yards per catch is close to noise. Modeling yards
directly dilutes the one signal with the one unpredictable thing, which is
exactly why receptions is 0.275 and receiving is 0.093.

```
receptions   ~ negative binomial   (now validated across the whole ladder)
yards/catch  ~ gamma               (player and league prior, no edge claimed)
yards        = sum of that many draws
```

**This got stronger today.** The ladder calibration result means the reception
count distribution is trustworthy at every rung, which is precisely the input
a compound model needs. It also predicts the fix for the gamma over-skew in
receiving, since a compound distribution has a genuine point mass at zero
where a plain gamma has zero density there.

Same machinery applies to rushing (carries x yards per carry) and passing
(completions x yards per completion), both currently measured at zero.

Testable offline on four seasons with no lines and no credits: does the
compound distribution predict the realized yards distribution better than the
plain gamma, and specifically does it get the rate of zero-yard games right.

### 3. Openers versus closes, and CLV as the metric

Untested, because it needs multi-snapshot data that only started accumulating
yesterday. The automated capture is now generating it.

The question: do lines move toward the truth between posting and kickoff. If
they do, the opener is the soft number and the correct play is to bet early.
That inverts the old "wait for movement toward the model" rule, which by
construction only confirmed after the price had deteriorated.

**Change the feedback metric from ROI to closing line value.** Two hundred
graded bets tell you almost nothing about ROI because outcome variance swamps
everything, and a great deal about CLV because the noise is far smaller. This
matters more now: the candidate rule produces 2 to 3 bets a week, so ROI
confirmation would take multiple seasons while CLV converges in weeks.

### 4. Role-change detection from snap counts and depth charts

Both data sources are already loaded and nothing tests whether the market
underweights them. A WR2 whose snap share jumped from 45 to 78 percent, a
backup promoted after an injury, a player in a new offense after a trade.

This is information the market UNDERWEIGHTS rather than information it lacks,
which is the only kind realistically exploitable with public data.

**Supported indirectly by today's result.** The stratification found beta
HIGHER on frequently quoted players, monotone at spearman +1.00 (B1 at 33
weeks gives 0.194, B4 at 119 weeks gives 0.302). The edge is on established
players with stable roles and plenty of history, which is exactly the
population where a role CHANGE is most detectable against a long baseline.

### 5. Injury and inactive speed

A starter ruled out at 11am Sunday whose backup's line has not moved is a real
edge requiring no forecasting. Purely about being faster than the book.
nflverse publishes injury reports and the capture pipeline already runs on a
schedule that could catch the gap. Also kills a class of fake edge: props on
players who will not play.

### 6. Team plays-per-game as a separate model

Vegas totals encode expected SCORING, not expected PLAYS. Pace, no-huddle
rate and time of possession drive opportunity volume. If the player models
take `total_line` as a feature they are using a proxy for the thing that
matters. Natural pairing with the volume term of the compound distribution in
item 2.

### 7. Proper two-sided devigging

Raw implied probability from American odds double-counts the hold.
Multiplicative, additive and Shin devigging differ materially, and on
receptions where FanDuel uses per-player odds the gap is 1 to 2 points.
Cheap, and it affects every edge computed anywhere.

### 8. Limits-aware strategy

If soft pockets are real they are real because LIMITS keep the big money out.
That implies many small bets on obscure props rather than few large bets on
marquee ones, which is the opposite of what a tier system encourages. No
dataset shows a limit. Only a bet slip does.

**The candidate rule produces 2 to 3 bets a week, which is small enough to
paper-trade by hand.** That is the cheapest possible way to answer the limit
question, and it should start this week.

### 9. Pinnacle via the `eu` region

Still worth adding, but **downgraded from v1's framing.** The reasoning was
that a sharp reference line would improve everything. Today's book sharpness
work partly undercuts that: FanDuel came out second sharpest of eleven books at
beta 0.191 and has the TIGHTEST hold at 0.0608. You are already anchored on one
of the best available prices.

Pinnacle would still be a better anchor and is one line of code plus about 80
credits a slate. Just do not expect it to transform anything.

---

## TIER 3: EVIDENCE AGAINST. Do not spend time here.

### 10. Line shopping across books. TESTED, NEGATIVE.

v1 called this "the highest-confidence edge available, requiring no
forecasting skill". **That was wrong and the test was unambiguous.**

| settlement | holdout ROI | t |
|---|---|---|
| FanDuel only | **+0.1475** | +1.74 |
| best real price, all 11 books | +0.0999 | +1.17 |
| best real price, 8 live books | +0.0758 | +0.89 |

Shopping across eight books earns LESS than betting FanDuel alone. The reason
is adverse selection: when a book offers the best price on a low-line
reception, it is usually because its NUMBER is bad in a way the model likes and
the market has not corrected. Choosing the outlier price and choosing the
outlier number are the same act.

Per-book ROI when that book won the best price: DraftKings -0.045 on 842 bets,
BetOnline -0.059 on 594, William Hill -0.068 on 308, FanDuel -0.019 on 931.

`best_price.py` and `best_price_v2.py` implement this properly, with the
reference set separable from the settlement menu, if it is ever worth
revisiting for another market.

### 11. The synthetic consensus result. ARTIFACT.

Pricing and settling against the consensus median produced +0.2991 at t +3.83,
all three seasons above +0.25. It is not real. A median across eight books is
not a price anyone offered, and the settlement used FanDuel's odds applied to a
line FanDuel did not post. The best-available-real-price test at +0.0999
confirms it: if the consensus figure were genuine, shopping to real prices
would have come close to it.

**This is worth remembering as a methodological trap.** The result had every
surface marker of validity: stable thresholds, three positive seasons, a smooth
monotone curve, good calibration. It was still an artifact of settling on a
number that did not exist.

### 12. BetRivers as a soft book. ARTIFACT.

BetRivers showed beta 0.654 at t +20.33 against a field clustered at 0.18 to
0.23, and Unibet US (the same operator, Rush Street Interactive) showed 0.759.
Two books, same company, both three times the field. It looked like the
clearest soft-book lead imaginable.

The posting profile explains it:

| book | distinct lines | mean dev | mean abs dev | over / under |
|---|---|---|---|---|
| draftkings | 4 | +0.000 | 0.020 | -107 / -120 |
| williamhill_us | 4 | -0.008 | 0.014 | -110 / -123 |
| fanduel | 4 | +0.001 | 0.071 | -106 / -120 |
| **betrivers** | **6** | +0.117 | **0.547** | **+100 / -136** |
| **unibet_us** | **6** | +0.240 | **0.909** | **+123 / -159** |

Rush Street expresses its opinion through the LINE where other books use the
PRICE. Six distinct line values against four, sitting half a point off
consensus, with wildly asymmetric odds that compensate.

Settling at BetRivers: **holdout -0.0373 on 548 bets.** Loses money despite
the highest beta in the field.

**The lesson generalises: beta measures where a book puts its line, not
whether the book is beatable.** The sharpness table must always be read with
the odds column beside it. v1's "lowest beta is the sharpest book" was
incomplete.

### 13. Winner's curse correction. TESTED, UNSTABLE.

The curse is real and measured at roughly 4.3 points of win probability,
larger than the vig. But the correction fitted slopes of 0.995, 0.849, 0.688
and 0.852 across seasons, which is too unstable to act on, and applying it cut
qualifying props from 67 to 22 percent while dropping the holdout from +0.1475
to +0.0237.

Two reasons, one of them mine. The slope is estimated on 2,800 to 4,000 rows,
which is not enough for a stable calibration curve. And the diagnostic table
still reads the raw probability after correction, so it was impossible to see
the fix working. **The idea is sound and the implementation is not ready.**
`--curse-correct` exists in `edge_threshold_v4.py` onward if someone wants to
fix it.

### 14. Obscurity as the soft pocket. PARTLY WRONG.

The hypothesis was that the market is soft on WR3s and backup backs. The
result is more interesting than a yes or no.

**Line level: confirmed.** Beta falls monotonically as the line rises, at
spearman -1.00 in two markets independently. qb_passing Q1 +0.174 down to Q4
-0.303; receiving Q1 +0.122 down to Q4 +0.012. Receptions Q1 was the
best-powered cell in the run at +0.216, t +4.83. The pre-registered holdout
came in at **+0.282, t +4.60** on 1,557 rows in 2025-26, higher than the
2023-24 value it was chosen from.

**Player obscurity: refuted, and reversed.** Receptions beta by how often a
player is quoted is monotone at spearman +1.00 in the OPPOSITE direction:
33 weeks gives 0.194, 119 weeks gives 0.302. The edge is on the most
frequently quoted players, not the unknowns. They have the most history for an
EWMA to exploit.

**Number of books: the wrong proxy entirely.** Median is 7 of 11, because
books auto-price everything. It measures aggregator coverage, not attention.

So the pocket is **low lines on familiar players**, which sounds contradictory
and is not: a 2.5-reception line on a player quoted every week is a slot
receiver or tight end with a stable role.

---

## METHOD LESSONS FROM TODAY

Worth recording because each cost real time.

**`qcut` collapses onto ties.** Asking for the lowest line quartile in
receptions returned only the 1.5 line, because p10, p50 and p90 were all 1.5.
That stratum then lost its level-dependent sigma (no span to fit against), its
tail went to 99 percent overs, and the holdout came back -0.0684. Use exact
line values for discrete markets, which is what `--line-value` exists for.

**A theoretically imposed parameter beat a fitted one.** Imposing sigma = a *
sqrt(mean) rather than fitting the exponent raised the holdout from +0.1081 to
+0.1475 and tightened the unconditional calibration to +0.1 points at z +0.10.
Variance proportional to the mean is correct for a count, the fitted exponents
were 0.47 to 0.53 anyway, and imposing it needs no level span so it survives a
restricted stratum.

**Decomposition beats attribution.** The +0.1475 contains a model term and a
book term. Forcing beta to zero gave +0.0524, restricting to props where
FanDuel matches consensus gave +0.0742. Both contribute and neither carries it
alone. FanDuel differs from consensus on only 7 percent of these props, and
removing that 7 percent halves the result.

**Three of my own errors, recorded so they are not repeated.** I recommended
shopping as the highest-confidence edge and it is negative. I designed a
winner's curse correction whose diagnostic could not show whether it worked. I
called the consensus run invalid and said the FanDuel settlement needed
building, when `price()` already anchored on consensus and settled at the bet
line, so the +0.1475 was already the correct test.

**Bugs found in my own v1 stratification script**, all silent: `n_books` bins
covered 0 to 8 while the data reaches 10, dropping 19 percent of rows;
receptions lost its entire spread table to a swallowed exception; and the
holdout tested the single highest in-sample cell rather than the pre-registered
trend, which is the maximum of 48 cells and high by construction.

**Book naming is not what the handoff says.** Eleven names, not eight, and
three are defunct brands: `pointsbetus` (bought by Fanatics in 2023),
`barstool` (became ESPN Bet), `unibet_us`. They contributed 3.8 percent of best
prices. Any consensus computed across all seasons is inconsistent in
composition, which is why the all-seasons mean book spread is 4.67 against
2025's 2.09.

---

## ORDER OF OPERATIONS

**This week, free.**

1. Export the current-week fair-odds ladder and compare a dozen players
   against FanDuel's alt prices by hand. Highest expected value of anything on
   this list and it costs nothing.
2. Paper-trade the candidate rule. 2 to 3 bets a week, tracked by hand, which
   answers the limit question that no dataset can.
3. Confirm the Sunday cron actually fired. Those pre-kickoff captures are how
   2026 becomes measurable against the 354,554 historical rows.

**Next, cheap.**

4. Add the alternate reception market to the capture once the hand comparison
   justifies it. About 3,000 credits a month.
5. Compound distribution test, offline, no credits.
6. Two-sided devigging.

**Then.**

7. CLV as the primary metric once six to eight weeks of multi-snapshot data
   exist.
8. Role-change detection and the inactive filter.
9. Pinnacle via `eu`, with modest expectations.

**Do not.**

Line shopping as a strategy. The BetRivers soft-book lead. Consensus
settlement. Anything that reads a beta without reading the odds beside it.

---

## WHAT THE APP BECOMES

Not an edge column with tiers. Tier size is measured anti-predictive and the
reason is structural: side selection costs about 4.3 points of win
probability, which exceeds the vig, and it worsens as the displayed edge grows.

The shape that today's results support:

- **receptions at 2.5 and 3.5 only**, with a shrunk edge and a stated
  threshold near 0.09, expected to fire 2 to 3 times a week and stay silent
  otherwise
- **an alternate ladder view** showing the model's fair odds at every rung
  beside FanDuel's actual prices, which is where the best evidence currently
  points
- **the other five markets shown as reference only**, projection beside line,
  no verdict, because rushing and qb_passing are measured at zero and
  receiving is near it
- per-book lines with FanDuel marked as the sharpest available anchor
- a flag for large recent moves, since those are usually news the model cannot
  see
- an inactive filter so dead props never appear

**Suggestions come from one market, one line band, and a ladder comparison.
Not from a general-purpose edge number across six markets.** That is a
narrower product than the original vision, and it is the one the data
supports.

---

## THE HONEST POSITION

One candidate rule at t +1.74, which is promising and not proven. One
genuinely strong distributional result at 10,022 observations with the failure
mode ruled out. Four markets measured at or near zero. Three ideas actively
killed today, including the one v1 rated highest.

That is a real position. It took two days, about $59, and a measurement
apparatus that caught four separate artifacts including three that presented
as t-statistics above +3. Most people who build a betting model never find out
which parts of it do not work.

The next honest step is not another backtest. It is a dozen alt ladders read
off a phone, and 2 to 3 paper bets a week, because both answer questions the
354,554 rows cannot.
