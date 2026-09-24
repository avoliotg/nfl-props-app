# OpalScales: New Directions, v3
### Rewritten Wednesday, September 23, 2026, evening

Supersedes `NEW_DIRECTIONS_2026-09-23.md` (the morning version). That document
rated the **alternate reception ladder** as the top item and the strongest
evidence the project had produced. **It was tested against live FanDuel prices
this afternoon and it is refuted.** Two other items moved, one closed, and one
new item appeared from a screenshot.

Read with `HANDOFF_2026-09-23_EVENING.md` for numbers and
`OPALSCALES_PLAN.md` for the phased plan.

---

## WHAT CHANGED SINCE THIS MORNING

| item | morning verdict | evening verdict |
|---|---|---|
| 1. Alt reception ladder | **Tier 1, do this next** | **REFUTED. Closed.** |
| 2. Compound distribution | Tier 2 | Tier 1. Unchanged in substance, now the best remaining modeling idea by default |
| 3. Openers vs closes, CLV | Tier 2 | Tier 1, with a prerequisite |
| 13. Winner's curse correction | tested, unstable | Still live. The mechanism is now understood, and it must come after the sigma fix |
| NEW | | **Template pricing at the bottom of FanDuel's board** |
| NEW | | **Threshold-selection robustness** |

The candidate rule reproduced exactly and its weakness turned out to be
different from what the morning document assumed.

---

## THE CANDIDATE RULE, REVISITED

Reproduced to every digit with `edge_threshold_v5.py --markets receptions
--line-value 2.5,3.5 --sigma-sqrt`:

```
holdout ROI   +0.1475   SE 0.0849   t +1.74   176 bets
thresholds    0.090 / 0.090 / 0.095
by season     2023 +0.0373 (71)  2024 +0.1488 (67)  2025 +0.3509 (38)
```

**A proper cluster bootstrap, re-running the whole protocol per replicate so
threshold-selection variance is included:**

```
point         +0.1475
boot median   +0.1482
95% CI        [-0.0709, +0.3133]
share > 0     91.2%
placebo       -0.0701, share > 0 24.4%
```

Not established, so J1 holds and the app stays silent. But 91 percent of
replicates positive against a placebo at 24 percent is a real asymmetry, and
the search carries no upward bias because the 100-bet training floor keeps it
off thin cells.

### The weakness is NOT the 38-bet 2025 season

This was the morning document's implicit concern and the first thing I went
after. It is wrong.

| dropped | bets | pooled ROI |
|---|---|---|
| none | 176 | +0.1475 |
| 2023 | 769 | **-0.0432** |
| 2024 | 787 | **-0.0795** |
| 2025 | 207 | **+0.0530** |

2025 is the least load-bearing of the three. Removing any season leaves a
two-season training set that picks a different threshold, and the bet count
explodes from 176 to 769. **The argmax over 51 cutoffs needs three seasons to
land near 0.090; with two it lands somewhere that loses money.**

That is the thing to fix, and it is more tractable than "one lucky season".

### The 82 percent unders is mostly the line restriction

The base population at line values 2.5 and 3.5 is already **61 percent
unders** (39 percent overs at min edge 0.00). The edge cutoff sharpens 61 to
82. And the unrestricted run goes the OTHER way, 37 percent overs at min edge
0.00 rising to 50 percent at 0.15.

Separately, the extreme under-heaviness reported under the default sigma (9
percent overs at min edge 0.10, 5 percent at 0.15) was a **flat-sigma
artifact**. A correct sigma produces a balanced tail.

### Neither component survives alone

```
alpha                        0.106
MODEL  beta*(proj-consensus) 0.108
BOOK   consensus-bet line    0.071
FanDuel differs from consensus on 298 of 4234 props (7.0%)

model only  (--same-line)  +0.0742   t +0.84
book only   (--no-model)   +0.0524   t +0.66
both                       +0.1475   t +1.74
```

And the `--no-model` arm is worse than its headline: 2023 gives **-0.0222 on
129 bets** while 2024 gives +0.2004 on 65. Sixty-five bets in one season carry
the whole arm, which is the adverse-selection mechanism that made line
shopping negative.

### EVERY ARM LOSES IN 2023 AND WINS IN 2024 AND 2025

Full rule, model only and book only all show the same shape. Beta is stable
across seasons at 0.184, 0.143, 0.213, so this is not forecast quality.

**Leading candidate: book composition.** 2023 and 2024 had eleven book names
including three defunct brands; 2025 had eight. `line_consensus` is a median
over a changing set, and mean book spread was 4.67 all-seasons against 2.09 in
2025. If the BOOK component is driven by consensus composition, 2023's loss is
mechanical rather than informative.

**This is the cheapest untested idea on the list.** Restrict to books present
in all four seasons, re-run the per-season betas and the holdout.

---

## TIER 1: DO THESE NEXT

### 1. Fix the threshold-selection step

See above. The argmax is too hungry on two seasons of training data.
Candidates: a one-standard-error rule, a smoothed threshold curve, or choosing
the cutoff on CALIBRATION rather than on realized ROI. The last is the most
appealing because calibration is measured on 4,234 rows rather than on the
176 that clear a cutoff, so it has far more power.

This is worth more than another backtest of the same rule, because it is
precisely the part that will not generalise to 2026.

### 2. Book composition across seasons

The 2023 pattern above. One flag's worth of work if `eval_harness` can filter
on book, and it would either explain the season pattern or eliminate the
explanation.

### 3. Compound distribution: receptions x yards per reception

```
receptions   ~ negative binomial   (now validated at every ladder rung AND
                                    correctly parameterised at k = 1.25)
yards/catch  ~ gamma               (player and league prior, no edge claimed)
yards        = sum of that many draws
```

Receiving yards equals receptions times yards per catch. Receptions is a usage
quantity where the model has real signal (beta 0.275). Yards per catch is close
to noise. Modeling yards directly dilutes the one signal with the one
unpredictable thing, which is exactly why receptions is 0.275 and receiving is
0.093.

**Strengthened twice today.** The reception distribution is validated across
the whole alternate ladder, and the variance form is now known to one
parameter. Both are the input a compound model needs. It also predicts the fix
for the gamma over-skew in receiving, because a compound distribution has a
genuine point mass at zero where a plain gamma has zero density.

Testable offline on four seasons with no lines and no credits. **One check
first: the correlation between reception count and yards per catch.** If it is
negative (a blowout gives more short catches; a deep threat gives few long
ones) the compound model overstates variance and needs the dependence modeled.

Same machinery applies to rushing (carries x yards per carry) and passing
(completions x yards per completion), both currently measured at zero.

### 4. CLV as the feedback metric, with its prerequisite

Two hundred graded bets say almost nothing about ROI because outcome variance
swamps everything, and a great deal about closing line value. That matters more
now: the candidate rule fires 2 to 3 times a week, so ROI confirmation would
take multiple seasons while CLV converges in weeks.

**Prerequisite that was not flagged this morning: reception movement must be
computed in PROBABILITY space.** Receptions is 93 percent flat on the line
because FanDuel moves reception prices through the ODDS. Line-space CLV will
measure zero movement forever on the one market that matters.

The automated capture is now generating the multi-snapshot data this needs.
Four scheduled runs fired overnight with row counts climbing 1,826 to 2,414.

### 5. Template pricing at the bottom of FanDuel's board (NEW)

Found by screenshot, not by fitting. Skyy Moore is a Packer, Olamide Zaccheaus
is a Falcon, different offenses and different roles:

| market | Skyy Moore | Zaccheaus |
|---|---|---|
| receiving yards 15+/20+/25+ | +130 / +194 / +280 | +132 / +194 / +280 |
| receptions 2+/3+/4+ | +126 / +370 / +920 | +134 / +400 / +920 |

Jonnu Smith and MarShawn Lloyd both sit at +880 for 4 or more receptions.
FanDuel is not pricing the bottom of the board player by player, it stamps a
template below some threshold.

That is the least defended surface anyone has found, and it requires no
distributional work. What it needs is a population study: identify where the
template boundary sits (probably a line level or a projected-volume cutoff),
then ask whether the model can separate players inside the template who
differ.

Caveats: one game, and the main lines on those players still carry real
per-player odds, so the template may live only in the alt ladder, which is
where the hold is highest. Check the MAIN line prices for template repetition
before getting excited.

---

## TIER 2: STILL LIVE

### 6. Role-change detection from snap counts and depth charts

Both sources already loaded, nothing tests whether the market underweights
them. A WR2 whose snap share jumped from 45 to 78 percent, a backup promoted
after an injury, a player in a new offense after a trade. This is information
the market UNDERWEIGHTS rather than lacks, which is the only kind realistically
exploitable with public data.

Supported indirectly by beta being HIGHER on frequently quoted players
(spearman +1.00, 33 weeks gives 0.194 against 119 weeks giving 0.302), which
is exactly the population where a role CHANGE is detectable against a long
baseline.

### 7. Injury and inactive speed

A starter ruled out at 11am Sunday whose backup's line has not moved is a real
edge requiring no forecasting. Also kills a class of fake edge: props on
players who will not play. The capture already runs on a schedule that could
catch the gap.

### 8. Proper two-sided devigging

Raw implied probability double-counts the hold. On receptions, where FanDuel
uses per-player odds, multiplicative, additive and Shin devigging differ by 1
to 2 points. Cheap, and it affects every edge computed anywhere.

**Now demonstrably useful:** the overlapping-rung method that measured
FanDuel's alt hold at 8.4 percent depends on devigging the main line
correctly, and it worked. That is a proof of concept for the technique.

### 9. Team plays-per-game as a separate model

Vegas totals encode expected SCORING, not expected PLAYS. If the player models
take `total_line` as a feature they are using a proxy for the thing that
matters. Natural pairing with the volume term of the compound distribution.

### 10. Limits-aware strategy

If soft pockets are real they are real because LIMITS keep the big money out.
That implies many small bets on obscure props rather than few large bets on
marquee ones. No dataset shows a limit. Only a bet slip does. The candidate
rule produces 2 to 3 bets a week, small enough to paper-trade by hand, and
that is the cheapest way to answer it.

**Do it after the sigma fix**, because the sigma change moves the edge
distribution and therefore which props qualify.

### 11. Winner's curse correction

**Still live, and the mechanism is now understood.** The over-dispersion tilt
reproduces (+7.2, +1.1, -0.4, -1.5, -5.0, -12.7 under level mode, against the
September 22 table's +7.3, +2.6, +1.1, -0.3, -4.7, -11.6).

What is new is why it cannot be a sigma problem: **dP/dsigma is negative in
every calibration band and both distribution families.** A sigma change shifts
all probabilities the same way. A tilt needs the ends moved in opposite
directions, and only error in the MEAN does that.

So the shrinkage targets the right thing. Its instability across seasons
(slopes 0.995, 0.849, 0.688, 0.852) is most likely because it was fitted
against a sigma that is wrong, so it absorbed a uniform shift it cannot
correct. **Refit it after the sigma change, not before.**

### 12. Pinnacle via the `eu` region

Downgraded and staying downgraded. FanDuel came out second sharpest of eleven
books with the tightest hold at 0.0608, so you are already anchored on one of
the best available prices. One line of code, about 80 credits a slate, modest
expectations.

---

## TIER 3: EVIDENCE AGAINST. DO NOT SPEND TIME HERE.

### 13. The alternate reception ladder. TESTED, REFUTED.

The morning document called this "the strongest single piece of evidence the
project has produced" and put it first. Tested against live FanDuel prices for
ATL at GB, and the case does not survive.

**First, the method that does not work.** Alt ladders are **overs only**, so
each rung gives one price. Fitting a distribution plus an overround to three
one-sided rungs is exactly identified and therefore degenerate. Proof: Skyy
Moore and Zaccheaus have ladders within two cents of each other, and the fits
returned mean 1.18 against 0.50 and hold 43 percent against 267 percent.

**The method that works.** A receptions main line is a half integer, so "over
4.5" and the alt rung "5+" are the SAME EVENT. The main line is two-sided, so
its vig strips out exactly and pins FanDuel's true probability at one rung.
**This technique is the reusable part of the whole exercise.**

**Result 1: the alt ladder is a strictly worse way to place the same bet.**
Twelve players, alt price worse on eleven and equal on one, never better. Mean
alt hold 8.4 percent (range 6.9 to 9.8) against 7.0 on the main line.

**Result 2: FanDuel prices the whole ladder off a single distribution.** With
the hold measured rather than fitted, one negative binomial fits every rung of
every ladder to within 1.3 pp, on 4 to 9 rungs per player. There is no drift
to exploit.

**Result 3: the far rungs are worse than a single negative binomial, not
better.** Pooled residuals +0.64, +1.05, +1.25 pp at four, five and six rungs
out. FanDuel's devigged probability sits ABOVE the fitted distribution, so the
odds are shorter than a pure count model justifies, on top of a higher hold.

The morning document's reasoning was that thin volume leaves the upper rungs
undefended. **FanDuel defends them by charging more.**

**Do not add the alternate market key to the capture.** It would cost about
3,000 credits a month to collect a surface that is dominated by the main line
at the overlap and worse than a coherent model at the tails.

**What the ladder work DID produce**, and it is worth more than the original
hypothesis: the FanDuel-implied var/mean of 1.38 was the first of four
independent routes to the receptions sigma fix, and the template pricing
observation is now Tier 1 item 5.

### 14. Line shopping across books. TESTED, NEGATIVE.

Unchanged. Shopping across eight books earns LESS than betting FanDuel alone
(+0.0758 against +0.1475) because choosing the outlier price and choosing the
outlier number are the same act.

### 15. The synthetic consensus result. ARTIFACT.

Unchanged. Pricing and settling against the consensus median produced +0.2991
at t +3.83, and it is not real: a median across eight books is not a price
anyone offered. Note this does NOT mean consensus is banned as an evaluation
anchor, where it is the better estimate of truth and used throughout the
harness. It is banned as a SETTLEMENT price.

### 16. BetRivers as a soft book. ARTIFACT.

Unchanged. Rush Street expresses its opinion through the LINE where other
books use the PRICE: six distinct line values against four, mean absolute
deviation 0.547 against 0.071, with asymmetric odds that compensate. Settling
at BetRivers loses 3.73 percent on 548 bets despite the highest beta in the
field. **Beta measures where a book puts its line, not whether the book is
beatable. Always read the sharpness table with the odds column beside it.**

### 17. A deviation term in sigma. TESTED, ZERO.

C within line value +0.0000, cluster bootstrap CI [-0.0476, +0.0525], with
level fully absorbed by per-line constants. Do not add the term.

---

## METHOD LESSONS, CUMULATIVE

From the morning document and still true: `qcut` collapses onto ties, so use
exact line values for discrete markets. A theoretically imposed parameter beat
a fitted one. Decomposition beats attribution. Book naming is not what the
handoff says.

Added this afternoon:

**Log loss is nearly blind to sigma at half-integer lines.** Eight candidate
forms landed within 0.0005 of each other on synthetic data where the planted
truth was one of the candidates, because P(over) is driven by the mean. Judge
sigma forms on calibration, not on proper scoring rules.

**Three sigmas existed simultaneously and I compared across them twice.** The
app's floored linear form, `edge_threshold`'s default flat residual SD, and
the sqrt form. No betting result has ever used the app's. **Always check which
parameters a result was computed under before comparing two results.**

**Record the sign convention in every calibration table.** `actual -
predicted` and `predicted - actual` both appear in this project's history, and
the mismatch produced a confident wrong conclusion that the September 22 table
did not reproduce.

**Import, do not restate.** The day's two worst errors both came from
reimplementing something that already existed: `get_line_movement`
paraphrasing the harness's book collapse, and a bootstrap script paraphrasing
v5's threshold selection. Import the constant and call the function.

**A script that reimplements an existing procedure must reproduce that
procedure's published output before producing a new number, and must ABORT
otherwise.** A synthetic test of the machinery is not a test of the
measurement.

---

## ORDER OF OPERATIONS

**This week, free.**

1. Fix `db.get_line_movement`. The bug is live and the columns it corrupts are
   ones bets have been chosen from.
2. Ship the sigma fix for receptions, remove the floor, verify against
   `mc_pricing` rather than assuming agreement.
3. Apply the same to rushing and qb_passing floors. Neither becomes
   profitable; both stop displaying confidently wrong numbers.
4. Verify the Sunday pre-kickoff captures fired.
5. Phase 3.4, the `targets_roll` gate. Still not done, still the highest-value
   model fix.

**Next, cheap.**

6. Threshold-selection robustness.
7. Book composition across seasons.
8. Compound distribution test, offline, no credits.
9. Two-sided devigging.

**Then.**

10. Refit the winner's curse correction, after sigma.
11. Paper-trade the candidate rule, after sigma.
12. CLV as the primary metric, after reception movement moves to probability
    space.
13. Template-pricing population study.
14. The line-as-feature model.

**Do not.**

The alt reception ladder. Line shopping as a strategy. Consensus settlement.
The BetRivers soft-book lead. A deviation term in sigma. Anything that reads a
beta without reading the odds beside it.

---

## THE HONEST POSITION

One candidate rule at t +1.74 with 91 percent of bootstrap replicates positive
and a placebo at 24 percent. Not established, and the app stays silent.

One genuine pricing fix with four independent confirmations, plus the
discovery that the sigma floor was causing the worst miscalibration in the
project across three markets. That is the most consequential thing found
today, and it improves the app whether or not a bet is ever placed.

Two leads killed cleanly, one of them this morning's headline. Four markets
measured at or near zero on the mean, unchanged.

What is genuinely untested and cheap: book composition across seasons, the
template pricing at the bottom of FanDuel's board, and the compound
distribution. All three need no credits.

The next honest step is not another backtest. It is shipping a pricing layer
that is correct, because that is the only finding from today that is settled
enough to act on.
