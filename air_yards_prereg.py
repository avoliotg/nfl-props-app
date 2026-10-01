"""air_yards_prereg.py - pre-register the air-yards test, then capture the
baseline. READ ONLY: fits nothing, changes no feature list, writes no
constant.

    python air_yards_prereg.py              print the pre-registration
    python air_yards_prereg.py --baseline   also run the harness and record
                                            the before numbers

WHY A PRE-REGISTRATION. Three findings in this project were artifacts of
choosing a specification after seeing the data: a qb_passing rule that swung
from 248 bets to 194 on a two-row input change because its threshold was
picked by argmax, a qb_rushing beta of 0.655 that came entirely from props
FanDuel never posted, and a receptions candidate whose ROI argmax needed
three seasons to land anywhere. Deciding to keep air yards BECAUSE it
improved a number on this data would be the same mistake in a new place.

So the predictions below are fixed before the feature exists, and the
baseline is recorded before it is added.

========================================================================
WHY THE BENCHMARK IS THE LINE, WHICH REVERSES YESTERDAY'S PLAN
========================================================================

Yesterday's plan was to test against FanDuel's DEVIGGED PRICE, on the
reasoning that settled receptions: the line is 93 percent flat there, so
the book speaks through the odds, and a model that improves on a stale
line adds nothing to a sharp price. That reasoning is sound for receptions
and does not transfer.

receptions_shipped_vs_price.py aborted on receiving with "book_p barely
varies", and the raw odds say why. FanDuel closing prices, all seasons:

    market       distinct over odds     sd      range
    qb_passing                    6   1.17     -115 to -105
    receiving                    30  17.09     -144 to +186
    rushing                      51  38.04     -160 to +710
    receptions                  119 127.41     -260 to +235

qb_passing has SIX distinct prices across 1,684 props. The price channel
there carries nothing. receiving's devigged probability has sd 0.0032
against receptions' 0.0634.

THE MECHANISM IS LINE GRANULARITY. A receiving line moves in half-yard
steps on a 40-yard number, so the book can place it at the median and
charge the vig on both sides. A reception line can only be 2.5 or 3.5, and
half a reception is a large fraction of the distribution, so the book
cannot sit at the median and must express the difference in the price.

    The same fact appeared twice more yesterday, independently: six books
    on one Dak Prescott snapshot at 260.5-270.5 all priced near -113, and
    receptions moving its price on 82.5 percent of book-paths while its
    line moved on 8.7 percent.

So "the line is stale" is a claim about a market that does not move its
line. For receiving and qb_passing it is close to the opposite of true, and
the line is the only instrument that exists.

========================================================================
THE QUESTION
========================================================================

Does adding a rolling air-yards feature to receiving's projection improve
it against the CLOSING LINE, out of sample, by enough to matter?

"By enough to matter" needs saying in advance, because beta is not the
quantity that pays. eval_harness reports `blend oos`, the out-of-sample
RMSE of the blended mean, against `RMSE line`. Its own note is the
standard: "If it does not beat RMSE line, the blend adds nothing usable
even where beta is significantly positive, because a real but tiny
R-squared moves RMSE by almost nothing."

========================================================================
PRE-REGISTERED PREDICTIONS
========================================================================

F1. BETA RISES. Currently +0.066 with a 95 percent interval of
    [-0.021, +0.153], so it does not exclude zero. Prediction: it rises
    above +0.10 and the interval excludes zero.

    Reasoning: air yards is the one input independently MEASURED as
    mispriced by this market. Rule 2 exists because FanDuel's weight on
    recent downfield usage stayed flat while its true predictive weight
    collapsed. A feature the market weights wrongly should carry signal
    the line lacks.

F2. RMSE BARELY MOVES. Prediction: `blend oos` improves by less than 0.3
    yards against the current 28.37, and `RMSE line` of 28.75 is beaten by
    under half a yard.

    Reasoning: residual sd on receiving is 28.4 yards. A feature would have
    to explain an implausible share of that variance to move RMSE a full
    yard. F1 and F2 are not in tension: a beta can be significantly
    positive and worth nothing, which is the single most repeated lesson in
    this project.

F3. THE OVER RATE DOES NOT REACH BREAKEVEN. Currently 0.4898 against a
    0.5305 breakeven at -113. Prediction: it stays below 0.52 either way,
    because adding a feature to a projection does not change which side the
    book's number sits on.

F4. RULE 2 IS DISTURBED. If F1 holds, Rule 2's veto stops being
    independent of its trigger: the rule fires on an air-yards shock and
    vetoes using the projection, and the projection would then contain air
    yards. Its measured +0.0840 was estimated with an independent veto.

    Prediction: this coupling is real and Rule 2 needs remeasuring before
    both can ship together. This is the prediction with a consequence
    attached, and it is the reason to decide the halflife BEFORE looking
    at results: a halflife chosen to maximise beta would be chosen to
    maximise the coupling too.

WHAT WOULD MAKE THIS WORTH SHIPPING. All three of: F1 holds, F2 is WRONG
in the model's favour by more than half a yard on `blend oos`, and the
per-season betas agree in sign. Anything less is a measurement, not a
feature.

WHAT WOULD MAKE IT A DEAD END. F1 fails, or F1 holds while F2 holds, which
is the receptions pattern exactly: a real beta against a quantity that does
not pay.

========================================================================
THE FEATURE, SPECIFIED BEFORE IT IS BUILT
========================================================================

Name:      air_yards_roll
Source:    receiving_air_yards, in nflverse player stats
Form:      shift(1).ewm(halflife=HL).mean(), matching every other rolling
           feature in receiving.py, so it can only ever see prior games
Halflife:  2.0 games

WHY 2.0 AND NOT A SWEEP. receiving.py's HL table already groups its
features: target_share_roll and snap_roll at 2.0, targets_roll and
ypt_roll at 12.0. The short halflives are the USAGE features and the long
ones are the efficiency features. Air yards is usage, specifically where
the targets are, so it belongs at 2.0 by the module's own logic.

    Sweeping the halflife and keeping the best would be selecting a
    specification on the test data, which is the defect behind three
    earlier artifacts. One value, chosen by analogy, fixed in advance.

WHERE IT MUST GO. Three places, and missing the third is a silent failure:
    HL                        the halflife entry
    LEAN_FEATS                the feature list load_model fits on
    build_dataset             the rolling computation
    build_upcoming_week       the SAME computation for unplayed weeks,
                              which is a separate code path and is what
                              the live board uses

    receiving.py line 33 already records a feature computed but left out
    of LEAN_FEATS, so the pattern for adding one without using it exists.
"""
import argparse
import subprocess
import sys

BASELINE_CMD = [sys.executable, "eval_harness.py", "--all-seasons",
                "--cache", "lines_cache.parquet", "--markets", "receiving"]

# Recorded from the eval_harness run of 2026-09-30, FanDuel line source,
# so the "after" run has something fixed to be compared against.
BASELINE = """
BASELINE, receiving, FanDuel closing line, seasons 2023-2026, 8,677 rows:

    beta            +0.066   SE 0.044   95% [-0.021, +0.153]   t(b=0) 1.49
    alpha           +4.45
    resid sd        28.4
    RMSE line        28.75
    RMSE proj        29.50
    blend oos        28.37
    blend ins        28.70
    beta oos        +0.066
    over rate        0.4898   SE 0.0054   [0.4791, 0.5005]
    breakeven        0.5305 at -113, so -0.0407 against it

Per-season scored rows: 3,641 / 3,219 / 3,463 / 351 for 2023-2026, with
n_train 4,608 / 9,765 / 14,956 / 20,324. A beta that rises monotonically
with n_train is a training-size effect rather than instability, which is
why the per-season breakdown has to be read alongside the pooled figure.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", action="store_true",
                    help="also run eval_harness and print the current "
                         "numbers, to confirm the recorded baseline still "
                         "reproduces before the feature is added")
    args = ap.parse_args()

    print(__doc__)
    print("=" * 72)
    print("RECORDED BASELINE")
    print("=" * 72)
    print(BASELINE)

    if not args.baseline:
        print("=" * 72)
        print("Rerun with --baseline to confirm these still reproduce.")
        print("=" * 72)
        return

    print("=" * 72)
    print("CONFIRMING THE BASELINE REPRODUCES")
    print("=" * 72)
    print("  %s" % " ".join(BASELINE_CMD))
    print()
    try:
        proc = subprocess.run(BASELINE_CMD, capture_output=True, text=True,
                              timeout=2400)
    except Exception as e:
        print("  could not run: %s: %s" % (type(e).__name__, e))
        sys.exit(1)
    out = (proc.stdout or "") + "\n" + (proc.stderr or "")
    print(out.rstrip())
    if proc.returncode != 0:
        print()
        print("  the harness exited %d; the baseline is not confirmed"
              % proc.returncode)
        sys.exit(1)

    print()
    print("=" * 72)
    print("CHECK THESE AGAINST THE RECORDED BASELINE ABOVE")
    print("=" * 72)
    for probe in ("0.066", "28.75", "28.37", "0.4898", "8677", "4.45"):
        print("  %-10s %s" % (probe, "present" if probe in out
                              or probe.replace("0.", ".") in out
                              else "NOT FOUND"))
    print()
    print("  If any is absent the baseline has moved, and the air-yards")
    print("  comparison needs the new numbers rather than these.")
    print()


if __name__ == "__main__":
    main()
