"""td_calibration.py - why does the TD board say 39 percent where the book says 5?

Reports only. Writes nothing, ships no constant, places no bet.

THE OBSERVATION

    The anytime_td board, September 25 2026:

        Tyrone Tracy Jr   NYG   model 39.4   odds +1900 (5.0%)   edge +34.4
        Bam Knight        ARI   model 34.7   odds +3000 (3.2%)   edge +31.5
        Kameron Johnson   TB    model 38.5   odds  +850 (10.5%)  edge +28.0
        Luke Farrell      SF    model 36.4   odds  +850 (10.5%)  edge +25.9
        Jahdae Walker     CHI   model 28.0   odds +2200 (4.3%)   edge +23.7
        Michael Penix Jr  ATL   model 27.2   odds +2700 (3.6%)   edge +23.6
        Isaac TeSlaa      DET   model 40.0   odds  +460 (17.9%)  edge +22.1

    Every row is a fringe player, every row is tagged Max, and the model
    disagrees with the market by 22 to 34 POINTS in the same direction. A
    one-sided disagreement of that size against a market that
    market_calibration.py measured as well calibrated is a model defect, not
    an edge.

    Note the arithmetic is fine: 39.4 minus 5.0 is 34.4. The edge calculation
    is correct and the INPUT to it is wrong.

TWO BUGS VISIBLE IN THE SOURCE, BOTH ALREADY DOCUMENTED IN rushing.py

    1. THE TRAINING POPULATION DOES NOT MATCH THE SERVED POPULATION.

       load_model  (line 64)  trains on touches_roll >= 3
       project_week(line 106) serves  touches_roll >= 1.5

       So the model is fitted on established contributors and then asked to
       score players with half that workload. This is rushing.py note 2
       exactly, where training on carries_roll >= 5 while serving 1.5
       inflated the intercept by roughly 66 percent for backups.

       Worse, load_model's own docstring asserts that
       "touches_roll >= 3 ... is the same rule the board uses to decide who
       to display". That is false, and a future session reading it would
       conclude the populations already match.

    2. build_upcoming_week BRIDGES FROM THE FILTERED FRAME.

       Line 15 reads build_dataset(), which applies touches >= 3, an IN-GAME
       quantity. So a bridged player's td_rate_roll is the mean of `scored`
       over only his busiest weeks. This is rushing.py note 3, fixed there
       and never fixed here, and build_all_rows' own docstring quantifies it:
       td_rate_roll reads about 0.29 where the unconditional rate is 0.25.

    Both defects push P(score) UP for exactly the low-volume players on the
    board. That is the right direction to explain what is on screen, which is
    why this script exists: to measure whether they are big enough to explain
    22 to 34 points, or whether something else is also wrong.

WHAT THIS SCRIPT DECIDES

    Section 3 is the one that matters. If the model's predicted probabilities
    are badly calibrated in the high-prediction bins, then every Max tier row
    on the TD board is noise and the market is not suppressing it (anytime_td
    is a probability market, so plan item J1's reference-only path does not
    apply to it).

Run from the repo root:
    python td_calibration.py
    python td_calibration.py --season 2025
"""
import argparse
import sys

import numpy as np
import pandas as pd

from models import anytime_td as td
from sklearn.linear_model import LogisticRegression

FEATS = list(td.FEATS)
EPS = 1e-12

# From anytime_td.load_model's docstring: "Backtested on 2025: log loss
# 0.5452 vs 0.5537 shipped". The gate reproduces the first figure.
PUB_LOGLOSS_2025 = 0.5452


def _rule(t):
    print()
    print("=" * 90)
    print(t)
    print("=" * 90)


def logloss(p, y):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    y = np.asarray(y, float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def fit(train_gate, train_max_season=2024):
    """Refit the shipped specification at a chosen training gate."""
    s = td.build_all_rows()
    s = s.to_pandas() if hasattr(s, "to_pandas") else s
    tr = s[(s["season"] <= train_max_season) &
           (s["touches_roll"] >= train_gate)].dropna(subset=FEATS + ["scored"])
    if len(tr) < 500:
        return None, None
    m = LogisticRegression(max_iter=1000).fit(tr[FEATS], tr["scored"])
    return m, s


def gate(season):
    _rule("GATE: reproduce the shipped model's published backtest")
    m, s = fit(3)
    if m is None:
        print("  could not fit. Stopping.")
        sys.exit(1)
    te = s[(s["season"] == season) & (s["touches_roll"] >= 3)].dropna(
        subset=FEATS + ["scored"])
    if not len(te):
        print(f"  no {season} rows at the training gate. Stopping.")
        sys.exit(1)
    ll = logloss(m.predict_proba(te[FEATS])[:, 1], te["scored"])
    print(f"  {season} log loss at the TRAINING gate (touches_roll >= 3): "
          f"{ll:.4f}")
    print(f"  published in load_model's docstring:                      "
          f"{PUB_LOGLOSS_2025:.4f}")
    print(f"  difference: {ll - PUB_LOGLOSS_2025:+.4f}")
    if abs(ll - PUB_LOGLOSS_2025) > 0.02:
        print()
        print("  NOTE: this does not reproduce closely. The published figure")
        print("  may have been computed on a different population or feature")
        print("  set. Treat every number below as describing THIS refit")
        print("  rather than the shipped model, and check load_model before")
        print("  acting on any of it.")
    else:
        print("\n  GATE PASSED.")
    return m, s


def section_gate_mismatch(m, s, season):
    _rule("SECTION 2: THE BAND THE MODEL WAS NEVER TRAINED ON")
    print("load_model trains on touches_roll >= 3. project_week serves 1.5.")
    print("So rows with touches_roll between 1.5 and 3 are scored by a model")
    print("that never saw players like them. This is the band the board's")
    print("Max tier rows live in.")
    print()
    te = s[(s["season"] == season)].dropna(subset=FEATS + ["scored"])
    bands = [("below 1.5 (not served)", 0.0, 1.5),
             ("1.5 to 3 (SERVED, NOT TRAINED)", 1.5, 3.0),
             ("3 to 8 (served and trained)", 3.0, 8.0),
             ("8 and up", 8.0, 1e9)]
    print(f"  {'band':<34}{'rows':>7}{'mean pred':>11}{'actual':>9}"
          f"{'bias':>9}{'logloss':>9}")
    for lab, lo, hi in bands:
        g = te[(te["touches_roll"] >= lo) & (te["touches_roll"] < hi)]
        if len(g) < 30:
            print(f"  {lab:<34}{len(g):>7,}   too few")
            continue
        p = m.predict_proba(g[FEATS])[:, 1]
        y = g["scored"].to_numpy()
        print(f"  {lab:<34}{len(g):>7,}{p.mean():>11.4f}{y.mean():>9.4f}"
              f"{p.mean() - y.mean():>+9.4f}{logloss(p, y):>9.4f}")
    print()
    print("  A positive bias in the 1.5-to-3 band is the gate mismatch doing")
    print("  exactly what rushing.py note 2 describes.")

    _rule("SECTION 2b: WHAT HAPPENS IF THE GATES ARE MADE TO MATCH")
    print("Refit at the SERVING gate (1.5) and score the same rows. If the")
    print("bias shrinks, the fix is a one-word change to load_model.")
    print()
    m15, _ = fit(1.5)
    if m15 is None:
        print("  could not fit at 1.5.")
        return
    print(f"  {'band':<34}{'rows':>7}{'gate 3 bias':>13}{'gate 1.5 bias':>15}")
    for lab, lo, hi in bands:
        g = te[(te["touches_roll"] >= lo) & (te["touches_roll"] < hi)]
        if len(g) < 30:
            continue
        y = g["scored"].to_numpy()
        b3 = m.predict_proba(g[FEATS])[:, 1].mean() - y.mean()
        b15 = m15.predict_proba(g[FEATS])[:, 1].mean() - y.mean()
        print(f"  {lab:<34}{len(g):>7,}{b3:>+13.4f}{b15:>+15.4f}")
    print()
    print("  Note what this CANNOT fix: the intercept moves, but the features")
    print("  are still only touches_roll, td_rate_roll, is_rb and is_te. A")
    print("  quarterback scores through designed runs and a blocking tight")
    print("  end barely scores at all, and neither is expressible in those")
    print("  four numbers. is_rb=0 and is_te=0 makes a QB a generic receiver.")


def section_calibration(m, s, season):
    _rule("SECTION 3: IS A 39 PERCENT PREDICTION WORTH ANYTHING?")
    print("Predicted probability against realized TD rate, in bins, on the")
    print("SERVED population (touches_roll >= 1.5). The board's Max rows sit")
    print("in the top bins, so those are the ones that decide whether the")
    print("display is trustworthy.")
    print()
    te = s[(s["season"] == season) &
           (s["touches_roll"] >= 1.5)].dropna(subset=FEATS + ["scored"])
    if len(te) < 200:
        print("  too few served rows.")
        return
    te = te.copy()
    te["p"] = m.predict_proba(te[FEATS])[:, 1]
    edges = [0, .05, .10, .15, .20, .25, .30, .35, .40, 1.0]
    te["_b"] = pd.cut(te["p"], edges, include_lowest=True)
    print(f"  {'predicted bin':<16}{'rows':>7}{'mean pred':>11}{'actual':>9}"
          f"{'bias':>9}{'binom SE':>10}{'z':>7}")
    for b, g in te.groupby("_b", observed=True):
        if len(g) < 25:
            continue
        p = g["p"].to_numpy()
        y = g["scored"].to_numpy()
        se = np.sqrt(max(y.mean() * (1 - y.mean()), 1e-9) / len(y))
        z = (p.mean() - y.mean()) / se if se > 0 else np.nan
        star = "  <--" if abs(z) > 3 else ""
        print(f"  {str(b):<16}{len(g):>7,}{p.mean():>11.4f}{y.mean():>9.4f}"
              f"{p.mean() - y.mean():>+9.4f}{se:>10.4f}{z:>7.1f}{star}")
    print()
    print("  Binomial SE, not clustered, so the z values are optimistic. They")
    print("  are here to rank bins by severity rather than to test one.")
    print()
    print("  If the top bins show a large positive bias, the board is")
    print("  advertising Max tier on rows the model systematically")
    print("  over-rates, and the edge column is measuring the model's own")
    print("  error rather than the market's.")


def section_bridge(season):
    _rule("SECTION 4: DOES build_upcoming_week INFLATE THE FEATURES?")
    print("Line 15 bridges from build_dataset(), which applies touches >= 3,")
    print("an IN-GAME quantity. So td_rate_roll is measured over a player's")
    print("busiest weeks only. rushing.py note 3 fixed this same defect by")
    print("reading build_all_rows instead. Measured here rather than assumed.")
    print()
    filt = td.build_dataset()
    allr = td.build_all_rows()
    filt = filt.to_pandas() if hasattr(filt, "to_pandas") else filt
    allr = allr.to_pandas() if hasattr(allr, "to_pandas") else allr
    prior_season = season - 1

    def bridge(frame, label):
        pr = frame[frame["season"] == prior_season].sort_values(
            ["player_id", "week"])
        if not len(pr):
            return None
        out = pr.groupby("player_id").apply(
            lambda g: pd.Series({
                "td_rate_roll": g.tail(10)["scored"].mean(),
                "touches_roll": g.tail(6)["touches"].mean(),
            }), include_groups=False)
        out["src"] = label
        return out

    a = bridge(filt, "build_dataset (SHIPPED)")
    b = bridge(allr, "build_all_rows (the fix)")
    if a is None or b is None:
        print(f"  no {prior_season} rows to bridge from.")
        return
    j = a.join(b, lsuffix="_filt", rsuffix="_all", how="inner")
    print(f"  {len(j):,} players bridgeable from both frames")
    print()
    print(f"  {'quantity':<20}{'shipped':>10}{'fixed':>10}{'inflation':>12}")
    for q in ("td_rate_roll", "touches_roll"):
        f_, a_ = j[f"{q}_filt"].mean(), j[f"{q}_all"].mean()
        print(f"  {q:<20}{f_:>10.4f}{a_:>10.4f}{f_ - a_:>+12.4f}")
    print()
    print("  and by how thin the player is, since that is where it should")
    print("  bite hardest:")
    j = j.copy()
    j["_q"] = pd.qcut(j["touches_roll_all"], 4, labels=False,
                      duplicates="drop")
    print(f"  {'touches quartile':<20}{'players':>9}{'td_rate shipped':>17}"
          f"{'fixed':>9}{'inflation':>12}")
    for q, g in j.groupby("_q"):
        print(f"  {'q' + str(int(q) + 1) + ' (thinnest first)':<20}"
              f"{len(g):>9,}{g['td_rate_roll_filt'].mean():>17.4f}"
              f"{g['td_rate_roll_all'].mean():>9.4f}"
              f"{(g['td_rate_roll_filt'] - g['td_rate_roll_all']).mean():>+12.4f}")
    print()
    print("  Inflation concentrated in the thinnest quartile is the")
    print("  survivorship signature, and those are the players on the board.")


def section_min_periods(season):
    _rule("SECTION 4b: THE BRIDGE ENFORCES NO MINIMUM GAME COUNT")
    print("This is almost certainly the cause of the MAGNITUDE, and it is a")
    print("third defect distinct from the other two.")
    print()
    print("  build_all_rows:  rolling(10, min_periods=4) for td_rate_roll")
    print("                   rolling(6,  min_periods=3) for touches_roll")
    print("  _bridge:         last10[\"scored\"].mean()   <- no minimum")
    print("                   last6[\"touches\"].mean()   <- no minimum")
    print()
    print("A mean over ONE row is that row's value. So a bridged player with")
    print("a single qualifying prior game in which he scored gets")
    print("td_rate_roll = 1.0. Two games and one TD gives 0.5. Since the")
    print("bridge also reads the touches >= 3 frame, 'qualifying games' is")
    print("already a thin subset, so fringe players are the MOST likely to")
    print("land on these degenerate values, not the least.")
    print()
    print("build_all_rows would have returned NaN for those players and")
    print("project_week would have dropped them. The bridge lets them")
    print("through with a fabricated rate.")
    print()
    filt = td.build_dataset()
    filt = filt.to_pandas() if hasattr(filt, "to_pandas") else filt
    pr = filt[filt["season"] == season - 1]
    if not len(pr):
        print(f"  no {season - 1} rows in the filtered frame.")
        return
    g = pr.groupby("player_id").agg(
        games=("scored", "size"),
        td_rate=("scored", lambda x: x.tail(10).mean()),
        touches=("touches", lambda x: x.tail(6).mean()))
    print(f"  {len(g):,} players bridgeable from {season - 1}")
    print()
    print(f"  {'qualifying games':<20}{'players':>9}{'mean td_rate':>14}"
          f"{'max td_rate':>13}{'at 1.0':>8}{'at >=0.5':>10}")
    bands = [("1", 1, 1), ("2", 2, 2), ("3", 3, 3),
             ("4 to 9 (min met)", 4, 9), ("10 or more", 10, 10 ** 6)]
    for lab, lo, hi in bands:
        b = g[(g["games"] >= lo) & (g["games"] <= hi)]
        if not len(b):
            continue
        print(f"  {lab:<20}{len(b):>9,}{b['td_rate'].mean():>14.4f}"
              f"{b['td_rate'].max():>13.4f}"
              f"{int((b['td_rate'] >= 0.999).sum()):>8,}"
              f"{int((b['td_rate'] >= 0.5).sum()):>10,}")
    thin = g[g["games"] < 4]
    print()
    print(f"  players BELOW build_all_rows' min_periods of 4: {len(thin):,} "
          f"({len(thin) / max(len(g), 1):.1%})")
    print(f"  of those, td_rate_roll >= 0.5: "
          f"{int((thin['td_rate'] >= 0.5).sum()):,}")
    print(f"  of those, td_rate_roll == 1.0: "
          f"{int((thin['td_rate'] >= 0.999).sum()):,}")
    print()
    print("  Any nonzero count in that last line is a player the board can")
    print("  show at an absurd probability, because td_rate_roll is the")
    print("  model's strongest feature and it is being handed a 1.0.")


def section_shipped(season, week):
    _rule("SECTION 6: DOES THE SHIPPED MODULE NOW BEHAVE?")
    print("Everything above measures the defect with hardcoded values, so it")
    print("cannot confirm a fix. This section calls the REAL module: its")
    print("constants, its load_model, its build_upcoming_week.")
    print()
    print("  constants as shipped:")
    for name, expect in (("MIN_TOUCHES_ROLL", 1.5),
                         ("MIN_GAMES_TD_RATE", 4),
                         ("MIN_GAMES_TOUCHES", 3)):
        got = getattr(td, name, None)
        ok = "ok" if got == expect else "MISSING or unexpected"
        print(f"    {name:<20}{str(got):>8}   (expected {expect})  {ok}")
    susp = getattr(td, "CALIBRATION_SUSPENDED", None)
    print(f"    {'CALIBRATION_SUSPENDED':<20}{str(susp):>8}"
          f"   verdict withheld on the board: "
          f"{'YES' if susp else 'NO'}")

    gate_now = getattr(td, "MIN_TOUCHES_ROLL", None)
    if gate_now is None:
        print()
        print("  MIN_TOUCHES_ROLL absent, so fix 1 is not in place. Stopping.")
        return

    _rule("SECTION 6b: THE BAND BIAS AT THE SHIPPED GATE")
    print(f"Refit at the module's own MIN_TOUCHES_ROLL of {gate_now}, which is")
    print("what load_model now uses. Compare against section 2's gate-3 row.")
    print()
    m, s = fit(gate_now)
    if m is None:
        print("  could not fit.")
        return
    te = s[(s["season"] == season)].dropna(subset=FEATS + ["scored"])
    print(f"  {'band':<34}{'rows':>7}{'mean pred':>11}{'actual':>9}{'bias':>9}")
    for lab, lo, hi in [("1.5 to 3 (was +0.0554)", 1.5, 3.0),
                        ("3 to 8 (was +0.0027)", 3.0, 8.0),
                        ("8 and up (was -0.0141)", 8.0, 1e9)]:
        g = te[(te["touches_roll"] >= lo) & (te["touches_roll"] < hi)]
        if len(g) < 30:
            continue
        p = m.predict_proba(g[FEATS])[:, 1]
        y = g["scored"].to_numpy()
        print(f"  {lab:<34}{len(g):>7,}{p.mean():>11.4f}{y.mean():>9.4f}"
              f"{p.mean() - y.mean():>+9.4f}")
    print()
    print("  The gate fix moves the intercept. It does NOT fix the features,")
    print("  so expect an improvement rather than a cure.")

    _rule("SECTION 6c: WHAT build_upcoming_week NOW RETURNS")
    print("The decisive check on fix 3. Before the fix, 9 of 460 bridgeable")
    print("players carried td_rate_roll = 1.0 from a SINGLE prior game, and")
    print("136 (29.6 percent) sat below the 4-game minimum. Those players")
    print("should now be dropped rather than scored.")
    print()
    try:
        up = td.build_upcoming_week(season, week)
    except Exception as e:
        print(f"  build_upcoming_week raised {type(e).__name__}: {e}")
        print("  (needs network for rosters; run again when available)")
        return
    if not len(up):
        print(f"  returned 0 rows for {season} week {week}. Nothing to check.")
        return
    print(f"  rows returned: {len(up):,}")
    for c in ("td_rate_roll", "touches_roll"):
        if c not in up.columns:
            continue
        v = up[c].dropna()
        print(f"  {c:<16} n={len(v):>5,}  mean {v.mean():.4f}  "
              f"max {v.max():.4f}  at >=0.5: {int((v >= 0.5).sum()):,}  "
              f"at 1.0: {int((v >= 0.999).sum()):,}")
    bad = int((up["td_rate_roll"] >= 0.999).sum()) if \
        "td_rate_roll" in up.columns else 0
    print()
    if bad == 0:
        print("  NO player carries td_rate_roll = 1.0. Fix 3 is working: the")
        print("  degenerate one-game rates are now NaN and dropped.")
    else:
        print(f"  {bad} player(s) STILL carry td_rate_roll = 1.0. Either the")
        print("  minimum is not being applied, or these players genuinely")
        print("  scored in every one of 4+ prior games, which is possible for")
        print("  a small sample and is not the same defect. Check the count of")
        print("  prior games for them before concluding.")


def section_verdict():
    _rule("SECTION 5: WHAT TO DO")
    print("  TWO CODE FIXES, both one-liners, both already precedented:")
    print()
    print("    1. load_model's gate: touches_roll >= 3  ->  >= 1.5, so the")
    print("       training population matches what project_week serves. Add a")
    print("       shared constant (rushing.py uses MIN_CARRIES_ROLL) so they")
    print("       cannot drift apart again, and FIX THE DOCSTRING, which")
    print("       currently claims they already match.")
    print()
    print("    2. build_upcoming_week: build_dataset() -> build_all_rows(),")
    print("       matching what rushing.py note 3 did.")
    print()
    print("    3. AND THE ONE THAT EXPLAINS THE MAGNITUDE: _bridge must")
    print("       enforce the same minimums build_all_rows does. Return NaN")
    print("       below 4 qualifying games for td_rate_roll and below 3 for")
    print("       touches_roll, so project_week's dropna removes those")
    print("       players instead of scoring them off a fabricated rate. A")
    print("       mean over one game is not a rate.")
    print()
    print("  AND A DISPLAY DECISION, which matters more than either fix.")
    print()
    print("  anytime_td is a probability market, so plan item J1's")
    print("  reference-only path does not cover it: J1 suppresses edge and")
    print("  side where BETA is clipped to zero, and this market has no beta.")
    print("  Nothing currently stops the board advertising Max tier on a")
    print("  fringe player at 39 percent against a market price of 5.")
    print()
    print("  If section 3 shows the top bins badly biased, the honest move is")
    print("  to withhold the tier and side for anytime_td until the two fixes")
    print("  land and the calibration is re-measured. A 22 to 34 point")
    print("  one-sided disagreement with a market this project measured as")
    print("  WELL CALIBRATED is not an edge, and displaying it as Max invites")
    print("  exactly the bet the evidence says to avoid.")
    print()
    print("  Note also that the market's own price is not available to check")
    print("  against here: anytime_td is ONE-SIDED, so devig.py refuses it")
    print("  and there is no fair probability to compare with. That is why")
    print("  this script measures against REALIZED outcomes instead.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=2025,
                    help="season to score out of sample (training is <=2024)")
    ap.add_argument("--week", type=int, default=1,
                    help="week for section 6c's build_upcoming_week check")
    args = ap.parse_args()

    print("=" * 90)
    print("ANYTIME TD: why does the board say 39 percent where the book says 5?")
    print(f"  scoring season {args.season}, training through 2024")
    print("=" * 90)
    print()
    print("  READ THE SECTION NUMBERS CAREFULLY.")
    print()
    print("  Sections 1 to 4b characterise the DEFECT as it was found on")
    print("  September 25. They hardcode the old training gate and compute")
    print("  the bridge with their own inline means, so they do NOT read the")
    print("  module's constants and will keep reporting the same figures")
    print("  after the fixes land. That is deliberate: it keeps the")
    print("  measurement of the bug stable and reproducible.")
    print()
    print("  SECTION 6 is the one that reads the SHIPPED module and tells you")
    print("  whether the fixes are actually in place and working.")

    m, s = gate(args.season)
    section_gate_mismatch(m, s, args.season)
    section_calibration(m, s, args.season)
    section_bridge(args.season)
    section_min_periods(args.season)
    section_shipped(args.season, args.week)
    section_verdict()

    _rule("DONE. Nothing was written.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
