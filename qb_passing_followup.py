"""qb_passing_followup.py - what the side-picker result actually is.

Reports only. Writes nothing, places no bet.

THE ERROR IN THE PREVIOUS SCRIPT

    qb_passing_sidepicker.py reported book_p coefficients of -0.0057, -0.0102
    and -0.0285, and its own output text read that as "the logistic is
    discounting the market's price". That interpretation is wrong.

    The real reason is that book_p is NEARLY CONSTANT. Mean devigged book_p
    came out 0.5000, and market_calibration.py placed all 1,682 FanDuel
    qb_passing rows in a single probability bin at mean 0.5000. FanDuel prices
    this market symmetrically on essentially every prop, so after devigging
    the fair probability is 0.5 on almost every row. A regressor with no
    variance has an unidentified coefficient, and reading a near-zero
    coefficient as a behavioural fact was a mistake.

    CONSEQUENCE, AND IT CHANGES WHAT THE RESULT MEANS. If book_p is constant
    then disagree = p_model - 0.5 is just the model's own probability
    re-centred. The result is therefore NOT a disagreement-with-the-market
    signal. It is a logistic on the module's five features picking a side of
    the half-integer line, with the odds contributing nothing.

    That is arguably a cleaner and better finding: because FanDuel's
    qb_passing prices are uniform, ALL of the market's information sits in the
    line, and the model has measurable out-of-sample skill at calling which
    side of it the outcome lands. But it has to be stated as that, not as
    beating a price.

THE THREE TESTS

    SECTION 1  Confirm the degeneracy numerically. Report book_p's standard
               deviation, its distinct-value count, and the share of rows
               within a whisker of 0.5. Then REFIT with book_p dropped
               entirely. If the results are materially identical, the
               reframing is established rather than argued.

    SECTION 2  Season split of the best cell. The previous run skipped 2023
               because nothing precedes it, so the whole result rests on
               2024, 2025 and a 64-row 2026. That is effectively two seasons
               and it was never split.

    SECTION 3  Re-run the placebo with the grid floor raised from 200 to 300
               bets. The winning cell had 249, barely over the old floor, and
               a result that depends on one marginal cell is not a result.

WHAT THIS CANNOT ADDRESS

    p = 0.010 is corrected for the threshold grid. It is NOT corrected for
    choosing qb_passing out of six markets, for this feature set, or for the
    prior sessions that searched this same rule on this same data. Only
    forward testing addresses those.

Run from the repo root:
    python qb_passing_followup.py --all-seasons --cache lines_cache.parquet
"""
import argparse
import sys
import types

import numpy as np
import pandas as pd

import eval_harness as eh
import qb_passing_sidepicker as sp
import recency_test as rt
from sklearn.linear_model import LogisticRegression

# From the previous run. The gate aborts unless these reproduce.
PUB_SLOPE = 0.7776
PUB_BEST_ROI = 0.0839
PUB_BEST_N = 249


def _rule(t):
    print()
    print("=" * 92)
    print(t)
    print("=" * 92)


def refit(g, feats, use_book_p):
    """Walk-forward refit, with or without book_p in the design."""
    cols = (["book_p"] if use_book_p else []) + feats
    out = []
    for season in sorted(g["season"].unique()):
        tr = g[g["season"] < season]
        te = g[g["season"] == season]
        if len(tr) < 300 or len(te) < 50:
            continue
        m = LogisticRegression(max_iter=2000).fit(tr[cols], tr["over"])
        t = te.copy()
        t["p_model"] = m.predict_proba(te[cols])[:, 1]
        # With book_p dropped there is no market probability to subtract, so
        # the signal is the model's own probability re-centred on 0.5. That is
        # what the WITH-book_p version was already computing, given book_p is
        # constant at 0.5. Making it explicit is the point of the comparison.
        t["disagree"] = t["p_model"] - (t["book_p"] if use_book_p else 0.5)
        out.append(t)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def section_degeneracy(g, feats):
    _rule("SECTION 1: IS book_p CONSTANT, AND DOES DROPPING IT CHANGE ANYTHING?")
    bp = g["book_p"]
    print(f"  rows                          {len(bp):,}")
    print(f"  mean                          {bp.mean():.6f}")
    print(f"  standard deviation            {bp.std():.6f}")
    print(f"  distinct values               {bp.nunique():,}")
    print(f"  within 0.001 of 0.5           {(bp.sub(0.5).abs() < 0.001).mean():.2%}")
    print(f"  within 0.01 of 0.5            {(bp.sub(0.5).abs() < 0.01).mean():.2%}")
    print(f"  min / max                     {bp.min():.4f} / {bp.max():.4f}")
    print()
    print("  raw over/under odds, distinct pairs:")
    pairs = (g.groupby(["over_odds", "under_odds"]).size()
             .sort_values(ascending=False).head(8))
    for (oo, uo), n in pairs.items():
        print(f"    {oo:>6} / {uo:<6}  {n:>6,} rows "
              f"({n / len(g):.1%})")
    print()
    if bp.std() < 0.01:
        print("  DEGENERACY CONFIRMED. book_p carries essentially no")
        print("  variance, so its coefficient was unidentified and the")
        print("  'disagreement' was the model's own probability all along.")
    else:
        print("  book_p has real variance, so the original framing stands and")
        print("  this whole script is built on a wrong premise. Stop here.")
        return None

    _rule("SECTION 1b: REFIT WITH book_p DROPPED")
    print("If the reframing is right, these two should be near-identical.")
    print("Any large difference means book_p was doing something after all.")
    print()
    print(f"  {'design':<22}{'n':>7}{'slope':>10}{'SE':>9}{'t':>7}"
          f"{'best ROI':>10}{'best n':>8}")
    res = {}
    for lab, use in (("with book_p", True), ("without book_p", False)):
        r = refit(g, feats, use)
        if not len(r):
            continue
        X = np.column_stack([np.ones(len(r)), r["disagree"].to_numpy()])
        f = rt.ols_cluster(X, r["over"].to_numpy(), r["event_id"].to_numpy())
        best_roi, best_n = -9.0, 0
        for thr in sp.THRESHOLDS:
            sel = r[r["disagree"].abs() >= thr]
            if len(sel) < 200:
                continue
            prof = np.where(sel["disagree"] > 0, sel["roi_over"],
                            sel["roi_under"])
            if float(prof.mean()) > best_roi:
                best_roi, best_n = float(prof.mean()), len(sel)
        t = f["beta"][1] / f["se"][1] if f and f["se"][1] else np.nan
        print(f"  {lab:<22}{len(r):>7,}{f['beta'][1]:>+10.4f}"
              f"{f['se'][1]:>9.4f}{t:>7.2f}{best_roi:>+10.4f}{best_n:>8,}")
        res[lab] = r
    print()
    print("  The 'without' row is the honest description of the finding: a")
    print("  model picking a side of the LINE, with the odds contributing")
    print("  nothing because they are uniform.")
    return res.get("without book_p")


def section_seasons(r, n_boot):
    _rule("SECTION 2: SEASON SPLIT OF THE BEST CELL")
    print("The previous run never did this. 2023 cannot be scored because")
    print("nothing precedes it, so the whole result rests on 2024, 2025 and a")
    print("64-row 2026, which is effectively two seasons.")
    for thr in (0.05, 0.06):
        sel = r[r["disagree"].abs() >= thr].copy()
        if len(sel) < 150:
            continue
        sel["prof"] = np.where(sel["disagree"] > 0, sel["roi_over"],
                               sel["roi_under"])
        sel["won"] = np.where(sel["disagree"] > 0, sel["over"],
                              1 - sel["over"])
        print()
        print(f"  |disagree| >= {thr}   ({len(sel):,} bets, pooled "
              f"{sel['prof'].mean():+.4f})")
        print(f"    {'season':<8}{'bets':>7}{'win':>8}{'ROI':>9}{'lo95':>9}"
              f"{'hi95':>9}")
        signs = []
        for season, gg in sel.groupby("season"):
            if len(gg) < 40:
                print(f"    {season:<8}{len(gg):>7,}   too few")
                continue
            lo, hi = sp.boot_ci(gg["prof"], gg["event_id"], n_boot)
            print(f"    {season:<8}{len(gg):>7,}{gg['won'].mean():>8.4f}"
                  f"{gg['prof'].mean():>+9.4f}{lo:>+9.4f}{hi:>+9.4f}")
            signs.append(np.sign(gg["prof"].mean()))
        if len(signs) < 2:
            print("    ONE SEASON ONLY. Not a split, and not evidence of")
            print("    stability. This is the weakest point in the finding.")
        elif all(s > 0 for s in signs):
            print("    SIGN CONSISTENT across the seasons that can be scored")
        else:
            print("    SIGN FLIPS")


def section_floor(r, n_perm):
    _rule("SECTION 3: DOES THE RESULT SURVIVE A HIGHER CELL FLOOR?")
    print("The winning cell had 249 bets against a floor of 200. A finding")
    print("that depends on one marginal cell is not a finding, so this")
    print("re-runs the placebo at floors of 200, 300 and 400 and reports how")
    print("the p-value moves.")
    print()
    print(f"  {'floor':<8}{'cells used':>12}{'best ROI':>10}{'best t':>9}"
          f"{'null 95th t':>13}{'p':>8}")
    for floor in (200, 300, 400):
        usable = sum(1 for thr in sp.THRESHOLDS
                     if len(r[r["disagree"].abs() >= thr]) >= floor)
        if usable == 0:
            print(f"  {floor:<8}{0:>12}   no cell survives this floor")
            continue
        obs_roi, obs_t = sp._grid_best("disagree", r, min_bets=floor)
        rng = np.random.default_rng(0)
        g = r.copy()
        nt = []
        for _ in range(n_perm):
            g["_d"] = g.groupby(["season", "week"])["disagree"].transform(
                lambda s: rng.permutation(s.to_numpy()))
            _, b = sp._grid_best("_d", g, min_bets=floor)
            if b > -9.0:
                nt.append(b)
        if not nt:
            print(f"  {floor:<8}{usable:>12}   placebo produced nothing")
            continue
        nt = np.array(nt)
        p = float((nt >= obs_t).mean())
        print(f"  {floor:<8}{usable:>12}{obs_roi:>+10.4f}{obs_t:>+9.2f}"
              f"{np.percentile(nt, 95):>+13.2f}{p:>8.3f}")
    print()
    print("  A p-value that holds as the floor rises means the signal is in")
    print("  the wide cells too, not just the narrow one. A p-value that")
    print("  collapses means the 249-bet cell was carrying it.")


def section_summary(r):
    _rule("SECTION 4: HOW TO STATE THIS")
    print("  The finding, stated correctly:")
    print()
    print("    FanDuel prices qb_passing symmetrically on nearly every prop,")
    print("    so the devigged fair probability is 0.5 almost everywhere and")
    print("    ALL of the market's information is in the half-integer line.")
    print("    A logistic on attempts_roll, team_spread, total_line, wind_eff")
    print("    and def_pass_roll, fitted walk-forward, has measurable skill")
    print("    at calling which side of that line the outcome lands.")
    print()
    print("  What it is NOT: a signal from disagreeing with the book's PRICE.")
    print("  There is no price signal to disagree with in this market.")
    print()
    print("  This is notable for a second reason. qb_passing's beta is 0.085")
    print("  with a CI spanning zero, which says the projection adds nothing")
    print("  to the line on a YARDS scale. A side-picker can still work where")
    print("  beta does not, because calling the SIGN of a threshold crossing")
    print("  is an easier problem than improving the conditional mean. That")
    print("  is the reframe that was proposed earlier in this project and")
    print("  never tested, and this is the first evidence for it.")
    print()
    print("  If sections 2 and 3 hold, that argues for trying the same")
    print("  threshold-classification approach in receptions, where the")
    print("  projection has real signal (beta 0.226) and the same")
    print("  side-picking question is still open.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", default=None)
    ap.add_argument("--all-seasons", action="store_true")
    ap.add_argument("--cache", default=None)
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--book", default="fanduel")
    ap.add_argument("--boot", type=int, default=1500)
    ap.add_argument("--perm", type=int, default=300)
    args = ap.parse_args()

    if args.all_seasons:
        seasons = list(eh.ALL_SEASONS)
    elif args.seasons:
        seasons = [int(s) for s in args.seasons.split(",") if s.strip()]
    else:
        seasons = [2025]

    print("=" * 92)
    print("QB_PASSING FOLLOW-UP: what the result actually is")
    print(f"  seasons {seasons}   book {args.book}")
    print("=" * 92)

    # Reuse the previous script's loader rather than restating it, so this
    # cannot measure a different population than the run it is correcting.
    g, feats = sp.load(types.SimpleNamespace(
        cache=args.cache, refresh=args.refresh, book=args.book), seasons)

    _rule("GATE: the previous run must reproduce")
    r0 = refit(g, feats, True)
    X = np.column_stack([np.ones(len(r0)), r0["disagree"].to_numpy()])
    f = rt.ols_cluster(X, r0["over"].to_numpy(), r0["event_id"].to_numpy())
    print(f"  slope    {f['beta'][1]:+.4f}   published {PUB_SLOPE:+.4f}")
    ok = abs(f["beta"][1] - PUB_SLOPE) <= 0.01
    br, _ = sp._grid_best("disagree", r0)
    print(f"  best ROI {br:+.4f}   published {PUB_BEST_ROI:+.4f}")
    ok = ok and abs(br - PUB_BEST_ROI) <= 0.005
    if not ok:
        print("\n  ABORT: the previous run does not reproduce.")
        sys.exit(1)
    print("  GATE PASSED.")

    r = section_degeneracy(g, feats)
    if r is None or not len(r):
        return 0
    section_seasons(r, args.boot)
    section_floor(r, args.perm)
    section_summary(r)

    _rule("DONE. Nothing was written. No bet was placed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
