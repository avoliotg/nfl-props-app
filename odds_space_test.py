"""
ODDS-SPACE TEST: is the book's price channel informative, beyond its line?

Every beta in this project lives in MEAN space against the line. The odds have
never entered a single calculation. That leaves one question unanswered and it
decides whether a side-picking rule is possible in a beta-zero market:

    at a FIXED line, does FanDuel's two-sided price predict the outcome?

If yes, the odds carry player-specific information, and a model with beta
clipped to zero (which returns line + alpha for every player at that line)
is betting that information is noise. That is the line_best / BetRivers
artifact family and the idea is dead.

If no, the price variation at a fixed line is closer to template pricing, and
a distributional rule has room to work.

PRE-REGISTERED PREDICTION, recorded before the first run:
    Test B comes back significantly positive in all four markets, strongest in
    receptions, because the plan states FanDuel moves reception prices through
    the ODDS (93 percent flat on the line). If receptions comes back FLAT, the
    prediction is refuted and the reference-only markets are worth revisiting.

WHAT THIS SCRIPT DOES NOT DO
    It does not fit anything new to decide a bet. It measures the book. No
    threshold, no rule, no ROI. Those come after, and only if Test B says the
    channel is not already efficient.

METHOD NOTES
  Devigging is PROPORTIONAL: normalise the two raw implied probabilities to
  sum to 1. Near even money, which is where half-integer yardage and reception
  lines sit, this is nearly method-independent. It is least reliable on
  longshots, so results are also broken out by price bucket rather than only
  pooled.

  Pushes are DROPPED, not counted as unders. Integer reception lines exist
  (plan 6.13) and an outcome exactly equal to the line is a push.

  Test B uses the WITHIN estimator: book_p is demeaned inside each
  (market, line) stratum, so the slope is identified only by variation in
  price at an identical line. Line level cannot contribute.

  Standard errors are clustered on event_id throughout, reusing the harness's
  own ols_clustered. Props in one game move together.

  Every null result prints a MINIMUM DETECTABLE EFFECT. "No difference" and
  "no difference we could see at this n" are different claims.

  Every p-value prints a Bonferroni threshold beside it.

THE GATE
  This script reimplements nothing, but it does re-derive the FanDuel-anchored
  betas from the same pipeline. It ABORTS unless those reproduce the published
  values within tolerance. A script that cannot reproduce a published number
  has no business producing a new one.

Run from the repo root with the venv active:
    python odds_space_test.py --all-seasons --cache lines_cache.parquet
    python odds_space_test.py --all-seasons --cache lines_cache.parquet --with-model
    python odds_space_test.py --all-seasons --cache lines_cache.parquet --save-rows odds_rows.csv

--with-model adds Test C, which needs mc.prob_over and therefore the app's
pricing layer. Leave it off for the first run: Tests A and B need no model at
all and they are what answers the question.
"""
import argparse
import sys

import numpy as np
import pandas as pd

import eval_harness as eh

# Published FanDuel-anchored betas, from the September 24 handoff anchor table.
# The gate compares against these. Do not edit to make a run pass.
PUBLISHED_FD_BETA = {
    "receptions": 0.226,
    "receiving": 0.066,
    "rushing": -0.043,
    "qb_passing": 0.085,
}
GATE_TOL = 0.030


# ------------------------------------------------------------------- devigging

def american_to_decimal(o):
    """American odds to decimal. Returns nan on anything unusable."""
    try:
        o = float(o)
    except (TypeError, ValueError):
        return np.nan
    if not np.isfinite(o) or o == 0:
        return np.nan
    if o > 0:
        return 1.0 + o / 100.0
    return 1.0 + 100.0 / (-o)


def devig_proportional(over_odds, under_odds):
    """Two-sided American odds to a vig-free P(over), and the hold.

    Proportional (multiplicative) normalisation. Returns (p_over, hold), both
    nan if either side is missing, because a one-sided price cannot be
    devigged and substituting a nominal other side would invent data.
    """
    do = american_to_decimal(over_odds)
    du = american_to_decimal(under_odds)
    if not (np.isfinite(do) and np.isfinite(du)):
        return np.nan, np.nan
    ro, ru = 1.0 / do, 1.0 / du
    tot = ro + ru
    if tot <= 0:
        return np.nan, np.nan
    return ro / tot, tot - 1.0


# ---------------------------------------------------------------- stats extras

def mde(se, power=0.80):
    """Minimum detectable effect at the given power, two-sided alpha 0.05.

    2.80 * SE for 80 percent power. Printed beside every null so a flat slope
    is not read as a demonstrated zero.
    """
    if se is None or not np.isfinite(se) or se <= 0:
        return np.nan
    return 2.80 * se


def two_sided_p(t):
    """Normal-approximation two-sided p-value. Adequate at these n."""
    if t is None or not np.isfinite(t):
        return np.nan
    from math import erfc, sqrt
    return erfc(abs(t) / sqrt(2.0))


def fmt_p(p, thresh):
    if p is None or not np.isfinite(p):
        return "     na"
    star = "*" if p < thresh else " "
    return f"{p:>7.4f}{star}"


# ------------------------------------------------------------------- the build

def build(args):
    """Reuse the harness end to end: cache, collapse, score, join."""
    if args.all_seasons:
        seasons = list(eh.ALL_SEASONS)
    elif args.seasons:
        seasons = [int(s) for s in args.seasons.split(",") if s.strip()]
    else:
        seasons = list(eh.ALL_SEASONS)
    markets = [m.strip() for m in args.markets.split(",") if m.strip()]
    bad = [m for m in markets if m not in eh.MARKET_SPEC]
    if bad:
        print(f"unknown markets: {bad}")
        sys.exit(1)

    print("=" * 96)
    print("ODDS-SPACE TEST: is the price channel informative beyond the line?")
    print(f"  seasons {seasons}   markets {markets}")
    print("  devig: proportional   SEs: clustered on event_id")
    print("=" * 96)

    sec = eh._secrets()
    box = {}

    def client_factory():
        if "c" not in box:
            box["c"] = eh.connect(sec)
        return box["c"]

    lines = eh.load_lines(client_factory, seasons, markets, args.cache,
                          args.refresh)
    if lines.empty:
        print("  no lines found")
        sys.exit(1)
    print(f"  {len(lines)} prop rows, {lines['book'].nunique()} books")
    nullw = int(lines["week"].isna().sum())
    if nullw:
        print(f"  dropping {nullw} rows with a null week")
        lines = lines[lines["week"].notna()]

    props = eh.collapse_books(lines)
    print(f"  {len(props)} distinct player-week props")

    print("\nscoring models walk-forward (harness score_market, unchanged)")
    scored = []
    for m in markets:
        print(f"  {m}")
        s = eh.score_market(m, seasons, mode="walk_forward",
                            population=args.score_population)
        if len(s):
            print(f"    {len(s)} player-weeks scored")
            scored.append(s)
    if not scored:
        print("  nothing scored")
        sys.exit(1)
    proj = pd.concat(scored, ignore_index=True)

    from models import data_utils
    props["_key"] = data_utils.norm_join_name(props["player"])
    proj["_key"] = data_utils.norm_join_name(proj["player"])
    props["week"] = props["week"].astype(int)
    proj["week"] = proj["week"].astype(int)

    j = props.merge(
        proj[["season", "week", "market", "_key", "projection", "actual"]],
        on=["season", "week", "market", "_key"], how="inner")
    print(f"\n  joined {len(j)} rows "
          f"({100.0 * len(j) / max(len(props), 1):.1f}% of props)")
    if len(j) < 200:
        print("  too few joined rows")
        sys.exit(1)
    return j, markets


# ------------------------------------------------------------------- the gate

def gate(j, markets):
    """Reproduce the published FanDuel betas or abort.

    Fits the harness's own regression, (actual - line) on (projection - line),
    anchored on the FanDuel line, clustered on event_id. These are published
    numbers. If this pipeline cannot land on them, nothing downstream is
    trustworthy.
    """
    print("\n" + "=" * 96)
    print("GATE: reproduce the published FanDuel-anchored betas")
    print("=" * 96)
    print(f"  {'market':<14}{'n':>7}{'beta':>9}{'published':>11}"
          f"{'diff':>9}{'SE':>8}   verdict")
    failures = []
    for m in markets:
        sub = j[j["market"] == m].copy()
        sub = sub.dropna(subset=["line_fanduel", "projection", "actual"])
        if len(sub) < 50:
            print(f"  {m:<14}{len(sub):>7}   too few rows to gate")
            failures.append(m)
            continue
        dev = sub["projection"] - sub["line_fanduel"]
        out = sub["actual"] - sub["line_fanduel"]
        f = eh.ols_clustered(dev, out, sub["event_id"])
        if f is None:
            print(f"  {m:<14}{len(sub):>7}   fit failed")
            failures.append(m)
            continue
        pub = PUBLISHED_FD_BETA.get(m)
        if pub is None:
            print(f"  {m:<14}{f['n']:>7}{f['beta']:>9.3f}{'--':>11}"
                  f"{'--':>9}{f['se_beta']:>8.3f}   no published value")
            continue
        d = f["beta"] - pub
        ok = abs(d) <= GATE_TOL
        print(f"  {m:<14}{f['n']:>7}{f['beta']:>9.3f}{pub:>11.3f}"
              f"{d:>+9.3f}{f['se_beta']:>8.3f}   {'PASS' if ok else 'FAIL'}")
        if not ok:
            failures.append(m)

    if failures:
        print(f"\n  GATE FAILED for {failures}. Tolerance is {GATE_TOL:.3f}.")
        print("  This pipeline does not reproduce the published betas, so no")
        print("  new number from it is trustworthy. Stop and find out why")
        print("  before reading anything below. Do not widen the tolerance.")
        sys.exit(2)
    print(f"\n  GATE PASSED for all markets at tolerance {GATE_TOL:.3f}.")


# -------------------------------------------------------------- the derivation

def derive(j):
    """Add devigged book probability and the realised over indicator."""
    po, hold = [], []
    for o, u in zip(j["over_odds"], j["under_odds"]):
        p, h = devig_proportional(o, u)
        po.append(p)
        hold.append(h)
    j["book_p_over"] = po
    j["hold"] = hold

    n0 = len(j)
    j = j.dropna(subset=["line_fanduel", "actual", "book_p_over"]).copy()
    print(f"\n  rows with a usable two-sided FanDuel price: {len(j)} of {n0}")

    push = (j["actual"] == j["line_fanduel"])
    n_push = int(push.sum())
    j = j[~push].copy()
    j["over"] = (j["actual"] > j["line_fanduel"]).astype(float)
    print(f"  pushes dropped (actual exactly on the line): {n_push}")
    print(f"  median hold: {j['hold'].median():.4f}   "
          f"mean hold: {j['hold'].mean():.4f}")
    return j


# --------------------------------------------------------------------- test A

def test_a(j, markets):
    """Is the book itself calibrated, once the vig is stripped out?

    This is the baseline nothing in the project has measured. If the book is
    well calibrated, the bar for any rule is the hold. If it is not, the
    miscalibration is where a rule would live.
    """
    print("\n" + "=" * 96)
    print("TEST A: is FanDuel's devigged price calibrated?")
    print("=" * 96)
    print("  Sign convention: gap = actual over rate minus devigged price.")
    print("  Positive means overs hit MORE often than the price implied.\n")
    edges = [0.0, 0.35, 0.42, 0.46, 0.50, 0.54, 0.58, 0.65, 1.0]
    for m in markets:
        sub = j[j["market"] == m]
        if len(sub) < 100:
            print(f"  {m}: {len(sub)} rows, skipped")
            continue
        print(f"  {m}  (n={len(sub)})")
        print(f"    {'price band':<16}{'n':>7}{'mean price':>12}"
              f"{'over rate':>11}{'gap pp':>9}{'SE pp':>8}")
        for lo, hi in zip(edges[:-1], edges[1:]):
            b = sub[(sub["book_p_over"] >= lo) & (sub["book_p_over"] < hi)]
            if len(b) < 40:
                continue
            cm = eh.cluster_mean(b["over"], b["event_id"])
            if cm is None:
                continue
            mp = b["book_p_over"].mean()
            gap = (cm["mean"] - mp) * 100.0
            se = cm["se"] * 100.0
            print(f"    {lo:.2f} to {hi:.2f}    {len(b):>7}{mp:>12.4f}"
                  f"{cm['mean']:>11.4f}{gap:>+9.1f}{se:>8.1f}")
        print()


# --------------------------------------------------------------------- test B

def test_b(j, markets):
    """THE KEY TEST. At a fixed line, does the price predict the outcome?

    Within estimator: book_p_over is demeaned inside each (market, line)
    stratum, so only price variation at an identical line identifies the
    slope. Strata with fewer than 40 rows are dropped, and strata with no
    price variation contribute nothing by construction.

    Slope near 1 means the price is fully informative at the margin.
    Slope near 0 means price variation at a fixed line predicts nothing,
    which is what template pricing would look like.
    """
    print("=" * 96)
    print("TEST B: does price variation at a FIXED line predict the outcome?")
    print("=" * 96)
    n_tests = len(markets)
    thresh = 0.05 / max(n_tests, 1)
    print(f"  {n_tests} markets tested. Bonferroni threshold {thresh:.4f}, "
          f"raw threshold 0.0500.")
    print("  A slope of 1.0 means the price is fully informative at the")
    print("  margin. A slope of 0.0 means it carries nothing the line does")
    print("  not already carry.\n")
    print(f"  {'market':<14}{'n':>7}{'strata':>8}{'slope':>9}{'SE':>8}"
          f"{'t':>8}{'p':>9}{'MDE':>8}")
    results = {}
    for m in markets:
        sub = j[j["market"] == m].copy()
        sub = sub[sub["line_fanduel"].notna()]
        if len(sub) < 100:
            print(f"  {m:<14}{len(sub):>7}   too few rows")
            continue
        g = sub.groupby("line_fanduel")
        keep = g["over"].transform("size") >= 40
        sub = sub[keep].copy()
        if len(sub) < 100:
            print(f"  {m:<14}{len(sub):>7}   too few rows after strata filter")
            continue
        sub["p_within"] = (sub["book_p_over"]
                           - sub.groupby("line_fanduel")["book_p_over"]
                           .transform("mean"))
        sub["over_within"] = (sub["over"]
                              - sub.groupby("line_fanduel")["over"]
                              .transform("mean"))
        n_strata = int(sub["line_fanduel"].nunique())
        f = eh.ols_clustered(sub["p_within"], sub["over_within"],
                             sub["event_id"])
        if f is None:
            print(f"  {m:<14}{len(sub):>7}   fit failed")
            continue
        t = f["beta"] / f["se_beta"] if f["se_beta"] > 0 else np.nan
        p = two_sided_p(t)
        print(f"  {m:<14}{f['n']:>7}{n_strata:>8}{f['beta']:>9.3f}"
              f"{f['se_beta']:>8.3f}{t:>8.2f}{fmt_p(p, thresh)}"
              f"{mde(f['se_beta']):>8.3f}")
        results[m] = (f, p, thresh)

    print("\n  Read the MDE column before calling any row flat. A slope of")
    print("  0.05 with an MDE of 0.60 is not evidence of no effect.")
    return results


def test_b_by_line(j, markets, min_rows=150):
    """Per-line detail for the markets where Test B has something to show."""
    print("\n" + "=" * 96)
    print("TEST B DETAIL: slope within each individual line")
    print("=" * 96)
    for m in markets:
        sub = j[j["market"] == m]
        lines_ = (sub.groupby("line_fanduel").size()
                  .sort_values(ascending=False))
        lines_ = lines_[lines_ >= min_rows]
        if not len(lines_):
            continue
        print(f"  {m}")
        print(f"    {'line':>8}{'n':>7}{'price sd':>10}{'slope':>9}"
              f"{'SE':>8}{'t':>8}{'over rate':>11}")
        for ln in sorted(lines_.index):
            b = sub[sub["line_fanduel"] == ln]
            f = eh.ols_clustered(b["book_p_over"], b["over"], b["event_id"])
            if f is None:
                continue
            t = f["beta"] / f["se_beta"] if f["se_beta"] > 0 else np.nan
            print(f"    {ln:>8.1f}{f['n']:>7}{b['book_p_over'].std():>10.4f}"
                  f"{f['beta']:>9.3f}{f['se_beta']:>8.3f}{t:>8.2f}"
                  f"{b['over'].mean():>11.4f}")
        print()


# --------------------------------------------------------------------- test C

def test_c(j, markets):
    """Does the MODEL's probability beat the book's, in probability space?

    Needs the app's pricing layer, so this is the only part that depends on
    mc / mc_pricing. Post-4.4 mc.prob_over prices the blended mean, so in a
    beta-clipped market the model probability is a function of the LINE alone
    and carries no player-specific content. In those markets this test is
    mechanically the negative of Test B, which is the point: it shows directly
    whether disagreeing with the price means disagreeing with information.

    MIRRORED FILTER RULE: prob_over returns None below the projection floor,
    so those rows never display in the app. They are dropped here and counted,
    rather than scored as though the board showed them.
    """
    print("=" * 96)
    print("TEST C: does the model's probability add anything to the price?")
    print("=" * 96)
    try:
        import mc
    except Exception as e:
        print(f"  could not import mc ({e}). Skipping Test C.")
        return
    rows = []
    for m in markets:
        sub = j[j["market"] == m]
        for _, r in sub.iterrows():
            try:
                p = mc.prob_over(m, float(r["projection"]),
                                 float(r["line_fanduel"]), 0)
            except Exception:
                p = None
            rows.append(p if p is not None else np.nan)
    j = j.copy()
    j["model_p"] = rows
    n_floor = int(j["model_p"].isna().sum())
    print(f"  rows returning None from prob_over (below the projection "
          f"floor, never displayed): {n_floor} of {len(j)}")
    j = j.dropna(subset=["model_p"]).copy()
    j["disagree"] = j["model_p"] - j["book_p_over"]

    n_tests = len(markets)
    thresh = 0.05 / max(n_tests, 1)
    print(f"  Bonferroni threshold {thresh:.4f}\n")
    print(f"  {'market':<14}{'n':>7}{'slope':>9}{'SE':>8}{'t':>8}"
          f"{'p':>9}{'MDE':>8}{'mean|dis|':>11}")
    for m in markets:
        sub = j[j["market"] == m]
        if len(sub) < 100:
            print(f"  {m:<14}{len(sub):>7}   too few rows")
            continue
        f = eh.ols_clustered(sub["disagree"], sub["over"], sub["event_id"])
        if f is None:
            print(f"  {m:<14}{len(sub):>7}   fit failed")
            continue
        t = f["beta"] / f["se_beta"] if f["se_beta"] > 0 else np.nan
        p = two_sided_p(t)
        print(f"  {m:<14}{f['n']:>7}{f['beta']:>9.3f}{f['se_beta']:>8.3f}"
              f"{t:>8.2f}{fmt_p(p, thresh)}{mde(f['se_beta']):>8.3f}"
              f"{sub['disagree'].abs().mean():>11.4f}")
    print("\n  A positive slope means the model's disagreement with the price")
    print("  predicts the outcome. A negative slope means disagreeing with")
    print("  the price is disagreeing with real information.")


# ----------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all-seasons", action="store_true")
    ap.add_argument("--seasons", default=None, help="comma separated")
    ap.add_argument("--markets", default=",".join(eh.DEFAULT_MARKETS))
    ap.add_argument("--cache", default="lines_cache.parquet")
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--score-population", default="all",
                    choices=["all", "board"])
    ap.add_argument("--with-model", action="store_true",
                    help="run Test C, which needs mc.prob_over")
    ap.add_argument("--save-rows", default=None,
                    help="write the derived row set to CSV")
    ap.add_argument("--skip-gate", action="store_true",
                    help=argparse.SUPPRESS)
    args = ap.parse_args()

    j, markets = build(args)

    if args.skip_gate:
        print("\n  WARNING: gate skipped. Every number below is unverified.")
    else:
        gate(j, markets)

    j = derive(j)

    test_a(j, markets)
    test_b(j, markets)
    test_b_by_line(j, markets)
    if args.with_model:
        test_c(j, markets)

    if args.save_rows:
        cols = ["season", "week", "market", "player", "line_fanduel",
                "over_odds", "under_odds", "book_p_over", "hold",
                "projection", "actual", "over", "event_id", "n_books"]
        cols = [c for c in cols if c in j.columns]
        j[cols].to_csv(args.save_rows, index=False, encoding="utf-8")
        print(f"\n  wrote {len(j)} rows to {args.save_rows}")

    print("\n" + "=" * 96)
    print("HOW TO READ THIS")
    print("=" * 96)
    print("  Test B is the one that decides. If the slope is significantly")
    print("  positive, the price channel carries player-specific information")
    print("  and a beta-zero model cannot be used to pick a side against it.")
    print("  If the slope is flat AND the MDE is small enough to make that")
    print("  meaningful, the channel is not informative at that line and a")
    print("  distributional rule has room. Report the MDE either way.")
    print("=" * 96)


if __name__ == "__main__":
    main()
