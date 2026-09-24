"""
ODDS-SPACE TEST v2: does the model know anything the PRICE does not?

WHAT v1 ESTABLISHED (run 2026-09-24, gate passed to three decimals)

  Receptions Test B slope 1.008, SE 0.090, t 11.18. FanDuel's devigged price
  at a fixed reception line is a fully efficient forecast at the margin: one
  point of price is worth one point of outcome probability.

  Receiving, rushing and qb_passing could not be tested. FanDuel prices those
  at flat -113 both sides, so the price standard deviation at a fixed line is
  0.0012 to 0.0082 against 0.047 to 0.075 for receptions, and 97 to 99.8
  percent of rows sit in the single band 0.50 to 0.54. Those slopes are not
  flat, they are undefined. MDE 4.096 for receiving and 27.068 for rushing.

  So FanDuel uses ONE channel per market and never both: a coarse line plus
  informative odds for receptions, a fine line plus flat odds for yardage.

WHAT v1 COULD NOT ANSWER, AND WHY THIS VERSION EXISTS

  Beta 0.226 says the model knows something the LINE does not. Nothing has
  tested whether it knows something the PRICE does not, and the price is the
  efficient part.

  v1's Test C was the wrong specification. Regressing the outcome on
  (model_p - book_p) forces the two terms to carry equal and opposite
  coefficients, which assumes the answer instead of measuring it.

  TEST D puts both in together:

      over = b0 + b1 * book_p_over + b2 * (projection - line)

  b2 positive with the price already in the regression means the model
  carries information FanDuel's odds do not. That is a side-picking rule for
  receptions.

  b2 collapsing to zero once book_p is included means beta 0.226 was the
  model rediscovering what the odds already said. In that case the candidate
  rule's +0.1475 needs re-examining on exactly this basis, because its
  threshold was selected in line space with the odds never consulted.

  The NESTED pair is printed side by side so the collapse is visible:
      over ~ dev                (model alone, no price)
      over ~ book_p + dev       (model against the price)

PRE-REGISTERED PREDICTION, recorded before the first v2 run

  b2 collapses by more than half and loses significance at the Bonferroni
  threshold. Reasoning: v1 measured the price slope at 1.008, which is what
  an efficient forecast looks like, and an efficient forecast should already
  contain a public-data projection. If b2 SURVIVES at close to its line-space
  value, the prediction is refuted and receptions has a real edge over the
  price rather than over the line.

  Stated so that being wrong is informative either way.

FIXED FROM v1

  The per-line detail table printed slopes of +46.15 at a price sd of 0.0012
  and a t of 71.64. That is division by near-zero variance, not a finding, and
  it is exactly what "a t-statistic of +6 is a bug until proven otherwise"
  exists to catch. Strata below MIN_PRICE_SD are now suppressed and counted.

THE GATES. TWO OF THEM.

  Gate 1: reproduce the published FanDuel-anchored betas (0.226, 0.066,
  -0.043, 0.085) within 0.030.

  Gate 2: reproduce v1's own published Test B receptions slope of 1.008
  within 0.030. v1's output is now a published number and this script
  reimplements part of its path, so the same rule applies to it.

  Neither gate may be widened to make a run pass.

Run from the repo root with the venv active:
    python odds_space_test_v2.py --all-seasons --cache lines_cache.parquet
    python odds_space_test_v2.py --all-seasons --cache lines_cache.parquet --placebo 200
    python odds_space_test_v2.py --all-seasons --cache lines_cache.parquet --save-rows odds_rows_v2.csv
"""
import argparse
import sys

import numpy as np
import pandas as pd

import eval_harness as eh

PUBLISHED_FD_BETA = {
    "receptions": 0.226,
    "receiving": 0.066,
    "rushing": -0.043,
    "qb_passing": 0.085,
}
PUBLISHED_V1_TESTB_RECEPTIONS = 1.008
GATE_TOL = 0.030

# Below this, price variation at a fixed line is numerically indistinguishable
# from none and a slope is a ratio of two near-zero quantities. Measured in
# v1: receptions strata run 0.047 to 0.075, yardage strata 0.0012 to 0.0082.
MIN_PRICE_SD = 0.015

# Strata smaller than this are dropped from the within estimator.
MIN_STRATUM_N = 40


# ------------------------------------------------------------------- devigging

def american_to_decimal(o):
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
    """Two-sided American odds to a vig-free P(over), and the hold."""
    do = american_to_decimal(over_odds)
    du = american_to_decimal(under_odds)
    if not (np.isfinite(do) and np.isfinite(du)):
        return np.nan, np.nan
    ro, ru = 1.0 / do, 1.0 / du
    tot = ro + ru
    if tot <= 0:
        return np.nan, np.nan
    return ro / tot, tot - 1.0


# -------------------------------------------------------- multivariate cluster

def ols_clustered_multi(X, y, cluster, names):
    """y = Xb with standard errors clustered on `cluster`.

    Same estimator as eval_harness.ols_clustered, generalised past one
    regressor. An intercept is added here, so do not pass one in X.

    Returns dict with coef, se, t and the design rank, or None if the design
    is rank deficient. Rank deficiency is reported rather than silently
    producing a pseudo-inverse answer, because a collinear regressor is
    exactly the failure mode this test is exposed to: if book_p and dev were
    perfectly collinear the horse race would be meaningless.
    """
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    c = np.asarray(cluster)
    if X.ndim == 1:
        X = X[:, None]
    ok = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    X, y, c = X[ok], y[ok], c[ok]
    n, k = X.shape
    if n < 20:
        return None
    D = np.column_stack([np.ones(n), X])
    rank = int(np.linalg.matrix_rank(D))
    if rank < k + 1:
        return {"rank_deficient": True, "n": n, "rank": rank, "k": k + 1}
    coef, *_ = np.linalg.lstsq(D, y, rcond=None)
    resid = y - D @ coef
    dtd_inv = np.linalg.inv(D.T @ D)

    meat = np.zeros((k + 1, k + 1))
    order = np.argsort(c)
    Ds, rs, cs = D[order], resid[order], c[order]
    starts = np.r_[0, np.flatnonzero(cs[1:] != cs[:-1]) + 1, len(cs)]
    g = len(starts) - 1
    for i in range(g):
        sl = slice(starts[i], starts[i + 1])
        u = Ds[sl].T @ rs[sl]
        meat += np.outer(u, u)
    scale = (g / max(g - 1, 1)) * ((n - 1) / max(n - k - 1, 1))
    cov = dtd_inv @ meat @ dtd_inv * scale
    se = np.sqrt(np.diag(cov))

    out = {"rank_deficient": False, "n": n, "clusters": g,
           "intercept": coef[0], "se_intercept": se[0],
           "resid_sd": float(np.sqrt((resid @ resid) / max(n - k - 1, 1)))}
    for i, nm in enumerate(names, start=1):
        out[f"b_{nm}"] = coef[i]
        out[f"se_{nm}"] = se[i]
        out[f"t_{nm}"] = coef[i] / se[i] if se[i] > 0 else np.nan
    return out


def mde(se, power=0.80):
    if se is None or not np.isfinite(se) or se <= 0:
        return np.nan
    return 2.80 * se


def two_sided_p(t):
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
    print("ODDS-SPACE TEST v2: does the model know anything the PRICE does not?")
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


# -------------------------------------------------------------------- gate one

def gate_betas(j, markets):
    print("\n" + "=" * 96)
    print("GATE 1: reproduce the published FanDuel-anchored betas")
    print("=" * 96)
    print(f"  {'market':<14}{'n':>7}{'beta':>9}{'published':>11}"
          f"{'diff':>9}{'SE':>8}   verdict")
    failures = []
    for m in markets:
        sub = j[j["market"] == m].dropna(
            subset=["line_fanduel", "projection", "actual"])
        if len(sub) < 50:
            print(f"  {m:<14}{len(sub):>7}   too few rows to gate")
            failures.append(m)
            continue
        f = eh.ols_clustered(sub["projection"] - sub["line_fanduel"],
                             sub["actual"] - sub["line_fanduel"],
                             sub["event_id"])
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
        print(f"\n  GATE 1 FAILED for {failures}. Tolerance {GATE_TOL:.3f}.")
        print("  Stop. Do not widen the tolerance.")
        sys.exit(2)
    print(f"\n  GATE 1 PASSED at tolerance {GATE_TOL:.3f}.")


# -------------------------------------------------------------------- gate two

def within_price_slope(sub):
    """v1's Test B within estimator, unchanged, for one market's frame."""
    sub = sub[sub["line_fanduel"].notna()].copy()
    keep = sub.groupby("line_fanduel")["over"].transform("size") >= MIN_STRATUM_N
    sub = sub[keep].copy()
    if len(sub) < 100:
        return None, sub
    sub["p_within"] = (sub["book_p_over"]
                       - sub.groupby("line_fanduel")["book_p_over"]
                       .transform("mean"))
    sub["over_within"] = (sub["over"]
                          - sub.groupby("line_fanduel")["over"]
                          .transform("mean"))
    f = eh.ols_clustered(sub["p_within"], sub["over_within"], sub["event_id"])
    return f, sub


def gate_v1(j):
    print("\n" + "=" * 96)
    print("GATE 2: reproduce v1's published Test B receptions slope")
    print("=" * 96)
    sub = j[j["market"] == "receptions"]
    f, _ = within_price_slope(sub)
    if f is None:
        print("  could not fit. GATE 2 FAILED.")
        sys.exit(2)
    d = f["beta"] - PUBLISHED_V1_TESTB_RECEPTIONS
    ok = abs(d) <= GATE_TOL
    print(f"  slope {f['beta']:.3f}   published {PUBLISHED_V1_TESTB_RECEPTIONS:.3f}"
          f"   diff {d:+.3f}   SE {f['se_beta']:.3f}   "
          f"{'PASS' if ok else 'FAIL'}")
    if not ok:
        print(f"\n  GATE 2 FAILED. Tolerance {GATE_TOL:.3f}. v2 does not")
        print("  reproduce v1, so nothing below is comparable to it.")
        sys.exit(2)
    print(f"  GATE 2 PASSED at tolerance {GATE_TOL:.3f}.")


# -------------------------------------------------------------- the derivation

def derive(j):
    po, hold = [], []
    for o, u in zip(j["over_odds"], j["under_odds"]):
        p, h = devig_proportional(o, u)
        po.append(p)
        hold.append(h)
    j = j.copy()
    j["book_p_over"] = po
    j["hold"] = hold

    n0 = len(j)
    j = j.dropna(subset=["line_fanduel", "actual", "book_p_over",
                         "projection"]).copy()
    print(f"\n  rows with a usable two-sided FanDuel price: {len(j)} of {n0}")

    push = (j["actual"] == j["line_fanduel"])
    n_push = int(push.sum())
    j = j[~push].copy()
    j["over"] = (j["actual"] > j["line_fanduel"]).astype(float)
    j["dev"] = j["projection"] - j["line_fanduel"]
    print(f"  pushes dropped: {n_push}")
    print(f"  median hold: {j['hold'].median():.4f}")
    return j


def price_variation_report(j, markets):
    """Say plainly which markets CAN be tested before testing them.

    v1's headline error was reading undefined slopes as flat ones. This table
    is printed first so no row below gets interpreted without it.
    """
    print("\n" + "=" * 96)
    print("PRICE VARIATION AT A FIXED LINE: which markets can be tested")
    print("=" * 96)
    print(f"  A slope needs variation in the regressor. Strata with a price")
    print(f"  SD below {MIN_PRICE_SD:.3f} are suppressed everywhere below.\n")
    print(f"  {'market':<14}{'n':>7}{'median stratum price SD':>26}"
          f"{'testable':>10}")
    testable = []
    for m in markets:
        sub = j[j["market"] == m]
        if len(sub) < 100:
            print(f"  {m:<14}{len(sub):>7}{'--':>26}{'no':>10}")
            continue
        sds = sub.groupby("line_fanduel")["book_p_over"].std()
        sizes = sub.groupby("line_fanduel").size()
        sds = sds[sizes >= MIN_STRATUM_N]
        med = float(sds.median()) if len(sds) else np.nan
        ok = np.isfinite(med) and med >= MIN_PRICE_SD
        print(f"  {m:<14}{len(sub):>7}{med:>26.4f}"
              f"{'YES' if ok else 'no':>10}")
        if ok:
            testable.append(m)
    print("\n  Markets marked no are priced at flat odds both sides. Their")
    print("  slopes are UNDEFINED rather than zero, and v1's MDE of 4.096")
    print("  for receiving and 27.068 for rushing is what that looks like.")
    return testable


# --------------------------------------------------------------------- test D

def test_d(j, testable):
    """THE HORSE RACE. Does dev survive once the price is in the regression?

    Nested pair, printed side by side:
        model alone   over ~ dev
        horse race    over ~ book_p_over + dev

    Units: dev is in the market's own units, so b2 is the change in P(over)
    per one reception of model deviation. book_p_over is a probability, so b1
    near 1.0 means the price is an efficient forecast.
    """
    print("\n" + "=" * 96)
    print("TEST D: over ~ book_p_over + dev, the horse race")
    print("=" * 96)
    if not testable:
        print("  no market has enough price variation. Nothing to race.")
        return {}
    n_tests = len(testable)
    thresh = 0.05 / max(n_tests, 1)
    print(f"  {n_tests} market(s) testable. Bonferroni threshold "
          f"{thresh:.4f}, raw 0.0500.\n")
    out = {}
    for m in testable:
        sub = j[j["market"] == m].copy()
        solo = eh.ols_clustered(sub["dev"], sub["over"], sub["event_id"])
        race = ols_clustered_multi(
            sub[["book_p_over", "dev"]].to_numpy(), sub["over"].to_numpy(),
            sub["event_id"].to_numpy(), ["book_p", "dev"])
        if solo is None or race is None:
            print(f"  {m}: fit failed")
            continue
        if race.get("rank_deficient"):
            print(f"  {m}: design rank deficient "
                  f"(rank {race['rank']} of {race['k']}). book_p and dev are")
            print("    collinear, so the horse race cannot be run. That is")
            print("    itself a finding: the price IS the projection.")
            continue

        p_solo = two_sided_p(solo["beta"] / solo["se_beta"])
        p_dev = two_sided_p(race["t_dev"])
        p_bp = two_sided_p(race["t_book_p"])
        shrink = (1.0 - race["b_dev"] / solo["beta"]) * 100.0 \
            if solo["beta"] != 0 else np.nan

        print(f"  {m}  (n={race['n']}, {race['clusters']} games)")
        print(f"    {'specification':<28}{'coef':>9}{'SE':>8}{'t':>8}"
              f"{'p':>9}{'MDE':>8}")
        print(f"    {'dev alone':<28}{solo['beta']:>9.4f}"
              f"{solo['se_beta']:>8.4f}"
              f"{solo['beta'] / solo['se_beta']:>8.2f}"
              f"{fmt_p(p_solo, thresh)}{mde(solo['se_beta']):>8.4f}")
        print(f"    {'dev, price controlled':<28}{race['b_dev']:>9.4f}"
              f"{race['se_dev']:>8.4f}{race['t_dev']:>8.2f}"
              f"{fmt_p(p_dev, thresh)}{mde(race['se_dev']):>8.4f}")
        print(f"    {'book_p, dev controlled':<28}{race['b_book_p']:>9.4f}"
              f"{race['se_book_p']:>8.4f}{race['t_book_p']:>8.2f}"
              f"{fmt_p(p_bp, thresh)}{mde(race['se_book_p']):>8.4f}")
        print(f"\n    dev shrinks {shrink:+.1f}% once the price is included.")
        if np.isfinite(p_dev) and p_dev >= thresh:
            print(f"    dev does NOT survive at the Bonferroni threshold.")
            print(f"    MDE on dev is {mde(race['se_dev']):.4f}, so read this")
            print(f"    as 'no effect larger than that', not as a zero.")
        else:
            print(f"    dev SURVIVES the price control. This is the result")
            print(f"    that would support a side-picking rule.")
        print()
        out[m] = {"solo": solo, "race": race, "thresh": thresh}
    return out


def placebo_dev(j, testable, n_shuffles, seed=0):
    """Shuffle dev within (market, line) and refit the horse race.

    Confirms the b2 null distribution sits at zero under this design. If the
    placebo mean is not near zero, b2 carries a mechanical component and the
    honest value is the difference, same structure as the harness's
    best-line placebo.
    """
    if not n_shuffles or not testable:
        return
    print("=" * 96)
    print(f"PLACEBO: dev shuffled within (market, line), {n_shuffles} draws")
    print("=" * 96)
    rng = np.random.default_rng(seed)
    for m in testable:
        sub = j[j["market"] == m].copy()
        vals = []
        for _ in range(n_shuffles):
            s = sub.copy()
            s["dev_shuf"] = (s.groupby("line_fanduel")["dev"]
                             .transform(lambda v: rng.permutation(v.to_numpy())))
            r = ols_clustered_multi(
                s[["book_p_over", "dev_shuf"]].to_numpy(),
                s["over"].to_numpy(), s["event_id"].to_numpy(),
                ["book_p", "dev"])
            if r and not r.get("rank_deficient"):
                vals.append(r["b_dev"])
        if not vals:
            print(f"  {m}: no usable draws")
            continue
        vals = np.asarray(vals)
        print(f"  {m}: placebo b_dev mean {vals.mean():+.5f}, "
              f"sd {vals.std():.5f}, "
              f"95% range [{np.percentile(vals, 2.5):+.5f}, "
              f"{np.percentile(vals, 97.5):+.5f}]")
    print()


def test_b_detail(j, testable, min_rows=150):
    """Per-line detail, with low-variance strata suppressed. v1 bug fixed."""
    print("=" * 96)
    print("TEST B DETAIL: price slope within each line (low-SD suppressed)")
    print("=" * 96)
    for m in testable:
        sub = j[j["market"] == m]
        sizes = sub.groupby("line_fanduel").size()
        cand = sizes[sizes >= min_rows].index
        print(f"  {m}")
        print(f"    {'line':>8}{'n':>7}{'price sd':>10}{'slope':>9}"
              f"{'SE':>8}{'t':>8}{'over rate':>11}")
        n_supp = 0
        for ln in sorted(cand):
            b = sub[sub["line_fanduel"] == ln]
            sd = float(b["book_p_over"].std())
            if not np.isfinite(sd) or sd < MIN_PRICE_SD:
                n_supp += 1
                continue
            f = eh.ols_clustered(b["book_p_over"], b["over"], b["event_id"])
            if f is None:
                continue
            print(f"    {ln:>8.1f}{f['n']:>7}{sd:>10.4f}{f['beta']:>9.3f}"
                  f"{f['se_beta']:>8.3f}"
                  f"{f['beta'] / f['se_beta']:>8.2f}"
                  f"{b['over'].mean():>11.4f}")
        if n_supp:
            print(f"    {n_supp} line(s) suppressed for price SD below "
                  f"{MIN_PRICE_SD:.3f}")
        print()


# ----------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all-seasons", action="store_true")
    ap.add_argument("--seasons", default=None)
    ap.add_argument("--markets", default=",".join(eh.DEFAULT_MARKETS))
    ap.add_argument("--cache", default="lines_cache.parquet")
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--score-population", default="all",
                    choices=["all", "board"])
    ap.add_argument("--placebo", type=int, default=0,
                    help="shuffles for the dev placebo, e.g. 200")
    ap.add_argument("--save-rows", default=None)
    args = ap.parse_args()

    j, markets = build(args)
    gate_betas(j, markets)
    j = derive(j)
    gate_v1(j)

    testable = price_variation_report(j, markets)
    res = test_d(j, testable)
    placebo_dev(j, testable, args.placebo)
    test_b_detail(j, testable)

    if args.save_rows:
        cols = ["season", "week", "market", "player", "line_fanduel",
                "over_odds", "under_odds", "book_p_over", "hold",
                "projection", "dev", "actual", "over", "event_id", "n_books"]
        cols = [c for c in cols if c in j.columns]
        j[cols].to_csv(args.save_rows, index=False, encoding="utf-8")
        print(f"  wrote {len(j)} rows to {args.save_rows}")

    print("=" * 96)
    print("HOW TO READ TEST D")
    print("=" * 96)
    print("  b_book_p near 1.0 confirms the price is an efficient forecast.")
    print("  b_dev is the question. If it survives with the price")
    print("  controlled, the model carries information the odds do not, and")
    print("  that is a side-picking rule. If it collapses, beta 0.226 was")
    print("  the model rediscovering the odds, and the candidate rule's")
    print("  threshold was selected in line space without ever consulting")
    print("  the price it would have been betting against.")
    print("  Either way, report the MDE beside the verdict.")
    print("=" * 96)


if __name__ == "__main__":
    main()
