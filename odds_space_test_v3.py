"""
ODDS-SPACE TEST v3: the corrected horse race, and the slope-above-1 loose end

WHAT v2 ESTABLISHED, AND THE ERROR IT CONTAINED (run 2026-09-24)

  Test D: dev alone +0.0431 (t 5.39). dev with book_p controlled +0.0126
  (t 1.46, p 0.1433). book_p with dev controlled +0.9856 (t 10.39). dev
  shrank 70.7 percent once the price entered.

  THE PLACEBO CAUGHT A REAL DEFECT. Shuffling dev within (market, line)
  returned a mean b_dev of +0.00850, sd 0.00710. That is about 17 standard
  errors of the mean above zero, so the specification carried a mechanical
  component rather than a null at zero.

  MY ERROR: Test D pooled all six line strata with NO line fixed effects.
  The over rate falls monotonically with the line (0.5004 at 1.5 down to
  0.4407 at 5.5) and mean dev differs by stratum, so between-stratum
  covariance loaded onto dev. Shuffling WITHIN stratum preserves each
  stratum's mean, which is why the placebo retained the artifact instead of
  destroying it. Same structure the harness documents at point 7: the placebo
  mean is the artifact, the honest value is the difference.

  Taking the difference by hand gave +0.0126 - 0.0085 = about +0.0041 against
  a placebo sd of 0.0071. TEST E below measures it properly instead of
  subtracting two numbers, by demeaning every term inside its line stratum so
  the line level cannot contribute to anything.

THE LOOSE END v2 TURNED UP

  Per-line price slopes rise monotonically with the line:

      1.5   0.838   SE 0.188
      2.5   0.927   SE 0.150
      3.5   1.017   SE 0.188
      4.5   1.158   SE 0.255
      5.5   1.636   SE 0.378
      6.5   1.594   SE 0.631

  A slope above 1 means the price UNDER-reacts: when FanDuel moves its
  reception odds, the outcome moves further than the price did. Every SE
  overlaps 1.0 individually, but six strata in monotone order is suggestive,
  and it is a claim about the BOOK rather than about the projection, so it
  survives everything Test E is about to conclude.

  TEST F tests it two ways: each slope against 1 rather than against 0, and a
  pooled interaction of price with line level.

  TEST G asks the only question that pays. A slope above 1 is exploitable
  only if the under-reaction exceeds the hold, so G measures REALISED ROI of
  price-extremity rules directly from the American odds. No devigging, no
  probability conversion, no hand arithmetic: profit is (decimal - 1) on a
  win and -1 on a loss. That is the number a bet actually returns.

PRE-REGISTERED PREDICTIONS, recorded before the first v3 run

  E1. b_dev under line fixed effects lands within one SE of zero, and the
      placebo mean drops to within one placebo sd of zero. If the placebo is
      STILL offset, the line-FE fix was not the whole defect and something
      else is loading onto dev.
  E2. b_book_p under line fixed effects stays near 1.0, close to the 1.008
      that v1 measured and v2's gate reproduced.
  F1. The slope-by-line interaction is positive but does NOT clear its
      Bonferroni threshold. Six overlapping SEs rarely survive a pooled test.
  G1. Every price-extremity ROI lands negative, between -0.02 and -0.06,
      because the hold is about 5.7 points and a slope of 1.0 to 1.6 does not
      obviously beat it.

  If G1 is refuted the project has a betting route that does not use the
  model at all, which would be the largest result in it.

THE GATES. THREE NOW.

  Gate 1: the published FanDuel betas (0.226, 0.066, -0.043, 0.085).
  Gate 2: v1's Test B receptions slope, 1.008.
  Gate 3: v2's own Test D triple, 0.0431 / 0.0126 / 0.9856, reproduced by
          re-running v2's exact pooled specification before the corrected one
          runs. A script that supersedes a number must first reproduce it.

  No gate may be widened to make a run pass.

Run from the repo root with the venv active:
    python odds_space_test_v3.py --all-seasons --cache lines_cache.parquet --placebo 200
    python odds_space_test_v3.py --all-seasons --cache lines_cache.parquet --save-rows odds_rows_v3.csv
"""
import argparse
import sys

import numpy as np
import pandas as pd

import eval_harness as eh

PUBLISHED_FD_BETA = {
    "receptions": 0.226, "receiving": 0.066,
    "rushing": -0.043, "qb_passing": 0.085,
}
PUBLISHED_V1_TESTB_RECEPTIONS = 1.008
PUBLISHED_V2_TESTD = {"dev_solo": 0.0431, "dev_race": 0.0126,
                      "book_p_race": 0.9856}
GATE_TOL = 0.030
GATE_TOL_D = 0.010

MIN_PRICE_SD = 0.015
MIN_STRATUM_N = 40


# ------------------------------------------------------------------- devigging

def american_to_decimal(o):
    try:
        o = float(o)
    except (TypeError, ValueError):
        return np.nan
    if not np.isfinite(o) or o == 0:
        return np.nan
    return 1.0 + o / 100.0 if o > 0 else 1.0 + 100.0 / (-o)


def devig_proportional(over_odds, under_odds):
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
    """y = Xb, SEs clustered on `cluster`. Intercept added internally.

    Self-tested against planted coefficients: recovers 1.4913 and -0.7967
    against planted 1.5 and -0.8 with correlated regressors over 400
    clusters, and flags rank deficiency on a collinear pair.
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
    if int(np.linalg.matrix_rank(D)) < k + 1:
        return {"rank_deficient": True, "n": n, "k": k + 1}
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
           "intercept": coef[0], "se_intercept": se[0]}
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
    return f"{p:>7.4f}{'*' if p < thresh else ' '}"


def demean_within(df, col, by):
    return df[col] - df.groupby(by)[col].transform("mean")


# ------------------------------------------------------------------- the build

def build(args):
    seasons = (list(eh.ALL_SEASONS) if args.all_seasons or not args.seasons
               else [int(s) for s in args.seasons.split(",") if s.strip()])
    markets = [m.strip() for m in args.markets.split(",") if m.strip()]
    bad = [m for m in markets if m not in eh.MARKET_SPEC]
    if bad:
        print(f"unknown markets: {bad}")
        sys.exit(1)

    print("=" * 96)
    print("ODDS-SPACE TEST v3: corrected horse race, and the slope-above-1 test")
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
        lines = lines[lines["week"].notna()]
        print(f"  dropped {nullw} rows with a null week")

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
        sys.exit(1)
    return j, markets


def derive(j):
    po, hold, do, du = [], [], [], []
    for o, u in zip(j["over_odds"], j["under_odds"]):
        p, h = devig_proportional(o, u)
        po.append(p)
        hold.append(h)
        do.append(american_to_decimal(o))
        du.append(american_to_decimal(u))
    j = j.copy()
    j["book_p_over"] = po
    j["hold"] = hold
    j["dec_over"] = do
    j["dec_under"] = du

    n0 = len(j)
    j = j.dropna(subset=["line_fanduel", "actual", "book_p_over",
                         "projection", "dec_over", "dec_under"]).copy()
    print(f"\n  rows with a usable two-sided FanDuel price: {len(j)} of {n0}")
    push = (j["actual"] == j["line_fanduel"])
    j = j[~push].copy()
    j["over"] = (j["actual"] > j["line_fanduel"]).astype(float)
    j["dev"] = j["projection"] - j["line_fanduel"]
    print(f"  pushes dropped: {int(push.sum())}")
    print(f"  median hold: {j['hold'].median():.4f}")
    return j


def strata_filter(sub):
    """Keep line strata with enough rows AND real price variation."""
    sizes = sub.groupby("line_fanduel")["over"].transform("size")
    sds = sub.groupby("line_fanduel")["book_p_over"].transform("std")
    return sub[(sizes >= MIN_STRATUM_N) & (sds >= MIN_PRICE_SD)].copy()


# ---------------------------------------------------------------------- gates

def gate_betas(j, markets):
    print("\n" + "=" * 96)
    print("GATE 1: reproduce the published FanDuel-anchored betas")
    print("=" * 96)
    fails = []
    print(f"  {'market':<14}{'n':>7}{'beta':>9}{'published':>11}{'diff':>9}"
          f"   verdict")
    for m in markets:
        sub = j[j["market"] == m].dropna(
            subset=["line_fanduel", "projection", "actual"])
        f = eh.ols_clustered(sub["projection"] - sub["line_fanduel"],
                             sub["actual"] - sub["line_fanduel"],
                             sub["event_id"]) if len(sub) >= 50 else None
        pub = PUBLISHED_FD_BETA.get(m)
        if f is None or pub is None:
            print(f"  {m:<14}{len(sub):>7}   not gated")
            continue
        d = f["beta"] - pub
        ok = abs(d) <= GATE_TOL
        print(f"  {m:<14}{f['n']:>7}{f['beta']:>9.3f}{pub:>11.3f}{d:>+9.3f}"
              f"   {'PASS' if ok else 'FAIL'}")
        if not ok:
            fails.append(m)
    if fails:
        print(f"\n  GATE 1 FAILED for {fails}. Stop. Do not widen tolerance.")
        sys.exit(2)
    print(f"  GATE 1 PASSED at {GATE_TOL:.3f}.")


def gate_v1(j):
    print("\n" + "=" * 96)
    print("GATE 2: reproduce v1's Test B receptions price slope")
    print("=" * 96)
    sub = strata_filter(j[j["market"] == "receptions"])
    pw = demean_within(sub, "book_p_over", "line_fanduel")
    ow = demean_within(sub, "over", "line_fanduel")
    f = eh.ols_clustered(pw, ow, sub["event_id"])
    if f is None:
        print("  fit failed. GATE 2 FAILED.")
        sys.exit(2)
    d = f["beta"] - PUBLISHED_V1_TESTB_RECEPTIONS
    ok = abs(d) <= GATE_TOL
    print(f"  slope {f['beta']:.3f}   published "
          f"{PUBLISHED_V1_TESTB_RECEPTIONS:.3f}   diff {d:+.3f}   "
          f"{'PASS' if ok else 'FAIL'}")
    if not ok:
        sys.exit(2)


def gate_v2_testd(j):
    """Re-run v2's EXACT pooled specification, artifact and all.

    This reproduces a number this script is about to supersede. That is the
    point: without it, a different answer from Test E could be the fix or
    could be a new bug, and the two are indistinguishable.
    """
    print("\n" + "=" * 96)
    print("GATE 3: reproduce v2's Test D (pooled, no line fixed effects)")
    print("=" * 96)
    sub = j[j["market"] == "receptions"].copy()
    solo = eh.ols_clustered(sub["dev"], sub["over"], sub["event_id"])
    race = ols_clustered_multi(sub[["book_p_over", "dev"]].to_numpy(),
                               sub["over"].to_numpy(),
                               sub["event_id"].to_numpy(), ["book_p", "dev"])
    got = {"dev_solo": solo["beta"], "dev_race": race["b_dev"],
           "book_p_race": race["b_book_p"]}
    fails = []
    print(f"  {'quantity':<16}{'got':>10}{'published':>11}{'diff':>9}"
          f"   verdict")
    for k, pub in PUBLISHED_V2_TESTD.items():
        d = got[k] - pub
        ok = abs(d) <= GATE_TOL_D
        print(f"  {k:<16}{got[k]:>10.4f}{pub:>11.4f}{d:>+9.4f}"
              f"   {'PASS' if ok else 'FAIL'}")
        if not ok:
            fails.append(k)
    if fails:
        print(f"\n  GATE 3 FAILED for {fails} at tolerance {GATE_TOL_D:.3f}.")
        print("  v3 does not reproduce v2, so Test E is not comparable to it.")
        sys.exit(2)
    print(f"  GATE 3 PASSED at {GATE_TOL_D:.3f}.")


# --------------------------------------------------------------------- test E

def test_e(j, n_shuffles, seed=0):
    """The corrected horse race: every term demeaned inside its line stratum.

    over_within = b1 * book_p_within + b2 * dev_within

    Line level cannot contribute to anything, which is what v2 got wrong.
    The placebo shuffles dev within stratum and should now centre on zero.
    """
    print("\n" + "=" * 96)
    print("TEST E: horse race WITH line fixed effects (supersedes Test D)")
    print("=" * 96)
    sub = strata_filter(j[j["market"] == "receptions"])
    sub["pw"] = demean_within(sub, "book_p_over", "line_fanduel")
    sub["dw"] = demean_within(sub, "dev", "line_fanduel")
    sub["ow"] = demean_within(sub, "over", "line_fanduel")

    solo = eh.ols_clustered(sub["dw"], sub["ow"], sub["event_id"])
    race = ols_clustered_multi(sub[["pw", "dw"]].to_numpy(),
                               sub["ow"].to_numpy(),
                               sub["event_id"].to_numpy(), ["book_p", "dev"])
    if solo is None or race is None or race.get("rank_deficient"):
        print("  fit failed or rank deficient")
        return
    thresh = 0.05 / 3.0
    print(f"  n={race['n']}, {race['clusters']} games, "
          f"{sub['line_fanduel'].nunique()} line strata")
    print(f"  3 coefficients reported, Bonferroni threshold {thresh:.4f}\n")
    print(f"  {'specification':<30}{'coef':>10}{'SE':>9}{'t':>8}{'p':>9}"
          f"{'MDE':>9}")
    rows = [
        ("dev alone, line FE", solo["beta"], solo["se_beta"]),
        ("dev, price controlled", race["b_dev"], race["se_dev"]),
        ("book_p, dev controlled", race["b_book_p"], race["se_book_p"]),
    ]
    for lab, b, se in rows:
        t = b / se if se > 0 else np.nan
        print(f"  {lab:<30}{b:>10.4f}{se:>9.4f}{t:>8.2f}"
              f"{fmt_p(two_sided_p(t), thresh)}{mde(se):>9.4f}")

    if n_shuffles:
        rng = np.random.default_rng(seed)
        vals = []
        for _ in range(n_shuffles):
            s = sub.copy()
            s["dw_shuf"] = (s.groupby("line_fanduel")["dw"]
                            .transform(lambda v: rng.permutation(v.to_numpy())))
            r = ols_clustered_multi(s[["pw", "dw_shuf"]].to_numpy(),
                                    s["ow"].to_numpy(),
                                    s["event_id"].to_numpy(),
                                    ["book_p", "dev"])
            if r and not r.get("rank_deficient"):
                vals.append(r["b_dev"])
        if vals:
            v = np.asarray(vals)
            se_mean = v.std() / np.sqrt(len(v))
            offset_t = v.mean() / se_mean if se_mean > 0 else np.nan
            print(f"\n  PLACEBO, dev shuffled within stratum, "
                  f"{len(v)} draws")
            print(f"    mean {v.mean():+.5f}   sd {v.std():.5f}   "
                  f"SE of mean {se_mean:.5f}   t vs 0 {offset_t:+.2f}")
            print(f"    95% range [{np.percentile(v, 2.5):+.5f}, "
                  f"{np.percentile(v, 97.5):+.5f}]")
            if abs(offset_t) > 3:
                print("    PLACEBO STILL OFFSET. The line-FE fix was not the")
                print("    whole defect. The honest b_dev is the difference")
                print("    from this mean, and something else loads onto dev.")
            else:
                print("    Placebo centred on zero. b_dev above is honest.")


# --------------------------------------------------------------------- test F

def test_f(j):
    """Is the price slope genuinely ABOVE 1, and does it rise with the line?

    Two forms. Per-stratum slopes tested against 1 rather than 0, and a
    pooled interaction of the within-stratum price with the line level.
    """
    print("\n" + "=" * 96)
    print("TEST F: is the price slope above 1, and does it rise with line?")
    print("=" * 96)
    sub = strata_filter(j[j["market"] == "receptions"])
    lines_ = sorted(sub["line_fanduel"].unique())
    thresh = 0.05 / max(len(lines_) + 1, 1)
    print(f"  {len(lines_)} strata plus one pooled interaction = "
          f"{len(lines_) + 1} tests, Bonferroni {thresh:.4f}")
    print("  A slope above 1 means the price UNDER-reacts.\n")
    print(f"  {'line':>8}{'n':>7}{'slope':>9}{'SE':>8}{'t vs 0':>9}"
          f"{'t vs 1':>9}{'p vs 1':>9}")
    for ln in lines_:
        b = sub[sub["line_fanduel"] == ln]
        f = eh.ols_clustered(b["book_p_over"], b["over"], b["event_id"])
        if f is None:
            continue
        t0 = f["beta"] / f["se_beta"]
        t1 = (f["beta"] - 1.0) / f["se_beta"]
        print(f"  {ln:>8.1f}{f['n']:>7}{f['beta']:>9.3f}{f['se_beta']:>8.3f}"
              f"{t0:>9.2f}{t1:>9.2f}{fmt_p(two_sided_p(t1), thresh)}")

    sub["pw"] = demean_within(sub, "book_p_over", "line_fanduel")
    sub["ow"] = demean_within(sub, "over", "line_fanduel")
    cl = sub["line_fanduel"] - sub["line_fanduel"].mean()
    sub["pw_x_line"] = sub["pw"] * cl
    r = ols_clustered_multi(sub[["pw", "pw_x_line"]].to_numpy(),
                            sub["ow"].to_numpy(),
                            sub["event_id"].to_numpy(), ["pw", "inter"])
    if r and not r.get("rank_deficient"):
        print(f"\n  POOLED INTERACTION, price x (line - mean line)")
        print(f"    {'term':<28}{'coef':>10}{'SE':>9}{'t':>8}{'p':>9}"
              f"{'MDE':>9}")
        for nm, lab in (("pw", "price at the mean line"),
                        ("inter", "price x line")):
            b, se = r[f"b_{nm}"], r[f"se_{nm}"]
            t = r[f"t_{nm}"]
            print(f"    {lab:<28}{b:>10.4f}{se:>9.4f}{t:>8.2f}"
                  f"{fmt_p(two_sided_p(t), thresh)}{mde(se):>9.4f}")
        print("\n    A positive interaction means the under-reaction grows")
        print("    with the line. Read the MDE before calling it absent.")


# --------------------------------------------------------------------- test G

def roi(profit, cluster):
    cm = eh.cluster_mean(profit, cluster)
    return cm


def test_g(j):
    """The only question that pays: does price under-reaction beat the hold?

    Realised ROI straight from the American odds. Profit is (decimal - 1) on
    a win and -1 on a loss, so the hold is inside the number by construction
    and no devigging or probability conversion enters.

    A slope above 1 says extreme prices are too conservative, so the rule is
    to bet WITH the price: back the over when the over is already the
    expensive side. Thresholds are swept, which is a multiple comparison, so
    the Bonferroni threshold is printed and the whole table should be read as
    exploratory rather than as a selected rule.
    """
    print("\n" + "=" * 96)
    print("TEST G: realised ROI of price-extremity rules, receptions")
    print("=" * 96)
    sub = strata_filter(j[j["market"] == "receptions"]).copy()
    sub["prof_over"] = np.where(sub["over"] > 0.5,
                                sub["dec_over"] - 1.0, -1.0)
    sub["prof_under"] = np.where(sub["over"] < 0.5,
                                 sub["dec_under"] - 1.0, -1.0)

    cuts = [0.50, 0.52, 0.54, 0.56, 0.58, 0.60]
    n_tests = len(cuts) * 2 + 2
    thresh = 0.05 / n_tests
    print(f"  {n_tests} cells swept. Bonferroni threshold {thresh:.4f}.")
    print("  EXPLORATORY. A cell that clears a raw 0.05 in a sweep this wide")
    print("  is expected by chance.\n")
    print(f"  {'rule':<34}{'bets':>7}{'ROI':>9}{'SE':>8}{'t':>7}{'p':>9}"
          f"{'MDE':>9}")

    def report(lab, frame, col):
        if len(frame) < 100:
            print(f"  {lab:<34}{len(frame):>7}   too few bets")
            return
        cm = roi(frame[col], frame["event_id"])
        if cm is None:
            return
        t = cm["mean"] / cm["se"] if cm["se"] > 0 else np.nan
        print(f"  {lab:<34}{cm['n']:>7}{cm['mean']:>+9.4f}{cm['se']:>8.4f}"
              f"{t:>7.2f}{fmt_p(two_sided_p(t), thresh)}{mde(cm['se']):>9.4f}")

    report("every over, no filter", sub, "prof_over")
    report("every under, no filter", sub, "prof_under")
    for c in cuts:
        hi = sub[sub["book_p_over"] >= c]
        report(f"back over when price >= {c:.2f}", hi, "prof_over")
    for c in cuts:
        lo = sub[sub["book_p_over"] <= 1.0 - c]
        report(f"back under when over price <= {1.0 - c:.2f}", lo,
               "prof_under")

    print("\n  BY LINE BAND, backing the expensive side")
    print(f"  {'line':>8}{'bets':>7}{'ROI':>9}{'SE':>8}{'t':>7}")
    sub["exp_side_prof"] = np.where(sub["book_p_over"] >= 0.5,
                                    sub["prof_over"], sub["prof_under"])
    for ln in sorted(sub["line_fanduel"].unique()):
        b = sub[sub["line_fanduel"] == ln]
        cm = roi(b["exp_side_prof"], b["event_id"])
        if cm is None:
            continue
        t = cm["mean"] / cm["se"] if cm["se"] > 0 else np.nan
        print(f"  {ln:>8.1f}{cm['n']:>7}{cm['mean']:>+9.4f}"
              f"{cm['se']:>8.4f}{t:>7.2f}")
    print("\n  Break-even is 0.0000. The hold is about 5.7 points, so a rule")
    print("  needs to overcome roughly -0.057 before it returns anything.")


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
    ap.add_argument("--placebo", type=int, default=0)
    ap.add_argument("--save-rows", default=None)
    args = ap.parse_args()

    j, markets = build(args)
    gate_betas(j, markets)
    j = derive(j)
    gate_v1(j)
    gate_v2_testd(j)

    test_e(j, args.placebo)
    test_f(j)
    test_g(j)

    if args.save_rows:
        cols = ["season", "week", "market", "player", "line_fanduel",
                "over_odds", "under_odds", "dec_over", "dec_under",
                "book_p_over", "hold", "projection", "dev", "actual",
                "over", "event_id", "n_books"]
        cols = [c for c in cols if c in j.columns]
        j[cols].to_csv(args.save_rows, index=False, encoding="utf-8")
        print(f"\n  wrote {len(j)} rows to {args.save_rows}")

    print("\n" + "=" * 96)
    print("READING ORDER")
    print("=" * 96)
    print("  Gate 3 first. If v2's Test D did not reproduce, stop.")
    print("  Then Test E's placebo line. If it is still offset, Test E's")
    print("  b_dev is not honest either and the defect is elsewhere.")
    print("  Only then read b_dev, and read the MDE beside it.")
    print("  Test G is the one that could pay. Break-even is 0.0000.")
    print("=" * 96)


if __name__ == "__main__":
    main()
