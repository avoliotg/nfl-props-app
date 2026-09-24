"""
ODDS-SPACE TEST v4: the BOOK term, which is the channel v1 to v3 never tested

WHY THIS EXISTS

  v3 concluded the model adds nothing to FanDuel's price: with the price
  controlled, b_dev is +0.0056 (t 0.54) and b_book_p is 0.9901 (t 10.21),
  placebo centred at -0.00022 (t -0.33).

  That did NOT close the candidate rule, and edge_threshold_v5's own
  docstring says why at its line 378. The blend is anchored on the CONSENSUS
  line while the bet settles at the FANDUEL line, so the distance driving the
  edge has three parts, not two:

      blend - bet_line =  alpha
                       +  beta * (projection - consensus)    MODEL
                       +  (consensus - bet_line)             BOOK

  v1 through v3 tested only the MODEL term. The BOOK term is a different
  claim: that when the other seven books disagree with FanDuel's LINE,
  FanDuel's own ODDS do not fully absorb the disagreement.

  Inside a fixed FanDuel line stratum, (consensus - fanduel) varies only
  because other books moved. So this asks whether other books know something
  FanDuel's price does not. Nothing in the project has measured that. It is
  also the only remaining channel that could produce +0.1475 on 176 bets at
  receptions 2.5 and 3.5 in a market where every simple rule in v3's Test G
  came back negative.

  Note this is NOT line shopping, which is already dead. Shopping takes a
  better PRICE for the same side. This is other books' lines as a FORECAST,
  used while still betting at FanDuel's number.

TWO PARAMETERISATIONS, BOTH REPORTED

  Spec 1 mirrors v5 exactly:
      over ~ book_p + (projection - consensus) + (consensus - fanduel)
  Spec 2 anchors the model term on the line actually bet, so the model term
  is not entangled with the book term through the consensus:
      over ~ book_p + (projection - fanduel) + (consensus - fanduel)

  Algebraically (proj - consensus) = (proj - fanduel) - (consensus - fanduel),
  so the two specs carry the same information in different coordinates. If
  they disagree about which term is significant, the coordinate choice is
  doing the work and neither should be believed. Printing both is the check.

  Every term is demeaned inside its FanDuel line stratum, as in v3.

A CAVEAT THAT COULD MAKE b_book MECHANICAL

  line_consensus is a MEDIAN THAT INCLUDES FANDUEL. So when FanDuel is an
  outlier, (consensus - fanduel) is large partly by construction. That is a
  milder version of the line_min / line_max artifact already documented,
  because a median is not an order statistic, but it is not zero either.
  TEST I shuffles the book term within stratum to locate the null, exactly
  the way v2's placebo caught my line-FE defect.

  Second caveat: the consensus book set CHANGES, eleven books in 2023 and
  2024 against eight in 2025, three of them defunct. So the book term is
  measured against a moving object. TEST J splits it by season, which is also
  the cheapest test yet of why every arm of the candidate rule loses in 2023.

PRE-REGISTERED PREDICTIONS

  H1. b_book comes back POSITIVE and significant in both specs. Reasoning: it
      is the only untested channel left, the candidate rule is built on it,
      and v5's decomposition note flags it as un-attributed rather than as
      measured. If b_book is ALSO zero, the candidate rule has no surviving
      mechanism and its +0.1475 is threshold selection, which would close the
      last open betting route in the project.
  H2. b_dev stays near zero in both specs, replicating v3.
  I1. The book-term placebo centres on zero. If it does not, b_book is partly
      the median-includes-FanDuel artifact and the honest value is the
      difference from the placebo mean.
  J1. b_book is largest in 2023, when the consensus had eleven books
      including three that no longer exist.
  K1. A direct book-term ROI rule lands NEGATIVE, because v3's Test G found
      every receptions cell negative and the hold is 5.66 points. If K1 is
      refuted while H1 holds, that is a real rule and it uses no model.

THE GATES. FOUR.

  Gate 1: published FanDuel betas, 0.226 / 0.066 / -0.043 / 0.085.
  Gate 2: v1's Test B receptions price slope, 1.008.
  Gate 3: v2's pooled Test D triple, 0.0431 / 0.0126 / 0.9856.
  Gate 4: v3's Test E triple under line FE, 0.0411 / 0.0056 / 0.9901.

  No gate may be widened to make a run pass.

ALSO RUN THESE TWO, they are published numbers this script does not compute:
    python edge_threshold_v5.py --all-seasons --cache lines_cache.parquet
    python edge_threshold_v5.py --all-seasons --cache lines_cache.parquet --no-model
    python edge_threshold_v5.py --all-seasons --cache lines_cache.parquet --same-line
  Expect pooled holdout about +0.1475 full, about +0.0524 with --no-model
  (book only) and about +0.0742 with --same-line (model only). If those three
  no longer reproduce, stop, because the target this script is explaining has
  moved.

Run from the repo root with the venv active:
    python odds_space_test_v4.py --all-seasons --cache lines_cache.parquet --placebo 200
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
PUBLISHED_V1_TESTB = 1.008
PUBLISHED_V2_TESTD = {"dev_solo": 0.0431, "dev_race": 0.0126,
                      "book_p_race": 0.9856}
PUBLISHED_V3_TESTE = {"dev_solo_fe": 0.0411, "dev_race_fe": 0.0056,
                      "book_p_race_fe": 0.9901}
GATE_TOL = 0.030
GATE_TOL_D = 0.010

MIN_PRICE_SD = 0.015
MIN_STRATUM_N = 40

# The candidate rule's own population, for the restricted cut in Test K.
CANDIDATE_LINES = (2.5, 3.5)


# ------------------------------------------------------------------ odds math

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

    Self-tested: recovers planted coefficients with correlated regressors,
    centres at zero under a planted null with within-stratum demeaning (mean
    -0.00071 over 40 replications, t -0.68), reported SE matches realised sd
    to 0.0065 against 0.0066, and flags rank deficiency on a collinear pair.
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
    print("ODDS-SPACE TEST v4: the BOOK term")
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
    j = j.dropna(subset=["line_fanduel", "line_consensus", "actual",
                         "book_p_over", "projection", "dec_over",
                         "dec_under"]).copy()
    print(f"\n  rows with a two-sided FanDuel price AND a consensus: "
          f"{len(j)} of {n0}")
    push = (j["actual"] == j["line_fanduel"])
    j = j[~push].copy()
    j["over"] = (j["actual"] > j["line_fanduel"]).astype(float)
    j["dev_fd"] = j["projection"] - j["line_fanduel"]
    j["dev_cons"] = j["projection"] - j["line_consensus"]
    j["dev_book"] = j["line_consensus"] - j["line_fanduel"]
    print(f"  pushes dropped: {int(push.sum())}")
    print(f"  median hold: {j['hold'].median():.4f}")
    return j


def strata_filter(sub):
    sizes = sub.groupby("line_fanduel")["over"].transform("size")
    sds = sub.groupby("line_fanduel")["book_p_over"].transform("std")
    return sub[(sizes >= MIN_STRATUM_N) & (sds >= MIN_PRICE_SD)].copy()


def book_term_report(j):
    """How much does the book term actually MOVE? Before testing it.

    v1's error was reading an undefined slope as a flat one, because receiving
    prices barely vary. The same trap applies here: if FanDuel's line equals
    the consensus on almost every row, the book term has no variation and its
    coefficient is a ratio of near-zero quantities.
    """
    print("\n" + "=" * 96)
    print("BOOK TERM VARIATION: can it be tested at all?")
    print("=" * 96)
    sub = strata_filter(j[j["market"] == "receptions"])
    db = sub["dev_book"]
    print(f"  receptions, {len(sub)} rows in usable strata")
    print(f"  dev_book = consensus - fanduel")
    print(f"    exactly zero      {int((db == 0).sum()):>7} "
          f"({100.0 * (db == 0).mean():.1f}%)")
    print(f"    nonzero           {int((db != 0).sum()):>7} "
          f"({100.0 * (db != 0).mean():.1f}%)")
    print(f"    sd                {db.std():>7.4f}")
    print(f"    within-stratum sd "
          f"{demean_within(sub, 'dev_book', 'line_fanduel').std():>7.4f}")
    print(f"    range             [{db.min():+.1f}, {db.max():+.1f}]")
    vc = db.value_counts().sort_index()
    print("\n    value      rows")
    for v, n in vc.items():
        if n >= 20:
            print(f"    {v:>+6.1f}{n:>10}")
    if db.std() < 0.05:
        print("\n  The book term barely varies. Any coefficient below is a")
        print("  ratio of near-zero quantities and must not be read as flat.")
    return sub


# ---------------------------------------------------------------------- gates

def gate_betas(j, markets):
    print("\n" + "=" * 96)
    print("GATE 1: published FanDuel-anchored betas")
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
        print(f"\n  GATE 1 FAILED for {fails}. Stop.")
        sys.exit(2)
    print(f"  GATE 1 PASSED at {GATE_TOL:.3f}.")


def gate_v1(j):
    print("\n" + "=" * 96)
    print("GATE 2: v1 Test B receptions price slope")
    print("=" * 96)
    sub = strata_filter(j[j["market"] == "receptions"])
    f = eh.ols_clustered(demean_within(sub, "book_p_over", "line_fanduel"),
                         demean_within(sub, "over", "line_fanduel"),
                         sub["event_id"])
    d = f["beta"] - PUBLISHED_V1_TESTB
    ok = abs(d) <= GATE_TOL
    print(f"  slope {f['beta']:.3f}   published {PUBLISHED_V1_TESTB:.3f}"
          f"   diff {d:+.3f}   {'PASS' if ok else 'FAIL'}")
    if not ok:
        sys.exit(2)


def gate_v2(j):
    print("\n" + "=" * 96)
    print("GATE 3: v2 Test D (pooled, no line fixed effects)")
    print("=" * 96)
    sub = j[j["market"] == "receptions"].copy()
    solo = eh.ols_clustered(sub["dev_fd"], sub["over"], sub["event_id"])
    race = ols_clustered_multi(sub[["book_p_over", "dev_fd"]].to_numpy(),
                               sub["over"].to_numpy(),
                               sub["event_id"].to_numpy(), ["book_p", "dev"])
    got = {"dev_solo": solo["beta"], "dev_race": race["b_dev"],
           "book_p_race": race["b_book_p"]}
    _check(got, PUBLISHED_V2_TESTD, GATE_TOL_D, 3)


def gate_v3(j):
    print("\n" + "=" * 96)
    print("GATE 4: v3 Test E (with line fixed effects)")
    print("=" * 96)
    sub = strata_filter(j[j["market"] == "receptions"])
    pw = demean_within(sub, "book_p_over", "line_fanduel")
    dw = demean_within(sub, "dev_fd", "line_fanduel")
    ow = demean_within(sub, "over", "line_fanduel")
    solo = eh.ols_clustered(dw, ow, sub["event_id"])
    race = ols_clustered_multi(np.column_stack([pw, dw]), ow.to_numpy(),
                               sub["event_id"].to_numpy(), ["book_p", "dev"])
    got = {"dev_solo_fe": solo["beta"], "dev_race_fe": race["b_dev"],
           "book_p_race_fe": race["b_book_p"]}
    _check(got, PUBLISHED_V3_TESTE, GATE_TOL_D, 4)


def _check(got, published, tol, n):
    fails = []
    print(f"  {'quantity':<18}{'got':>10}{'published':>11}{'diff':>9}"
          f"   verdict")
    for k, pub in published.items():
        d = got[k] - pub
        ok = abs(d) <= tol
        print(f"  {k:<18}{got[k]:>10.4f}{pub:>11.4f}{d:>+9.4f}"
              f"   {'PASS' if ok else 'FAIL'}")
        if not ok:
            fails.append(k)
    if fails:
        print(f"\n  GATE {n} FAILED for {fails} at tolerance {tol:.3f}. Stop.")
        sys.exit(2)
    print(f"  GATE {n} PASSED at {tol:.3f}.")


# --------------------------------------------------------------------- test H

def test_h(sub):
    """THE THREE-WAY HORSE RACE. Both parameterisations."""
    print("\n" + "=" * 96)
    print("TEST H: over ~ price + model + book, line fixed effects")
    print("=" * 96)
    ow = demean_within(sub, "over", "line_fanduel").to_numpy()
    pw = demean_within(sub, "book_p_over", "line_fanduel").to_numpy()
    bw = demean_within(sub, "dev_book", "line_fanduel").to_numpy()
    mw_cons = demean_within(sub, "dev_cons", "line_fanduel").to_numpy()
    mw_fd = demean_within(sub, "dev_fd", "line_fanduel").to_numpy()
    ev = sub["event_id"].to_numpy()

    thresh = 0.05 / 6.0
    print(f"  6 coefficients across two specs, Bonferroni {thresh:.4f}\n")
    specs = [
        ("SPEC 1, v5 coordinates: model anchored on consensus",
         np.column_stack([pw, mw_cons, bw]),
         ["price", "model_cons", "book"]),
        ("SPEC 2: model anchored on the line actually bet",
         np.column_stack([pw, mw_fd, bw]),
         ["price", "model_fd", "book"]),
    ]
    out = {}
    for label, X, names in specs:
        r = ols_clustered_multi(X, ow, ev, names)
        print(f"  {label}")
        if r is None or r.get("rank_deficient"):
            print("    rank deficient or fit failed")
            print()
            continue
        print(f"    {'term':<20}{'coef':>10}{'SE':>9}{'t':>8}{'p':>9}"
              f"{'MDE':>9}")
        for nm in names:
            b, se, t = r[f"b_{nm}"], r[f"se_{nm}"], r[f"t_{nm}"]
            print(f"    {nm:<20}{b:>10.4f}{se:>9.4f}{t:>8.2f}"
                  f"{fmt_p(two_sided_p(t), thresh)}{mde(se):>9.4f}")
        print(f"    n={r['n']}, {r['clusters']} games")
        print()
        out[label] = r

    print("  If the two specs disagree about which term matters, the")
    print("  coordinate choice is doing the work and neither is believable.")
    return out


# --------------------------------------------------------------------- test I

def test_i(sub, n_shuffles, seed=0):
    """Placebo on the BOOK term. Locates the null under this design.

    The consensus median INCLUDES FanDuel, so a large book term partly means
    FanDuel was an outlier. Shuffling the book term within stratum destroys
    any real forecast content while preserving that structure, so a nonzero
    placebo mean is the mechanical part.
    """
    if not n_shuffles:
        return
    print("=" * 96)
    print(f"TEST I: book-term placebo, {n_shuffles} draws")
    print("=" * 96)
    ow = demean_within(sub, "over", "line_fanduel").to_numpy()
    pw = demean_within(sub, "book_p_over", "line_fanduel").to_numpy()
    mw = demean_within(sub, "dev_fd", "line_fanduel").to_numpy()
    s2 = sub.copy()
    s2["bw"] = demean_within(sub, "dev_book", "line_fanduel")
    ev = sub["event_id"].to_numpy()
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_shuffles):
        shuf = (s2.groupby("line_fanduel")["bw"]
                .transform(lambda v: rng.permutation(v.to_numpy())).to_numpy())
        r = ols_clustered_multi(np.column_stack([pw, mw, shuf]), ow, ev,
                                ["price", "model", "book"])
        if r and not r.get("rank_deficient"):
            vals.append(r["b_book"])
    if not vals:
        print("  no usable draws")
        return
    v = np.asarray(vals)
    se_mean = v.std() / np.sqrt(len(v))
    t = v.mean() / se_mean if se_mean > 0 else np.nan
    print(f"  placebo b_book mean {v.mean():+.5f}   sd {v.std():.5f}   "
          f"SE of mean {se_mean:.5f}   t vs 0 {t:+.2f}")
    print(f"  95% range [{np.percentile(v, 2.5):+.5f}, "
          f"{np.percentile(v, 97.5):+.5f}]")
    if abs(t) > 3:
        print("  PLACEBO OFFSET. Subtract this mean from Test H's b_book.")
    else:
        print("  Placebo centred. Test H's b_book is honest as printed.")
    print()


# --------------------------------------------------------------------- test J

def test_j(sub):
    """Book term by season. The consensus book set changes: 11 books in 2023
    and 2024, 8 in 2025, three defunct brands. Also the cheapest test yet of
    why every arm of the candidate rule loses in 2023.
    """
    print("=" * 96)
    print("TEST J: book term by season (the consensus book set changes)")
    print("=" * 96)
    seasons = sorted(sub["season"].unique())
    thresh = 0.05 / max(len(seasons), 1)
    print(f"  {len(seasons)} seasons, Bonferroni {thresh:.4f}\n")
    print(f"  {'season':>8}{'n':>7}{'b_book':>10}{'SE':>9}{'t':>8}{'p':>9}"
          f"{'MDE':>9}{'book sd':>10}")
    for s in seasons:
        g = sub[sub["season"] == s].copy()
        if len(g) < 200:
            print(f"  {int(s):>8}{len(g):>7}   too few rows")
            continue
        ow = demean_within(g, "over", "line_fanduel").to_numpy()
        pw = demean_within(g, "book_p_over", "line_fanduel").to_numpy()
        mw = demean_within(g, "dev_fd", "line_fanduel").to_numpy()
        bw = demean_within(g, "dev_book", "line_fanduel").to_numpy()
        r = ols_clustered_multi(np.column_stack([pw, mw, bw]), ow,
                                g["event_id"].to_numpy(),
                                ["price", "model", "book"])
        if r is None or r.get("rank_deficient"):
            print(f"  {int(s):>8}{len(g):>7}   rank deficient")
            continue
        b, se, t = r["b_book"], r["se_book"], r["t_book"]
        print(f"  {int(s):>8}{r['n']:>7}{b:>10.4f}{se:>9.4f}{t:>8.2f}"
              f"{fmt_p(two_sided_p(t), thresh)}{mde(se):>9.4f}"
              f"{g['dev_book'].std():>10.4f}")
    print()


# --------------------------------------------------------------------- test K

def test_k(sub):
    """Does a book-term rule PAY? Realised ROI from the American odds.

    Profit is (decimal - 1) on a win and -1 on a loss, so the hold is inside
    the number. The rule: when the consensus sits ABOVE FanDuel's line, other
    books think the true number is higher, so back the over at FanDuel; and
    the mirror. Swept by threshold, which is a multiple comparison.

    Also reported restricted to the candidate rule's own lines, 2.5 and 3.5,
    which is the population whose +0.1475 this whole script is explaining.
    """
    print("=" * 96)
    print("TEST K: realised ROI of a book-term rule, receptions")
    print("=" * 96)
    s = sub.copy()
    s["prof_over"] = np.where(s["over"] > 0.5, s["dec_over"] - 1.0, -1.0)
    s["prof_under"] = np.where(s["over"] < 0.5, s["dec_under"] - 1.0, -1.0)
    # follow the consensus: consensus above FanDuel means back the over
    s["prof_follow"] = np.where(s["dev_book"] > 0, s["prof_over"],
                                np.where(s["dev_book"] < 0, s["prof_under"],
                                         np.nan))
    cuts = [0.0, 0.5, 1.0]
    n_tests = len(cuts) * 2 + 2
    thresh = 0.05 / n_tests
    print(f"  {n_tests} cells. Bonferroni {thresh:.4f}. EXPLORATORY.")
    print("  Break-even is 0.0000. The hold is about 5.7 points.\n")
    print(f"  {'rule':<42}{'bets':>7}{'ROI':>9}{'SE':>8}{'t':>7}{'p':>9}"
          f"{'MDE':>9}")

    def rep(lab, frame, col):
        f = frame.dropna(subset=[col])
        if len(f) < 100:
            print(f"  {lab:<42}{len(f):>7}   too few bets")
            return
        cm = eh.cluster_mean(f[col].to_numpy(), f["event_id"].to_numpy())
        if cm is None:
            return
        t = cm["mean"] / cm["se"] if cm["se"] > 0 else np.nan
        print(f"  {lab:<42}{cm['n']:>7}{cm['mean']:>+9.4f}{cm['se']:>8.4f}"
              f"{t:>7.2f}{fmt_p(two_sided_p(t), thresh)}{mde(cm['se']):>9.4f}")

    for c in cuts:
        g = s[s["dev_book"].abs() > c] if c > 0 else s[s["dev_book"] != 0]
        rep(f"follow consensus, |gap| > {c:.1f}", g, "prof_follow")
    for c in cuts:
        g = s[s["dev_book"].abs() > c] if c > 0 else s[s["dev_book"] != 0]
        g = g.copy()
        g["prof_fade"] = np.where(g["dev_book"] > 0, g["prof_under"],
                                  g["prof_over"])
        rep(f"FADE consensus, |gap| > {c:.1f}", g, "prof_fade")
    rep("all rows, back the over", s, "prof_over")
    rep("all rows, back the under", s, "prof_under")

    print(f"\n  RESTRICTED to the candidate rule's lines "
          f"{CANDIDATE_LINES}")
    cand = s[s["line_fanduel"].isin(CANDIDATE_LINES)]
    print(f"  {'rule':<42}{'bets':>7}{'ROI':>9}{'SE':>8}{'t':>7}")
    for lab, col, frame in (
            ("follow consensus, nonzero gap", "prof_follow",
             cand[cand["dev_book"] != 0]),
            ("back the over", "prof_over", cand),
            ("back the under", "prof_under", cand)):
        f = frame.dropna(subset=[col])
        if len(f) < 50:
            print(f"  {lab:<42}{len(f):>7}   too few")
            continue
        cm = eh.cluster_mean(f[col].to_numpy(), f["event_id"].to_numpy())
        if cm is None:
            continue
        t = cm["mean"] / cm["se"] if cm["se"] > 0 else np.nan
        print(f"  {lab:<42}{cm['n']:>7}{cm['mean']:>+9.4f}"
              f"{cm['se']:>8.4f}{t:>7.2f}")
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
    ap.add_argument("--placebo", type=int, default=0)
    ap.add_argument("--save-rows", default=None)
    args = ap.parse_args()

    j, markets = build(args)
    gate_betas(j, markets)
    j = derive(j)
    gate_v1(j)
    gate_v2(j)
    gate_v3(j)

    sub = book_term_report(j)
    test_h(sub)
    test_i(sub, args.placebo)
    test_j(sub)
    test_k(sub)

    if args.save_rows:
        cols = ["season", "week", "market", "player", "line_fanduel",
                "line_consensus", "n_books", "over_odds", "under_odds",
                "dec_over", "dec_under", "book_p_over", "hold",
                "projection", "dev_fd", "dev_cons", "dev_book",
                "actual", "over", "event_id"]
        cols = [c for c in cols if c in j.columns]
        j[cols].to_csv(args.save_rows, index=False, encoding="utf-8")
        print(f"  wrote {len(j)} rows to {args.save_rows}")

    print("=" * 96)
    print("READING ORDER")
    print("=" * 96)
    print("  1. All four gates. If any failed, nothing below counts.")
    print("  2. BOOK TERM VARIATION. If the book term barely moves, its")
    print("     coefficient is undefined rather than flat. That was v1's")
    print("     mistake with receiving and it applies identically here.")
    print("  3. Test I's placebo. If offset, subtract it from b_book.")
    print("  4. Then Test H. If b_book is zero in BOTH specs, the candidate")
    print("     rule has no surviving mechanism and +0.1475 is selection.")
    print("  5. Test K decides whether any of it pays. Break-even 0.0000.")
    print("=" * 96)


if __name__ == "__main__":
    main()
