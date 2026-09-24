"""
QB_PASSING, SEASON STABILITY: testing a claim I made without testing it

THE ERROR THIS CORRECTS, AND IT IS MINE

  The evening handoff lists, as caveat 1 against the qb_passing signal:
  "Season concentration, the same pattern that killed receiving. At a fixed
  cutoff, 2024 gives +5.87 at t 1.39 and 2025 gives +14.90 at t 3.08."

  I read two point estimates and called it concentration WITHOUT TESTING
  WHETHER THEY DIFFER. By hand:

      difference +9.03   SE 6.41   t +1.41   p 0.1592
      MDE at 80 percent power: 17.96 pp

  They are not distinguishable. The slopes are clearer still: difference
  -0.213, SE 0.667, t -0.32. So qb_passing shows NO detectable season
  heterogeneity, and the handoff's caveat 1 was my error rather than a
  property of the data.

  RECEIVING IS THE CONTRAST AND IT IS GENUINELY DIFFERENT. Its slope moves
  +0.290 to +1.314 between 2024 and 2025, difference +1.024, SE 0.460,
  t +2.23, p 0.0261. That one IS heterogeneous, which is why setting it aside
  was right for a reason I had not actually established at the time.

  This is the same failure mode the project's own rules exist to catch:
  comparing two numbers with intervals by looking at their point estimates.
  "Report the minimum detectable effect beside every null result" applies to
  differences between subgroups too, not just to a single null.

WHAT THIS SCRIPT DOES PROPERLY

  H1. A FORMAL INTERACTION TEST on the slope. One pooled regression with a
      season interaction, clustered on game, so the heterogeneity has a
      single coefficient and a single standard error rather than being
      eyeballed across two separate fits.

  H2. THE SAME TEST ON THE FIXED-CUTOFF GAP, which is the other metric that
      appeared to disagree with the slope. If both interaction terms are
      null, the two metrics were never in tension and the apparent conflict
      was noise in two thin subgroups.

  H3. RECEIVING RUN THROUGH BOTH, as the positive control. If H1 and H2
      cannot detect heterogeneity anywhere, they are underpowered rather
      than informative, and receiving is the case where we already believe
      it exists.

  H4. THE DISAGREEMENT DISTRIBUTION BY SEASON. A stable slope with an
      unstable tail gap has one clean mechanical explanation: the slope is
      the same but the model disagrees MORE in one season, so more rows
      clear the cutoff and the tail is composed differently. This checks it
      directly rather than inferring it.

  H5. LINEARITY. The slope assumes the disagreement is informative
      proportionally. Non-cumulative bins of disagreement against the
      realised over rate show whether that holds, which matters because a
      curved relationship would make the slope and the tail gap genuinely
      different quantities rather than two estimates of one thing.

PRE-REGISTERED PREDICTIONS

  I1. The qb_passing slope interaction is null, t under 1.
  I2. The qb_passing gap interaction is null but with an MDE large enough
      that the null is weak, roughly 15 to 20 points.
  I3. The receiving slope interaction IS significant, confirming the tests
      can detect heterogeneity when it exists.
  I4. The disagreement distribution is WIDER in 2025 for qb_passing, which
      would explain the larger tail gap without any change in the underlying
      relationship.
  I5. The disagreement-to-outcome relationship is roughly linear, so the
      slope and the tail gap are two views of one quantity.

  If I3 fails, H1 and H2 are underpowered and this script settles nothing.
  That is the check that makes the rest of it readable.

THE GATE

  Reproduce the published qb_passing figures before testing anything: beta
  0.085, model 3 slope 0.832, holdout ROI 0.1278, 248 bets. Then the
  per-season slopes 0.947 and 0.734 and the fixed-cutoff gaps +5.87 and
  +14.90, which are the numbers this script is arguing about.

Run from the repo root with the venv active:
    python season_stability.py --all-seasons --cache lines_cache.parquet
"""
import argparse
import sys

import numpy as np
import pandas as pd

import eval_harness as eh

PUB = {
    "qb_passing": {"beta": 0.085, "slope": 0.832, "roi": 0.1278,
                   "bets": 248, "s2024": 0.947, "s2025": 0.734,
                   "gap2024": 5.87, "gap2025": 14.90, "cut": 0.040},
    "receiving": {"beta": 0.066, "slope": 0.570,
                  "s2024": 0.290, "s2025": 1.314, "cut": None},
}
TOL = 0.030
TOL_ROI = 0.010
TOL_PP = 0.30


def american_to_decimal(o):
    try:
        o = float(o)
    except (TypeError, ValueError):
        return np.nan
    if not np.isfinite(o) or o == 0:
        return np.nan
    return 1.0 + o / 100.0 if o > 0 else 1.0 + 100.0 / (-o)


def devig(a, b):
    da, db = american_to_decimal(a), american_to_decimal(b)
    if not (np.isfinite(da) and np.isfinite(db)):
        return np.nan
    ra, rb = 1.0 / da, 1.0 / db
    return ra / (ra + rb) if (ra + rb) > 0 else np.nan


def two_sided_p(t):
    if t is None or not np.isfinite(t):
        return np.nan
    from math import erfc, sqrt
    return erfc(abs(t) / sqrt(2.0))


def mde(se):
    if se is None or not np.isfinite(se) or se <= 0:
        return np.nan
    return 2.80 * se


def ols_multi(X, y, cluster, names):
    """Clustered OLS. Self-tested earlier today against planted coefficients."""
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    c = np.asarray(cluster)
    if X.ndim == 1:
        X = X[:, None]
    ok = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    X, y, c = X[ok], y[ok], c[ok]
    n, k = X.shape
    if n < 50:
        return None
    D = np.column_stack([np.ones(n), X])
    if int(np.linalg.matrix_rank(D)) < k + 1:
        return {"rank_deficient": True}
    coef, *_ = np.linalg.lstsq(D, y, rcond=None)
    resid = y - D @ coef
    inv = np.linalg.inv(D.T @ D)
    meat = np.zeros((k + 1, k + 1))
    o = np.argsort(c)
    Ds, rs, cs = D[o], resid[o], c[o]
    st = np.r_[0, np.flatnonzero(cs[1:] != cs[:-1]) + 1, len(cs)]
    g = len(st) - 1
    for i in range(g):
        sl = slice(st[i], st[i + 1])
        u = Ds[sl].T @ rs[sl]
        meat += np.outer(u, u)
    scale = (g / max(g - 1, 1)) * ((n - 1) / max(n - k - 1, 1))
    se = np.sqrt(np.diag(inv @ meat @ inv * scale))
    out = {"rank_deficient": False, "n": n, "clusters": g}
    for i, nm in enumerate(names, start=1):
        out[f"b_{nm}"] = coef[i]
        out[f"se_{nm}"] = se[i]
        out[f"t_{nm}"] = coef[i] / se[i] if se[i] > 0 else np.nan
    return out


def build(market, seasons, cache, refresh):
    import importlib
    from models import data_utils
    mod_name, actual_col = eh.MARKET_SPEC[market]
    mod = importlib.import_module(f"models.{mod_name}")
    feats = list(getattr(mod, "LEAN_FEATS", None)
                 or getattr(mod, "FEATS", None) or [])
    builder = getattr(mod, "build_all_rows", None) or mod.build_dataset
    full = builder()
    keep = feats + [actual_col, "season", "week", "player_display_name"]
    full = full[[c for c in keep if c in full.columns]].dropna(
        subset=feats + [actual_col]).copy()
    full["_key"] = data_utils.norm_join_name(full["player_display_name"])
    full["week"] = full["week"].astype(int)

    def factory():
        raise SystemExit("cache incomplete; run eval_harness once first")

    lines = eh.load_lines(factory, seasons, [market], cache, refresh)
    lines = lines[lines["week"].notna()]
    props = eh.collapse_books(lines)
    props["_key"] = data_utils.norm_join_name(props["player"])
    props["week"] = props["week"].astype(int)
    j = props.merge(full, on=["season", "week", "_key"], how="inner")
    j["book_p"] = [devig(o, u)
                   for o, u in zip(j["over_odds"], j["under_odds"])]
    j["dec_over"] = [american_to_decimal(o) for o in j["over_odds"]]
    j["dec_under"] = [american_to_decimal(o) for o in j["under_odds"]]
    j = j.dropna(subset=["line_fanduel", "book_p", actual_col,
                         "dec_over", "dec_under"]).copy()
    j = j[j[actual_col] != j["line_fanduel"]].copy()
    j["over"] = (j[actual_col] > j["line_fanduel"]).astype(int)
    return j, full, feats, actual_col


def score(j, feats):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    cols = ["book_p", "line_fanduel"] + feats
    d = j.reset_index(drop=True).copy()
    p = np.full(len(d), np.nan)
    for s in sorted(d["season"].unique()):
        tr = d.index[d["season"] < s]
        te = d.index[d["season"] == s]
        if len(tr) < 500 or not len(te):
            continue
        y = d.loc[tr, "over"].to_numpy(int)
        if len(np.unique(y)) < 2:
            continue
        m = make_pipeline(StandardScaler(),
                          LogisticRegression(max_iter=2000))
        m.fit(d.loc[tr, cols].to_numpy(float), y)
        p[te] = m.predict_proba(d.loc[te, cols].to_numpy(float))[:, 1]
    d["p"] = p
    d = d.dropna(subset=["p"]).copy()
    d["dis"] = d["p"] - d["book_p"]
    d["be_over"] = 1.0 / d["dec_over"]
    d["be_under"] = 1.0 / d["dec_under"]
    d["e_over"] = d["p"] - d["be_over"]
    d["e_under"] = (1.0 - d["p"]) - d["be_under"]
    d["side_over"] = d["e_over"] >= d["e_under"]
    d["edge"] = np.where(d["side_over"], d["e_over"], d["e_under"])
    won = np.where(d["side_over"], d["over"] == 1, d["over"] == 0)
    pay = np.where(d["side_over"], d["dec_over"] - 1.0, d["dec_under"] - 1.0)
    d["won_f"] = won.astype(float)
    d["profit"] = np.where(won, pay, -1.0)
    be_ch = np.where(d["side_over"], d["be_over"], d["be_under"])
    be_ot = np.where(d["side_over"], d["be_under"], d["be_over"])
    d["fair_chosen"] = be_ch / (be_ch + be_ot)
    return d


def slope(sub):
    r = ols_multi(np.column_stack([sub["dis"], sub["book_p"]]),
                  sub["over"], sub["event_id"], ["dis", "bp"])
    return r


def gap(g):
    if len(g) < 20:
        return None
    cw = eh.cluster_mean(g["won_f"].to_numpy(), g["event_id"].to_numpy())
    if cw is None:
        return None
    fair = float(g["fair_chosen"].mean())
    return {"n": len(g), "gap": cw["mean"] - fair, "se": cw["se"],
            "t": (cw["mean"] - fair) / cw["se"] if cw["se"] > 0 else np.nan,
            "roi": float(g["profit"].mean())}


def gate(market, j, full, d, feats, actual_col):
    from sklearn.linear_model import LinearRegression
    print(f"\n  GATE, {market}")
    out = []
    for s in sorted(j["season"].unique()):
        tr = full[full["season"] < s]
        te = j[j["season"] == s]
        if len(tr) < 300 or not len(te):
            continue
        m = LinearRegression().fit(tr[feats], tr[actual_col])
        te = te.copy()
        te["proj"] = m.predict(te[feats])
        out.append(te)
    g = pd.concat(out, ignore_index=True)
    f = eh.ols_clustered(g["proj"] - g["line_fanduel"],
                         g[actual_col] - g["line_fanduel"], g["event_id"])
    r = slope(d)
    pub = PUB[market]
    checks = [("beta", f["beta"], pub["beta"], TOL),
              ("slope", r["b_dis"], pub["slope"], TOL)]
    for yr in (2024, 2025):
        sub = d[d["season"] == yr]
        rr = slope(sub)
        if rr and not rr.get("rank_deficient"):
            checks.append((f"slope {yr}", rr["b_dis"],
                           pub[f"s{yr}"], TOL))
    if pub.get("cut") is not None:
        for yr in (2024, 2025):
            gg = gap(d[(d["season"] == yr) & (d["edge"] >= pub["cut"])])
            if gg:
                checks.append((f"gap {yr} pp", 100 * gg["gap"],
                               pub[f"gap{yr}"], TOL_PP))
    fails = []
    print(f"    {'quantity':<16}{'got':>10}{'published':>11}{'diff':>9}"
          f"   verdict")
    for name, got, p_, tol in checks:
        dd = got - p_
        ok = np.isfinite(dd) and abs(dd) <= tol
        print(f"    {name:<16}{got:>10.4f}{p_:>11.4f}{dd:>+9.4f}"
              f"   {'PASS' if ok else 'FAIL'}")
        if not ok:
            fails.append(name)
    if fails:
        print(f"    GATE FAILED for {fails}")
        return False
    print("    GATE PASSED")
    return True


def h1_slope_interaction(d, market):
    """One pooled regression with a season interaction on the slope."""
    print(f"\n  H1. SLOPE INTERACTION, {market}")
    sub = d[d["season"].isin([2024, 2025])].copy()
    if len(sub) < 200:
        print("    too few rows")
        return
    is25 = (sub["season"] == 2025).astype(float)
    X = np.column_stack([sub["dis"], sub["book_p"], is25,
                         sub["dis"] * is25])
    r = ols_multi(X, sub["over"], sub["event_id"],
                  ["dis", "bp", "is25", "inter"])
    if r is None or r.get("rank_deficient"):
        print("    fit failed")
        return
    print(f"    n {r['n']}, {r['clusters']} games")
    print(f"    {'term':<22}{'coef':>10}{'SE':>9}{'t':>7}{'p':>9}{'MDE':>9}")
    for nm, lab in (("dis", "slope in 2024"),
                    ("inter", "2025 minus 2024")):
        b, se, t = r[f"b_{nm}"], r[f"se_{nm}"], r[f"t_{nm}"]
        print(f"    {lab:<22}{b:>10.4f}{se:>9.4f}{t:>7.2f}"
              f"{two_sided_p(t):>9.4f}{mde(se):>9.4f}")


def h2_gap_interaction(d, market, cut):
    """The same interaction test on the fixed-cutoff outcome."""
    if cut is None:
        return
    print(f"\n  H2. GAP INTERACTION at a fixed {cut:.3f}, {market}")
    sub = d[(d["edge"] >= cut) & d["season"].isin([2024, 2025])].copy()
    if len(sub) < 60:
        print("    too few rows")
        return
    is25 = (sub["season"] == 2025).astype(float)
    # outcome minus the fair price, so the coefficient is a gap difference
    y = sub["won_f"] - sub["fair_chosen"]
    r = ols_multi(is25.to_numpy(), y, sub["event_id"], ["is25"])
    if r is None or r.get("rank_deficient"):
        print("    fit failed")
        return
    b, se, t = r["b_is25"], r["se_is25"], r["t_is25"]
    print(f"    n {r['n']}, {r['clusters']} games")
    print(f"    2025 minus 2024 gap  {100 * b:+.2f} pp   "
          f"SE {100 * se:.2f}   t {t:+.2f}   p {two_sided_p(t):.4f}")
    print(f"    MDE {100 * mde(se):.2f} pp")
    if abs(t) < 2:
        print(f"    NOT distinguishable. And the MDE says a difference below")
        print(f"    {100 * mde(se):.0f} points cannot be seen at this n, so this null is")
        print(f"    weak rather than reassuring.")


def h4_dispersion(d, market, cut):
    """Does the model simply disagree MORE in one season?"""
    print(f"\n  H4. DISAGREEMENT DISTRIBUTION BY SEASON, {market}")
    print(f"    {'season':>8}{'rows':>7}{'sd dis':>9}{'p90 |dis|':>11}"
          f"{'rows over cut':>15}{'share':>8}")
    for s, g in d.groupby("season"):
        n_cut = int((g["edge"] >= cut).sum()) if cut is not None else 0
        print(f"    {int(s):>8}{len(g):>7}{g['dis'].std():>9.4f}"
              f"{np.percentile(g['dis'].abs(), 90):>11.4f}"
              f"{n_cut:>15}{100.0 * n_cut / max(len(g), 1):>7.1f}%")
    print("    A stable slope with an unstable tail gap is explained if the")
    print("    model disagrees more in one season: more rows clear the cut")
    print("    and the tail is composed differently, with no change in the")
    print("    underlying relationship.")


def h5_linearity(d, market):
    """Is the disagreement informative proportionally?"""
    print(f"\n  H5. LINEARITY of disagreement against the outcome, {market}")
    sub = d.copy()
    sub["q"] = pd.qcut(sub["dis"], 8, labels=False, duplicates="drop")
    print(f"    {'octile':>8}{'rows':>7}{'mean dis':>10}{'over rate':>11}"
          f"{'book_p':>9}{'gap pp':>9}{'SE pp':>8}")
    for q, g in sub.groupby("q"):
        cw = eh.cluster_mean(g["over"].to_numpy().astype(float),
                             g["event_id"].to_numpy())
        if cw is None:
            continue
        bp = float(g["book_p"].mean())
        print(f"    {int(q):>8}{len(g):>7}{g['dis'].mean():>10.4f}"
              f"{cw['mean']:>11.4f}{bp:>9.4f}"
              f"{100 * (cw['mean'] - bp):>+9.2f}{100 * cw['se']:>8.2f}")
    print("    A monotone gap column means the disagreement is informative")
    print("    proportionally, so the slope and the tail gap are two views")
    print("    of one quantity rather than different objects.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all-seasons", action="store_true")
    ap.add_argument("--seasons", default=None)
    ap.add_argument("--cache", default="lines_cache.parquet")
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()
    seasons = (list(eh.ALL_SEASONS) if args.all_seasons or not args.seasons
               else [int(x) for x in args.seasons.split(",") if x.strip()])

    print("=" * 96)
    print("SEASON STABILITY: is qb_passing's signal concentrated in 2025?")
    print("  receiving is the positive control, because its slope IS")
    print("  heterogeneous (t +2.23 by hand). If the tests cannot detect")
    print("  that, they are underpowered and settle nothing.")
    print("=" * 96)

    for market in ("qb_passing", "receiving"):
        print(f"\n{'=' * 96}\n{market.upper()}\n{'=' * 96}")
        j, full, feats, actual_col = build(market, seasons, args.cache,
                                           args.refresh)
        print(f"  joined {len(j)} props, full dataset {len(full)} rows")
        d = score(j, feats)
        print(f"  scored {len(d)} rows out of sample")
        if not gate(market, j, full, d, feats, actual_col):
            print("  gate failed, results not comparable to the published")
            print("  figures. Skipping this market.")
            continue
        cut = PUB[market].get("cut")
        h1_slope_interaction(d, market)
        h2_gap_interaction(d, market, cut)
        h4_dispersion(d, market, cut if cut else 0.04)
        h5_linearity(d, market)

    print("\n" + "=" * 96)
    print("HOW TO READ THIS")
    print("=" * 96)
    print("  Receiving first. If its slope interaction is significant, the")
    print("  test works and qb_passing's null means something. If receiving")
    print("  also comes back null, both tests are underpowered and the")
    print("  season question is simply unanswerable on this data, which is")
    print("  itself the finding.")
    print("  Then H4. A wider disagreement distribution in 2025 explains a")
    print("  larger tail gap with no change in the relationship, which")
    print("  would resolve the apparent conflict between the two metrics")
    print("  mechanically rather than statistically.")
    print("=" * 96)


if __name__ == "__main__":
    main()
